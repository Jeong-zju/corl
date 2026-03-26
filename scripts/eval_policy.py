from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

from env import get_env_choices, get_env_module
from eval_helpers import resolve_eval_policy_path
from policy_defaults import load_policy_mode_defaults


def ensure_streaming_act_importable(repo_root: Path) -> None:
    streaming_act_src = repo_root / "main/policy/lerobot_policy_streaming_act/src"
    if not streaming_act_src.exists():
        raise FileNotFoundError(
            f"Streaming ACT package source not found: {streaming_act_src}"
        )
    sys.path.insert(0, str(streaming_act_src))


def patch_lerobot_processor_factory(act_config_cls) -> None:
    import lerobot.policies.factory as policy_factory

    policy_factory.ACTConfig = act_config_cls


def validate_first_frame_anchor_support(
    *,
    env_name: str,
    use_first_frame_anchor: bool,
) -> None:
    if not use_first_frame_anchor:
        return
    if env_name != "braidedhub":
        raise NotImplementedError(
            "First-frame anchor evaluation is currently implemented only for `braidedhub`. "
            f"Got env={env_name!r}."
        )


def validate_prefix_sequence_support(
    *,
    env_name: str,
    policy_name: str,
    use_prefix_sequence_training: bool,
) -> None:
    if not use_prefix_sequence_training:
        return
    if policy_name != "streaming_act":
        raise NotImplementedError(
            "Prefix-sequence evaluation is currently implemented only for `streaming_act`. "
            f"Got policy={policy_name!r}."
        )


def validate_visual_prefix_memory_support(
    *,
    policy_name: str,
    use_visual_prefix_memory: bool,
) -> None:
    if not use_visual_prefix_memory:
        return
    if policy_name != "streaming_act":
        raise NotImplementedError(
            "Visual prefix memory evaluation is currently implemented only for "
            f"`streaming_act`. Got policy={policy_name!r}."
        )


def validate_delta_signature_support(
    *,
    policy_name: str,
    use_delta_signature: bool,
) -> None:
    if not use_delta_signature:
        return
    if policy_name != "streaming_act":
        raise NotImplementedError(
            "Delta-signature evaluation is currently implemented only for "
            f"`streaming_act`. Got policy={policy_name!r}."
        )


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env", choices=get_env_choices(), default="h_shape")
    bootstrap.add_argument(
        "--policy",
        choices=["act", "streaming_act"],
        default="act",
    )
    known_args, _ = bootstrap.parse_known_args(argv)
    defaults = load_policy_mode_defaults("eval", known_args.env, known_args.policy)

    parser = argparse.ArgumentParser(
        description="Evaluate LeRobot ACT or Streaming ACT in a selected environment."
    )
    parser.add_argument("--env", choices=get_env_choices(), default=known_args.env)
    parser.add_argument(
        "--policy",
        choices=["act", "streaming_act"],
        default=known_args.policy,
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=defaults.get("policy_path"),
        help=(
            "Checkpoint dir, pretrained_model dir, or training run dir. "
            "If omitted, the latest run under --train-output-root is used."
        ),
    )
    parser.add_argument(
        "--latest-run-dir",
        type=Path,
        default=defaults.get("latest_run_dir"),
        help="Explicit training run directory used when --policy-path is omitted.",
    )
    parser.add_argument(
        "--train-output-root",
        type=Path,
        default=defaults.get("train_output_root"),
        help="Training output root used to infer the latest run.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Value substituted into --output-dir when it contains '{run_tag}'. "
            "Defaults to the current timestamp."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=defaults.get("output_dir"),
        help="Directory where rollout videos and summary are saved.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=defaults.get("num_rollouts", 20),
        help="Number of evaluation rollouts.",
    )
    parser.add_argument("--max-steps", type=int, default=defaults.get("max_steps", 120))
    parser.add_argument("--fps", type=int, default=defaults.get("fps", 20))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=defaults.get("n_action_steps"),
        help=(
            "Optional override for policy n_action_steps during rollout. "
            "Set to 1 for per-step replanning. Defaults to the checkpoint config."
        ),
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=defaults.get("success_threshold", 0.0),
        help="Only used by h_shape evaluation.",
    )
    parser.add_argument(
        "--max-action-step",
        type=float,
        default=defaults.get("max_action_step", 1.0),
        help="Clamp action magnitude to avoid implausibly large jumps.",
    )
    parser.add_argument(
        "--collision-mode",
        type=str,
        default=defaults.get("collision_mode", "reject"),
        choices=["reject", "detect"],
        help=(
            "Only used by braidedhub evaluation. "
            "`reject` blocks invalid moves; `detect` records them but allows penetration."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.get("device", "cuda"),
        help="cuda/cpu/mps",
    )

    randomize_group = parser.add_mutually_exclusive_group()
    randomize_group.add_argument(
        "--enable-randomize",
        dest="enable_randomize",
        action="store_true",
        help=(
            "Randomize the reset start state within "
            "the task start region instead of using the region center."
        ),
    )
    randomize_group.add_argument(
        "--disable-randomize",
        dest="enable_randomize",
        action="store_false",
        help="Disable randomized reset start states during evaluation.",
    )
    parser.set_defaults(enable_randomize=defaults.get("enable_randomize", False))

    if known_args.policy == "streaming_act":
        parser.add_argument(
            "--signature-backend",
            type=str,
            default=defaults.get("signature_backend", "auto"),
            choices=["auto", "signatory", "simple"],
            help="Backend for online path-signature computation during streaming eval.",
        )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(argv)
    args = parser.parse_args(argv)

    if isinstance(args.output_dir, Path):
        output_dir_s = str(args.output_dir)
    else:
        output_dir_s = args.output_dir
    if output_dir_s is not None and "{run_tag}" in output_dir_s:
        run_tag = args.run_tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_s = output_dir_s.format(run_tag=run_tag)
    args.output_dir = Path(output_dir_s)

    if args.train_output_root is not None and not isinstance(args.train_output_root, Path):
        args.train_output_root = Path(args.train_output_root)
    if args.policy_path is not None and not isinstance(args.policy_path, Path):
        args.policy_path = Path(args.policy_path)
    if args.latest_run_dir is not None and not isinstance(args.latest_run_dir, Path):
        args.latest_run_dir = Path(args.latest_run_dir)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import torch

    repo_root = Path(__file__).resolve().parents[2]
    ensure_streaming_act_importable(repo_root)

    try:
        from lerobot.policies.factory import make_pre_post_processors
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing LeRobot evaluation dependencies. Install the pip package first, "
            "for example `pip install lerobot`, and ensure torch is installed for "
            "your platform."
        ) from exc

    from lerobot_policy_streaming_act.configuration_act import ACTConfig, StreamingACTConfig
    from lerobot_policy_streaming_act.modeling_act import ACTPolicy, StreamingACTPolicy

    patch_lerobot_processor_factory(act_config_cls=ACTConfig)
    policy_cls = StreamingACTPolicy if args.policy == "streaming_act" else ACTPolicy

    policy_dir = resolve_eval_policy_path(
        policy_path=args.policy_path,
        latest_run_dir=args.latest_run_dir,
        train_output_root=args.train_output_root,
    )
    print(f"Using policy path: {policy_dir}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = policy_cls.from_pretrained(policy_dir)
    cfg = policy.config
    cfg.device = args.device
    validate_first_frame_anchor_support(
        env_name=args.env,
        use_first_frame_anchor=bool(getattr(cfg, "use_first_frame_anchor", False)),
    )
    validate_prefix_sequence_support(
        env_name=args.env,
        policy_name=args.policy,
        use_prefix_sequence_training=bool(
            getattr(cfg, "use_prefix_sequence_training", False)
        ),
    )
    validate_visual_prefix_memory_support(
        policy_name=args.policy,
        use_visual_prefix_memory=bool(
            getattr(cfg, "use_visual_prefix_memory", False)
        ),
    )
    validate_delta_signature_support(
        policy_name=args.policy,
        use_delta_signature=bool(getattr(cfg, "use_delta_signature", False)),
    )
    if args.n_action_steps is not None:
        cfg.n_action_steps = int(args.n_action_steps)
        if cfg.n_action_steps <= 0:
            raise ValueError(
                f"`--n-action-steps` must be positive, got {cfg.n_action_steps}."
            )
        if hasattr(cfg, "chunk_size") and cfg.n_action_steps > int(cfg.chunk_size):
            raise ValueError(
                "`--n-action-steps` cannot exceed the checkpoint chunk_size. "
                f"Got n_action_steps={cfg.n_action_steps}, chunk_size={cfg.chunk_size}."
            )
    policy.eval()
    policy.to(args.device)
    if args.n_action_steps is not None and hasattr(policy, "reset"):
        policy.reset()

    preprocessor_overrides = {
        "device_processor": {"device": args.device},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_dir,
        preprocessor_overrides=preprocessor_overrides,
    )

    env_module = get_env_module(args.env)
    env_module.evaluate_policy(
        policy_type=args.policy,
        args=args,
        policy=policy,
        cfg=cfg,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        policy_dir=policy_dir,
    )


if __name__ == "__main__":
    main()
