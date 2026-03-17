from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from env import get_env_choices, get_env_module
from eval_helpers import resolve_policy_dir


def ensure_streaming_act_importable(repo_root: Path) -> None:
    streaming_act_src = repo_root / "main/policy/lerobot_policy_streaming_act/src"
    if not streaming_act_src.exists():
        raise FileNotFoundError(
            f"Streaming ACT package source not found: {streaming_act_src}"
        )
    sys.path.insert(0, str(streaming_act_src))


def patch_lerobot_processor_factory(streaming_config_cls) -> None:
    import lerobot.policies.factory as policy_factory

    policy_factory.ACTConfig = streaming_config_cls


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env", choices=get_env_choices(), default="h_shape")
    bootstrap.add_argument(
        "--policy",
        choices=["act", "streaming_act"],
        default="act",
    )
    known_args, _ = bootstrap.parse_known_args(argv)
    env_module = get_env_module(known_args.env)
    defaults = env_module.get_eval_defaults(known_args.policy)

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
        required=True,
        help="Checkpoint dir, pretrained_model dir, or training run dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults["output_dir"],
        help="Directory where rollout videos and summary are saved.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=defaults["num_rollouts"],
        help="Number of evaluation rollouts.",
    )
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"])
    parser.add_argument("--fps", type=int, default=defaults["fps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=defaults["success_threshold"],
        help="Only used by h_shape evaluation.",
    )
    parser.add_argument(
        "--max-action-step",
        type=float,
        default=defaults["max_action_step"],
        help="Clamp action magnitude to avoid implausibly large jumps.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu/mps")
    parser.add_argument(
        "--signature-backend",
        type=str,
        default="auto",
        choices=["auto", "signatory", "simple"],
        help="Backend for online path-signature computation during streaming eval.",
    )
    parser.add_argument(
        "--enable-randomize",
        action="store_true",
        help=(
            "Randomize the reset start state within "
            "the task start region instead of using the region center."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(argv)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import torch

    repo_root = Path(__file__).resolve().parents[2]
    if args.policy == "streaming_act":
        ensure_streaming_act_importable(repo_root)

    try:
        from lerobot.policies.factory import make_pre_post_processors
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing LeRobot evaluation dependencies. Install the pip package first, "
            "for example `pip install lerobot`, and ensure torch is installed for "
            "your platform."
        ) from exc

    if args.policy == "streaming_act":
        from lerobot_policy_streaming_act.configuration_act import StreamingACTConfig
        from lerobot_policy_streaming_act.modeling_act import StreamingACTPolicy

        patch_lerobot_processor_factory(streaming_config_cls=StreamingACTConfig)
        policy_cls = StreamingACTPolicy
    else:
        from lerobot.policies.act.modeling_act import ACTPolicy

        policy_cls = ACTPolicy

    policy_dir = resolve_policy_dir(args.policy_path)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = policy_cls.from_pretrained(policy_dir)
    cfg = policy.config
    cfg.device = args.device
    policy.eval()
    policy.to(args.device)

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
