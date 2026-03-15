from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import warnings

from env import get_env_choices, get_env_module

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated*",
    category=UserWarning,
)


def ensure_streaming_act_importable(repo_root: Path) -> None:
    streaming_act_src = repo_root / "main/policy/lerobot_policy_streaming_act/src"
    if not streaming_act_src.exists():
        raise FileNotFoundError(
            f"Streaming ACT package source not found: {streaming_act_src}"
        )
    sys.path.insert(0, str(streaming_act_src))


def patch_lerobot_act_factory(streaming_policy_cls, streaming_config_cls) -> None:
    import lerobot.policies.factory as policy_factory

    original_get_policy_class = policy_factory.get_policy_class

    def get_policy_class_with_streaming_act(name: str):
        if name in {"act", "streaming_act"}:
            return streaming_policy_cls
        return original_get_policy_class(name)

    policy_factory.get_policy_class = get_policy_class_with_streaming_act
    policy_factory.ACTConfig = streaming_config_cls


def teardown_wandb_safely(exit_code: int) -> None:
    try:
        import wandb
    except Exception:
        return

    try:
        wandb.teardown(exit_code=exit_code)
    except BaseException as exc:
        print(f"[WARN] wandb teardown failed during shutdown: {exc}")


def validate_dataset_root(dataset_root: Path) -> None:
    required = [
        dataset_root / "meta/info.json",
        dataset_root / "meta/stats.json",
        dataset_root / "meta/episodes/chunk-000/file-000.parquet",
        dataset_root / "data/chunk-000/file-000.parquet",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_s = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Dataset path is missing required LeRobot v3.0 files:\n"
            f"{missing_s}\n"
            f"dataset_root={dataset_root}"
        )


def resolve_use_imagenet_stats(
    dataset_root: Path,
    disable_imagenet_stats: bool,
) -> bool:
    if disable_imagenet_stats:
        return False

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((dataset_root / "meta/stats.json").read_text(encoding="utf-8"))

    camera_keys = [
        key
        for key, spec in info.get("features", {}).items()
        if isinstance(spec, dict) and spec.get("dtype") in ("image", "video")
    ]
    missing_camera_stats = [key for key in camera_keys if key not in stats]
    if missing_camera_stats:
        print(
            "[WARN] meta/stats.json is missing camera stats keys required by "
            "LeRobot's ImageNet-stats override:\n"
            + "\n".join(f"  - {key}" for key in missing_camera_stats)
            + "\n[WARN] Auto-switching to --disable-imagenet-stats behavior."
        )
        return False
    return True


def resolve_signature_dim(
    dataset_root: Path,
    use_path_signature: bool,
    signature_dim: int,
) -> int:
    if not use_path_signature:
        return signature_dim

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    sig_key = "observation.path_signature"
    sig_spec = info.get("features", {}).get(sig_key)
    if sig_spec is None:
        raise KeyError(
            f"Dataset feature '{sig_key}' not found in {dataset_root / 'meta/info.json'}. "
            "Please run path-signature preprocessing first or disable path signature."
        )

    shape = sig_spec.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 1 or int(shape[0]) <= 0:
        raise ValueError(
            f"Invalid shape for '{sig_key}' in dataset info: {shape}. Expected [signature_dim]."
        )
    dataset_sig_dim = int(shape[0])
    if signature_dim > 0 and signature_dim != dataset_sig_dim:
        raise ValueError(
            f"signature_dim mismatch: cli={signature_dim} vs dataset={dataset_sig_dim} "
            f"for feature '{sig_key}'."
        )
    return dataset_sig_dim if signature_dim <= 0 else signature_dim


def resolve_history_length(dataset_root: Path, history_length: int) -> int:
    if history_length > 0:
        return history_length

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    episodes_file = dataset_root / "meta/episodes/chunk-000/file-000.parquet"

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        pq = None

    if pq is not None and episodes_file.exists():
        episode_table = pq.read_table(episodes_file, columns=["length"])
        episode_lengths = np.asarray(
            episode_table["length"].to_pylist(), dtype=np.int64
        )
        if episode_lengths.size == 0:
            raise ValueError(f"No episode lengths found in {episodes_file}.")
        return int(episode_lengths.max())

    total_frames = int(info.get("total_frames", 0))
    total_episodes = int(info.get("total_episodes", 0))
    if total_frames > 0 and total_episodes > 0 and total_frames % total_episodes == 0:
        inferred_length = total_frames // total_episodes
        if inferred_length > 0:
            print(
                "[WARN] pyarrow is unavailable, so max episode length was inferred "
                f"from info.json as total_frames / total_episodes = {inferred_length}."
            )
            return int(inferred_length)

    raise RuntimeError(
        "Could not auto-resolve history_length from dataset metadata. "
        "Install pyarrow or pass --history-length explicitly."
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
    env_module = get_env_module(known_args.env)
    defaults = env_module.get_train_defaults(known_args.policy)

    parser = argparse.ArgumentParser(
        description="Train LeRobot ACT or Streaming ACT on a selected environment dataset."
    )
    parser.add_argument("--env", choices=get_env_choices(), default=known_args.env)
    parser.add_argument(
        "--policy",
        choices=["act", "streaming_act"],
        default=known_args.policy,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=defaults["dataset_root"],
        help="Path to local LeRobotDataset v3.0 directory.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=defaults["dataset_repo_id"],
        help="Logical dataset repo_id used by LeRobot metadata APIs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=defaults["output_root"],
        help="Root folder for training outputs.",
    )
    parser.add_argument("--job-name", type=str, default=defaults["job_name"])
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help=(
            "Optional explicit Weights & Biases run name. "
            "Defaults to '<job-name>_s<seed>_<timestamp>'."
        ),
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument(
        "--eval-freq", type=int, default=-1, help="Set -1 to disable eval."
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu / mps")
    parser.add_argument(
        "--chunk-size", type=int, default=5, help="ACT action chunk size."
    )
    parser.add_argument(
        "--disable-imagenet-stats",
        action="store_true",
        help="Disable replacing visual stats with ImageNet stats.",
    )
    parser.add_argument("--wandb-project", type=str, default=defaults["wandb_project"])
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--disable-wandb", action="store_true")

    parser.add_argument(
        "--disable-path-signature",
        action="store_true",
        help="Disable path-signature token injection in StreamingACT.",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=0,
        help=(
            "History window size used by path-signature settings in config. "
            "Set 0 to auto-read the maximum episode length from the dataset."
        ),
    )
    parser.add_argument(
        "--signature-dim",
        type=int,
        default=0,
        help=("Path-signature feature dim. " "Set 0 to auto-read from meta/info.json."),
    )
    parser.add_argument("--signature-depth", type=int, default=3)
    parser.add_argument("--signature-hidden-dim", type=int, default=512)
    parser.add_argument("--signature-dropout", type=float, default=0.1)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(argv)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]

    try:
        from lerobot.configs.default import DatasetConfig, WandBConfig
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.scripts.lerobot_train import train
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing LeRobot training dependencies. Install the pip package first, "
            "for example `pip install lerobot`, and ensure torch is installed for "
            "your platform."
        ) from exc

    dataset_root = args.dataset_root.resolve()
    validate_dataset_root(dataset_root)
    use_imagenet_stats = resolve_use_imagenet_stats(
        dataset_root=dataset_root,
        disable_imagenet_stats=args.disable_imagenet_stats,
    )

    if args.policy == "streaming_act":
        ensure_streaming_act_importable(repo_root)
        from lerobot_policy_streaming_act.configuration_act import StreamingACTConfig
        from lerobot_policy_streaming_act.modeling_act import StreamingACTPolicy

        patch_lerobot_act_factory(
            streaming_policy_cls=StreamingACTPolicy,
            streaming_config_cls=StreamingACTConfig,
        )

        use_path_signature = not args.disable_path_signature
        resolved_history_length = resolve_history_length(
            dataset_root=dataset_root,
            history_length=args.history_length,
        )
        signature_dim = resolve_signature_dim(
            dataset_root=dataset_root,
            use_path_signature=use_path_signature,
            signature_dim=args.signature_dim,
        )
    else:
        from lerobot.policies.act.configuration_act import ACTConfig

        use_path_signature = False
        resolved_history_length = 0
        signature_dim = 0

    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_root / run_stamp).resolve()

    wandb_enable = (not args.disable_wandb) and (args.wandb_mode != "disabled")
    resolved_job_name = args.job_name
    if wandb_enable:
        resolved_job_name = (
            args.wandb_run_name
            if args.wandb_run_name
            else f"{args.job_name}_s{args.seed}_{run_stamp}"
        )

    if (
        wandb_enable
        and args.wandb_mode == "online"
        and "WANDB_API_KEY" not in os.environ
    ):
        print(
            "[WARN] WANDB_API_KEY not found in environment. "
            "If you are not already logged in, run `wandb login` first."
        )

    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=str(dataset_root),
        use_imagenet_stats=use_imagenet_stats,
    )

    if args.policy == "streaming_act":
        from lerobot_policy_streaming_act.configuration_act import StreamingACTConfig

        policy_cfg = StreamingACTConfig(
            device=args.device,
            push_to_hub=False,
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            chunk_size=args.chunk_size,
            n_action_steps=args.chunk_size,
            use_path_signature=use_path_signature,
            history_length=resolved_history_length,
            signature_dim=signature_dim,
            signature_depth=args.signature_depth,
            signature_hidden_dim=args.signature_hidden_dim,
            signature_dropout=args.signature_dropout,
        )
    else:
        from lerobot.policies.act.configuration_act import ACTConfig

        policy_cfg = ACTConfig(
            device=args.device,
            push_to_hub=False,
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            chunk_size=args.chunk_size,
            n_action_steps=args.chunk_size,
        )

    wandb_cfg = WandBConfig(
        enable=wandb_enable,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
    )

    cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=resolved_job_name,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        wandb=wandb_cfg,
    )

    print("Starting LeRobot imitation training with config:")
    print(f"- env: {args.env}")
    print(f"- policy: {args.policy}")
    print(f"- dataset_root: {dataset_root}")
    print(f"- dataset_repo_id: {args.dataset_repo_id}")
    print(f"- output_dir: {output_dir}")
    print(f"- device: {args.device}")
    print(f"- job_name: {resolved_job_name}")
    print(f"- steps: {args.steps}")
    print(f"- batch_size: {args.batch_size}")
    print(f"- use_imagenet_stats: {use_imagenet_stats}")
    if args.policy == "streaming_act":
        print(f"- use_path_signature: {use_path_signature}")
        if use_path_signature:
            print(
                f"- signature: dim={signature_dim}, depth={args.signature_depth}, "
                f"history={resolved_history_length}, hidden={args.signature_hidden_dim}, "
                f"dropout={args.signature_dropout}"
            )
    print(
        f"- wandb: enable={wandb_enable}, project={args.wandb_project}, mode={args.wandb_mode}"
    )

    try:
        train(cfg)
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user. Cleaning up wandb before exit.")
        teardown_wandb_safely(exit_code=130)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
