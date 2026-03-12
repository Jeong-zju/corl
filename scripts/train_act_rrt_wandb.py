import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import warnings

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated*",
    category=UserWarning,
)


def build_args():
    parser = argparse.ArgumentParser(
        description="Train LeRobot ACT on local rrt_connect_h_v30 dataset and sync metrics to Weights & Biases."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/zeno-ai/rrt_connect_h_v30"),
        help="Path to local LeRobotDataset v3.0 directory.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="zeno-ai/rrt_connect_h_v30",
        help="Logical dataset repo_id used by LeRobot metadata APIs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/train/rrt_connect_h_act"),
        help="Root folder for training outputs.",
    )
    parser.add_argument("--job-name", type=str, default="act_rrt_connect_h")
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
    parser.add_argument("--wandb-project", type=str, default="lerobot-rrt-act")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def ensure_lerobot_importable(repo_root: Path):
    lerobot_src = repo_root / "ACT-wholebody-torque/lerobot/src"
    if not lerobot_src.exists():
        raise FileNotFoundError(f"LeRobot source not found: {lerobot_src}")
    sys.path.insert(0, str(lerobot_src))


def validate_dataset_root(dataset_root: Path):
    required = [
        dataset_root / "meta/info.json",
        dataset_root / "meta/stats.json",
        dataset_root / "meta/episodes/chunk-000/file-000.parquet",
        dataset_root / "data/chunk-000/file-000.parquet",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_s = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "Dataset path is missing required LeRobot v3.0 files:\n"
            f"{missing_s}\n"
            f"dataset_root={dataset_root}"
        )


def resolve_use_imagenet_stats(
    dataset_root: Path, disable_imagenet_stats: bool
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

    missing_camera_stats = [k for k in camera_keys if k not in stats]
    if missing_camera_stats:
        print(
            "[WARN] meta/stats.json is missing camera stats keys required by "
            "LeRobot's ImageNet-stats override:\n"
            + "\n".join(f"  - {k}" for k in missing_camera_stats)
            + "\n[WARN] Auto-switching to --disable-imagenet-stats behavior."
        )
        return False

    return True


def main():
    args = build_args()

    repo_root = Path(__file__).resolve().parents[2]
    ensure_lerobot_importable(repo_root)

    try:
        from lerobot.configs.default import DatasetConfig, WandBConfig
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.scripts.lerobot_train import train
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing LeRobot training dependencies. Please install LeRobot first, for example:\n"
            "  cd ACT-wholebody-torque/lerobot\n"
            "  pip install -e .\n"
            "and ensure torch is installed for your platform."
        ) from exc

    dataset_root = args.dataset_root.resolve()
    validate_dataset_root(dataset_root)
    use_imagenet_stats = resolve_use_imagenet_stats(
        dataset_root=dataset_root,
        disable_imagenet_stats=args.disable_imagenet_stats,
    )

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

    if wandb_enable and args.wandb_mode == "online":
        if "WANDB_API_KEY" not in os.environ:
            print(
                "[WARN] WANDB_API_KEY not found in environment. "
                "If you are not already logged in, run `wandb login` first."
            )

    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=str(dataset_root),
        use_imagenet_stats=use_imagenet_stats,
    )
    policy_cfg = ACTConfig(
        device=args.device,
        push_to_hub=False,
        pretrained_backbone_weights=None,
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

    print("Starting LeRobot ACT imitation training with config:")
    print(f"- dataset_root: {dataset_root}")
    print(f"- dataset_repo_id: {args.dataset_repo_id}")
    print(f"- output_dir: {output_dir}")
    print(f"- device: {args.device}")
    print(f"- job_name: {resolved_job_name}")
    print(f"- steps: {args.steps}")
    print(f"- batch_size: {args.batch_size}")
    print(f"- use_imagenet_stats: {use_imagenet_stats}")
    print(
        f"- wandb: enable={wandb_enable}, project={args.wandb_project}, mode={args.wandb_mode}"
    )

    train(cfg)


if __name__ == "__main__":
    main()
