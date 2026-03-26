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
from policy_defaults import load_policy_mode_defaults

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated*",
    category=UserWarning,
)

FIRST_FRAME_ANCHOR_KEY = "observation.anchor_image"


def ensure_streaming_act_importable(repo_root: Path) -> None:
    streaming_act_src = repo_root / "main/policy/lerobot_policy_streaming_act/src"
    if not streaming_act_src.exists():
        raise FileNotFoundError(
            f"Streaming ACT package source not found: {streaming_act_src}"
        )
    sys.path.insert(0, str(streaming_act_src))


def patch_lerobot_act_factory(act_policy_cls, act_config_cls, streaming_policy_cls) -> None:
    import lerobot.policies.factory as policy_factory

    original_get_policy_class = policy_factory.get_policy_class

    def get_policy_class_with_local_act(name: str):
        if name == "act":
            return act_policy_cls
        if name == "streaming_act":
            return streaming_policy_cls
        return original_get_policy_class(name)

    policy_factory.get_policy_class = get_policy_class_with_local_act
    policy_factory.ACTConfig = act_config_cls


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
    use_imagenet_stats: bool,
) -> bool:
    if not use_imagenet_stats:
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


def validate_first_frame_anchor_support(
    *,
    env_name: str,
    use_first_frame_anchor: bool,
    context: str,
) -> None:
    if not use_first_frame_anchor:
        return
    if env_name != "braidedhub":
        raise NotImplementedError(
            "First-frame anchor support is currently implemented only for `braidedhub` "
            f"during {context}. Got env={env_name!r}."
        )


def validate_first_frame_anchor_dataset(dataset_root: Path, use_first_frame_anchor: bool) -> None:
    if not use_first_frame_anchor:
        return

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((dataset_root / "meta/stats.json").read_text(encoding="utf-8"))
    anchor_spec = info.get("features", {}).get(FIRST_FRAME_ANCHOR_KEY)
    if anchor_spec is None:
        raise KeyError(
            f"Dataset feature '{FIRST_FRAME_ANCHOR_KEY}' not found in {dataset_root / 'meta/info.json'}. "
            "Regenerate the dataset with "
            "`main/scripts/collect_imitation_dataset.py --env braidedhub --enable-first-frame-anchor`."
        )
    if anchor_spec.get("dtype") not in {"image", "video"}:
        raise ValueError(
            f"Dataset feature '{FIRST_FRAME_ANCHOR_KEY}' must be stored as image/video, "
            f"got dtype={anchor_spec.get('dtype')!r}."
        )
    if FIRST_FRAME_ANCHOR_KEY not in stats:
        raise KeyError(
            f"Dataset stats for '{FIRST_FRAME_ANCHOR_KEY}' are missing from {dataset_root / 'meta/stats.json'}. "
            "Regenerate the dataset so the anchor feature participates in normalization."
        )


def validate_prefix_sequence_support(
    *,
    env_name: str,
    policy_name: str,
    use_prefix_sequence_training: bool,
    context: str,
) -> None:
    if not use_prefix_sequence_training:
        return
    if policy_name != "streaming_act":
        raise NotImplementedError(
            "Prefix-sequence training is currently implemented only for `streaming_act`. "
            f"Got policy={policy_name!r} during {context}."
        )


def validate_prefix_sequence_dataset(
    dataset_root: Path,
    *,
    use_prefix_sequence_training: bool,
    use_path_signature: bool,
    use_delta_signature: bool,
) -> None:
    if not use_prefix_sequence_training:
        return

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((dataset_root / "meta/stats.json").read_text(encoding="utf-8"))
    features = info.get("features", {})

    state_spec = features.get("observation.state")
    if state_spec is None:
        raise KeyError(
            "Prefix-sequence mode requires dataset feature `observation.state`."
        )

    camera_keys = [
        key
        for key, spec in features.items()
        if isinstance(spec, dict)
        and spec.get("dtype") in {"image", "video"}
        and key != FIRST_FRAME_ANCHOR_KEY
    ]
    if not camera_keys:
        raise KeyError(
            "Prefix-sequence mode requires at least one regular observation image feature."
        )
    missing_camera_stats = [key for key in camera_keys if key not in stats]
    if missing_camera_stats:
        raise KeyError(
            "Prefix-sequence mode requires image stats for all regular observation cameras. "
            f"Missing: {missing_camera_stats}."
        )

    if "observation.state" not in stats:
        raise KeyError(
            "Prefix-sequence mode requires `observation.state` stats in meta/stats.json."
        )

    if use_path_signature:
        sig_key = "observation.path_signature"
        if sig_key not in features:
            raise KeyError(
                f"Prefix-sequence mode requires dataset feature `{sig_key}`. "
                "Regenerate the dataset with path-signature export enabled."
            )
        if sig_key not in stats:
            raise KeyError(
                f"Prefix-sequence mode requires dataset stats for `{sig_key}`."
            )
    if use_delta_signature:
        delta_sig_key = "observation.delta_signature"
        if delta_sig_key not in features:
            raise KeyError(
                f"Prefix-sequence mode requires dataset feature `{delta_sig_key}` "
                "when delta signatures are enabled."
            )
        if delta_sig_key not in stats:
            raise KeyError(
                f"Prefix-sequence mode requires dataset stats for `{delta_sig_key}`."
            )


def validate_delta_signature_dataset(
    dataset_root: Path,
    *,
    use_delta_signature: bool,
) -> None:
    if not use_delta_signature:
        return

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((dataset_root / "meta/stats.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    delta_sig_key = "observation.delta_signature"
    delta_sig_spec = features.get(delta_sig_key)
    if delta_sig_spec is None:
        raise KeyError(
            f"Dataset feature `{delta_sig_key}` not found in {dataset_root / 'meta/info.json'}. "
            "Regenerate the dataset with delta-signature export enabled."
        )
    shape = delta_sig_spec.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 1 or int(shape[0]) <= 0:
        raise ValueError(
            f"Invalid shape for `{delta_sig_key}` in dataset info: {shape}. "
            "Expected [signature_dim]."
        )
    if delta_sig_key not in stats:
        raise KeyError(
            f"Dataset stats for `{delta_sig_key}` are missing from {dataset_root / 'meta/stats.json'}."
        )


def validate_visual_prefix_memory_support(
    *,
    policy_name: str,
    use_visual_prefix_memory: bool,
    use_prefix_sequence_training: bool,
) -> None:
    if not use_visual_prefix_memory:
        return
    if policy_name != "streaming_act":
        raise NotImplementedError(
            "Visual prefix memory is currently implemented only for `streaming_act`. "
            f"Got policy={policy_name!r}."
        )
    if not use_prefix_sequence_training:
        raise ValueError(
            "`--enable-visual-prefix-memory` requires "
            "`--enable-prefix-sequence-training`."
        )


def build_policy_feature_overrides(
    dataset_root: Path,
    *,
    use_prefix_sequence_training: bool,
    prefix_train_max_steps: int,
    use_path_signature: bool,
    use_delta_signature: bool,
):
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))

    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    from lerobot_policy_streaming_act.prefix_sequence import (
        build_prefix_sequence_input_features,
    )

    dataset_features = dataset_to_policy_features(info.get("features", {}))
    output_features = {
        key: feature
        for key, feature in dataset_features.items()
        if feature.type is FeatureType.ACTION
    }
    input_features = {
        key: feature
        for key, feature in dataset_features.items()
        if key not in output_features
    }
    if use_prefix_sequence_training:
        input_features = build_prefix_sequence_input_features(
            base_input_features=input_features,
            prefix_train_max_steps=prefix_train_max_steps,
            use_path_signature=use_path_signature,
            use_delta_signature=use_delta_signature,
        )
    return input_features, output_features


def install_prefix_sequence_dataset_patch() -> None:
    import lerobot.datasets.factory as dataset_factory
    import lerobot.scripts.lerobot_train as lerobot_train_module
    from lerobot_policy_streaming_act.prefix_sequence import PrefixSequenceDataset

    if getattr(lerobot_train_module, "_prefix_sequence_patch_installed", False):
        return

    original_make_dataset = dataset_factory.make_dataset

    def make_dataset_with_prefix(cfg):
        dataset = original_make_dataset(cfg)
        policy_cfg = cfg.policy
        if not bool(getattr(policy_cfg, "use_prefix_sequence_training", False)):
            return dataset
        if isinstance(dataset, PrefixSequenceDataset):
            return dataset
        return PrefixSequenceDataset(
            dataset,
            prefix_train_max_steps=int(policy_cfg.prefix_train_max_steps),
            prefix_frame_stride=int(policy_cfg.prefix_frame_stride),
            prefix_pad_value=float(policy_cfg.prefix_pad_value),
            use_path_signature=bool(getattr(policy_cfg, "use_path_signature", False)),
            use_delta_signature=bool(getattr(policy_cfg, "use_delta_signature", False)),
        )

    dataset_factory.make_dataset = make_dataset_with_prefix
    lerobot_train_module.make_dataset = make_dataset_with_prefix
    lerobot_train_module._prefix_sequence_patch_installed = True


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env", choices=get_env_choices(), default="h_shape")
    bootstrap.add_argument(
        "--policy",
        choices=["act", "streaming_act"],
        default="act",
    )
    known_args, _ = bootstrap.parse_known_args(argv)
    defaults = load_policy_mode_defaults("train", known_args.env, known_args.policy)

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
        default=defaults.get("dataset_root"),
        help="Path to local LeRobotDataset v3.0 directory.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=defaults.get("dataset_repo_id"),
        help="Logical dataset repo_id used by LeRobot metadata APIs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=defaults.get("output_root"),
        help="Root folder for training outputs.",
    )
    parser.add_argument("--job-name", type=str, default=defaults.get("job_name"))
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=defaults.get("wandb_run_name"),
        help=(
            "Optional explicit Weights & Biases run name. "
            "Defaults to '<job-name>_s<seed>_<timestamp>'."
        ),
    )
    parser.add_argument("--steps", type=int, default=defaults.get("steps", 10000))
    parser.add_argument(
        "--batch-size", type=int, default=defaults.get("batch_size", 32)
    )
    parser.add_argument(
        "--num-workers", type=int, default=defaults.get("num_workers", 4)
    )
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--log-freq", type=int, default=defaults.get("log_freq", 50))
    parser.add_argument(
        "--save-freq", type=int, default=defaults.get("save_freq", 1000)
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=defaults.get("eval_freq", -1),
        help="Set -1 to disable eval.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.get("device", "cuda"),
        help="cuda / cpu / mps",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.get("chunk_size", 5),
        help="ACT action chunk size.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=defaults.get("n_action_steps", 1),
        help=(
            "Number of predicted actions executed before querying the policy again. "
            "Set to 1 for per-step replanning."
        ),
    )
    anchor_group = parser.add_mutually_exclusive_group()
    anchor_group.add_argument(
        "--enable-first-frame-anchor",
        dest="use_first_frame_anchor",
        action="store_true",
        help="Enable an episode-constant first-frame anchor token from observation.anchor_image.",
    )
    anchor_group.add_argument(
        "--disable-first-frame-anchor",
        dest="use_first_frame_anchor",
        action="store_false",
        help="Disable the first-frame anchor token input.",
    )
    parser.set_defaults(
        use_first_frame_anchor=defaults.get("use_first_frame_anchor", False),
    )

    imagenet_group = parser.add_mutually_exclusive_group()
    imagenet_group.add_argument(
        "--enable-imagenet-stats",
        dest="use_imagenet_stats",
        action="store_true",
        help="Replace visual dataset stats with ImageNet stats when available.",
    )
    imagenet_group.add_argument(
        "--disable-imagenet-stats",
        dest="use_imagenet_stats",
        action="store_false",
        help="Use dataset-provided visual stats instead of ImageNet stats.",
    )
    parser.set_defaults(
        use_imagenet_stats=defaults.get("use_imagenet_stats", True),
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default=defaults.get("wandb_project"),
    )
    parser.add_argument("--wandb-entity", type=str, default=defaults.get("wandb_entity"))
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=defaults.get("wandb_mode", "online"),
        choices=["online", "offline", "disabled"],
    )

    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--enable-wandb",
        dest="enable_wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    wandb_group.add_argument(
        "--disable-wandb",
        dest="enable_wandb",
        action="store_false",
        help="Disable Weights & Biases logging.",
    )
    parser.set_defaults(enable_wandb=defaults.get("enable_wandb", True))

    parser.add_argument(
        "--wandb-console",
        type=str,
        default=defaults.get("wandb_console", "off"),
        help="Value exported to WANDB_CONSOLE before training starts.",
    )
    parser.add_argument(
        "--wandb-service-wait",
        type=int,
        default=defaults.get("wandb_service_wait", 10),
        help="Value exported to WANDB__SERVICE_WAIT before training starts.",
    )

    if known_args.policy == "streaming_act":
        path_signature_group = parser.add_mutually_exclusive_group()
        path_signature_group.add_argument(
            "--enable-path-signature",
            dest="use_path_signature",
            action="store_true",
            help="Enable path-signature token injection in StreamingACT.",
        )
        path_signature_group.add_argument(
            "--disable-path-signature",
            dest="use_path_signature",
            action="store_false",
            help="Disable path-signature token injection in StreamingACT.",
        )
        parser.set_defaults(
            use_path_signature=defaults.get("use_path_signature", True),
        )
        parser.add_argument(
            "--history-length",
            type=int,
            default=defaults.get("history_length", 0),
            help=(
                "History window size used by path-signature settings in config. "
                "Set 0 to auto-read the maximum episode length from the dataset."
            ),
        )
        parser.add_argument(
            "--signature-dim",
            type=int,
            default=defaults.get("signature_dim", 0),
            help=(
                "Path-signature feature dim. "
                "Set 0 to auto-read from meta/info.json."
            ),
        )
        parser.add_argument(
            "--signature-depth",
            type=int,
            default=defaults.get("signature_depth", 3),
        )
        parser.add_argument(
            "--signature-hidden-dim",
            type=int,
            default=defaults.get("signature_hidden_dim", 512),
        )
        parser.add_argument(
            "--signature-dropout",
            type=float,
            default=defaults.get("signature_dropout", 0.1),
        )
        delta_signature_group = parser.add_mutually_exclusive_group()
        delta_signature_group.add_argument(
            "--enable-delta-signature",
            dest="use_delta_signature",
            action="store_true",
            help=(
                "Enable observation.delta_signature and optional delta-signature "
                "encoder-memory token support."
            ),
        )
        delta_signature_group.add_argument(
            "--disable-delta-signature",
            dest="use_delta_signature",
            action="store_false",
            help="Disable delta-signature inputs and token injection.",
        )
        parser.set_defaults(
            use_delta_signature=defaults.get("use_delta_signature", False),
        )
        prefix_group = parser.add_mutually_exclusive_group()
        prefix_group.add_argument(
            "--enable-prefix-sequence-training",
            dest="use_prefix_sequence_training",
            action="store_true",
            help=(
                "Enable prefix-sequence training inputs derived from the full episode "
                "prefix up to the current step."
            ),
        )
        prefix_group.add_argument(
            "--disable-prefix-sequence-training",
            dest="use_prefix_sequence_training",
            action="store_false",
            help="Disable prefix-sequence training inputs.",
        )
        parser.set_defaults(
            use_prefix_sequence_training=defaults.get("use_prefix_sequence_training", False),
        )
        parser.add_argument(
            "--prefix-train-max-steps",
            type=int,
            default=defaults.get("prefix_train_max_steps", 32),
            help=(
                "Maximum number of prefix elements kept per training sample. "
                "Prefix tensors are right padded to this length."
            ),
        )
        parser.add_argument(
            "--prefix-frame-stride",
            type=int,
            default=defaults.get("prefix_frame_stride", 1),
            help=(
                "Stride used when subsampling the episode prefix. "
                "The current step is always kept as the last valid element."
            ),
        )
        parser.add_argument(
            "--prefix-pad-value",
            type=float,
            default=defaults.get("prefix_pad_value", 0.0),
            help="Padding value used for prefix state/signature tensors.",
        )
        visual_prefix_memory_group = parser.add_mutually_exclusive_group()
        visual_prefix_memory_group.add_argument(
            "--enable-visual-prefix-memory",
            dest="use_visual_prefix_memory",
            action="store_true",
            help=(
                "Enable GRU-style visual prefix memory tokens built from prefix "
                "images and prefix states."
            ),
        )
        visual_prefix_memory_group.add_argument(
            "--disable-visual-prefix-memory",
            dest="use_visual_prefix_memory",
            action="store_false",
            help="Disable the visual prefix memory token.",
        )
        parser.set_defaults(
            use_visual_prefix_memory=defaults.get("use_visual_prefix_memory", False),
        )
        parser.add_argument(
            "--num-memory-slots",
            type=int,
            default=defaults.get("num_memory_slots", 1),
            help=(
                "Number of visual prefix memory slots. Each slot keeps an "
                "independent GRU-style memory state."
            ),
        )
        signature_conditioned_memory_group = parser.add_mutually_exclusive_group()
        signature_conditioned_memory_group.add_argument(
            "--enable-signature-conditioned-visual-prefix-memory",
            dest="use_signature_conditioned_visual_prefix_memory",
            action="store_true",
            help=(
                "Condition visual prefix memory updates on path signatures and, "
                "when enabled, delta signatures."
            ),
        )
        signature_conditioned_memory_group.add_argument(
            "--disable-signature-conditioned-visual-prefix-memory",
            dest="use_signature_conditioned_visual_prefix_memory",
            action="store_false",
            help="Disable signature-conditioned visual prefix memory updates.",
        )
        parser.set_defaults(
            use_signature_conditioned_visual_prefix_memory=defaults.get(
                "use_signature_conditioned_visual_prefix_memory", False
            ),
        )
        memory_conditioning_group = parser.add_mutually_exclusive_group()
        memory_conditioning_group.add_argument(
            "--enable-memory-conditioned-encoder-film",
            dest="use_memory_conditioned_encoder_film",
            action="store_true",
            help=(
                "Let the pooled visual prefix memory FiLM-modulate the current-step "
                "ACT encoder tokens before the transformer encoder."
            ),
        )
        memory_conditioning_group.add_argument(
            "--disable-memory-conditioned-encoder-film",
            dest="use_memory_conditioned_encoder_film",
            action="store_false",
            help="Disable memory-conditioned encoder FiLM modulation.",
        )
        parser.set_defaults(
            use_memory_conditioned_encoder_film=defaults.get(
                "use_memory_conditioned_encoder_film", False
            ),
        )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    if not isinstance(args.dataset_root, Path):
        args.dataset_root = Path(args.dataset_root)
    if not isinstance(args.output_root, Path):
        args.output_root = Path(args.output_root)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    ensure_streaming_act_importable(repo_root)

    os.environ["WANDB_CONSOLE"] = str(args.wandb_console)
    os.environ["WANDB__SERVICE_WAIT"] = str(args.wandb_service_wait)

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
    env_module = get_env_module(args.env)
    if hasattr(env_module, "validate_training_dataset_root"):
        env_module.validate_training_dataset_root(dataset_root)
    use_first_frame_anchor = bool(args.use_first_frame_anchor)
    validate_first_frame_anchor_support(
        env_name=args.env,
        use_first_frame_anchor=use_first_frame_anchor,
        context="training",
    )
    validate_first_frame_anchor_dataset(
        dataset_root=dataset_root,
        use_first_frame_anchor=use_first_frame_anchor,
    )
    use_imagenet_stats = resolve_use_imagenet_stats(
        dataset_root=dataset_root,
        use_imagenet_stats=args.use_imagenet_stats,
    )

    from lerobot_policy_streaming_act.configuration_act import ACTConfig, StreamingACTConfig
    from lerobot_policy_streaming_act.modeling_act import ACTPolicy, StreamingACTPolicy

    patch_lerobot_act_factory(
        act_policy_cls=ACTPolicy,
        act_config_cls=ACTConfig,
        streaming_policy_cls=StreamingACTPolicy,
    )

    if args.policy == "streaming_act":
        use_path_signature = args.use_path_signature
        use_delta_signature = bool(args.use_delta_signature)
        use_prefix_sequence_training = bool(args.use_prefix_sequence_training)
        use_visual_prefix_memory = bool(args.use_visual_prefix_memory)
        validate_prefix_sequence_support(
            env_name=args.env,
            policy_name=args.policy,
            use_prefix_sequence_training=use_prefix_sequence_training,
            context="training",
        )
        resolved_history_length = resolve_history_length(
            dataset_root=dataset_root,
            history_length=args.history_length,
        )
        signature_dim = resolve_signature_dim(
            dataset_root=dataset_root,
            use_path_signature=use_path_signature,
            signature_dim=args.signature_dim,
        )
        validate_delta_signature_dataset(
            dataset_root=dataset_root,
            use_delta_signature=use_delta_signature,
        )
        validate_prefix_sequence_dataset(
            dataset_root=dataset_root,
            use_prefix_sequence_training=use_prefix_sequence_training,
            use_path_signature=use_path_signature,
            use_delta_signature=use_delta_signature,
        )
        validate_visual_prefix_memory_support(
            policy_name=args.policy,
            use_visual_prefix_memory=use_visual_prefix_memory,
            use_prefix_sequence_training=use_prefix_sequence_training,
        )
    else:
        use_path_signature = False
        use_delta_signature = False
        use_prefix_sequence_training = False
        use_visual_prefix_memory = False
        resolved_history_length = 0
        signature_dim = 0

    if use_prefix_sequence_training:
        install_prefix_sequence_dataset_patch()

    input_features_override = None
    output_features_override = None
    if use_prefix_sequence_training:
        input_features_override, output_features_override = build_policy_feature_overrides(
            dataset_root=dataset_root,
            use_prefix_sequence_training=use_prefix_sequence_training,
            prefix_train_max_steps=int(args.prefix_train_max_steps),
            use_path_signature=use_path_signature,
            use_delta_signature=use_delta_signature,
        )

    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_root / run_stamp).resolve()

    wandb_enable = args.enable_wandb and (args.wandb_mode != "disabled")
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
        policy_cfg = StreamingACTConfig(
            device=args.device,
            push_to_hub=False,
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            use_first_frame_anchor=use_first_frame_anchor,
            use_path_signature=use_path_signature,
            history_length=resolved_history_length,
            signature_dim=signature_dim,
            signature_depth=args.signature_depth,
            signature_hidden_dim=args.signature_hidden_dim,
            signature_dropout=args.signature_dropout,
            use_delta_signature=use_delta_signature,
            use_prefix_sequence_training=use_prefix_sequence_training,
            prefix_train_max_steps=(
                int(args.prefix_train_max_steps) if use_prefix_sequence_training else 32
            ),
            prefix_frame_stride=(
                int(args.prefix_frame_stride) if use_prefix_sequence_training else 1
            ),
            prefix_pad_value=(
                float(args.prefix_pad_value) if use_prefix_sequence_training else 0.0
            ),
            use_visual_prefix_memory=use_visual_prefix_memory,
            use_signature_conditioned_visual_prefix_memory=bool(
                args.use_signature_conditioned_visual_prefix_memory
            ),
            use_memory_conditioned_encoder_film=bool(
                args.use_memory_conditioned_encoder_film
            ),
            num_memory_slots=int(args.num_memory_slots),
            input_features=input_features_override,
            output_features=output_features_override,
        )
    else:
        policy_cfg = ACTConfig(
            device=args.device,
            push_to_hub=False,
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            use_first_frame_anchor=use_first_frame_anchor,
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
    print(
        f"- action_execution: chunk_size={args.chunk_size}, "
        f"n_action_steps={args.n_action_steps}"
    )
    print(f"- use_imagenet_stats: {use_imagenet_stats}")
    print(f"- use_first_frame_anchor: {use_first_frame_anchor}")
    if args.policy == "streaming_act":
        print(f"- use_path_signature: {use_path_signature}")
        if use_path_signature:
            print(
                f"- signature: dim={signature_dim}, depth={args.signature_depth}, "
                f"history={resolved_history_length}, hidden={args.signature_hidden_dim}, "
                f"dropout={args.signature_dropout}"
            )
        print(f"- use_delta_signature: {use_delta_signature}")
        print(f"- use_prefix_sequence_training: {use_prefix_sequence_training}")
        if use_prefix_sequence_training:
            print(
                f"- prefix_sequence: max_steps={args.prefix_train_max_steps}, "
                f"stride={args.prefix_frame_stride}, pad_value={args.prefix_pad_value}"
            )
        print(f"- use_visual_prefix_memory: {use_visual_prefix_memory}")
        if use_visual_prefix_memory:
            print(
                "- visual_prefix_memory: "
                f"num_memory_slots={args.num_memory_slots}, "
                "signature_conditioned="
                f"{bool(args.use_signature_conditioned_visual_prefix_memory)}, "
                "encoder_film="
                f"{bool(args.use_memory_conditioned_encoder_film)}"
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
