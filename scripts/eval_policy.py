from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from dataset_utils import (
    DEFAULT_LOCAL_DATA_ROOT,
    build_dataset_split,
    find_dataset_split_file,
    infer_dataset_repo_id,
    load_dataset_split,
    resolve_dataset_root,
    resolve_episode_indices_from_dataset_info,
    validate_dataset_root,
)
from env import get_env_choices, get_env_module
from eval_helpers import (
    build_prefix_sequence_eval_inputs,
    compute_delta_signature_step_np,
    compute_signatory_signature_np,
    compute_simple_signature_np,
    ensure_prefix_sequence_batch_dims,
    resolve_eval_policy_path,
    resolve_signature_backend,
    write_summary,
)
from policy_defaults import (
    load_policy_mode_defaults,
    load_policy_mode_defaults_for_dataset,
)


FIRST_FRAME_ANCHOR_KEY = "observation.anchor_image"
DEFAULT_PATH_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_DELTA_SIGNATURE_KEY = "observation.delta_signature"


def maybe_create_tqdm(*, total: int, desc: str, unit: str):
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit)


def progress_write(progress, message: str) -> None:
    if progress is not None and hasattr(progress, "write"):
        progress.write(message)
        return
    print(message)


def format_elapsed_s(elapsed_s: float) -> str:
    return f"{elapsed_s:.1f}s"


def ensure_streaming_act_importable(repo_root: Path) -> None:
    streaming_act_src = repo_root / "main/policy/lerobot_policy_streaming_act/src"
    if not streaming_act_src.exists():
        raise FileNotFoundError(
            f"Streaming ACT package source not found: {streaming_act_src}"
        )
    sys.path.insert(0, str(streaming_act_src))


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


def default_policy_series_name(policy_name: str) -> str:
    return str(policy_name).replace("_", "-")


def normalize_output_path_part(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "item"


def default_dataset_output_subdir(dataset_selector: str | None) -> Path | None:
    if not dataset_selector:
        return None

    raw = str(dataset_selector).strip().replace("\\", "/")
    if not raw:
        return None
    if raw.startswith("./"):
        raw = raw[2:]
    for prefix in ("main/data/", "data/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    marker = "/main/data/"
    if marker in raw:
        raw = raw.split(marker, 1)[1]

    parts = [
        normalize_output_path_part(part)
        for part in raw.split("/")
        if part not in {"", ".", ".."}
    ]
    if not parts:
        return None
    return Path(*parts)


def default_train_output_root(
    policy_name: str,
    dataset_selector: str | None = None,
) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root / "main" / "outputs" / "train"
    dataset_subdir = default_dataset_output_subdir(dataset_selector)
    if dataset_subdir is not None:
        return base / dataset_subdir / default_policy_series_name(policy_name)
    return base / default_policy_series_name(policy_name)


def default_eval_output_dir(
    policy_name: str,
    dataset_selector: str | None = None,
) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root / "main" / "outputs" / "eval"
    dataset_subdir = default_dataset_output_subdir(dataset_selector)
    if dataset_subdir is not None:
        return str(
            base / dataset_subdir / default_policy_series_name(policy_name) / "{run_tag}"
        )
    return str(base / default_policy_series_name(policy_name) / "{run_tag}")


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--dataset", type=str, default=None)
    bootstrap.add_argument(
        "--env",
        choices=get_env_choices(),
        default=None,
    )
    bootstrap.add_argument(
        "--policy",
        choices=["act", "streaming_act"],
        default="act",
    )
    known_args, _ = bootstrap.parse_known_args(argv)
    defaults = {}
    dataset_train_defaults = {}
    defaults_path = None
    if known_args.dataset:
        defaults, defaults_path = load_policy_mode_defaults_for_dataset(
            mode="eval",
            dataset_selector=known_args.dataset,
            policy_name=known_args.policy,
        )
        dataset_train_defaults, _ = load_policy_mode_defaults_for_dataset(
            mode="train",
            dataset_selector=known_args.dataset,
            policy_name=known_args.policy,
        )
    if known_args.env:
        try:
            env_defaults = load_policy_mode_defaults(
                mode="eval",
                env_name=known_args.env,
                policy_name=known_args.policy,
            )
        except FileNotFoundError:
            env_defaults = {}
        defaults.update(env_defaults)

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a LeRobot ACT or Streaming ACT checkpoint either with env "
            "rollouts (`--env`) or on a held-out dataset split (`--dataset`)."
        )
    )
    parser.add_argument(
        "--policy",
        choices=["act", "streaming_act"],
        default=known_args.policy,
    )
    parser.add_argument(
        "--env",
        choices=get_env_choices(),
        default=None,
        help=(
            "Optional simulator environment for online rollout evaluation. "
            "If omitted, eval_policy runs offline against a held-out dataset split."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=known_args.dataset,
        help=(
            "Dataset ID or path under main/data. This value is also used to resolve "
            "`bash/defaults/<dataset_key>/<policy>.yaml` when present. "
            "If omitted in dataset mode, reuse the dataset recorded in the nearest "
            "dataset_split.json next to the training run."
        ),
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=defaults.get(
            "dataset_repo_id",
            dataset_train_defaults.get("dataset_repo_id"),
        ),
        help="Optional logical repo_id override used when loading the local LeRobot dataset.",
    )
    parser.add_argument(
        "--local-data-root",
        type=Path,
        default=defaults.get("local_data_root", DEFAULT_LOCAL_DATA_ROOT),
        help="Root directory used to resolve --dataset when a relative dataset ID is provided.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help=(
            "Checkpoint dir, pretrained_model dir, or training run dir. "
            "If omitted, the latest run under --train-output-root is used."
        ),
    )
    parser.add_argument(
        "--latest-run-dir",
        type=Path,
        default=None,
        help="Explicit training run directory used when --policy-path is omitted.",
    )
    parser.add_argument(
        "--train-output-root",
        type=Path,
        default=defaults.get(
            "train_output_root",
            default_train_output_root(known_args.policy, known_args.dataset),
        ),
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
        default=defaults.get(
            "output_dir",
            default_eval_output_dir(known_args.policy, known_args.dataset),
        ),
        help="Directory where evaluation artifacts are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.get("seed", 42),
        help="Random seed used by env rollouts and any stochastic split reconstruction.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=defaults.get("n_action_steps"),
        help=(
            "Optional override for policy n_action_steps during evaluation. "
            "Set to 1 for per-step replanning. Defaults to the checkpoint config."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.get("device", "cuda"),
        help="cuda/cpu/mps",
    )

    parser.add_argument(
        "--eval-split",
        type=str,
        default=defaults.get("eval_split", "test"),
        choices=["train", "test"],
        help="Held-out split to evaluate in dataset mode.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=defaults.get("test_ratio"),
        help=(
            "Optional held-out ratio used when no saved dataset_split.json is found. "
            "If omitted, dataset mode falls back to the dataset's info.json split metadata."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=defaults.get("split_seed", 42),
        help="Random seed used together with --test-ratio in dataset mode.",
    )
    split_shuffle_group = parser.add_mutually_exclusive_group()
    split_shuffle_group.add_argument(
        "--shuffle-split-episodes",
        dest="split_shuffle",
        action="store_true",
        help="Shuffle episode IDs before applying the train/test split ratio.",
    )
    split_shuffle_group.add_argument(
        "--preserve-split-order",
        dest="split_shuffle",
        action="store_false",
        help="Split train/test episodes by original episode order without shuffling.",
    )
    parser.set_defaults(split_shuffle=bool(defaults.get("split_shuffle", True)))
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=defaults.get("max_episodes"),
        help="Optional cap on the number of evaluated episodes in dataset mode.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=defaults.get("max_steps_per_episode"),
        help="Optional cap on the number of timesteps evaluated per episode in dataset mode.",
    )

    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=defaults.get("num_rollouts", 20),
        help="Number of evaluation rollouts in env mode.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=defaults.get("max_steps", 120),
        help="Maximum rollout length in env mode.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=defaults.get("fps", 20),
        help="Video fps in env rollout mode.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=defaults.get("task", defaults.get("cil")),
        help=(
            "Optional Meta-World task subset for env mode. Use a comma-separated "
            "task list such as `assembly-v3,dial-turn-v3,handle-press-side-v3`."
        ),
    )
    parser.add_argument(
        "--cil",
        dest="task",
        type=str,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-episodes-rendered",
        type=int,
        default=defaults.get("max_episodes_rendered"),
        help=(
            "Maximum number of rollout videos to save per task in env mode when "
            "supported by the environment evaluator."
        ),
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=defaults.get("success_threshold", 0.0),
        help="Only used by h_shape env evaluation.",
    )
    parser.add_argument(
        "--max-action-step",
        type=float,
        default=defaults.get("max_action_step", 1.0),
        help="Clamp action magnitude during env rollout evaluation.",
    )
    parser.add_argument(
        "--collision-mode",
        type=str,
        default=defaults.get("collision_mode", "reject"),
        choices=["reject", "detect"],
        help=(
            "Only used by braidedhub env evaluation. "
            "`reject` blocks invalid moves; `detect` records them but allows penetration."
        ),
    )
    randomize_group = parser.add_mutually_exclusive_group()
    randomize_group.add_argument(
        "--enable-randomize",
        dest="enable_randomize",
        action="store_true",
        help="Randomize reset start states during env evaluation.",
    )
    randomize_group.add_argument(
        "--disable-randomize",
        dest="enable_randomize",
        action="store_false",
        help="Disable randomized reset start states during env evaluation.",
    )
    parser.set_defaults(enable_randomize=bool(defaults.get("enable_randomize", False)))

    if known_args.policy == "streaming_act":
        parser.add_argument(
            "--signature-backend",
            type=str,
            default=defaults.get("signature_backend", "auto"),
            choices=["auto", "signatory", "simple"],
            help=(
                "Backend for online path-signature computation during env evaluation "
                "and for dataset evaluation when the dataset lacks signature features."
            ),
        )
    parser.set_defaults(
        _policy_defaults_path=(
            None if defaults_path is None else str(defaults_path)
        ),
        _policy_defaults_dataset_root=dataset_train_defaults.get("dataset_root"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(argv)
    args = parser.parse_args(argv)

    output_dir_s = str(args.output_dir)
    if "{run_tag}" in output_dir_s:
        run_tag = args.run_tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_s = output_dir_s.format(run_tag=run_tag)
    args.output_dir = Path(output_dir_s)

    for attr in ("local_data_root", "train_output_root", "policy_path", "latest_run_dir"):
        value = getattr(args, attr)
        if value is not None and not isinstance(value, Path):
            setattr(args, attr, Path(value))
    return args


def select_visual_observation_keys(cfg) -> list[str]:
    visual_features = getattr(cfg, "visual_observation_features", None)
    if visual_features is None:
        visual_features = getattr(cfg, "image_features", {})
    keys = list(visual_features)
    if not keys:
        raise RuntimeError("Policy has no visual observation input features.")
    return keys


def resolve_state_key(cfg) -> str:
    input_features = getattr(cfg, "input_features", None)
    if isinstance(input_features, dict):
        if "observation.state" in input_features:
            return "observation.state"
        for key in input_features:
            if key.endswith(".state"):
                return str(key)
    return "observation.state"


def resolve_env_state_key(cfg) -> str | None:
    input_features = getattr(cfg, "input_features", None)
    if isinstance(input_features, dict):
        if "observation.environment_state" in input_features:
            return "observation.environment_state"
        for key in input_features:
            if key.endswith(".environment_state"):
                return str(key)
    return None


def resolve_action_key(cfg) -> str:
    output_features = getattr(cfg, "output_features", None)
    if isinstance(output_features, dict):
        if "action" in output_features:
            return "action"
        if len(output_features) == 1:
            return str(next(iter(output_features)))
    return "action"


def as_tensor_copy(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().clone()
    return torch.as_tensor(value)


def tensor_to_numpy_vector(value) -> np.ndarray:
    import torch

    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return np.asarray(array, dtype=np.float32).reshape(-1)


def build_episode_groups(dataset) -> list[tuple[int, list[int]]]:
    episode_values = dataset.hf_dataset["episode_index"]
    groups: dict[int, list[int]] = {}
    order: list[int] = []
    for rel_idx, episode_value in enumerate(episode_values):
        episode_index = int(episode_value)
        if episode_index not in groups:
            groups[episode_index] = []
            order.append(episode_index)
        groups[episode_index].append(rel_idx)
    return [(episode_index, groups[episode_index]) for episode_index in order]


def compute_online_signature_prefix(
    state_history: deque[np.ndarray],
    sig_depth: int,
    signature_backend: str,
) -> np.ndarray:
    if len(state_history) == 0:
        raise ValueError("state_history is empty; cannot compute path signature.")
    window = np.stack(list(state_history), axis=0).astype(np.float32, copy=False)
    if signature_backend == "signatory":
        return compute_signatory_signature_np(window, sig_depth)
    return compute_simple_signature_np(window, sig_depth)


def resolve_dataset_selection(
    *,
    args,
    policy_dir: Path,
) -> tuple[Path, str, list[int], str]:
    saved_split_path = find_dataset_split_file(policy_dir)
    saved_split = (
        None if saved_split_path is None else load_dataset_split(saved_split_path)
    )

    if args.dataset is not None:
        defaults_dataset_root = getattr(args, "_policy_defaults_dataset_root", None)
        try:
            dataset_root = resolve_dataset_root(
                args.dataset,
                local_data_root=args.local_data_root.resolve(),
            )
        except FileNotFoundError:
            if not defaults_dataset_root:
                raise
            dataset_root = resolve_dataset_root(
                defaults_dataset_root,
                local_data_root=args.local_data_root.resolve(),
            )
        dataset_repo_id = (
            str(args.dataset_repo_id)
            if args.dataset_repo_id
            else infer_dataset_repo_id(
                dataset_root,
                local_data_root=args.local_data_root.resolve(),
            )
        )
    elif saved_split is not None:
        dataset_root = Path(saved_split.dataset_root).resolve()
        dataset_repo_id = str(saved_split.dataset_repo_id)
    else:
        raise ValueError(
            "Could not infer the evaluation dataset. Pass --dataset explicitly or "
            "evaluate a checkpoint inside a training run that contains dataset_split.json."
        )

    validate_dataset_root(dataset_root)

    if (
        saved_split is not None
        and dataset_root.resolve() == Path(saved_split.dataset_root).resolve()
        and args.test_ratio is None
    ):
        episode_indices = (
            saved_split.train_episode_indices
            if args.eval_split == "train"
            else saved_split.test_episode_indices
        )
        return (
            dataset_root,
            dataset_repo_id,
            [int(ep) for ep in episode_indices],
            f"training_artifact:{saved_split_path}",
        )

    if args.test_ratio is not None:
        split_spec = build_dataset_split(
            dataset_arg=args.dataset or dataset_repo_id,
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            test_ratio=float(args.test_ratio),
            split_seed=int(args.split_seed),
            split_shuffle=bool(args.split_shuffle),
        )
        episode_indices = (
            split_spec.train_episode_indices
            if args.eval_split == "train"
            else split_spec.test_episode_indices
        )
        return (
            dataset_root,
            dataset_repo_id,
            [int(ep) for ep in episode_indices],
            "cli_ratio_split",
        )

    split_name = "train" if args.eval_split == "train" else "test"
    episode_indices = resolve_episode_indices_from_dataset_info(dataset_root, split_name)
    if episode_indices is None:
        raise ValueError(
            "Could not resolve evaluation episodes from dataset metadata. "
            "Pass --test-ratio/--split-seed to reconstruct the split or evaluate a "
            "checkpoint inside a training run that contains dataset_split.json."
        )
    return dataset_root, dataset_repo_id, [int(ep) for ep in episode_indices], "dataset_info"


def run_env_evaluation(
    *,
    args,
    policy,
    cfg,
    preprocessor,
    postprocessor,
    policy_dir: Path,
) -> None:
    validate_first_frame_anchor_support(
        env_name=args.env,
        use_first_frame_anchor=bool(getattr(cfg, "use_first_frame_anchor", False)),
    )
    validate_prefix_sequence_support(
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


def run_dataset_evaluation(
    *,
    args,
    policy,
    cfg,
    preprocessor,
    postprocessor,
    policy_dir: Path,
) -> None:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root, dataset_repo_id, episode_indices, split_source = resolve_dataset_selection(
        args=args,
        policy_dir=policy_dir,
    )
    if not episode_indices:
        raise ValueError(
            f"No episodes were resolved for eval_split={args.eval_split!r}."
        )

    total_resolved_episodes = len(episode_indices)
    selected_episode_indices = [int(ep) for ep in episode_indices]
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("`--max-episodes` must be positive when provided.")
        selected_episode_indices = selected_episode_indices[: args.max_episodes]
    if not selected_episode_indices:
        raise RuntimeError("The selected dataset split produced zero episodes.")

    dataset_load_start_s = time.perf_counter()
    print(
        "[load] Building dataset view: "
        f"selected_episodes={len(selected_episode_indices)}/{total_resolved_episodes}, "
        f"dataset_root={dataset_root}"
    )
    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=str(dataset_root),
        episodes=selected_episode_indices,
        image_transforms=None,
    )
    print(
        "[timing] Dataset view loaded in "
        f"{format_elapsed_s(time.perf_counter() - dataset_load_start_s)} "
        f"({dataset.num_episodes} episodes, {dataset.num_frames} frames)"
    )
    episode_groups = build_episode_groups(dataset)
    if not episode_groups:
        raise RuntimeError("The selected dataset split produced zero decodable episodes.")

    print(
        "Resolved dataset split: "
        f"source={split_source}, dataset_root={dataset_root}, "
        f"episodes={len(episode_groups)}/{total_resolved_episodes}"
    )

    visual_keys = select_visual_observation_keys(cfg)
    state_key = resolve_state_key(cfg)
    env_state_key = resolve_env_state_key(cfg)
    action_key = resolve_action_key(cfg)

    use_path_signature = bool(
        args.policy == "streaming_act" and getattr(cfg, "use_path_signature", False)
    )
    use_prefix_sequence_training = bool(
        args.policy == "streaming_act"
        and getattr(cfg, "use_prefix_sequence_training", False)
    )
    use_visual_prefix_memory = bool(
        args.policy == "streaming_act"
        and getattr(cfg, "use_visual_prefix_memory", False)
    )
    use_delta_signature = bool(
        args.policy == "streaming_act" and getattr(cfg, "use_delta_signature", False)
    )
    use_first_frame_anchor = bool(getattr(cfg, "use_first_frame_anchor", False))
    build_explicit_prefix_eval_inputs = (
        use_prefix_sequence_training and not use_visual_prefix_memory
    )
    signature_backend = None
    if use_path_signature:
        signature_backend = resolve_signature_backend(
            getattr(args, "signature_backend", "auto")
        )
        print(
            "[info] path-signature eval enabled: "
            f"backend={signature_backend}, depth={cfg.signature_depth}, dim={cfg.signature_dim}"
        )
    if use_delta_signature:
        print(
            "[info] delta-signature eval enabled: "
            f"key={DEFAULT_DELTA_SIGNATURE_KEY}, first_step=zeros"
        )
    if build_explicit_prefix_eval_inputs:
        print(
            "[info] explicit prefix-sequence eval enabled: "
            f"max_steps={cfg.prefix_train_max_steps}, stride={cfg.prefix_frame_stride}, "
            f"cameras={visual_keys}"
        )
    elif use_visual_prefix_memory:
        print(
            "[info] visual prefix memory eval enabled: "
            "the policy updates recurrent memory from the true current observation at each step"
        )
    if use_first_frame_anchor:
        print(
            "[info] first-frame anchor eval enabled: "
            f"key={FIRST_FRAME_ANCHOR_KEY}, fallback_camera={visual_keys[0]}"
        )

    action_dim: int | None = None
    total_abs_error: np.ndarray | None = None
    total_sq_error: np.ndarray | None = None
    total_l2_error = 0.0
    total_cosine = 0.0
    total_cosine_count = 0
    total_steps = 0
    results: list[dict[str, object]] = []
    warned_anchor_fallback = False
    planned_steps_per_episode: list[int] = []
    for _, rel_indices in episode_groups:
        planned_steps = len(rel_indices)
        if args.max_steps_per_episode is not None:
            if args.max_steps_per_episode <= 0:
                raise ValueError(
                    "`--max-steps-per-episode` must be positive when provided."
                )
            planned_steps = min(planned_steps, int(args.max_steps_per_episode))
        planned_steps_per_episode.append(planned_steps)
    total_planned_steps = int(sum(planned_steps_per_episode))
    step_progress = maybe_create_tqdm(
        total=total_planned_steps,
        desc="Dataset eval",
        unit="step",
    )
    print(
        "[load] Starting dataset evaluation: "
        f"episodes={len(episode_groups)}, planned_steps={total_planned_steps}"
    )
    textual_progress_step = max(1, total_planned_steps // 20) if total_planned_steps > 0 else 1

    for episode_pos, ((episode_index, rel_indices), planned_episode_steps) in enumerate(
        zip(episode_groups, planned_steps_per_episode),
        start=1,
    ):
        if hasattr(policy, "reset"):
            policy.reset()

        state_history = deque() if use_path_signature else None
        prefix_state_history = [] if build_explicit_prefix_eval_inputs else None
        prefix_signature_history = (
            [] if build_explicit_prefix_eval_inputs and use_path_signature else None
        )
        prefix_delta_signature_history = (
            [] if build_explicit_prefix_eval_inputs and use_delta_signature else None
        )
        prefix_image_histories = (
            {key: [] for key in visual_keys}
            if build_explicit_prefix_eval_inputs
            else None
        )
        previous_signature_vec: np.ndarray | None = None
        first_frame_anchor = None
        episode_abs_error: np.ndarray | None = None
        episode_sq_error: np.ndarray | None = None
        episode_l2_error = 0.0
        episode_cosine = 0.0
        episode_cosine_count = 0

        evaluated_steps = planned_episode_steps

        for rel_idx in rel_indices[:evaluated_steps]:
            item = dataset[rel_idx]
            if state_key not in item:
                raise KeyError(
                    f"Dataset item is missing required state key `{state_key}`."
                )
            if env_state_key is not None and env_state_key not in item:
                raise KeyError(
                    "Dataset item is missing required environment-state key "
                    f"`{env_state_key}`."
                )
            if action_key not in item:
                raise KeyError(
                    f"Dataset item is missing required action key `{action_key}`."
                )
            missing_visual_keys = [key for key in visual_keys if key not in item]
            if missing_visual_keys:
                raise KeyError(
                    "Dataset item is missing required visual observation keys: "
                    f"{missing_visual_keys}."
                )

            obs = {state_key: as_tensor_copy(item[state_key])}
            if env_state_key is not None:
                obs[env_state_key] = as_tensor_copy(item[env_state_key])
            for visual_key in visual_keys:
                obs[visual_key] = as_tensor_copy(item[visual_key])

            if use_first_frame_anchor:
                if FIRST_FRAME_ANCHOR_KEY in item:
                    anchor_tensor = as_tensor_copy(item[FIRST_FRAME_ANCHOR_KEY])
                    if first_frame_anchor is None:
                        first_frame_anchor = anchor_tensor.detach().clone()
                else:
                    if not warned_anchor_fallback:
                        print(
                            "[WARN] Dataset does not contain `observation.anchor_image`; "
                            f"falling back to the first frame from `{visual_keys[0]}`."
                        )
                        warned_anchor_fallback = True
                    if first_frame_anchor is None:
                        first_frame_anchor = as_tensor_copy(item[visual_keys[0]])
                    anchor_tensor = first_frame_anchor.detach().clone()
                obs[FIRST_FRAME_ANCHOR_KEY] = anchor_tensor

            signature_vec: np.ndarray | None = None
            if use_path_signature:
                assert state_history is not None
                state_history.append(tensor_to_numpy_vector(item[state_key]))
                if DEFAULT_PATH_SIGNATURE_KEY in item:
                    obs[DEFAULT_PATH_SIGNATURE_KEY] = as_tensor_copy(
                        item[DEFAULT_PATH_SIGNATURE_KEY]
                    )
                    signature_vec = tensor_to_numpy_vector(item[DEFAULT_PATH_SIGNATURE_KEY])
                else:
                    signature_vec = compute_online_signature_prefix(
                        state_history=state_history,
                        sig_depth=int(cfg.signature_depth),
                        signature_backend=str(signature_backend),
                    )
                    obs[DEFAULT_PATH_SIGNATURE_KEY] = torch.from_numpy(
                        signature_vec.astype(np.float32, copy=False)
                    )
                if signature_vec.shape[0] != int(cfg.signature_dim):
                    raise RuntimeError(
                        "Signature dimension mismatch during offline evaluation: "
                        f"got {signature_vec.shape[0]}, expected {cfg.signature_dim}."
                    )

                if use_delta_signature:
                    if DEFAULT_DELTA_SIGNATURE_KEY in item:
                        obs[DEFAULT_DELTA_SIGNATURE_KEY] = as_tensor_copy(
                            item[DEFAULT_DELTA_SIGNATURE_KEY]
                        )
                    else:
                        delta_signature_vec = compute_delta_signature_step_np(
                            signature_vec,
                            previous_signature_vec,
                        )
                        obs[DEFAULT_DELTA_SIGNATURE_KEY] = torch.from_numpy(
                            delta_signature_vec.astype(np.float32, copy=False)
                        )
                    previous_signature_vec = signature_vec.astype(np.float32, copy=True)

            if build_explicit_prefix_eval_inputs:
                assert prefix_state_history is not None
                assert prefix_image_histories is not None
                build_prefix_sequence_eval_inputs(
                    obs=obs,
                    cfg=cfg,
                    state_key=state_key,
                    image_keys=visual_keys,
                    signature_key=(
                        DEFAULT_PATH_SIGNATURE_KEY if use_path_signature else None
                    ),
                    delta_signature_key=(
                        DEFAULT_DELTA_SIGNATURE_KEY if use_delta_signature else None
                    ),
                    prefix_state_history=prefix_state_history,
                    prefix_signature_history=prefix_signature_history,
                    prefix_delta_signature_history=prefix_delta_signature_history,
                    prefix_image_histories=prefix_image_histories,
                )

            obs = preprocessor(obs)
            if use_path_signature:
                if DEFAULT_PATH_SIGNATURE_KEY not in obs:
                    raise KeyError(
                        f"`{DEFAULT_PATH_SIGNATURE_KEY}` missing after preprocessor."
                    )
                path_signature = obs[DEFAULT_PATH_SIGNATURE_KEY]
                if path_signature.ndim == 1:
                    path_signature = path_signature.unsqueeze(0)
                elif path_signature.ndim != 2:
                    raise RuntimeError(
                        f"`{DEFAULT_PATH_SIGNATURE_KEY}` must be 1D/2D after preprocessing, "
                        f"got shape={tuple(path_signature.shape)}"
                    )
                obs[DEFAULT_PATH_SIGNATURE_KEY] = path_signature.to(
                    device=obs[state_key].device,
                    dtype=obs[state_key].dtype,
                )
            if use_delta_signature:
                if DEFAULT_DELTA_SIGNATURE_KEY not in obs:
                    raise KeyError(
                        f"`{DEFAULT_DELTA_SIGNATURE_KEY}` missing after preprocessor."
                    )
                delta_signature = obs[DEFAULT_DELTA_SIGNATURE_KEY]
                if delta_signature.ndim == 1:
                    delta_signature = delta_signature.unsqueeze(0)
                elif delta_signature.ndim != 2:
                    raise RuntimeError(
                        f"`{DEFAULT_DELTA_SIGNATURE_KEY}` must be 1D/2D after preprocessing, "
                        f"got shape={tuple(delta_signature.shape)}"
                    )
                obs[DEFAULT_DELTA_SIGNATURE_KEY] = delta_signature.to(
                    device=obs[state_key].device,
                    dtype=obs[state_key].dtype,
                )
            if build_explicit_prefix_eval_inputs:
                ensure_prefix_sequence_batch_dims(
                    obs=obs,
                    state_key=state_key,
                    image_keys=visual_keys,
                    use_path_signature=use_path_signature,
                    use_delta_signature=use_delta_signature,
                )
            if use_first_frame_anchor:
                if FIRST_FRAME_ANCHOR_KEY not in obs:
                    raise KeyError(
                        f"`{FIRST_FRAME_ANCHOR_KEY}` missing after preprocessor."
                    )
                anchor_image = obs[FIRST_FRAME_ANCHOR_KEY]
                if anchor_image.ndim == 3:
                    anchor_image = anchor_image.unsqueeze(0)
                elif anchor_image.ndim != 4:
                    raise RuntimeError(
                        f"`{FIRST_FRAME_ANCHOR_KEY}` must be 3D/4D after preprocessing, "
                        f"got shape={tuple(anchor_image.shape)}"
                    )
                obs[FIRST_FRAME_ANCHOR_KEY] = anchor_image.to(
                    device=obs[state_key].device,
                    dtype=obs[state_key].dtype,
                )

            with torch.no_grad():
                predicted_action = policy.select_action(obs)
            predicted_action = postprocessor(predicted_action)
            predicted_np = tensor_to_numpy_vector(predicted_action.squeeze(0))
            ground_truth_np = tensor_to_numpy_vector(item[action_key])

            if action_dim is None:
                action_dim = int(ground_truth_np.shape[0])
                total_abs_error = np.zeros((action_dim,), dtype=np.float64)
                total_sq_error = np.zeros((action_dim,), dtype=np.float64)
            if predicted_np.shape[0] != action_dim or ground_truth_np.shape[0] != action_dim:
                raise RuntimeError(
                    "Action dimension mismatch during offline evaluation. "
                    f"predicted={predicted_np.shape[0]}, ground_truth={ground_truth_np.shape[0]}, "
                    f"expected={action_dim}."
                )

            error = predicted_np - ground_truth_np
            abs_error = np.abs(error).astype(np.float32, copy=False)
            sq_error = np.square(error).astype(np.float32, copy=False)
            l2_error = float(np.linalg.norm(error))

            pred_norm = float(np.linalg.norm(predicted_np))
            gt_norm = float(np.linalg.norm(ground_truth_np))
            cosine = None
            if pred_norm > 1e-8 and gt_norm > 1e-8:
                cosine = float(
                    np.dot(predicted_np, ground_truth_np) / (pred_norm * gt_norm)
                )

            total_abs_error += abs_error
            total_sq_error += sq_error
            total_l2_error += l2_error
            total_steps += 1
            if step_progress is not None:
                step_progress.set_postfix_str(
                    f"episode={episode_pos}/{len(episode_groups)} id={episode_index}"
                )
                step_progress.update(1)
            elif (
                total_steps == 1
                or total_steps == total_planned_steps
                or (total_steps % textual_progress_step) == 0
            ):
                print(
                    "[Dataset eval] "
                    f"steps={total_steps}/{total_planned_steps} "
                    f"episode={episode_pos}/{len(episode_groups)} id={episode_index}"
                )
            if cosine is not None:
                total_cosine += cosine
                total_cosine_count += 1

            if episode_abs_error is None:
                episode_abs_error = np.zeros((action_dim,), dtype=np.float64)
                episode_sq_error = np.zeros((action_dim,), dtype=np.float64)
            episode_abs_error += abs_error
            episode_sq_error += sq_error
            episode_l2_error += l2_error
            if cosine is not None:
                episode_cosine += cosine
                episode_cosine_count += 1

        assert action_dim is not None
        assert episode_abs_error is not None
        assert episode_sq_error is not None
        result = {
            "episode_index": int(episode_index),
            "steps": int(evaluated_steps),
            "mae": float(episode_abs_error.sum() / (evaluated_steps * action_dim)),
            "rmse": float(
                np.sqrt(episode_sq_error.sum() / (evaluated_steps * action_dim))
            ),
            "mean_l2_error": float(episode_l2_error / evaluated_steps),
            "cosine_similarity": (
                None
                if episode_cosine_count == 0
                else float(episode_cosine / episode_cosine_count)
            ),
            "per_dim_mae": (episode_abs_error / evaluated_steps).tolist(),
            "per_dim_rmse": np.sqrt(episode_sq_error / evaluated_steps).tolist(),
        }
        results.append(result)
        cosine_text = (
            "n/a"
            if result["cosine_similarity"] is None
            else f"{result['cosine_similarity']:.4f}"
        )
        progress_write(
            step_progress,
            f"[{episode_pos:03d}/{len(episode_groups):03d}] "
            f"episode={episode_index} steps={evaluated_steps} "
            f"mae={result['mae']:.6f} rmse={result['rmse']:.6f} "
            f"l2={result['mean_l2_error']:.6f} cosine={cosine_text}"
        )

    if step_progress is not None:
        step_progress.close()

    assert action_dim is not None
    assert total_abs_error is not None
    assert total_sq_error is not None

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "policy_type": args.policy,
        "policy_dir": str(policy_dir),
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "eval_split": args.eval_split,
        "split_source": split_source,
        "num_episodes": len(results),
        "num_steps": int(total_steps),
        "action_dim": int(action_dim),
        "metrics": {
            "mae": float(total_abs_error.sum() / (total_steps * action_dim)),
            "rmse": float(np.sqrt(total_sq_error.sum() / (total_steps * action_dim))),
            "mean_l2_error": float(total_l2_error / total_steps),
            "cosine_similarity": (
                None
                if total_cosine_count == 0
                else float(total_cosine / total_cosine_count)
            ),
            "per_dim_mae": (total_abs_error / total_steps).tolist(),
            "per_dim_rmse": np.sqrt(total_sq_error / total_steps).tolist(),
        },
        "results": results,
    }
    summary_path = write_summary(output_dir, summary)

    print(f"\nSummary: {summary_path}")
    print(
        f"MAE: {summary['metrics']['mae']:.6f}, "
        f"RMSE: {summary['metrics']['rmse']:.6f}, "
        f"mean_L2: {summary['metrics']['mean_l2_error']:.6f}"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import torch

    repo_root = Path(__file__).resolve().parents[2]
    if args.policy == "streaming_act":
        ensure_streaming_act_importable(repo_root)

    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing LeRobot evaluation dependencies. Install the pip package first, "
            "for example `pip install lerobot`, and ensure torch is installed for "
            "your platform."
        ) from exc

    if args.policy == "streaming_act":
        from lerobot_policy_streaming_act.configuration_streaming_act import (
            StreamingACTConfig,
        )
        from lerobot_policy_streaming_act.modeling_streaming_act import (
            StreamingACTPolicy,
        )

        policy_cls = StreamingACTPolicy
    else:
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        policy_cls = ACTPolicy

    policy_dir = resolve_eval_policy_path(
        policy_path=args.policy_path,
        latest_run_dir=args.latest_run_dir,
        train_output_root=args.train_output_root,
    )
    print(f"Using policy path: {policy_dir}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    local_files_only = policy_dir.is_dir()
    config_load_start_s = time.perf_counter()
    print(f"[load] Loading policy config for device={args.device}...")
    cfg = PreTrainedConfig.from_pretrained(
        policy_dir,
        local_files_only=local_files_only,
    )
    checkpoint_device = getattr(cfg, "device", None)
    if checkpoint_device != args.device:
        print(
            "[load] Overriding checkpoint device: "
            f"{checkpoint_device!r} -> {args.device!r}"
        )
    cfg.device = args.device
    print(
        "[timing] Policy config loaded in "
        f"{format_elapsed_s(time.perf_counter() - config_load_start_s)}"
    )

    policy_load_start_s = time.perf_counter()
    print("[load] Loading policy weights...")
    policy = policy_cls.from_pretrained(
        policy_dir,
        config=cfg,
        local_files_only=local_files_only,
    )
    print(
        "[timing] Policy weights loaded in "
        f"{format_elapsed_s(time.perf_counter() - policy_load_start_s)}"
    )
    cfg = policy.config
    cfg.device = args.device
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
    if (
        args.policy == "streaming_act"
        and getattr(cfg, "use_visual_prefix_memory", False)
        and cfg.n_action_steps > 1
    ):
        print(
            "[warn] `streaming_act` is evaluating with `n_action_steps>1`. "
            "Online visual prefix memory is updated every observation step, but new "
            "memory state does not affect actions already queued for execution. "
            "For the most responsive streaming evaluation, prefer `--n-action-steps 1`."
        )
    policy.eval()
    if args.n_action_steps is not None and hasattr(policy, "reset"):
        policy.reset()

    preprocessor_overrides = {
        "device_processor": {"device": args.device},
        "rename_observations_processor": {"rename_map": {}},
    }
    processor_load_start_s = time.perf_counter()
    print("[load] Initializing pre/post processors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_dir,
        preprocessor_overrides=preprocessor_overrides,
    )
    print(
        "[timing] Pre/post processors initialized in "
        f"{format_elapsed_s(time.perf_counter() - processor_load_start_s)}"
    )

    if args.env is not None:
        run_env_evaluation(
            args=args,
            policy=policy,
            cfg=cfg,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_dir=policy_dir,
        )
        return

    run_dataset_evaluation(
        args=args,
        policy=policy,
        cfg=cfg,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        policy_dir=policy_dir,
    )


if __name__ == "__main__":
    main()
