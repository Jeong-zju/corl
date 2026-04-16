from __future__ import annotations

from collections import deque
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dataset_utils import (
    DEFAULT_LOCAL_DATA_ROOT,
    LEGACY_EPISODES_JSONL_PATH,
    find_dataset_split_file,
    is_supported_lerobot_dataset_root,
    load_dataset_split,
    resolve_dataset_root,
)
from eval_helpers import (
    build_prefix_sequence_eval_inputs,
    compute_delta_signature_step_np,
    compute_signatory_signature_np,
    compute_simple_signature_np,
    ensure_prefix_sequence_batch_dims,
    resolve_signature_backend,
    write_summary,
)
from policy_capabilities import (
    get_visual_memory_debug_stats,
    resolve_policy_capability_flags,
)


MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from benchmarks.robocasa import (  # noqa: E402
    ROBOCASA_TASK_DESCRIPTION_KEY,
    RoboCasaEvaluationResult,
    RoboCasaRolloutResult,
    _close_video_writer,
    _normalize_video_frame,
    _write_video_frame,
    create_robocasa_benchmark_env,
    flatten_robocasa_state_observation,
    list_available_robocasa_tasks,
    start_ffmpeg_raw_writer,
)


ENV_NAME = "robocasa"
DEFAULT_ROBOCASA_CONDA_ENV = "robocasa"
DEFAULT_CAMERA_HEIGHT = 256
DEFAULT_CAMERA_WIDTH = 256
TASK_NAME_FEATURE_KEY = "annotation.human.task_name"
TASK_DESCRIPTION_FEATURE_KEY = "annotation.human.task_description"
ROBOCASA_MAX_STEPS_MARGIN_RATIO = 0.10
ROBOCASA_MAX_STEPS_MARGIN_MIN = 10
SUPPORTED_TASK_FEATURE_KEYS = {
    TASK_NAME_FEATURE_KEY,
    TASK_DESCRIPTION_FEATURE_KEY,
}
DEFAULT_PATH_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_DELTA_SIGNATURE_KEY = "observation.delta_signature"
DEFAULT_PREFIX_STATE_KEY = "observation.prefix_state"
DEFAULT_PREFIX_MASK_KEY = "observation.prefix_mask"
DEFAULT_PREFIX_PATH_SIGNATURE_KEY = "observation.prefix_path_signature"
DEFAULT_PREFIX_DELTA_SIGNATURE_KEY = "observation.prefix_delta_signature"
DEFAULT_PREFIX_IMAGES_PREFIX = "observation.prefix_images."


def _maybe_create_tqdm(
    *,
    total: int,
    desc: str,
    unit: str,
    leave: bool = True,
):
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit, leave=leave)


def _progress_write(progress, message: str) -> None:
    if progress is not None and hasattr(progress, "write"):
        progress.write(message)
        return
    print(message)


def _format_elapsed_s(elapsed_s: float) -> str:
    return f"{elapsed_s:.1f}s"


def _normalize_output_path_part(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "item"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(output_path: Path, payload: Mapping[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return output_path


def _feature_shape(feature: Any) -> tuple[int, ...]:
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, Mapping):
        shape = feature.get("shape")
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


def _resolve_visual_features(cfg) -> list[tuple[str, str, tuple[int, ...]]]:
    visual_features = getattr(cfg, "visual_observation_features", None)
    if visual_features is None:
        visual_features = getattr(cfg, "image_features", {})
    if not isinstance(visual_features, Mapping):
        return []

    specs: list[tuple[str, str, tuple[int, ...]]] = []
    for policy_key, feature in visual_features.items():
        policy_key_s = str(policy_key)
        if not policy_key_s.startswith("observation.images."):
            raise NotImplementedError(
                "RoboCasa ACT eval currently supports image inputs only under "
                "`observation.images.*`. "
                f"Got unsupported visual feature {policy_key_s!r}."
            )
        camera_name = policy_key_s[len("observation.images.") :]
        specs.append((policy_key_s, f"video.{camera_name}", _feature_shape(feature)))
    return specs


def _resolve_state_feature_key(cfg) -> tuple[str | None, tuple[int, ...]]:
    input_features = getattr(cfg, "input_features", None)
    if not isinstance(input_features, Mapping):
        return None, ()
    if "observation.state" in input_features:
        feature = input_features["observation.state"]
        return "observation.state", _feature_shape(feature)
    for key, feature in input_features.items():
        key_s = str(key)
        if key_s.endswith(".state") and not key_s.startswith("observation.images."):
            return key_s, _feature_shape(feature)
    return None, ()


def _resolve_required_task_feature_keys(cfg) -> list[str]:
    input_features = getattr(cfg, "input_features", None)
    if not isinstance(input_features, Mapping):
        return []
    return [
        str(key)
        for key in input_features
        if str(key) in SUPPORTED_TASK_FEATURE_KEYS
    ]


def _is_supported_aux_input_feature_key(key: str) -> bool:
    # Some saved checkpoint schemas keep signature/prefix placeholders in
    # `input_features` even when the runtime policy ignores missing values.
    return key in {
        DEFAULT_PATH_SIGNATURE_KEY,
        DEFAULT_DELTA_SIGNATURE_KEY,
        DEFAULT_PREFIX_STATE_KEY,
        DEFAULT_PREFIX_MASK_KEY,
        DEFAULT_PREFIX_PATH_SIGNATURE_KEY,
        DEFAULT_PREFIX_DELTA_SIGNATURE_KEY,
    } or key.startswith(DEFAULT_PREFIX_IMAGES_PREFIX)


def _validate_supported_input_features(cfg) -> None:
    input_features = getattr(cfg, "input_features", None)
    if not isinstance(input_features, Mapping):
        raise RuntimeError("RoboCasa ACT eval expected `cfg.input_features` to be present.")

    supported: set[str] = set()
    supported.update(key for key, _, _ in _resolve_visual_features(cfg))
    state_key, _state_shape = _resolve_state_feature_key(cfg)
    if state_key is not None:
        supported.add(state_key)
    supported.update(_resolve_required_task_feature_keys(cfg))
    supported.update(
        str(key)
        for key in input_features
        if _is_supported_aux_input_feature_key(str(key))
    )

    unsupported = [str(key) for key in input_features if str(key) not in supported]
    if unsupported:
        raise NotImplementedError(
            "RoboCasa eval currently supports image, state, task-id, signature, "
            "and prefix-memory inputs only. "
            f"Unsupported input features: {unsupported}"
        )


def _resolve_camera_setup(
    visual_specs: list[tuple[str, str, tuple[int, ...]]],
) -> tuple[list[str] | None, int, int]:
    if not visual_specs:
        return None, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT

    camera_names: list[str] = []
    heights: set[int] = set()
    widths: set[int] = set()
    for _policy_key, env_key, shape in visual_specs:
        camera_name = env_key.split(".", 1)[1]
        if camera_name not in camera_names:
            camera_names.append(camera_name)
        if len(shape) == 3:
            heights.add(int(shape[-2]))
            widths.add(int(shape[-1]))

    if len(heights) > 1 or len(widths) > 1:
        raise RuntimeError(
            "RoboCasa ACT eval expects all image features to share one resolution. "
            f"Got heights={sorted(heights)}, widths={sorted(widths)}."
        )

    camera_height = next(iter(heights), DEFAULT_CAMERA_HEIGHT)
    camera_width = next(iter(widths), DEFAULT_CAMERA_WIDTH)
    return camera_names, camera_width, camera_height


def _choose_video_image_key(
    visual_specs: list[tuple[str, str, tuple[int, ...]]],
) -> str | None:
    env_keys = [env_key for _policy_key, env_key, _shape in visual_specs]
    for preferred in (
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ):
        if preferred in env_keys:
            return preferred
    return env_keys[0] if env_keys else None


def _resolve_video_image_keys(
    visual_specs: list[tuple[str, str, tuple[int, ...]]],
) -> list[str]:
    keys: list[str] = []
    for _policy_key, env_key, _shape in visual_specs:
        if env_key not in keys:
            keys.append(env_key)
    return keys


def _build_video_frame_from_observation(
    observation: Mapping[str, Any],
    *,
    video_image_keys: list[str],
) -> np.ndarray:
    if not video_image_keys:
        raise ValueError("`video_image_keys` must not be empty.")

    frames: list[np.ndarray] = []
    max_height = 0
    for image_key in video_image_keys:
        if image_key not in observation:
            raise KeyError(
                f"RoboCasa observation is missing requested video image key {image_key!r}."
            )
        frame = _normalize_video_frame(observation[image_key])
        frames.append(frame)
        max_height = max(max_height, int(frame.shape[0]))

    padded_frames: list[np.ndarray] = []
    for frame in frames:
        if int(frame.shape[0]) == max_height:
            padded_frames.append(frame)
            continue
        pad_height = max_height - int(frame.shape[0])
        padded_frames.append(
            np.pad(
                frame,
                ((0, pad_height), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        )
    return np.ascontiguousarray(np.concatenate(padded_frames, axis=1))


def _normalize_conda_env_name(raw_value: str | None) -> str | None:
    if raw_value is None:
        return DEFAULT_ROBOCASA_CONDA_ENV
    value = str(raw_value).strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def _normalize_robocasa_split(raw_value: str | None) -> str:
    if raw_value is None:
        return "target"
    value = str(raw_value).strip().lower()
    if value in {"", "none", "null"}:
        return "target"
    if value not in {"target", "pretrain", "all"}:
        raise ValueError(
            "`--robocasa-split` must be one of {'target', 'pretrain', 'all'}, "
            f"got {raw_value!r}."
        )
    return value


def _compute_online_signature(
    state_history: deque[np.ndarray],
    history_length: int,
    sig_depth: int,
    signature_backend: str,
) -> np.ndarray:
    if history_length <= 0:
        raise ValueError(f"history_length must be > 0, got {history_length}")
    if len(state_history) == 0:
        raise ValueError("state_history is empty; cannot compute path signature.")

    window = np.stack(list(state_history), axis=0).astype(np.float32, copy=False)
    if window.shape[0] < history_length:
        pad_len = history_length - window.shape[0]
        pad = np.repeat(window[:1], pad_len, axis=0)
        window = np.concatenate([pad, window], axis=0)

    if signature_backend == "signatory":
        return compute_signatory_signature_np(window, sig_depth)
    return compute_simple_signature_np(window, sig_depth)


def _parse_tasks(task_arg: str | None) -> list[str]:
    raw = "" if task_arg is None else str(task_arg)
    tasks: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        task_name = part.strip()
        if not task_name or task_name in seen:
            continue
        seen.add(task_name)
        tasks.append(task_name)
    if not tasks:
        raise ValueError(
            "RoboCasa eval requires `--task` or `--tasks` with one task name or a "
            "comma-separated task list."
        )
    return tasks


def _resolve_dataset_root_for_task_indices(args, policy_dir: Path) -> Path | None:
    local_data_root = Path(
        getattr(args, "local_data_root", DEFAULT_LOCAL_DATA_ROOT)
    ).resolve()
    dataset_arg = getattr(args, "dataset", None)
    defaults_dataset_root = getattr(args, "_policy_defaults_dataset_root", None)

    if dataset_arg is not None:
        try:
            return resolve_dataset_root(dataset_arg, local_data_root=local_data_root)
        except FileNotFoundError:
            if defaults_dataset_root:
                return resolve_dataset_root(
                    defaults_dataset_root,
                    local_data_root=local_data_root,
                )
            raise

    split_path = find_dataset_split_file(policy_dir)
    if split_path is not None:
        split_spec = load_dataset_split(split_path)
        dataset_root = Path(split_spec.dataset_root)
        if dataset_root.exists():
            return dataset_root.resolve()

    if defaults_dataset_root:
        return resolve_dataset_root(
            defaults_dataset_root,
            local_data_root=local_data_root,
        )
    return None


def _iter_exact_dataset_root_candidates(
    selector: str | Path | None,
    *,
    local_data_root: Path,
) -> list[Path]:
    if selector is None:
        return []

    dataset_text = str(selector).strip()
    if not dataset_text:
        return []

    normalized_dataset_text = dataset_text.replace("\\", "/")
    raw_path = Path(dataset_text).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute() or raw_path.exists() or dataset_text.startswith("."):
        candidates.append(raw_path)

    if not raw_path.is_absolute():
        if normalized_dataset_text.startswith("main/data/"):
            candidates.append(
                local_data_root / normalized_dataset_text[len("main/data/") :]
            )
        elif normalized_dataset_text.startswith("data/"):
            candidates.append(
                local_data_root / normalized_dataset_text[len("data/") :]
            )
        else:
            candidates.append(local_data_root / dataset_text)

    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(candidate)
    return ordered


def _task_dataset_matches(dataset_root: Path, task_name: str) -> bool:
    if dataset_root.name == str(task_name).strip():
        return True

    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return False

    for raw_line in tasks_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if str(payload.get("task", "")).strip() == str(task_name).strip():
            return True
    return False


def _resolve_task_dataset_root(
    *,
    task_name: str,
    args,
    policy_dir: Path,
) -> Path | None:
    local_data_root = Path(
        getattr(args, "local_data_root", DEFAULT_LOCAL_DATA_ROOT)
    ).resolve()

    selectors: list[str | Path] = []
    dataset_arg = getattr(args, "dataset", None)
    if dataset_arg is not None:
        selectors.append(dataset_arg)

    defaults_dataset_root = getattr(args, "_policy_defaults_dataset_root", None)
    if defaults_dataset_root:
        selectors.append(defaults_dataset_root)

    split_path = find_dataset_split_file(policy_dir)
    if split_path is not None:
        selectors.append(load_dataset_split(split_path).dataset_root)

    seen: set[Path] = set()
    for selector in selectors:
        for base_candidate in _iter_exact_dataset_root_candidates(
            selector,
            local_data_root=local_data_root,
        ):
            for candidate in (
                base_candidate,
                base_candidate / task_name,
            ):
                resolved = candidate.resolve(strict=False)
                if resolved in seen:
                    continue
                seen.add(resolved)
                if is_supported_lerobot_dataset_root(candidate) and _task_dataset_matches(
                    candidate,
                    task_name,
                ):
                    return candidate.resolve()

            if is_supported_lerobot_dataset_root(base_candidate):
                sibling_candidate = base_candidate.parent / task_name
                resolved_sibling = sibling_candidate.resolve(strict=False)
                if (
                    resolved_sibling not in seen
                    and is_supported_lerobot_dataset_root(sibling_candidate)
                    and _task_dataset_matches(sibling_candidate, task_name)
                ):
                    return sibling_candidate.resolve()
                seen.add(resolved_sibling)

    for candidate in sorted(local_data_root.glob(f"**/{task_name}")):
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if is_supported_lerobot_dataset_root(candidate) and _task_dataset_matches(
            candidate,
            task_name,
        ):
            return candidate.resolve()
    return None


def _estimate_conservative_max_steps_from_dataset(
    dataset_root: Path,
) -> tuple[int, int, int]:
    lengths: list[int] = []

    episodes_meta_dir = dataset_root / "meta" / "episodes"
    episode_parquet_paths = sorted(episodes_meta_dir.rglob("*.parquet"))
    if episode_parquet_paths:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError:
            pq = None
        if pq is not None:
            for parquet_path in episode_parquet_paths:
                episode_table = pq.read_table(parquet_path, columns=["length"])
                parquet_lengths = np.asarray(
                    episode_table["length"].to_pylist(),
                    dtype=np.int64,
                )
                lengths.extend(int(length) for length in parquet_lengths if int(length) > 0)

    if not lengths:
        episodes_path = dataset_root / LEGACY_EPISODES_JSONL_PATH
        if episodes_path.exists():
            for raw_line in episodes_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                length = int(payload.get("length", 0))
                if length > 0:
                    lengths.append(length)

    if not lengths:
        extras_root = dataset_root / "extras"
        for states_path in sorted(extras_root.glob("episode_*/states.npz")):
            with np.load(states_path) as payload:
                states = payload.get("states")
                if states is None or states.ndim == 0:
                    continue
                length = int(states.shape[0])
                if length > 0:
                    lengths.append(length)

    if not lengths:
        info_path = dataset_root / "meta" / "info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            total_frames = int(info.get("total_frames", 0))
            total_episodes = int(info.get("total_episodes", 0))
            if (
                total_frames > 0
                and total_episodes > 0
                and total_frames % total_episodes == 0
            ):
                inferred_length = total_frames // total_episodes
                if inferred_length > 0:
                    lengths.append(inferred_length)

    if not lengths:
        raise FileNotFoundError(
            "RoboCasa eval could not infer episode lengths. Checked "
            f"{episodes_meta_dir}, {dataset_root / LEGACY_EPISODES_JSONL_PATH}, "
            f"{dataset_root / 'extras'}, and {dataset_root / 'meta' / 'info.json'}."
        )

    longest_length = int(max(lengths))
    conservative_margin = max(
        int(ROBOCASA_MAX_STEPS_MARGIN_MIN),
        int(math.ceil(longest_length * ROBOCASA_MAX_STEPS_MARGIN_RATIO)),
    )
    conservative_max_steps = int(longest_length + conservative_margin)
    return conservative_max_steps, longest_length, conservative_margin


def _resolve_task_max_steps(
    *,
    requested_max_steps: int | None,
    task_name: str,
    task_dataset_root: Path | None,
) -> tuple[int, dict[str, Any]]:
    if requested_max_steps is not None:
        resolved = int(requested_max_steps)
        if resolved <= 0:
            raise ValueError("`--max-steps` must be positive when provided.")
        return resolved, {
            "source": "cli",
            "requested_max_steps": resolved,
            "dataset_root": None if task_dataset_root is None else str(task_dataset_root),
            "dataset_longest_episode_length": None,
            "conservative_margin": None,
        }

    if task_dataset_root is None:
        raise ValueError(
            "RoboCasa eval could not infer `max_steps` for task "
            f"{task_name!r} because no matching dataset root was resolved. "
            "Pass `--dataset robocasa/.../<task>` (or a base RoboCasa dataset path) "
            "or set `--max-steps` explicitly."
        )

    resolved, longest_length, conservative_margin = (
        _estimate_conservative_max_steps_from_dataset(task_dataset_root)
    )
    return resolved, {
        "source": "dataset_longest_plus_margin",
        "requested_max_steps": None,
        "dataset_root": str(task_dataset_root),
        "dataset_longest_episode_length": int(longest_length),
        "conservative_margin": int(conservative_margin),
    }


class _TaskIndexResolver:
    def __init__(self, dataset_root: Path | None) -> None:
        self.dataset_root = dataset_root
        self._task_to_index: dict[str, int] = {}
        if dataset_root is None:
            return
        tasks_path = dataset_root / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            return
        for raw_line in tasks_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            task_text = str(payload.get("task", "")).strip()
            if not task_text:
                continue
            self._task_to_index[task_text] = int(payload["task_index"])

    @property
    def available(self) -> bool:
        return bool(self._task_to_index)

    def resolve(self, task_text: str) -> int | None:
        return self._task_to_index.get(str(task_text).strip())


class RoboCasaACTPolicyAdapter:
    def __init__(
        self,
        *,
        policy,
        cfg,
        preprocessor,
        postprocessor,
        policy_dir: Path,
        task_index_resolver: _TaskIndexResolver,
        signature_backend: str | None = None,
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.policy_dir = policy_dir
        self.task_index_resolver = task_index_resolver
        self.visual_specs = _resolve_visual_features(cfg)
        self.policy_image_keys = [policy_key for policy_key, _env_key, _shape in self.visual_specs]
        self.state_feature_key, self.state_shape = _resolve_state_feature_key(cfg)
        self.required_task_feature_keys = _resolve_required_task_feature_keys(cfg)
        capability_flags = resolve_policy_capability_flags(cfg)
        self.use_path_signature = capability_flags.use_path_signature
        self.use_prefix_sequence_training = (
            capability_flags.use_prefix_sequence_training
        )
        self.use_visual_prefix_memory = capability_flags.use_visual_prefix_memory
        self.use_delta_signature = capability_flags.use_delta_signature
        self.build_explicit_prefix_eval_inputs = (
            capability_flags.build_explicit_prefix_eval_inputs
        )
        self.signature_backend = signature_backend
        _validate_supported_input_features(cfg)
        if self.state_feature_key is None:
            raise RuntimeError(
                "RoboCasa ACT eval requires a state input feature such as "
                "`observation.state`."
            )
        if self.required_task_feature_keys and not self.task_index_resolver.available:
            raise RuntimeError(
                "This RoboCasa ACT checkpoint expects task-id conditioning "
                f"features {self.required_task_feature_keys}, but no `meta/tasks.jsonl` "
                "mapping was found. Pass `--dataset robocasa/...` or evaluate from a "
                "training run that still has `dataset_split.json`."
            )
        if self.use_path_signature and self.signature_backend is None:
            raise RuntimeError(
                "RoboCasa eval requires a resolved signature backend "
                "when `use_path_signature=True`."
            )
        self.reset()

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if self.use_path_signature:
            self.state_history = deque(maxlen=int(self.cfg.history_length))
            self.previous_signature_vec: np.ndarray | None = None
        else:
            self.state_history = None
            self.previous_signature_vec = None
        if self.build_explicit_prefix_eval_inputs:
            self.prefix_state_history: list[Any] = []
            self.prefix_image_histories = {
                image_key: [] for image_key in self.policy_image_keys
            }
            self.prefix_signature_history = [] if self.use_path_signature else None
            self.prefix_delta_signature_history = [] if self.use_delta_signature else None
        else:
            self.prefix_state_history = None
            self.prefix_image_histories = None
            self.prefix_signature_history = None
            self.prefix_delta_signature_history = None

    def _build_task_feature(
        self,
        *,
        feature_key: str,
        observation: Mapping[str, Any],
        env,
    ) -> np.ndarray:
        if feature_key == TASK_NAME_FEATURE_KEY:
            task_index = self.task_index_resolver.resolve(str(env.task))
        elif feature_key == TASK_DESCRIPTION_FEATURE_KEY:
            task_description = str(observation.get(ROBOCASA_TASK_DESCRIPTION_KEY, "")).strip()
            task_index = self.task_index_resolver.resolve(task_description)
        else:
            raise KeyError(f"Unsupported RoboCasa task feature {feature_key!r}.")

        if task_index is None:
            raise KeyError(
                "Could not resolve RoboCasa task index for feature "
                f"{feature_key!r}. task={env.task!r}."
            )
        return np.asarray([task_index], dtype=np.int64)

    def _build_policy_observation(
        self,
        observation: Mapping[str, Any],
        *,
        env,
    ) -> dict[str, Any]:
        import torch

        obs: dict[str, Any] = {}
        for policy_key, env_key, _shape in self.visual_specs:
            if env_key not in observation:
                raise KeyError(
                    f"RoboCasa observation is missing required image key {env_key!r}."
                )
            image = np.asarray(observation[env_key])
            if image.ndim != 3:
                raise RuntimeError(
                    f"Expected RoboCasa image observation {env_key!r} to be HWC, "
                    f"got shape={tuple(image.shape)}."
                )
            obs[policy_key] = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .contiguous()
                .float()
                / 255.0
            )

        state = flatten_robocasa_state_observation(
            observation,
            state_keys=env.state_keys,
        )
        if len(self.state_shape) == 1 and int(self.state_shape[0]) != int(state.shape[0]):
            raise RuntimeError(
                "RoboCasa flattened state dimension mismatch: "
                f"got {state.shape[0]}, expected {self.state_shape[0]}."
            )
        obs[self.state_feature_key] = torch.from_numpy(
            state.astype(np.float32, copy=False)
        )

        for feature_key in self.required_task_feature_keys:
            obs[feature_key] = torch.from_numpy(
                self._build_task_feature(
                    feature_key=feature_key,
                    observation=observation,
                    env=env,
                )
            )
        return obs

    def act(self, observation: Mapping[str, Any], *, env) -> np.ndarray:
        import torch

        obs = self._build_policy_observation(observation, env=env)
        if self.use_path_signature:
            assert self.state_history is not None
            state_now = (
                obs[self.state_feature_key]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            self.state_history.append(state_now.copy())
            signature_vec = _compute_online_signature(
                state_history=self.state_history,
                history_length=int(self.cfg.history_length),
                sig_depth=int(self.cfg.signature_depth),
                signature_backend=str(self.signature_backend),
            )
            if signature_vec.shape[0] != int(self.cfg.signature_dim):
                raise RuntimeError(
                    "Online signature dimension mismatch: "
                    f"got {signature_vec.shape[0]}, expected cfg.signature_dim={self.cfg.signature_dim}."
                )
            obs[DEFAULT_PATH_SIGNATURE_KEY] = torch.from_numpy(
                signature_vec.astype(np.float32, copy=False)
            )
            if self.use_delta_signature:
                delta_signature_vec = compute_delta_signature_step_np(
                    signature_vec,
                    self.previous_signature_vec,
                )
                obs[DEFAULT_DELTA_SIGNATURE_KEY] = torch.from_numpy(
                    delta_signature_vec.astype(np.float32, copy=False)
                )
                self.previous_signature_vec = signature_vec.astype(np.float32, copy=True)

        if self.build_explicit_prefix_eval_inputs:
            assert self.prefix_state_history is not None
            assert self.prefix_image_histories is not None
            build_prefix_sequence_eval_inputs(
                obs=obs,
                cfg=self.cfg,
                state_key=self.state_feature_key,
                image_keys=self.policy_image_keys,
                signature_key=(
                    DEFAULT_PATH_SIGNATURE_KEY if self.use_path_signature else None
                ),
                delta_signature_key=(
                    DEFAULT_DELTA_SIGNATURE_KEY if self.use_delta_signature else None
                ),
                prefix_state_history=self.prefix_state_history,
                prefix_signature_history=self.prefix_signature_history,
                prefix_delta_signature_history=self.prefix_delta_signature_history,
                prefix_image_histories=self.prefix_image_histories,
            )

        obs = self.preprocessor(obs)
        if self.use_path_signature:
            if DEFAULT_PATH_SIGNATURE_KEY not in obs:
                raise KeyError(
                    f"`{DEFAULT_PATH_SIGNATURE_KEY}` missing after preprocessing; "
                    "cannot run policy with use_path_signature=True."
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
                device=obs[self.state_feature_key].device,
                dtype=obs[self.state_feature_key].dtype,
            )
        if self.use_delta_signature:
            if DEFAULT_DELTA_SIGNATURE_KEY not in obs:
                raise KeyError(
                    f"`{DEFAULT_DELTA_SIGNATURE_KEY}` missing after preprocessing; "
                    "cannot run policy with use_delta_signature=True."
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
                device=obs[self.state_feature_key].device,
                dtype=obs[self.state_feature_key].dtype,
            )
        if self.build_explicit_prefix_eval_inputs:
            ensure_prefix_sequence_batch_dims(
                obs=obs,
                state_key=self.state_feature_key,
                image_keys=self.policy_image_keys,
                use_path_signature=self.use_path_signature,
                use_delta_signature=self.use_delta_signature,
            )
        with torch.no_grad():
            action = self.policy.select_action(obs)
        action = self.postprocessor(action)
        if isinstance(action, Mapping):
            if "action" in action:
                action = action["action"]
            elif len(action) == 1:
                action = next(iter(action.values()))
            else:
                raise RuntimeError(
                    "RoboCasa ACT eval expected a single action tensor after "
                    f"postprocessing, got keys={list(action)}."
                )
        if torch.is_tensor(action):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)
        action_np = np.asarray(action_np, dtype=np.float32).reshape(-1)
        if action_np.shape[0] != int(env.action_dim):
            raise RuntimeError(
                "RoboCasa ACT action dimension mismatch: "
                f"got {action_np.shape[0]}, expected {env.action_dim}."
            )
        return action_np


def _run_single_rollout(
    *,
    env,
    policy_adapter: RoboCasaACTPolicyAdapter,
    max_steps: int,
    seed: int | None,
    video_path: Path | None,
    details_path: Path,
    fps: int,
    video_image_key: str | None,
    video_image_keys: list[str] | None,
    step_progress=None,
) -> RoboCasaRolloutResult:
    if max_steps <= 0:
        raise ValueError("`--max-steps` must be positive for RoboCasa eval.")

    policy_adapter.reset()
    writer = None

    def write_frame(observation: Mapping[str, Any] | None) -> None:
        nonlocal writer
        if video_path is None:
            return
        if observation is not None and video_image_keys:
            frame = _build_video_frame_from_observation(
                observation,
                video_image_keys=video_image_keys,
            )
        elif observation is not None and video_image_key is not None:
            if video_image_key not in observation:
                raise KeyError(
                    f"RoboCasa observation is missing requested video image key {video_image_key!r}."
                )
            frame = np.asarray(observation[video_image_key], dtype=np.uint8).copy()
        else:
            frame = env.render_frame(image_key=video_image_key)
        frame = _normalize_video_frame(frame)
        if writer is None:
            writer = start_ffmpeg_raw_writer(
                video_path,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                fps=int(fps),
            )
            if writer.stdin is None:
                raise RuntimeError("Failed to open ffmpeg stdin for RoboCasa eval video writing.")
        _write_video_frame(writer, frame)

    try:
        observation, initial_info = env.reset(seed=seed, record_observation=True)
        write_frame(observation)

        total_reward = 0.0
        num_steps = 0
        terminated = False
        truncated = False
        success = bool(initial_info.get("success", False))
        success_details = dict(initial_info.get("success_details", {"task": success}))
        final_info = dict(initial_info)
        done_reason = "reset_success" if success else "horizon"

        if not success:
            for _step_idx in range(int(max_steps)):
                action = policy_adapter.act(observation, env=env)
                observation, reward, terminated, truncated, step_info = env.step(
                    action,
                    record_observation=True,
                )
                write_frame(observation)
                num_steps += 1
                if step_progress is not None:
                    step_progress.update(1)
                total_reward += float(reward)
                final_info = dict(step_info)
                success = bool(step_info.get("success", False))
                success_details = dict(
                    step_info.get("success_details", {"task": success})
                )
                if step_progress is not None:
                    step_progress.set_postfix(
                        reward=f"{total_reward:.2f}",
                        success=int(success),
                        refresh=False,
                    )

                if success:
                    done_reason = "success"
                    break
                if terminated:
                    done_reason = "terminated"
                    break
                if truncated:
                    done_reason = "truncated"
                    break

        result = RoboCasaRolloutResult(
            task=str(env.task),
            seed=seed,
            max_steps=int(max_steps),
            num_steps=int(num_steps),
            total_reward=float(total_reward),
            success=bool(success),
            success_details={
                str(key): bool(value) for key, value in success_details.items()
            },
            terminated=bool(terminated),
            truncated=bool(truncated),
            done_reason=str(done_reason),
            initial_info=dict(initial_info),
            final_info=dict(final_info),
            initial_observation=None,
            final_observation=None,
            trajectory=None,
            video_path=None if video_path is None else str(video_path),
            details_path=str(details_path),
        )
        _write_json(
            details_path,
            {
                "task": result.task,
                "seed": result.seed,
                "max_steps": result.max_steps,
                "num_steps": result.num_steps,
                "total_reward": result.total_reward,
                "success": result.success,
                "success_details": result.success_details,
                "terminated": result.terminated,
                "truncated": result.truncated,
                "done_reason": result.done_reason,
                "task_description": result.final_info.get("task_description"),
                "video_path": result.video_path,
                "policy_dir": str(policy_adapter.policy_dir),
                "env": ENV_NAME,
            },
        )
        return result
    finally:
        if writer is not None and video_path is not None:
            _close_video_writer(writer, output_path=video_path)


def evaluate_policy(
    *,
    policy_type: str,
    args,
    policy,
    cfg,
    preprocessor,
    postprocessor,
    policy_dir: Path,
) -> None:
    if policy_type == "prism_diffusion":
        raise NotImplementedError(
            "RoboCasa env eval does not yet support `prism_diffusion`. "
            "No fallback is attempted; use dataset evaluation or another online env."
        )
    if policy_type not in {"act", "streaming_act"}:
        raise NotImplementedError(
            "RoboCasa env eval currently supports `act` and `streaming_act` checkpoints."
        )
    if int(args.num_rollouts) <= 0:
        raise ValueError("`--num-rollouts` must be positive for RoboCasa eval.")
    if args.max_steps is not None and int(args.max_steps) <= 0:
        raise ValueError("`--max-steps` must be positive for RoboCasa eval.")
    if args.max_episodes_rendered is not None and int(args.max_episodes_rendered) < 0:
        raise ValueError("`--max-episodes-rendered` must be >= 0 when provided.")

    tasks = _parse_tasks(getattr(args, "task", None))
    conda_env = _normalize_conda_env_name(
        getattr(args, "robocasa_conda_env", DEFAULT_ROBOCASA_CONDA_ENV)
    )
    robocasa_split = _normalize_robocasa_split(
        getattr(args, "robocasa_split", None)
    )

    print(
        "[load] Validating RoboCasa tasks: "
        f"tasks={tasks}, split={robocasa_split}, conda_env={conda_env or '<current>'}"
    )
    available_tasks = list_available_robocasa_tasks(conda_env=conda_env)
    missing_tasks = [task_name for task_name in tasks if task_name not in available_tasks]
    if missing_tasks:
        preview = ", ".join(available_tasks[:12])
        raise ValueError(
            "Unknown RoboCasa task(s): "
            f"{missing_tasks}. Available examples: {preview}"
        )

    _validate_supported_input_features(cfg)
    capability_flags = resolve_policy_capability_flags(cfg)
    use_path_signature = capability_flags.use_path_signature
    use_prefix_sequence_training = capability_flags.use_prefix_sequence_training
    use_visual_prefix_memory = capability_flags.use_visual_prefix_memory
    use_delta_signature = capability_flags.use_delta_signature
    signature_backend = None
    if use_path_signature:
        signature_backend = resolve_signature_backend(
            getattr(args, "signature_backend", "auto")
        )
        print(
            "[load] RoboCasa online signatures: "
            f"backend={signature_backend}, history={cfg.history_length}, "
            f"depth={cfg.signature_depth}, dim={cfg.signature_dim}"
        )
    if use_prefix_sequence_training:
        mode_text = "online_visual_memory" if use_visual_prefix_memory else "full_prefix"
        print(
            "[load] RoboCasa prefix context: "
            f"mode={mode_text}, max_steps={cfg.prefix_train_max_steps}, "
            f"stride={cfg.prefix_frame_stride}, delta_signature={use_delta_signature}"
        )
    if use_visual_prefix_memory:
        initial_memory_debug = get_visual_memory_debug_stats(policy)
        if initial_memory_debug is not None:
            print(
                "[load] RoboCasa visual memory debug: "
                f"enabled={bool(initial_memory_debug.get('enabled', False))}, "
                f"num_slots={int(initial_memory_debug.get('num_slots', 0))}, "
                f"updates={int(initial_memory_debug.get('update_count', 0))}"
            )
    visual_specs = _resolve_visual_features(cfg)
    camera_names, camera_width, camera_height = _resolve_camera_setup(visual_specs)
    video_image_key = _choose_video_image_key(visual_specs)
    video_image_keys = _resolve_video_image_keys(visual_specs)
    max_videos_per_task = (
        int(args.max_episodes_rendered)
        if args.max_episodes_rendered is not None
        else int(args.num_rollouts)
    )
    # RoboCasaGymEnv zeros all camera observations when `enable_render=False`,
    # so ACT image inputs and saved rollout videos both require rendering enabled.
    enable_render = bool(visual_specs) or bool(video_image_key)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_success_count = 0
    overall_reward = 0.0
    overall_steps = 0
    overall_rollouts = 0
    overall_video_paths: list[str] = []
    per_task_summary: list[dict[str, Any]] = []
    resolved_max_steps_by_task: dict[str, int] = {}
    dataset_roots_by_task: dict[str, str] = {}
    eval_start_s = time.perf_counter()
    total_rollout_count = int(len(tasks) * int(args.num_rollouts))
    overall_progress = _maybe_create_tqdm(
        total=total_rollout_count,
        desc="RoboCasa Eval",
        unit="rollout",
        leave=True,
    )

    try:
        for task_name in tasks:
            task_dir = output_dir / _normalize_output_path_part(task_name)
            task_dataset_root = _resolve_task_dataset_root(
                task_name=task_name,
                args=args,
                policy_dir=policy_dir,
            )
            task_max_steps, task_max_steps_info = _resolve_task_max_steps(
                requested_max_steps=(
                    None if args.max_steps is None else int(args.max_steps)
                ),
                task_name=task_name,
                task_dataset_root=task_dataset_root,
            )
            resolved_max_steps_by_task[str(task_name)] = int(task_max_steps)
            if task_dataset_root is not None:
                dataset_roots_by_task[str(task_name)] = str(task_dataset_root)

            task_index_resolver = _TaskIndexResolver(task_dataset_root)
            policy_adapter = RoboCasaACTPolicyAdapter(
                policy=policy,
                cfg=cfg,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                policy_dir=policy_dir,
                task_index_resolver=task_index_resolver,
                signature_backend=signature_backend,
            )
            _progress_write(
                overall_progress,
                "[load] Building RoboCasa env: "
                f"task={task_name}, cameras={camera_names}, "
                f"resolution={camera_width}x{camera_height}, "
                f"max_steps={task_max_steps} "
                f"(source={task_max_steps_info['source']}, "
                f"dataset_longest={task_max_steps_info['dataset_longest_episode_length']}, "
                f"margin={task_max_steps_info['conservative_margin']})",
            )
            env = create_robocasa_benchmark_env(
                conda_env=conda_env,
                task=task_name,
                camera_names=camera_names,
                camera_width=camera_width,
                camera_height=camera_height,
                enable_render=enable_render,
                split=robocasa_split,
            )
            task_start_s = time.perf_counter()
            try:
                rollout_results: list[RoboCasaRolloutResult] = []
                task_success_count = 0
                task_total_reward = 0.0
                task_total_steps = 0

                for rollout_index in range(int(args.num_rollouts)):
                    rollout_seed = (
                        None
                        if getattr(args, "seed", None) is None
                        else int(args.seed) + rollout_index
                    )
                    save_video = bool(video_image_keys or video_image_key) and rollout_index < max_videos_per_task
                    video_path = (
                        task_dir / f"rollout_{rollout_index:03d}.mp4"
                        if save_video
                        else None
                    )
                    details_path = task_dir / f"rollout_{rollout_index:03d}.json"
                    if overall_progress is not None:
                        overall_progress.set_description(
                            f"RoboCasa Eval [{task_name}]"
                        )
                    step_progress = _maybe_create_tqdm(
                        total=int(task_max_steps),
                        desc=(
                            f"{task_name} "
                            f"[{rollout_index + 1}/{int(args.num_rollouts)}]"
                        ),
                        unit="step",
                        leave=False,
                    )
                    result = None
                    rollout_start_s = time.perf_counter()
                    try:
                        result = _run_single_rollout(
                            env=env,
                            policy_adapter=policy_adapter,
                            max_steps=int(task_max_steps),
                            seed=rollout_seed,
                            video_path=video_path,
                            details_path=details_path,
                            fps=int(args.fps),
                            video_image_key=video_image_key,
                            video_image_keys=video_image_keys,
                            step_progress=step_progress,
                        )
                    finally:
                        if step_progress is not None:
                            if result is not None:
                                step_progress.set_postfix(
                                    reward=f"{result.total_reward:.2f}",
                                    success=int(result.success),
                                    refresh=False,
                                )
                            step_progress.close()

                    assert result is not None
                    rollout_results.append(result)
                    task_success_count += int(result.success)
                    task_total_reward += float(result.total_reward)
                    task_total_steps += int(result.num_steps)
                    memory_debug_stats = (
                        get_visual_memory_debug_stats(policy)
                        if use_visual_prefix_memory
                        else None
                    )
                    if memory_debug_stats is not None:
                        result.final_info["visual_memory_debug"] = memory_debug_stats
                    if result.video_path is not None:
                        overall_video_paths.append(result.video_path)
                    overall_rollouts += 1
                    if overall_progress is not None:
                        overall_progress.update(1)
                        overall_progress.set_postfix(
                            success=f"{overall_success_count + task_success_count}/{overall_rollouts}",
                            reward=f"{(overall_reward + task_total_reward) / max(1, overall_rollouts):.2f}",
                            refresh=False,
                        )
                    _progress_write(
                        overall_progress,
                        "[eval][robocasa] "
                        f"task={task_name} rollout={rollout_index + 1}/{int(args.num_rollouts)} "
                        f"steps={result.num_steps} reward={result.total_reward:.2f} "
                        f"success={int(result.success)} elapsed="
                        f"{_format_elapsed_s(time.perf_counter() - rollout_start_s)}",
                    )

                task_eval = RoboCasaEvaluationResult(
                    task=str(task_name),
                    num_rollouts=int(args.num_rollouts),
                    max_steps=int(task_max_steps),
                    success_count=int(task_success_count),
                    success_rate=float(task_success_count / max(1, int(args.num_rollouts))),
                    average_reward=float(task_total_reward / max(1, int(args.num_rollouts))),
                    average_steps=float(task_total_steps / max(1, int(args.num_rollouts))),
                    rollout_results=rollout_results,
                    output_dir=str(task_dir),
                    summary_path=str(task_dir / "summary.json"),
                )
                task_summary = task_eval.to_summary_dict()
                task_summary.update(
                    {
                        "requested_max_steps": task_max_steps_info["requested_max_steps"],
                        "max_steps_source": task_max_steps_info["source"],
                        "dataset_root": task_max_steps_info["dataset_root"],
                        "dataset_longest_episode_length": task_max_steps_info[
                            "dataset_longest_episode_length"
                        ],
                        "max_steps_conservative_margin": task_max_steps_info[
                            "conservative_margin"
                        ],
                    }
                )
                write_summary(task_dir, task_summary)

                overall_success_count += int(task_success_count)
                overall_reward += float(task_total_reward)
                overall_steps += int(task_total_steps)
                per_task_summary.append(
                    {
                        "task": task_eval.task,
                        "num_rollouts": task_eval.num_rollouts,
                        "success_count": task_eval.success_count,
                        "success_rate": task_eval.success_rate,
                        "average_reward": task_eval.average_reward,
                        "average_steps": task_eval.average_steps,
                        "max_steps": int(task_max_steps),
                        "requested_max_steps": task_max_steps_info["requested_max_steps"],
                        "max_steps_source": task_max_steps_info["source"],
                        "dataset_root": task_max_steps_info["dataset_root"],
                        "dataset_longest_episode_length": task_max_steps_info[
                            "dataset_longest_episode_length"
                        ],
                        "max_steps_conservative_margin": task_max_steps_info[
                            "conservative_margin"
                        ],
                        "output_dir": task_eval.output_dir,
                        "summary_path": task_eval.summary_path,
                        "video_paths": [
                            result.video_path
                            for result in task_eval.rollout_results
                            if result.video_path is not None
                        ],
                    }
                )
                _progress_write(
                    overall_progress,
                    "[task][robocasa] "
                    f"task={task_name} success_rate={task_eval.success_rate:.3f} "
                    f"avg_reward={task_eval.average_reward:.2f} avg_steps={task_eval.average_steps:.1f} "
                    f"max_steps={task_max_steps} "
                    f"elapsed={_format_elapsed_s(time.perf_counter() - task_start_s)}",
                )
            finally:
                env.close()
    finally:
        if overall_progress is not None:
            overall_progress.close()

    summary = {
        "policy_type": policy_type,
        "policy_dir": str(policy_dir),
        "env": ENV_NAME,
        "task_spec": str(args.task),
        "tasks": tasks,
        "num_tasks": int(len(tasks)),
        "rollouts_per_task": int(args.num_rollouts),
        "total_rollouts": int(overall_rollouts),
        "max_steps": None if args.max_steps is None else int(args.max_steps),
        "resolved_max_steps_by_task": resolved_max_steps_by_task,
        "seed": None if getattr(args, "seed", None) is None else int(args.seed),
        "fps": int(args.fps),
        "robocasa_conda_env": conda_env,
        "robocasa_split": robocasa_split,
        "camera_names": camera_names,
        "camera_width": int(camera_width),
        "camera_height": int(camera_height),
        "video_image_key": video_image_key,
        "video_image_keys": video_image_keys,
        "max_episodes_rendered_per_task": int(max_videos_per_task),
        "dataset_root": None,
        "dataset_roots_by_task": dataset_roots_by_task,
        "success_count": int(overall_success_count),
        "success_rate": float(overall_success_count / max(1, overall_rollouts)),
        "average_reward": float(overall_reward / max(1, overall_rollouts)),
        "average_steps": float(overall_steps / max(1, overall_rollouts)),
        "per_task": per_task_summary,
        "video_paths": overall_video_paths,
    }
    summary_path = write_summary(output_dir, summary)
    print(
        "[done] RoboCasa eval summary saved to "
        f"{summary_path} "
        f"(elapsed={_format_elapsed_s(time.perf_counter() - eval_start_s)})"
    )
