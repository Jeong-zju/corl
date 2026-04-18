from __future__ import annotations

import argparse
import bisect
import datetime as dt
import inspect
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warnings

from dataset_utils import (
    DATASET_SPLIT_FILENAME,
    DEFAULT_LOCAL_DATA_ROOT,
    DatasetSplitSpec,
    build_dataset_split,
    ensure_lerobot_dataset_v30_compat,
    infer_dataset_repo_id,
    is_supported_lerobot_dataset_root,
    load_dataset_split,
    resolve_dataset_root,
    save_dataset_split,
    validate_dataset_root,
)
from policy_capabilities import policy_supports_signature_features
from policy_defaults import load_policy_mode_defaults_for_dataset

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated*",
    category=UserWarning,
)

RAW_IMAGE_ARRAY_STORAGE_ENCODING = "raw_uint8_array"
RAW_IMAGE_ARRAY_STORAGE_DTYPE = "uint8"
SIGNATURE_CACHE_LAYOUT_VERSION = 1
PATH_SIGNATURE_FEATURE_KEY = "observation.path_signature"
DELTA_SIGNATURE_FEATURE_KEY = "observation.delta_signature"
PREFIX_PATH_SIGNATURE_FEATURE_KEY = "observation.prefix_path_signature"
PREFIX_DELTA_SIGNATURE_FEATURE_KEY = "observation.prefix_delta_signature"
CHECKPOINTS_DIRNAME = "checkpoints"
LAST_CHECKPOINT_LINK_NAME = "last"
PRETRAINED_MODEL_DIRNAME = "pretrained_model"
TRAIN_CONFIG_FILENAME = "train_config.json"
TRAINING_STATE_DIRNAME = "training_state"
TRAINING_STEP_FILENAME = "training_step.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ResumeRunState:
    run_dir: Path
    checkpoint_dir: Path
    pretrained_model_dir: Path
    train_config_path: Path
    split_path: Path | None


def _sanitize_signature_cache_path_part(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "item"


def resolve_signature_cache_dir(
    dataset_root: Path,
    *,
    dataset_repo_id: str,
    cache_root: Path | None,
) -> Path:
    root = (
        dataset_root / ".signature_cache" / _sanitize_signature_cache_path_part(dataset_repo_id)
        if cache_root is None
        else Path(cache_root).resolve()
    )
    return root / f"signature_cache_v{SIGNATURE_CACHE_LAYOUT_VERSION}"


def _normalize_dataset_repo_id_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\\", "/").strip("/")


def resolve_effective_dataset_repo_id(
    *,
    requested_repo_id: str | None,
    default_repo_id: str | None,
    dataset_root: Path,
    local_data_root: Path,
) -> str:
    inferred_repo_id = infer_dataset_repo_id(
        dataset_root,
        local_data_root=local_data_root.resolve(),
    )
    requested = _normalize_dataset_repo_id_text(requested_repo_id)
    if not requested:
        return inferred_repo_id

    defaulted = _normalize_dataset_repo_id_text(default_repo_id)
    inferred = _normalize_dataset_repo_id_text(inferred_repo_id)
    if (
        defaulted
        and requested == defaulted
        and inferred
        and inferred != requested
        and inferred.startswith(f"{requested}/")
    ):
        print(
            "[INFO] dataset_repo_id: overriding broad defaults value "
            f"`{requested}` with inferred dataset id `{inferred_repo_id}`"
        )
        return inferred_repo_id
    return requested_repo_id if requested_repo_id is not None else inferred_repo_id


def _normalize_dataset_task_names(
    value: list[str] | tuple[str, ...] | set[str] | str | None,
) -> tuple[str, ...]:
    if value is None:
        return ()

    raw_items = value if isinstance(value, (list, tuple, set)) else str(value).split(",")
    task_names: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        task_name = str(raw_item).strip()
        if not task_name or task_name in seen:
            continue
        seen.add(task_name)
        task_names.append(task_name)
    return tuple(task_names)


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
        if normalized_dataset_text.startswith("data/"):
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


def _resolve_dataset_root_from_exact_task_names(
    selector: str | Path | None,
    *,
    local_data_root: Path,
    exact_task_names: tuple[str, ...],
) -> Path | None:
    if not exact_task_names:
        return None

    for base_candidate in _iter_exact_dataset_root_candidates(
        selector,
        local_data_root=local_data_root,
    ):
        if is_supported_lerobot_dataset_root(base_candidate):
            if len(exact_task_names) == 1 and base_candidate.name == exact_task_names[0]:
                return base_candidate.resolve()
            continue

        matched_roots: list[Path] = []
        for task_name in exact_task_names:
            task_candidate = base_candidate / task_name
            if not (
                is_supported_lerobot_dataset_root(task_candidate)
                and task_candidate.name == task_name
            ):
                matched_roots = []
                break
            matched_roots.append(task_candidate.resolve())

        if not matched_roots:
            continue
        if len(matched_roots) == 1:
            return matched_roots[0]

        raise NotImplementedError(
            "The selected defaults declare multiple exact dataset task names under "
            f"{base_candidate}, but train_policy currently expects a single local "
            "dataset root. "
            f"dataset_tasks={list(exact_task_names)}. "
            "Create a merged dataset root first or use a task-specific defaults file."
        )

    return None


def resolve_training_dataset_root(
    *,
    dataset: str | Path | None,
    defaults_dataset_root: str | Path | None,
    local_data_root: Path,
    exact_task_names: list[str] | tuple[str, ...] | set[str] | str | None = None,
) -> Path:
    resolved_local_data_root = local_data_root.resolve()
    normalized_task_names = _normalize_dataset_task_names(exact_task_names)
    last_error: FileNotFoundError | None = None

    if dataset is not None:
        matched_root = _resolve_dataset_root_from_exact_task_names(
            dataset,
            local_data_root=resolved_local_data_root,
            exact_task_names=normalized_task_names,
        )
        if matched_root is not None:
            return matched_root
        try:
            return resolve_dataset_root(
                dataset,
                local_data_root=resolved_local_data_root,
            )
        except FileNotFoundError as exc:
            last_error = exc

    if defaults_dataset_root is not None:
        matched_root = _resolve_dataset_root_from_exact_task_names(
            defaults_dataset_root,
            local_data_root=resolved_local_data_root,
            exact_task_names=normalized_task_names,
        )
        if matched_root is not None:
            return matched_root
        try:
            return resolve_dataset_root(
                defaults_dataset_root,
                local_data_root=resolved_local_data_root,
            )
        except FileNotFoundError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise FileNotFoundError("No dataset selector or defaults dataset_root was provided.")


def load_signature_cache_metadata(
    dataset_root: Path,
    *,
    dataset_repo_id: str,
    cache_root: Path | None,
) -> dict | None:
    metadata_path = (
        resolve_signature_cache_dir(
            dataset_root,
            dataset_repo_id=dataset_repo_id,
            cache_root=cache_root,
        )
        / "metadata.json"
    )
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _signature_feature_spec_from_cache_metadata(
    metadata: dict | None,
    *,
    key: str,
) -> dict | None:
    if not isinstance(metadata, dict):
        return None
    feature_shapes = metadata.get("feature_shapes", {})
    shape = feature_shapes.get(key) if isinstance(feature_shapes, dict) else None
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        return None
    feature_dim = int(shape[0])
    if feature_dim <= 0:
        return None

    prefix = "path_sig" if key.endswith("path_signature") else "delta_path_sig"
    return {
        "dtype": "float32",
        "shape": [feature_dim],
        "names": [f"{prefix}_{index}" for index in range(feature_dim)],
    }


def _identity_signature_stats(feature_dim: int) -> dict[str, list[float] | list[int]]:
    zeros = [0.0] * int(feature_dim)
    ones = [1.0] * int(feature_dim)
    return {
        "min": zeros,
        "max": zeros,
        "mean": zeros,
        "std": ones,
        "count": [0],
    }


def resolve_pre_normalized_signature_observation_keys(
    *,
    feature_keys: tuple[str, ...],
    reader_pre_normalized: bool,
    use_prefix_sequence_training: bool,
    use_path_signature: bool,
    use_delta_signature: bool,
) -> tuple[str, ...]:
    if not reader_pre_normalized:
        return ()

    resolved_keys = list(str(key) for key in feature_keys)
    if use_prefix_sequence_training:
        if use_path_signature and PATH_SIGNATURE_FEATURE_KEY in resolved_keys:
            resolved_keys.append(PREFIX_PATH_SIGNATURE_FEATURE_KEY)
        if use_delta_signature and DELTA_SIGNATURE_FEATURE_KEY in resolved_keys:
            resolved_keys.append(PREFIX_DELTA_SIGNATURE_FEATURE_KEY)
    return tuple(dict.fromkeys(resolved_keys))


def augment_dataset_metadata_with_signature_cache(
    *,
    info: dict,
    stats: dict,
    cache_metadata: dict | None,
    feature_keys: tuple[str, ...],
) -> tuple[dict, dict]:
    if not isinstance(cache_metadata, dict):
        return info, stats

    feature_specs = info.setdefault("features", {})
    for key in feature_keys:
        if key in feature_specs:
            continue
        cache_spec = _signature_feature_spec_from_cache_metadata(cache_metadata, key=key)
        if cache_spec is None:
            continue
        feature_specs[key] = cache_spec
        if key not in stats:
            stats[key] = _identity_signature_stats(int(cache_spec["shape"][0]))
    return info, stats


def get_signature_cache_only_feature_keys(info: dict | None) -> set[str]:
    """Return feature keys whose payload lives in `.signature_cache`, not parquet.

    Older datasets record `storage=signature_cache` only in top-level metadata
    like `info["path_signature"]`, while newer metadata may also attach storage
    directly on the feature spec. We support both layouts here.
    """
    if not isinstance(info, dict):
        return set()

    keys: set[str] = set()
    features = info.get("features", {})
    if isinstance(features, dict):
        for key, feature_spec in features.items():
            if (
                isinstance(feature_spec, dict)
                and str(feature_spec.get("storage", "")).lower() == "signature_cache"
            ):
                keys.add(str(key))

    for metadata_key in ("path_signature", "delta_signature"):
        metadata = info.get(metadata_key)
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("storage", "")).lower() != "signature_cache":
            continue
        feature_key = metadata.get("key")
        if feature_key:
            keys.add(str(feature_key))

    return keys


def ensure_streaming_act_importable(project_root: Path) -> None:
    streaming_act_src = (
        project_root / "policy" / "lerobot_policy_streaming_act" / "src"
    )
    if not streaming_act_src.exists():
        raise FileNotFoundError(
            f"Streaming ACT package source not found: {streaming_act_src}"
        )
    sys.path.insert(0, str(streaming_act_src))


def ensure_prism_diffusion_importable(project_root: Path) -> None:
    prism_diffusion_src = (
        project_root / "policy" / "lerobot_policy_prism_diffusion" / "src"
    )
    if not prism_diffusion_src.exists():
        raise FileNotFoundError(
            f"PRISM Diffusion package source not found: {prism_diffusion_src}"
        )
    sys.path.insert(0, str(prism_diffusion_src))


def teardown_wandb_safely(exit_code: int) -> None:
    try:
        import wandb
    except Exception:
        return

    try:
        wandb.teardown(exit_code=exit_code)
    except BaseException as exc:
        print(f"[WARN] wandb teardown failed during shutdown: {exc}")


def ensure_writable_hf_cache_env(project_root: Path) -> None:
    cache_root = (project_root / ".cache" / "huggingface").resolve()
    hf_home = cache_root / "home"
    hf_datasets_cache = cache_root / "datasets"
    xdg_cache_home = cache_root / "xdg"
    torch_home = cache_root / "torch"
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache_home.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_datasets_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_home))
    if "TORCH_HOME" not in os.environ:
        cached_torch_home = Path.home() / ".cache" / "torch"
        if (cached_torch_home / "hub" / "checkpoints").exists():
            os.environ["TORCH_HOME"] = str(cached_torch_home)
        else:
            os.environ["TORCH_HOME"] = str(torch_home)


def configure_torch_sharing_strategy(strategy: str | None) -> str | None:
    resolved_strategy = strategy or "auto"
    if resolved_strategy == "auto":
        resolved_strategy = "file_system" if os.name == "posix" else None

    if resolved_strategy is None:
        return None

    try:
        import torch.multiprocessing as torch_mp
    except Exception as exc:
        print(
            "[WARN] Failed to import torch.multiprocessing while configuring "
            f"sharing strategy: {exc}"
        )
        return None

    try:
        current_strategy = torch_mp.get_sharing_strategy()
    except Exception:
        current_strategy = None

    if current_strategy == resolved_strategy:
        return current_strategy

    try:
        torch_mp.set_sharing_strategy(resolved_strategy)
    except (AttributeError, RuntimeError, ValueError) as exc:
        print(
            "[WARN] Failed to set torch multiprocessing sharing strategy to "
            f"{resolved_strategy!r}: {exc}"
        )
        return current_strategy

    return resolved_strategy


def resolve_accelerator_mixed_precision(
    *,
    device: str,
    use_amp: bool,
    amp_dtype: str,
) -> str:
    if not use_amp:
        return "no"

    device_type = str(device).split(":", 1)[0]
    if device_type == "cuda":
        try:
            import torch
        except Exception as exc:
            print(f"[WARN] Failed to import torch while resolving AMP mode: {exc}")
            return "no"

        if not torch.cuda.is_available():
            print(
                "[WARN] CUDA AMP requested but CUDA is not available. Falling back to fp32."
            )
            return "no"

        if amp_dtype == "auto":
            if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                return "bf16"
            return "fp16"

        if (
            amp_dtype == "bf16"
            and not getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        ):
            print(
                "[WARN] bf16 AMP requested but this CUDA device does not support bf16. Falling back to fp16."
            )
            return "fp16"

        return amp_dtype

    if device_type == "xpu":
        return "bf16" if amp_dtype == "auto" else amp_dtype

    print(
        f"[WARN] AMP is not configured for device={device_type!r}. Falling back to fp32."
    )
    return "no"


def _read_distributed_env_int(
    *env_names: str,
    default: int,
) -> int:
    for env_name in env_names:
        raw_value = os.environ.get(env_name)
        if raw_value is None or not str(raw_value).strip():
            continue
        try:
            return int(raw_value)
        except ValueError:
            print(
                f"[WARN] Ignoring invalid distributed env var {env_name}={raw_value!r}."
            )
    return int(default)


def resolve_distributed_world_size_from_env() -> int:
    return max(1, _read_distributed_env_int("WORLD_SIZE", default=1))


def resolve_distributed_process_index_from_env() -> int:
    return max(
        0,
        _read_distributed_env_int(
            "RANK",
            "ACCELERATE_PROCESS_INDEX",
            default=0,
        ),
    )


def is_distributed_main_process_from_env() -> bool:
    return resolve_distributed_process_index_from_env() == 0


def resolve_train_run_stamp(now: dt.datetime | None = None) -> str:
    shared_run_stamp = os.environ.get("CORL_TRAIN_RUN_STAMP")
    if shared_run_stamp:
        return str(shared_run_stamp)
    moment = dt.datetime.now() if now is None else now
    return moment.strftime("%Y%m%d_%H%M%S")


def install_torch_dataloader_patch(
    *,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> None:
    import torch.utils.data

    original_dataloader_cls = getattr(
        install_torch_dataloader_patch,
        "_original_dataloader_cls",
        None,
    )
    if original_dataloader_cls is None:
        original_dataloader_cls = torch.utils.data.DataLoader
        install_torch_dataloader_patch._original_dataloader_cls = (  # type: ignore[attr-defined]
            original_dataloader_cls
        )

    resolved_prefetch_factor = (
        None if prefetch_factor is None else max(1, int(prefetch_factor))
    )
    patch_signature = (
        bool(persistent_workers),
        resolved_prefetch_factor,
    )
    current_dataloader_cls = torch.utils.data.DataLoader
    if (
        getattr(current_dataloader_cls, "_corl_patch_signature", None)
        == patch_signature
    ):
        return

    class PatchedDataLoader(original_dataloader_cls):
        _corl_patch_signature = patch_signature

        def __init__(self, *args, **kwargs):
            try:
                bound = inspect.signature(
                    original_dataloader_cls.__init__
                ).bind_partial(self, *args, **kwargs)
                num_workers = int(bound.arguments.get("num_workers", 0))
            except Exception:
                num_workers = int(kwargs.get("num_workers", 0) or 0)

            if num_workers > 0:
                kwargs.setdefault("persistent_workers", bool(persistent_workers))
                if resolved_prefetch_factor is not None:
                    kwargs.setdefault("prefetch_factor", resolved_prefetch_factor)

            super().__init__(*args, **kwargs)

    PatchedDataLoader.__name__ = original_dataloader_cls.__name__
    PatchedDataLoader.__qualname__ = original_dataloader_cls.__qualname__
    PatchedDataLoader.__module__ = original_dataloader_cls.__module__
    torch.utils.data.DataLoader = PatchedDataLoader


def install_episode_aware_sampler_patch() -> None:
    import lerobot.datasets.sampler as sampler_module
    import lerobot.scripts.lerobot_train as lerobot_train_module
    import torch

    if getattr(sampler_module, "_corl_relative_index_patch_installed", False):
        return

    class RelativeEpisodeAwareSampler:
        def __init__(
            self,
            dataset_from_indices: list[int],
            dataset_to_indices: list[int],
            episode_indices_to_use: list | None = None,
            drop_n_first_frames: int = 0,
            drop_n_last_frames: int = 0,
            shuffle: bool = False,
        ):
            selected_episode_indices = (
                None
                if episode_indices_to_use is None
                else {int(ep) for ep in episode_indices_to_use}
            )
            resolved_drop_n_first_frames = max(0, int(drop_n_first_frames))
            resolved_drop_n_last_frames = max(0, int(drop_n_last_frames))

            indices: list[int] = []
            relative_episode_start = 0
            for episode_idx, (start_index, end_index) in enumerate(
                zip(dataset_from_indices, dataset_to_indices, strict=True)
            ):
                start_index = int(start_index)
                end_index = int(end_index)
                episode_length = max(0, end_index - start_index)

                if (
                    selected_episode_indices is not None
                    and episode_idx not in selected_episode_indices
                ):
                    continue

                relative_episode_end = relative_episode_start + episode_length
                sample_start = relative_episode_start + resolved_drop_n_first_frames
                sample_end = relative_episode_end - resolved_drop_n_last_frames
                if sample_end > sample_start:
                    indices.extend(range(sample_start, sample_end))
                relative_episode_start = relative_episode_end

            self.indices = indices
            self.shuffle = bool(shuffle)

        def __iter__(self):
            if self.shuffle:
                for i in torch.randperm(len(self.indices)):
                    yield self.indices[int(i)]
            else:
                for i in self.indices:
                    yield i

        def __len__(self) -> int:
            return len(self.indices)

    sampler_module.EpisodeAwareSampler = RelativeEpisodeAwareSampler
    lerobot_train_module.EpisodeAwareSampler = RelativeEpisodeAwareSampler
    sampler_module._corl_relative_index_patch_installed = True


def install_lerobot_dataset_load_patch() -> None:
    import datasets
    import pyarrow.dataset as pa_ds
    import torch
    import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import FrameTimestampError
    from lerobot.datasets.utils import (
        get_hf_features_from_features,
        hf_transform_to_torch,
        load_nested_dataset,
    )
    try:
        from lerobot_policy_streaming_act.signature_cache import (
            get_signature_cache_reader_for_dataset,
            get_signature_cache_runtime,
            signature_cache_feature_keys_for_dataset,
        )
    except ModuleNotFoundError:
        def signature_cache_feature_keys_for_dataset(_dataset_root: Path) -> tuple[str, ...]:
            return ()

        def get_signature_cache_reader_for_dataset(_dataset_root: Path):
            return None

        def get_signature_cache_runtime():
            return None

    if getattr(LeRobotDataset, "_custom_dataset_load_patch_installed", False):
        return

    original_load_metadata = LeRobotDatasetMetadata.load_metadata
    original_getitem = LeRobotDataset.__getitem__

    def is_raw_array_image_feature(feature_spec: dict | None) -> bool:
        return bool(
            isinstance(feature_spec, dict)
            and feature_spec.get("dtype") == "image"
            and feature_spec.get("storage_encoding") == RAW_IMAGE_ARRAY_STORAGE_ENCODING
        )

    def build_hf_array_feature(*, shape: list[int] | tuple[int, ...], dtype: str):
        resolved_shape = tuple(int(dim) for dim in shape)
        if resolved_shape == (1,):
            return datasets.Value(dtype=dtype)
        if len(resolved_shape) == 1:
            return datasets.Sequence(
                length=int(resolved_shape[0]),
                feature=datasets.Value(dtype=dtype),
            )
        if len(resolved_shape) == 2:
            return datasets.Array2D(shape=resolved_shape, dtype=dtype)
        if len(resolved_shape) == 3:
            return datasets.Array3D(shape=resolved_shape, dtype=dtype)
        if len(resolved_shape) == 4:
            return datasets.Array4D(shape=resolved_shape, dtype=dtype)
        if len(resolved_shape) == 5:
            return datasets.Array5D(shape=resolved_shape, dtype=dtype)
        raise ValueError(
            f"Unsupported HF array feature shape: shape={resolved_shape}, dtype={dtype}"
        )

    def get_hf_features_from_features_with_raw_images(
        features: dict,
    ) -> datasets.Features:
        hf_features = {}
        for key, feature_spec in features.items():
            dtype = feature_spec["dtype"]
            shape = tuple(feature_spec.get("shape", ()))
            if dtype == "video":
                continue
            if dtype == "image":
                if is_raw_array_image_feature(feature_spec):
                    storage_dtype = str(
                        feature_spec.get(
                            "storage_dtype",
                            RAW_IMAGE_ARRAY_STORAGE_DTYPE,
                        )
                    )
                    hf_features[key] = build_hf_array_feature(
                        shape=shape,
                        dtype=storage_dtype,
                    )
                else:
                    hf_features[key] = datasets.Image()
                continue
            hf_features[key] = build_hf_array_feature(shape=shape, dtype=dtype)
        return datasets.Features(hf_features)

    def convert_raw_image_tensor_to_chw_float(
        value: torch.Tensor | np.ndarray | list,
        *,
        key: str,
        storage_dtype: str,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            np_value = np.asarray(value, dtype=np.dtype(storage_dtype))
            tensor = torch.from_numpy(np_value)

        if tensor.ndim != 3:
            raise ValueError(
                f"Raw array image feature `{key}` must have rank 3 (H, W, C). "
                f"Got shape={tuple(tensor.shape)}."
            )

        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] not in (1, 3, 4):
            raise ValueError(
                f"Raw array image feature `{key}` has ambiguous shape={tuple(tensor.shape)}. "
                "Expected HWC with 1/3/4 channels."
            )

        tensor = tensor.contiguous().to(dtype=torch.float32)
        if tensor.numel() > 0 and float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        return tensor

    def build_hf_transform_to_torch(feature_specs: dict):
        raw_image_storage_dtype_by_key = {
            key: str(
                feature_spec.get(
                    "storage_dtype",
                    RAW_IMAGE_ARRAY_STORAGE_DTYPE,
                )
            )
            for key, feature_spec in feature_specs.items()
            if is_raw_array_image_feature(feature_spec)
        }
        if not raw_image_storage_dtype_by_key:
            return hf_transform_to_torch

        def transform(items_dict: dict[str, list[object]]) -> dict[str, list[object]]:
            raw_items = {
                key: items_dict[key]
                for key in raw_image_storage_dtype_by_key
                if key in items_dict
            }
            non_raw_items = {
                key: value
                for key, value in items_dict.items()
                if key not in raw_image_storage_dtype_by_key
            }

            converted = hf_transform_to_torch(non_raw_items)
            for key, values in raw_items.items():
                converted[key] = [
                    convert_raw_image_tensor_to_chw_float(
                        value,
                        key=key,
                        storage_dtype=raw_image_storage_dtype_by_key[key],
                    )
                    for value in values
                ]
            return converted

        return transform

    def _install_compact_relative_index_layout(self):
        layout_start_s = time.perf_counter()
        self._absolute_to_relative_idx = None
        self._compact_relative_index_abs_starts = None
        self._compact_relative_index_abs_ends = None
        self._compact_relative_index_rel_starts = None
        self._compact_relative_index_episode_ids = None

        if self.episodes is None:
            return

        selected_episode_ids = sorted(int(ep) for ep in self.episodes)
        abs_starts: list[int] = []
        abs_ends: list[int] = []
        rel_starts: list[int] = []
        rel_offset = 0
        for ep_idx in selected_episode_ids:
            episode_meta = self.meta.episodes[ep_idx]
            abs_start = int(episode_meta["dataset_from_index"])
            abs_end = int(episode_meta["dataset_to_index"])
            abs_starts.append(abs_start)
            abs_ends.append(abs_end)
            rel_starts.append(rel_offset)
            rel_offset += abs_end - abs_start

        hf_num_rows = (
            len(self.hf_dataset) if self.hf_dataset is not None else rel_offset
        )
        if rel_offset != hf_num_rows:
            raise RuntimeError(
                "Compact relative-index layout mismatch: expected "
                f"{hf_num_rows} rows from filtered HF dataset, but selected episode spans "
                f"cover {rel_offset} rows."
            )

        self._compact_relative_index_episode_ids = tuple(selected_episode_ids)
        self._compact_relative_index_abs_starts = tuple(abs_starts)
        self._compact_relative_index_abs_ends = tuple(abs_ends)
        self._compact_relative_index_rel_starts = tuple(rel_starts)
        layout_elapsed_s = time.perf_counter() - layout_start_s
        print(
            "[INFO] dataset.relative_index_layout: "
            f"{layout_elapsed_s:.1f}s [episode_offsets] "
            f"(episodes={len(selected_episode_ids)}, rows={hf_num_rows})"
        )

    def _map_absolute_index_to_relative(self, abs_idx: int) -> int:
        compact_abs_starts = getattr(self, "_compact_relative_index_abs_starts", None)
        if not compact_abs_starts:
            return abs_idx

        compact_abs_ends = self._compact_relative_index_abs_ends
        compact_rel_starts = self._compact_relative_index_rel_starts
        position = bisect.bisect_right(compact_abs_starts, int(abs_idx)) - 1
        if position < 0 or int(abs_idx) >= compact_abs_ends[position]:
            raise KeyError(
                f"Absolute frame index {abs_idx} is not covered by the selected episode layout."
            )
        return compact_rel_starts[position] + (
            int(abs_idx) - compact_abs_starts[position]
        )

    def _map_absolute_indices_to_relative(
        self, absolute_indices: list[int]
    ) -> list[int]:
        explicit_mapping = getattr(self, "_absolute_to_relative_idx", None)
        if explicit_mapping is not None:
            return [
                explicit_mapping[idx.item() if isinstance(idx, torch.Tensor) else idx]
                for idx in absolute_indices
            ]

        compact_abs_starts = getattr(self, "_compact_relative_index_abs_starts", None)
        if not compact_abs_starts:
            return [
                idx.item() if isinstance(idx, torch.Tensor) else int(idx)
                for idx in absolute_indices
            ]

        return [
            _map_absolute_index_to_relative(
                self,
                idx.item() if isinstance(idx, torch.Tensor) else int(idx),
            )
            for idx in absolute_indices
        ]

    def init_with_compact_relative_index_layout(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        vcodec: str = "libsvtav1",
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = (
            Path(root) if root else lerobot_dataset_module.HF_LEROBOT_HOME / repo_id
        )
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = (
            revision if revision else lerobot_dataset_module.CODEBASE_VERSION
        )
        self.video_backend = (
            video_backend
            if video_backend
            else lerobot_dataset_module.get_safe_default_codec()
        )
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0
        self.vcodec = lerobot_dataset_module.resolve_vcodec(vcodec)
        self._encoder_threads = encoder_threads

        self.image_writer = None
        self.episode_buffer = None
        self.writer = None
        self.latest_episode = None
        self._current_file_start_frame = None
        self._streaming_encoder = None

        self.root.mkdir(exist_ok=True, parents=True)

        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )

        self._lazy_loading = False
        self._recorded_frames = self.meta.total_frames
        self._writer_closed_for_reading = False

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.hf_dataset = self.load_hf_dataset()
            if not self._check_cached_episodes_sufficient():
                raise FileNotFoundError(
                    "Cached dataset doesn't contain all requested episodes"
                )
        except (FileNotFoundError, NotADirectoryError):
            if lerobot_dataset_module.is_valid_version(self.revision):
                self.revision = lerobot_dataset_module.get_safe_version(
                    self.repo_id, self.revision
                )
            self.download(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        _install_compact_relative_index_layout(self)
        self._signature_cache_reader = get_signature_cache_reader_for_dataset(
            self.root
        )
        if self._signature_cache_reader is not None:
            print(
                "[INFO] signature_cache: attached "
                f"(keys={list(self._signature_cache_reader.feature_keys)}, mode={self._signature_cache_reader.mode})"
            )

        if self.delta_timestamps is not None:
            lerobot_dataset_module.check_delta_timestamps(
                self.delta_timestamps, self.fps, self.tolerance_s
            )
            self.delta_indices = lerobot_dataset_module.get_delta_indices(
                self.delta_timestamps, self.fps
            )

        if streaming_encoding and len(self.meta.video_keys) > 0:
            self._streaming_encoder = lerobot_dataset_module.StreamingVideoEncoder(
                fps=self.meta.fps,
                vcodec=self.vcodec,
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                preset=None,
                queue_maxsize=encoder_queue_maxsize,
                encoder_threads=encoder_threads,
            )

    def load_metadata_with_timing(self):
        start_s = time.perf_counter()
        original_load_metadata(self)
        runtime = get_signature_cache_runtime()
        if (
            runtime is not None
            and getattr(runtime, "enabled", False)
            and Path(self.root).resolve() == Path(runtime.dataset_root).resolve()
        ):
            cache_metadata = load_signature_cache_metadata(
                Path(self.root),
                dataset_repo_id=str(runtime.dataset_repo_id),
                cache_root=runtime.cache_root,
            )
            self.info, self.stats = augment_dataset_metadata_with_signature_cache(
                info=self.info,
                stats=self.stats,
                cache_metadata=cache_metadata,
                feature_keys=tuple(str(key) for key in runtime.feature_keys),
            )
        elapsed_s = time.perf_counter() - start_s
        print(
            f"[INFO] dataset.load_metadata: {elapsed_s:.1f}s "
            f"(root={self.root / 'meta'})"
        )

    def load_hf_dataset_with_timing(self):
        load_start_s = time.perf_counter()
        runtime_cached_feature_keys = set(
            signature_cache_feature_keys_for_dataset(self.root)
        )
        metadata_cached_feature_keys = get_signature_cache_only_feature_keys(
            getattr(self.meta, "info", None)
        )
        cached_feature_keys = (
            runtime_cached_feature_keys | metadata_cached_feature_keys
        )
        load_feature_specs = {
            key: feature_spec
            for key, feature_spec in self.features.items()
            if key not in cached_feature_keys
        }
        if metadata_cached_feature_keys:
            print(
                "[INFO] dataset.load_hf_dataset: excluding signature-cache-only "
                f"features from parquet load: {sorted(metadata_cached_feature_keys)}"
            )
        if runtime_cached_feature_keys:
            print(
                "[INFO] signature_cache: excluding cached parquet columns from HF load: "
                f"{sorted(runtime_cached_feature_keys)}"
            )
        hf_transform = build_hf_transform_to_torch(load_feature_specs)
        parquet_paths = sorted((self.root / "data").glob("*/*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(
                f"Provided directory does not contain any parquet file: {self.root / 'data'}"
            )
        filters = (
            pa_ds.field("episode_index").isin(self.episodes)
            if self.episodes is not None
            else None
        )
        load_columns = [
            key
            for key, feature_spec in load_feature_specs.items()
            if feature_spec.get("dtype") != "video"
        ]
        hf_dataset = datasets.Dataset.from_parquet(
            [str(path) for path in parquet_paths],
            columns=load_columns,
            filters=filters,
            features=get_hf_features_from_features_with_raw_images(load_feature_specs),
        )
        hf_dataset.set_transform(hf_transform)
        load_elapsed_s = time.perf_counter() - load_start_s
        print(
            "[INFO] dataset.load_hf_dataset: "
            f"{load_elapsed_s:.1f}s "
            f"(rows={len(hf_dataset)})"
        )
        return hf_dataset

    def get_query_timestamps_with_compact_relative_index(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                absolute_indices = tuple(int(idx) for idx in query_indices[key])
                relative_indices = _map_absolute_indices_to_relative(
                    self, list(absolute_indices)
                )
                timestamps = torch.stack(
                    self.hf_dataset[relative_indices]["timestamp"]
                ).tolist()
                query_timestamps[key] = timestamps
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def query_videos_with_timestamp_layout_detection(
        self,
        query_timestamps: dict[str, list[float]],
        ep_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Handle both episode-relative and merged-file absolute parquet timestamps.

        Upstream LeRobot expects parquet `timestamp` values to be episode-relative,
        then shifts them by `meta/episodes[*].videos/*/from_timestamp` before
        decoding. Some older locally processed datasets instead stored absolute
        timestamps after video merging, which would double-apply the episode offset
        and push frame indices past the end of the shared mp4.
        """
        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = float(ep[f"videos/{vid_key}/from_timestamp"])
            to_timestamp = float(
                ep.get(f"videos/{vid_key}/to_timestamp", from_timestamp)
            )
            episode_duration = max(to_timestamp - from_timestamp, 0.0)
            query_ts_values = [float(ts) for ts in query_ts]

            # Absolute timestamps are typically much larger than the episode-local
            # duration once videos from many episodes have been merged into one mp4.
            use_absolute_timestamps = bool(
                from_timestamp > self.tolerance_s
                and query_ts_values
                and max(query_ts_values)
                > episode_duration + max(float(self.tolerance_s), 1e-6)
            )

            if use_absolute_timestamps:
                shifted_query_ts = query_ts_values
                if not getattr(
                    self,
                    "_corl_warned_absolute_video_timestamp_layout",
                    False,
                ):
                    print(
                        "[WARN] Detected absolute parquet video timestamps in the "
                        "dataset. Enabling compatibility mode and skipping the "
                        "per-episode from_timestamp offset during video decoding."
                    )
                    self._corl_warned_absolute_video_timestamp_layout = True
            else:
                shifted_query_ts = [from_timestamp + ts for ts in query_ts_values]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            try:
                frames = lerobot_dataset_module.decode_video_frames(
                    video_path,
                    shifted_query_ts,
                    self.tolerance_s,
                    self.video_backend,
                )
            except FrameTimestampError:
                relaxed_tolerance_s = max(
                    float(self.tolerance_s),
                    1.0 / max(float(self.fps), 1.0),
                )
                if relaxed_tolerance_s <= float(self.tolerance_s):
                    raise
                frames = lerobot_dataset_module.decode_video_frames(
                    video_path,
                    shifted_query_ts,
                    relaxed_tolerance_s,
                    self.video_backend,
                )
                if not getattr(
                    self,
                    "_corl_warned_relaxed_video_tolerance",
                    False,
                ):
                    print(
                        "[WARN] Video frame timestamps were not perfectly aligned "
                        "with the dataset metadata. Retrying video decode with a "
                        f"relaxed tolerance of {relaxed_tolerance_s:.4f}s."
                    )
                    self._corl_warned_relaxed_video_tolerance = True
            item[vid_key] = frames.squeeze(0)
        return item

    def query_hf_dataset_with_compact_relative_index(
        self, query_indices: dict[str, list[int]]
    ) -> dict:
        result: dict = {}
        signature_cache_reader = getattr(self, "_signature_cache_reader", None)
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            if signature_cache_reader is not None and signature_cache_reader.has_key(key):
                absolute_indices = [
                    idx.item() if isinstance(idx, torch.Tensor) else int(idx)
                    for idx in q_idx
                ]
                result[key] = signature_cache_reader.get_many(key, absolute_indices)
                continue
            relative_indices = _map_absolute_indices_to_relative(self, q_idx)
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def getitem_with_signature_cache(self, idx):
        item = original_getitem(self, idx)
        signature_cache_reader = getattr(self, "_signature_cache_reader", None)
        if signature_cache_reader is None:
            return item
        absolute_index = int(
            item["index"].item() if isinstance(item["index"], torch.Tensor) else item["index"]
        )
        for key in signature_cache_reader.feature_keys:
            if key not in item:
                item[key] = signature_cache_reader.get(key, absolute_index)
        return item

    LeRobotDataset.__init__ = init_with_compact_relative_index_layout
    LeRobotDataset.__getitem__ = getitem_with_signature_cache
    LeRobotDatasetMetadata.load_metadata = load_metadata_with_timing
    LeRobotDataset.load_hf_dataset = load_hf_dataset_with_timing
    LeRobotDataset._get_query_timestamps = (
        get_query_timestamps_with_compact_relative_index
    )
    LeRobotDataset._query_videos = query_videos_with_timestamp_layout_detection
    LeRobotDataset._query_hf_dataset = query_hf_dataset_with_compact_relative_index
    LeRobotDataset._custom_dataset_load_patch_installed = True # pyright: ignore[reportAttributeAccessIssue]


def summarize_visual_storage_modes(dataset_root: Path) -> dict[str, int]:
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    counts = {"image": 0, "video": 0}
    for spec in info.get("features", {}).values():
        if not isinstance(spec, dict):
            continue
        dtype = spec.get("dtype")
        if dtype in counts:
            counts[str(dtype)] += 1
    return counts


def resolve_signature_dim(
    dataset_root: Path,
    *,
    dataset_repo_id: str,
    signature_cache_root: Path | None,
    use_path_signature: bool,
    signature_dim: int,
) -> int:
    if not use_path_signature:
        return signature_dim

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    sig_key = "observation.path_signature"
    sig_spec = info.get("features", {}).get(sig_key)
    if sig_spec is None:
        cache_metadata = load_signature_cache_metadata(
            dataset_root,
            dataset_repo_id=dataset_repo_id,
            cache_root=signature_cache_root,
        )
        sig_spec = _signature_feature_spec_from_cache_metadata(
            cache_metadata,
            key=sig_key,
        )
    if sig_spec is None:
        raise KeyError(
            f"Dataset feature '{sig_key}' not found in {dataset_root / 'meta/info.json'}, "
            "and no compatible entry was found in .signature_cache metadata. "
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


def validate_delta_signature_dataset(
    dataset_root: Path,
    *,
    dataset_repo_id: str,
    signature_cache_root: Path | None,
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
        cache_metadata = load_signature_cache_metadata(
            dataset_root,
            dataset_repo_id=dataset_repo_id,
            cache_root=signature_cache_root,
        )
        delta_sig_spec = _signature_feature_spec_from_cache_metadata(
            cache_metadata,
            key=delta_sig_key,
        )
    if delta_sig_spec is None:
        raise KeyError(
            f"Dataset feature `{delta_sig_key}` not found in {dataset_root / 'meta/info.json'}, "
            "and no compatible entry was found in .signature_cache metadata. "
            "Regenerate the dataset with delta-signature export enabled."
        )
    shape = delta_sig_spec.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 1 or int(shape[0]) <= 0:
        raise ValueError(
            f"Invalid shape for `{delta_sig_key}` in dataset info: {shape}. "
            "Expected [signature_dim]."
        )
    if delta_sig_key not in stats and delta_sig_key in features:
        raise KeyError(
            f"Dataset stats for `{delta_sig_key}` are missing from {dataset_root / 'meta/stats.json'}."
        )


def validate_prefix_sequence_support(
    *,
    policy_name: str,
    use_prefix_sequence_training: bool,
    context: str,
) -> None:
    if not use_prefix_sequence_training:
        return
    if policy_name not in {"streaming_act", "prism_diffusion"}:
        raise NotImplementedError(
            "Prefix-sequence training is currently implemented only for "
            "`streaming_act` and `prism_diffusion`. "
            f"Got policy={policy_name!r} during {context}."
        )


def validate_prefix_sequence_dataset(
    dataset_root: Path,
    *,
    dataset_repo_id: str,
    signature_cache_root: Path | None,
    use_prefix_sequence_training: bool,
    use_imagenet_stats: bool,
    use_path_signature: bool,
    use_delta_signature: bool,
) -> None:
    if not use_prefix_sequence_training:
        return

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((dataset_root / "meta/stats.json").read_text(encoding="utf-8"))
    cache_metadata = load_signature_cache_metadata(
        dataset_root,
        dataset_repo_id=dataset_repo_id,
        cache_root=signature_cache_root,
    )
    info, stats = augment_dataset_metadata_with_signature_cache(
        info=info,
        stats=stats,
        cache_metadata=cache_metadata,
        feature_keys=tuple(
            key
            for key, enabled in (
                ("observation.path_signature", bool(use_path_signature)),
                ("observation.delta_signature", bool(use_delta_signature)),
            )
            if enabled
        ),
    )
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
        and key != "observation.anchor_image"
        and not str(key).startswith("observation.prefix_images.")
    ]
    if not camera_keys:
        raise KeyError(
            "Prefix-sequence mode requires at least one regular observation image feature."
        )
    missing_camera_stats = [key for key in camera_keys if key not in stats]
    if missing_camera_stats:
        if use_imagenet_stats:
            raise KeyError(
                "Prefix-sequence mode could not apply ImageNet camera stats because "
                f"meta/stats.json is missing {missing_camera_stats}."
            )
        print(
            "[WARN] Prefix-sequence mode found observation cameras without stats in "
            f"{dataset_root / 'meta/stats.json'}:\n"
            + "\n".join(f"  - {key}" for key in missing_camera_stats)
            + "\n[WARN] Current and prefix image features will use identity "
            "normalization for those keys."
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


def validate_visual_prefix_memory_support(
    *,
    use_visual_prefix_memory: bool,
    use_prefix_sequence_training: bool,
) -> None:
    if not use_visual_prefix_memory:
        return
    if not use_prefix_sequence_training:
        raise ValueError(
            "`--enable-visual-prefix-memory` requires "
            "`--enable-prefix-sequence-training`."
        )


def build_policy_feature_overrides(
    dataset_root: Path,
    *,
    dataset_repo_id: str,
    signature_cache_root: Path | None,
    use_prefix_sequence_training: bool,
    prefix_train_max_steps: int,
    use_path_signature: bool,
    use_delta_signature: bool,
):
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    cache_metadata = load_signature_cache_metadata(
        dataset_root,
        dataset_repo_id=dataset_repo_id,
        cache_root=signature_cache_root,
    )
    info, _stats = augment_dataset_metadata_with_signature_cache(
        info=info,
        stats={},
        cache_metadata=cache_metadata,
        feature_keys=tuple(
            key
            for key, enabled in (
                ("observation.path_signature", bool(use_path_signature)),
                ("observation.delta_signature", bool(use_delta_signature)),
            )
            if enabled
        ),
    )

    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
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

    try:
        from lerobot_policy_streaming_act.prefix_image_cache import (
            get_prefix_image_cache_reader_for_dataset,
        )
    except ModuleNotFoundError:

        def get_prefix_image_cache_reader_for_dataset(_dataset_root: Path):
            return None

    if getattr(lerobot_train_module, "_prefix_sequence_patch_installed", False):
        return

    original_make_dataset = dataset_factory.make_dataset

    def make_dataset_with_prefix(cfg):
        dataset_load_start_s = time.perf_counter()
        print(
            "[INFO] Building local dataset view. Large parquet/video datasets "
            "can take several minutes before training starts."
        )
        dataset = original_make_dataset(cfg)
        dataset_load_elapsed_s = time.perf_counter() - dataset_load_start_s
        print(
            "[INFO] Dataset ready in "
            f"{dataset_load_elapsed_s:.1f}s "
            f"({dataset.num_episodes} episodes, {dataset.num_frames} frames)."
        )
        policy_cfg = cfg.policy
        if not bool(getattr(policy_cfg, "use_prefix_sequence_training", False)):
            return dataset
        policy_name = getattr(policy_cfg, "type", None)
        if policy_name == "streaming_act":
            from lerobot_policy_streaming_act.prefix_sequence import (
                PrefixSequenceDataset,
            )

            wrapper_cls = PrefixSequenceDataset
        elif policy_name == "prism_diffusion":
            from lerobot_policy_prism_diffusion.prefix_dataset import (
                PrismDiffusionPrefixDataset,
            )

            wrapper_cls = PrismDiffusionPrefixDataset
        else:
            return dataset
        if isinstance(dataset, wrapper_cls):
            return dataset
        prefix_image_cache_reader = get_prefix_image_cache_reader_for_dataset(
            Path(dataset.root)
        )
        if prefix_image_cache_reader is not None:
            print(
                "[INFO] prefix_image_cache: attached "
                f"(keys={list(prefix_image_cache_reader.camera_keys)}, "
                f"mode={prefix_image_cache_reader.mode})"
            )
        return wrapper_cls(
            dataset,
            prefix_train_max_steps=int(policy_cfg.prefix_train_max_steps),
            prefix_frame_stride=int(policy_cfg.prefix_frame_stride),
            prefix_pad_value=float(policy_cfg.prefix_pad_value),
            use_path_signature=bool(getattr(policy_cfg, "use_path_signature", False)),
            use_delta_signature=bool(getattr(policy_cfg, "use_delta_signature", False)),
            prefix_image_cache_reader=prefix_image_cache_reader,
        )

    dataset_factory.make_dataset = make_dataset_with_prefix
    lerobot_train_module.make_dataset = make_dataset_with_prefix
    lerobot_train_module._prefix_sequence_patch_installed = True


def parquet_has_columns(dataset_root: Path, required_keys: list[str]) -> bool:
    if not required_keys:
        return True
    data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(
            f"No parquet files were found under {dataset_root / 'data'}."
        )
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`pyarrow` is required to inspect parquet schema for signature storage."
        ) from exc
    schema = pq.read_schema(data_files[0])
    schema_names = set(schema.names)
    return all(key in schema_names for key in required_keys)


def default_train_output_root(policy_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "train" / default_policy_series_name(policy_name)


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
    for prefix in ("data/",):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break

    parts = [
        normalize_output_path_part(part)
        for part in raw.split("/")
        if part not in {"", ".", ".."}
    ]
    if not parts:
        return None
    return Path(*parts)


def resolve_default_train_output_root(
    *,
    policy_name: str,
    dataset_selector: str | None,
) -> Path:
    base = PROJECT_ROOT / "outputs" / "train"
    dataset_subdir = default_dataset_output_subdir(dataset_selector)
    if dataset_subdir is not None:
        return base / dataset_subdir / default_policy_series_name(policy_name)
    return default_train_output_root(policy_name)


def _resolve_train_output_root_candidates(raw: Path) -> list[Path]:
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, PROJECT_ROOT / raw])

    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def iter_training_run_dirs(train_output_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[Path] = set()

    for resolved_root in _resolve_train_output_root_candidates(
        train_output_root.expanduser()
    ):
        if not resolved_root.is_dir():
            continue
        candidates = [
            path
            for path in resolved_root.iterdir()
            if path.is_dir()
        ]
        candidates.sort(
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            run_dirs.append(resolved)

    return run_dirs


def iter_resume_checkpoint_dirs(run_dir: Path) -> list[Path]:
    checkpoints_dir = run_dir / CHECKPOINTS_DIRNAME
    if not checkpoints_dir.is_dir():
        return []

    candidates: list[Path] = []
    last_dir = checkpoints_dir / LAST_CHECKPOINT_LINK_NAME
    if last_dir.is_dir():
        candidates.append(last_dir.resolve(strict=False))

    numbered_dirs = [
        path
        for path in checkpoints_dir.iterdir()
        if path.is_dir() and path.name != LAST_CHECKPOINT_LINK_NAME
    ]
    numbered_dirs.sort(
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    candidates.extend(path.resolve(strict=False) for path in numbered_dirs)

    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def is_resumable_checkpoint_dir(checkpoint_dir: Path) -> bool:
    pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIRNAME
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIRNAME
    return (
        pretrained_model_dir.is_dir()
        and (pretrained_model_dir / TRAIN_CONFIG_FILENAME).is_file()
        and training_state_dir.is_dir()
        and (training_state_dir / TRAINING_STEP_FILENAME).is_file()
    )


def resolve_resume_run_state(train_output_root: Path) -> ResumeRunState:
    run_dirs = iter_training_run_dirs(train_output_root)
    if not run_dirs:
        raise FileNotFoundError(
            "Resume requested, but no prior training runs were found under "
            f"{train_output_root}."
        )

    for run_dir in run_dirs:
        for checkpoint_dir in iter_resume_checkpoint_dirs(run_dir):
            if not is_resumable_checkpoint_dir(checkpoint_dir):
                continue
            pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIRNAME
            split_path = run_dir / DATASET_SPLIT_FILENAME
            return ResumeRunState(
                run_dir=run_dir,
                checkpoint_dir=checkpoint_dir,
                pretrained_model_dir=pretrained_model_dir,
                train_config_path=pretrained_model_dir / TRAIN_CONFIG_FILENAME,
                split_path=split_path if split_path.is_file() else None,
            )

    raise FileNotFoundError(
        "Resume requested, but no resumable checkpoint was found under "
        f"{train_output_root}."
    )


def load_resume_dataset_split(
    *,
    resume_run_state: ResumeRunState,
    source_dataset_root: Path,
    dataset_repo_id: str,
) -> DatasetSplitSpec | None:
    if resume_run_state.split_path is None:
        return None

    split_spec = load_dataset_split(resume_run_state.split_path)
    saved_dataset_root = Path(split_spec.dataset_root).expanduser().resolve()
    expected_dataset_root = source_dataset_root.resolve()

    if saved_dataset_root != expected_dataset_root:
        raise ValueError(
            "The latest resumable run was created from a different dataset root. "
            f"saved={saved_dataset_root}, expected={expected_dataset_root}."
        )
    if str(split_spec.dataset_repo_id) != str(dataset_repo_id):
        raise ValueError(
            "The latest resumable run was created from a different dataset_repo_id. "
            f"saved={split_spec.dataset_repo_id!r}, expected={dataset_repo_id!r}."
        )
    return split_spec


def default_wandb_project_name(
    dataset_repo_id: str | None,
    dataset_root: Path,
) -> str:
    candidate = str(dataset_repo_id or "").strip()
    if candidate:
        candidate = candidate.replace("\\", "/").rstrip("/")
        if "/" in candidate:
            candidate = candidate.rsplit("/", 1)[-1]
    if not candidate:
        candidate = dataset_root.name
    return candidate or "dataset"


def resolve_diffusion_drop_n_last_frames(
    *,
    n_obs_steps: int,
    horizon: int,
    n_action_steps: int,
    drop_n_last_frames: int | None,
) -> int:
    if n_obs_steps <= 0:
        raise ValueError(f"`--n-obs-steps` must be positive, got {n_obs_steps}.")
    if horizon <= 0:
        raise ValueError(f"`--horizon` must be positive, got {horizon}.")
    if n_action_steps <= 0:
        raise ValueError(f"`--n-action-steps` must be positive, got {n_action_steps}.")

    max_n_action_steps = horizon - n_obs_steps + 1
    if max_n_action_steps <= 0:
        raise ValueError(
            "Diffusion policy requires `horizon >= n_obs_steps`. "
            f"Got horizon={horizon}, n_obs_steps={n_obs_steps}."
        )
    if n_action_steps > max_n_action_steps:
        raise ValueError(
            "Diffusion policy requires `n_action_steps <= horizon - n_obs_steps + 1`. "
            f"Got n_action_steps={n_action_steps}, horizon={horizon}, "
            f"n_obs_steps={n_obs_steps}, max={max_n_action_steps}."
        )

    max_drop_n_last_frames = max_n_action_steps - n_action_steps
    if drop_n_last_frames is None:
        return max_drop_n_last_frames

    resolved_drop_n_last_frames = int(drop_n_last_frames)
    if resolved_drop_n_last_frames < 0:
        raise ValueError(
            "`--drop-n-last-frames` must be non-negative when provided. "
            f"Got {resolved_drop_n_last_frames}."
        )
    if resolved_drop_n_last_frames > max_drop_n_last_frames:
        raise ValueError(
            "Diffusion policy `drop_n_last_frames` is too large for the requested "
            "observation/action horizon. "
            f"Got drop_n_last_frames={resolved_drop_n_last_frames}, "
            f"max={max_drop_n_last_frames} for horizon={horizon}, "
            f"n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}."
        )
    return resolved_drop_n_last_frames


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--dataset", type=str, default=None)
    bootstrap.add_argument(
        "--policy",
        choices=["act", "diffusion", "prism_diffusion", "streaming_act"],
        default="act",
    )
    known_args, _ = bootstrap.parse_known_args(argv)
    defaults, defaults_path = ({}, None)
    if known_args.dataset:
        defaults, defaults_path = load_policy_mode_defaults_for_dataset(
            mode="train",
            dataset_selector=known_args.dataset,
            policy_name=known_args.policy,
        )

    parser = argparse.ArgumentParser(
        description=(
            "Train LeRobot ACT, Diffusion, PRISM Diffusion, or Streaming ACT "
            "on a local LeRobot dataset."
        )
    )
    parser.add_argument(
        "--policy",
        choices=["act", "diffusion", "prism_diffusion", "streaming_act"],
        default=known_args.policy,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default=known_args.dataset,
        help=(
            "Dataset ID or path under data. This value is also used to resolve "
            "`bash/defaults/<dataset_key>/<policy>.yaml` when present. "
            "Examples: zeno-ai/day3_5_Exp1_processed, "
            "robocasa/composite/ArrangeBreadBasket, "
            "./data/zeno-ai/day3_5_Exp1."
        ),
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=defaults.get("dataset_repo_id"),
        help=(
            "Optional logical repo_id override used by LeRobot metadata APIs. "
            "Defaults to the dataset path relative to main/data."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default=defaults.get("task", defaults.get("cil")),
        help=(
            "Optional Meta-World task subset recorded in the training config. "
            "Use a comma-separated task list such as "
            "`assembly-v3,dial-turn-v3,handle-press-side-v3`."
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
        "--local-data-root",
        type=Path,
        default=defaults.get("local_data_root", DEFAULT_LOCAL_DATA_ROOT),
        help="Root directory used to resolve --dataset when a relative dataset ID is provided.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=defaults.get("test_ratio", 0.2),
        help=(
            "Held-out test-set ratio computed over episodes. "
            "Use 0 to place every episode in the training split."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=defaults.get("split_seed", 42),
        help="Random seed used when sampling train/test episodes.",
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
    parser.set_defaults(split_shuffle=defaults.get("split_shuffle", True))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=defaults.get(
            "output_root",
            resolve_default_train_output_root(
                policy_name=known_args.policy,
                dataset_selector=known_args.dataset,
            ),
        ),
        help="Root folder for training outputs.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help=(
            "Resume from the newest resumable checkpoint under --output-root. "
            "The latest run directory is reused instead of creating a new timestamped run."
        ),
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and start a fresh run.",
    )
    parser.set_defaults(resume=bool(defaults.get("resume", False)))
    parser.add_argument(
        "--job-name",
        type=str,
        default=defaults.get("job_name", default_policy_series_name(known_args.policy)),
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=defaults.get("wandb_run_name"),
        help=(
            "Optional explicit Weights & Biases run name. "
            "Defaults to '<job-name>-s<seed>-<timestamp>'."
        ),
    )
    parser.add_argument("--steps", type=int, default=defaults.get("steps", 10000))
    parser.add_argument(
        "--batch-size", type=int, default=defaults.get("batch_size", 32)
    )
    parser.add_argument(
        "--num-workers", type=int, default=defaults.get("num_workers", 4)
    )
    dataloader_persistent_workers_group = parser.add_mutually_exclusive_group()
    dataloader_persistent_workers_group.add_argument(
        "--enable-persistent-workers",
        dest="dataloader_persistent_workers",
        action="store_true",
        help=(
            "Keep DataLoader workers alive across epochs so video decoders and "
            "worker startup cost are reused."
        ),
    )
    dataloader_persistent_workers_group.add_argument(
        "--disable-persistent-workers",
        dest="dataloader_persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers.",
    )
    parser.set_defaults(
        dataloader_persistent_workers=defaults.get(
            "dataloader_persistent_workers", True
        ),
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=defaults.get("dataloader_prefetch_factor", 4),
        help=(
            "Number of prefetched batches kept per worker. Larger values help "
            "video-heavy pipelines hide decode latency."
        ),
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default=defaults.get("video_backend"),
        help=(
            "Optional LeRobot video decoder backend override. "
            "Examples: torchcodec, pyav, video_reader."
        ),
    )
    parser.add_argument(
        "--torch-sharing-strategy",
        choices=["auto", "file_system", "file_descriptor"],
        default=defaults.get("torch_sharing_strategy", "auto"),
        help=(
            "Torch multiprocessing sharing strategy used by DataLoader workers. "
            "`file_system` is more robust for large video datasets on Linux."
        ),
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
        help=(
            "In-training simulator eval is disabled in dataset-only mode. "
            "Use -1 (recommended) and run eval_policy.py on the held-out split."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.get("device", "cuda"),
        help="cuda / cpu / mps",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--enable-amp",
        dest="use_amp",
        action="store_true",
        help="Enable mixed-precision training through Accelerate.",
    )
    amp_group.add_argument(
        "--disable-amp",
        dest="use_amp",
        action="store_false",
        help="Disable mixed-precision training and keep fp32 updates.",
    )
    parser.set_defaults(use_amp=defaults.get("use_amp", False))
    parser.add_argument(
        "--amp-dtype",
        choices=["auto", "bf16", "fp16"],
        default=defaults.get("amp_dtype", "auto"),
        help=(
            "Preferred AMP dtype when --enable-amp is active. "
            "`auto` selects bf16 on supported CUDA GPUs and otherwise falls back to fp16."
        ),
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
    if known_args.policy in {"diffusion", "prism_diffusion"}:
        parser.add_argument(
            "--n-obs-steps",
            type=int,
            default=defaults.get("n_obs_steps", 2),
            help=(
                "Number of observation steps passed to the diffusion policy "
                "at each decision."
            ),
        )
        parser.add_argument(
            "--horizon",
            type=int,
            default=defaults.get("horizon", 16),
            help=(
                "Diffusion action prediction horizon. This should be an integer "
                "multiple of the U-Net downsampling factor."
            ),
        )
        parser.add_argument(
            "--drop-n-last-frames",
            type=int,
            default=defaults.get("drop_n_last_frames"),
            help=(
                "Number of tail frames skipped when sampling training windows for "
                "diffusion. If omitted, it is derived from horizon, n_obs_steps, "
                "and n_action_steps."
            ),
        )
    else:
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=defaults.get("chunk_size", 5),
            help="ACT-style action chunk size.",
        )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=defaults.get("wandb_project"),
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=defaults.get("wandb_entity"),
    )
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
                "Path-signature feature dim. " "Set 0 to auto-read from meta/info.json."
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
                "Enable fixed-budget historical memory tokens built from prefix "
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
                "Number of legacy GRU-style visual prefix memory slots. "
                "Ignored when --enable-signature-indexed-slot-memory is used."
            ),
        )
        signature_indexed_slot_memory_group = parser.add_mutually_exclusive_group()
        signature_indexed_slot_memory_group.add_argument(
            "--enable-signature-indexed-slot-memory",
            dest="use_signature_indexed_slot_memory",
            action="store_true",
            help=(
                "Replace the legacy GRU-style visual prefix memory updater with "
                "a Signature-Indexed Slot Memory (SISM / PRISM core) updater."
            ),
        )
        signature_indexed_slot_memory_group.add_argument(
            "--disable-signature-indexed-slot-memory",
            dest="use_signature_indexed_slot_memory",
            action="store_false",
            help="Disable the SISM updater and use the legacy GRU-style updater instead.",
        )
        parser.set_defaults(
            use_signature_indexed_slot_memory=defaults.get(
                "use_signature_indexed_slot_memory", False
            ),
        )
        parser.add_argument(
            "--slot-memory-num-slots",
            type=int,
            default=defaults.get("slot_memory_num_slots", 4),
            help=(
                "Number of SISM slots when --enable-signature-indexed-slot-memory "
                "is active."
            ),
        )
        parser.add_argument(
            "--slot-memory-routing-hidden-dim",
            type=int,
            default=defaults.get("slot_memory_routing_hidden_dim", 512),
            help="Hidden dimension used by the SISM routing network.",
        )
        slot_memory_delta_routing_group = parser.add_mutually_exclusive_group()
        slot_memory_delta_routing_group.add_argument(
            "--enable-slot-memory-delta-routing",
            dest="slot_memory_use_delta_routing",
            action="store_true",
            help="Include delta signatures in the explicit SISM routing signal.",
        )
        slot_memory_delta_routing_group.add_argument(
            "--disable-slot-memory-delta-routing",
            dest="slot_memory_use_delta_routing",
            action="store_false",
            help="Route SISM slots using path signatures only.",
        )
        parser.set_defaults(
            slot_memory_use_delta_routing=defaults.get(
                "slot_memory_use_delta_routing", False
            ),
        )
        slot_memory_softmax_group = parser.add_mutually_exclusive_group()
        slot_memory_softmax_group.add_argument(
            "--enable-slot-memory-softmax-routing",
            dest="slot_memory_use_softmax_routing",
            action="store_true",
            help="Use softmax routing weights across SISM slots.",
        )
        slot_memory_softmax_group.add_argument(
            "--disable-slot-memory-softmax-routing",
            dest="slot_memory_use_softmax_routing",
            action="store_false",
            help="Use independent sigmoid routing strengths for SISM slots.",
        )
        parser.set_defaults(
            slot_memory_use_softmax_routing=defaults.get(
                "slot_memory_use_softmax_routing", True
            ),
        )
        slot_memory_readout_group = parser.add_mutually_exclusive_group()
        slot_memory_readout_group.add_argument(
            "--enable-slot-memory-readout-pooling",
            dest="slot_memory_use_readout_pooling",
            action="store_true",
            help=(
                "Use an attention readout over SISM slots to produce the pooled "
                "memory context for encoder FiLM."
            ),
        )
        slot_memory_readout_group.add_argument(
            "--disable-slot-memory-readout-pooling",
            dest="slot_memory_use_readout_pooling",
            action="store_false",
            help="Use mean pooling over SISM slots for encoder FiLM context.",
        )
        parser.set_defaults(
            slot_memory_use_readout_pooling=defaults.get(
                "slot_memory_use_readout_pooling", True
            ),
        )
        parser.add_argument(
            "--slot-memory-balance-loss-coef",
            type=float,
            default=defaults.get("slot_memory_balance_loss_coef", 0.0),
            help="Optional routing-balance loss coefficient for SISM.",
        )
        parser.add_argument(
            "--slot-memory-consistency-loss-coef",
            type=float,
            default=defaults.get("slot_memory_consistency_loss_coef", 0.0),
            help="Optional readout/write consistency loss coefficient for SISM.",
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
        memory_conditioned_encoder_film_group = parser.add_mutually_exclusive_group()
        memory_conditioned_encoder_film_group.add_argument(
            "--enable-memory-conditioned-encoder-film",
            dest="use_memory_conditioned_encoder_film",
            action="store_true",
            help=(
                "FiLM-modulate the current-step encoder tokens using the pooled "
                "visual prefix memory context."
            ),
        )
        memory_conditioned_encoder_film_group.add_argument(
            "--disable-memory-conditioned-encoder-film",
            dest="use_memory_conditioned_encoder_film",
            action="store_false",
            help="Disable memory-conditioned encoder FiLM.",
        )
        parser.set_defaults(
            use_memory_conditioned_encoder_film=defaults.get(
                "use_memory_conditioned_encoder_film", False
            ),
        )
        parser.add_argument(
            "--signature-cache-mode",
            choices=["off", "memmap", "ram"],
            default=defaults.get("signature_cache_mode", "off"),
            help=(
                "Fast path for path/delta signature loading. `memmap` reads from a "
                "disk-backed contiguous cache, `ram` preloads that cache once into "
                "shared memory, and `off` falls back to parquet columns."
            ),
        )
        parser.add_argument(
            "--signature-cache-dtype",
            choices=["float16", "float32"],
            default=defaults.get("signature_cache_dtype", "float16"),
            help=(
                "Storage dtype used by the signature cache. The cache always keeps "
                "the same normalized semantics as runtime normalization."
            ),
        )
        parser.add_argument(
            "--refresh-signature-cache",
            action="store_true",
            default=bool(defaults.get("refresh_signature_cache", False)),
            help="Force a signature cache rebuild before training starts.",
        )
        parser.add_argument(
            "--signature-cache-root",
            type=Path,
            default=defaults.get("signature_cache_root"),
            help=(
                "Optional cache directory override. Defaults to a hidden cache "
                "folder under the dataset root."
            ),
        )
        parser.add_argument(
            "--prefix-image-cache-mode",
            choices=["off", "memmap", "ram"],
            default=defaults.get("prefix_image_cache_mode", "off"),
            help=(
                "Optional raw prefix-image cache. `memmap` reads contiguous cached "
                "frames from disk, `ram` preloads that cache once, and `off` falls "
                "back to the dataset video decode path."
            ),
        )
        parser.add_argument(
            "--prefix-image-cache-dtype",
            choices=["uint8", "float16", "float32"],
            default=defaults.get("prefix_image_cache_dtype", "uint8"),
            help=(
                "Storage dtype used by the optional raw prefix-image cache. "
                "`uint8` minimizes disk and RAM footprint."
            ),
        )
        parser.add_argument(
            "--refresh-prefix-image-cache",
            action="store_true",
            default=bool(defaults.get("refresh_prefix_image_cache", False)),
            help="Force a prefix-image cache rebuild before training starts.",
        )
        parser.add_argument(
            "--prefix-image-cache-root",
            type=Path,
            default=defaults.get("prefix_image_cache_root"),
            help=(
                "Optional prefix-image cache directory override. Defaults to a "
                "hidden cache folder under the dataset root."
            ),
        )
    elif known_args.policy == "prism_diffusion":
        path_signature_group = parser.add_mutually_exclusive_group()
        path_signature_group.add_argument(
            "--enable-path-signature",
            dest="use_path_signature",
            action="store_true",
            help=(
                "Record path-signature conditioning in the PRISM Diffusion "
                "checkpoint config."
            ),
        )
        path_signature_group.add_argument(
            "--disable-path-signature",
            dest="use_path_signature",
            action="store_false",
            help="Disable path-signature conditioning in PRISM Diffusion configs.",
        )
        parser.set_defaults(
            use_path_signature=defaults.get("use_path_signature", False),
        )
        parser.add_argument(
            "--history-length",
            type=int,
            default=defaults.get("history_length", 0),
            help=(
                "History window size recorded in PrismDiffusionConfig. "
                "Set 0 to defer inference to the local policy package."
            ),
        )
        parser.add_argument(
            "--signature-dim",
            type=int,
            default=defaults.get("signature_dim", 0),
            help=(
                "Serialized signature feature dim for PRISM Diffusion. "
                "Set 0 to defer inference to the local policy package."
            ),
        )
        parser.add_argument(
            "--signature-depth",
            type=int,
            default=defaults.get("signature_depth", 3),
            help="Serialized signature truncation depth for PRISM Diffusion.",
        )
        parser.add_argument(
            "--signature-hidden-dim",
            type=int,
            default=defaults.get("signature_hidden_dim", 512),
            help="Serialized signature projection hidden dim for PRISM Diffusion.",
        )
        parser.add_argument(
            "--signature-dropout",
            type=float,
            default=defaults.get("signature_dropout", 0.1),
            help="Serialized signature projection dropout for PRISM Diffusion.",
        )
        delta_signature_group = parser.add_mutually_exclusive_group()
        delta_signature_group.add_argument(
            "--enable-delta-signature",
            dest="use_delta_signature",
            action="store_true",
            help="Record delta-signature conditioning in PrismDiffusionConfig.",
        )
        delta_signature_group.add_argument(
            "--disable-delta-signature",
            dest="use_delta_signature",
            action="store_false",
            help="Disable delta-signature conditioning in PRISM Diffusion configs.",
        )
        parser.set_defaults(
            use_delta_signature=defaults.get("use_delta_signature", False),
        )
        prefix_group = parser.add_mutually_exclusive_group()
        prefix_group.add_argument(
            "--enable-prefix-sequence-training",
            dest="use_prefix_sequence_training",
            action="store_true",
            help="Record prefix-sequence training support in PrismDiffusionConfig.",
        )
        prefix_group.add_argument(
            "--disable-prefix-sequence-training",
            dest="use_prefix_sequence_training",
            action="store_false",
            help="Disable prefix-sequence training fields in PRISM Diffusion configs.",
        )
        parser.set_defaults(
            use_prefix_sequence_training=defaults.get(
                "use_prefix_sequence_training", False
            ),
        )
        parser.add_argument(
            "--prefix-train-max-steps",
            type=int,
            default=defaults.get("prefix_train_max_steps", 32),
            help="Serialized prefix length budget for PRISM Diffusion configs.",
        )
        parser.add_argument(
            "--prefix-frame-stride",
            type=int,
            default=defaults.get("prefix_frame_stride", 1),
            help="Serialized prefix subsampling stride for PRISM Diffusion configs.",
        )
        parser.add_argument(
            "--prefix-pad-value",
            type=float,
            default=defaults.get("prefix_pad_value", 0.0),
            help="Serialized prefix padding value for PRISM Diffusion configs.",
        )
        parser.add_argument(
            "--signature-cache-mode",
            choices=["off", "memmap", "ram"],
            default=defaults.get("signature_cache_mode", "off"),
            help=(
                "Fast path for path/delta signature loading in PRISM Diffusion. "
                "`memmap` reads from a disk-backed contiguous cache, `ram` preloads "
                "that cache once into shared memory, and `off` falls back to parquet "
                "columns."
            ),
        )
        parser.add_argument(
            "--signature-cache-dtype",
            choices=["float16", "float32"],
            default=defaults.get("signature_cache_dtype", "float16"),
            help=(
                "Storage dtype used by the PRISM Diffusion signature cache. The "
                "cache preserves runtime normalization semantics."
            ),
        )
        parser.add_argument(
            "--refresh-signature-cache",
            action="store_true",
            default=bool(defaults.get("refresh_signature_cache", False)),
            help="Force a signature cache rebuild before PRISM Diffusion training starts.",
        )
        parser.add_argument(
            "--signature-cache-root",
            type=Path,
            default=defaults.get("signature_cache_root"),
            help=(
                "Optional signature cache directory override. Defaults to a hidden "
                "cache folder under the dataset root."
            ),
        )
        parser.add_argument(
            "--prefix-image-cache-mode",
            choices=["off", "memmap", "ram"],
            default=defaults.get("prefix_image_cache_mode", "off"),
            help=(
                "Optional raw prefix-image cache for PRISM Diffusion. `memmap` "
                "reads contiguous cached frames from disk, `ram` preloads that "
                "cache once, and `off` falls back to the dataset image/video path."
            ),
        )
        parser.add_argument(
            "--prefix-image-cache-dtype",
            choices=["uint8", "float16", "float32"],
            default=defaults.get("prefix_image_cache_dtype", "uint8"),
            help=(
                "Storage dtype used by the optional PRISM Diffusion prefix-image "
                "cache. `uint8` minimizes disk and RAM footprint."
            ),
        )
        parser.add_argument(
            "--refresh-prefix-image-cache",
            action="store_true",
            default=bool(defaults.get("refresh_prefix_image_cache", False)),
            help="Force a prefix-image cache rebuild before PRISM Diffusion training starts.",
        )
        parser.add_argument(
            "--prefix-image-cache-root",
            type=Path,
            default=defaults.get("prefix_image_cache_root"),
            help=(
                "Optional prefix-image cache directory override. Defaults to a "
                "hidden cache folder under the dataset root."
            ),
        )
        visual_prefix_memory_group = parser.add_mutually_exclusive_group()
        visual_prefix_memory_group.add_argument(
            "--enable-visual-prefix-memory",
            dest="use_visual_prefix_memory",
            action="store_true",
            help="Record visual prefix memory support in PrismDiffusionConfig.",
        )
        visual_prefix_memory_group.add_argument(
            "--disable-visual-prefix-memory",
            dest="use_visual_prefix_memory",
            action="store_false",
            help="Disable visual prefix memory fields in PRISM Diffusion configs.",
        )
        parser.set_defaults(
            use_visual_prefix_memory=defaults.get("use_visual_prefix_memory", False),
        )
        signature_indexed_slot_memory_group = parser.add_mutually_exclusive_group()
        signature_indexed_slot_memory_group.add_argument(
            "--enable-signature-indexed-slot-memory",
            dest="use_signature_indexed_slot_memory",
            action="store_true",
            help="Record Signature-Indexed Slot Memory support in PrismDiffusionConfig.",
        )
        signature_indexed_slot_memory_group.add_argument(
            "--disable-signature-indexed-slot-memory",
            dest="use_signature_indexed_slot_memory",
            action="store_false",
            help="Disable Signature-Indexed Slot Memory fields in PRISM Diffusion configs.",
        )
        parser.set_defaults(
            use_signature_indexed_slot_memory=defaults.get(
                "use_signature_indexed_slot_memory", False
            ),
        )
        parser.add_argument(
            "--slot-memory-num-slots",
            type=int,
            default=defaults.get("slot_memory_num_slots", 4),
            help="Serialized number of PRISM slot-memory slots.",
        )
        parser.add_argument(
            "--slot-memory-routing-hidden-dim",
            type=int,
            default=defaults.get("slot_memory_routing_hidden_dim", 512),
            help="Serialized PRISM slot-routing hidden dim.",
        )
        slot_memory_delta_routing_group = parser.add_mutually_exclusive_group()
        slot_memory_delta_routing_group.add_argument(
            "--enable-slot-memory-delta-routing",
            dest="slot_memory_use_delta_routing",
            action="store_true",
            help="Record delta-signature slot routing in PrismDiffusionConfig.",
        )
        slot_memory_delta_routing_group.add_argument(
            "--disable-slot-memory-delta-routing",
            dest="slot_memory_use_delta_routing",
            action="store_false",
            help="Disable delta-signature slot routing in PRISM Diffusion configs.",
        )
        parser.set_defaults(
            slot_memory_use_delta_routing=defaults.get(
                "slot_memory_use_delta_routing", False
            ),
        )
        slot_memory_softmax_group = parser.add_mutually_exclusive_group()
        slot_memory_softmax_group.add_argument(
            "--enable-slot-memory-softmax-routing",
            dest="slot_memory_use_softmax_routing",
            action="store_true",
            help="Use softmax slot routing in serialized PRISM Diffusion configs.",
        )
        slot_memory_softmax_group.add_argument(
            "--disable-slot-memory-softmax-routing",
            dest="slot_memory_use_softmax_routing",
            action="store_false",
            help="Use sigmoid slot routing in serialized PRISM Diffusion configs.",
        )
        parser.set_defaults(
            slot_memory_use_softmax_routing=defaults.get(
                "slot_memory_use_softmax_routing", True
            ),
        )
        slot_memory_readout_group = parser.add_mutually_exclusive_group()
        slot_memory_readout_group.add_argument(
            "--enable-slot-memory-readout-pooling",
            dest="slot_memory_use_readout_pooling",
            action="store_true",
            help="Use readout pooling in serialized PRISM Diffusion configs.",
        )
        slot_memory_readout_group.add_argument(
            "--disable-slot-memory-readout-pooling",
            dest="slot_memory_use_readout_pooling",
            action="store_false",
            help="Disable readout pooling in serialized PRISM Diffusion configs.",
        )
        parser.set_defaults(
            slot_memory_use_readout_pooling=defaults.get(
                "slot_memory_use_readout_pooling", True
            ),
        )
        parser.add_argument(
            "--slot-memory-balance-loss-coef",
            type=float,
            default=defaults.get("slot_memory_balance_loss_coef", 0.0),
            help="Serialized PRISM slot-memory balance loss coefficient.",
        )
        parser.add_argument(
            "--slot-memory-consistency-loss-coef",
            type=float,
            default=defaults.get("slot_memory_consistency_loss_coef", 0.0),
            help="Serialized PRISM slot-memory consistency loss coefficient.",
        )
        parser.add_argument(
            "--prism-adapter-hidden-dim",
            type=int,
            default=defaults.get("prism_adapter_hidden_dim", 512),
            help="Serialized hidden dim for the PRISM adapter MLP.",
        )
        prism_adapter_zero_init_group = parser.add_mutually_exclusive_group()
        prism_adapter_zero_init_group.add_argument(
            "--enable-prism-adapter-zero-init",
            dest="prism_adapter_zero_init",
            action="store_true",
            help="Zero-initialize the serialized PRISM adapter parameters.",
        )
        prism_adapter_zero_init_group.add_argument(
            "--disable-prism-adapter-zero-init",
            dest="prism_adapter_zero_init",
            action="store_false",
            help="Disable zero-init for serialized PRISM adapter parameters.",
        )
        parser.set_defaults(
            prism_adapter_zero_init=defaults.get("prism_adapter_zero_init", True),
        )
    parser.set_defaults(
        _policy_defaults_path=(None if defaults_path is None else str(defaults_path)),
        _policy_defaults_dataset_root=defaults.get("dataset_root"),
        _policy_defaults_dataset_repo_id=defaults.get("dataset_repo_id"),
        _policy_defaults_dataset_tasks=defaults.get("dataset_tasks"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    if not isinstance(args.local_data_root, Path):
        args.local_data_root = Path(args.local_data_root)
    if not isinstance(args.output_root, Path):
        args.output_root = Path(args.output_root)
    if getattr(args, "signature_cache_root", None) is not None and not isinstance(
        args.signature_cache_root, Path
    ):
        args.signature_cache_root = Path(args.signature_cache_root)
    if getattr(args, "prefix_image_cache_root", None) is not None and not isinstance(
        args.prefix_image_cache_root, Path
    ):
        args.prefix_image_cache_root = Path(args.prefix_image_cache_root)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    ensure_writable_hf_cache_env(PROJECT_ROOT)
    if args.policy == "streaming_act":
        ensure_streaming_act_importable(PROJECT_ROOT)
    elif args.policy == "prism_diffusion":
        ensure_prism_diffusion_importable(PROJECT_ROOT)
        ensure_streaming_act_importable(PROJECT_ROOT)

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

    defaults_dataset_root = getattr(args, "_policy_defaults_dataset_root", None)
    source_dataset_root = resolve_training_dataset_root(
        dataset=args.dataset,
        defaults_dataset_root=defaults_dataset_root,
        local_data_root=args.local_data_root.resolve(),
        exact_task_names=getattr(args, "_policy_defaults_dataset_tasks", None),
    )
    dataset_repo_id = resolve_effective_dataset_repo_id(
        requested_repo_id=args.dataset_repo_id,
        default_repo_id=getattr(args, "_policy_defaults_dataset_repo_id", None),
        dataset_root=source_dataset_root,
        local_data_root=args.local_data_root.resolve(),
    )
    dataset_root = ensure_lerobot_dataset_v30_compat(
        source_dataset_root,
        dataset_repo_id=dataset_repo_id,
        local_data_root=args.local_data_root.resolve(),
    )
    validate_dataset_root(dataset_root)
    resume_run_state = (
        resolve_resume_run_state(args.output_root) if bool(args.resume) else None
    )
    split_spec = (
        load_resume_dataset_split(
            resume_run_state=resume_run_state,
            source_dataset_root=source_dataset_root,
            dataset_repo_id=dataset_repo_id,
        )
        if resume_run_state is not None
        else None
    )
    if split_spec is None:
        split_spec = build_dataset_split(
            dataset_arg=args.dataset,
            dataset_root=source_dataset_root,
            dataset_repo_id=dataset_repo_id,
            test_ratio=float(args.test_ratio),
            split_seed=int(args.split_seed),
            split_shuffle=bool(args.split_shuffle),
        )
    visual_storage_modes = summarize_visual_storage_modes(dataset_root)

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    PrismDiffusionConfig = None
    if policy_supports_signature_features(args.policy):
        from lerobot_policy_streaming_act.prefix_image_cache import (
            PrefixImageCacheRuntimeConfig,
            configure_prefix_image_cache_runtime,
            prepare_prefix_image_cache_runtime,
        )
        from lerobot_policy_streaming_act.signature_cache import (
            SignatureCacheRuntimeConfig,
            configure_signature_cache_runtime,
            prepare_signature_cache_runtime,
        )

    if args.policy == "streaming_act":
        from lerobot_policy_streaming_act.configuration_streaming_act import (
            DELTA_SIGNATURE_KEY,
            PATH_SIGNATURE_KEY,
            StreamingACTConfig,
        )
    elif args.policy == "prism_diffusion":
        from lerobot_policy_prism_diffusion.configuration_diffusion import (
            PrismDiffusionConfig,
        )
    else:
        configure_prefix_image_cache_runtime = None
        prepare_prefix_image_cache_runtime = None
        configure_signature_cache_runtime = None
        prepare_signature_cache_runtime = None
    install_torch_dataloader_patch(
        persistent_workers=bool(args.dataloader_persistent_workers),
        prefetch_factor=int(args.dataloader_prefetch_factor),
    )
    install_lerobot_dataset_load_patch()
    install_episode_aware_sampler_patch()

    prism_use_path_signature = False
    prism_use_delta_signature = False
    prism_use_prefix_sequence_training = False
    prism_use_visual_prefix_memory = False
    prism_history_length = 0
    prism_signature_dim = 0
    prism_required_signature_parquet_keys: list[str] = []
    required_signature_parquet_keys: list[str] = []

    if args.policy == "streaming_act":
        use_path_signature = args.use_path_signature
        use_delta_signature = bool(args.use_delta_signature)
        use_prefix_sequence_training = bool(args.use_prefix_sequence_training)
        use_visual_prefix_memory = bool(args.use_visual_prefix_memory)
        use_imagenet_stats = False
        required_signature_parquet_keys = [
            key
            for key, enabled in (
                ("observation.path_signature", bool(use_path_signature)),
                ("observation.delta_signature", bool(use_delta_signature)),
            )
            if enabled
        ]
        validate_prefix_sequence_support(
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
            dataset_repo_id=dataset_repo_id,
            signature_cache_root=args.signature_cache_root,
            use_path_signature=use_path_signature,
            signature_dim=args.signature_dim,
        )
        validate_delta_signature_dataset(
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            signature_cache_root=args.signature_cache_root,
            use_delta_signature=use_delta_signature,
        )
        validate_prefix_sequence_dataset(
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            signature_cache_root=args.signature_cache_root,
            use_prefix_sequence_training=use_prefix_sequence_training,
            use_imagenet_stats=use_imagenet_stats,
            use_path_signature=use_path_signature,
            use_delta_signature=use_delta_signature,
        )
        validate_visual_prefix_memory_support(
            use_visual_prefix_memory=use_visual_prefix_memory,
            use_prefix_sequence_training=use_prefix_sequence_training,
        )
        if (
            required_signature_parquet_keys
            and str(args.signature_cache_mode) == "off"
            and not parquet_has_columns(dataset_root, required_signature_parquet_keys)
        ):
            raise ValueError(
                "This dataset does not store the requested signature features as parquet columns. "
                "Enable `--signature-cache-mode memmap` or `--signature-cache-mode ram` "
                "so the training loader materializes signatures from the dataset cache."
            )
    elif args.policy == "prism_diffusion":
        use_path_signature = False
        use_delta_signature = False
        use_prefix_sequence_training = False
        use_visual_prefix_memory = False
        resolved_history_length = 0
        signature_dim = 0

        prism_use_path_signature = bool(args.use_path_signature)
        prism_use_delta_signature = bool(args.use_delta_signature)
        prism_use_prefix_sequence_training = bool(args.use_prefix_sequence_training)
        prism_use_visual_prefix_memory = bool(args.use_visual_prefix_memory)
        prism_required_signature_parquet_keys = [
            key
            for key, enabled in (
                ("observation.path_signature", bool(prism_use_path_signature)),
                ("observation.delta_signature", bool(prism_use_delta_signature)),
            )
            if enabled
        ]
        validate_prefix_sequence_support(
            policy_name=args.policy,
            use_prefix_sequence_training=prism_use_prefix_sequence_training,
            context="training",
        )
        prism_history_length = (
            resolve_history_length(
                dataset_root=dataset_root,
                history_length=args.history_length,
            )
            if prism_use_path_signature
            else int(args.history_length)
        )
        prism_signature_dim = resolve_signature_dim(
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            signature_cache_root=args.signature_cache_root,
            use_path_signature=prism_use_path_signature,
            signature_dim=args.signature_dim,
        )
        validate_delta_signature_dataset(
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            signature_cache_root=args.signature_cache_root,
            use_delta_signature=prism_use_delta_signature,
        )
        validate_prefix_sequence_dataset(
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            signature_cache_root=args.signature_cache_root,
            use_prefix_sequence_training=prism_use_prefix_sequence_training,
            use_imagenet_stats=False,
            use_path_signature=prism_use_path_signature,
            use_delta_signature=prism_use_delta_signature,
        )
        if (
            prism_required_signature_parquet_keys
            and str(args.signature_cache_mode) == "off"
            and not parquet_has_columns(dataset_root, prism_required_signature_parquet_keys)
        ):
            raise ValueError(
                "This dataset does not store the requested signature features as parquet columns. "
                "Enable `--signature-cache-mode memmap` or `--signature-cache-mode ram` "
                "so the training loader materializes signatures from the dataset cache."
            )
    else:
        use_path_signature = False
        use_delta_signature = False
        use_prefix_sequence_training = False
        use_visual_prefix_memory = False
        resolved_history_length = 0
        signature_dim = 0

    active_use_prefix_sequence_training = bool(
        use_prefix_sequence_training
        if args.policy == "streaming_act"
        else prism_use_prefix_sequence_training
    )
    active_use_path_signature = bool(
        use_path_signature if args.policy == "streaming_act" else prism_use_path_signature
    )
    active_use_delta_signature = bool(
        use_delta_signature if args.policy == "streaming_act" else prism_use_delta_signature
    )

    resolved_diffusion_drop_n_last_frames = None
    if args.policy in {"diffusion", "prism_diffusion"}:
        resolved_diffusion_drop_n_last_frames = resolve_diffusion_drop_n_last_frames(
            n_obs_steps=int(args.n_obs_steps),
            horizon=int(args.horizon),
            n_action_steps=int(args.n_action_steps),
            drop_n_last_frames=args.drop_n_last_frames,
        )

    input_features_override = None
    output_features_override = None
    if active_use_prefix_sequence_training:
        install_prefix_sequence_dataset_patch()
        input_features_override, output_features_override = build_policy_feature_overrides(
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            signature_cache_root=args.signature_cache_root,
            use_prefix_sequence_training=active_use_prefix_sequence_training,
            prefix_train_max_steps=int(args.prefix_train_max_steps),
            use_path_signature=active_use_path_signature,
            use_delta_signature=active_use_delta_signature,
        )

    dataset_cfg = DatasetConfig(
        repo_id=dataset_repo_id,
        root=str(dataset_root),
        episodes=split_spec.train_episode_indices,
        use_imagenet_stats=False,
        video_backend=args.video_backend,
    )

    if args.policy == "streaming_act":
        policy_cfg = StreamingACTConfig(
            device=args.device,
            use_amp=bool(args.use_amp),
            push_to_hub=False,
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            use_path_signature=bool(use_path_signature),
            use_delta_signature=bool(use_delta_signature),
            history_length=int(resolved_history_length),
            signature_dim=int(signature_dim),
            signature_depth=int(args.signature_depth),
            signature_hidden_dim=int(args.signature_hidden_dim),
            signature_dropout=float(args.signature_dropout),
            use_prefix_sequence_training=bool(use_prefix_sequence_training),
            prefix_train_max_steps=(
                int(args.prefix_train_max_steps) if use_prefix_sequence_training else 32
            ),
            prefix_frame_stride=(
                int(args.prefix_frame_stride) if use_prefix_sequence_training else 1
            ),
            prefix_pad_value=(
                float(args.prefix_pad_value) if use_prefix_sequence_training else 0.0
            ),
            use_visual_prefix_memory=bool(use_visual_prefix_memory),
            use_signature_conditioned_visual_prefix_memory=bool(
                args.use_signature_conditioned_visual_prefix_memory
            ),
            use_signature_indexed_slot_memory=bool(
                args.use_signature_indexed_slot_memory
            ),
            use_memory_conditioned_encoder_film=bool(
                args.use_memory_conditioned_encoder_film
            ),
            num_memory_slots=int(args.num_memory_slots),
            slot_memory_num_slots=int(args.slot_memory_num_slots),
            slot_memory_routing_hidden_dim=int(args.slot_memory_routing_hidden_dim),
            slot_memory_use_delta_routing=bool(args.slot_memory_use_delta_routing),
            slot_memory_use_softmax_routing=bool(args.slot_memory_use_softmax_routing),
            slot_memory_use_readout_pooling=bool(args.slot_memory_use_readout_pooling),
            slot_memory_balance_loss_coef=float(args.slot_memory_balance_loss_coef),
            slot_memory_consistency_loss_coef=float(
                args.slot_memory_consistency_loss_coef
            ),
            input_features=input_features_override,
            output_features=output_features_override,
        )
        active_signature_keys = tuple(
            key
            for key, enabled in (
                (PATH_SIGNATURE_KEY, bool(use_path_signature)),
                (DELTA_SIGNATURE_KEY, bool(use_delta_signature)),
            )
            if enabled
        )
        configure_signature_cache_runtime(
            SignatureCacheRuntimeConfig(
                dataset_root=dataset_root,
                dataset_repo_id=dataset_repo_id,
                mode=str(args.signature_cache_mode),
                cache_dtype=str(args.signature_cache_dtype),
                feature_keys=active_signature_keys,
                normalization_mode=policy_cfg.normalization_mapping.get(
                    "STATE", "mean_std"
                ),
                refresh=bool(args.refresh_signature_cache),
                cache_root=args.signature_cache_root,
            )
        )
        signature_cache_reader = prepare_signature_cache_runtime()
        if use_prefix_sequence_training:
            configure_prefix_image_cache_runtime(
                PrefixImageCacheRuntimeConfig(
                    dataset_root=dataset_root,
                    dataset_repo_id=dataset_repo_id,
                    mode=str(args.prefix_image_cache_mode),
                    cache_dtype=str(args.prefix_image_cache_dtype),
                    refresh=bool(args.refresh_prefix_image_cache),
                    cache_root=args.prefix_image_cache_root,
                )
            )
            prepare_prefix_image_cache_runtime()
        elif configure_prefix_image_cache_runtime is not None:
            configure_prefix_image_cache_runtime(None)
        if signature_cache_reader is None and required_signature_parquet_keys and not parquet_has_columns(
            dataset_root,
            required_signature_parquet_keys,
        ):
            raise RuntimeError(
                "Signature cache is required for this dataset because the parquet files "
                "do not contain the requested signature columns, but the cache could not be prepared."
            )
        policy_cfg.pre_normalized_observation_keys = (
            resolve_pre_normalized_signature_observation_keys(
                feature_keys=tuple(signature_cache_reader.feature_keys)
                if signature_cache_reader is not None
                else (),
                reader_pre_normalized=bool(
                    signature_cache_reader is not None
                    and signature_cache_reader.pre_normalized
                ),
                use_prefix_sequence_training=bool(use_prefix_sequence_training),
                use_path_signature=bool(use_path_signature),
                use_delta_signature=bool(use_delta_signature),
            )
        )
    elif args.policy in {"diffusion", "prism_diffusion"}:
        diffusion_config_cls = (
            PrismDiffusionConfig if args.policy == "prism_diffusion" else DiffusionConfig
        )
        prism_config_kwargs = {}
        if args.policy == "prism_diffusion":
            prism_config_kwargs = {
                "use_path_signature": prism_use_path_signature,
                "use_delta_signature": prism_use_delta_signature,
                "history_length": prism_history_length,
                "signature_dim": prism_signature_dim,
                "signature_depth": int(args.signature_depth),
                "signature_hidden_dim": int(args.signature_hidden_dim),
                "signature_dropout": float(args.signature_dropout),
                "use_prefix_sequence_training": prism_use_prefix_sequence_training,
                "prefix_train_max_steps": int(args.prefix_train_max_steps),
                "prefix_frame_stride": int(args.prefix_frame_stride),
                "prefix_pad_value": float(args.prefix_pad_value),
                "use_visual_prefix_memory": prism_use_visual_prefix_memory,
                "use_signature_indexed_slot_memory": bool(
                    args.use_signature_indexed_slot_memory
                ),
                "slot_memory_num_slots": int(args.slot_memory_num_slots),
                "slot_memory_routing_hidden_dim": int(
                    args.slot_memory_routing_hidden_dim
                ),
                "slot_memory_use_delta_routing": bool(
                    args.slot_memory_use_delta_routing
                ),
                "slot_memory_use_softmax_routing": bool(
                    args.slot_memory_use_softmax_routing
                ),
                "slot_memory_use_readout_pooling": bool(
                    args.slot_memory_use_readout_pooling
                ),
                "slot_memory_balance_loss_coef": float(
                    args.slot_memory_balance_loss_coef
                ),
                "slot_memory_consistency_loss_coef": float(
                    args.slot_memory_consistency_loss_coef
                ),
                "prism_adapter_hidden_dim": int(args.prism_adapter_hidden_dim),
                "prism_adapter_zero_init": bool(args.prism_adapter_zero_init),
            }
        policy_cfg = diffusion_config_cls(
            device=args.device,
            use_amp=bool(args.use_amp),
            push_to_hub=False,
            n_obs_steps=int(args.n_obs_steps),
            horizon=int(args.horizon),
            n_action_steps=int(args.n_action_steps),
            drop_n_last_frames=int(resolved_diffusion_drop_n_last_frames),
            input_features=input_features_override,
            output_features=output_features_override,
            **prism_config_kwargs,
        )
        if args.policy == "prism_diffusion":
            active_signature_keys = tuple(
                key
                for key, enabled in (
                    ("observation.path_signature", bool(prism_use_path_signature)),
                    ("observation.delta_signature", bool(prism_use_delta_signature)),
                )
                if enabled
            )
            configure_signature_cache_runtime(
                SignatureCacheRuntimeConfig(
                    dataset_root=dataset_root,
                    dataset_repo_id=dataset_repo_id,
                    mode=str(args.signature_cache_mode),
                    cache_dtype=str(args.signature_cache_dtype),
                    feature_keys=active_signature_keys,
                    normalization_mode=policy_cfg.normalization_mapping.get(
                        "STATE", "mean_std"
                    ),
                    refresh=bool(args.refresh_signature_cache),
                    cache_root=args.signature_cache_root,
                )
            )
            signature_cache_reader = prepare_signature_cache_runtime()
            if prism_use_prefix_sequence_training:
                configure_prefix_image_cache_runtime(
                    PrefixImageCacheRuntimeConfig(
                        dataset_root=dataset_root,
                        dataset_repo_id=dataset_repo_id,
                        mode=str(args.prefix_image_cache_mode),
                        cache_dtype=str(args.prefix_image_cache_dtype),
                        refresh=bool(args.refresh_prefix_image_cache),
                        cache_root=args.prefix_image_cache_root,
                    )
                )
                prepare_prefix_image_cache_runtime()
            elif configure_prefix_image_cache_runtime is not None:
                configure_prefix_image_cache_runtime(None)
            if (
                signature_cache_reader is None
                and prism_required_signature_parquet_keys
                and not parquet_has_columns(
                    dataset_root,
                    prism_required_signature_parquet_keys,
                )
            ):
                raise RuntimeError(
                    "Signature cache is required for this dataset because the parquet "
                    "files do not contain the requested signature columns, but the "
                    "cache could not be prepared."
                )
            policy_cfg.pre_normalized_observation_keys = (
                resolve_pre_normalized_signature_observation_keys(
                    feature_keys=tuple(signature_cache_reader.feature_keys)
                    if signature_cache_reader is not None
                    else (),
                    reader_pre_normalized=bool(
                        signature_cache_reader is not None
                        and signature_cache_reader.pre_normalized
                    ),
                    use_prefix_sequence_training=bool(
                        prism_use_prefix_sequence_training
                    ),
                    use_path_signature=bool(prism_use_path_signature),
                    use_delta_signature=bool(prism_use_delta_signature),
                )
            )
    else:
        if configure_signature_cache_runtime is not None:
            configure_signature_cache_runtime(None)
        if configure_prefix_image_cache_runtime is not None:
            configure_prefix_image_cache_runtime(None)
        policy_cfg = ACTConfig(
            device=args.device,
            use_amp=bool(args.use_amp),
            push_to_hub=False,
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
        )

    resume_saved_train_cfg = None
    resume_optimizer_cfg = None
    resume_scheduler_cfg = None
    if resume_run_state is not None:
        policy_cfg.pretrained_path = resume_run_state.pretrained_model_dir
        try:
            resume_saved_train_cfg = TrainPipelineConfig.from_pretrained(
                resume_run_state.train_config_path,
                local_files_only=True,
            )
        except Exception as exc:
            print(
                "[WARN] Failed to load saved train config from "
                f"{resume_run_state.train_config_path}: {exc}"
            )
        else:
            saved_policy_type = getattr(resume_saved_train_cfg.policy, "type", None)
            if saved_policy_type and str(saved_policy_type) != str(args.policy):
                raise ValueError(
                    "The latest resumable checkpoint was created with a different "
                    f"policy type: saved={saved_policy_type!r}, requested={args.policy!r}."
                )
            resume_optimizer_cfg = resume_saved_train_cfg.optimizer
            resume_scheduler_cfg = resume_saved_train_cfg.scheduler

        if resume_optimizer_cfg is None:
            resume_optimizer_cfg = policy_cfg.get_optimizer_preset()
        if resume_scheduler_cfg is None:
            resume_scheduler_cfg = policy_cfg.get_scheduler_preset()

    distributed_world_size = resolve_distributed_world_size_from_env()
    run_stamp = resolve_train_run_stamp()
    output_dir = (
        resume_run_state.run_dir.resolve()
        if resume_run_state is not None
        else (args.output_root / run_stamp).resolve()
    )
    split_path = output_dir / DATASET_SPLIT_FILENAME

    wandb_enable = args.enable_wandb and (args.wandb_mode != "disabled")
    resolved_wandb_project = (
        str(args.wandb_project)
        if args.wandb_project
        else default_wandb_project_name(
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
        )
    )
    resolved_job_name = (
        str(resume_saved_train_cfg.job_name)
        if (
            resume_saved_train_cfg is not None
            and resume_saved_train_cfg.job_name is not None
        )
        else str(args.job_name)
    )
    if wandb_enable:
        if args.wandb_run_name:
            resolved_job_name = str(args.wandb_run_name)
        elif resume_run_state is None:
            resolved_job_name = f"{args.job_name}-s{args.seed}-{run_stamp}"
        elif (
            resume_saved_train_cfg is not None
            and resume_saved_train_cfg.job_name is not None
        ):
            resolved_job_name = str(resume_saved_train_cfg.job_name)
        else:
            resolved_job_name = output_dir.name

    if (
        wandb_enable
        and args.wandb_mode == "online"
        and "WANDB_API_KEY" not in os.environ
    ):
        print(
            "[WARN] WANDB_API_KEY not found in environment. "
            "If you are not already logged in, run `wandb login` first."
        )

    env_cfg = None
    if args.task:
        from lerobot.envs.configs import MetaworldEnv as MetaworldEnvConfig

        env_cfg = MetaworldEnvConfig(task=str(args.task))

    wandb_cfg = WandBConfig(
        enable=wandb_enable,
        project=resolved_wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
    )
    resolved_mixed_precision = resolve_accelerator_mixed_precision(
        device=str(policy_cfg.device),
        use_amp=bool(getattr(policy_cfg, "use_amp", False)),
        amp_dtype=str(args.amp_dtype),
    )

    effective_eval_freq = -1
    if int(args.eval_freq) > 0:
        print(
            "[WARN] `--eval-freq` is ignored in dataset-only mode because no simulator "
            "environment is configured. Use `scripts/eval_policy.py` on the saved "
            "held-out test split instead."
        )

    cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        env=env_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=resolved_job_name,
        resume=bool(resume_run_state is not None),
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=effective_eval_freq,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        optimizer=resume_optimizer_cfg,
        scheduler=resume_scheduler_cfg,
        wandb=wandb_cfg,
    )

    print("Starting LeRobot imitation training with config:")
    if getattr(args, "_policy_defaults_path", None):
        print(f"- defaults_file: {args._policy_defaults_path}")
    print(f"- policy: {args.policy}")
    print(f"- dataset: {args.dataset}")
    if dataset_root == source_dataset_root:
        print(f"- dataset_root: {dataset_root}")
    else:
        print(f"- dataset_root_source: {source_dataset_root}")
        print(f"- dataset_root_runtime: {dataset_root}")
    print(f"- dataset_repo_id: {dataset_repo_id}")
    normalized_dataset_tasks = _normalize_dataset_task_names(
        getattr(args, "_policy_defaults_dataset_tasks", None)
    )
    if normalized_dataset_tasks:
        print(f"- dataset_tasks: {list(normalized_dataset_tasks)}")
    if args.task:
        print(f"- task: {args.task}")
    print(f"- resume: {bool(resume_run_state is not None)}")
    if resume_run_state is not None:
        print(f"- resume_run_dir: {resume_run_state.run_dir}")
        print(f"- resume_checkpoint_dir: {resume_run_state.checkpoint_dir}")
        print(f"- resume_train_config_path: {resume_run_state.train_config_path}")
        print(
            "- train_split_source: "
            f"{resume_run_state.split_path or 'recomputed from current dataset/split args'}"
        )
    print(
        f"- train_test_split: train={split_spec.train_count}, "
        f"test={split_spec.test_count}, "
        f"test_ratio={split_spec.test_ratio:.3f}, "
        f"shuffle={split_spec.split_shuffle}, seed={split_spec.split_seed}"
    )
    print(f"- train_episode_indices_path: {split_path}")
    print(f"- output_dir: {output_dir}")
    print(f"- device: {args.device}")
    print(
        "- amp: "
        f"enable={bool(args.use_amp)}, dtype={args.amp_dtype}, "
        f"mixed_precision={resolved_mixed_precision}"
    )
    print(f"- job_name: {resolved_job_name}")
    print(f"- steps: {args.steps}")
    print(f"- batch_size: {args.batch_size}")
    print(f"- num_workers: {args.num_workers}")
    print(
        "- dataloader: "
        f"persistent_workers={bool(args.dataloader_persistent_workers)}, "
        f"prefetch_factor={int(args.dataloader_prefetch_factor)}"
    )
    print(
        "- visual_storage: "
        f"images={int(visual_storage_modes.get('image', 0))}, "
        f"videos={int(visual_storage_modes.get('video', 0))}"
    )
    if int(visual_storage_modes.get("video", 0)) > 0:
        print(f"- video_backend: {args.video_backend or 'lerobot-default'}")
    else:
        print(
            "- video_backend: ignored "
            "(dataset stores visual observations directly in parquet image columns)"
        )
    if args.policy in {"diffusion", "prism_diffusion"}:
        print(
            "- action_execution: "
            f"n_obs_steps={int(args.n_obs_steps)}, "
            f"horizon={int(args.horizon)}, "
            f"n_action_steps={int(args.n_action_steps)}, "
            "drop_n_last_frames="
            f"{int(resolved_diffusion_drop_n_last_frames)}"
        )
    else:
        print(
            f"- action_execution: chunk_size={args.chunk_size}, "
            f"n_action_steps={args.n_action_steps}"
        )
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
                f"num_memory_slots={int(args.num_memory_slots)}, "
                "signature_indexed_slot_memory="
                f"{bool(args.use_signature_indexed_slot_memory)}, "
                "signature_conditioned="
                f"{bool(args.use_signature_conditioned_visual_prefix_memory)}, "
                "encoder_film="
                f"{bool(args.use_memory_conditioned_encoder_film)}"
            )
            if bool(args.use_signature_indexed_slot_memory):
                print(
                    "- slot_memory: "
                    f"num_slots={int(args.slot_memory_num_slots)}, "
                    f"routing_hidden={int(args.slot_memory_routing_hidden_dim)}, "
                    "delta_routing="
                    f"{bool(args.slot_memory_use_delta_routing)}, "
                    "softmax_routing="
                    f"{bool(args.slot_memory_use_softmax_routing)}, "
                    "readout_pooling="
                    f"{bool(args.slot_memory_use_readout_pooling)}, "
                    "balance_loss_coef="
                    f"{float(args.slot_memory_balance_loss_coef)}, "
                    "consistency_loss_coef="
                    f"{float(args.slot_memory_consistency_loss_coef)}"
                )
        print(
            "- signature_cache: "
            f"mode={args.signature_cache_mode}, dtype={args.signature_cache_dtype}, "
            f"root={args.signature_cache_root or (dataset_root / '.signature_cache')}, "
            f"refresh={bool(args.refresh_signature_cache)}"
        )
        print(
            "- prefix_image_cache: "
            f"mode={args.prefix_image_cache_mode}, dtype={args.prefix_image_cache_dtype}, "
            f"root={args.prefix_image_cache_root or (dataset_root / '.prefix_image_cache')}, "
            f"refresh={bool(args.refresh_prefix_image_cache)}"
        )
        print(
            "- signature_runtime_normalization: "
            f"skip_keys={list(getattr(policy_cfg, 'pre_normalized_observation_keys', ()))}"
        )
    elif args.policy in {"diffusion", "prism_diffusion"}:
        print(
            "- diffusion: "
            f"n_obs_steps={int(args.n_obs_steps)}, "
            f"horizon={int(args.horizon)}, "
            "drop_n_last_frames="
            f"{int(resolved_diffusion_drop_n_last_frames)}"
        )
        if args.policy == "prism_diffusion":
            print(f"- use_path_signature: {prism_use_path_signature}")
            if prism_use_path_signature:
                print(
                    f"- signature: dim={prism_signature_dim}, "
                    f"depth={int(args.signature_depth)}, "
                    f"history={prism_history_length}, "
                    f"hidden={int(args.signature_hidden_dim)}, "
                    f"dropout={float(args.signature_dropout)}"
                )
            print(f"- use_delta_signature: {prism_use_delta_signature}")
            print(
                "- prefix_sequence: "
                f"enabled={prism_use_prefix_sequence_training}, "
                f"max_steps={int(args.prefix_train_max_steps)}, "
                f"stride={int(args.prefix_frame_stride)}, "
                f"pad_value={float(args.prefix_pad_value)}"
            )
            print(f"- use_visual_prefix_memory: {prism_use_visual_prefix_memory}")
            print(
                "- slot_memory: "
                "enabled="
                f"{bool(args.use_signature_indexed_slot_memory)}, "
                f"num_slots={int(args.slot_memory_num_slots)}, "
                f"routing_hidden={int(args.slot_memory_routing_hidden_dim)}, "
                "delta_routing="
                f"{bool(args.slot_memory_use_delta_routing)}, "
                "softmax_routing="
                f"{bool(args.slot_memory_use_softmax_routing)}, "
                "readout_pooling="
                f"{bool(args.slot_memory_use_readout_pooling)}, "
                "balance_loss_coef="
                f"{float(args.slot_memory_balance_loss_coef)}, "
                "consistency_loss_coef="
                f"{float(args.slot_memory_consistency_loss_coef)}"
            )
            print(
                "- prism_adapter: "
                f"hidden_dim={int(args.prism_adapter_hidden_dim)}, "
                f"zero_init={bool(args.prism_adapter_zero_init)}"
            )
    print(
        f"- wandb: enable={wandb_enable}, project={resolved_wandb_project}, mode={args.wandb_mode}"
    )

    resolved_torch_sharing_strategy = configure_torch_sharing_strategy(
        args.torch_sharing_strategy
    )
    print(
        "- torch_sharing_strategy: " f"{resolved_torch_sharing_strategy or 'default'}"
    )

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    force_cpu = str(policy_cfg.device).split(":", 1)[0] == "cpu"
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        cpu=force_cpu,
        mixed_precision=resolved_mixed_precision,
    )
    accelerator_world_size = max(
        int(getattr(accelerator, "num_processes", distributed_world_size)),
        1,
    )
    if accelerator.is_main_process:
        print(
            "- distributed: "
            f"world_size={accelerator_world_size}, "
            f"per_device_batch_size={int(args.batch_size)}, "
            f"global_batch_size={int(args.batch_size) * accelerator_world_size}"
        )

    resume_config_arg = None
    if resume_run_state is not None and not any(
        str(arg).startswith("--config_path=") for arg in sys.argv[1:]
    ):
        resume_config_arg = f"--config_path={resume_run_state.train_config_path}"
        sys.argv.append(resume_config_arg)

    try:
        train(cfg, accelerator=accelerator)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            saved_split_path = save_dataset_split(output_dir, split_spec)
            print(f"Saved dataset split: {saved_split_path}")
        accelerator.wait_for_everyone()
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user. Cleaning up wandb before exit.")
        teardown_wandb_safely(exit_code=130)
        raise SystemExit(130)
    finally:
        if resume_config_arg is not None and sys.argv[-1] == resume_config_arg:
            sys.argv.pop()


if __name__ == "__main__":
    main()
