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
from pathlib import Path

import numpy as np
import warnings

from dataset_utils import (
    DEFAULT_LOCAL_DATA_ROOT,
    build_dataset_split,
    ensure_lerobot_dataset_v30_compat,
    infer_dataset_repo_id,
    resolve_dataset_root,
    save_dataset_split,
    validate_dataset_root,
)
from policy_defaults import load_policy_mode_defaults_for_dataset

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated*",
    category=UserWarning,
)

FIRST_FRAME_ANCHOR_KEY = "observation.anchor_image"
RAW_IMAGE_ARRAY_STORAGE_ENCODING = "raw_uint8_array"
RAW_IMAGE_ARRAY_STORAGE_DTYPE = "uint8"
SIGNATURE_CACHE_LAYOUT_VERSION = 1


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


def ensure_streaming_act_importable(repo_root: Path) -> None:
    streaming_act_src = repo_root / "main/policy/lerobot_policy_streaming_act/src"
    if not streaming_act_src.exists():
        raise FileNotFoundError(
            f"Streaming ACT package source not found: {streaming_act_src}"
        )
    sys.path.insert(0, str(streaming_act_src))


def teardown_wandb_safely(exit_code: int) -> None:
    try:
        import wandb
    except Exception:
        return

    try:
        wandb.teardown(exit_code=exit_code)
    except BaseException as exc:
        print(f"[WARN] wandb teardown failed during shutdown: {exc}")


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
        cached_feature_keys = set(signature_cache_feature_keys_for_dataset(self.root))
        load_feature_specs = {
            key: feature_spec
            for key, feature_spec in self.features.items()
            if key not in cached_feature_keys
        }
        if cached_feature_keys:
            print(
                "[INFO] signature_cache: excluding cached parquet columns from HF load: "
                f"{sorted(cached_feature_keys)}"
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
    LeRobotDataset._custom_dataset_load_patch_installed = True


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


def validate_first_frame_anchor_dataset(
    dataset_root: Path, use_first_frame_anchor: bool
) -> None:
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
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root
        / "main"
        / "outputs"
        / "train"
        / default_policy_series_name(policy_name)
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


def resolve_default_train_output_root(
    *,
    policy_name: str,
    dataset_selector: str | None,
) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root / "main" / "outputs" / "train"
    dataset_subdir = default_dataset_output_subdir(dataset_selector)
    if dataset_subdir is not None:
        return base / dataset_subdir / default_policy_series_name(policy_name)
    return default_train_output_root(policy_name)


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
        choices=["act", "diffusion", "streaming_act"],
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
            "Train LeRobot ACT, Diffusion, or Streaming ACT on a local "
            "LeRobot dataset."
        )
    )
    parser.add_argument(
        "--policy",
        choices=["act", "diffusion", "streaming_act"],
        default=known_args.policy,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default=known_args.dataset,
        help=(
            "Dataset ID or path under main/data. This value is also used to resolve "
            "`bash/defaults/<dataset_key>/<policy>.yaml` when present. "
            "Examples: zeno-ai/day3_5_Exp1_processed, "
            "robocasa/composite/ArrangeBreadBasket, "
            "./main/data/zeno-ai/day3_5_Exp1."
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
        help="Held-out test-set ratio computed over episodes.",
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
    if known_args.policy == "diffusion":
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
    parser.set_defaults(
        _policy_defaults_path=(None if defaults_path is None else str(defaults_path)),
        _policy_defaults_dataset_root=defaults.get("dataset_root"),
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
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    if args.policy == "streaming_act":
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

    defaults_dataset_root = getattr(args, "_policy_defaults_dataset_root", None)
    try:
        source_dataset_root = resolve_dataset_root(
            args.dataset,
            local_data_root=args.local_data_root.resolve(),
        )
    except FileNotFoundError:
        if not defaults_dataset_root:
            raise
        source_dataset_root = resolve_dataset_root(
            defaults_dataset_root,
            local_data_root=args.local_data_root.resolve(),
        )
    dataset_repo_id = (
        str(args.dataset_repo_id)
        if args.dataset_repo_id
        else infer_dataset_repo_id(
            source_dataset_root,
            local_data_root=args.local_data_root.resolve(),
        )
    )
    dataset_root = ensure_lerobot_dataset_v30_compat(
        source_dataset_root,
        dataset_repo_id=dataset_repo_id,
        local_data_root=args.local_data_root.resolve(),
    )
    validate_dataset_root(dataset_root)
    split_spec = build_dataset_split(
        dataset_arg=args.dataset,
        dataset_root=source_dataset_root,
        dataset_repo_id=dataset_repo_id,
        test_ratio=float(args.test_ratio),
        split_seed=int(args.split_seed),
        split_shuffle=bool(args.split_shuffle),
    )
    use_first_frame_anchor = bool(args.use_first_frame_anchor)
    validate_first_frame_anchor_dataset(
        dataset_root=dataset_root,
        use_first_frame_anchor=use_first_frame_anchor,
    )
    if args.policy != "streaming_act" and use_first_frame_anchor:
        raise NotImplementedError(
            "`--enable-first-frame-anchor` is only supported by the local "
            "`streaming_act` implementation. Other current policies, including "
            "`act` and `diffusion`, should be run without it."
        )
    use_imagenet_stats = resolve_use_imagenet_stats(
        dataset_root=dataset_root,
        use_imagenet_stats=args.use_imagenet_stats,
    )
    visual_storage_modes = summarize_visual_storage_modes(dataset_root)

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    if args.policy == "streaming_act":
        from lerobot_policy_streaming_act.configuration_streaming_act import (
            DELTA_SIGNATURE_KEY,
            PATH_SIGNATURE_KEY,
            StreamingACTConfig,
        )
        from lerobot_policy_streaming_act.signature_cache import (
            SignatureCacheRuntimeConfig,
            configure_signature_cache_runtime,
            prepare_signature_cache_runtime,
        )
    else:
        configure_signature_cache_runtime = None
        prepare_signature_cache_runtime = None
    install_torch_dataloader_patch(
        persistent_workers=bool(args.dataloader_persistent_workers),
        prefetch_factor=int(args.dataloader_prefetch_factor),
    )
    install_lerobot_dataset_load_patch()
    install_episode_aware_sampler_patch()

    if args.policy == "streaming_act":
        use_path_signature = args.use_path_signature
        use_delta_signature = bool(args.use_delta_signature)
        required_signature_parquet_keys = [
            key
            for key, enabled in (
                ("observation.path_signature", bool(use_path_signature)),
                ("observation.delta_signature", bool(use_delta_signature)),
            )
            if enabled
        ]
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
    else:
        use_path_signature = False
        use_delta_signature = False
        resolved_history_length = 0
        signature_dim = 0

    resolved_diffusion_drop_n_last_frames = None
    if args.policy == "diffusion":
        resolved_diffusion_drop_n_last_frames = resolve_diffusion_drop_n_last_frames(
            n_obs_steps=int(args.n_obs_steps),
            horizon=int(args.horizon),
            n_action_steps=int(args.n_action_steps),
            drop_n_last_frames=args.drop_n_last_frames,
        )

    input_features_override = None
    output_features_override = None

    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_root / run_stamp).resolve()
    split_path = output_dir / "dataset_split.json"

    wandb_enable = args.enable_wandb and (args.wandb_mode != "disabled")
    resolved_wandb_project = (
        str(args.wandb_project)
        if args.wandb_project
        else default_wandb_project_name(
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
        )
    )
    resolved_job_name = args.job_name
    if wandb_enable:
        resolved_job_name = (
            args.wandb_run_name
            if args.wandb_run_name
            else f"{args.job_name}-s{args.seed}-{run_stamp}"
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
        repo_id=dataset_repo_id,
        root=str(dataset_root),
        episodes=split_spec.train_episode_indices,
        use_imagenet_stats=use_imagenet_stats,
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
        if signature_cache_reader is None and required_signature_parquet_keys and not parquet_has_columns(
            dataset_root,
            required_signature_parquet_keys,
        ):
            raise RuntimeError(
                "Signature cache is required for this dataset because the parquet files "
                "do not contain the requested signature columns, but the cache could not be prepared."
            )
        if signature_cache_reader is not None and signature_cache_reader.pre_normalized:
            policy_cfg.pre_normalized_observation_keys = tuple(
                signature_cache_reader.feature_keys
            )
        else:
            policy_cfg.pre_normalized_observation_keys = ()
    elif args.policy == "diffusion":
        policy_cfg = DiffusionConfig(
            device=args.device,
            use_amp=bool(args.use_amp),
            push_to_hub=False,
            n_obs_steps=int(args.n_obs_steps),
            horizon=int(args.horizon),
            n_action_steps=int(args.n_action_steps),
            drop_n_last_frames=int(resolved_diffusion_drop_n_last_frames),
        )
    else:
        if configure_signature_cache_runtime is not None:
            configure_signature_cache_runtime(None)
        policy_cfg = ACTConfig(
            device=args.device,
            use_amp=bool(args.use_amp),
            push_to_hub=False,
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
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
            "environment is configured. Use `main/scripts/eval_policy.py` on the saved "
            "held-out test split instead."
        )

    cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        env=env_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=resolved_job_name,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=effective_eval_freq,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
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
    if args.task:
        print(f"- task: {args.task}")
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
    if args.policy == "diffusion":
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
        print(
            "- signature_cache: "
            f"mode={args.signature_cache_mode}, dtype={args.signature_cache_dtype}, "
            f"root={args.signature_cache_root or (dataset_root / '.signature_cache')}, "
            f"refresh={bool(args.refresh_signature_cache)}"
        )
        print(
            "- signature_runtime_normalization: "
            f"skip_keys={list(getattr(policy_cfg, 'pre_normalized_observation_keys', ()))}"
        )
    elif args.policy == "diffusion":
        print(
            "- diffusion: "
            f"n_obs_steps={int(args.n_obs_steps)}, "
            f"horizon={int(args.horizon)}, "
            "drop_n_last_frames="
            f"{int(resolved_diffusion_drop_n_last_frames)}"
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

    try:
        train(cfg, accelerator=accelerator)
        saved_split_path = save_dataset_split(output_dir, split_spec)
        print(f"Saved dataset split: {saved_split_path}")
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user. Cleaning up wandb before exit.")
        teardown_wandb_safely(exit_code=130)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
