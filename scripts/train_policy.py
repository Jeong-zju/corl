from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import hashlib
import inspect
import json
import os
import re
import shutil
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
            print("[WARN] CUDA AMP requested but CUDA is not available. Falling back to fp32.")
            return "no"

        if amp_dtype == "auto":
            if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                return "bf16"
            return "fp16"

        if amp_dtype == "bf16" and not getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            print("[WARN] bf16 AMP requested but this CUDA device does not support bf16. Falling back to fp16.")
            return "fp16"

        return amp_dtype

    if device_type == "xpu":
        return "bf16" if amp_dtype == "auto" else amp_dtype

    print(f"[WARN] AMP is not configured for device={device_type!r}. Falling back to fp32.")
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
    if getattr(current_dataloader_cls, "_corl_patch_signature", None) == patch_signature:
        return

    class PatchedDataLoader(original_dataloader_cls):
        _corl_patch_signature = patch_signature

        def __init__(self, *args, **kwargs):
            try:
                bound = inspect.signature(original_dataloader_cls.__init__).bind_partial(
                    self, *args, **kwargs
                )
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


def install_torchcodec_decoder_cache_patch(max_cached_decoders: int) -> None:
    import fsspec
    import importlib
    from threading import Lock

    import lerobot.datasets.video_utils as video_utils
    import torch

    resolved_max_cached_decoders = max(1, int(max_cached_decoders))

    class BoundedVideoDecoderCache:
        def __init__(self, max_size: int):
            self._max_size = max(1, int(max_size))
            self._cache: collections.OrderedDict[str, tuple[object, object]] = (
                collections.OrderedDict()
            )
            self._lock = Lock()

        def get_decoder(self, video_path: str):
            if importlib.util.find_spec("torchcodec"):
                from torchcodec.decoders import VideoDecoder
            else:
                raise ImportError("torchcodec is required but not available.")

            video_path = str(video_path)

            with self._lock:
                if video_path in self._cache:
                    decoder, file_handle = self._cache.pop(video_path)
                    self._cache[video_path] = (decoder, file_handle)
                    return decoder

                while len(self._cache) >= self._max_size:
                    _, (_, evicted_file_handle) = self._cache.popitem(last=False)
                    evicted_file_handle.close()

                file_handle = fsspec.open(video_path).__enter__()
                decoder = VideoDecoder(file_handle, seek_mode="approximate")
                self._cache[video_path] = (decoder, file_handle)
                return decoder

        def clear(self):
            with self._lock:
                for _, file_handle in self._cache.values():
                    file_handle.close()
                self._cache.clear()

        def size(self) -> int:
            with self._lock:
                return len(self._cache)

    existing_cache = getattr(video_utils, "_default_decoder_cache", None)
    if existing_cache is not None and hasattr(existing_cache, "clear"):
        try:
            existing_cache.clear()
        except Exception:
            pass

    video_utils.VideoDecoderCache = BoundedVideoDecoderCache
    video_utils._default_decoder_cache = BoundedVideoDecoderCache(
        max_size=resolved_max_cached_decoders
    )

    def decode_video_frames_torchcodec_fast(
        video_path: Path | str,
        timestamps: list[float],
        tolerance_s: float,
        log_loaded_timestamps: bool = False,
        decoder_cache: BoundedVideoDecoderCache | None = None,
    ) -> torch.Tensor:
        if decoder_cache is None:
            decoder_cache = video_utils._default_decoder_cache

        decoder = decoder_cache.get_decoder(str(video_path))
        average_fps = decoder.metadata.average_fps
        frame_indices = [round(ts * average_fps) for ts in timestamps]
        frames_batch = decoder.get_frames_at(indices=frame_indices)

        loaded_ts = frames_batch.pts_seconds
        if not isinstance(loaded_ts, torch.Tensor):
            loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)
        else:
            loaded_ts = loaded_ts.to(dtype=torch.float32)

        query_ts = torch.tensor(timestamps, dtype=torch.float32)
        if len(query_ts) != len(loaded_ts):
            raise video_utils.FrameTimestampError(
                f"Number of retrieved timestamps ({len(loaded_ts)}) does not match "
                f"number of queried timestamps ({len(query_ts)})"
            )

        min_ = torch.abs(query_ts - loaded_ts)
        is_within_tol = min_ < tolerance_s
        if not is_within_tol.all():
            raise video_utils.FrameTimestampError(
                f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
                " It means that the closest frame that can be loaded from the video is too far away in time."
                " This might be due to synchronization issues with timestamps during data collection."
                " To be safe, we advise to ignore this item during training."
                f"\nqueried timestamps: {query_ts}"
                f"\nloaded timestamps: {loaded_ts}"
                f"\nvideo: {video_path}"
            )

        closest_frames = frames_batch.data
        if not isinstance(closest_frames, torch.Tensor):
            closest_frames = torch.stack(list(closest_frames), dim=0)
        closest_frames = closest_frames.to(dtype=torch.float32) / 255.0

        if log_loaded_timestamps:
            print(f"[INFO] torchcodec.loaded_timestamps: {loaded_ts.tolist()}")

        return closest_frames

    video_utils.decode_video_frames_torchcodec = decode_video_frames_torchcodec_fast


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


def resolve_filtered_hf_cache_root(
    dataset_root: Path,
    filtered_hf_cache_root: Path | None,
) -> Path:
    if filtered_hf_cache_root is None:
        return dataset_root / ".cache" / "hf_datasets"
    if filtered_hf_cache_root.is_absolute():
        return filtered_hf_cache_root
    return dataset_root / filtered_hf_cache_root


def compute_dataset_metadata_fingerprint(dataset_root: Path) -> str:
    meta_root = dataset_root / "meta"
    digest = hashlib.sha256()
    candidate_paths = [
        meta_root / "info.json",
        meta_root / "stats.json",
        *sorted((meta_root / "episodes").rglob("*.parquet")),
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        stat = path.stat()
        rel_path = path.relative_to(dataset_root)
        digest.update(str(rel_path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()[:16]


def build_filtered_hf_cache_dir(
    *,
    filtered_hf_cache_root: Path,
    dataset_root: Path,
    dataset_repo_id: str,
    episodes: list[int],
) -> Path:
    metadata_fingerprint = compute_dataset_metadata_fingerprint(dataset_root)
    selection_payload = {
        "dataset_repo_id": dataset_repo_id,
        "dataset_root": str(dataset_root.resolve()),
        "episodes": [int(ep) for ep in episodes],
    }
    selection_fingerprint = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    sanitized_repo_id = dataset_repo_id.replace("/", "__")
    return (
        filtered_hf_cache_root
        / sanitized_repo_id
        / f"meta-{metadata_fingerprint}"
        / f"episodes-{selection_fingerprint}"
    )


def install_lerobot_dataset_load_patch(
    *,
    dataset_root: Path,
    filtered_hf_cache_root: Path,
    enable_filtered_hf_cache: bool,
    refresh_filtered_hf_cache: bool,
) -> None:
    import datasets
    import torch
    import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import (
        get_hf_features_from_features,
        hf_transform_to_torch,
        load_nested_dataset,
    )

    patch_config = {
        "dataset_root": dataset_root.resolve(),
        "filtered_hf_cache_root": filtered_hf_cache_root.resolve(),
        "enable_filtered_hf_cache": bool(enable_filtered_hf_cache),
        "refresh_filtered_hf_cache": bool(refresh_filtered_hf_cache),
    }

    if getattr(LeRobotDataset, "_custom_dataset_load_patch_installed", False):
        LeRobotDataset._custom_dataset_load_patch_config = patch_config
        return

    original_load_metadata = LeRobotDatasetMetadata.load_metadata
    original_cache_check = LeRobotDataset._check_cached_episodes_sufficient

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

    def get_hf_features_from_features_with_raw_images(features: dict) -> datasets.Features:
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

        hf_num_rows = len(self.hf_dataset) if self.hf_dataset is not None else rel_offset
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
        return compact_rel_starts[position] + (int(abs_idx) - compact_abs_starts[position])

    def _map_absolute_indices_to_relative(self, absolute_indices: list[int]) -> list[int]:
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
        self.revision = revision if revision else lerobot_dataset_module.CODEBASE_VERSION
        self.video_backend = (
            video_backend if video_backend else lerobot_dataset_module.get_safe_default_codec()
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
                raise FileNotFoundError("Cached dataset doesn't contain all requested episodes")
        except (FileNotFoundError, NotADirectoryError):
            if lerobot_dataset_module.is_valid_version(self.revision):
                self.revision = lerobot_dataset_module.get_safe_version(
                    self.repo_id, self.revision
                )
            self.download(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        _install_compact_relative_index_layout(self)

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
        elapsed_s = time.perf_counter() - start_s
        print(
            f"[INFO] dataset.load_metadata: {elapsed_s:.1f}s "
            f"(root={self.root / 'meta'})"
        )

    def load_hf_dataset_with_timing(self):
        patch_cfg = LeRobotDataset._custom_dataset_load_patch_config
        resolved_root = self.root.resolve()
        hf_transform = build_hf_transform_to_torch(self.features)
        cache_info = {
            "status": "disabled",
            "path": None,
        }
        can_use_filtered_cache = (
            patch_cfg["enable_filtered_hf_cache"]
            and resolved_root == patch_cfg["dataset_root"]
            and self.episodes is not None
            and len(self.episodes) > 0
        )
        load_start_s = time.perf_counter()

        if can_use_filtered_cache:
            cache_dir = build_filtered_hf_cache_dir(
                filtered_hf_cache_root=patch_cfg["filtered_hf_cache_root"],
                dataset_root=resolved_root,
                dataset_repo_id=self.repo_id,
                episodes=self.episodes,
            )
            cache_info["path"] = str(cache_dir)
            if patch_cfg["refresh_filtered_hf_cache"] and cache_dir.exists():
                shutil.rmtree(cache_dir)

            if cache_dir.exists():
                try:
                    hf_dataset = datasets.load_from_disk(str(cache_dir))
                    hf_dataset.set_transform(hf_transform)
                    load_elapsed_s = time.perf_counter() - load_start_s
                    cache_info["status"] = "filtered_cache_hit"
                    self._custom_hf_dataset_cache_info = cache_info
                    print(
                        "[INFO] dataset.load_hf_dataset: "
                        f"{load_elapsed_s:.1f}s [{cache_info['status']}] "
                        f"(rows={len(hf_dataset)}, cache_dir={cache_dir})"
                    )
                    return hf_dataset
                except Exception as exc:
                    print(
                        "[WARN] Failed to load filtered HF dataset cache from "
                        f"{cache_dir}: {exc}. Rebuilding it."
                    )
                    shutil.rmtree(cache_dir, ignore_errors=True)

            features = get_hf_features_from_features_with_raw_images(self.features)
            hf_dataset = load_nested_dataset(
                self.root / "data",
                features=features,
                episodes=self.episodes,
            )
            load_elapsed_s = time.perf_counter() - load_start_s

            persist_start_s = time.perf_counter()
            temp_cache_dir = cache_dir.with_name(f"{cache_dir.name}.tmp")
            shutil.rmtree(temp_cache_dir, ignore_errors=True)
            temp_cache_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                hf_dataset.save_to_disk(str(temp_cache_dir))
                cache_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.rmtree(cache_dir, ignore_errors=True)
                temp_cache_dir.rename(cache_dir)
                cache_info["status"] = "filtered_cache_miss_saved"
                persist_elapsed_s = time.perf_counter() - persist_start_s
                cache_meta = {
                    "dataset_repo_id": self.repo_id,
                    "dataset_root": str(resolved_root),
                    "episodes": [int(ep) for ep in self.episodes],
                    "num_rows": len(hf_dataset),
                    "created_at": dt.datetime.now().isoformat(),
                }
                (cache_dir / "cache_meta.json").write_text(
                    json.dumps(cache_meta, indent=2, ensure_ascii=True),
                    encoding="utf-8",
                )
                print(
                    "[INFO] dataset.load_hf_dataset: "
                    f"{load_elapsed_s:.1f}s [filtered_cache_miss] "
                    f"(rows={len(hf_dataset)}, cache_dir={cache_dir})"
                )
                print(
                    "[INFO] dataset.persist_filtered_hf_cache: "
                    f"{persist_elapsed_s:.1f}s (cache_dir={cache_dir})"
                )
            except Exception as exc:
                cache_info["status"] = "filtered_cache_miss_unsaved"
                shutil.rmtree(temp_cache_dir, ignore_errors=True)
                print(
                    "[WARN] Failed to persist filtered HF dataset cache at "
                    f"{cache_dir}: {exc}"
                )

            hf_dataset.set_transform(hf_transform)
            self._custom_hf_dataset_cache_info = cache_info
            return hf_dataset

        hf_dataset = load_nested_dataset(
            self.root / "data",
            features=get_hf_features_from_features_with_raw_images(self.features),
            episodes=self.episodes,
        )
        hf_dataset.set_transform(hf_transform)
        load_elapsed_s = time.perf_counter() - load_start_s
        cache_reason = "full_dataset_or_disabled"
        if not patch_cfg["enable_filtered_hf_cache"]:
            cache_reason = "cache_disabled"
        elif resolved_root != patch_cfg["dataset_root"]:
            cache_reason = "different_dataset_root"
        elif self.episodes is None:
            cache_reason = "all_episodes_requested"
        cache_info["status"] = cache_reason
        self._custom_hf_dataset_cache_info = cache_info
        print(
            "[INFO] dataset.load_hf_dataset: "
            f"{load_elapsed_s:.1f}s [{cache_reason}] "
            f"(rows={len(hf_dataset)})"
        )
        return hf_dataset

    def cache_check_with_timing(self):
        start_s = time.perf_counter()
        is_sufficient = original_cache_check(self)
        elapsed_s = time.perf_counter() - start_s
        cache_info = getattr(self, "_custom_hf_dataset_cache_info", None) or {}
        cache_status = cache_info.get("status", "unknown")
        print(
            "[INFO] dataset.cache_check: "
            f"{elapsed_s:.1f}s (ok={is_sufficient}, hf_cache={cache_status})"
        )
        return is_sufficient

    def get_query_timestamps_with_compact_relative_index(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        shared_timestamp_cache: dict[tuple[int, ...], list[float]] = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                absolute_indices = tuple(int(idx) for idx in query_indices[key])
                timestamps = shared_timestamp_cache.get(absolute_indices)
                if timestamps is None:
                    relative_indices = _map_absolute_indices_to_relative(
                        self, list(absolute_indices)
                    )
                    timestamps = torch.stack(
                        self.hf_dataset[relative_indices]["timestamp"]
                    ).tolist()
                    shared_timestamp_cache[absolute_indices] = timestamps
                query_timestamps[key] = timestamps
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def query_hf_dataset_with_compact_relative_index(self, query_indices: dict[str, list[int]]) -> dict:
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            relative_indices = _map_absolute_indices_to_relative(self, q_idx)
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    LeRobotDataset.__init__ = init_with_compact_relative_index_layout
    LeRobotDatasetMetadata.load_metadata = load_metadata_with_timing
    LeRobotDataset.load_hf_dataset = load_hf_dataset_with_timing
    LeRobotDataset._check_cached_episodes_sufficient = cache_check_with_timing
    LeRobotDataset._get_query_timestamps = get_query_timestamps_with_compact_relative_index
    LeRobotDataset._query_hf_dataset = query_hf_dataset_with_compact_relative_index
    LeRobotDataset._custom_dataset_load_patch_installed = True
    LeRobotDataset._custom_dataset_load_patch_config = patch_config

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
    use_imagenet_stats: bool,
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


def default_train_output_root(policy_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "main" / "outputs" / "train" / default_policy_series_name(
        policy_name
    )


def ensure_writable_hf_cache_env(repo_root: Path) -> None:
    cache_root = (repo_root / "main" / ".cache" / "huggingface").resolve()
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
        raise ValueError(
            f"`--n-action-steps` must be positive, got {n_action_steps}."
        )

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
    parser.add_argument(
        "--torchcodec-decoder-cache-size",
        type=int,
        default=defaults.get("torchcodec_decoder_cache_size", 32),
        help=(
            "Maximum number of torchcodec video decoders kept open per process. "
            "Lower values reduce open-file pressure."
        ),
    )
    filtered_hf_cache_group = parser.add_mutually_exclusive_group()
    filtered_hf_cache_group.add_argument(
        "--enable-filtered-hf-cache",
        dest="use_filtered_hf_cache",
        action="store_true",
        help=(
            "Persist the episode-filtered HF dataset view to disk and reuse it "
            "on later runs with the same split."
        ),
    )
    filtered_hf_cache_group.add_argument(
        "--disable-filtered-hf-cache",
        dest="use_filtered_hf_cache",
        action="store_false",
        help="Disable the persisted filtered HF dataset cache.",
    )
    parser.set_defaults(
        use_filtered_hf_cache=defaults.get("enable_filtered_hf_cache", False),
    )
    parser.add_argument(
        "--filtered-hf-cache-root",
        type=Path,
        default=defaults.get("filtered_hf_cache_root"),
        help=(
            "Optional cache root for filtered HF datasets. Relative paths are "
            "resolved under the dataset root."
        ),
    )
    parser.add_argument(
        "--refresh-filtered-hf-cache",
        action="store_true",
        default=defaults.get("refresh_filtered_hf_cache", False),
        help="Rebuild the filtered HF dataset cache even if a matching cache exists.",
    )
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--log-freq", type=int, default=defaults.get("log_freq", 50))
    parser.add_argument("--save-freq", type=int, default=defaults.get("save_freq", 1000))
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
                "a Signature-Indexed Slot Memory (SISM) updater."
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
    parser.set_defaults(
        _policy_defaults_path=(
            None if defaults_path is None else str(defaults_path)
        ),
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
    if args.filtered_hf_cache_root is not None and not isinstance(
        args.filtered_hf_cache_root, Path
    ):
        args.filtered_hf_cache_root = Path(args.filtered_hf_cache_root)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    ensure_writable_hf_cache_env(repo_root)
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
    filtered_hf_cache_root = resolve_filtered_hf_cache_root(
        dataset_root=dataset_root,
        filtered_hf_cache_root=args.filtered_hf_cache_root,
    )

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    if args.policy == "streaming_act":
        from lerobot_policy_streaming_act.configuration_streaming_act import (
            StreamingACTConfig,
        )
    install_torch_dataloader_patch(
        persistent_workers=bool(args.dataloader_persistent_workers),
        prefetch_factor=int(args.dataloader_prefetch_factor),
    )
    if int(visual_storage_modes.get("video", 0)) > 0:
        install_torchcodec_decoder_cache_patch(
            max_cached_decoders=int(args.torchcodec_decoder_cache_size)
        )
    install_lerobot_dataset_load_patch(
        dataset_root=dataset_root,
        filtered_hf_cache_root=filtered_hf_cache_root,
        enable_filtered_hf_cache=bool(args.use_filtered_hf_cache),
        refresh_filtered_hf_cache=bool(args.refresh_filtered_hf_cache),
    )
    install_episode_aware_sampler_patch()

    if args.policy == "streaming_act":
        use_path_signature = args.use_path_signature
        use_delta_signature = bool(args.use_delta_signature)
        use_prefix_sequence_training = bool(args.use_prefix_sequence_training)
        use_visual_prefix_memory = bool(args.use_visual_prefix_memory)
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
            use_imagenet_stats=use_imagenet_stats,
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

    resolved_diffusion_drop_n_last_frames = None
    if args.policy == "diffusion":
        resolved_diffusion_drop_n_last_frames = (
            resolve_diffusion_drop_n_last_frames(
                n_obs_steps=int(args.n_obs_steps),
                horizon=int(args.horizon),
                n_action_steps=int(args.n_action_steps),
                drop_n_last_frames=args.drop_n_last_frames,
            )
        )

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
            use_signature_indexed_slot_memory=bool(
                args.use_signature_indexed_slot_memory
            ),
            use_memory_conditioned_encoder_film=bool(
                args.use_memory_conditioned_encoder_film
            ),
            num_memory_slots=int(args.num_memory_slots),
            slot_memory_num_slots=int(args.slot_memory_num_slots),
            slot_memory_routing_hidden_dim=int(
                args.slot_memory_routing_hidden_dim
            ),
            slot_memory_use_delta_routing=bool(
                args.slot_memory_use_delta_routing
            ),
            slot_memory_use_softmax_routing=bool(
                args.slot_memory_use_softmax_routing
            ),
            slot_memory_use_readout_pooling=bool(
                args.slot_memory_use_readout_pooling
            ),
            slot_memory_balance_loss_coef=float(
                args.slot_memory_balance_loss_coef
            ),
            slot_memory_consistency_loss_coef=float(
                args.slot_memory_consistency_loss_coef
            ),
            input_features=input_features_override,
            output_features=output_features_override,
        )
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
        print(
            "- torchcodec_decoder_cache_size: "
            f"{int(args.torchcodec_decoder_cache_size)}"
        )
    else:
        print(
            "- video_backend: ignored "
            "(dataset stores visual observations directly in parquet image columns)"
        )
    print(
        "- filtered_hf_cache: "
        f"enable={bool(args.use_filtered_hf_cache)}, "
        f"root={filtered_hf_cache_root}, "
        f"refresh={bool(args.refresh_filtered_hf_cache)}"
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
        "- torch_sharing_strategy: "
        f"{resolved_torch_sharing_strategy or 'default'}"
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
