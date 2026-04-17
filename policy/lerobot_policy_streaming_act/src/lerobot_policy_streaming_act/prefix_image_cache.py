from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

_CACHE_LAYOUT_VERSION = 1
_CACHE_BUILD_LOCK_STALE_TIMEOUT_S = 6 * 60 * 60
_CACHE_BUILD_LOCK_POLL_INTERVAL_S = 1.0


def _maybe_create_tqdm(*, total: int, desc: str, unit: str):
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return None
    return tqdm(total=total, desc=desc, unit=unit)


def _sanitize_path_part(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "item"


def _dtype_name(dtype: str) -> str:
    dtype_name = str(dtype).lower()
    if dtype_name not in {"uint8", "float16", "float32"}:
        raise ValueError(
            "`prefix-image-cache-dtype` must be one of "
            "{'uint8', 'float16', 'float32'}. "
            f"Got {dtype!r}."
        )
    return dtype_name


def _mode_name(mode: str) -> str:
    mode_name = str(mode).lower()
    if mode_name not in {"off", "memmap", "ram"}:
        raise ValueError(
            "`prefix-image-cache-mode` must be one of {'off', 'memmap', 'ram'}. "
            f"Got {mode!r}."
        )
    return mode_name


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_dataset_manifest(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}.")

    episodes_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    episode_meta = pq.read_table(
        episodes_path,
        columns=["dataset_from_index", "dataset_to_index", "length"],
    )
    total_frames = int(max(episode_meta.column("dataset_to_index").to_pylist()))

    return {
        "dataset_root": str(dataset_root),
        "info_path": str(dataset_root / "meta/info.json"),
        "episodes_path": str(episodes_path),
        "total_frames": total_frames,
        "data_files": [
            {
                "path": str(path.relative_to(dataset_root)),
                "size": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
            }
            for path in data_files
        ],
    }


@contextmanager
def _cache_build_lock(lock_path: Path, *, label: str):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    warned_waiting = False

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                lock_age_s = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue

            if lock_age_s > _CACHE_BUILD_LOCK_STALE_TIMEOUT_S:
                try:
                    lock_path.unlink()
                    print(
                        "[WARN] "
                        f"{label}: removed stale cache lock {lock_path} "
                        f"(age={lock_age_s:.0f}s)"
                    )
                    continue
                except FileNotFoundError:
                    continue

            if not warned_waiting:
                print(f"[INFO] {label}: waiting for cache lock {lock_path}")
                warned_waiting = True
            time.sleep(_CACHE_BUILD_LOCK_POLL_INTERVAL_S)
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at": dt.datetime.utcnow().isoformat() + "Z"}))
        break

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _default_cache_root(dataset_root: Path, dataset_repo_id: str) -> Path:
    return dataset_root / ".prefix_image_cache" / _sanitize_path_part(dataset_repo_id)


def _cache_dir(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    cache_root: Path | None,
) -> Path:
    root = (
        _default_cache_root(dataset_root, dataset_repo_id)
        if cache_root is None
        else Path(cache_root).resolve()
    )
    return root / f"prefix_image_cache_v{_CACHE_LAYOUT_VERSION}"


def _normalize_image_shape_to_chw(
    shape: list[int] | tuple[int, int, int],
    *,
    key: str,
) -> tuple[int, int, int]:
    dims = tuple(int(dim) for dim in shape)
    if len(dims) != 3:
        raise ValueError(
            f"Invalid camera feature shape for {key!r}: {shape}. Expected a 3D shape."
        )

    first_is_channel = dims[0] in {1, 3, 4}
    last_is_channel = dims[2] in {1, 3, 4}

    if first_is_channel and not last_is_channel:
        return dims
    if last_is_channel and not first_is_channel:
        return (dims[2], dims[0], dims[1])
    if first_is_channel and last_is_channel:
        # Small synthetic images can make the layout ambiguous (for example 3x3x3).
        # Prefer channel-first to match the runtime tensors emitted by LeRobotDataset.
        return dims

    raise ValueError(
        "Could not infer channel layout for camera feature "
        f"{key!r} with shape={shape}. Expected either CHW or HWC with 1/3/4 channels."
    )


def _resolve_camera_specs(info: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    specs: dict[str, tuple[int, int, int]] = {}
    for key, feature_spec in info.get("features", {}).items():
        if not isinstance(feature_spec, dict):
            continue
        dtype = str(feature_spec.get("dtype", ""))
        if dtype not in {"image", "video"}:
            continue
        key = str(key)
        if key == "observation.anchor_image":
            continue
        if key.startswith("observation.prefix_images."):
            continue
        if not (key.startswith("observation.images.") or key == "observation.image"):
            continue
        shape = feature_spec.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            raise ValueError(
                f"Invalid camera feature shape for {key!r}: {shape}. Expected a 3D image shape."
            )
        specs[key] = _normalize_image_shape_to_chw(shape, key=key)
    if not specs:
        raise KeyError(
            "Could not resolve any regular observation image keys for the prefix "
            "image cache."
        )
    return specs


def _convert_image_for_storage(image: torch.Tensor, *, cache_dtype: str) -> np.ndarray:
    tensor = torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(
            "Prefix image cache expects individual images with shape (C, H, W). "
            f"Got {tuple(tensor.shape)}."
        )
    if tensor.shape[0] not in {1, 3, 4} and tensor.shape[-1] in {1, 3, 4}:
        tensor = tensor.permute(2, 0, 1).contiguous()
    tensor = tensor.detach().cpu().to(dtype=torch.float32)
    if cache_dtype == "uint8":
        if tensor.numel() > 0 and float(tensor.max().item()) <= 1.0:
            tensor = (tensor * 255.0).round().clamp(0.0, 255.0)
        return tensor.to(dtype=torch.uint8).numpy()
    return tensor.to(dtype=getattr(torch, cache_dtype)).numpy()


def _convert_loaded_images(array: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        tensor = array
    else:
        tensor = torch.from_numpy(np.array(array, copy=True))
    if tensor.dtype == torch.uint8:
        return tensor.to(dtype=torch.float32) / 255.0
    return tensor.to(dtype=torch.float32)


@dataclass
class PrefixImageCacheReader:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray | torch.Tensor]
    mode: str

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return tuple(self.metadata["camera_keys"])

    @property
    def estimated_bytes(self) -> int:
        return int(self.metadata.get("estimated_bytes", 0))

    def has_key(self, key: str) -> bool:
        return key in self.arrays

    def get(self, key: str, absolute_index: int) -> torch.Tensor:
        source = self.arrays[key]
        index = int(absolute_index)
        if isinstance(source, torch.Tensor):
            return _convert_loaded_images(source[index])
        return _convert_loaded_images(source[index])

    def get_many(self, key: str, absolute_indices: list[int]) -> torch.Tensor:
        source = self.arrays[key]
        if isinstance(source, torch.Tensor):
            return _convert_loaded_images(
                source[torch.as_tensor(absolute_indices, dtype=torch.long)]
            )
        return _convert_loaded_images(
            source[np.asarray(absolute_indices, dtype=np.int64)]
        )


@dataclass
class PrefixImageCacheRuntimeConfig:
    dataset_root: Path
    dataset_repo_id: str
    mode: str
    cache_dtype: str
    refresh: bool = False
    cache_root: Path | None = None
    reader: PrefixImageCacheReader | None = field(default=None, init=False)
    status: str = field(default="off", init=False)
    camera_specs: dict[str, tuple[int, int, int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root).resolve()
        self.mode = _mode_name(self.mode)
        self.cache_dtype = _dtype_name(self.cache_dtype)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


_PREFIX_IMAGE_CACHE_RUNTIME: PrefixImageCacheRuntimeConfig | None = None


def configure_prefix_image_cache_runtime(
    runtime: PrefixImageCacheRuntimeConfig | None,
) -> None:
    global _PREFIX_IMAGE_CACHE_RUNTIME
    _PREFIX_IMAGE_CACHE_RUNTIME = runtime


def get_prefix_image_cache_runtime() -> PrefixImageCacheRuntimeConfig | None:
    return _PREFIX_IMAGE_CACHE_RUNTIME


def get_prefix_image_cache_reader_for_dataset(
    dataset_root: Path,
) -> PrefixImageCacheReader | None:
    runtime = get_prefix_image_cache_runtime()
    if runtime is None or runtime.reader is None:
        return None
    if runtime.dataset_root != Path(dataset_root).resolve():
        return None
    return runtime.reader


def _load_cache_metadata(cache_dir: Path) -> dict[str, Any] | None:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    return _json_load(metadata_path)


def _manifests_are_compatible(
    cached_manifest: dict[str, Any] | None,
    current_manifest: dict[str, Any],
) -> bool:
    if not isinstance(cached_manifest, dict):
        return False
    if cached_manifest.get("dataset_root") != current_manifest.get("dataset_root"):
        return False
    if cached_manifest.get("info_path") != current_manifest.get("info_path"):
        return False
    if cached_manifest.get("episodes_path") != current_manifest.get("episodes_path"):
        return False
    if int(cached_manifest.get("total_frames", -1)) != int(
        current_manifest.get("total_frames", -1)
    ):
        return False

    cached_files = cached_manifest.get("data_files")
    current_files = current_manifest.get("data_files")
    if not isinstance(cached_files, list) or not isinstance(current_files, list):
        return False
    if len(cached_files) != len(current_files):
        return False

    for cached_entry, current_entry in zip(cached_files, current_files, strict=True):
        if cached_entry.get("path") != current_entry.get("path"):
            return False
        if int(cached_entry.get("size", -1)) != int(current_entry.get("size", -1)):
            return False
    return True


def _metadata_matches_runtime(
    metadata: dict[str, Any],
    *,
    runtime: PrefixImageCacheRuntimeConfig,
    manifest: dict[str, Any],
    camera_specs: dict[str, tuple[int, int, int]],
) -> bool:
    metadata_specs = {
        key: tuple(int(dim) for dim in value)
        for key, value in metadata.get("camera_specs", {}).items()
    }
    return (
        metadata.get("layout_version") == _CACHE_LAYOUT_VERSION
        and metadata.get("dataset_root") == str(runtime.dataset_root)
        and metadata.get("dataset_repo_id") == runtime.dataset_repo_id
        and metadata.get("cache_dtype") == runtime.cache_dtype
        and metadata_specs == camera_specs
        and _manifests_are_compatible(metadata.get("dataset_manifest"), manifest)
    )


def _open_prefix_image_cache_reader(
    cache_dir: Path,
    metadata: dict[str, Any],
    *,
    mode: str,
) -> PrefixImageCacheReader:
    arrays: dict[str, np.ndarray | torch.Tensor] = {}
    estimated_bytes = 0
    for camera_key in metadata["camera_keys"]:
        filename = re.sub(r"[^A-Za-z0-9._-]+", "-", str(camera_key)) + ".npy"
        array_path = cache_dir / filename
        if not array_path.exists():
            raise FileNotFoundError(f"Prefix image cache array is missing: {array_path}")
        if mode == "ram":
            tensor = torch.from_numpy(np.load(array_path, mmap_mode=None)).clone()
            arrays[camera_key] = tensor
            estimated_bytes += int(tensor.numel() * tensor.element_size())
        else:
            mmap_array = np.load(array_path, mmap_mode="r")
            arrays[camera_key] = mmap_array
            estimated_bytes += int(mmap_array.size * mmap_array.dtype.itemsize)
    reader_metadata = dict(metadata)
    reader_metadata["estimated_bytes"] = estimated_bytes
    return PrefixImageCacheReader(metadata=reader_metadata, arrays=arrays, mode=mode)


def _write_cache_metadata(cache_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = cache_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_prefix_image_cache(runtime: PrefixImageCacheRuntimeConfig) -> PrefixImageCacheReader:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = _json_load(runtime.dataset_root / "meta/info.json")
    camera_specs = _resolve_camera_specs(info)
    manifest = _build_dataset_manifest(runtime.dataset_root)
    total_frames = int(manifest["total_frames"])

    cache_dir = _cache_dir(
        dataset_root=runtime.dataset_root,
        dataset_repo_id=runtime.dataset_repo_id,
        cache_root=runtime.cache_root,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[INFO] prefix_image_cache: build "
        f"(mode={runtime.mode}, dtype={runtime.cache_dtype}, path={cache_dir})"
    )
    build_start = dt.datetime.now()

    arrays: dict[str, np.memmap] = {}
    total_bytes = 0
    for camera_key, shape in camera_specs.items():
        filename = re.sub(r"[^A-Za-z0-9._-]+", "-", str(camera_key)) + ".npy"
        mmap_array = np.lib.format.open_memmap(
            cache_dir / filename,
            mode="w+",
            dtype=np.dtype(runtime.cache_dtype),
            shape=(total_frames, *shape),
        )
        arrays[camera_key] = mmap_array
        total_bytes += int(mmap_array.size * mmap_array.dtype.itemsize)

    dataset = LeRobotDataset(
        repo_id=runtime.dataset_repo_id,
        root=str(runtime.dataset_root),
        image_transforms=None,
    )
    dataset_size = len(dataset)
    progress = _maybe_create_tqdm(
        total=dataset_size,
        desc="Build prefix_image_cache",
        unit="frame",
    )

    try:
        for dataset_index in range(dataset_size):
            item = dataset[dataset_index]
            absolute_index = int(
                item["index"].item() if isinstance(item["index"], torch.Tensor) else item["index"]
            )
            for camera_key in camera_specs:
                arrays[camera_key][absolute_index] = _convert_image_for_storage(
                    item[camera_key],
                    cache_dtype=runtime.cache_dtype,
                )
            if progress is not None:
                progress.update(1)
                if (dataset_index + 1) % 256 == 0 or (dataset_index + 1) == dataset_size:
                    progress.set_postfix_str(
                        f"last_abs_idx={absolute_index}, cameras={len(camera_specs)}"
                    )
    finally:
        if progress is not None:
            progress.close()

    for array in arrays.values():
        array.flush()

    metadata = {
        "layout_version": _CACHE_LAYOUT_VERSION,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "dataset_root": str(runtime.dataset_root),
        "dataset_repo_id": runtime.dataset_repo_id,
        "cache_dtype": runtime.cache_dtype,
        "camera_keys": list(camera_specs.keys()),
        "camera_specs": {
            key: list(shape) for key, shape in camera_specs.items()
        },
        "estimated_bytes": total_bytes,
        "dataset_manifest": manifest,
    }
    _write_cache_metadata(cache_dir, metadata)
    reader = _open_prefix_image_cache_reader(cache_dir, metadata, mode=runtime.mode)
    build_elapsed_s = (dt.datetime.now() - build_start).total_seconds()
    print(
        "[INFO] prefix_image_cache: ready "
        f"(cameras={len(camera_specs)}, frames={dataset_size}, "
        f"size={reader.estimated_bytes / (1024 ** 3):.2f} GiB, "
        f"elapsed={build_elapsed_s:.1f}s)"
    )
    if runtime.mode == "ram":
        print(
            "[INFO] prefix_image_cache: preloaded RAM "
            f"({reader.estimated_bytes / (1024 ** 3):.2f} GiB)"
        )
    return reader


def prepare_prefix_image_cache_runtime() -> PrefixImageCacheReader | None:
    runtime = get_prefix_image_cache_runtime()
    if runtime is None or not runtime.enabled:
        return None
    if runtime.reader is not None:
        return runtime.reader

    info = _json_load(runtime.dataset_root / "meta/info.json")
    camera_specs = _resolve_camera_specs(info)
    manifest = _build_dataset_manifest(runtime.dataset_root)
    cache_dir = _cache_dir(
        dataset_root=runtime.dataset_root,
        dataset_repo_id=runtime.dataset_repo_id,
        cache_root=runtime.cache_root,
    )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    metadata = None if runtime.refresh else _load_cache_metadata(cache_dir)

    if metadata is not None and _metadata_matches_runtime(
        metadata,
        runtime=runtime,
        manifest=manifest,
        camera_specs=camera_specs,
    ):
        runtime.reader = _open_prefix_image_cache_reader(
            cache_dir,
            metadata,
            mode=runtime.mode,
        )
        print(
            "[INFO] prefix_image_cache: hit "
            f"(mode={runtime.mode}, dtype={runtime.cache_dtype}, path={cache_dir})"
        )
        if runtime.mode == "ram":
            print(
                "[INFO] prefix_image_cache: preloaded RAM "
                f"({runtime.reader.estimated_bytes / (1024 ** 3):.2f} GiB)"
            )
        runtime.status = "ready"
        runtime.camera_specs = camera_specs
        return runtime.reader

    lock_path = cache_dir / ".build.lock"
    with _cache_build_lock(lock_path, label="prefix_image_cache"):
        metadata = None if runtime.refresh else _load_cache_metadata(cache_dir)
        if metadata is not None and _metadata_matches_runtime(
            metadata,
            runtime=runtime,
            manifest=manifest,
            camera_specs=camera_specs,
        ):
            runtime.reader = _open_prefix_image_cache_reader(
                cache_dir,
                metadata,
                mode=runtime.mode,
            )
            print(
                "[INFO] prefix_image_cache: hit "
                f"(mode={runtime.mode}, dtype={runtime.cache_dtype}, path={cache_dir})"
            )
            if runtime.mode == "ram":
                print(
                    "[INFO] prefix_image_cache: preloaded RAM "
                    f"({runtime.reader.estimated_bytes / (1024 ** 3):.2f} GiB)"
                )
            runtime.status = "ready"
            runtime.camera_specs = camera_specs
            return runtime.reader

        runtime.reader = _build_prefix_image_cache(runtime)
        runtime.status = "ready"
        runtime.camera_specs = camera_specs
        return runtime.reader
