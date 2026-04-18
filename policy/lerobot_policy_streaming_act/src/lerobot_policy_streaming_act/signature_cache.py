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

from lerobot.configs.types import NormalizationMode

PATH_SIGNATURE_KEY = "observation.path_signature"
DELTA_SIGNATURE_KEY = "observation.delta_signature"
SIGNATURE_FEATURE_KEYS = (PATH_SIGNATURE_KEY, DELTA_SIGNATURE_KEY)
_CACHE_LAYOUT_VERSION = 1
_CACHE_BUILD_LOCK_STALE_TIMEOUT_S = 6 * 60 * 60
_CACHE_BUILD_LOCK_POLL_INTERVAL_S = 1.0


def _sanitize_path_part(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "item"


def _npy_dtype_name(dtype: str) -> str:
    dtype_name = str(dtype).lower()
    if dtype_name not in {"float16", "float32"}:
        raise ValueError(
            "`signature-cache-dtype` must be one of {'float16', 'float32'}. "
            f"Got {dtype}."
        )
    return dtype_name


def _mode_name(mode: str) -> str:
    mode_name = str(mode).lower()
    if mode_name not in {"off", "memmap", "ram"}:
        raise ValueError(
            "`signature-cache-mode` must be one of {'off', 'memmap', 'ram'}. "
            f"Got {mode}."
        )
    return mode_name


def _normalize_mode_name(mode: NormalizationMode | str) -> str:
    if isinstance(mode, NormalizationMode):
        return mode.value
    return str(mode).upper()


def _stats_to_numpy(stats: dict[str, Any], key: str) -> np.ndarray:
    value = stats.get(key)
    if value is None:
        raise KeyError(f"Normalization stats are missing `{key}`.")
    return np.asarray(value, dtype=np.float32)


def normalize_signature_array(
    values: np.ndarray,
    *,
    stats: dict[str, Any],
    normalization_mode: NormalizationMode | str,
    eps: float,
) -> np.ndarray:
    mode_name = _normalize_mode_name(normalization_mode)
    values = np.asarray(values, dtype=np.float32)
    if mode_name == NormalizationMode.IDENTITY.value:
        return values

    if mode_name == NormalizationMode.MEAN_STD.value:
        mean = _stats_to_numpy(stats, "mean")
        std = _stats_to_numpy(stats, "std")
        return (values - mean) / (std + float(eps))

    if mode_name == NormalizationMode.MIN_MAX.value:
        min_value = _stats_to_numpy(stats, "min")
        max_value = _stats_to_numpy(stats, "max")
        return ((values - min_value) / (max_value - min_value + float(eps))) * 2.0 - 1.0

    if mode_name == NormalizationMode.QUANTILES.value:
        q01 = _stats_to_numpy(stats, "q01")
        q99 = _stats_to_numpy(stats, "q99")
        return ((values - q01) / (q99 - q01 + float(eps))) * 2.0 - 1.0

    if mode_name == NormalizationMode.QUANTILE10.value:
        q10 = _stats_to_numpy(stats, "q10")
        q90 = _stats_to_numpy(stats, "q90")
        return ((values - q10) / (q90 - q10 + float(eps))) * 2.0 - 1.0

    raise ValueError(f"Unsupported normalization mode for signature cache: {mode_name}.")


def _fixed_size_list_column_to_numpy(column, feature_dim: int) -> np.ndarray:
    flattened = column.combine_chunks()
    try:
        values = flattened.values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float32).reshape(len(flattened), feature_dim)
    except Exception:
        return np.asarray(flattened.to_pylist(), dtype=np.float32).reshape(
            len(flattened), feature_dim
        )


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
        "stats_path": str(dataset_root / "meta/stats.json"),
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
    return dataset_root / ".signature_cache" / _sanitize_path_part(dataset_repo_id)


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
    return root / f"signature_cache_v{_CACHE_LAYOUT_VERSION}"


@dataclass
class SignatureCacheReader:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray | torch.Tensor]
    mode: str

    @property
    def feature_keys(self) -> tuple[str, ...]:
        return tuple(self.metadata["feature_keys"])

    @property
    def pre_normalized(self) -> bool:
        return bool(self.metadata.get("pre_normalized", False))

    @property
    def estimated_bytes(self) -> int:
        return int(self.metadata.get("estimated_bytes", 0))

    def has_key(self, key: str) -> bool:
        return key in self.arrays

    def get(self, key: str, absolute_index: int) -> torch.Tensor:
        source = self.arrays[key]
        index = int(absolute_index)
        if isinstance(source, torch.Tensor):
            return source[index]
        return torch.from_numpy(np.array(source[index], copy=True))

    def get_many(self, key: str, absolute_indices: list[int]) -> torch.Tensor:
        source = self.arrays[key]
        if isinstance(source, torch.Tensor):
            return source[torch.as_tensor(absolute_indices, dtype=torch.long)]
        return torch.from_numpy(
            np.array(source[np.asarray(absolute_indices, dtype=np.int64)], copy=True)
        )


@dataclass
class SignatureCacheRuntimeConfig:
    dataset_root: Path
    dataset_repo_id: str
    mode: str
    cache_dtype: str
    feature_keys: tuple[str, ...]
    normalization_mode: NormalizationMode | str
    eps: float = 1e-8
    refresh: bool = False
    cache_root: Path | None = None
    reader: SignatureCacheReader | None = field(default=None, init=False)
    status: str = field(default="off", init=False)
    matched_keys: tuple[str, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root).resolve()
        self.mode = _mode_name(self.mode)
        self.cache_dtype = _npy_dtype_name(self.cache_dtype)
        self.feature_keys = tuple(str(key) for key in self.feature_keys)

    @property
    def enabled(self) -> bool:
        return self.mode != "off" and bool(self.feature_keys)


_SIGNATURE_CACHE_RUNTIME: SignatureCacheRuntimeConfig | None = None


def configure_signature_cache_runtime(
    runtime: SignatureCacheRuntimeConfig | None,
) -> None:
    global _SIGNATURE_CACHE_RUNTIME
    _SIGNATURE_CACHE_RUNTIME = runtime


def get_signature_cache_runtime() -> SignatureCacheRuntimeConfig | None:
    return _SIGNATURE_CACHE_RUNTIME


def prepare_signature_cache_runtime() -> SignatureCacheReader | None:
    runtime = get_signature_cache_runtime()
    if runtime is None or not runtime.enabled:
        return None
    if runtime.reader is not None:
        return runtime.reader
    try:
        runtime.reader = _prepare_signature_cache(runtime)
        runtime.status = "ready"
        runtime.matched_keys = runtime.reader.feature_keys
        return runtime.reader
    except Exception as exc:
        runtime.reader = None
        runtime.status = "fallback"
        runtime.matched_keys = ()
        print(
            "[WARN] signature_cache: failed to prepare cache, falling back to parquet columns. "
            f"reason={exc}"
        )
        return None


def should_use_signature_cache_for_dataset(dataset_root: Path) -> bool:
    runtime = get_signature_cache_runtime()
    if runtime is None or runtime.reader is None:
        return False
    return runtime.dataset_root == Path(dataset_root).resolve()


def signature_cache_feature_keys_for_dataset(dataset_root: Path) -> tuple[str, ...]:
    if not should_use_signature_cache_for_dataset(dataset_root):
        return ()
    runtime = get_signature_cache_runtime()
    assert runtime is not None
    return runtime.reader.feature_keys if runtime.reader is not None else ()


def get_signature_cache_reader_for_dataset(
    dataset_root: Path,
) -> SignatureCacheReader | None:
    if not should_use_signature_cache_for_dataset(dataset_root):
        return None
    runtime = get_signature_cache_runtime()
    assert runtime is not None
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
    if cached_manifest.get("stats_path") != current_manifest.get("stats_path"):
        return False

    current_episodes_path = current_manifest.get("episodes_path")
    current_info_path = current_manifest.get("info_path")
    cached_episodes_path = cached_manifest.get("episodes_path")
    cached_episode_metadata_path = cached_manifest.get("episode_metadata_path")
    legacy_info_backed_episode_path = (
        cached_episodes_path == current_info_path
        or cached_episode_metadata_path == current_info_path
    )
    if not legacy_info_backed_episode_path:
        if (
            cached_episodes_path != current_episodes_path
            and cached_episode_metadata_path != current_episodes_path
        ):
            return False
    if int(cached_manifest.get("total_frames", -1)) != int(current_manifest.get("total_frames", -1)):
        return False

    cached_files = cached_manifest.get("data_files")
    current_files = current_manifest.get("data_files")
    if not isinstance(cached_files, list) or not isinstance(current_files, list):
        return False
    if len(cached_files) != len(current_files):
        return False

    for cached_entry, current_entry in zip(cached_files, current_files, strict=True):
        if not isinstance(cached_entry, dict) or not isinstance(current_entry, dict):
            return False
        if cached_entry.get("path") != current_entry.get("path"):
            return False
        if int(cached_entry.get("size", -1)) != int(current_entry.get("size", -1)):
            return False

    return True


def _metadata_matches_runtime(
    metadata: dict[str, Any],
    *,
    runtime: SignatureCacheRuntimeConfig,
    manifest: dict[str, Any],
) -> bool:
    metadata_keys = tuple(metadata.get("feature_keys", ()))
    return (
        metadata.get("layout_version") == _CACHE_LAYOUT_VERSION
        and metadata.get("dataset_root") == str(runtime.dataset_root)
        and metadata.get("dataset_repo_id") == runtime.dataset_repo_id
        and all(key in metadata_keys for key in runtime.feature_keys)
        and metadata.get("cache_dtype") == runtime.cache_dtype
        and metadata.get("pre_normalized") is True
        and metadata.get("normalization_mode")
        == _normalize_mode_name(runtime.normalization_mode)
        and _manifests_are_compatible(metadata.get("dataset_manifest"), manifest)
    )


def _open_signature_cache_reader(
    cache_dir: Path,
    metadata: dict[str, Any],
    *,
    mode: str,
    feature_keys: tuple[str, ...] | None = None,
) -> SignatureCacheReader:
    keys_to_open = tuple(metadata["feature_keys"] if feature_keys is None else feature_keys)
    arrays: dict[str, np.ndarray | torch.Tensor] = {}
    estimated_bytes = 0
    for key in keys_to_open:
        filename = metadata["files"][key]
        path = cache_dir / filename
        if mode == "ram":
            np_array = np.load(path, mmap_mode="r")
            tensor = torch.from_numpy(np.array(np_array, copy=True)).share_memory_()
            arrays[key] = tensor
            estimated_bytes += int(tensor.numel() * tensor.element_size())
        else:
            mmap_array = np.load(path, mmap_mode="r")
            arrays[key] = mmap_array
            estimated_bytes += int(mmap_array.size * mmap_array.dtype.itemsize)
    reader_metadata = dict(metadata)
    reader_metadata["feature_keys"] = list(keys_to_open)
    reader_metadata["estimated_bytes"] = estimated_bytes
    return SignatureCacheReader(metadata=reader_metadata, arrays=arrays, mode=mode)


def _prepare_signature_cache(runtime: SignatureCacheRuntimeConfig) -> SignatureCacheReader:
    cache_dir = _cache_dir(
        dataset_root=runtime.dataset_root,
        dataset_repo_id=runtime.dataset_repo_id,
        cache_root=runtime.cache_root,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = _build_dataset_manifest(runtime.dataset_root)
    metadata = None if runtime.refresh else _load_cache_metadata(cache_dir)

    if metadata is not None and _metadata_matches_runtime(
        metadata,
        runtime=runtime,
        manifest=manifest,
    ):
        reader = _open_signature_cache_reader(
            cache_dir,
            metadata,
            mode=runtime.mode,
            feature_keys=runtime.feature_keys,
        )
        print(
            "[INFO] signature_cache: hit "
            f"(mode={runtime.mode}, dtype={runtime.cache_dtype}, path={cache_dir})"
        )
        if runtime.mode == "ram":
            print(
                "[INFO] signature_cache: preloaded shared RAM "
                f"({reader.estimated_bytes / (1024 ** 2):.1f} MiB)"
            )
        return reader

    lock_path = cache_dir / ".build.lock"
    with _cache_build_lock(lock_path, label="signature_cache"):
        metadata = None if runtime.refresh else _load_cache_metadata(cache_dir)
        if metadata is not None and _metadata_matches_runtime(
            metadata,
            runtime=runtime,
            manifest=manifest,
        ):
            reader = _open_signature_cache_reader(
                cache_dir,
                metadata,
                mode=runtime.mode,
                feature_keys=runtime.feature_keys,
            )
            print(
                "[INFO] signature_cache: hit "
                f"(mode={runtime.mode}, dtype={runtime.cache_dtype}, path={cache_dir})"
            )
            if runtime.mode == "ram":
                print(
                    "[INFO] signature_cache: preloaded shared RAM "
                    f"({reader.estimated_bytes / (1024 ** 2):.1f} MiB)"
                )
            return reader

        print(
            "[INFO] signature_cache: build "
            f"(mode={runtime.mode}, dtype={runtime.cache_dtype}, path={cache_dir})"
        )
        build_start = dt.datetime.now()
        info = _json_load(runtime.dataset_root / "meta/info.json")
        stats_json = _json_load(runtime.dataset_root / "meta/stats.json")

        files: dict[str, str] = {}
        arrays: dict[str, np.memmap] = {}
        total_bytes = 0

        for key in runtime.feature_keys:
            feature_spec = info.get("features", {}).get(key)
            if feature_spec is None:
                raise KeyError(f"Dataset feature `{key}` was not found in meta/info.json.")
            shape = tuple(int(dim) for dim in feature_spec.get("shape", ()))
            if len(shape) != 1 or shape[0] <= 0:
                raise ValueError(
                    f"Dataset feature `{key}` must have shape [signature_dim]. Got {shape}."
                )
            filename = f"{_sanitize_path_part(key)}.{runtime.cache_dtype}.npy"
            files[key] = filename
            array_path = cache_dir / filename
            arrays[key] = np.lib.format.open_memmap(
                array_path,
                mode="w+",
                dtype=np.dtype(runtime.cache_dtype),
                shape=(manifest["total_frames"], shape[0]),
            )
            total_bytes += int(arrays[key].size * arrays[key].dtype.itemsize)

        seen = np.zeros(manifest["total_frames"], dtype=bool)
        data_files = [runtime.dataset_root / entry["path"] for entry in manifest["data_files"]]
        for parquet_path in data_files:
            table = pq.read_table(parquet_path, columns=["index", *runtime.feature_keys])
            absolute_indices = np.asarray(table.column("index").to_pylist(), dtype=np.int64)
            if absolute_indices.size == 0:
                continue
            seen[absolute_indices] = True
            for key in runtime.feature_keys:
                feature_dim = arrays[key].shape[1]
                raw = _fixed_size_list_column_to_numpy(table.column(key), feature_dim)
                normalized = normalize_signature_array(
                    raw,
                    stats=stats_json[key],
                    normalization_mode=runtime.normalization_mode,
                    eps=runtime.eps,
                )
                arrays[key][absolute_indices] = normalized.astype(runtime.cache_dtype, copy=False)

        if not bool(np.all(seen)):
            missing = int((~seen).sum())
            raise RuntimeError(
                "Signature cache build detected missing absolute frame indices. "
                f"missing={missing}"
            )

        for array in arrays.values():
            array.flush()

        metadata = {
            "layout_version": _CACHE_LAYOUT_VERSION,
            "dataset_root": str(runtime.dataset_root),
            "dataset_repo_id": runtime.dataset_repo_id,
            "feature_keys": list(runtime.feature_keys),
            "feature_shapes": {key: list(arrays[key].shape[1:]) for key in runtime.feature_keys},
            "cache_dtype": runtime.cache_dtype,
            "pre_normalized": True,
            "normalization_mode": _normalize_mode_name(runtime.normalization_mode),
            "eps": float(runtime.eps),
            "files": files,
            "estimated_bytes": total_bytes,
            "built_at": build_start.isoformat(),
            "dataset_manifest": manifest,
        }
        (cache_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        reader = _open_signature_cache_reader(
            cache_dir,
            metadata,
            mode=runtime.mode,
            feature_keys=runtime.feature_keys,
        )
        elapsed = dt.datetime.now() - build_start
        print(
            "[INFO] signature_cache: ready "
            f"(elapsed={elapsed.total_seconds():.1f}s, size={total_bytes / (1024 ** 2):.1f} MiB)"
        )
        if runtime.mode == "ram":
            print(
                "[INFO] signature_cache: preloaded shared RAM "
                f"({reader.estimated_bytes / (1024 ** 2):.1f} MiB)"
            )
        return reader
