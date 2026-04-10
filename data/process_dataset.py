#!/usr/bin/env python3
"""Process LeRobot-style datasets by splitting trajectories and recomputing signatures.

Examples:
    python data/process_dataset.py zeno-ai/day3_5_Exp1 --operations split
    python data/process_dataset.py zeno-ai/day3_5_Exp1 --operations update-signatures
    python data/process_dataset.py zeno-ai/day3_5_Exp1 --operations split update-signatures
    python data/process_dataset.py zeno-ai/day3_5_Exp1 --operations split --in-place
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_helpers import (  # noqa: E402
    compute_delta_signature_sequence_np,
    compute_signatory_signature_np,
    compute_simple_signature_np,
    resolve_signature_backend,
)


DEFAULT_STATE_KEY = "observation.state"
DEFAULT_PATH_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_DELTA_SIGNATURE_KEY = "observation.delta_signature"
DEFAULT_SIGNATURE_WINDOW_SIZE = 0
DEFAULT_SIGNATURE_DEPTH = 3
DEFAULT_SIGNATURE_BACKEND = "auto"
DEFAULT_COPY_SUFFIX = "_processed"
DEFAULT_LOCAL_DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"
SKIPPED_ROOT_COPY_NAMES = frozenset({".cache", "__pycache__"})


def require_pyarrow_dependencies():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "This script requires `pyarrow` to read and write LeRobot parquet files. "
            "Install it in the current environment first."
        ) from exc
    return pa, pq


def require_hf_image_dependencies():
    try:
        import datasets
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Embedding decoded video frames into parquet image columns requires "
            "`datasets` from Hugging Face. Install it in the current environment first."
        ) from exc
    return datasets


def maybe_import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Downloading datasets requires `huggingface_hub`. "
            "Install it first, or pre-download the dataset under the local data root."
        ) from exc
    return snapshot_download


def require_binary(binary_name: str) -> None:
    if shutil.which(binary_name) is None:
        raise RuntimeError(
            f"Required binary `{binary_name}` was not found in PATH. "
            "Install it before running this dataset processing mode."
        )


def maybe_import_tqdm():
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm


class SimpleProgressBar:
    def __init__(
        self,
        total: int,
        *,
        desc: str,
        unit: str = "item",
        width: int = 28,
    ) -> None:
        self.total = max(0, int(total))
        self.desc = str(desc)
        self.unit = str(unit)
        self.width = max(10, int(width))
        self.count = 0
        self._closed = False
        self._render()

    def _format_bar(self) -> str:
        if self.total <= 0:
            return "[" + ("-" * self.width) + "]"
        filled = min(
            self.width,
            int(round((self.count / max(self.total, 1)) * self.width)),
        )
        return "[" + ("#" * filled) + ("-" * (self.width - filled)) + "]"

    def _render(self) -> None:
        if self._closed:
            return
        if self.total > 0:
            progress = min(self.count, self.total)
            percent = (100.0 * progress) / self.total
            suffix = (
                f"{self._format_bar()} {progress}/{self.total} "
                f"{self.unit} ({percent:5.1f}%)"
            )
        else:
            suffix = f"{self._format_bar()} 0/0 {self.unit}"
        end = "\n" if self.total > 0 and self.count >= self.total else ""
        print(f"\r{self.desc}: {suffix}", end=end, flush=True)

    def update(self, n: int = 1) -> None:
        if self._closed:
            return
        self.count += int(n)
        self._render()

    def close(self) -> None:
        if self._closed:
            return
        if self.total > 0 and self.count < self.total:
            self.count = self.total
            self._render()
        elif self.total <= 0:
            print("", flush=True)
        self._closed = True

    def __enter__(self) -> "SimpleProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def iter_with_progress(
    items: Sequence[Any],
    *,
    desc: str,
    unit: str = "item",
) -> Iterator[Any]:
    total = len(items)
    tqdm = maybe_import_tqdm()

    if tqdm is not None:
        yield from tqdm(items, total=total, desc=desc, unit=unit, dynamic_ncols=True)
        return

    if total == 0:
        return

    with SimpleProgressBar(total, desc=desc, unit=unit) as progress:
        for item in items:
            yield item
            progress.update(1)


def iter_completed_futures(
    futures: Sequence[cf.Future],
    *,
    desc: str,
    unit: str = "item",
) -> Iterator[cf.Future]:
    total = len(futures)
    tqdm = maybe_import_tqdm()

    if tqdm is not None:
        with tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True) as progress:
            for future in cf.as_completed(futures):
                progress.update(1)
                yield future
        return

    if total == 0:
        return

    with SimpleProgressBar(total, desc=desc, unit=unit) as progress:
        for future in cf.as_completed(futures):
            progress.update(1)
            yield future


def is_lerobot_dataset_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "meta/info.json").exists()
        and (path / DEFAULT_EPISODES_PATH).exists()
    )


def find_lerobot_dataset_root(candidate: Path) -> Path | None:
    raw = candidate.expanduser()
    if is_lerobot_dataset_root(raw):
        return raw.resolve()

    if not raw.exists() or not raw.is_dir():
        return None

    for info_path in sorted(raw.rglob("meta/info.json")):
        root = info_path.parent.parent
        if is_lerobot_dataset_root(root):
            return root.resolve()
    return None


def resolve_source_dataset_root(
    dataset_id: str,
    *,
    local_data_root: Path,
    download_if_missing: bool,
    cache_dir: Path | None,
    repo_type: str,
) -> Path:
    candidates: list[Path] = []
    raw_path = Path(dataset_id).expanduser()
    if raw_path.exists():
        candidates.append(raw_path)

    candidates.append(local_data_root / dataset_id.replace("/", "_"))
    candidates.append(local_data_root / dataset_id)

    for candidate in candidates:
        found = find_lerobot_dataset_root(candidate)
        if found is not None:
            return found

    if not download_if_missing:
        candidate_text = "\n".join(f"- {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "Could not resolve a local LeRobot dataset root for "
            f"`{dataset_id}`. Checked:\n{candidate_text}"
        )

    snapshot_download = maybe_import_snapshot_download()
    target_dir = local_data_root / dataset_id.replace("/", "_")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset {dataset_id} to {target_dir}")
    snapshot_download(
        repo_id=dataset_id,
        repo_type=repo_type,
        local_dir=str(target_dir),
        cache_dir=None if cache_dir is None else str(cache_dir),
        resume_download=True,
        max_workers=4,
    )
    found = find_lerobot_dataset_root(target_dir)
    if found is None:
        raise FileNotFoundError(
            f"Downloaded {dataset_id}, but no LeRobot dataset root was found under {target_dir}."
        )
    return found


def copy_dataset_without_data(
    source_root: Path,
    target_root: Path,
    *,
    include_videos: bool,
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    children_to_copy: list[Path] = []
    skipped_children: list[str] = []
    for child in sorted(source_root.iterdir(), key=lambda path: path.name):
        if child.name == "data":
            skipped_children.append(child.name)
            continue
        if child.name == "videos" and not include_videos:
            skipped_children.append(child.name)
            continue
        if child.name in SKIPPED_ROOT_COPY_NAMES:
            skipped_children.append(child.name)
            continue
        children_to_copy.append(child)

    print(
        "Preparing output root: "
        f"copying {len(children_to_copy)} entries from {source_root}"
    )
    if skipped_children:
        print(
            "Skipping source root entries that are regenerated or cache-only: "
            + ", ".join(skipped_children)
        )

    for child in iter_with_progress(
        children_to_copy,
        desc="Setup",
        unit="entry",
    ):
        destination = target_root / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


def prepare_target_root(
    source_root: Path,
    *,
    in_place: bool,
    output_dir: Path | None,
    copy_suffix: str,
    overwrite_output: bool,
    include_videos: bool,
) -> Path:
    if in_place:
        return source_root

    target_root = (
        output_dir.expanduser()
        if output_dir is not None
        else source_root.parent / f"{source_root.name}{copy_suffix}"
    )
    target_root = target_root.resolve()
    source_resolved = source_root.resolve()

    if target_root == source_resolved:
        raise ValueError(
            "`--output-dir` resolves to the source dataset root. "
            "Use `--in-place` if you want to overwrite the source dataset."
        )
    if source_resolved in target_root.parents:
        raise ValueError(
            "Output directory cannot live inside the source dataset root. "
            "Choose a sibling directory or use `--in-place`."
        )
    if target_root.exists():
        if not overwrite_output:
            raise FileExistsError(
                f"Output directory already exists: {target_root}. "
                "Pass `--overwrite-output` to replace it."
            )
        shutil.rmtree(target_root)

    copy_dataset_without_data(
        source_root,
        target_root,
        include_videos=include_videos,
    )
    return target_root


def clear_processing_outputs(dataset_root: Path) -> None:
    data_dir = dataset_root / "data"
    episodes_dir = dataset_root / "meta" / "episodes"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    if episodes_dir.exists():
        shutil.rmtree(episodes_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)


def copy_processed_outputs_back(
    staging_root: Path,
    destination_root: Path,
    *,
    include_videos: bool,
) -> None:
    destination_meta_dir = destination_root / "meta"
    destination_meta_dir.mkdir(parents=True, exist_ok=True)

    destination_data_dir = destination_root / "data"
    destination_episodes_dir = destination_meta_dir / "episodes"
    if destination_data_dir.exists():
        shutil.rmtree(destination_data_dir)
    if destination_episodes_dir.exists():
        shutil.rmtree(destination_episodes_dir)

    shutil.copytree(staging_root / "data", destination_data_dir)
    shutil.copytree(staging_root / "meta" / "episodes", destination_episodes_dir)
    shutil.copy2(staging_root / "meta" / "info.json", destination_meta_dir / "info.json")
    shutil.copy2(staging_root / "meta" / "stats.json", destination_meta_dir / "stats.json")
    if include_videos:
        destination_videos_dir = destination_root / "videos"
        if destination_videos_dir.exists():
            shutil.rmtree(destination_videos_dir)
        shutil.copytree(staging_root / "videos", destination_videos_dir)


def parse_split_spec(split_spec: str) -> tuple[int, int]:
    start_text, end_text = split_spec.split(":", 1)
    return int(start_text), int(end_text)


def build_source_split_lookup(
    source_splits: dict[str, str],
    total_source_episodes: int,
) -> tuple[list[str], list[str]]:
    if not source_splits:
        return ["train"] * total_source_episodes, ["train"]

    lookup: list[str | None] = [None] * total_source_episodes
    split_order = list(source_splits)
    for split_name, split_spec in source_splits.items():
        split_from, split_to = parse_split_spec(split_spec)
        if not (0 <= split_from <= split_to <= total_source_episodes):
            raise ValueError(f"Invalid split range for {split_name}: {split_spec}")
        for episode_idx in range(split_from, split_to):
            if lookup[episode_idx] is not None:
                raise ValueError(
                    f"Episode {episode_idx} belongs to multiple splits. "
                    f"Existing={lookup[episode_idx]}, new={split_name}"
                )
            lookup[episode_idx] = split_name

    if any(value is None for value in lookup):
        missing = [idx for idx, value in enumerate(lookup) if value is None]
        raise ValueError(
            "Source split definitions do not cover all episodes. Missing indices: "
            + ", ".join(str(idx) for idx in missing[:20])
        )
    return [str(value) for value in lookup], split_order


def build_output_splits(
    output_split_assignments: Sequence[str],
    split_order: Sequence[str],
) -> dict[str, str]:
    if not output_split_assignments:
        return {split_order[0]: "0:0"} if split_order else {"train": "0:0"}

    split_counts = {split_name: 0 for split_name in split_order}
    for split_name in output_split_assignments:
        if split_name not in split_counts:
            raise ValueError(f"Unknown output split name: {split_name}")
        split_counts[split_name] += 1

    cursor = 0
    output_splits: dict[str, str] = {}
    for split_name in split_order:
        count = split_counts[split_name]
        output_splits[split_name] = f"{cursor}:{cursor + count}"
        cursor += count
    return output_splits


def get_chunk_and_file_index(
    episode_index: int,
    episodes_per_chunk: int,
) -> tuple[int, int]:
    return (
        int(episode_index // episodes_per_chunk),
        int(episode_index % episodes_per_chunk),
    )


def collect_chunk_file_indices(
    source_episodes_meta: Sequence[dict[str, Any]],
) -> dict[int, list[int]]:
    chunk_to_file_indices: dict[int, set[int]] = {}
    for episode_meta in source_episodes_meta:
        chunk_index = int(episode_meta["data/chunk_index"])
        file_index = int(episode_meta["data/file_index"])
        chunk_to_file_indices.setdefault(chunk_index, set()).add(file_index)
    return {
        chunk_index: sorted(file_indices)
        for chunk_index, file_indices in chunk_to_file_indices.items()
    }


def infer_source_episodes_per_chunk(
    source_episodes_meta: Sequence[dict[str, Any]],
) -> int | None:
    chunk_file_indices = collect_chunk_file_indices(source_episodes_meta)
    if not chunk_file_indices:
        return None
    return max(len(file_indices) for file_indices in chunk_file_indices.values())


def is_declared_chunk_layout_consistent(
    source_episodes_meta: Sequence[dict[str, Any]],
    declared_episodes_per_chunk: int,
) -> bool:
    if declared_episodes_per_chunk <= 0:
        return False

    chunk_file_indices = collect_chunk_file_indices(source_episodes_meta)
    if not chunk_file_indices:
        return False

    ordered_chunk_indices = sorted(chunk_file_indices)
    for position, chunk_index in enumerate(ordered_chunk_indices):
        file_indices = chunk_file_indices[chunk_index]
        if file_indices != list(range(len(file_indices))):
            return False

        file_count = len(file_indices)
        is_last_chunk = position == len(ordered_chunk_indices) - 1
        if file_count > declared_episodes_per_chunk:
            return False
        if not is_last_chunk and file_count != declared_episodes_per_chunk:
            return False
    return True


def resolve_episodes_per_chunk(
    *,
    requested_episodes_per_chunk: int | None,
    source_info: dict[str, Any],
    source_episodes_meta: Sequence[dict[str, Any]],
) -> int:
    if requested_episodes_per_chunk is not None:
        resolved = int(requested_episodes_per_chunk)
        if resolved <= 0:
            raise ValueError("`episodes_per_chunk` must be positive.")
        return resolved

    declared_raw = source_info.get("chunks_size")
    declared = None if declared_raw is None else int(declared_raw)
    if declared is not None and is_declared_chunk_layout_consistent(source_episodes_meta, declared):
        return declared

    inferred = infer_source_episodes_per_chunk(source_episodes_meta)
    if inferred is not None and inferred > 0:
        if declared is not None and declared != inferred:
            print(
                "[info] Source metadata chunks_size does not match the observed data layout. "
                f"declared={declared}, inferred={inferred}. Using inferred value."
            )
        return inferred

    fallback = 1000
    if declared is not None and declared > 0:
        return declared
    return fallback


def estimate_total_size_mb(paths: Sequence[Path]) -> int:
    if not paths:
        return 0
    total_bytes = sum(path.stat().st_size for path in paths if path.exists())
    return int(round(total_bytes / (1024 * 1024)))


def build_stats(values: Sequence[Any]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def compute_path_signature_sequence(
    states: Sequence[Sequence[float]],
    *,
    window_size: int,
    sig_depth: int,
    signature_backend: str,
) -> np.ndarray:
    states_array = np.asarray(states, dtype=np.float32)
    if states_array.ndim != 2:
        raise ValueError(
            "Expected state trajectory with shape (T, state_dim). "
            f"Got {tuple(states_array.shape)}."
        )
    if states_array.shape[0] == 0:
        raise ValueError("State trajectory must contain at least one step.")
    if sig_depth <= 0:
        raise ValueError(f"`sig_depth` must be positive, got {sig_depth}.")

    outputs: list[np.ndarray] = []
    signature_dim: int | None = None
    for end_idx in range(int(states_array.shape[0])):
        start_idx = 0 if window_size <= 0 else max(0, end_idx + 1 - int(window_size))
        window = states_array[start_idx : end_idx + 1]
        if signature_backend == "signatory":
            signature = compute_signatory_signature_np(window, sig_depth)
        else:
            signature = compute_simple_signature_np(window, sig_depth)
        if signature_dim is None:
            signature_dim = int(signature.shape[0])
        elif int(signature.shape[0]) != signature_dim:
            raise RuntimeError(
                "Path signature dimension changed across timesteps. "
                f"Expected {signature_dim}, got {int(signature.shape[0])}."
            )
        outputs.append(signature.astype(np.float32, copy=False))
    return np.stack(outputs, axis=0)


def normalize_numeric_batch(values: Sequence[Any]) -> np.ndarray | None:
    if not values:
        return None
    array = np.asarray(values)
    if array.dtype.kind not in {"i", "u", "f"}:
        return None
    array = array.astype(np.float64)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[:, None]
    return array


@dataclass
class NumericStatsAccumulator:
    sum_values: np.ndarray
    sumsq_values: np.ndarray
    min_values: np.ndarray
    max_values: np.ndarray
    count: int

    @classmethod
    def from_batch(cls, batch: np.ndarray) -> "NumericStatsAccumulator":
        return cls(
            sum_values=batch.sum(axis=0),
            sumsq_values=np.square(batch).sum(axis=0),
            min_values=batch.min(axis=0),
            max_values=batch.max(axis=0),
            count=int(batch.shape[0]),
        )

    def update(self, batch: np.ndarray) -> None:
        self.sum_values += batch.sum(axis=0)
        self.sumsq_values += np.square(batch).sum(axis=0)
        self.min_values = np.minimum(self.min_values, batch.min(axis=0))
        self.max_values = np.maximum(self.max_values, batch.max(axis=0))
        self.count += int(batch.shape[0])

    def merge(self, other: "NumericStatsAccumulator") -> None:
        self.sum_values += other.sum_values
        self.sumsq_values += other.sumsq_values
        self.min_values = np.minimum(self.min_values, other.min_values)
        self.max_values = np.maximum(self.max_values, other.max_values)
        self.count += int(other.count)

    def finalize(self) -> dict[str, Any]:
        mean = self.sum_values / self.count
        variance = self.sumsq_values / self.count - np.square(mean)
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)
        return {
            "min": self.min_values.tolist(),
            "max": self.max_values.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "count": [int(self.count)],
        }


def update_numeric_accumulators(
    accumulators: dict[str, NumericStatsAccumulator],
    frame_columns: dict[str, list[Any]],
    feature_specs: dict[str, dict[str, Any]] | None = None,
) -> None:
    for key, values in frame_columns.items():
        if feature_specs is not None:
            spec = feature_specs.get(key)
            if isinstance(spec, dict) and spec.get("dtype") in {"image", "video"}:
                continue
        batch = normalize_numeric_batch(values)
        if batch is None:
            continue
        if key not in accumulators:
            accumulators[key] = NumericStatsAccumulator.from_batch(batch)
        else:
            accumulators[key].update(batch)


def merge_numeric_accumulators(
    destination: dict[str, NumericStatsAccumulator],
    source: dict[str, NumericStatsAccumulator],
) -> None:
    for key, accumulator in source.items():
        if key not in destination:
            destination[key] = accumulator
        else:
            destination[key].merge(accumulator)


class EpisodeDataCache:
    def __init__(self, pq_module: Any) -> None:
        self._pq = pq_module
        self._current_path: Path | None = None
        self._current_columns: dict[str, list[Any]] | None = None
        self._current_schema = None

    def load(self, data_file: Path) -> tuple[dict[str, list[Any]], Any]:
        if self._current_path != data_file:
            table = self._pq.read_table(data_file)
            self._current_columns = table.to_pydict()
            self._current_schema = table.schema
            self._current_path = data_file
        assert self._current_columns is not None
        return self._current_columns, self._current_schema


def build_data_file_path(
    dataset_root: Path,
    data_path_pattern: str,
    *,
    chunk_index: int,
    file_index: int,
) -> Path:
    return dataset_root / data_path_pattern.format(
        chunk_index=int(chunk_index),
        file_index=int(file_index),
    )


def build_video_file_path(
    dataset_root: Path,
    video_path_pattern: str,
    *,
    video_key: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    return dataset_root / video_path_pattern.format(
        video_key=video_key,
        chunk_index=int(chunk_index),
        file_index=int(file_index),
    )


def ffprobe_video_info(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]

    raw_rate = str(stream.get("avg_frame_rate", "0/1"))
    if "/" in raw_rate:
        numerator, denominator = raw_rate.split("/", 1)
        fps = float(numerator) / max(float(denominator), 1.0)
    else:
        fps = float(raw_rate)

    raw_frames = stream.get("nb_frames")
    if raw_frames in (None, "N/A"):
        duration = float(stream.get("duration", 0.0))
        frame_count = int(round(duration * fps))
    else:
        frame_count = int(raw_frames)

    return {
        "codec": str(stream.get("codec_name", "unknown")),
        "pix_fmt": str(stream.get("pix_fmt", "unknown")),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": float(fps),
        "frames": int(frame_count),
        "duration": float(stream.get("duration", 0.0)),
    }


def inspect_available_video_keys(
    *,
    source_root: Path,
    video_path_pattern: str,
    source_episodes_meta: Sequence[dict[str, Any]],
    video_keys: Sequence[str],
) -> tuple[list[str], dict[str, list[Path]]]:
    available_keys: list[str] = []
    missing_paths_by_key: dict[str, list[Path]] = {}

    for video_key in video_keys:
        unique_refs = sorted(
            {
                (
                    int(episode[f"videos/{video_key}/chunk_index"]),
                    int(episode[f"videos/{video_key}/file_index"]),
                )
                for episode in source_episodes_meta
                if f"videos/{video_key}/chunk_index" in episode
                and f"videos/{video_key}/file_index" in episode
            }
        )
        missing_paths: list[Path] = []
        for chunk_index, file_index in unique_refs:
            candidate = build_video_file_path(
                source_root,
                video_path_pattern,
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if not candidate.exists():
                missing_paths.append(candidate)
        if missing_paths:
            missing_paths_by_key[video_key] = missing_paths
        else:
            available_keys.append(video_key)

    return available_keys, missing_paths_by_key


def slice_video_segment(
    *,
    source_video: Path,
    target_video: Path,
    start_frame: int,
    num_frames: int,
) -> dict[str, Any]:
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    if start_frame < 0:
        raise ValueError(f"`start_frame` must be non-negative, got {start_frame}.")

    target_video.parent.mkdir(parents=True, exist_ok=True)
    end_frame = start_frame + num_frames
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-i",
        str(source_video),
        "-vf",
        f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(target_video),
    ]
    subprocess.run(command, check=True)
    video_info = ffprobe_video_info(target_video)
    if int(video_info["frames"]) != int(num_frames):
        raise RuntimeError(
            "Sliced video frame count mismatch. "
            f"Expected {num_frames}, got {video_info['frames']} for {target_video}."
        )
    return video_info


def extract_episode_columns(
    cache: EpisodeDataCache,
    data_file: Path,
    episode_meta: dict[str, Any],
) -> tuple[dict[str, list[Any]], Any]:
    columns, schema = cache.load(data_file)
    if "index" not in columns:
        raise KeyError(f"`index` column is missing from data file {data_file}")

    start_index = int(episode_meta["dataset_from_index"])
    end_index = int(episode_meta["dataset_to_index"])
    expected_length = end_index - start_index
    file_indices = [int(value) for value in columns["index"]]

    if start_index in file_indices:
        start_pos = file_indices.index(start_index)
        end_pos = start_pos + expected_length
    elif "episode_index" in columns:
        source_episode_index = int(episode_meta["episode_index"])
        positions = [
            pos
            for pos, value in enumerate(columns["episode_index"])
            if int(value) == source_episode_index
        ]
        if not positions:
            raise KeyError(
                f"Could not locate source episode {source_episode_index} in {data_file}"
            )
        start_pos = positions[0]
        end_pos = positions[-1] + 1
    else:
        raise KeyError(
            "Could not locate episode rows in the source data file because both "
            "`index` lookup and `episode_index` fallback failed."
        )

    episode_columns = {
        key: values[start_pos:end_pos]
        for key, values in columns.items()
    }
    if len(episode_columns["index"]) != expected_length:
        raise ValueError(
            f"Episode slice length mismatch for data file {data_file}. "
            f"Expected {expected_length}, got {len(episode_columns['index'])}."
        )
    return episode_columns, schema


def compute_split_ranges(
    episode_columns: dict[str, list[Any]],
    *,
    split_strategy: str,
) -> list[tuple[int, int]]:
    num_steps = len(episode_columns["index"])
    if num_steps == 0:
        return []
    if split_strategy == "episode_index":
        return [(0, num_steps)]

    done_values = [bool(value) for value in episode_columns.get("next.done", [False] * num_steps)]
    frame_index_values = (
        None
        if "frame_index" not in episode_columns
        else [int(value) for value in episode_columns["frame_index"]]
    )
    timestamp_values = (
        None
        if "timestamp" not in episode_columns
        else [float(value) for value in episode_columns["timestamp"]]
    )

    boundaries = [0]
    for idx in range(num_steps - 1):
        should_split = False
        if done_values[idx]:
            should_split = True
        if split_strategy == "auto":
            if (
                frame_index_values is not None
                and frame_index_values[idx + 1] <= frame_index_values[idx]
            ):
                should_split = True
            if (
                timestamp_values is not None
                and timestamp_values[idx + 1] < timestamp_values[idx]
            ):
                should_split = True
        if should_split and boundaries[-1] != idx + 1:
            boundaries.append(idx + 1)

    if boundaries[-1] != num_steps:
        boundaries.append(num_steps)

    ranges = [
        (boundaries[pos], boundaries[pos + 1])
        for pos in range(len(boundaries) - 1)
        if boundaries[pos] < boundaries[pos + 1]
    ]
    return ranges or [(0, num_steps)]


def slice_frame_columns(
    frame_columns: dict[str, list[Any]],
    start: int,
    end: int,
) -> dict[str, list[Any]]:
    return {key: values[start:end] for key, values in frame_columns.items()}


def normalize_tasks(raw_tasks: Any) -> list[str]:
    if raw_tasks is None:
        return []
    if isinstance(raw_tasks, list):
        return [str(value) for value in raw_tasks]
    return [str(raw_tasks)]


def insert_after(column_order: list[str], target: str, anchor: str) -> None:
    if target in column_order:
        return
    if anchor in column_order:
        column_order.insert(column_order.index(anchor) + 1, target)
    else:
        column_order.append(target)


@dataclass(frozen=True)
class SignaturePlan:
    path_key: str
    delta_key: str
    state_key: str
    window_size: int
    depth: int
    resolved_backend: str
    write_path: bool
    write_delta: bool


def build_signature_plan(
    *,
    operations: Sequence[str],
    source_info: dict[str, Any],
    source_field_names: Sequence[str],
    path_key: str,
    delta_key: str,
    state_key: str,
    requested_signature_type: str,
    requested_window_size: int,
    requested_depth: int,
    requested_backend: str,
) -> SignaturePlan | None:
    source_path_meta = source_info.get("path_signature", {})
    source_delta_meta = source_info.get("delta_signature", {})
    source_path_meta_key = source_path_meta.get("key")
    source_delta_meta_key = source_delta_meta.get("key")
    if source_path_meta_key and str(source_path_meta_key) != path_key:
        raise ValueError(
            "The source dataset uses a different path-signature key. "
            f"Expected `{path_key}`, found `{source_path_meta_key}`. "
            "Pass `--path-signature-key` with the source key to avoid stale columns."
        )
    if source_delta_meta_key and str(source_delta_meta_key) != delta_key:
        raise ValueError(
            "The source dataset uses a different delta-signature key. "
            f"Expected `{delta_key}`, found `{source_delta_meta_key}`. "
            "Pass `--delta-signature-key` with the source key to avoid stale columns."
        )

    source_has_path = path_key in source_field_names or path_key in source_info.get("features", {})
    source_has_delta = delta_key in source_field_names or delta_key in source_info.get("features", {})
    split_requested = "split" in operations
    update_requested = "update-signatures" in operations
    wants_path = update_requested and requested_signature_type in {"path", "both"}
    wants_delta = update_requested and requested_signature_type in {"delta", "both"}

    write_path = bool(source_has_path or source_has_delta or wants_path or wants_delta)
    write_delta = bool(source_has_delta or wants_delta)
    if not write_path and not write_delta:
        return None

    if update_requested:
        window_size = int(requested_window_size)
        depth = int(requested_depth)
        backend = resolve_signature_backend(requested_backend)
    elif split_requested and (source_has_path or source_has_delta):
        window_size = int(source_path_meta.get("window_size", DEFAULT_SIGNATURE_WINDOW_SIZE))
        depth = int(source_path_meta.get("sig_depth", DEFAULT_SIGNATURE_DEPTH))
        source_backend = str(
            source_path_meta.get("backend", DEFAULT_SIGNATURE_BACKEND)
        )
        if requested_backend != DEFAULT_SIGNATURE_BACKEND:
            backend_request = requested_backend
        elif source_backend == "signatory":
            # Keep split-only processing usable on machines where signatory is
            # unavailable or broken by letting auto-fallback pick `simple`.
            backend_request = DEFAULT_SIGNATURE_BACKEND
        else:
            backend_request = source_backend
        backend = resolve_signature_backend(backend_request)
    else:
        return None

    return SignaturePlan(
        path_key=path_key,
        delta_key=delta_key,
        state_key=state_key,
        window_size=window_size,
        depth=depth,
        resolved_backend=backend,
        write_path=write_path,
        write_delta=write_delta,
    )


def compute_segment_video_timestamps(
    *,
    source_episode_meta: dict[str, Any],
    source_episode_columns: dict[str, list[Any]],
    segment_start: int,
    segment_length: int,
    video_key: str,
    fps: int,
) -> tuple[float, float]:
    from_key = f"videos/{video_key}/from_timestamp"
    to_key = f"videos/{video_key}/to_timestamp"
    source_video_from = float(source_episode_meta.get(from_key, 0.0))
    source_video_to = float(source_episode_meta.get(to_key, source_video_from))
    source_episode_length = len(source_episode_columns["index"])

    if "timestamp" in source_episode_columns and source_episode_columns["timestamp"]:
        source_timestamps = [float(value) for value in source_episode_columns["timestamp"]]
        episode_origin = source_timestamps[0]
        offset = source_timestamps[segment_start] - episode_origin
    else:
        offset = segment_start / max(int(fps), 1)

    duration = source_video_to - source_video_from
    if source_episode_length > 0 and duration > 0:
        frame_period = duration / source_episode_length
    else:
        frame_period = 1.0 / max(int(fps), 1)

    segment_video_from = source_video_from + offset
    segment_video_to = segment_video_from + frame_period * segment_length
    return float(segment_video_from), float(segment_video_to)


def validate_processed_dataset(
    episodes_meta: Sequence[dict[str, Any]],
    splits: dict[str, str],
    total_frames: int,
    total_episodes: int,
) -> None:
    if total_episodes != len(episodes_meta):
        raise ValueError("Episode count mismatch after processing.")

    cursor = 0
    for expected_episode_index, episode in enumerate(episodes_meta):
        if int(episode["episode_index"]) != expected_episode_index:
            raise ValueError(
                "Output episodes must be sequentially indexed. "
                f"Expected {expected_episode_index}, got {episode['episode_index']}."
            )
        episode_length = int(episode["length"])
        if int(episode["dataset_from_index"]) != cursor:
            raise ValueError(
                "Non-contiguous dataset_from_index in processed episodes. "
                f"Expected {cursor}, got {episode['dataset_from_index']}."
            )
        cursor += episode_length
        if int(episode["dataset_to_index"]) != cursor:
            raise ValueError(
                "Invalid dataset_to_index in processed episodes. "
                f"Expected {cursor}, got {episode['dataset_to_index']}."
            )
    if cursor != total_frames:
        raise ValueError(
            f"Total frame mismatch after validation. Expected {total_frames}, got {cursor}."
        )

    for split_name, split_spec in splits.items():
        split_from, split_to = parse_split_spec(split_spec)
        if not (0 <= split_from <= split_to <= total_episodes):
            raise ValueError(f"Invalid split range for {split_name}: {split_spec}")


def build_arrow_table(
    pa_module: Any,
    *,
    frame_columns: dict[str, list[Any]],
    column_order: Sequence[str],
    field_types: dict[str, Any],
):
    arrays = []
    names = []
    for name in column_order:
        if name not in frame_columns:
            continue
        field_type = field_types.get(name)
        if field_type is None:
            arrays.append(pa_module.array(frame_columns[name]))
        else:
            arrays.append(pa_module.array(frame_columns[name], type=field_type))
        names.append(name)
    return pa_module.Table.from_arrays(arrays, names=names)


def get_hf_features_from_lerobot_features(feature_specs: dict[str, dict[str, Any]]):
    datasets = require_hf_image_dependencies()
    hf_features = {}
    for key, spec in feature_specs.items():
        dtype = str(spec["dtype"])
        shape = tuple(spec.get("shape", ()))
        if dtype == "video":
            continue
        if dtype == "image":
            hf_features[key] = datasets.Image()
        elif shape == (1,):
            hf_features[key] = datasets.Value(dtype=dtype)
        elif len(shape) == 1:
            hf_features[key] = datasets.Sequence(
                length=int(shape[0]),
                feature=datasets.Value(dtype=dtype),
            )
        elif len(shape) == 2:
            hf_features[key] = datasets.Array2D(shape=shape, dtype=dtype)
        elif len(shape) == 3:
            hf_features[key] = datasets.Array3D(shape=shape, dtype=dtype)
        elif len(shape) == 4:
            hf_features[key] = datasets.Array4D(shape=shape, dtype=dtype)
        elif len(shape) == 5:
            hf_features[key] = datasets.Array5D(shape=shape, dtype=dtype)
        else:
            raise ValueError(f"Unsupported feature spec for parquet writing: key={key}, spec={spec}")
    return datasets.Features(hf_features)


def write_frame_columns_with_hf_images(
    *,
    frame_columns: dict[str, list[Any]],
    output_path: Path,
    feature_specs: dict[str, dict[str, Any]],
    column_order: Sequence[str],
) -> None:
    datasets = require_hf_image_dependencies()
    ordered_columns = {
        name: frame_columns[name]
        for name in column_order
        if name in frame_columns
    }
    ordered_features = {
        name: feature_specs[name]
        for name in ordered_columns
        if name in feature_specs
    }
    hf_features = get_hf_features_from_lerobot_features(ordered_features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets.Dataset.from_dict(
        ordered_columns,
        features=hf_features,
        split="train",
    )
    dataset.to_parquet(str(output_path))


def write_frame_columns(
    *,
    pa_module: Any,
    pq_module: Any,
    frame_columns: dict[str, list[Any]],
    output_path: Path,
    feature_specs: dict[str, dict[str, Any]],
    column_order: Sequence[str],
    field_types: dict[str, Any],
) -> None:
    has_image_feature = any(
        name in frame_columns and feature_specs.get(name, {}).get("dtype") == "image"
        for name in column_order
    )
    if has_image_feature:
        write_frame_columns_with_hf_images(
            frame_columns=frame_columns,
            output_path=output_path,
            feature_specs=feature_specs,
            column_order=column_order,
        )
        return

    output_table = build_arrow_table(
        pa_module,
        frame_columns=frame_columns,
        column_order=column_order,
        field_types=field_types,
    )
    pq_module.write_table(output_table, output_path, compression="snappy")


def decode_video_segment_frames(
    *,
    source_video: Path,
    start_frame: int,
    num_frames: int,
    video_info_cache: dict[Path, dict[str, Any]],
) -> list[np.ndarray]:
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    if start_frame < 0:
        raise ValueError(f"`start_frame` must be non-negative, got {start_frame}.")

    video_info = video_info_cache.get(source_video)
    if video_info is None:
        video_info = ffprobe_video_info(source_video)
        video_info_cache[source_video] = video_info

    width = int(video_info["width"])
    height = int(video_info["height"])
    end_frame = int(start_frame + num_frames)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-i",
        str(source_video),
        "-vf",
        f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
    )
    frame_bytes = width * height * 3
    raw = completed.stdout
    if len(raw) != num_frames * frame_bytes:
        actual_frames = 0 if frame_bytes == 0 else len(raw) // frame_bytes
        raise RuntimeError(
            "Decoded frame count mismatch while embedding video frames into parquet. "
            f"Expected {num_frames}, got {actual_frames} for {source_video}."
        )

    frames = np.frombuffer(raw, dtype=np.uint8).reshape(num_frames, height, width, 3)
    return [np.ascontiguousarray(frame) for frame in frames]


def write_episodes_table(
    pa_module: Any,
    pq_module: Any,
    *,
    episodes_meta: Sequence[dict[str, Any]],
    output_path: Path,
    video_keys: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = [
        pa_module.array([int(ep["episode_index"]) for ep in episodes_meta], type=pa_module.int64()),
        pa_module.array([normalize_tasks(ep.get("tasks")) for ep in episodes_meta], type=pa_module.list_(pa_module.string())),
        pa_module.array([int(ep["length"]) for ep in episodes_meta], type=pa_module.int64()),
        pa_module.array([int(ep["data/chunk_index"]) for ep in episodes_meta], type=pa_module.int64()),
        pa_module.array([int(ep["data/file_index"]) for ep in episodes_meta], type=pa_module.int64()),
        pa_module.array([int(ep["dataset_from_index"]) for ep in episodes_meta], type=pa_module.int64()),
        pa_module.array([int(ep["dataset_to_index"]) for ep in episodes_meta], type=pa_module.int64()),
    ]
    names = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]

    for video_key in video_keys:
        arrays.extend(
            [
                pa_module.array(
                    [int(ep[f"videos/{video_key}/chunk_index"]) for ep in episodes_meta],
                    type=pa_module.int64(),
                ),
                pa_module.array(
                    [int(ep[f"videos/{video_key}/file_index"]) for ep in episodes_meta],
                    type=pa_module.int64(),
                ),
                pa_module.array(
                    [float(ep[f"videos/{video_key}/from_timestamp"]) for ep in episodes_meta],
                    type=pa_module.float32(),
                ),
                pa_module.array(
                    [float(ep[f"videos/{video_key}/to_timestamp"]) for ep in episodes_meta],
                    type=pa_module.float32(),
                ),
            ]
        )
        names.extend(
            [
                f"videos/{video_key}/chunk_index",
                f"videos/{video_key}/file_index",
                f"videos/{video_key}/from_timestamp",
                f"videos/{video_key}/to_timestamp",
            ]
        )

    arrays.extend(
        [
            pa_module.array([0] * len(episodes_meta), type=pa_module.int64()),
            pa_module.array([0] * len(episodes_meta), type=pa_module.int64()),
        ]
    )
    names.extend(["meta/episodes/chunk_index", "meta/episodes/file_index"])

    episodes_table = pa_module.Table.from_arrays(arrays, names=names)
    pq_module.write_table(episodes_table, output_path, compression="snappy")


def process_dataset(args: argparse.Namespace) -> Path:
    pa, pq = require_pyarrow_dependencies()
    split_requested = "split" in args.operations
    decode_videos_to_images = bool(args.decode_videos_to_images)

    if split_requested or decode_videos_to_images:
        require_binary("ffmpeg")
        require_binary("ffprobe")
    if decode_videos_to_images:
        require_hf_image_dependencies()

    source_root = resolve_source_dataset_root(
        args.dataset_id,
        local_data_root=args.local_data_root,
        download_if_missing=args.download_if_missing,
        cache_dir=args.cache_dir,
        repo_type=args.repo_type,
    )
    if args.in_place:
        target_root = source_root.parent / f".{source_root.name}_processing_tmp"
        if target_root.exists():
            shutil.rmtree(target_root)
        copy_dataset_without_data(
            source_root,
            target_root,
            include_videos=False,
        )
    else:
        target_root = prepare_target_root(
            source_root,
            in_place=False,
            output_dir=args.output_dir,
            copy_suffix=args.copy_suffix,
            overwrite_output=args.overwrite_output,
            include_videos=(not split_requested) and (not decode_videos_to_images),
        )
    clear_processing_outputs(target_root)

    source_info_path = source_root / "meta" / "info.json"
    source_stats_path = source_root / "meta" / "stats.json"
    source_info = json.loads(source_info_path.read_text(encoding="utf-8"))
    source_stats = (
        json.loads(source_stats_path.read_text(encoding="utf-8"))
        if source_stats_path.exists()
        else {}
    )

    data_path_pattern = str(source_info.get("data_path", DEFAULT_DATA_PATH))
    video_path_pattern = str(source_info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"))
    fps = int(source_info.get("fps", args.fps))

    source_episodes_path = source_root / DEFAULT_EPISODES_PATH
    source_episodes_table = pq.read_table(source_episodes_path)
    source_episodes_meta = source_episodes_table.to_pylist()
    if not source_episodes_meta:
        raise ValueError(f"No source episodes found in {source_episodes_path}")
    episodes_per_chunk = resolve_episodes_per_chunk(
        requested_episodes_per_chunk=args.episodes_per_chunk,
        source_info=source_info,
        source_episodes_meta=source_episodes_meta,
    )

    first_episode_data_file = build_data_file_path(
        source_root,
        data_path_pattern,
        chunk_index=int(source_episodes_meta[0]["data/chunk_index"]),
        file_index=int(source_episodes_meta[0]["data/file_index"]),
    )
    first_schema = pq.read_table(first_episode_data_file).schema
    source_field_types = {field.name: field.type for field in first_schema}
    output_field_types = dict(source_field_types)
    output_column_order = list(first_schema.names)

    signature_plan = build_signature_plan(
        operations=args.operations,
        source_info=source_info,
        source_field_names=first_schema.names,
        path_key=args.path_signature_key,
        delta_key=args.delta_signature_key,
        state_key=args.state_key,
        requested_signature_type=args.signature_type,
        requested_window_size=args.path_signature_window_size,
        requested_depth=args.path_signature_depth,
        requested_backend=args.signature_backend,
    )

    if signature_plan is not None:
        insert_after(output_column_order, signature_plan.path_key, signature_plan.state_key)
        insert_after(output_column_order, signature_plan.delta_key, signature_plan.path_key)
        print(
            "Signature recomputation enabled: "
            f"path={signature_plan.write_path}, "
            f"delta={signature_plan.write_delta}, "
            f"window={signature_plan.window_size}, "
            f"depth={signature_plan.depth}, "
            f"backend={signature_plan.resolved_backend}"
        )

    source_split_lookup, split_order = build_source_split_lookup(
        dict(source_info.get("splits", {})),
        len(source_episodes_meta),
    )
    output_features = copy.deepcopy(dict(source_info.get("features", {})))
    output_info = copy.deepcopy(source_info)
    output_stats = copy.deepcopy(source_stats)
    declared_video_keys = [
        key
        for key, value in output_features.items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    source_video_keys, missing_video_paths_by_key = inspect_available_video_keys(
        source_root=source_root,
        video_path_pattern=video_path_pattern,
        source_episodes_meta=source_episodes_meta,
        video_keys=declared_video_keys,
    )
    unavailable_video_keys = [key for key in declared_video_keys if key not in source_video_keys]
    if unavailable_video_keys:
        print("Warning: some declared video features are missing local mp4 files and will be dropped:")
        for video_key in unavailable_video_keys:
            missing_paths = missing_video_paths_by_key.get(video_key, [])
            sample_text = ", ".join(str(path) for path in missing_paths[:3])
            suffix = "" if len(missing_paths) <= 3 else f", ... ({len(missing_paths)} missing files)"
            print(f"  - {video_key}: {sample_text}{suffix}")
            output_features.pop(video_key, None)
            output_stats.pop(video_key, None)

    output_video_keys = list(source_video_keys)
    if decode_videos_to_images and source_video_keys:
        print(
            "Embedding source video features into parquet image columns: "
            + ", ".join(source_video_keys)
        )
        output_video_keys = []
        for video_key in source_video_keys:
            feature_spec = output_features.get(video_key)
            if isinstance(feature_spec, dict):
                feature_spec["dtype"] = "image"
                feature_spec.pop("info", None)

    for feature_key, feature_spec in output_features.items():
        if (
            isinstance(feature_spec, dict)
            and feature_spec.get("dtype") == "image"
            and feature_key not in output_column_order
        ):
            output_column_order.append(feature_key)

    planning_cache = EpisodeDataCache(pq)
    episode_tasks: list[dict[str, Any]] = []
    output_split_assignments: list[str] = []
    next_output_episode_index = 0
    next_global_index = 0
    for source_episode_index, source_episode_meta in enumerate(
        iter_with_progress(source_episodes_meta, desc="Planning", unit="episode")
    ):
        data_file = build_data_file_path(
            source_root,
            data_path_pattern,
            chunk_index=int(source_episode_meta["data/chunk_index"]),
            file_index=int(source_episode_meta["data/file_index"]),
        )
        source_episode_columns, _ = extract_episode_columns(
            planning_cache,
            data_file,
            source_episode_meta,
        )
        split_ranges = (
            compute_split_ranges(source_episode_columns, split_strategy=args.split_strategy)
            if split_requested
            else [(0, len(source_episode_columns["index"]))]
        )

        segment_specs: list[dict[str, int]] = []
        for segment_start, segment_end in split_ranges:
            segment_length = int(segment_end - segment_start)
            if segment_length <= 0:
                continue
            data_chunk_index, data_file_index = get_chunk_and_file_index(
                next_output_episode_index,
                episodes_per_chunk,
            )
            segment_specs.append(
                {
                    "segment_start": int(segment_start),
                    "segment_end": int(segment_end),
                    "segment_length": int(segment_length),
                    "new_episode_index": int(next_output_episode_index),
                    "global_index_start": int(next_global_index),
                    "data_chunk_index": int(data_chunk_index),
                    "data_file_index": int(data_file_index),
                }
            )
            output_split_assignments.append(source_split_lookup[source_episode_index])
            next_output_episode_index += 1
            next_global_index += segment_length

        if segment_specs:
            episode_tasks.append(
                {
                    "source_episode_index": int(source_episode_index),
                    "source_episode_meta": source_episode_meta,
                    "data_file": data_file,
                    "segment_specs": segment_specs,
                }
            )

    numeric_accumulators: dict[str, NumericStatsAccumulator] = {}
    output_episodes_meta: list[dict[str, Any]] = []
    written_data_files: list[Path] = []
    written_video_files: list[Path] = []
    first_output_video_info_by_key: dict[str, dict[str, Any]] = {}
    path_signature_dim: int | None = None
    delta_signature_dim: int | None = None

    max_workers = int(
        args.workers
        if args.workers is not None
        else min(8, max(1, os.cpu_count() or 1))
    )
    max_workers = max(1, max_workers)
    print(f"Parallel workers: {max_workers}")

    def process_episode_task(task: dict[str, Any]) -> dict[str, Any]:
        local_cache = EpisodeDataCache(pq)
        local_video_info_cache: dict[Path, dict[str, Any]] = {}
        source_episode_meta = dict(task["source_episode_meta"])
        source_episode_columns, _ = extract_episode_columns(
            local_cache,
            Path(task["data_file"]),
            source_episode_meta,
        )

        local_field_types = dict(output_field_types)
        local_numeric_accumulators: dict[str, NumericStatsAccumulator] = {}
        local_output_episodes_meta: list[dict[str, Any]] = []
        local_written_data_files: list[Path] = []
        local_written_video_files: list[Path] = []
        local_first_output_video_info_by_key: dict[str, dict[str, Any]] = {}
        local_path_signature_dim: int | None = None
        local_delta_signature_dim: int | None = None

        for segment_spec in task["segment_specs"]:
            segment_start = int(segment_spec["segment_start"])
            segment_end = int(segment_spec["segment_end"])
            segment_length = int(segment_spec["segment_length"])
            new_episode_index = int(segment_spec["new_episode_index"])
            global_index_start = int(segment_spec["global_index_start"])
            data_chunk_index = int(segment_spec["data_chunk_index"])
            data_file_index = int(segment_spec["data_file_index"])

            segment_columns = slice_frame_columns(
                source_episode_columns,
                segment_start,
                segment_end,
            )
            segment_columns["frame_index"] = list(range(segment_length))
            if "timestamp" in segment_columns and segment_columns["timestamp"]:
                base_timestamp = float(segment_columns["timestamp"][0])
                segment_columns["timestamp"] = [
                    float(value) - base_timestamp
                    for value in segment_columns["timestamp"]
                ]
            else:
                segment_columns["timestamp"] = [
                    frame_idx / max(int(fps), 1)
                    for frame_idx in range(segment_length)
                ]
            segment_columns["episode_index"] = [new_episode_index] * segment_length
            segment_columns["index"] = list(
                range(global_index_start, global_index_start + segment_length)
            )
            segment_columns["next.done"] = [False] * max(segment_length - 1, 0) + [True]

            if signature_plan is not None:
                if signature_plan.state_key not in segment_columns:
                    raise KeyError(
                        f"State key `{signature_plan.state_key}` is missing from the dataset."
                    )
                path_sequence = compute_path_signature_sequence(
                    segment_columns[signature_plan.state_key],
                    window_size=signature_plan.window_size,
                    sig_depth=signature_plan.depth,
                    signature_backend=signature_plan.resolved_backend,
                )
                current_path_signature_dim = int(path_sequence.shape[1])
                if (
                    local_path_signature_dim is not None
                    and local_path_signature_dim != current_path_signature_dim
                ):
                    raise RuntimeError(
                        "Path-signature dimension changed across worker outputs. "
                        f"Expected {local_path_signature_dim}, got {current_path_signature_dim}."
                    )
                local_path_signature_dim = current_path_signature_dim
                local_field_types[signature_plan.path_key] = pa.list_(
                    pa.float32(),
                    current_path_signature_dim,
                )
                segment_columns[signature_plan.path_key] = path_sequence.astype(
                    np.float32
                ).tolist()

                if signature_plan.write_delta:
                    delta_sequence = compute_delta_signature_sequence_np(path_sequence)
                    current_delta_signature_dim = int(delta_sequence.shape[1])
                    if (
                        local_delta_signature_dim is not None
                        and local_delta_signature_dim != current_delta_signature_dim
                    ):
                        raise RuntimeError(
                            "Delta-signature dimension changed across worker outputs. "
                            f"Expected {local_delta_signature_dim}, got {current_delta_signature_dim}."
                        )
                    local_delta_signature_dim = current_delta_signature_dim
                    local_field_types[signature_plan.delta_key] = pa.list_(
                        pa.float32(),
                        current_delta_signature_dim,
                    )
                    segment_columns[signature_plan.delta_key] = delta_sequence.astype(
                        np.float32
                    ).tolist()
                elif signature_plan.delta_key in segment_columns:
                    del segment_columns[signature_plan.delta_key]

            if decode_videos_to_images:
                for video_key in source_video_keys:
                    source_video_from, _ = compute_segment_video_timestamps(
                        source_episode_meta=source_episode_meta,
                        source_episode_columns=source_episode_columns,
                        segment_start=segment_start,
                        segment_length=segment_length,
                        video_key=video_key,
                        fps=fps,
                    )
                    source_video_chunk_index = int(
                        source_episode_meta[f"videos/{video_key}/chunk_index"]
                    )
                    source_video_file_index = int(
                        source_episode_meta[f"videos/{video_key}/file_index"]
                    )
                    source_video_file = build_video_file_path(
                        source_root,
                        video_path_pattern,
                        video_key=video_key,
                        chunk_index=source_video_chunk_index,
                        file_index=source_video_file_index,
                    )
                    segment_columns[video_key] = decode_video_segment_frames(
                        source_video=source_video_file,
                        start_frame=int(round(source_video_from * fps)),
                        num_frames=segment_length,
                        video_info_cache=local_video_info_cache,
                    )

            target_data_file = build_data_file_path(
                target_root,
                DEFAULT_DATA_PATH,
                chunk_index=data_chunk_index,
                file_index=data_file_index,
            )
            write_frame_columns(
                pa_module=pa,
                pq_module=pq,
                frame_columns=segment_columns,
                output_path=target_data_file,
                feature_specs=output_features,
                column_order=output_column_order,
                field_types=local_field_types,
            )
            local_written_data_files.append(target_data_file)
            update_numeric_accumulators(
                local_numeric_accumulators,
                segment_columns,
                feature_specs=output_features,
            )

            episode_meta = {
                "episode_index": new_episode_index,
                "tasks": normalize_tasks(source_episode_meta.get("tasks")),
                "length": segment_length,
                "data/chunk_index": data_chunk_index,
                "data/file_index": data_file_index,
                "dataset_from_index": global_index_start,
                "dataset_to_index": global_index_start + segment_length,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for video_key in output_video_keys:
                if split_requested:
                    source_video_from, _ = compute_segment_video_timestamps(
                        source_episode_meta=source_episode_meta,
                        source_episode_columns=source_episode_columns,
                        segment_start=segment_start,
                        segment_length=segment_length,
                        video_key=video_key,
                        fps=fps,
                    )
                    source_video_chunk_index = int(
                        source_episode_meta[f"videos/{video_key}/chunk_index"]
                    )
                    source_video_file_index = int(
                        source_episode_meta[f"videos/{video_key}/file_index"]
                    )
                    source_video_file = build_video_file_path(
                        source_root,
                        video_path_pattern,
                        video_key=video_key,
                        chunk_index=source_video_chunk_index,
                        file_index=source_video_file_index,
                    )
                    target_video_file = build_video_file_path(
                        target_root,
                        video_path_pattern,
                        video_key=video_key,
                        chunk_index=data_chunk_index,
                        file_index=data_file_index,
                    )
                    sliced_video_info = slice_video_segment(
                        source_video=source_video_file,
                        target_video=target_video_file,
                        start_frame=int(round(source_video_from * fps)),
                        num_frames=segment_length,
                    )
                    local_written_video_files.append(target_video_file)
                    local_first_output_video_info_by_key.setdefault(
                        video_key,
                        sliced_video_info,
                    )
                    episode_meta[f"videos/{video_key}/chunk_index"] = int(data_chunk_index)
                    episode_meta[f"videos/{video_key}/file_index"] = int(data_file_index)
                    episode_meta[f"videos/{video_key}/from_timestamp"] = 0.0
                    episode_meta[f"videos/{video_key}/to_timestamp"] = float(
                        segment_length / max(float(sliced_video_info["fps"]), 1.0)
                    )
                else:
                    episode_meta[f"videos/{video_key}/chunk_index"] = int(
                        source_episode_meta[f"videos/{video_key}/chunk_index"]
                    )
                    episode_meta[f"videos/{video_key}/file_index"] = int(
                        source_episode_meta[f"videos/{video_key}/file_index"]
                    )
                    (
                        episode_meta[f"videos/{video_key}/from_timestamp"],
                        episode_meta[f"videos/{video_key}/to_timestamp"],
                    ) = compute_segment_video_timestamps(
                        source_episode_meta=source_episode_meta,
                        source_episode_columns=source_episode_columns,
                        segment_start=segment_start,
                        segment_length=segment_length,
                        video_key=video_key,
                        fps=fps,
                    )

            local_output_episodes_meta.append(episode_meta)

        return {
            "episodes_meta": local_output_episodes_meta,
            "numeric_accumulators": local_numeric_accumulators,
            "written_data_files": local_written_data_files,
            "written_video_files": local_written_video_files,
            "first_output_video_info_by_key": local_first_output_video_info_by_key,
            "path_signature_dim": local_path_signature_dim,
            "delta_signature_dim": local_delta_signature_dim,
        }

    def merge_result(result: dict[str, Any]) -> None:
        nonlocal path_signature_dim, delta_signature_dim

        output_episodes_meta.extend(result["episodes_meta"])
        written_data_files.extend(result["written_data_files"])
        written_video_files.extend(result["written_video_files"])
        merge_numeric_accumulators(numeric_accumulators, result["numeric_accumulators"])

        for video_key, video_info in result["first_output_video_info_by_key"].items():
            first_output_video_info_by_key.setdefault(video_key, video_info)

        worker_path_signature_dim = result["path_signature_dim"]
        if worker_path_signature_dim is not None:
            if path_signature_dim is None:
                path_signature_dim = int(worker_path_signature_dim)
            elif path_signature_dim != int(worker_path_signature_dim):
                raise RuntimeError(
                    "Path-signature dimension changed across workers. "
                    f"Expected {path_signature_dim}, got {worker_path_signature_dim}."
                )

        worker_delta_signature_dim = result["delta_signature_dim"]
        if worker_delta_signature_dim is not None:
            if delta_signature_dim is None:
                delta_signature_dim = int(worker_delta_signature_dim)
            elif delta_signature_dim != int(worker_delta_signature_dim):
                raise RuntimeError(
                    "Delta-signature dimension changed across workers. "
                    f"Expected {delta_signature_dim}, got {worker_delta_signature_dim}."
                )

    if max_workers == 1 or len(episode_tasks) <= 1:
        for task in iter_with_progress(episode_tasks, desc="Episodes", unit="episode"):
            merge_result(process_episode_task(task))
    else:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_episode_task, task)
                for task in episode_tasks
            ]
            for future in iter_completed_futures(futures, desc="Episodes", unit="episode"):
                merge_result(future.result())

    output_episodes_meta.sort(key=lambda episode: int(episode["episode_index"]))
    total_frames = next_global_index
    total_episodes = next_output_episode_index
    output_splits = build_output_splits(output_split_assignments, split_order)
    validate_processed_dataset(
        output_episodes_meta,
        output_splits,
        total_frames,
        total_episodes,
    )

    if signature_plan is not None:
        if path_signature_dim is None:
            raise RuntimeError("Failed to compute path-signature dimensions.")
        output_features[signature_plan.path_key] = {
            "dtype": "float32",
            "shape": [path_signature_dim],
            "names": [f"path_sig_{idx}" for idx in range(path_signature_dim)],
        }
        output_info["path_signature"] = {
            "key": signature_plan.path_key,
            "window_size": int(signature_plan.window_size),
            "window_mode": (
                "full_prefix"
                if int(signature_plan.window_size) <= 0
                else "sliding_window"
            ),
            "sig_depth": int(signature_plan.depth),
            "signature_dim": int(path_signature_dim),
            "kind": (
                "signature"
                if signature_plan.resolved_backend == "signatory"
                else "simple_signature"
            ),
            "backend": signature_plan.resolved_backend,
        }
        if signature_plan.write_delta:
            if delta_signature_dim is None:
                raise RuntimeError("Failed to compute delta-signature dimensions.")
            output_features[signature_plan.delta_key] = {
                "dtype": "float32",
                "shape": [delta_signature_dim],
                "names": [f"delta_path_sig_{idx}" for idx in range(delta_signature_dim)],
            }
            output_info["delta_signature"] = {
                "key": signature_plan.delta_key,
                "signature_key": signature_plan.path_key,
                "definition": "path_signature_t - path_signature_{t-1}",
                "first_step_rule": "zeros",
                "signature_dim": int(delta_signature_dim),
            }
        else:
            output_features.pop(signature_plan.delta_key, None)
            output_info.pop("delta_signature", None)

    for video_key in source_video_keys:
        if video_key in source_stats:
            output_stats[video_key] = source_stats[video_key]
    for video_key in output_video_keys:
        if split_requested and video_key in first_output_video_info_by_key:
            video_info = first_output_video_info_by_key[video_key]
            if video_key in output_features and isinstance(output_features[video_key], dict):
                output_features[video_key]["shape"] = [
                    int(video_info["height"]),
                    int(video_info["width"]),
                    3,
                ]
                output_features[video_key]["names"] = ["height", "width", "channels"]
                output_features[video_key]["info"] = {
                    "video.height": int(video_info["height"]),
                    "video.width": int(video_info["width"]),
                    "video.codec": str(video_info["codec"]),
                    "video.pix_fmt": str(video_info["pix_fmt"]),
                    "video.is_depth_map": False,
                    "video.fps": int(round(float(video_info["fps"]))),
                    "video.channels": 3,
                    "has_audio": False,
                }

    for key, accumulator in numeric_accumulators.items():
        output_stats[key] = accumulator.finalize()

    output_info["features"] = output_features
    output_info["total_episodes"] = int(total_episodes)
    output_info["total_frames"] = int(total_frames)
    output_info["total_tasks"] = int(output_info.get("total_tasks", 0) or 0)
    output_info["chunks_size"] = int(episodes_per_chunk)
    if split_requested and first_output_video_info_by_key:
        first_video_info = next(iter(first_output_video_info_by_key.values()))
        output_info["fps"] = int(round(float(first_video_info["fps"])))
    else:
        output_info["fps"] = int(fps)
    output_info["splits"] = output_splits
    output_info["data_path"] = DEFAULT_DATA_PATH
    output_info["video_path"] = video_path_pattern if output_video_keys else None
    output_info["data_files_size_in_mb"] = estimate_total_size_mb(written_data_files)
    output_info["video_files_size_in_mb"] = (
        estimate_total_size_mb(
            written_video_files
            if split_requested
            else sorted(
                ((source_root if args.in_place else target_root) / "videos").rglob("*.mp4")
            )
        )
        if output_video_keys
        else 0
    )

    target_episodes_path = target_root / DEFAULT_EPISODES_PATH
    write_episodes_table(
        pa,
        pq,
        episodes_meta=output_episodes_meta,
        output_path=target_episodes_path,
        video_keys=output_video_keys,
    )

    (target_root / "meta").mkdir(parents=True, exist_ok=True)
    (target_root / "meta" / "info.json").write_text(
        json.dumps(output_info, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    (target_root / "meta" / "stats.json").write_text(
        json.dumps(output_stats, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    final_root = target_root
    if args.in_place:
        copy_processed_outputs_back(
            target_root,
            source_root,
            include_videos=split_requested and bool(output_video_keys),
        )
        shutil.rmtree(target_root)
        final_root = source_root

    print(f"Processed dataset written to: {final_root}")
    print(f"Episodes: {total_episodes}, Frames: {total_frames}")
    return final_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process LeRobot datasets by splitting nested trajectories and "
            "recomputing path/delta signatures."
        )
    )
    parser.add_argument(
        "dataset_id",
        type=str,
        help=(
            "Dataset identifier, usually a Hugging Face repo id like `org/name`, "
            "or a local dataset path."
        ),
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=["split", "update-signatures"],
        required=True,
        help="One or both operations to run. If both are provided, split happens first.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["auto", "done", "episode_index"],
        default="auto",
        help=(
            "`auto` splits on `next.done` and on timestamp/frame-index resets. "
            "`done` only uses `next.done`. `episode_index` disables nested splitting."
        ),
    )
    parser.add_argument(
        "--signature-type",
        choices=["path", "delta", "both"],
        default="both",
        help="Requested signature update mode when `update-signatures` is enabled.",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default=DEFAULT_STATE_KEY,
        help="Feature key used to compute signatures.",
    )
    parser.add_argument(
        "--path-signature-key",
        type=str,
        default=DEFAULT_PATH_SIGNATURE_KEY,
        help="Feature key used to store path signatures.",
    )
    parser.add_argument(
        "--delta-signature-key",
        type=str,
        default=DEFAULT_DELTA_SIGNATURE_KEY,
        help="Feature key used to store delta signatures.",
    )
    parser.add_argument(
        "--path-signature-window-size",
        type=int,
        default=DEFAULT_SIGNATURE_WINDOW_SIZE,
        help="Sliding window size for path signatures. Use 0 for the full prefix.",
    )
    parser.add_argument(
        "--path-signature-depth",
        type=int,
        default=DEFAULT_SIGNATURE_DEPTH,
        help="Signature depth used during recomputation.",
    )
    parser.add_argument(
        "--signature-backend",
        choices=["auto", "simple", "signatory"],
        default=DEFAULT_SIGNATURE_BACKEND,
        help="Backend used to compute path signatures.",
    )
    parser.add_argument(
        "--episodes-per-chunk",
        type=int,
        default=None,
        help="Number of episode files per chunk in the rewritten dataset.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel workers used for episode processing. "
            "Defaults to min(8, cpu_count). Use 1 to disable parallelism."
        ),
    )
    parser.add_argument(
        "--decode-videos-to-images",
        action="store_true",
        help=(
            "Decode source mp4 visual features into parquet-embedded `image` columns. "
            "This removes runtime video decoding at training time, but increases "
            "dataset size and processing time."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Fallback FPS used when the source dataset metadata does not provide one.",
    )
    parser.add_argument(
        "--local-data-root",
        type=Path,
        default=DEFAULT_LOCAL_DATA_ROOT,
        help="Local base directory used to resolve or download datasets.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download the dataset with huggingface_hub if no local copy is found.",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        help="Repository type passed to huggingface_hub when downloading.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for huggingface_hub downloads.",
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the source dataset in place.",
    )
    overwrite_group.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write the processed dataset to a separate directory.",
    )
    parser.add_argument(
        "--copy-suffix",
        type=str,
        default=DEFAULT_COPY_SUFFIX,
        help="Suffix used for the default output directory when not running in place.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete an existing output directory before writing the processed dataset.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.operations = list(dict.fromkeys(args.operations))
    process_dataset(args)


if __name__ == "__main__":
    main()
