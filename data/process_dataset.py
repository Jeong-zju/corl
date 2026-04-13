from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_OBSERVATION_KEY = "observation.state"
DEFAULT_PATH_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_DELTA_SIGNATURE_KEY = "observation.delta_signature"
SIGNATURE_CACHE_LAYOUT_VERSION = 1


def _sanitize_path_part(value: str) -> str:
    normalized = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    normalized = normalized.strip("-")
    return normalized or "item"


@dataclass(slots=True)
class VectorStatsAccumulator:
    count: int = 0
    min: np.ndarray | None = None
    max: np.ndarray | None = None
    sum: np.ndarray | None = None
    sumsq: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(
                "Expected a 2D array when accumulating stats. "
                f"Got shape={array.shape}."
            )
        if array.shape[0] == 0:
            return

        batch_min = np.min(array, axis=0)
        batch_max = np.max(array, axis=0)
        batch_sum = np.sum(array, axis=0, dtype=np.float64)
        batch_sumsq = np.sum(np.square(array, dtype=np.float64), axis=0, dtype=np.float64)

        if self.min is None:
            self.min = batch_min
            self.max = batch_max
            self.sum = batch_sum
            self.sumsq = batch_sumsq
        else:
            self.min = np.minimum(self.min, batch_min)
            self.max = np.maximum(self.max, batch_max)
            self.sum = self.sum + batch_sum
            self.sumsq = self.sumsq + batch_sumsq
        self.count += int(array.shape[0])

    def finalize(self) -> dict[str, list[float] | list[int]]:
        if (
            self.count <= 0
            or self.min is None
            or self.max is None
            or self.sum is None
            or self.sumsq is None
        ):
            raise ValueError("Cannot finalize empty stats accumulator.")

        mean = self.sum / float(self.count)
        variance = np.maximum(self.sumsq / float(self.count) - np.square(mean), 0.0)
        std = np.sqrt(variance, dtype=np.float64)
        return {
            "min": self.min.astype(np.float32).tolist(),
            "max": self.max.astype(np.float32).tolist(),
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "count": [int(self.count)],
        }


def get_tqdm():
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return None
    return tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute full-prefix path signatures for every frame of every episode in "
            "a LeRobot dataset, save them into a signature cache directory, and "
            "update metadata without storing signature columns in parquet."
        )
    )
    parser.add_argument(
        "dataset",
        type=str,
        nargs="?",
        help=(
            "Dataset root path, a path inside `main/data/`, or a dataset id relative "
            "to `main/data/`."
        ),
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_option",
        type=str,
        default=None,
        help=(
            "Dataset root path, a path inside `main/data/`, or a dataset id relative "
            "to `main/data/`. This is an alias for the positional dataset argument."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional output dataset directory. If omitted, the input dataset is "
            "updated in place."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacing an existing output directory when --output-dir is used.",
    )
    parser.add_argument(
        "--observation-key",
        type=str,
        default=DEFAULT_OBSERVATION_KEY,
        help="State feature used to compute signatures.",
    )
    parser.add_argument(
        "--path-signature-key",
        type=str,
        default=DEFAULT_PATH_SIGNATURE_KEY,
        help="Logical dataset feature key used for path signatures.",
    )
    parser.add_argument(
        "--delta-signature-key",
        type=str,
        default=DEFAULT_DELTA_SIGNATURE_KEY,
        help="Logical dataset feature key used for delta signatures.",
    )
    parser.add_argument(
        "--signature-depth",
        type=int,
        default=3,
        help="Truncation depth passed to signatory.signature.",
    )
    parser.add_argument(
        "--signature-cache-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype used by the generated signature cache files.",
    )
    parser.add_argument(
        "--signature-cache-root",
        type=str,
        default=None,
        help=(
            "Optional cache directory override. Defaults to a hidden cache folder "
            "inside the target dataset directory."
        ),
    )
    return parser


def resolve_dataset_arg(
    parser: argparse.ArgumentParser,
    *,
    dataset: str | None,
    dataset_option: str | None,
) -> str:
    if dataset is not None and dataset_option is not None:
        parser.error("Specify the dataset either positionally or via --dataset, not both.")
    resolved = dataset_option if dataset_option is not None else dataset
    if resolved is None:
        parser.error("the following arguments are required: dataset")
    return resolved


def is_lerobot_dataset_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "meta/info.json").exists()
        and (path / "meta/stats.json").exists()
        and (path / "data").is_dir()
    )


def _resolve_existing_dataset_root(candidate: Path) -> Path | None:
    expanded = candidate.expanduser()

    direct_candidates = [expanded]
    if expanded.is_file() and expanded.name == "info.json" and expanded.parent.name == "meta":
        direct_candidates.append(expanded.parent.parent)
    if expanded.is_file():
        direct_candidates.append(expanded.parent)

    for direct_candidate in direct_candidates:
        for parent in [direct_candidate, *direct_candidate.parents]:
            if is_lerobot_dataset_root(parent):
                return parent.resolve()

    if not expanded.exists():
        return None

    for info_path in sorted(expanded.rglob("meta/info.json")):
        root = info_path.parent.parent
        if is_lerobot_dataset_root(root):
            return root.resolve()
    return None


def resolve_dataset_root(dataset_arg: str) -> Path:
    raw = Path(dataset_arg).expanduser()
    main_root = Path(__file__).resolve().parents[1]
    data_root = main_root / "data"

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                main_root / raw,
                data_root / raw,
            ]
        )

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            ordered_candidates.append(resolved)

    for candidate in ordered_candidates:
        resolved_root = _resolve_existing_dataset_root(candidate)
        if resolved_root is not None:
            return resolved_root

    checked = "\n".join(f"- {path}" for path in ordered_candidates)
    raise FileNotFoundError(
        "Could not resolve a LeRobot dataset root from the provided argument. "
        "Checked these paths:\n"
        f"{checked}"
    )


def resolve_output_dir(output_dir: str | None) -> Path | None:
    if output_dir is None:
        return None

    raw = Path(output_dir).expanduser()
    main_root = Path(__file__).resolve().parents[1]
    repo_root = main_root.parent
    if raw.is_absolute():
        return raw.resolve()

    text = raw.as_posix()
    if text.startswith("main/data/"):
        return (repo_root / raw).resolve()
    if text.startswith("data/"):
        return (main_root / raw).resolve()
    return (main_root / "data" / raw).resolve()


def infer_dataset_repo_id(dataset_root: Path) -> str:
    data_root = Path(__file__).resolve().parents[1] / "data"
    try:
        return dataset_root.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return dataset_root.resolve().name


def resolve_signature_cache_dir(
    dataset_root: Path,
    *,
    dataset_repo_id: str,
    signature_cache_root: str | None,
) -> Path:
    if signature_cache_root is None:
        root = dataset_root / ".signature_cache" / _sanitize_path_part(dataset_repo_id)
    else:
        root = Path(signature_cache_root).expanduser()
        if not root.is_absolute():
            root = (dataset_root / root).resolve()
    return root / f"signature_cache_v{SIGNATURE_CACHE_LAYOUT_VERSION}"


def prepare_target_dataset(
    dataset_root: Path,
    *,
    output_dir: Path | None,
    overwrite_output: bool,
) -> Path:
    if output_dir is None:
        return dataset_root

    output_dir = output_dir.resolve()
    if output_dir == dataset_root.resolve():
        return dataset_root

    if output_dir.exists():
        if not overwrite_output:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite-output to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dataset_root, output_dir)
    return output_dir.resolve()


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`pyarrow` is required to process LeRobot parquet files. "
            "Activate the intended environment and install pyarrow first."
        ) from exc
    return pa, pq


def compute_delta_signature_sequence_np(signatures: np.ndarray) -> np.ndarray:
    signatures_array = np.asarray(signatures, dtype=np.float32)
    if signatures_array.ndim != 2:
        raise ValueError(
            "Expected signature trajectory with shape (T, signature_dim). "
            f"Got {signatures_array.shape}."
        )
    if signatures_array.shape[0] == 0:
        raise ValueError("Signature trajectory must contain at least one step.")

    delta = np.zeros_like(signatures_array, dtype=np.float32)
    if signatures_array.shape[0] > 1:
        delta[1:] = signatures_array[1:] - signatures_array[:-1]
    return delta


def compute_episode_path_signatures(
    state_sequence: np.ndarray,
    *,
    signature_depth: int,
) -> np.ndarray:
    states = np.asarray(state_sequence, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(
            "State trajectory must have shape (T, state_dim). "
            f"Got {states.shape}."
        )
    if states.shape[0] == 0:
        raise ValueError("State trajectory must contain at least one frame.")
    if signature_depth <= 0:
        raise ValueError(
            f"`signature_depth` must be positive, got {signature_depth}."
        )

    try:
        import torch
        import signatory
    except ImportError as exc:
        raise ImportError(
            "`signatory` is required to compute path signatures. "
            "Install it in the active environment before running this script."
        ) from exc

    path = torch.from_numpy(states).unsqueeze(0)
    basepoint = path[:, 0, :]
    with torch.no_grad():
        streamed = signatory.signature(
            path,
            depth=int(signature_depth),
            stream=True,
            basepoint=basepoint,
        )
    signatures = streamed.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    if signatures.shape[0] != states.shape[0]:
        raise RuntimeError(
            "Unexpected streamed signature shape. "
            f"Expected {states.shape[0]} rows, got {signatures.shape[0]}."
        )
    return signatures


def _column_to_matrix(table, key: str) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f"Column {key!r} was not found in the parquet table.")

    values = table[key].to_pylist()
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(
            f"Column {key!r} must be a 2D vector feature. Got shape={matrix.shape}."
        )
    return matrix


def _column_to_int_vector(table, key: str) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f"Column {key!r} was not found in the parquet table.")
    return np.asarray(table[key].to_pylist(), dtype=np.int64)


def _fixed_size_list_array(pa_module, values: np.ndarray):
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(
            "Expected a 2D matrix when building a fixed-size list array. "
            f"Got shape={matrix.shape}."
        )
    flat = pa_module.array(matrix.reshape(-1), type=pa_module.float32())
    return pa_module.FixedSizeListArray.from_arrays(flat, int(matrix.shape[1]))


def _replace_or_append_column(table, *, key: str, array):
    if key in table.column_names:
        return table.set_column(table.column_names.index(key), key, array)
    return table.append_column(key, array)


def write_table_atomic(table, file_path: Path, pq_module) -> None:
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    pq_module.write_table(table, temp_path, compression="snappy")
    temp_path.replace(file_path)


def build_signature_cache_manifest(dataset_root: Path) -> dict[str, object]:
    data_files = iter_data_parquet_files(dataset_root)
    if not data_files:
        raise FileNotFoundError(
            f"No parquet files were found under {dataset_root / 'data'}."
        )
    episodes_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    _, pq = require_pyarrow()
    episode_meta = pq.read_table(
        episodes_path,
        columns=["dataset_to_index"],
    )
    total_frames = int(max(episode_meta.column("dataset_to_index").to_pylist()))
    return {
        "dataset_root": str(dataset_root.resolve()),
        "info_path": str((dataset_root / "meta/info.json").resolve()),
        "stats_path": str((dataset_root / "meta/stats.json").resolve()),
        "episodes_path": str(episodes_path.resolve()),
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


def update_dataset_metadata(
    dataset_root: Path,
    *,
    dataset_repo_id: str,
    observation_key: str,
    path_signature_key: str,
    delta_signature_key: str,
    signature_depth: int,
    signature_dim: int,
    path_signature_stats: VectorStatsAccumulator,
    delta_signature_stats: VectorStatsAccumulator,
    signature_cache_dir: Path,
    signature_cache_dtype: str,
) -> None:
    info_path = dataset_root / "meta/info.json"
    stats_path = dataset_root / "meta/stats.json"

    info = json.loads(info_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    try:
        cache_dir_value = str(signature_cache_dir.relative_to(dataset_root))
    except ValueError:
        cache_dir_value = str(signature_cache_dir)

    features = info.setdefault("features", {})
    features[path_signature_key] = {
        "dtype": "float32",
        "shape": [int(signature_dim)],
        "names": [f"path_sig_{index}" for index in range(int(signature_dim))],
    }
    features[delta_signature_key] = {
        "dtype": "float32",
        "shape": [int(signature_dim)],
        "names": [f"delta_path_sig_{index}" for index in range(int(signature_dim))],
    }

    info["path_signature"] = {
        "key": path_signature_key,
        "source_key": observation_key,
        "signature_depth": int(signature_depth),
        "signature_dim": int(signature_dim),
        "backend": "signatory",
        "window": "full_prefix",
        "basepoint": "first_frame",
        "kind": "signature",
        "storage": "signature_cache",
        "cache_dir": cache_dir_value,
        "cache_dtype": str(signature_cache_dtype),
        "dataset_repo_id": str(dataset_repo_id),
    }
    info["delta_signature"] = {
        "key": delta_signature_key,
        "signature_key": path_signature_key,
        "definition": "path_signature_t - path_signature_{t-1}",
        "first_step_rule": "zeros",
        "signature_dim": int(signature_dim),
        "storage": "signature_cache",
        "cache_dir": cache_dir_value,
        "cache_dtype": str(signature_cache_dtype),
        "dataset_repo_id": str(dataset_repo_id),
    }

    stats[path_signature_key] = path_signature_stats.finalize()
    stats[delta_signature_key] = delta_signature_stats.finalize()

    info_path.write_text(
        json.dumps(info, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(stats, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def iter_data_parquet_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "data").glob("chunk-*/*.parquet"))


def write_signature_cache_metadata(
    cache_dir: Path,
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    path_signature_key: str,
    delta_signature_key: str,
    signature_dim: int,
    signature_cache_dtype: str,
) -> None:
    manifest = build_signature_cache_manifest(dataset_root)
    total_frames = int(manifest["total_frames"])
    bytes_per_value = np.dtype(signature_cache_dtype).itemsize
    estimated_bytes = int(total_frames * signature_dim * 2 * bytes_per_value)
    metadata = {
        "layout_version": SIGNATURE_CACHE_LAYOUT_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "dataset_repo_id": str(dataset_repo_id),
        "feature_keys": [path_signature_key, delta_signature_key],
        "feature_shapes": {
            path_signature_key: [int(signature_dim)],
            delta_signature_key: [int(signature_dim)],
        },
        "cache_dtype": str(signature_cache_dtype),
        "pre_normalized": True,
        "normalization_mode": "MEAN_STD",
        "eps": 1e-8,
        "files": {
            path_signature_key: f"{_sanitize_path_part(path_signature_key)}.{signature_cache_dtype}.npy",
            delta_signature_key: f"{_sanitize_path_part(delta_signature_key)}.{signature_cache_dtype}.npy",
        },
        "estimated_bytes": estimated_bytes,
        "built_at": dt.datetime.now().isoformat(),
        "dataset_manifest": manifest,
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def process_dataset(
    dataset_root: Path,
    *,
    observation_key: str,
    path_signature_key: str,
    delta_signature_key: str,
    signature_depth: int,
    signature_cache_dtype: str,
    signature_cache_root: str | None,
) -> None:
    pa, pq = require_pyarrow()

    dataset_repo_id = infer_dataset_repo_id(dataset_root)
    cache_dir = resolve_signature_cache_dir(
        dataset_root,
        dataset_repo_id=dataset_repo_id,
        signature_cache_root=signature_cache_root,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_files = iter_data_parquet_files(dataset_root)
    if not data_files:
        raise FileNotFoundError(
            f"No parquet files were found under {dataset_root / 'data'}."
        )

    initial_manifest = build_signature_cache_manifest(dataset_root)
    total_frames = int(initial_manifest["total_frames"])
    tqdm = get_tqdm()
    expected_signature_dim: int | None = None
    total_rows = 0
    total_episodes = 0
    path_signature_stats = VectorStatsAccumulator()
    delta_signature_stats = VectorStatsAccumulator()
    cache_path_signatures = None
    cache_delta_signatures = None
    seen_indices = np.zeros(total_frames, dtype=bool)

    file_iterator = (
        tqdm(
            data_files,
            desc="Processing parquet files",
            unit="file",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else data_files
    )

    for file_index, parquet_path in enumerate(file_iterator, start=1):
        if tqdm is not None:
            file_iterator.set_postfix_str(parquet_path.name)
        table = pq.read_table(parquet_path)
        if table.num_rows == 0:
            print(
                f"[SKIP] {parquet_path} is empty. "
                "No signature cache rows were written."
            )
            continue

        states = _column_to_matrix(table, observation_key)
        absolute_indices = _column_to_int_vector(table, "index")
        episode_indices = _column_to_int_vector(table, "episode_index")
        if "frame_index" in table.column_names:
            frame_indices = _column_to_int_vector(table, "frame_index")
        else:
            frame_indices = np.arange(table.num_rows, dtype=np.int64)

        unique_episode_indices = list(dict.fromkeys(episode_indices.tolist()))
        episode_iterator = (
            tqdm(
                unique_episode_indices,
                desc=f"Episodes in {parquet_path.name}",
                unit="episode",
                leave=False,
                dynamic_ncols=True,
            )
            if tqdm is not None
            else unique_episode_indices
        )

        for episode_index in episode_iterator:
            row_indices = np.flatnonzero(episode_indices == episode_index)
            if row_indices.size == 0:
                continue

            order = np.argsort(frame_indices[row_indices], kind="stable")
            sorted_row_indices = row_indices[order]
            episode_states = states[sorted_row_indices]
            episode_path_signatures = compute_episode_path_signatures(
                episode_states,
                signature_depth=signature_depth,
            )
            episode_delta_signatures = compute_delta_signature_sequence_np(
                episode_path_signatures
            )

            signature_dim = int(episode_path_signatures.shape[1])
            if expected_signature_dim is None:
                expected_signature_dim = signature_dim
                cache_path_signatures = np.lib.format.open_memmap(
                    cache_dir
                    / f"{_sanitize_path_part(path_signature_key)}.{signature_cache_dtype}.npy",
                    mode="w+",
                    dtype=np.dtype(signature_cache_dtype),
                    shape=(total_frames, signature_dim),
                )
                cache_delta_signatures = np.lib.format.open_memmap(
                    cache_dir
                    / f"{_sanitize_path_part(delta_signature_key)}.{signature_cache_dtype}.npy",
                    mode="w+",
                    dtype=np.dtype(signature_cache_dtype),
                    shape=(total_frames, signature_dim),
                )
            elif signature_dim != expected_signature_dim:
                raise RuntimeError(
                    "Signature dimension changed across episodes. "
                    f"Expected {expected_signature_dim}, got {signature_dim} "
                    f"for episode_index={episode_index} in {parquet_path}."
                )

            assert cache_path_signatures is not None
            assert cache_delta_signatures is not None
            sorted_absolute_indices = absolute_indices[sorted_row_indices]
            cache_path_signatures[sorted_absolute_indices] = episode_path_signatures.astype(
                signature_cache_dtype,
                copy=False,
            )
            cache_delta_signatures[sorted_absolute_indices] = episode_delta_signatures.astype(
                signature_cache_dtype,
                copy=False,
            )
            seen_indices[sorted_absolute_indices] = True
            path_signature_stats.update(episode_path_signatures)
            delta_signature_stats.update(episode_delta_signatures)
            total_episodes += 1
            if tqdm is not None:
                episode_iterator.set_postfix(
                    episode=int(episode_index),
                    frames=int(sorted_row_indices.size),
                )

        if cache_path_signatures is None or cache_delta_signatures is None:
            raise RuntimeError(
                f"Failed to compute signatures for parquet file: {parquet_path}"
            )

        existing_signature_columns = [
            key
            for key in (path_signature_key, delta_signature_key)
            if key in table.column_names
        ]
        if existing_signature_columns:
            stripped_table = table.drop(existing_signature_columns)
            write_table_atomic(stripped_table, parquet_path, pq)
            parquet_status = f"removed_columns={existing_signature_columns}"
        else:
            parquet_status = "parquet_unchanged"

        total_rows += int(table.num_rows)
        print(
            f"[{file_index}/{len(data_files)}] Processed {parquet_path} "
            f"(rows={table.num_rows}, episodes={len(unique_episode_indices)}, {parquet_status})"
        )

    if expected_signature_dim is None:
        raise RuntimeError("No signatures were computed from the dataset.")
    if not bool(np.all(seen_indices)):
        missing = int((~seen_indices).sum())
        raise RuntimeError(
            "Signature cache build did not cover every absolute frame index. "
            f"missing={missing}"
        )

    assert cache_path_signatures is not None
    assert cache_delta_signatures is not None
    cache_path_signatures.flush()
    cache_delta_signatures.flush()

    update_dataset_metadata(
        dataset_root,
        dataset_repo_id=dataset_repo_id,
        observation_key=observation_key,
        path_signature_key=path_signature_key,
        delta_signature_key=delta_signature_key,
        signature_depth=signature_depth,
        signature_dim=expected_signature_dim,
        path_signature_stats=path_signature_stats,
        delta_signature_stats=delta_signature_stats,
        signature_cache_dir=cache_dir,
        signature_cache_dtype=signature_cache_dtype,
    )
    write_signature_cache_metadata(
        cache_dir,
        dataset_root=dataset_root,
        dataset_repo_id=dataset_repo_id,
        path_signature_key=path_signature_key,
        delta_signature_key=delta_signature_key,
        signature_dim=expected_signature_dim,
        signature_cache_dtype=signature_cache_dtype,
    )

    print(
        "Finished building dataset signature cache: "
        f"dataset_root={dataset_root}, rows={total_rows}, "
        f"episodes={total_episodes}, signature_dim={expected_signature_dim}, "
        f"depth={signature_depth}, cache_dir={cache_dir}, cache_dtype={signature_cache_dtype}"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dataset_arg = resolve_dataset_arg(
        parser,
        dataset=args.dataset,
        dataset_option=args.dataset_option,
    )
    dataset_root = resolve_dataset_root(dataset_arg)
    target_root = prepare_target_dataset(
        dataset_root,
        output_dir=resolve_output_dir(args.output_dir),
        overwrite_output=bool(args.overwrite_output),
    )

    process_dataset(
        target_root,
        observation_key=str(args.observation_key),
        path_signature_key=str(args.path_signature_key),
        delta_signature_key=str(args.delta_signature_key),
        signature_depth=int(args.signature_depth),
        signature_cache_dtype=str(args.signature_cache_dtype),
        signature_cache_root=args.signature_cache_root,
    )


if __name__ == "__main__":
    main()
