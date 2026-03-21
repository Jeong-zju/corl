#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval_helpers import (
    compute_signatory_signature_np,
    compute_simple_signature_np,
    resolve_signature_backend,
)


DEFAULT_STATE_KEY = "observation.state"
DEFAULT_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/path_signature_tsne")


@dataclass(slots=True)
class LoadedEpisodeFeatures:
    features: np.ndarray
    episode_indices: np.ndarray
    label_texts: list[str] | None
    label_source: str | None
    dataset_format: str
    feature_source: str
    signature_key: str | None
    state_key: str | None
    metadata: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a processed .npz or LeRobot dataset, extract per-episode path "
            "signature features, and produce a 2D t-SNE embedding."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to a processed .npz dataset or a LeRobot dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the embedding CSV, plots, and summary JSON.",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "npz", "lerobot"],
        default="auto",
        help="Dataset format. Defaults to automatic detection.",
    )
    parser.add_argument(
        "--feature-source",
        choices=["auto", "dataset", "compute"],
        default="auto",
        help=(
            "Use path signatures already stored in the dataset, or recompute them "
            "from the state trajectory."
        ),
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default=DEFAULT_STATE_KEY,
        help="State feature key for LeRobot datasets, or optional npz key override.",
    )
    parser.add_argument(
        "--signature-key",
        type=str,
        default=None,
        help="Optional explicit path-signature key override.",
    )
    parser.add_argument(
        "--label-source",
        choices=["auto", "task", "goal", "success", "episode", "none"],
        default="auto",
        help="Metadata used to color the visualization.",
    )
    parser.add_argument(
        "--signature-pool",
        choices=["last", "mean", "flatten"],
        default="last",
        help=(
            "How to reduce a per-timestep path-signature sequence to one vector per "
            "episode. `last` usually matches the full-trajectory signature."
        ),
    )
    parser.add_argument(
        "--path-signature-window-size",
        type=int,
        default=0,
        help="Sliding window size for recomputed signatures. 0 means full prefix.",
    )
    parser.add_argument(
        "--path-signature-depth",
        type=int,
        default=3,
        help="Signature depth used when signatures must be recomputed.",
    )
    parser.add_argument(
        "--signature-backend",
        choices=["auto", "signatory", "simple"],
        default="auto",
        help="Backend used when signatures must be recomputed.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Randomly sample at most this many episodes before fitting t-SNE.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for episode sampling, t-SNE, and optional KMeans.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Target t-SNE perplexity. It will be clamped to the dataset size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate. Use 0 to delegate to sklearn's auto mode.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of t-SNE optimization iterations.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Optional PCA preprocessing dimension before t-SNE. 0 disables PCA.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=0,
        help=(
            "Optional KMeans cluster count fitted on the path-signature features. "
            "0 disables KMeans."
        ),
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=36.0,
        help="Scatter marker size in the saved plots.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=14.0,
        help="Figure width in inches for 2D plots.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=10.0,
        help="Figure height in inches for 2D plots.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=180,
        help="DPI for the saved 2D PNG plots.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip PNG plot generation and only export CSV/JSON outputs.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help=(
            "Show the 2D figure in an interactive matplotlib window after saving, "
            "so you can pan and zoom it locally."
        ),
    )
    return parser


def require_sklearn():
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "scikit-learn is required for 2D t-SNE clustering. "
            "Install it first, for example: pip install scikit-learn"
        ) from exc
    return StandardScaler, PCA, TSNE, KMeans


def require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pyarrow is required to read LeRobot parquet datasets. "
            "Install it first, for example: pip install pyarrow"
        ) from exc
    return pq


def has_display_server() -> bool:
    if os.name != "posix":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def require_matplotlib(interactive: bool = False):
    try:
        import matplotlib

        use_interactive = bool(interactive and has_display_server())
        if not use_interactive:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "matplotlib is required to save 2D plots. "
            "Install it first, for example: pip install matplotlib"
        ) from exc
    if interactive and not use_interactive:
        print(
            "[WARN] --show-plot requested, but no GUI display was detected. "
            "Falling back to non-interactive plot saving."
        )
    return plt


def detect_dataset_format(dataset_path: Path, requested: str) -> str:
    resolved = dataset_path.expanduser().resolve()
    if requested != "auto":
        return requested
    if resolved.is_file() and resolved.suffix == ".npz":
        return "npz"
    if resolved.is_dir() and (resolved / "meta/info.json").exists():
        return "lerobot"
    raise ValueError(
        "Could not infer dataset format. Expected either a .npz file or a "
        f"LeRobot dataset directory with meta/info.json. Got: {resolved}"
    )


def ensure_trajectory_array(values: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Trajectory must be 2D [T, D], got shape={array.shape}.")
    if array.shape[0] == 0:
        raise ValueError("Trajectory must contain at least one timestep.")
    return array


def select_episode_positions(total: int, max_episodes: int | None, seed: int) -> np.ndarray:
    if total <= 0:
        raise ValueError("Dataset has no episodes.")
    if max_episodes is None or max_episodes >= total:
        return np.arange(total, dtype=np.int64)
    if max_episodes <= 0:
        raise ValueError(f"--max-episodes must be positive, got {max_episodes}.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total, size=max_episodes, replace=False)
    return np.sort(chosen.astype(np.int64))


def compute_signature_for_window(
    window: np.ndarray,
    sig_depth: int,
    resolved_backend: str,
) -> np.ndarray:
    if resolved_backend == "signatory":
        return compute_signatory_signature_np(window, sig_depth)
    return compute_simple_signature_np(window, sig_depth)


def build_signature_window(
    trajectory: np.ndarray,
    end_index: int,
    window_size: int,
) -> np.ndarray:
    if window_size <= 0:
        return trajectory[: end_index + 1]

    start_index = max(0, end_index - window_size + 1)
    window = trajectory[start_index : end_index + 1]
    if window.shape[0] < window_size:
        pad = np.repeat(trajectory[:1], window_size - window.shape[0], axis=0)
        window = np.concatenate([pad, window], axis=0)
    return window


def pool_signature_sequence(signature_sequence: np.ndarray, pool: str) -> np.ndarray:
    if signature_sequence.ndim == 1:
        return signature_sequence.astype(np.float32, copy=False)
    if signature_sequence.ndim != 2:
        raise ValueError(
            f"Signature sequence must be 1D or 2D, got shape={signature_sequence.shape}."
        )
    if pool == "last":
        return signature_sequence[-1].astype(np.float32, copy=False)
    if pool == "mean":
        return signature_sequence.mean(axis=0, dtype=np.float32).astype(
            np.float32, copy=False
        )
    if pool == "flatten":
        return signature_sequence.reshape(-1).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported signature pool: {pool}")


def compute_signature_feature_from_states(
    states: np.ndarray,
    pool: str,
    window_size: int,
    sig_depth: int,
    signature_backend: str,
) -> np.ndarray:
    trajectory = ensure_trajectory_array(states)
    resolved_backend = (
        signature_backend
        if signature_backend in {"signatory", "simple"}
        else resolve_signature_backend(signature_backend)
    )

    if pool == "last":
        window = build_signature_window(
            trajectory=trajectory,
            end_index=int(trajectory.shape[0] - 1),
            window_size=window_size,
        )
        return compute_signature_for_window(window, sig_depth, resolved_backend)

    signatures: list[np.ndarray] = []
    for time_index in range(trajectory.shape[0]):
        window = build_signature_window(
            trajectory=trajectory,
            end_index=int(time_index),
            window_size=window_size,
        )
        signatures.append(
            compute_signature_for_window(window, sig_depth, resolved_backend)
        )

    signature_sequence = np.stack(signatures, axis=0).astype(np.float32, copy=False)
    return pool_signature_sequence(signature_sequence, pool)


def resolve_output_dir(dataset_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    dataset_name = dataset_path.stem if dataset_path.is_file() else dataset_path.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_OUTPUT_ROOT / f"{dataset_name}_{timestamp}").resolve()


def resolve_npz_label_texts(
    data: Any,
    label_source: str,
    selected_positions: np.ndarray,
) -> tuple[list[str] | None, str | None]:
    total = len(selected_positions)
    if label_source == "none":
        return None, None

    if label_source == "auto":
        for candidate in ("goal", "task", "success", "episode"):
            labels, resolved = resolve_npz_label_texts(data, candidate, selected_positions)
            if labels is not None:
                return labels, resolved
        return None, None

    if label_source == "goal" and "target_goal_names" in data:
        values = np.asarray(data["target_goal_names"])[selected_positions]
        return [str(value) for value in values.tolist()], "target_goal_names"

    if label_source == "task" and "task_ids" in data:
        values = np.asarray(data["task_ids"], dtype=np.int64)[selected_positions]
        return [f"task_{int(value)}" for value in values.tolist()], "task_ids"

    if label_source == "success" and "success" in data:
        values = np.asarray(data["success"], dtype=bool)[selected_positions]
        return [f"success_{int(bool(value))}" for value in values.tolist()], "success"

    if label_source == "episode":
        if "episode_ids" in data:
            values = np.asarray(data["episode_ids"], dtype=np.int64)[selected_positions]
        else:
            values = selected_positions
        return [f"episode_{int(value)}" for value in values.tolist()], "episode"

    if total > 0:
        return None, None
    return None, None


def load_npz_episode_features(args: argparse.Namespace) -> LoadedEpisodeFeatures:
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as data:
        signature_keys = [key for key in (args.signature_key, "path_signatures", DEFAULT_SIGNATURE_KEY) if key]
        state_keys = [key for key in (args.state_key, "observations", DEFAULT_STATE_KEY) if key]

        signature_key = next((key for key in signature_keys if key in data), None)
        state_key = next((key for key in state_keys if key in data), None)

        if signature_key is not None:
            total_episodes = int(np.asarray(data[signature_key]).shape[0])
        elif state_key is not None:
            total_episodes = int(np.asarray(data[state_key]).shape[0])
        else:
            raise KeyError(
                "Could not find path signatures or trajectories in the npz file. "
                f"Tried signature keys={signature_keys} and state keys={state_keys}."
            )

        selected_positions = select_episode_positions(
            total=total_episodes,
            max_episodes=args.max_episodes,
            seed=args.seed,
        )

        use_dataset_signatures = (
            args.feature_source != "compute" and signature_key is not None
        )
        if args.feature_source == "dataset" and signature_key is None:
            raise KeyError(
                "Requested --feature-source dataset, but the npz file does not "
                "contain a path-signature array."
            )

        if use_dataset_signatures:
            signature_array = np.asarray(data[signature_key], dtype=np.float32)[
                selected_positions
            ]
            if signature_array.ndim == 2:
                features = signature_array.astype(np.float32, copy=False)
            elif signature_array.ndim == 3:
                features = np.stack(
                    [
                        pool_signature_sequence(signature_array[index], args.signature_pool)
                        for index in range(signature_array.shape[0])
                    ],
                    axis=0,
                ).astype(np.float32, copy=False)
            else:
                raise ValueError(
                    f"Unexpected signature array shape in npz: {signature_array.shape}"
                )
            feature_source = "dataset"
        else:
            if state_key is None:
                raise KeyError(
                    "Need state trajectories to compute path signatures, but no state "
                    "array was found in the npz file."
                )
            state_array = np.asarray(data[state_key], dtype=np.float32)[selected_positions]
            if state_array.ndim == 2:
                state_array = state_array[None, ...]
            if state_array.ndim != 3:
                raise ValueError(
                    "State trajectory array in npz must have shape [N, T, D]. "
                    f"Got: {state_array.shape}"
                )
            resolved_backend = resolve_signature_backend(args.signature_backend)
            features = np.stack(
                [
                    compute_signature_feature_from_states(
                        states=state_array[index],
                        pool=args.signature_pool,
                        window_size=args.path_signature_window_size,
                        sig_depth=args.path_signature_depth,
                        signature_backend=resolved_backend,
                    )
                    for index in range(state_array.shape[0])
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            feature_source = "computed"

        label_texts, resolved_label_source = resolve_npz_label_texts(
            data=data,
            label_source=args.label_source,
            selected_positions=selected_positions,
        )
        if "episode_ids" in data:
            episode_indices = np.asarray(data["episode_ids"], dtype=np.int64)[
                selected_positions
            ]
        else:
            episode_indices = selected_positions.astype(np.int64, copy=False)

        metadata: dict[str, Any] = {
            "dataset_path": str(dataset_path),
            "available_keys": sorted(data.files),
            "selected_episodes": int(len(selected_positions)),
            "total_episodes": int(total_episodes),
        }
        for key in (
            "path_signature_depth",
            "path_signature_window_size",
            "path_signature_backend",
            "path_signature_key",
            "t_fixed",
            "format_version",
        ):
            if key in data:
                raw = np.asarray(data[key])
                metadata[key] = raw.tolist() if raw.ndim > 0 else raw.item()

    return LoadedEpisodeFeatures(
        features=features,
        episode_indices=episode_indices,
        label_texts=label_texts,
        label_source=resolved_label_source,
        dataset_format="npz",
        feature_source=feature_source,
        signature_key=signature_key if feature_source == "dataset" else None,
        state_key=state_key,
        metadata=metadata,
    )


def arrow_column_to_numpy(column: Any, dtype: Any | None = None) -> np.ndarray:
    array = column.combine_chunks()
    list_size = getattr(array.type, "list_size", None)
    if list_size is not None:
        try:
            flat = array.values.to_numpy(zero_copy_only=False)
            result = np.asarray(flat, dtype=dtype).reshape(len(array), int(list_size))
        except Exception:
            result = np.asarray(array.to_pylist(), dtype=dtype)
        return result

    try:
        result = array.to_numpy(zero_copy_only=False)
    except Exception:
        result = np.asarray(array.to_pylist())
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return np.asarray(result)


def resolve_lerobot_label_text(
    label_source: str,
    episode_index: int,
    task_values: np.ndarray | None,
    success_values: np.ndarray | None,
) -> tuple[str | None, str | None]:
    if label_source == "none":
        return None, None
    if label_source == "auto":
        if task_values is not None:
            return f"task_{int(task_values[0])}", "task_index"
        return None, None
    if label_source == "task":
        if task_values is None:
            return None, None
        return f"task_{int(task_values[0])}", "task_index"
    if label_source == "success":
        if success_values is None:
            return None, None
        return f"success_{int(bool(success_values[-1]))}", "next.success"
    if label_source == "episode":
        return f"episode_{episode_index}", "episode_index"
    if label_source == "goal":
        return None, None
    raise ValueError(f"Unsupported label source: {label_source}")


def load_lerobot_episode_features(args: argparse.Namespace) -> LoadedEpisodeFeatures:
    pq = require_pyarrow_parquet()

    dataset_root = args.dataset_path.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    info_path = dataset_root / "meta/info.json"
    episodes_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    if not info_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(
            "Expected a LeRobot dataset root with meta/info.json and "
            f"meta/episodes/chunk-000/file-000.parquet. Got: {dataset_root}"
        )

    info = json.loads(info_path.read_text(encoding="utf-8"))
    dataset_signature_key = None
    if isinstance(info.get("path_signature"), dict):
        dataset_signature_key = str(info["path_signature"].get("key", "")) or None
    signature_key = args.signature_key or dataset_signature_key or DEFAULT_SIGNATURE_KEY
    state_key = args.state_key

    episode_table = pq.read_table(
        episodes_path,
        columns=["episode_index", "data/chunk_index", "data/file_index"],
    )
    episode_indices_all = arrow_column_to_numpy(
        episode_table.column("episode_index"), dtype=np.int64
    )
    chunk_indices_all = arrow_column_to_numpy(
        episode_table.column("data/chunk_index"), dtype=np.int64
    )
    file_indices_all = arrow_column_to_numpy(
        episode_table.column("data/file_index"), dtype=np.int64
    )

    selected_positions = select_episode_positions(
        total=int(episode_indices_all.shape[0]),
        max_episodes=args.max_episodes,
        seed=args.seed,
    )
    use_dataset_signatures = (
        args.feature_source != "compute"
        and isinstance(info.get("features"), dict)
        and signature_key in info["features"]
    )
    if args.feature_source == "dataset" and not use_dataset_signatures:
        raise KeyError(
            "Requested --feature-source dataset, but the LeRobot dataset metadata "
            f"does not advertise the signature key {signature_key!r}."
        )

    features: list[np.ndarray] = []
    episode_indices: list[int] = []
    label_texts: list[str] = []
    resolved_label_source: str | None = None
    progress_interval = max(1, min(25, int(selected_positions.shape[0])))
    resolved_backend = (
        None
        if use_dataset_signatures
        else resolve_signature_backend(args.signature_backend)
    )

    for loop_index, position in enumerate(selected_positions.tolist(), start=1):
        episode_index = int(episode_indices_all[position])
        chunk_index = int(chunk_indices_all[position])
        file_index = int(file_indices_all[position])
        relative_data_path = info["data_path"].format(
            chunk_index=chunk_index,
            file_index=file_index,
        )
        data_path = dataset_root / relative_data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data parquet for episode {episode_index}: {data_path}")

        columns = ["episode_index"]
        columns.append(signature_key if use_dataset_signatures else state_key)
        if args.label_source in {"auto", "task"}:
            columns.append("task_index")
        if args.label_source == "success":
            columns.append("next.success")

        data_table = pq.read_table(data_path, columns=list(dict.fromkeys(columns)))
        file_episode_indices = arrow_column_to_numpy(
            data_table.column("episode_index"), dtype=np.int64
        )
        mask = file_episode_indices == episode_index
        if not np.any(mask):
            raise ValueError(
                f"Episode {episode_index} has no rows in parquet file {data_path}."
            )

        if use_dataset_signatures:
            signature_sequence = arrow_column_to_numpy(
                data_table.column(signature_key),
                dtype=np.float32,
            )[mask]
            feature = pool_signature_sequence(signature_sequence, args.signature_pool)
        else:
            state_sequence = arrow_column_to_numpy(
                data_table.column(state_key),
                dtype=np.float32,
            )[mask]
            feature = compute_signature_feature_from_states(
                states=state_sequence,
                pool=args.signature_pool,
                window_size=args.path_signature_window_size,
                sig_depth=args.path_signature_depth,
                signature_backend=str(resolved_backend),
            )

        task_values = None
        if "task_index" in data_table.column_names:
            task_values = arrow_column_to_numpy(
                data_table.column("task_index"),
                dtype=np.int64,
            )[mask]
        success_values = None
        if "next.success" in data_table.column_names:
            success_values = arrow_column_to_numpy(
                data_table.column("next.success"),
                dtype=bool,
            )[mask]

        label_text, label_source = resolve_lerobot_label_text(
            label_source=args.label_source,
            episode_index=episode_index,
            task_values=task_values,
            success_values=success_values,
        )
        if label_text is not None:
            label_texts.append(label_text)
        if resolved_label_source is None and label_source is not None:
            resolved_label_source = label_source

        features.append(feature.astype(np.float32, copy=False))
        episode_indices.append(episode_index)

        if (
            loop_index == 1
            or loop_index == len(selected_positions)
            or (loop_index % progress_interval) == 0
        ):
            print(
                "[info] loaded lerobot episodes: "
                f"{loop_index}/{len(selected_positions)}"
            )

    label_texts_or_none = (
        label_texts if len(label_texts) == len(features) and resolved_label_source else None
    )

    metadata = {
        "dataset_path": str(dataset_root),
        "selected_episodes": int(len(selected_positions)),
        "total_episodes": int(len(episode_indices_all)),
        "path_signature_meta": info.get("path_signature"),
        "robot_type": info.get("robot_type"),
    }

    return LoadedEpisodeFeatures(
        features=np.stack(features, axis=0).astype(np.float32, copy=False),
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        label_texts=label_texts_or_none,
        label_source=resolved_label_source,
        dataset_format="lerobot",
        feature_source="dataset" if use_dataset_signatures else "computed",
        signature_key=signature_key if use_dataset_signatures else None,
        state_key=state_key,
        metadata=metadata,
    )


def load_episode_features(args: argparse.Namespace) -> LoadedEpisodeFeatures:
    dataset_format = detect_dataset_format(args.dataset_path, args.input_format)
    if dataset_format == "npz":
        return load_npz_episode_features(args)
    if dataset_format == "lerobot":
        return load_lerobot_episode_features(args)
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def fit_tsne_embedding(
    features: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    StandardScaler, PCA, TSNE, KMeans = require_sklearn()
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D [N, D], got shape={features.shape}.")
    if features.shape[0] < 3:
        raise ValueError(
            "Need at least 3 episodes to build a meaningful 2D t-SNE embedding. "
            f"Got {features.shape[0]}."
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca_dim = min(
        int(args.pca_components),
        int(scaled.shape[1]),
        max(1, int(scaled.shape[0] - 1)),
    )
    transformed = scaled
    if pca_dim > 0 and pca_dim < scaled.shape[1]:
        transformed = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(
            scaled
        )

    resolved_perplexity = min(float(args.perplexity), float(max(1, scaled.shape[0] - 1)))
    learning_rate: float | str = "auto" if float(args.learning_rate) <= 0 else float(args.learning_rate)

    tsne_kwargs: dict[str, Any] = {
        "n_components": 2,
        "perplexity": resolved_perplexity,
        "learning_rate": learning_rate,
        "init": "pca",
        "random_state": args.seed,
    }
    tsne_signature = inspect.signature(TSNE)
    if "max_iter" in tsne_signature.parameters:
        tsne_kwargs["max_iter"] = int(args.iterations)
    else:
        tsne_kwargs["n_iter"] = int(args.iterations)

    embedding = TSNE(**tsne_kwargs).fit_transform(transformed).astype(
        np.float32, copy=False
    )

    cluster_ids: np.ndarray | None = None
    if args.num_clusters > 0:
        num_clusters = min(int(args.num_clusters), int(features.shape[0]))
        cluster_ids = KMeans(
            n_clusters=num_clusters,
            random_state=args.seed,
            n_init=10,
        ).fit_predict(scaled).astype(np.int64, copy=False)

    return embedding, transformed.astype(np.float32, copy=False), cluster_ids


def save_embedding_csv(
    output_path: Path,
    embedding: np.ndarray,
    loaded: LoadedEpisodeFeatures,
    cluster_ids: np.ndarray | None,
) -> None:
    fieldnames = ["episode_index", "tsne_x", "tsne_y"]
    if loaded.label_texts is not None:
        fieldnames.append("label")
    if cluster_ids is not None:
        fieldnames.append("cluster_id")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(embedding.shape[0]):
            row: dict[str, Any] = {
                "episode_index": int(loaded.episode_indices[row_index]),
                "tsne_x": float(embedding[row_index, 0]),
                "tsne_y": float(embedding[row_index, 1]),
            }
            if loaded.label_texts is not None:
                row["label"] = loaded.label_texts[row_index]
            if cluster_ids is not None:
                row["cluster_id"] = int(cluster_ids[row_index])
            writer.writerow(row)


def configure_2d_axes(ax: Any, embedding: np.ndarray) -> None:
    mins = embedding.min(axis=0).astype(np.float64)
    maxs = embedding.max(axis=0).astype(np.float64)
    ranges = np.maximum(maxs - mins, 1e-6)
    padding = np.maximum(ranges * 0.08, 1e-3)

    ax.set_xlim(mins[0] - padding[0], maxs[0] + padding[0])
    ax.set_ylim(mins[1] - padding[1], maxs[1] + padding[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def plot_embedding_2d(
    output_path: Path,
    embedding: np.ndarray,
    color_labels: list[str],
    title: str,
    marker_size: float,
    figure_width: float,
    figure_height: float,
    figure_dpi: int,
    plt: Any,
    close_figure: bool = True,
) -> Any:

    unique_labels = list(dict.fromkeys(color_labels))
    color_map = plt.cm.get_cmap("tab20", max(1, len(unique_labels)))
    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_subplot(111)

    for label_index, label in enumerate(unique_labels):
        mask = np.asarray([value == label for value in color_labels], dtype=bool)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=marker_size,
            alpha=0.82,
            color=color_map(label_index),
            label=label,
            edgecolors="none",
        )

    configure_2d_axes(ax, embedding)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
    )
    fig.subplots_adjust(left=0.05, right=0.78, bottom=0.06, top=0.92)
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    if close_figure:
        plt.close(fig)
    return fig


def count_values(values: list[str] | np.ndarray | None) -> dict[str, int]:
    if values is None:
        return {}
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_cluster_label_breakdown(
    cluster_ids: np.ndarray | None,
    label_texts: list[str] | None,
) -> dict[str, dict[str, int]]:
    if cluster_ids is None or label_texts is None:
        return {}
    breakdown: dict[str, dict[str, int]] = {}
    for cluster_id, label_text in zip(cluster_ids.tolist(), label_texts, strict=False):
        cluster_key = str(int(cluster_id))
        cluster_counts = breakdown.setdefault(cluster_key, {})
        cluster_counts[label_text] = cluster_counts.get(label_text, 0) + 1
    return breakdown


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.skip_plot and args.show_plot:
        parser.error("--show-plot cannot be used together with --skip-plot.")
    args.dataset_path = args.dataset_path.expanduser().resolve()
    output_dir = resolve_output_dir(args.dataset_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_episode_features(args)
    print(
        "[info] episode features ready: "
        f"format={loaded.dataset_format}, "
        f"source={loaded.feature_source}, "
        f"episodes={loaded.features.shape[0]}, "
        f"feature_dim={loaded.features.shape[1]}"
    )

    embedding, transformed, cluster_ids = fit_tsne_embedding(loaded.features, args)
    embedding_csv_path = output_dir / "tsne_embedding.csv"
    save_embedding_csv(embedding_csv_path, embedding, loaded, cluster_ids)

    plot_by_label_path: str | None = None
    plot_by_cluster_path: str | None = None
    if not args.skip_plot:
        plt = require_matplotlib(interactive=args.show_plot)
        close_figure = not args.show_plot
        if loaded.label_texts is not None:
            plot_by_label_path = str(output_dir / "tsne_2d_by_label.png")
            plot_embedding_2d(
                output_path=Path(plot_by_label_path),
                embedding=embedding,
                color_labels=loaded.label_texts,
                title="2D t-SNE of Path Signatures by Label",
                marker_size=args.marker_size,
                figure_width=args.figure_width,
                figure_height=args.figure_height,
                figure_dpi=args.figure_dpi,
                plt=plt,
                close_figure=close_figure,
            )
        if cluster_ids is not None:
            plot_by_cluster_path = str(output_dir / "tsne_2d_by_cluster.png")
            plot_embedding_2d(
                output_path=Path(plot_by_cluster_path),
                embedding=embedding,
                color_labels=[f"cluster_{int(cluster_id)}" for cluster_id in cluster_ids],
                title="2D t-SNE of Path Signatures by KMeans Cluster",
                marker_size=args.marker_size,
                figure_width=args.figure_width,
                figure_height=args.figure_height,
                figure_dpi=args.figure_dpi,
                plt=plt,
                close_figure=close_figure,
            )
        if args.show_plot and plt.get_fignums():
            print("[info] showing interactive 2D plot window")
            plt.show()

    summary = {
        "dataset_path": str(args.dataset_path),
        "output_dir": str(output_dir),
        "dataset_format": loaded.dataset_format,
        "feature_source": loaded.feature_source,
        "signature_key": loaded.signature_key,
        "state_key": loaded.state_key,
        "label_source": loaded.label_source,
        "num_episodes": int(loaded.features.shape[0]),
        "feature_dim": int(loaded.features.shape[1]),
        "transformed_dim": int(transformed.shape[1]),
        "embedding_dim": int(embedding.shape[1]),
        "signature_pool": args.signature_pool,
        "path_signature_window_size": int(args.path_signature_window_size),
        "path_signature_depth": int(args.path_signature_depth),
        "signature_backend": args.signature_backend,
        "perplexity": float(min(args.perplexity, max(1, loaded.features.shape[0] - 1))),
        "learning_rate": "auto" if args.learning_rate <= 0 else float(args.learning_rate),
        "iterations": int(args.iterations),
        "pca_components": int(args.pca_components),
        "num_clusters": 0 if cluster_ids is None else int(np.unique(cluster_ids).shape[0]),
        "label_counts": count_values(loaded.label_texts),
        "cluster_counts": count_values(
            None if cluster_ids is None else [str(int(value)) for value in cluster_ids]
        ),
        "cluster_label_breakdown": build_cluster_label_breakdown(
            cluster_ids=cluster_ids,
            label_texts=loaded.label_texts,
        ),
        "artifacts": {
            "embedding_csv": str(embedding_csv_path),
            "plot_by_label": plot_by_label_path,
            "plot_by_cluster": plot_by_cluster_path,
        },
        "dataset_metadata": loaded.metadata,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[info] saved embedding csv: {embedding_csv_path}")
    print(f"[info] saved summary: {summary_path}")
    if not args.skip_plot:
        if plot_by_label_path is not None:
            print(f"[info] saved label plot: {plot_by_label_path}")
        if plot_by_cluster_path is not None:
            print(f"[info] saved cluster plot: {plot_by_cluster_path}")


if __name__ == "__main__":
    main()
