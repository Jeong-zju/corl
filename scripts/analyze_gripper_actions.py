from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def iter_parquet_paths(dataset_root: Path) -> list[Path]:
    paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}.")
    return paths


def resolve_dataset_root(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(DEFAULT_DATA_ROOT / str(value))
    for candidate in candidates:
        if (candidate / "meta" / "stats.json").exists() and (candidate / "data").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve LeRobot dataset root for {value!r}.")


def collect_action_values(
    *,
    dataset_root: Path,
    action_key: str,
    action_indices: tuple[int, ...],
    max_rows: int | None,
) -> dict[int, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "`pyarrow` is required to read LeRobot parquet files. Activate the "
            "project environment or install pyarrow before running this script."
        ) from exc

    collected: dict[int, list[np.ndarray]] = {idx: [] for idx in action_indices}
    rows_seen = 0
    for parquet_path in iter_parquet_paths(dataset_root):
        table = pq.read_table(parquet_path, columns=[action_key])
        actions = np.asarray(table.column(action_key).to_pylist(), dtype=np.float32)
        if actions.ndim != 2:
            raise RuntimeError(
                f"Expected `{action_key}` to be a 2D action array, got {actions.shape} "
                f"in {parquet_path}."
            )
        if max_rows is not None:
            remaining = int(max_rows) - rows_seen
            if remaining <= 0:
                break
            actions = actions[:remaining]
        for idx in action_indices:
            if idx < 0 or idx >= actions.shape[1]:
                raise IndexError(
                    f"Action index {idx} is outside action dim {actions.shape[1]}."
                )
            collected[idx].append(actions[:, idx].astype(np.float32, copy=False))
        rows_seen += actions.shape[0]
        if max_rows is not None and rows_seen >= int(max_rows):
            break
    return {idx: np.concatenate(chunks) for idx, chunks in collected.items()}


def kmeans_1d(values: np.ndarray, *, steps: int = 50) -> tuple[float, float]:
    centers = np.quantile(values.astype(np.float64), [0.1, 0.9]).astype(np.float64)
    for _ in range(int(steps)):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = distances.argmin(axis=1)
        next_centers = centers.copy()
        for label in (0, 1):
            mask = labels == label
            if np.any(mask):
                next_centers[label] = float(values[mask].mean())
        if np.allclose(next_centers, centers):
            break
        centers = next_centers
    low, high = sorted(float(value) for value in centers)
    return low, high


def thresholds_for_values(
    *,
    closed: float,
    opened: float,
    hysteresis_ratio: float,
) -> tuple[float, float]:
    direction = 1.0 if opened >= closed else -1.0
    low = direction * float(closed)
    high = direction * float(opened)
    midpoint = (low + high) * 0.5
    half_gap = abs(high - low) * float(hysteresis_ratio) * 0.5
    return direction * (midpoint - half_gap), direction * (midpoint + half_gap)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze discrete gripper action values from a LeRobot dataset."
    )
    parser.add_argument("dataset", help="Dataset root or ID under main/data.")
    parser.add_argument(
        "--action-key",
        default="action",
        help="Action column name in the LeRobot parquet files.",
    )
    parser.add_argument(
        "--action-indices",
        default="9,16",
        help="Comma-separated gripper action indices. Zeno default: 9,16.",
    )
    parser.add_argument(
        "--closed-is",
        choices=["low", "high"],
        default="low",
        help="Whether the lower or higher cluster center should be labeled closed.",
    )
    parser.add_argument(
        "--hysteresis-ratio",
        type=float,
        default=0.25,
        help="Fraction of closed/open span reserved as the keep-state band.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=5,
        help="Decimals used when reporting most common rounded values.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of rounded mode values to print per gripper.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for quick sampling.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset_root = resolve_dataset_root(args.dataset)
    action_indices = parse_csv_ints(args.action_indices)
    values_by_index = collect_action_values(
        dataset_root=dataset_root,
        action_key=str(args.action_key),
        action_indices=action_indices,
        max_rows=args.max_rows,
    )

    closed_values: list[float] = []
    open_values: list[float] = []
    close_thresholds: list[float] = []
    open_thresholds: list[float] = []
    summaries = []
    for idx in action_indices:
        values = values_by_index[idx]
        low, high = kmeans_1d(values)
        closed, opened = (low, high) if args.closed_is == "low" else (high, low)
        close_threshold, open_threshold = thresholds_for_values(
            closed=closed,
            opened=opened,
            hysteresis_ratio=float(args.hysteresis_ratio),
        )
        rounded, counts = np.unique(
            np.round(values, int(args.round_decimals)),
            return_counts=True,
        )
        order = np.argsort(counts)[::-1][: int(args.top_k)]
        top_values = [
            {"value": float(rounded[i]), "count": int(counts[i])}
            for i in order
        ]
        closed_values.append(float(closed))
        open_values.append(float(opened))
        close_thresholds.append(float(close_threshold))
        open_thresholds.append(float(open_threshold))
        summaries.append(
            {
                "action_index": int(idx),
                "count": int(values.shape[0]),
                "min": float(values.min()),
                "max": float(values.max()),
                "q01": float(np.quantile(values, 0.01)),
                "q50": float(np.quantile(values, 0.50)),
                "q99": float(np.quantile(values, 0.99)),
                "kmeans_low": float(low),
                "kmeans_high": float(high),
                "top_rounded_values": top_values,
            }
        )

    result = {
        "dataset_root": str(dataset_root),
        "action_key": str(args.action_key),
        "gripper_hysteresis": {
            "enabled": True,
            "action_indices": list(action_indices),
            "closed_values": closed_values,
            "open_values": open_values,
            "close_thresholds": close_thresholds,
            "open_thresholds": open_thresholds,
            "hysteresis_ratio": float(args.hysteresis_ratio),
            "initial_state": "current",
        },
        "summary": summaries,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nYAML snippet:")
    print("gripper_hysteresis:")
    for key, value in result["gripper_hysteresis"].items():
        print(f"  {key}: {json.dumps(value)}")


if __name__ == "__main__":
    main()
