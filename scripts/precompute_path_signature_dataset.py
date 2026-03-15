#!/usr/bin/env python3
"""Add prefix-history log-signature to a local LeRobot-format dataset.

This script:
1. Loads LeRobot parquet data with `datasets.load_dataset`.
2. Builds a per-frame prefix history over `observation.state` within each episode.
3. Computes log-signature features using `signatory`.
4. Adds a new column `observation.path_signature`.
5. Writes a new LeRobot dataset directory preserving chunked parquet layout.
6. Updates `meta/info.json` (feature spec + signature metadata) and `meta/stats.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import Dataset, Sequence, Value, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline LeRobot path-signature preprocessor.")
    parser.add_argument("--input-root", type=Path, required=True, help="Input LeRobot dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output LeRobot dataset root.")
    parser.add_argument("--state-key", type=str, default="observation.state")
    parser.add_argument("--output-key", type=str, default="observation.path_signature")
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="History window size. Use 0 to include all frames up to the current frame.",
    )
    parser.add_argument("--sig-depth", type=int, default=3)
    parser.add_argument("--map-batch-size", type=int, default=256)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument(
        "--signature-backend",
        type=str,
        default="auto",
        choices=["auto", "signatory", "simple"],
        help=(
            "Signature backend. "
            "'auto': try signatory then fallback to simple if unavailable/unstable; "
            "'signatory': force signatory (error if unusable); "
            "'simple': pure-numpy fallback features."
        ),
    )
    parser.add_argument(
        "--symlink-media",
        action="store_true",
        help="Symlink videos/images into output dataset instead of copying.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_data_parquet_files(dataset_root: Path) -> list[Path]:
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under: {dataset_root / 'data'}")
    return data_files


def load_data_dataset(data_files: list[Path]) -> Dataset:
    return load_dataset(
        "parquet",
        data_files={"train": [str(p) for p in data_files]},
        split="train",
    )


def build_episode_state_cache(
    dataset: Dataset,
    state_key: str,
) -> tuple[dict[int, np.ndarray], np.ndarray, int]:
    """Build in-memory episode states and global-row->position mapping.

    Returns:
        episode_states: {episode_index: np.ndarray[T_ep, state_dim]}
        row_pos_in_episode: np.ndarray[num_rows] giving local timestep within episode.
        state_dim: dimensionality of observation.state.
    """
    required_cols = {"episode_index", state_key}
    missing = required_cols - set(dataset.column_names)
    if missing:
        raise KeyError(f"Missing required columns in dataset: {sorted(missing)}")

    view = dataset.select_columns(["episode_index", state_key])
    num_rows = len(view)

    episode_state_lists: dict[int, list[np.ndarray]] = {}
    row_pos_in_episode = np.empty((num_rows,), dtype=np.int64)
    state_dim = None

    for idx, row in enumerate(view):
        ep_idx = int(row["episode_index"])
        state = np.asarray(row[state_key], dtype=np.float32)
        if state.ndim != 1:
            raise ValueError(
                f"Expected `{state_key}` to be 1D per row, got shape={state.shape} at idx={idx}."
            )

        if state_dim is None:
            state_dim = int(state.shape[0])
        elif state.shape[0] != state_dim:
            raise ValueError(
                f"Inconsistent state dim at idx={idx}: {state.shape[0]} vs expected {state_dim}."
            )

        ep_states = episode_state_lists.setdefault(ep_idx, [])
        row_pos_in_episode[idx] = len(ep_states)
        ep_states.append(state)

        if (idx + 1) % 200000 == 0:
            print(f"[cache] scanned {idx + 1}/{num_rows} rows")

    if state_dim is None:
        raise RuntimeError("Dataset appears empty; no rows found.")

    episode_states = {ep: np.stack(states, axis=0) for ep, states in episode_state_lists.items()}
    return episode_states, row_pos_in_episode, state_dim


def compute_logsignature_np(window: np.ndarray, sig_depth: int) -> np.ndarray:
    """Compute log-signature for one window.

    Args:
        window: (window_size, state_dim) float32 array
        sig_depth: truncation depth for log-signature
    Returns:
        (sig_dim,) float32 array
    """
    if window.ndim != 2:
        raise ValueError(f"Window must be 2D, got shape={window.shape}")

    try:
        import signatory
    except ImportError as exc:
        raise ImportError(
            "`signatory` is required. Install it first (example: pip install signatory)."
        ) from exc

    path = torch.from_numpy(window).unsqueeze(0)  # (1, T, C)
    with torch.no_grad():
        logsig = signatory.logsignature(path, depth=sig_depth)  # (1, sig_dim)
    return logsig.squeeze(0).cpu().numpy().astype(np.float32)


def compute_simple_signature_np(window: np.ndarray, sig_depth: int) -> np.ndarray:
    """Fallback signature-like feature extractor when signatory is unusable.

    This is not a true log-signature; it computes per-dimension moments of increments:
        [sum(dx), sum(dx^2), ..., sum(dx^depth)].
    Output dim = state_dim * sig_depth.
    """
    if window.ndim != 2:
        raise ValueError(f"Window must be 2D, got shape={window.shape}")
    if sig_depth <= 0:
        raise ValueError(f"sig_depth must be > 0, got {sig_depth}")

    deltas = np.diff(window, axis=0, prepend=window[:1]).astype(np.float32)
    feats = [np.sum(np.power(deltas, k, dtype=np.float32), axis=0) for k in range(1, sig_depth + 1)]
    return np.concatenate(feats, axis=0).astype(np.float32)


def check_signatory_usable() -> tuple[bool, str]:
    """Run a tiny subprocess probe to avoid crashing main process on signatory segfault."""
    probe = (
        "import torch\n"
        "import signatory\n"
        "x = torch.randn(1, 8, 2)\n"
        "y = signatory.logsignature(x, depth=2)\n"
        "print(tuple(y.shape))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ok = proc.returncode == 0
    detail = proc.stderr.strip() if proc.stderr.strip() else proc.stdout.strip()
    if not detail:
        detail = f"probe exited with returncode={proc.returncode}"
    return ok, detail


def resolve_signature_backend(requested_backend: str) -> str:
    if requested_backend == "simple":
        return "simple"

    ok, detail = check_signatory_usable()
    if requested_backend == "signatory":
        if not ok:
            raise RuntimeError(
                "signatory backend requested but precheck failed. "
                f"Detail: {detail or 'unknown error'}"
            )
        return "signatory"

    # auto mode
    if ok:
        return "signatory"
    print(
        "[WARN] signatory precheck failed; falling back to simple backend. "
        f"Detail: {detail or 'unknown error'}"
    )
    return "simple"


def build_signature_mapper(
    episode_states: dict[int, np.ndarray],
    row_pos_in_episode: np.ndarray,
    window_size: int,
    sig_depth: int,
    output_key: str,
    signature_backend: str,
):
    if sig_depth <= 0:
        raise ValueError(f"sig_depth must be > 0, got {sig_depth}")

    def map_batch(batch: dict, indices: list[int]) -> dict[str, list[list[float]]]:
        ep_indices = batch["episode_index"]
        signatures: list[list[float]] = []

        for abs_row_idx, ep_raw in zip(indices, ep_indices, strict=True):
            ep_idx = int(ep_raw)
            ep_states = episode_states[ep_idx]
            local_t = int(row_pos_in_episode[abs_row_idx])

            if window_size <= 0:
                window = ep_states[: local_t + 1]
            else:
                start = max(0, local_t - window_size + 1)
                window = ep_states[start : local_t + 1]
                if window.shape[0] < window_size:
                    pad_len = window_size - window.shape[0]
                    pad = np.repeat(ep_states[0:1], pad_len, axis=0)
                    window = np.concatenate([pad, window], axis=0)

            if signature_backend == "signatory":
                sig = compute_logsignature_np(window, sig_depth)
            else:
                sig = compute_simple_signature_np(window, sig_depth)
            signatures.append(sig.tolist())

        return {output_key: signatures}

    return map_batch


def compute_column_stats(dataset: Dataset, key: str, batch_size: int = 2048) -> dict:
    """Compute min/max/mean/std/count for one vector feature column."""
    total_count = 0
    sum_vec = None
    sum_sq_vec = None
    min_vec = None
    max_vec = None

    for batch in dataset.iter(batch_size=batch_size):
        arr = np.asarray(batch[key], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected {key} to be 2D in batched iteration, got shape={arr.shape}")

        if sum_vec is None:
            sum_vec = np.zeros((arr.shape[1],), dtype=np.float64)
            sum_sq_vec = np.zeros((arr.shape[1],), dtype=np.float64)
            min_vec = np.full((arr.shape[1],), np.inf, dtype=np.float32)
            max_vec = np.full((arr.shape[1],), -np.inf, dtype=np.float32)

        total_count += arr.shape[0]
        sum_vec += arr.sum(axis=0, dtype=np.float64)
        sum_sq_vec += np.square(arr, dtype=np.float64).sum(axis=0)
        min_vec = np.minimum(min_vec, arr.min(axis=0))
        max_vec = np.maximum(max_vec, arr.max(axis=0))

    if total_count == 0:
        raise RuntimeError(f"No rows found when computing stats for column `{key}`")

    mean = sum_vec / total_count
    var = np.maximum(sum_sq_vec / total_count - np.square(mean), 0.0)
    std = np.sqrt(var)

    return {
        "min": min_vec.tolist(),
        "max": max_vec.tolist(),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "count": int(total_count),
    }


def copy_or_link_tree(src: Path, dst: Path, symlink: bool) -> None:
    if not src.exists():
        return
    if symlink:
        try:
            os.symlink(src, dst, target_is_directory=True)
            return
        except OSError:
            pass
    shutil.copytree(src, dst)


def prepare_output_root(input_root: Path, output_root: Path, symlink_media: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    # Keep LeRobot metadata structure.
    shutil.copytree(input_root / "meta", output_root / "meta")

    # Preserve media dirs if they exist.
    for folder in ["videos", "images"]:
        src = input_root / folder
        dst = output_root / folder
        copy_or_link_tree(src, dst, symlink=symlink_media)

    # Optional top-level files.
    for file_name in ["README.md", ".gitattributes"]:
        src = input_root / file_name
        if src.exists():
            shutil.copy2(src, output_root / file_name)

    (output_root / "data").mkdir(parents=True, exist_ok=True)


def write_chunked_data(
    dataset: Dataset,
    input_root: Path,
    output_root: Path,
    input_data_files: list[Path],
) -> None:
    row_counts = [pq.read_metadata(str(p)).num_rows for p in input_data_files]
    if sum(row_counts) != len(dataset):
        raise RuntimeError(
            f"Row count mismatch: parquet rows={sum(row_counts)} vs dataset rows={len(dataset)}"
        )

    offset = 0
    for src, nrows in zip(input_data_files, row_counts, strict=True):
        rel = src.relative_to(input_root)
        out_file = output_root / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)

        shard = dataset.select(range(offset, offset + nrows))
        shard.to_parquet(str(out_file))
        offset += nrows


def update_info_json(
    output_root: Path,
    output_key: str,
    signature_dim: int,
    window_size: int,
    sig_depth: int,
) -> None:
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))

    info.setdefault("features", {})
    info["features"][output_key] = {
        "dtype": "float32",
        "shape": [int(signature_dim)],
        "names": [f"path_sig_{i}" for i in range(signature_dim)],
    }

    info["path_signature"] = {
        "key": output_key,
        "window_size": int(window_size),
        "window_mode": "full_prefix" if int(window_size) <= 0 else "sliding_window",
        "sig_depth": int(sig_depth),
        "signature_dim": int(signature_dim),
        "kind": "logsignature",
    }

    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")


def update_stats_json(output_root: Path, output_key: str, stats_value: dict) -> None:
    stats_path = output_root / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        stats = {}

    stats[output_key] = stats_value
    stats_path.write_text(json.dumps(stats, indent=4, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset root does not exist: {input_root}")
    if args.sig_depth <= 0:
        raise ValueError(f"sig_depth must be > 0, got {args.sig_depth}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    print(f"[1/7] loading data parquet from: {input_root}")
    data_files = list_data_parquet_files(input_root)
    dataset = load_data_dataset(data_files)
    print(f"Loaded {len(dataset)} rows from {len(data_files)} parquet files.")

    signature_backend = resolve_signature_backend(args.signature_backend)
    print(f"[info] signature backend: {signature_backend}")

    print("[2/7] building episode state cache")
    episode_states, row_pos_in_episode, state_dim = build_episode_state_cache(dataset, args.state_key)
    print(f"Cached {len(episode_states)} episodes, state_dim={state_dim}")

    window_label = "all_prefix" if args.window_size <= 0 else str(args.window_size)
    print(f"[3/7] computing path signatures with dataset.map (window={window_label})")
    mapper = build_signature_mapper(
        episode_states=episode_states,
        row_pos_in_episode=row_pos_in_episode,
        window_size=args.window_size,
        sig_depth=args.sig_depth,
        output_key=args.output_key,
        signature_backend=signature_backend,
    )
    map_kwargs = dict(
        batched=True,
        with_indices=True,
        batch_size=args.map_batch_size,
        load_from_cache_file=False,
        desc="Computing log-signatures",
    )
    # `datasets` may still spin worker subprocesses when num_proc is provided; avoid that for num_proc<=1.
    if args.num_proc and args.num_proc > 1:
        map_kwargs["num_proc"] = args.num_proc
    dataset = dataset.map(mapper, **map_kwargs)

    signature_dim = len(dataset[0][args.output_key])
    dataset = dataset.cast_column(
        args.output_key,
        Sequence(feature=Value(dtype="float32"), length=signature_dim),
    )
    print(f"Computed {args.output_key} with dim={signature_dim}")

    print("[4/7] computing signature stats")
    signature_stats = compute_column_stats(dataset, args.output_key)

    print(f"[5/7] preparing output dataset root: {output_root}")
    prepare_output_root(input_root, output_root, symlink_media=args.symlink_media)

    print("[6/7] writing chunked parquet data")
    write_chunked_data(dataset, input_root, output_root, data_files)

    print("[7/7] updating meta/info.json and meta/stats.json")
    update_info_json(
        output_root=output_root,
        output_key=args.output_key,
        signature_dim=signature_dim,
        window_size=args.window_size,
        sig_depth=args.sig_depth,
    )
    update_stats_json(output_root, args.output_key, signature_stats)

    print("Done.")
    print(f"Output dataset: {output_root}")
    print(f"Signature key: {args.output_key}")
    print(f"Signature dim: {signature_dim}")


if __name__ == "__main__":
    main()
