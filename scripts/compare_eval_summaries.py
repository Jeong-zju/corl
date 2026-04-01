#!/usr/bin/env python3
"""Compare and visualize offline dataset-eval summaries across algorithms.

Examples:
    python scripts/compare_eval_summaries.py --dataset zeno-ai/day3_5_Exp1_processed
    python scripts/compare_eval_summaries.py \
        --summary outputs/eval/day3_5_Exp1_processed_act/20260401_112614/summary.json \
        --summary outputs/eval/day3_5_Exp1_processed_streaming_act/20260401_114643/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = REPO_ROOT / "main" / "outputs" / "eval"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "main" / "outputs" / "eval_compare"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    lower_is_better: bool


@dataclass
class EvalRun:
    label: str
    summary_path: Path
    series_name: str
    run_name: str
    policy_type: str
    policy_dir: str
    dataset_repo_id: str
    dataset_root: str
    eval_split: str
    split_source: str
    num_episodes: int
    num_steps: int
    action_dim: int
    metrics: dict[str, Any]
    results: list[dict[str, Any]]


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("mae", "MAE", True),
    MetricSpec("rmse", "RMSE", True),
    MetricSpec("mean_l2_error", "Mean L2 Error", True),
    MetricSpec("cosine_similarity", "Cosine Similarity", False),
)


def require_matplotlib(interactive: bool = False):
    try:
        mpl_config_dir = os.environ.get("MPLCONFIGDIR")
        if not mpl_config_dir:
            fallback_dir = Path(tempfile.gettempdir()) / "matplotlib-corl"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(fallback_dir)

        import matplotlib

        has_display = os.environ.get("DISPLAY") not in (None, "")
        if not interactive or not has_display:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "matplotlib is required to save comparison plots. "
            "Install it first, for example: pip install matplotlib"
        ) from exc

    if interactive and not has_display:
        print(
            "[WARN] --show-plot requested, but no GUI display was detected. "
            "Falling back to non-interactive plot saving."
        )
    return plt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple offline dataset-eval summary.json files and save "
            "tables plus comparison plots."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Dataset ID or path fragment used to auto-discover the latest summary.json "
            "under outputs/eval for each algorithm series."
        ),
    )
    input_group.add_argument(
        "--summary",
        action="append",
        default=None,
        help="Explicit path to a summary.json file. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional display label. Repeat once per selected summary in the final order.",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help="Root directory used by --dataset auto-discovery.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Base directory for auto-generated compare reports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output directory. Overrides --output-root.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plot generation and only export CSV/JSON reports.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show matplotlib windows after saving plots when a display is available.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG DPI for generated plots.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=11.0,
        help="Figure width in inches for line and bar plots.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=6.0,
        help="Figure height in inches for line and bar plots.",
    )
    args = parser.parse_args(argv)
    if args.skip_plot and args.show_plot:
        parser.error("--show-plot cannot be used together with --skip-plot.")
    return args


def normalize_dataset_candidates(value: str | Path | None) -> set[str]:
    if value is None:
        return set()

    raw = str(value).strip().replace("\\", "/")
    if not raw:
        return set()

    candidates: set[str] = set()

    def add(text: str | Path | None) -> None:
        if text is None:
            return
        normalized = str(text).strip().replace("\\", "/")
        if not normalized:
            return
        if normalized.startswith("./"):
            normalized = normalized[2:]
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        if normalized:
            candidates.add(normalized)

    raw_path = Path(raw).expanduser()
    add(raw)
    add(raw.replace("/", "_"))
    add(raw_path.name)
    if len(raw_path.parts) >= 2:
        add("_".join(raw_path.parts[-2:]))

    for prefix in ("main/data/", "data/"):
        if raw.startswith(prefix):
            stripped = raw[len(prefix) :]
            add(stripped)
            add(stripped.replace("/", "_"))
            add(Path(stripped).name)
            if len(Path(stripped).parts) >= 2:
                add("_".join(Path(stripped).parts[-2:]))

    for root in (REPO_ROOT / "main" / "data", REPO_ROOT / "main", REPO_ROOT):
        try:
            relative = raw_path.resolve(strict=False).relative_to(root.resolve())
        except Exception:
            continue
        add(relative.as_posix())
        add(relative.as_posix().replace("/", "_"))
        add(relative.name)
        if len(relative.parts) >= 2:
            add("_".join(relative.parts[-2:]))

    return candidates


def infer_series_name(summary_path: Path, eval_root: Path) -> str:
    try:
        relative = summary_path.resolve().relative_to(eval_root.resolve())
    except Exception:
        relative = summary_path
    if len(relative.parts) >= 3:
        return relative.parts[0]
    if len(summary_path.parents) >= 2:
        return summary_path.parents[1].name
    return summary_path.parent.name


def load_summary_file(summary_path: Path, eval_root: Path) -> EvalRun:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    results = data.get("results")
    metrics = data.get("metrics")
    if not isinstance(results, list) or not isinstance(metrics, dict):
        raise ValueError(
            "Expected dataset-eval summary with top-level `results` list and `metrics` mapping. "
            f"Got: {summary_path}"
        )

    series_name = infer_series_name(summary_path, eval_root)
    run_name = summary_path.parent.name
    return EvalRun(
        label=series_name,
        summary_path=summary_path.resolve(),
        series_name=series_name,
        run_name=run_name,
        policy_type=str(data.get("policy_type", "unknown")),
        policy_dir=str(data.get("policy_dir", "")),
        dataset_repo_id=str(data.get("dataset_repo_id", "")),
        dataset_root=str(data.get("dataset_root", "")),
        eval_split=str(data.get("eval_split", "")),
        split_source=str(data.get("split_source", "")),
        num_episodes=int(data.get("num_episodes", len(results))),
        num_steps=int(data.get("num_steps", 0)),
        action_dim=int(data.get("action_dim", 0)),
        metrics=metrics,
        results=results,
    )


def summary_matches_dataset(summary_data: dict[str, Any], dataset_selector: str) -> bool:
    selector_candidates = normalize_dataset_candidates(dataset_selector)
    if not selector_candidates:
        return False

    summary_candidates = set()
    summary_candidates.update(normalize_dataset_candidates(summary_data.get("dataset_repo_id")))
    summary_candidates.update(normalize_dataset_candidates(summary_data.get("dataset_root")))
    return bool(selector_candidates.intersection(summary_candidates))


def discover_latest_summaries(dataset_selector: str, eval_root: Path) -> list[Path]:
    if not eval_root.exists():
        raise FileNotFoundError(f"Eval root not found: {eval_root}")

    latest_by_series: dict[str, tuple[tuple[int, str], Path]] = {}
    matched = 0
    for summary_path in sorted(eval_root.glob("*/*/summary.json")):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not summary_matches_dataset(data, dataset_selector):
            continue

        matched += 1
        series_name = infer_series_name(summary_path, eval_root)
        run_dir = summary_path.parent
        key = (run_dir.stat().st_mtime_ns, run_dir.name)
        prev = latest_by_series.get(series_name)
        if prev is None or key > prev[0]:
            latest_by_series[series_name] = (key, summary_path.resolve())

    if not latest_by_series:
        raise FileNotFoundError(
            "No summary.json files matched the requested dataset.\n"
            f"- dataset={dataset_selector}\n"
            f"- eval_root={eval_root}"
        )

    print(
        "[info] matched summaries: "
        f"{matched} candidates across {len(latest_by_series)} series"
    )
    return [path for _, path in sorted(latest_by_series.values(), key=lambda item: item[1].as_posix())]


def build_default_labels(runs: list[EvalRun]) -> list[str]:
    if not runs:
        return []

    series_names = [run.series_name for run in runs]
    common_prefix = os.path.commonprefix(series_names)
    trimmed_prefix = common_prefix
    if "_" in common_prefix:
        trimmed_prefix = common_prefix[: common_prefix.rfind("_") + 1]

    candidate_labels = [
        name[len(trimmed_prefix) :] if trimmed_prefix and name.startswith(trimmed_prefix) else name
        for name in series_names
    ]
    if all(candidate_labels) and len(set(candidate_labels)) == len(candidate_labels):
        return candidate_labels

    policy_labels = [run.policy_type for run in runs]
    if all(policy_labels) and len(set(policy_labels)) == len(policy_labels):
        return policy_labels

    return series_names


def sanitize_name(text: str) -> str:
    result = []
    for char in text:
        if char.isalnum() or char in ("-", "_"):
            result.append(char)
        else:
            result.append("_")
    sanitized = "".join(result).strip("_")
    return sanitized or "compare"


def validate_runs(runs: list[EvalRun]) -> None:
    if not runs:
        raise ValueError("No evaluation summaries were selected.")

    dataset_ids = {run.dataset_repo_id for run in runs}
    dataset_roots = {run.dataset_root for run in runs}
    if len(dataset_ids) > 1 or len(dataset_roots) > 1:
        raise ValueError(
            "All selected summaries must come from the same dataset.\n"
            f"- dataset_repo_ids={sorted(dataset_ids)}\n"
            f"- dataset_roots={sorted(dataset_roots)}"
        )


def assign_labels(runs: list[EvalRun], labels: list[str] | None) -> None:
    if labels is not None and len(labels) != len(runs):
        raise ValueError(
            f"`--label` count must match the number of selected summaries. Got {len(labels)} labels for {len(runs)} runs."
        )

    resolved_labels = labels if labels is not None else build_default_labels(runs)
    for run, label in zip(runs, resolved_labels, strict=True):
        run.label = label


def default_output_dir(runs: list[EvalRun], output_root: Path) -> Path:
    dataset_tag = runs[0].dataset_repo_id or Path(runs[0].dataset_root).name or "dataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{sanitize_name(dataset_tag)}_{timestamp}"


def build_aggregate_rows(runs: list[EvalRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        row: dict[str, object] = {
            "label": run.label,
            "series_name": run.series_name,
            "run_name": run.run_name,
            "policy_type": run.policy_type,
            "dataset_repo_id": run.dataset_repo_id,
            "dataset_root": run.dataset_root,
            "eval_split": run.eval_split,
            "split_source": run.split_source,
            "num_episodes": run.num_episodes,
            "num_steps": run.num_steps,
            "action_dim": run.action_dim,
            "summary_path": str(run.summary_path),
        }
        for metric in METRIC_SPECS:
            row[metric.key] = run.metrics.get(metric.key)
        rows.append(row)
    return rows


def build_episode_rows(runs: list[EvalRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for result in run.results:
            row: dict[str, object] = {
                "label": run.label,
                "series_name": run.series_name,
                "run_name": run.run_name,
                "policy_type": run.policy_type,
                "episode_index": result.get("episode_index"),
                "steps": result.get("steps"),
            }
            for metric in METRIC_SPECS:
                row[metric.key] = result.get(metric.key)
            rows.append(row)
    return rows


def build_per_dim_rows(runs: list[EvalRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for metric_key in ("per_dim_mae", "per_dim_rmse"):
            values = run.metrics.get(metric_key)
            if not isinstance(values, list):
                continue
            for dim_index, value in enumerate(values):
                rows.append(
                    {
                        "label": run.label,
                        "series_name": run.series_name,
                        "run_name": run.run_name,
                        "policy_type": run.policy_type,
                        "metric": metric_key,
                        "action_dim_index": dim_index,
                        "value": value,
                    }
                )
    return rows


def build_pairwise_rows(runs: list[EvalRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    per_run_results = {
        run.label: {
            int(result["episode_index"]): result
            for result in run.results
            if result.get("episode_index") is not None
        }
        for run in runs
    }

    tolerance = 1e-12
    for metric in METRIC_SPECS:
        for lhs_index, lhs in enumerate(runs):
            lhs_map = per_run_results[lhs.label]
            for rhs in runs[lhs_index + 1 :]:
                rhs_map = per_run_results[rhs.label]
                shared = sorted(set(lhs_map).intersection(rhs_map))
                lhs_better = 0
                rhs_better = 0
                ties = 0
                deltas: list[float] = []
                for episode_index in shared:
                    lhs_value = lhs_map[episode_index].get(metric.key)
                    rhs_value = rhs_map[episode_index].get(metric.key)
                    if lhs_value is None or rhs_value is None:
                        continue
                    lhs_value = float(lhs_value)
                    rhs_value = float(rhs_value)
                    deltas.append(lhs_value - rhs_value)
                    if abs(lhs_value - rhs_value) <= tolerance:
                        ties += 1
                    elif metric.lower_is_better:
                        if lhs_value < rhs_value:
                            lhs_better += 1
                        else:
                            rhs_better += 1
                    else:
                        if lhs_value > rhs_value:
                            lhs_better += 1
                        else:
                            rhs_better += 1
                rows.append(
                    {
                        "metric": metric.key,
                        "lhs_label": lhs.label,
                        "rhs_label": rhs.label,
                        "shared_episodes": len(shared),
                        "lhs_better_count": lhs_better,
                        "rhs_better_count": rhs_better,
                        "tie_count": ties,
                        "lhs_minus_rhs_mean": (
                            None if not deltas else float(np.mean(np.asarray(deltas, dtype=np.float64)))
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sort_runs_for_display(runs: list[EvalRun]) -> list[EvalRun]:
    return sorted(runs, key=lambda run: run.label)


def plot_overall_metrics(
    plt,
    runs: list[EvalRun],
    output_path: Path,
    *,
    width: float,
    height: float,
    dpi: int,
    close_figure: bool,
) -> None:
    ordered_runs = sort_runs_for_display(runs)
    labels = [run.label for run in ordered_runs]
    fig, axes = plt.subplots(2, 2, figsize=(width, height))
    axes_flat = axes.reshape(-1)
    for ax, metric in zip(axes_flat, METRIC_SPECS, strict=True):
        values = [float(run.metrics[metric.key]) for run in ordered_runs]
        bars = ax.bar(labels, values)
        best_index = int(np.argmin(values) if metric.lower_is_better else np.argmax(values))
        bars[best_index].set_alpha(1.0)
        bars[best_index].set_edgecolor("black")
        bars[best_index].set_linewidth(1.0)
        ax.set_title(metric.title)
        ax.set_ylabel(metric.title)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle(f"Overall Offline Eval Metrics ({ordered_runs[0].dataset_repo_id})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close_figure:
        plt.close(fig)


def plot_episode_metric(
    plt,
    runs: list[EvalRun],
    metric: MetricSpec,
    output_path: Path,
    *,
    width: float,
    height: float,
    dpi: int,
    close_figure: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height))
    for run in sort_runs_for_display(runs):
        episode_results = sorted(
            (
                (int(result["episode_index"]), float(result[metric.key]))
                for result in run.results
                if result.get("episode_index") is not None and result.get(metric.key) is not None
            ),
            key=lambda item: item[0],
        )
        if not episode_results:
            continue
        x_values = [episode_index for episode_index, _ in episode_results]
        y_values = [value for _, value in episode_results]
        ax.plot(x_values, y_values, marker="o", markersize=2.5, linewidth=1.4, label=run.label)
    ax.set_title(f"Per-Episode {metric.title}")
    ax.set_xlabel("Episode Index")
    ax.set_ylabel(metric.title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close_figure:
        plt.close(fig)


def plot_per_dim_metric(
    plt,
    runs: list[EvalRun],
    metric_key: str,
    metric_title: str,
    output_path: Path,
    *,
    width: float,
    height: float,
    dpi: int,
    close_figure: bool,
) -> None:
    ordered_runs = sort_runs_for_display(runs)
    arrays: list[np.ndarray] = []
    for run in ordered_runs:
        values = run.metrics.get(metric_key)
        if not isinstance(values, list):
            print(f"[WARN] Missing `{metric_key}` for {run.label}; skipping per-dim plot.")
            return
        arrays.append(np.asarray(values, dtype=np.float64))

    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1:
        print(f"[WARN] Inconsistent action dims for `{metric_key}`; skipping per-dim plot.")
        return

    num_dims = int(next(iter(lengths)))
    x_positions = np.arange(num_dims, dtype=np.float64)
    bar_width = 0.8 / max(1, len(ordered_runs))

    fig, ax = plt.subplots(figsize=(width, height))
    for run_index, (run, values) in enumerate(zip(ordered_runs, arrays, strict=True)):
        offsets = x_positions - 0.4 + bar_width * 0.5 + run_index * bar_width
        ax.bar(offsets, values, width=bar_width, label=run.label)
    ax.set_title(metric_title)
    ax.set_xlabel("Action Dimension")
    ax.set_ylabel(metric_title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(index) for index in range(num_dims)])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close_figure:
        plt.close(fig)


def render_report(
    runs: list[EvalRun],
    output_dir: Path,
    *,
    skip_plot: bool,
    show_plot: bool,
    dpi: int,
    width: float,
    height: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows = build_aggregate_rows(runs)
    episode_rows = build_episode_rows(runs)
    per_dim_rows = build_per_dim_rows(runs)
    pairwise_rows = build_pairwise_rows(runs)

    aggregate_csv = output_dir / "aggregate_metrics.csv"
    episode_csv = output_dir / "episode_metrics.csv"
    per_dim_csv = output_dir / "per_dim_metrics.csv"
    pairwise_csv = output_dir / "pairwise_wins.csv"
    write_csv(aggregate_csv, aggregate_rows)
    write_csv(episode_csv, episode_rows)
    write_csv(per_dim_csv, per_dim_rows)
    write_csv(pairwise_csv, pairwise_rows)

    generated_plots: dict[str, str] = {}
    if not skip_plot:
        plt = require_matplotlib(interactive=show_plot)
        close_figure = not show_plot
        overall_path = output_dir / "overall_metrics.png"
        plot_overall_metrics(
            plt,
            runs,
            overall_path,
            width=max(width, 10.0),
            height=max(height, 7.5),
            dpi=dpi,
            close_figure=close_figure,
        )
        generated_plots["overall_metrics"] = str(overall_path)

        for metric in METRIC_SPECS:
            plot_path = output_dir / f"per_episode_{metric.key}.png"
            plot_episode_metric(
                plt,
                runs,
                metric,
                plot_path,
                width=width,
                height=height,
                dpi=dpi,
                close_figure=close_figure,
            )
            generated_plots[f"per_episode_{metric.key}"] = str(plot_path)

        per_dim_specs = (
            ("per_dim_mae", "Per-Dimension MAE"),
            ("per_dim_rmse", "Per-Dimension RMSE"),
        )
        for metric_key, metric_title in per_dim_specs:
            plot_path = output_dir / f"{metric_key}.png"
            plot_per_dim_metric(
                plt,
                runs,
                metric_key,
                metric_title,
                plot_path,
                width=width,
                height=height,
                dpi=dpi,
                close_figure=close_figure,
            )
            if plot_path.exists():
                generated_plots[metric_key] = str(plot_path)

        if show_plot and plt.get_fignums():
            plt.show()

    report_payload: dict[str, object] = {
        "dataset_repo_id": runs[0].dataset_repo_id,
        "dataset_root": runs[0].dataset_root,
        "eval_split": runs[0].eval_split,
        "num_runs": len(runs),
        "runs": [
            {
                "label": run.label,
                "series_name": run.series_name,
                "run_name": run.run_name,
                "policy_type": run.policy_type,
                "summary_path": str(run.summary_path),
                "policy_dir": run.policy_dir,
            }
            for run in runs
        ],
        "artifacts": {
            "report_json": "",
            "aggregate_metrics_csv": str(aggregate_csv),
            "episode_metrics_csv": str(episode_csv),
            "per_dim_metrics_csv": str(per_dim_csv),
            "pairwise_wins_csv": str(pairwise_csv),
            "plots": generated_plots,
        },
    }
    report_json = output_dir / "report.json"
    report_payload["artifacts"]["report_json"] = str(report_json)
    save_json(report_json, report_payload)
    return report_payload


def print_selected_runs(runs: list[EvalRun]) -> None:
    print("[info] selected runs:")
    for run in sort_runs_for_display(runs):
        print(
            f"  - {run.label}: policy={run.policy_type}, "
            f"episodes={run.num_episodes}, steps={run.num_steps}, "
            f"summary={run.summary_path}"
        )


def print_metric_table(runs: list[EvalRun]) -> None:
    ordered_runs = sort_runs_for_display(runs)
    headers = ["label", "mae", "rmse", "mean_l2_error", "cosine_similarity"]
    rows = []
    for run in ordered_runs:
        rows.append(
            [
                run.label,
                f"{float(run.metrics['mae']):.6f}",
                f"{float(run.metrics['rmse']):.6f}",
                f"{float(run.metrics['mean_l2_error']):.6f}",
                f"{float(run.metrics['cosine_similarity']):.6f}",
            ]
        )
    column_widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    header_line = "  ".join(
        header.ljust(column_widths[index]) for index, header in enumerate(headers)
    )
    print("[info] aggregate metrics:")
    print(f"  {header_line}")
    for row in rows:
        print(
            "  "
            + "  ".join(
                value.ljust(column_widths[index]) for index, value in enumerate(row)
            )
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    eval_root = args.eval_root.expanduser().resolve()
    if args.summary is not None:
        summary_paths = [Path(path).expanduser().resolve() for path in args.summary]
    else:
        summary_paths = discover_latest_summaries(args.dataset, eval_root=eval_root)

    runs = [load_summary_file(path, eval_root=eval_root) for path in summary_paths]
    validate_runs(runs)
    assign_labels(runs, args.label)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(runs, args.output_root.expanduser().resolve())
    )

    print_selected_runs(runs)
    print_metric_table(runs)
    report_payload = render_report(
        runs,
        output_dir,
        skip_plot=bool(args.skip_plot),
        show_plot=bool(args.show_plot),
        dpi=int(args.dpi),
        width=float(args.figure_width),
        height=float(args.figure_height),
    )

    print(f"[info] compare report saved to: {output_dir}")
    print("[info] generated artifacts:")
    for key, value in report_payload["artifacts"].items():
        if key == "plots":
            if not value:
                continue
            print("  - plots:")
            for plot_key, plot_path in sorted(value.items()):
                print(f"    - {plot_key}: {plot_path}")
        else:
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main(sys.argv[1:])
