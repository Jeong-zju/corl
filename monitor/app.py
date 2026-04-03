from __future__ import annotations

import json
import mimetypes
from numbers import Real
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".gif"}
DEFAULT_EVAL_ROOT = BASE_DIR / ".." / "outputs" / "eval"
FALLBACK_EVAL_ROOT = BASE_DIR / "main" / "outputs" / "eval"


app = FastAPI(title="Local Eval Comparison Viewer")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def resolve_eval_root() -> Path:
    for candidate in (DEFAULT_EVAL_ROOT, FALLBACK_EVAL_ROOT):
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_EVAL_ROOT.resolve()


def safe_relative_label(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def to_percent(value: Any) -> float | None:
    if not isinstance(value, Real):
        return None
    numeric = float(value)
    if 0.0 <= numeric <= 1.0:
        return numeric * 100.0
    return numeric


def empty_run_metrics_summary() -> dict[str, Any]:
    return {
        "has_eval_metrics": False,
        "success_rate_pct": None,
        "success_rate_label": "N/A",
        "success_count": 0,
        "total_episodes": 0,
        "metrics_source": "",
        "metrics_error": "",
    }


def load_eval_info_summary(timestamp_dir: Path) -> dict[str, Any]:
    eval_info_path = timestamp_dir / "eval_info.json"
    summary = empty_run_metrics_summary()

    if not eval_info_path.is_file():
        return summary

    try:
        data = json.loads(eval_info_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        summary["metrics_error"] = f"读取 eval_info.json 失败: {exc}"
        return summary

    success_values: list[int] = []
    for item in data.get("per_task", []):
        metrics = item.get("metrics", {})
        successes = metrics.get("successes", [])
        if isinstance(successes, list):
            success_values.extend(1 if bool(value) else 0 for value in successes)

    if success_values:
        total_episodes = len(success_values)
        success_count = sum(success_values)
        success_rate_pct = (success_count / total_episodes) * 100
    else:
        overall = data.get("overall", {})
        pc_success = overall.get("pc_success")
        n_episodes = overall.get("n_episodes")
        pc_success_pct = to_percent(pc_success)
        if (
            pc_success_pct is not None
            and isinstance(n_episodes, Real)
            and n_episodes > 0
        ):
            total_episodes = int(n_episodes)
            success_rate_pct = pc_success_pct
            success_count = int(round((success_rate_pct / 100) * total_episodes))
        else:
            return summary

    summary.update(
        {
            "has_eval_metrics": True,
            "success_rate_pct": success_rate_pct,
            "success_rate_label": format_percent(success_rate_pct),
            "success_count": success_count,
            "total_episodes": total_episodes,
            "metrics_source": "eval_info.json",
        }
    )
    return summary


def load_summary_json_summary(timestamp_dir: Path) -> dict[str, Any]:
    summary_path = timestamp_dir / "summary.json"
    summary = empty_run_metrics_summary()

    if not summary_path.is_file():
        return summary

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        summary["metrics_error"] = f"读取 summary.json 失败: {exc}"
        return summary

    success_values: list[int] = []
    results = data.get("results", [])
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and "success" in item:
                success_values.append(1 if bool(item["success"]) else 0)

    if success_values:
        total_episodes = len(success_values)
        success_count = sum(success_values)
        success_rate_pct = (success_count / total_episodes) * 100
    else:
        success_count_raw = data.get("success_count")
        num_rollouts_raw = data.get("num_rollouts")
        success_rate_raw = data.get("success_rate")

        if (
            isinstance(success_count_raw, Real)
            and isinstance(num_rollouts_raw, Real)
            and num_rollouts_raw > 0
        ):
            total_episodes = int(num_rollouts_raw)
            success_count = int(success_count_raw)
            success_rate_pct = (success_count / total_episodes) * 100
        else:
            success_rate_pct = to_percent(success_rate_raw)
            if success_rate_pct is None:
                return summary

            total_episodes = (
                int(num_rollouts_raw)
                if isinstance(num_rollouts_raw, Real) and num_rollouts_raw > 0
                else 0
            )
            success_count = (
                int(round((success_rate_pct / 100) * total_episodes))
                if total_episodes
                else 0
            )

    summary.update(
        {
            "has_eval_metrics": True,
            "success_rate_pct": success_rate_pct,
            "success_rate_label": format_percent(success_rate_pct),
            "success_count": success_count,
            "total_episodes": total_episodes,
            "metrics_source": "summary.json",
        }
    )
    return summary


def load_run_metrics_summary(timestamp_dir: Path) -> dict[str, Any]:
    eval_info_summary = load_eval_info_summary(timestamp_dir)
    if eval_info_summary["has_eval_metrics"]:
        return eval_info_summary
    summary_json_summary = load_summary_json_summary(timestamp_dir)
    if summary_json_summary["has_eval_metrics"]:
        return summary_json_summary
    return (
        eval_info_summary
        if eval_info_summary["metrics_error"]
        else summary_json_summary
    )


def build_algorithm_success_summary(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    aggregates: dict[str, dict[str, Any]] = {}
    missing_count = 0

    for run in runs:
        if not run["has_eval_metrics"]:
            missing_count += 1
            continue

        algorithm = run["algorithm"]
        bucket = aggregates.setdefault(
            algorithm,
            {
                "algorithm": algorithm,
                "success_count": 0,
                "total_episodes": 0,
                "run_count": 0,
            },
        )
        bucket["success_count"] += run["success_count"]
        bucket["total_episodes"] += run["total_episodes"]
        bucket["run_count"] += 1

    bars: list[dict[str, Any]] = []
    for bucket in aggregates.values():
        total_episodes = bucket["total_episodes"]
        success_rate_pct = (
            (bucket["success_count"] / total_episodes) * 100 if total_episodes else 0.0
        )
        bars.append(
            {
                "algorithm": bucket["algorithm"],
                "success_rate_pct": success_rate_pct,
                "success_rate_label": format_percent(success_rate_pct),
                "bar_width": max(0.0, min(success_rate_pct, 100.0)),
                "run_count": bucket["run_count"],
                "total_episodes": total_episodes,
            }
        )

    bars.sort(key=lambda item: (-item["success_rate_pct"], item["algorithm"].lower()))
    return bars, missing_count


def scan_runs() -> dict[str, Any]:
    eval_root = resolve_eval_root()
    runs: list[dict[str, Any]] = []
    error_message = ""

    if not eval_root.exists():
        return {
            "eval_root": eval_root,
            "eval_root_label": safe_relative_label(eval_root),
            "runs": [],
            "error_message": "",
        }

    try:
        for env_dir in sorted(
            (item for item in eval_root.iterdir() if item.is_dir()),
            key=lambda item: item.name.lower(),
        ):
            for algorithm_dir in sorted(
                (item for item in env_dir.iterdir() if item.is_dir()),
                key=lambda item: item.name.lower(),
            ):
                for timestamp_dir in sorted(
                    (item for item in algorithm_dir.iterdir() if item.is_dir()),
                    key=lambda item: item.name,
                    reverse=True,
                ):
                    run_metrics_summary = load_run_metrics_summary(timestamp_dir)
                    videos: list[dict[str, Any]] = []
                    for video_path in sorted(
                        (
                            item
                            for item in timestamp_dir.rglob("*")
                            if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
                        ),
                        key=lambda item: item.relative_to(timestamp_dir)
                        .as_posix()
                        .lower(),
                    ):
                        relative_path = video_path.relative_to(eval_root).as_posix()
                        run_relative_path = video_path.relative_to(
                            timestamp_dir
                        ).as_posix()
                        videos.append(
                            {
                                "name": run_relative_path,
                                "relative_path": relative_path,
                                "is_gif": video_path.suffix.lower() == ".gif",
                            }
                        )

                    runs.append(
                        {
                            "env": env_dir.name,
                            "algorithm": algorithm_dir.name,
                            "timestamp": timestamp_dir.name,
                            "videos": videos,
                            "video_count": len(videos),
                            **run_metrics_summary,
                        }
                    )
    except (
        Exception
    ) as exc:  # pragma: no cover - defensive fallback for unexpected filesystem issues
        error_message = f"扫描失败: {exc}"

    return {
        "eval_root": eval_root,
        "eval_root_label": safe_relative_label(eval_root),
        "runs": runs,
        "error_message": error_message,
    }


def refresh_scan_cache() -> dict[str, Any]:
    cache = scan_runs()
    app.state.scan_cache = cache
    return cache


def get_scan_cache() -> dict[str, Any]:
    cache = getattr(app.state, "scan_cache", None)
    if cache is None:
        cache = refresh_scan_cache()
    return cache


def build_filter_options(
    runs: list[dict[str, Any]], selected_env: str
) -> tuple[list[str], list[str], list[str]]:
    env_options = sorted({run["env"] for run in runs}, key=lambda value: value.lower())

    scoped_runs = [
        run for run in runs if not selected_env or run["env"] == selected_env
    ]
    algorithm_options = sorted(
        {run["algorithm"] for run in scoped_runs}, key=lambda value: value.lower()
    )
    timestamp_options = sorted({run["timestamp"] for run in scoped_runs}, reverse=True)
    return env_options, algorithm_options, timestamp_options


def filter_runs(
    runs: list[dict[str, Any]],
    selected_env: str,
    selected_algorithms: list[str],
    selected_timestamps: list[str],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    algorithm_set = set(selected_algorithms)
    timestamp_set = set(selected_timestamps)

    for run in runs:
        if selected_env and run["env"] != selected_env:
            continue
        if algorithm_set and run["algorithm"] not in algorithm_set:
            continue
        if timestamp_set and run["timestamp"] not in timestamp_set:
            continue
        filtered.append(run)

    filtered.sort(
        key=lambda run: (
            run["env"].lower(),
            run["algorithm"].lower(),
            run["timestamp"],
        )
    )
    return filtered


@app.on_event("startup")
def startup_refresh() -> None:
    refresh_scan_cache()


@app.get("/")
async def index(request: Request) -> Any:
    cache = get_scan_cache()
    runs = cache["runs"]

    selected_env = request.query_params.get("env", "").strip()
    selected_algorithms = [
        value for value in request.query_params.getlist("algorithm") if value
    ]
    selected_timestamps = [
        value for value in request.query_params.getlist("timestamp") if value
    ]

    env_options, algorithm_options, timestamp_options = build_filter_options(
        runs, selected_env
    )
    filtered_runs = filter_runs(
        runs, selected_env, selected_algorithms, selected_timestamps
    )
    algorithm_success_bars, success_metrics_missing_count = (
        build_algorithm_success_summary(filtered_runs)
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "scan_root_label": cache["eval_root_label"],
            "scan_root_path": str(cache["eval_root"]),
            "error_message": cache["error_message"],
            "has_eval_root": cache["eval_root"].exists(),
            "env_options": env_options,
            "algorithm_options": algorithm_options,
            "timestamp_options": timestamp_options,
            "selected_env": selected_env,
            "selected_algorithms": set(selected_algorithms),
            "selected_timestamps": set(selected_timestamps),
            "runs": filtered_runs,
            "algorithm_success_bars": algorithm_success_bars,
            "success_metrics_missing_count": success_metrics_missing_count,
            "total_runs": len(runs),
            "filtered_runs_count": len(filtered_runs),
        },
    )


@app.post("/rescan")
async def rescan() -> RedirectResponse:
    refresh_scan_cache()
    return RedirectResponse(url="/", status_code=303)


@app.get("/video/{file_path:path}", name="video_file")
async def video_file(file_path: str) -> FileResponse:
    cache = get_scan_cache()
    eval_root = cache["eval_root"]

    if not eval_root.exists():
        raise HTTPException(status_code=404, detail="Eval directory does not exist.")

    target_path = (eval_root / file_path).resolve()
    if eval_root != target_path and eval_root not in target_path.parents:
        raise HTTPException(status_code=403, detail="Invalid file path.")
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found.")
    if target_path.suffix.lower() not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=404, detail="Unsupported file type.")

    media_type, _ = mimetypes.guess_type(target_path.name)
    return FileResponse(
        target_path, media_type=media_type or "application/octet-stream"
    )
