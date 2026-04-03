from __future__ import annotations

import json
import mimetypes
import re
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".gif"}
DEFAULT_EVAL_ROOT = BASE_DIR / ".." / "outputs" / "eval"
FALLBACK_EVAL_ROOT = BASE_DIR / "main" / "outputs" / "eval"
VIDEOS_PER_PAGE = 6
EPISODE_PATTERNS = (
    re.compile(r"(?:eval_)?episode[_-](\d+)", re.IGNORECASE),
    re.compile(r"rollout[_-](\d+)", re.IGNORECASE),
    re.compile(r"ep(?:isode)?[_-](\d+)", re.IGNORECASE),
)
TRAILING_NUMBER_PATTERN = re.compile(r"(\d+)$")
TASK_CODE_PATTERN = re.compile(r"task[_-]?([0-9A-Za-z]+)", re.IGNORECASE)
TASK_SUFFIX_PATTERN = re.compile(r"^(?P<label>.+?)_(?P<index>\d+)$")


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


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def normalize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "n/a"}:
        return ""
    return text


def build_failure_reason_display(reason: Any, stage: Any = None) -> str:
    normalized_reason = normalize_optional_text(reason)
    normalized_stage = normalize_optional_text(stage)

    if not normalized_reason:
        return ""

    if normalized_reason == "wrong_branch":
        if normalized_stage:
            return f"{normalized_stage} 阶段走错分支"
        return "走错分支"

    return normalized_reason.replace("_", " ").replace("-", " ")


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


def resolve_video_reference(
    video_reference: str, timestamp_dir: Path, eval_root: Path
) -> tuple[str, str] | None:
    if not isinstance(video_reference, str) or not video_reference.strip():
        return None

    candidate = Path(video_reference)
    candidate_paths = (
        (candidate,)
        if candidate.is_absolute()
        else (timestamp_dir / candidate, eval_root / candidate)
    )

    for candidate_path in candidate_paths:
        try:
            resolved_path = candidate_path.resolve()
            relative_path = resolved_path.relative_to(eval_root).as_posix()
            relative_run_path = resolved_path.relative_to(timestamp_dir).as_posix()
            return relative_path, relative_run_path
        except ValueError:
            continue

    return None


def extract_episode_index(path_or_name: str) -> int | None:
    stem = Path(PurePosixPath(path_or_name).name).stem

    for pattern in EPISODE_PATTERNS:
        match = pattern.search(stem)
        if match:
            return int(match.group(1))

    trailing_match = TRAILING_NUMBER_PATTERN.search(stem)
    if trailing_match:
        return int(trailing_match.group(1))

    return None


def build_episode_label(episode_index: int | None, fallback_name: str) -> str:
    if episode_index is not None:
        return f"Episode {episode_index}"
    return Path(PurePosixPath(fallback_name).name).stem


def derive_task_identity(
    *,
    task_group: Any = None,
    task_code: Any = None,
    task_id: Any = None,
    run_relative_path: str = "",
) -> dict[str, Any]:
    explicit_group = str(task_group).strip() if isinstance(task_group, str) else ""
    explicit_code = str(task_code).strip() if isinstance(task_code, str) else ""
    path_parent = ""
    path_label = ""
    path_index: int | None = None

    path_parent_name = PurePosixPath(run_relative_path).parent.name
    if path_parent_name and path_parent_name != ".":
        path_parent = path_parent_name
        suffix_match = TASK_SUFFIX_PATTERN.match(path_parent_name)
        if suffix_match:
            path_label = suffix_match.group("label")
            path_index = int(suffix_match.group("index"))
        else:
            path_label = path_parent_name

    numeric_task_id = int(task_id) if isinstance(task_id, Real) else None

    if explicit_group:
        return {
            "task_key": f"group:{explicit_group}",
            "task_label": explicit_group,
            "task_sort_key": (0, explicit_group.lower(), numeric_task_id or 0, ""),
        }

    if explicit_code:
        numeric_code = int(explicit_code) if explicit_code.isdigit() else None
        return {
            "task_key": f"code:{explicit_code}",
            "task_label": f"task {explicit_code}",
            "task_sort_key": (
                1,
                "",
                numeric_code if numeric_code is not None else 10**9,
                explicit_code.lower(),
            ),
        }

    if path_parent:
        return {
            "task_key": f"path:{path_label}",
            "task_label": path_label,
            "task_sort_key": (2, path_label.lower(), path_index or 0, ""),
        }

    if numeric_task_id is not None:
        label = f"task {numeric_task_id}"
        return {
            "task_key": f"id:{numeric_task_id}",
            "task_label": label,
            "task_sort_key": (3, "", numeric_task_id, ""),
        }

    filename = PurePosixPath(run_relative_path).name
    task_match = TASK_CODE_PATTERN.search(filename)
    if task_match:
        task_code_from_name = task_match.group(1)
        numeric_code = (
            int(task_code_from_name) if task_code_from_name.isdigit() else None
        )
        return {
            "task_key": f"code:{task_code_from_name}",
            "task_label": f"task {task_code_from_name}",
            "task_sort_key": (
                4,
                "",
                numeric_code if numeric_code is not None else 10**9,
                task_code_from_name.lower(),
            ),
        }

    return {
        "task_key": "ungrouped",
        "task_label": "未分组",
        "task_sort_key": (5, "ungrouped", 0, ""),
    }


def merge_video_detail(
    target: dict[str, dict[str, Any]],
    relative_path: str,
    detail: dict[str, Any],
) -> None:
    bucket = target.setdefault(relative_path, {})
    for key, value in detail.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if key not in bucket or bucket[key] in {None, ""}:
            bucket[key] = value


def load_eval_info_video_details(
    timestamp_dir: Path, eval_root: Path
) -> dict[str, dict[str, Any]]:
    eval_info_path = timestamp_dir / "eval_info.json"
    if not eval_info_path.is_file():
        return {}

    try:
        data = json.loads(eval_info_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    details: dict[str, dict[str, Any]] = {}
    for task_item in data.get("per_task", []):
        if not isinstance(task_item, dict):
            continue

        metrics = task_item.get("metrics", {})
        if not isinstance(metrics, dict):
            continue

        video_paths = metrics.get("video_paths", [])
        successes = metrics.get("successes", [])
        if not isinstance(video_paths, list):
            continue

        for index, video_reference in enumerate(video_paths):
            resolved_paths = resolve_video_reference(
                video_reference, timestamp_dir, eval_root
            )
            if resolved_paths is None:
                continue

            relative_path, relative_run_path = resolved_paths
            success = (
                coerce_bool(successes[index])
                if isinstance(successes, list) and index < len(successes)
                else None
            )
            merge_video_detail(
                details,
                relative_path,
                {
                    **derive_task_identity(
                        task_group=task_item.get("task_group"),
                        task_id=task_item.get("task_id"),
                        run_relative_path=relative_run_path,
                    ),
                    "episode_index": extract_episode_index(relative_run_path),
                    "success": success,
                },
            )

    return details


def load_summary_video_details(
    timestamp_dir: Path, eval_root: Path
) -> dict[str, dict[str, Any]]:
    summary_path = timestamp_dir / "summary.json"
    if not summary_path.is_file():
        return {}

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    details: dict[str, dict[str, Any]] = {}
    results = data.get("results", [])
    if not isinstance(results, list):
        return details

    for item in results:
        if not isinstance(item, dict):
            continue

        resolved_paths = resolve_video_reference(
            item.get("video_path", ""), timestamp_dir, eval_root
        )
        if resolved_paths is None:
            continue

        relative_path, relative_run_path = resolved_paths
        merge_video_detail(
            details,
            relative_path,
            {
                **derive_task_identity(
                    task_group=item.get("task_group"),
                    task_code=item.get("task_code"),
                    task_id=item.get("task_id"),
                    run_relative_path=relative_run_path,
                ),
                "episode_index": (
                    int(item["episode_index"])
                    if isinstance(item.get("episode_index"), Real)
                    else extract_episode_index(relative_run_path)
                ),
                "success": coerce_bool(item.get("success")),
                "failure_reason": build_failure_reason_display(
                    item.get("failure_reason"),
                    stage=item.get("branch_mismatch_stage"),
                ),
            },
        )

    return details


def load_run_video_details(
    timestamp_dir: Path, eval_root: Path
) -> dict[str, dict[str, Any]]:
    details = load_eval_info_video_details(timestamp_dir, eval_root)
    for relative_path, detail in load_summary_video_details(
        timestamp_dir, eval_root
    ).items():
        merge_video_detail(details, relative_path, detail)
    return details


def build_status_metadata(success: bool | None) -> tuple[str, str, str]:
    if success is True:
        return "success", "成功", "status-success"
    if success is False:
        return "failure", "失败", "status-failure"
    return "unknown", "未知", "status-unknown"


def build_run_video_groups(timestamp_dir: Path, eval_root: Path) -> dict[str, Any]:
    detail_by_relative_path = load_run_video_details(timestamp_dir, eval_root)
    video_records: list[dict[str, Any]] = []

    for video_path in sorted(
        (
            item
            for item in timestamp_dir.rglob("*")
            if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda item: item.relative_to(timestamp_dir).as_posix().lower(),
    ):
        relative_path = video_path.relative_to(eval_root).as_posix()
        relative_run_path = video_path.relative_to(timestamp_dir).as_posix()
        detail = detail_by_relative_path.get(relative_path, {})
        task_identity = (
            {
                "task_key": detail.get("task_key"),
                "task_label": detail.get("task_label"),
                "task_sort_key": detail.get("task_sort_key"),
            }
            if detail.get("task_key") and detail.get("task_label")
            else derive_task_identity(run_relative_path=relative_run_path)
        )
        episode_index = detail.get("episode_index")
        if not isinstance(episode_index, int):
            episode_index = extract_episode_index(relative_run_path)

        success = detail.get("success")
        if success not in {True, False}:
            success = None
        status_key, status_label, status_class = build_status_metadata(success)
        failure_reason = normalize_optional_text(detail.get("failure_reason", ""))
        if status_key != "failure":
            failure_reason = ""

        video_records.append(
            {
                "name": relative_run_path,
                "relative_path": relative_path,
                "is_gif": video_path.suffix.lower() == ".gif",
                "task_key": task_identity["task_key"],
                "task_label": task_identity["task_label"],
                "task_sort_key": task_identity["task_sort_key"],
                "episode_index": episode_index,
                "episode_label": build_episode_label(
                    episode_index, fallback_name=relative_run_path
                ),
                "success": success,
                "status_key": status_key,
                "status_label": status_label,
                "status_class": status_class,
                "failure_reason": failure_reason,
            }
        )

    video_records.sort(
        key=lambda item: (
            item["task_sort_key"],
            item["episode_index"] if item["episode_index"] is not None else 10**9,
            item["name"].lower(),
        )
    )

    task_buckets: dict[str, dict[str, Any]] = {}
    for video in video_records:
        bucket = task_buckets.setdefault(
            video["task_key"],
            {
                "task_key": video["task_key"],
                "task_label": video["task_label"],
                "task_sort_key": video["task_sort_key"],
                "videos": [],
                "video_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "unknown_count": 0,
            },
        )
        bucket["videos"].append(video)
        bucket["video_count"] += 1
        if video["status_key"] == "success":
            bucket["success_count"] += 1
        elif video["status_key"] == "failure":
            bucket["failure_count"] += 1
        else:
            bucket["unknown_count"] += 1

    task_groups = sorted(
        task_buckets.values(), key=lambda item: (item["task_sort_key"], item["task_label"])
    )
    for task_group in task_groups:
        task_group.pop("task_sort_key", None)

    return {
        "videos": video_records,
        "task_groups": task_groups,
        "task_count": len(task_groups),
    }


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
                    video_data = build_run_video_groups(timestamp_dir, eval_root)

                    runs.append(
                        {
                            "env": env_dir.name,
                            "algorithm": algorithm_dir.name,
                            "timestamp": timestamp_dir.name,
                            "videos": video_data["videos"],
                            "video_count": len(video_data["videos"]),
                            "task_groups": video_data["task_groups"],
                            "task_count": video_data["task_count"],
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
            "videos_per_page": VIDEOS_PER_PAGE,
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
