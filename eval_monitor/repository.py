from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from statistics import fmean
from typing import Any
from urllib.parse import quote


RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
VIDEO_NAME_RE = re.compile(r"rollout_(?P<episode>\d+)_task_(?P<task>.+)\.mp4$")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_run_datetime(run_name: str, fallback_ts_ns: int) -> datetime:
    try:
        naive = datetime.strptime(run_name, RUN_TIMESTAMP_FORMAT)
        return naive.astimezone()
    except ValueError:
        return datetime.fromtimestamp(fallback_ts_ns / 1_000_000_000, tz=timezone.utc).astimezone()


def datetime_to_payload(dt: datetime) -> tuple[str, int]:
    return dt.isoformat(timespec="seconds"), int(dt.timestamp())


def sort_timestamp_key(item: dict[str, Any]) -> tuple[int, str]:
    return (item.get("timestamp_epoch") or 0, item.get("id") or "")


def ensure_relative_to(base: Path, candidate: Path) -> None:
    base_resolved = base.resolve()
    candidate_resolved = candidate.resolve()
    if candidate_resolved != base_resolved and base_resolved not in candidate_resolved.parents:
        raise ValueError(f"Path escapes root: {candidate_resolved}")


def collect_failure_counts(payload: dict[str, Any]) -> dict[str, int]:
    collected: dict[str, int] = {}
    for key, value in payload.items():
        if key.endswith("_failures"):
            parsed = safe_int(value)
            if parsed is not None:
                collected[key] = parsed
    return collected


def task_sort_key(task: dict[str, Any]) -> tuple[int, int, str]:
    for key in ("task_id", "task_code", "goal_name", "task_key"):
        raw = normalize_text(task.get(key))
        if not raw:
            continue
        match = re.search(r"(\d+)", raw)
        if match:
            return (0, int(match.group(1)), raw)
        return (1, 0, raw)
    return (2, 0, "")


@dataclass(slots=True)
class RunScan:
    series_name: str
    run_name: str
    run_dir: Path
    status: str
    summary: dict[str, Any] | None
    summary_error: str | None
    video_files: list[Path]
    last_modified_ns: int
    fingerprint: str


class EvalSnapshotBuilder:
    def __init__(self, eval_root: Path) -> None:
        self.eval_root = eval_root.resolve()

    def build(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
        generated_at = iso_now()
        series_scans: list[tuple[Path, list[RunScan], str | None, str | None]] = []
        policy_hints: set[str] = set()
        fingerprints: list[str] = []

        if not self.eval_root.exists():
            snapshot = {
                "version": 0,
                "generated_at": generated_at,
                "eval_root": str(self.eval_root),
                "stats": {
                    "series_count": 0,
                    "run_count": 0,
                    "complete_runs": 0,
                    "incomplete_runs": 0,
                    "invalid_runs": 0,
                    "environment_count": 0,
                    "algorithm_count": 0,
                    "total_rollouts": 0,
                    "video_count": 0,
                },
                "filters": {"environments": [], "algorithms": [], "statuses": []},
                "matrix": {"algorithms": [], "rows": []},
                "series": [],
                "recent_runs": [],
            }
            return snapshot, {}, "missing-root"

        for series_dir in sorted(self.eval_root.iterdir(), key=lambda path: path.name):
            if not series_dir.is_dir():
                continue
            run_scans: list[RunScan] = []
            policy_hint: str | None = None
            env_family_hint: str | None = None
            for run_dir in sorted(series_dir.iterdir(), key=lambda path: path.name):
                if not run_dir.is_dir():
                    continue
                scan = self._scan_run_dir(series_dir.name, run_dir)
                run_scans.append(scan)
                fingerprints.append(scan.fingerprint)
                if scan.summary:
                    hint = normalize_text(scan.summary.get("policy_type"))
                    if hint:
                        policy_hint = hint
                        policy_hints.add(hint)
                    env_hint = normalize_text(scan.summary.get("env"))
                    if env_hint:
                        env_family_hint = env_hint
            series_scans.append((series_dir, run_scans, policy_hint, env_family_hint))

        provisional_algorithms: set[str] = set()
        for series_dir, _, policy_hint, _ in series_scans:
            _, provisional_algorithm = self._infer_series_parts(series_dir.name, policy_hint, [])
            if provisional_algorithm != "unknown":
                provisional_algorithms.add(provisional_algorithm)

        known_algorithms = sorted(provisional_algorithms | policy_hints, key=lambda value: (-len(value), value))
        algorithms: set[str] = set()
        environments: dict[str, dict[str, Any]] = {}
        details: dict[str, dict[str, Any]] = {}
        series_payloads: list[dict[str, Any]] = []
        all_run_summaries: list[dict[str, Any]] = []

        for series_dir, run_scans, policy_hint, env_family_hint in series_scans:
            environment, algorithm = self._infer_series_parts(series_dir.name, policy_hint, known_algorithms)
            environment_family = env_family_hint or environment
            algorithms.add(algorithm)
            environments.setdefault(
                environment,
                {
                    "id": environment,
                    "label": environment,
                    "family": environment_family,
                },
            )

            run_details: list[dict[str, Any]] = []
            for scan in run_scans:
                detail = self._build_run_detail(
                    scan=scan,
                    environment=environment,
                    environment_family=environment_family,
                    algorithm=algorithm,
                )
                run_details.append(detail)
                details[detail["run"]["id"]] = detail
                all_run_summaries.append(detail["run"])

            series_payload = self._build_series_payload(
                series_dir=series_dir,
                environment=environment,
                environment_family=environment_family,
                algorithm=algorithm,
                run_details=run_details,
            )
            series_payloads.append(series_payload)

        recent_runs = sorted(
            all_run_summaries,
            key=lambda item: (item.get("last_modified_epoch") or 0, item.get("id") or ""),
            reverse=True,
        )[:32]
        series_payloads.sort(key=lambda item: (item["environment"], item["algorithm"], item["id"]))
        matrix = self._build_matrix(series_payloads, sorted(algorithms))
        statuses = sorted({run["status"] for run in all_run_summaries})

        total_rollouts = sum(run.get("num_rollouts") or 0 for run in all_run_summaries if run["status"] == "complete")
        total_videos = sum(run.get("video_count") or 0 for run in all_run_summaries)

        snapshot = {
            "version": 0,
            "generated_at": generated_at,
            "eval_root": str(self.eval_root),
            "stats": {
                "series_count": len(series_payloads),
                "run_count": len(all_run_summaries),
                "complete_runs": sum(1 for run in all_run_summaries if run["status"] == "complete"),
                "incomplete_runs": sum(1 for run in all_run_summaries if run["status"] == "incomplete"),
                "invalid_runs": sum(1 for run in all_run_summaries if run["status"] == "invalid"),
                "environment_count": len(environments),
                "algorithm_count": len(algorithms),
                "total_rollouts": total_rollouts,
                "video_count": total_videos,
            },
            "filters": {
                "environments": sorted(environments.values(), key=lambda item: (item["family"], item["label"])),
                "algorithms": [{"id": algorithm, "label": algorithm} for algorithm in sorted(algorithms)],
                "statuses": [{"id": status, "label": status} for status in statuses],
            },
            "matrix": matrix,
            "series": series_payloads,
            "recent_runs": recent_runs,
        }
        fingerprint = "\n".join(sorted(fingerprints))
        return snapshot, details, fingerprint

    def _scan_run_dir(self, series_name: str, run_dir: Path) -> RunScan:
        children: list[Path] = []
        try:
            children = sorted(run_dir.iterdir(), key=lambda path: path.name)
        except FileNotFoundError:
            pass

        summary: dict[str, Any] | None = None
        summary_error: str | None = None
        video_files: list[Path] = []
        tokens = [series_name, run_dir.name]
        last_modified_ns = run_dir.stat().st_mtime_ns if run_dir.exists() else 0

        for child in children:
            try:
                stat = child.stat()
            except FileNotFoundError:
                continue
            tokens.append(f"{child.name}:{stat.st_size}:{stat.st_mtime_ns}")
            last_modified_ns = max(last_modified_ns, stat.st_mtime_ns)
            if child.suffix.lower() == ".mp4":
                video_files.append(child)
            elif child.name == "summary.json":
                try:
                    summary = json.loads(child.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    summary_error = f"{type(exc).__name__}: {exc}"

        if summary is not None and summary_error is None:
            status = "complete"
        elif summary_error:
            status = "invalid"
        else:
            status = "incomplete"

        fingerprint = "|".join(tokens)
        return RunScan(
            series_name=series_name,
            run_name=run_dir.name,
            run_dir=run_dir,
            status=status,
            summary=summary,
            summary_error=summary_error,
            video_files=video_files,
            last_modified_ns=last_modified_ns,
            fingerprint=fingerprint,
        )

    def _infer_series_parts(
        self,
        series_name: str,
        policy_hint: str | None,
        known_algorithms: list[str],
    ) -> tuple[str, str]:
        if policy_hint:
            marker = f"_{policy_hint}"
            index = series_name.rfind(marker)
            if index > 0:
                environment = series_name[:index]
                algorithm = series_name[index + 1 :]
                if environment and algorithm:
                    return environment, algorithm

        for algorithm in known_algorithms:
            suffix = f"_{algorithm}"
            if series_name.endswith(suffix):
                environment = series_name[: -len(suffix)]
                if environment:
                    return environment, algorithm

        if policy_hint and series_name.endswith(f"_{policy_hint}"):
            return series_name[: -(len(policy_hint) + 1)], policy_hint

        return series_name, policy_hint or "unknown"

    def _build_run_detail(
        self,
        scan: RunScan,
        environment: str,
        environment_family: str,
        algorithm: str,
    ) -> dict[str, Any]:
        dt = parse_run_datetime(scan.run_name, scan.last_modified_ns)
        timestamp_iso, timestamp_epoch = datetime_to_payload(dt)
        last_modified_dt = datetime.fromtimestamp(scan.last_modified_ns / 1_000_000_000, tz=timezone.utc).astimezone()
        last_modified_iso, last_modified_epoch = datetime_to_payload(last_modified_dt)
        run_id = f"{scan.series_name}/{scan.run_name}"
        results = self._build_results(scan)
        per_task = self._build_per_task(scan.summary or {}, results)
        failure_reason_counts = self._count_failure_reasons(results)
        top_level_failure_counts = collect_failure_counts(scan.summary or {})
        num_rollouts = safe_int((scan.summary or {}).get("num_rollouts"))
        success_count = safe_int((scan.summary or {}).get("success_count"))
        success_rate = safe_float((scan.summary or {}).get("success_rate"))

        if success_count is None and results:
            counted = sum(1 for result in results if result.get("success") is True)
            if counted or scan.status == "complete":
                success_count = counted
        if num_rollouts is None and scan.status == "complete":
            num_rollouts = len(results)
        if success_rate is None and success_count is not None and num_rollouts:
            success_rate = success_count / max(1, num_rollouts)

        steps_values = [result["steps"] for result in results if result.get("steps") is not None]
        reward_values = [result["sum_reward"] for result in results if result.get("sum_reward") is not None]
        collision_values = [
            result["collision_rejections"]
            for result in results
            if result.get("collision_rejections") is not None
        ]

        run_payload = {
            "id": run_id,
            "series_id": scan.series_name,
            "run_name": scan.run_name,
            "path": str(scan.run_dir),
            "status": scan.status,
            "environment": environment,
            "environment_family": environment_family,
            "algorithm": algorithm,
            "label": f"{environment} / {algorithm} / {scan.run_name}",
            "timestamp": timestamp_iso,
            "timestamp_epoch": timestamp_epoch,
            "timestamp_label": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "last_modified": last_modified_iso,
            "last_modified_epoch": last_modified_epoch,
            "num_rollouts": num_rollouts,
            "video_count": len(scan.video_files),
            "result_count": len(results),
            "task_count": len(per_task),
            "success_count": success_count,
            "success_rate": success_rate,
            "avg_steps": fmean(steps_values) if steps_values else None,
            "avg_reward": fmean(reward_values) if reward_values else None,
            "avg_collision_rejections": fmean(collision_values) if collision_values else None,
            "failure_counts": top_level_failure_counts,
            "failure_reason_counts": failure_reason_counts,
            "seed": safe_int((scan.summary or {}).get("seed")),
            "fps": safe_int((scan.summary or {}).get("fps")),
            "max_steps": safe_int((scan.summary or {}).get("max_steps")),
            "max_action_step": safe_float((scan.summary or {}).get("max_action_step")),
            "start_randomized": (scan.summary or {}).get("start_randomized"),
            "policy_dir": normalize_text((scan.summary or {}).get("policy_dir")),
            "has_summary": scan.summary is not None,
            "summary_error": scan.summary_error,
        }

        detail = {
            "run": run_payload,
            "per_task": per_task,
            "results": results,
        }
        return detail

    def _build_results(self, scan: RunScan) -> list[dict[str, Any]]:
        summary = scan.summary or {}
        raw_results = summary.get("results")
        if isinstance(raw_results, list):
            results: list[dict[str, Any]] = []
            for index, raw in enumerate(raw_results):
                if not isinstance(raw, dict):
                    continue
                result = self._normalize_result(scan, raw, index)
                if result:
                    results.append(result)
            return sorted(results, key=lambda item: (item.get("episode_index") or 0, item["video_name"]))

        partial_results = []
        for video_file in scan.video_files:
            episode_index, task_code = self._parse_video_name(video_file.name)
            relative_path = f"{scan.series_name}/{scan.run_name}/{video_file.name}"
            partial_results.append(
                {
                    "episode_index": episode_index,
                    "task_id": None,
                    "task_code": task_code,
                    "start_region_name": None,
                    "target_goal_name": None,
                    "final_phase_name": None,
                    "reached_goal": None,
                    "success": None,
                    "steps": None,
                    "sum_reward": None,
                    "collision_rejections": None,
                    "failure_reason": None,
                    "media_path": relative_path,
                    "media_url": f"/media/{quote(relative_path, safe='/')}",
                    "video_name": video_file.name,
                }
            )
        return sorted(partial_results, key=lambda item: (item.get("episode_index") or 0, item["video_name"]))

    def _normalize_result(
        self,
        scan: RunScan,
        raw: dict[str, Any],
        fallback_index: int,
    ) -> dict[str, Any] | None:
        video_path_text = normalize_text(raw.get("video_path"))
        video_name = None
        if video_path_text:
            video_name = Path(video_path_text).name
        if not video_name:
            video_name = f"rollout_{fallback_index:03d}.mp4"
        video_file = scan.run_dir / video_name
        relative_path = f"{scan.series_name}/{scan.run_name}/{video_name}"

        return {
            "episode_index": safe_int(raw.get("episode_index")),
            "task_id": normalize_text(raw.get("task_id")),
            "task_code": normalize_text(raw.get("task_code")),
            "start_region_name": normalize_text(raw.get("start_region_name")),
            "target_goal_name": normalize_text(raw.get("target_goal_name")),
            "final_phase_name": normalize_text(raw.get("final_phase_name")),
            "reached_goal": normalize_text(raw.get("reached_goal")),
            "success": raw.get("success") if isinstance(raw.get("success"), bool) else None,
            "steps": safe_int(raw.get("steps")),
            "sum_reward": safe_float(raw.get("sum_reward")),
            "collision_rejections": safe_int(raw.get("collision_rejections")),
            "failure_reason": normalize_text(raw.get("failure_reason")),
            "media_path": relative_path,
            "media_url": f"/media/{quote(relative_path, safe='/')}",
            "video_name": video_name,
            "video_exists": video_file.exists(),
        }

    def _parse_video_name(self, video_name: str) -> tuple[int | None, str | None]:
        match = VIDEO_NAME_RE.match(video_name)
        if not match:
            return None, None
        return safe_int(match.group("episode")), normalize_text(match.group("task"))

    def _build_per_task(self, summary: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        per_task_lookup: dict[str, dict[str, Any]] = {}
        raw_per_task = summary.get("per_task")
        if isinstance(raw_per_task, dict):
            for task_key, payload in raw_per_task.items():
                if not isinstance(payload, dict):
                    continue
                normalized_key = normalize_text(task_key) or str(task_key)
                per_task_lookup[normalized_key] = {
                    "task_key": normalized_key,
                    "task_id": normalize_text(payload.get("task_id")) or normalized_key,
                    "task_code": normalize_text(payload.get("task_code")),
                    "goal_name": normalize_text(payload.get("goal_name")),
                    "rollouts": safe_int(payload.get("rollouts")),
                    "success_count": safe_int(payload.get("success_count")),
                    "success_rate": safe_float(payload.get("success_rate")),
                    "failure_counts": collect_failure_counts(payload),
                }

        grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            task_key = normalize_text(result.get("task_id"))
            if not task_key:
                task_key = normalize_text(result.get("task_code"))
            if not task_key:
                task_key = normalize_text(result.get("target_goal_name"))
            if not task_key:
                task_key = "unknown"
            grouped_results[task_key].append(result)

        for task_key, grouped in grouped_results.items():
            entry = per_task_lookup.setdefault(
                task_key,
                {
                    "task_key": task_key,
                    "task_id": normalize_text(grouped[0].get("task_id")) or task_key,
                    "task_code": normalize_text(grouped[0].get("task_code")),
                    "goal_name": normalize_text(grouped[0].get("target_goal_name")),
                    "rollouts": None,
                    "success_count": None,
                    "success_rate": None,
                    "failure_counts": {},
                },
            )

            if entry.get("task_code") is None:
                entry["task_code"] = normalize_text(grouped[0].get("task_code"))
            if entry.get("goal_name") is None:
                entry["goal_name"] = normalize_text(grouped[0].get("target_goal_name"))

            if entry.get("rollouts") is None:
                entry["rollouts"] = len(grouped)
            if entry.get("success_count") is None:
                entry["success_count"] = sum(1 for result in grouped if result.get("success") is True)
            if entry.get("success_rate") is None and entry.get("rollouts"):
                entry["success_rate"] = entry["success_count"] / max(1, entry["rollouts"])

            if not entry["failure_counts"]:
                failure_counts: dict[str, int] = defaultdict(int)
                for result in grouped:
                    reason = normalize_text(result.get("failure_reason"))
                    if reason:
                        failure_counts[f"{reason}_failures"] += 1
                entry["failure_counts"] = dict(failure_counts)

        return sorted(per_task_lookup.values(), key=task_sort_key)

    def _count_failure_reasons(self, results: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for result in results:
            reason = normalize_text(result.get("failure_reason"))
            if reason:
                counts[reason] += 1
        return dict(counts)

    def _build_series_payload(
        self,
        series_dir: Path,
        environment: str,
        environment_family: str,
        algorithm: str,
        run_details: list[dict[str, Any]],
    ) -> dict[str, Any]:
        run_summaries = [detail["run"] for detail in run_details]
        run_summaries.sort(key=sort_timestamp_key, reverse=True)
        complete_runs = [run for run in run_summaries if run["status"] == "complete" and run.get("success_rate") is not None]
        incomplete_runs = [run for run in run_summaries if run["status"] == "incomplete"]
        invalid_runs = [run for run in run_summaries if run["status"] == "invalid"]

        latest_complete = max(complete_runs, key=sort_timestamp_key) if complete_runs else None
        latest_run = max(run_summaries, key=sort_timestamp_key) if run_summaries else None
        best_run = max(complete_runs, key=lambda item: (item.get("success_rate") or 0.0, sort_timestamp_key(item))) if complete_runs else None
        weighted_rollouts = sum(run.get("num_rollouts") or 0 for run in complete_runs)
        weighted_success = sum(run.get("success_count") or 0 for run in complete_runs)
        weighted_success_rate = (
            weighted_success / max(1, weighted_rollouts)
            if weighted_rollouts
            else None
        )
        average_success_rate = (
            fmean(run["success_rate"] for run in complete_runs if run.get("success_rate") is not None)
            if complete_runs
            else None
        )

        failure_totals: dict[str, int] = defaultdict(int)
        for run in complete_runs:
            for key, value in run["failure_counts"].items():
                failure_totals[key] += value

        task_totals: dict[str, dict[str, Any]] = {}
        for detail in run_details:
            if detail["run"]["status"] != "complete":
                continue
            for task in detail["per_task"]:
                task_key = normalize_text(task.get("task_id")) or normalize_text(task.get("task_code")) or normalize_text(task.get("goal_name")) or task["task_key"]
                if task_key not in task_totals:
                    task_totals[task_key] = {
                        "task_key": task_key,
                        "task_id": normalize_text(task.get("task_id")) or task_key,
                        "task_code": normalize_text(task.get("task_code")),
                        "goal_name": normalize_text(task.get("goal_name")),
                        "rollouts": 0,
                        "success_count": 0,
                        "failure_counts": defaultdict(int),
                    }
                bucket = task_totals[task_key]
                bucket["rollouts"] += task.get("rollouts") or 0
                bucket["success_count"] += task.get("success_count") or 0
                if bucket.get("task_code") is None:
                    bucket["task_code"] = normalize_text(task.get("task_code"))
                if bucket.get("goal_name") is None:
                    bucket["goal_name"] = normalize_text(task.get("goal_name"))
                for key, value in task.get("failure_counts", {}).items():
                    bucket["failure_counts"][key] += value

        aggregated_tasks = []
        for task in task_totals.values():
            rollouts = task["rollouts"]
            success_rate = task["success_count"] / max(1, rollouts) if rollouts else None
            aggregated_tasks.append(
                {
                    "task_key": task["task_key"],
                    "task_id": task["task_id"],
                    "task_code": task["task_code"],
                    "goal_name": task["goal_name"],
                    "rollouts": rollouts,
                    "success_count": task["success_count"],
                    "success_rate": success_rate,
                    "failure_counts": dict(task["failure_counts"]),
                }
            )
        aggregated_tasks.sort(key=task_sort_key)

        last_updated = max((run["last_modified_epoch"] for run in run_summaries), default=0)
        last_updated_iso = (
            datetime.fromtimestamp(last_updated, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
            if last_updated
            else None
        )

        return {
            "id": series_dir.name,
            "path": str(series_dir),
            "environment": environment,
            "environment_family": environment_family,
            "algorithm": algorithm,
            "label": f"{environment} / {algorithm}",
            "run_count": len(run_summaries),
            "complete_runs": len(complete_runs),
            "incomplete_runs": len(incomplete_runs),
            "invalid_runs": len(invalid_runs),
            "latest_run_id": latest_run["id"] if latest_run else None,
            "latest_complete_run_id": latest_complete["id"] if latest_complete else None,
            "best_run_id": best_run["id"] if best_run else None,
            "latest_success_rate": latest_complete["success_rate"] if latest_complete else None,
            "best_success_rate": best_run["success_rate"] if best_run else None,
            "average_success_rate": average_success_rate,
            "weighted_success_rate": weighted_success_rate,
            "total_rollouts": weighted_rollouts,
            "success_count_total": weighted_success,
            "failure_totals": dict(failure_totals),
            "per_task": aggregated_tasks,
            "runs": run_summaries,
            "last_updated": last_updated_iso,
            "last_updated_epoch": last_updated,
        }

    def _build_matrix(self, series_payloads: list[dict[str, Any]], algorithms: list[str]) -> dict[str, Any]:
        rows: dict[str, dict[str, Any]] = {}
        for series in series_payloads:
            row = rows.setdefault(
                series["environment"],
                {
                    "environment": series["environment"],
                    "label": series["environment"],
                    "family": series["environment_family"],
                    "cells": {},
                },
            )
            row["cells"][series["algorithm"]] = {
                "series_id": series["id"],
                "complete_runs": series["complete_runs"],
                "incomplete_runs": series["incomplete_runs"],
                "invalid_runs": series["invalid_runs"],
                "latest_success_rate": series["latest_success_rate"],
                "best_success_rate": series["best_success_rate"],
                "average_success_rate": series["average_success_rate"],
                "weighted_success_rate": series["weighted_success_rate"],
                "total_rollouts": series["total_rollouts"],
                "last_updated": series["last_updated"],
                "last_updated_epoch": series["last_updated_epoch"],
            }
        ordered_rows = sorted(rows.values(), key=lambda item: (item["family"], item["label"]))
        return {"algorithms": algorithms, "rows": ordered_rows}


class EvalMonitorStore:
    def __init__(self, eval_root: Path, poll_interval_seconds: float = 2.0) -> None:
        self.eval_root = eval_root.resolve()
        self.poll_interval_seconds = poll_interval_seconds
        self._lock = threading.RLock()
        self._listeners: set[Queue[dict[str, Any]]] = set()
        self._snapshot: dict[str, Any] | None = None
        self._details: dict[str, dict[str, Any]] = {}
        self._fingerprint = ""
        self._version = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.refresh(force=True)
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._poll_loop, name="eval-monitor-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval_seconds):
            self.refresh(force=False)

    def refresh(self, force: bool) -> bool:
        builder = EvalSnapshotBuilder(self.eval_root)
        snapshot, details, fingerprint = builder.build()
        listeners: list[Queue[dict[str, Any]]] = []
        changed = False

        with self._lock:
            if force or fingerprint != self._fingerprint or self._snapshot is None:
                self._version += 1
                snapshot["version"] = self._version
                self._snapshot = snapshot
                self._details = details
                self._fingerprint = fingerprint
                listeners = list(self._listeners)
                changed = True

        if changed:
            event = {
                "version": snapshot["version"],
                "generated_at": snapshot["generated_at"],
                "stats": snapshot["stats"],
            }
            for listener in listeners:
                try:
                    listener.put_nowait(event)
                except Exception:
                    continue
        return changed

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot or {
                "version": 0,
                "generated_at": iso_now(),
                "eval_root": str(self.eval_root),
                "stats": {},
                "filters": {"environments": [], "algorithms": [], "statuses": []},
                "matrix": {"algorithms": [], "rows": []},
                "series": [],
                "recent_runs": [],
            }

    def get_run_detail(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._details.get(run_id)

    def subscribe(self) -> Queue[dict[str, Any]]:
        queue: Queue[dict[str, Any]] = Queue()
        with self._lock:
            self._listeners.add(queue)
        return queue

    def unsubscribe(self, queue: Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._listeners.discard(queue)

    def delete_run(self, run_id: str) -> bool:
        series_name, run_name = self._split_run_id(run_id)
        run_dir = (self.eval_root / series_name / run_name).resolve()
        ensure_relative_to(self.eval_root, run_dir)
        if not run_dir.exists():
            return False

        for child in sorted(run_dir.iterdir(), key=lambda path: path.name, reverse=True):
            if child.is_dir():
                self._remove_tree(child)
            else:
                child.unlink(missing_ok=True)
        run_dir.rmdir()

        series_dir = run_dir.parent
        if series_dir.exists() and not any(series_dir.iterdir()):
            series_dir.rmdir()
        self.refresh(force=True)
        return True

    def delete_series(self, series_id: str) -> bool:
        if not series_id or "/" in series_id:
            raise ValueError("Invalid series id")
        series_dir = (self.eval_root / series_id).resolve()
        ensure_relative_to(self.eval_root, series_dir)
        if not series_dir.exists():
            return False
        self._remove_tree(series_dir)
        self.refresh(force=True)
        return True

    def _remove_tree(self, path: Path) -> None:
        ensure_relative_to(self.eval_root, path)
        if path.is_dir():
            for child in sorted(path.iterdir(), key=lambda item: item.name, reverse=True):
                self._remove_tree(child)
            path.rmdir()
        else:
            path.unlink(missing_ok=True)

    def _split_run_id(self, run_id: str) -> tuple[str, str]:
        parts = [part for part in run_id.split("/") if part]
        if len(parts) != 2:
            raise ValueError("Run id must be '<series>/<run>'")
        return parts[0], parts[1]
