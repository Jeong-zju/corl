from __future__ import annotations

import asyncio
import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlparse
from pathlib import Path

from starlette.datastructures import QueryParams


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("monitor_app", APP_PATH)
MONITOR_APP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MONITOR_APP)


class BuildRunVideoGroupsTests(unittest.TestCase):
    def test_eval_info_groups_by_task_and_sorts_episode_numerically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_root = Path(tmp_dir) / "outputs" / "eval"
            timestamp_dir = eval_root / "metaworld_mt50" / "act" / "20260403_140651"
            task_dir = timestamp_dir / "videos" / "assembly-v3_0"
            task_dir.mkdir(parents=True)

            video_paths = [
                task_dir / "eval_episode_10.mp4",
                task_dir / "eval_episode_2.mp4",
                task_dir / "eval_episode_1.mp4",
            ]
            for video_path in video_paths:
                video_path.touch()

            (timestamp_dir / "eval_info.json").write_text(
                json.dumps(
                    {
                        "per_task": [
                            {
                                "task_group": "assembly-v3",
                                "task_id": 0,
                                "metrics": {
                                    "video_paths": [str(path) for path in video_paths],
                                    "successes": [False, True, False],
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            grouped = MONITOR_APP.build_run_video_groups(
                timestamp_dir=timestamp_dir, eval_root=eval_root
            )

            self.assertEqual(grouped["task_count"], 1)
            task_group = grouped["task_groups"][0]
            self.assertEqual(task_group["task_label"], "assembly-v3")
            self.assertEqual(task_group["video_count"], 3)
            self.assertEqual(task_group["success_count"], 1)
            self.assertEqual(task_group["failure_count"], 2)
            self.assertEqual(
                [video["episode_index"] for video in task_group["videos"]],
                [1, 2, 10],
            )
            self.assertEqual(
                [video["status_key"] for video in task_group["videos"]],
                ["failure", "success", "failure"],
            )

    def test_summary_results_provide_task_code_and_failure_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_root = Path(tmp_dir) / "outputs" / "eval"
            timestamp_dir = eval_root / "braidedhub" / "act-baseline" / "20260402_205445"
            timestamp_dir.mkdir(parents=True)

            rollout_a = timestamp_dir / "rollout_010_task_01.mp4"
            rollout_b = timestamp_dir / "rollout_002_task_00.mp4"
            rollout_a.touch()
            rollout_b.touch()

            (timestamp_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "episode_index": 10,
                                "task_code": "01",
                                "video_path": str(rollout_a),
                                "failure_reason": "wrong_branch",
                                "branch_mismatch_stage": "H1",
                                "success": False,
                            },
                            {
                                "episode_index": 2,
                                "task_code": "00",
                                "video_path": str(rollout_b),
                                "failure_reason": None,
                                "success": True,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            grouped = MONITOR_APP.build_run_video_groups(
                timestamp_dir=timestamp_dir, eval_root=eval_root
            )

            self.assertEqual(
                [task_group["task_label"] for task_group in grouped["task_groups"]],
                ["task 00", "task 01"],
            )
            success_video = grouped["task_groups"][0]["videos"][0]
            self.assertEqual(success_video["status_key"], "success")
            self.assertEqual(success_video["failure_reason"], "")

            failure_video = grouped["task_groups"][1]["videos"][0]
            self.assertEqual(failure_video["episode_index"], 10)
            self.assertEqual(failure_video["status_key"], "failure")
            self.assertEqual(failure_video["failure_reason"], "H1 阶段走错分支")


class RescanRouteTests(unittest.TestCase):
    def test_rescan_redirect_preserves_filter_query_params(self) -> None:
        class FakeRequest:
            def __init__(self, query_params: QueryParams) -> None:
                self.query_params = query_params

        request = FakeRequest(
            QueryParams(
                [
                    ("env", "metaworld_mt50"),
                    ("algorithm", "act"),
                    ("algorithm", "streaming-act"),
                    ("timestamp", "20260403_144650"),
                    ("timestamp", "20260403_140651"),
                ]
            )
        )

        with mock.patch.object(MONITOR_APP, "refresh_scan_cache", return_value={}):
            response = asyncio.run(MONITOR_APP.rescan(request))

        self.assertEqual(response.status_code, 303)
        location = response.headers["location"]
        parsed = urlparse(location)
        self.assertEqual(parsed.path, "/")
        self.assertEqual(
            parse_qs(parsed.query),
            {
                "env": ["metaworld_mt50"],
                "algorithm": ["act", "streaming-act"],
                "timestamp": ["20260403_144650", "20260403_140651"],
            },
        )


if __name__ == "__main__":
    unittest.main()
