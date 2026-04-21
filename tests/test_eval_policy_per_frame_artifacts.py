from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from eval_policy import (
    build_per_episode_metric_series,
    write_per_frame_metrics_json,
)


def test_build_per_episode_metric_series_preserves_episode_boundaries() -> None:
    series = build_per_episode_metric_series(
        [
            {"episode_index": 3, "frame_mae": [0.1, 0.2]},
            {"episode_index": 7, "frame_mae": [0.3]},
            {"episode_index": 8, "frame_mae": []},
        ],
        metric_key="frame_mae",
    )

    assert len(series) == 2
    assert series[0]["episode_index"] == 3
    assert np.allclose(series[0]["step_index"], np.asarray([0, 1], dtype=np.int32))
    assert np.allclose(series[0]["values"], np.asarray([0.1, 0.2], dtype=np.float32))
    assert series[1]["episode_index"] == 7
    assert np.allclose(series[1]["step_index"], np.asarray([0], dtype=np.int32))
    assert np.allclose(series[1]["values"], np.asarray([0.3], dtype=np.float32))


def test_write_per_frame_metrics_json_persists_episode_payload(
    tmp_path: Path,
) -> None:
    metrics_path = write_per_frame_metrics_json(
        tmp_path,
        [
            {
                "episode_index": 11,
                "steps": 2,
                "frame_mae": [0.01, 0.02],
                "frame_rmse": [0.03, 0.04],
                "frame_l2_error": [0.5, 0.6],
            }
        ],
    )

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_path == tmp_path / "per_frame_metrics.json"
    assert payload == {
        "episodes": [
            {
                "episode_index": 11,
                "steps": 2,
                "frame_mae": [0.01, 0.02],
                "frame_rmse": [0.03, 0.04],
                "frame_l2_error": [0.5, 0.6],
            }
        ]
    }
