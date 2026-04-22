from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deploy.gripper_hysteresis import (
    GripperHysteresis,
    GripperHysteresisConfig,
    complete_gripper_hysteresis_config_from_dataset_stats,
    parse_gripper_hysteresis_config,
)


def test_gripper_hysteresis_holds_inside_deadband() -> None:
    filt = GripperHysteresis(
        GripperHysteresisConfig(
            enabled=True,
            action_indices=(1,),
            closed_values=(0.0,),
            open_values=(1.0,),
            close_thresholds=(0.4,),
            open_thresholds=(0.6,),
            initial_state="prediction",
        )
    )

    outputs = [
        filt.apply(np.asarray([0.0, 0.2], dtype=np.float32))[1],
        filt.apply(np.asarray([0.0, 0.55], dtype=np.float32))[1],
        filt.apply(np.asarray([0.0, 0.61], dtype=np.float32))[1],
        filt.apply(np.asarray([0.0, 0.45], dtype=np.float32))[1],
        filt.apply(np.asarray([0.0, 0.39], dtype=np.float32))[1],
    ]

    assert outputs == [0.0, 0.0, 1.0, 1.0, 0.0]


def test_gripper_hysteresis_can_initialize_from_current_state() -> None:
    filt = GripperHysteresis(
        GripperHysteresisConfig(
            enabled=True,
            action_indices=(1,),
            closed_values=(0.0,),
            open_values=(1.0,),
            close_thresholds=(0.4,),
            open_thresholds=(0.6,),
            initial_state="current",
        )
    )

    out = filt.apply(
        np.asarray([0.0, 0.5], dtype=np.float32),
        current_state=np.asarray([0.0, 0.9], dtype=np.float32),
    )

    assert out[1] == 1.0


def test_gripper_hysteresis_derives_values_from_dataset_stats(tmp_path: Path) -> None:
    stats_dir = tmp_path / "meta"
    stats_dir.mkdir(parents=True)
    (stats_dir / "stats.json").write_text(
        json.dumps(
            {
                "action": {
                    "min": [0.0, -0.02, 0.0, -0.01],
                    "max": [0.0, 0.08, 0.0, 0.07],
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = parse_gripper_hysteresis_config(
        {"enabled": True, "action_indices": [1, 3], "hysteresis_ratio": 0.2}
    )

    completed = complete_gripper_hysteresis_config_from_dataset_stats(
        cfg,
        dataset_root=tmp_path,
    )

    assert completed.closed_values == (-0.02, -0.01)
    assert completed.open_values == (0.08, 0.07)
    assert np.allclose(completed.close_thresholds, [0.02, 0.022])
    assert np.allclose(completed.open_thresholds, [0.04, 0.038])
