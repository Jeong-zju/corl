from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main"))
sys.path.insert(0, str(REPO_ROOT / "main" / "deploy"))

from deploy.bridge.sync import reorder_joint_positions


def test_reorder_joint_positions_uses_joint_names_when_available() -> None:
    positions = [30.0, 10.0, 20.0]
    names = ["joint3", "joint1", "joint2"]
    expected_names = ["joint1", "joint2", "joint3"]

    reordered = reorder_joint_positions(
        positions=positions,
        names=names,
        expected_names=expected_names,
        dof=3,
    )

    assert np.allclose(reordered, np.asarray([10.0, 20.0, 30.0], dtype=np.float32))


def test_reorder_joint_positions_falls_back_to_raw_order_when_names_do_not_match() -> None:
    reordered = reorder_joint_positions(
        positions=[1.0, 2.0],
        names=["unexpected_a", "unexpected_b"],
        expected_names=["joint1", "joint2", "joint3"],
        dof=3,
    )

    assert np.allclose(reordered, np.asarray([1.0, 2.0, 0.0], dtype=np.float32))
