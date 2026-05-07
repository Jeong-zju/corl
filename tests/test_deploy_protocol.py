from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main"))
sys.path.insert(0, str(REPO_ROOT / "main" / "deploy"))

from deploy.bridge.protocol import build_command_packet


def test_build_command_packet_applies_per_axis_base_deadzones() -> None:
    config = SimpleNamespace(
        policy=SimpleNamespace(base_action_dim=3, arm_dof=7),
        command=SimpleNamespace(
            publish_base=True,
            publish_arms=True,
            max_linear_x=1.0,
            max_linear_y=1.0,
            max_angular_z=1.0,
            deadzone_linear_x=0.1,
            deadzone_linear_y=0.2,
            deadzone_angular_z=0.05,
        ),
    )

    action = np.zeros(17, dtype=np.float32)
    action[:3] = np.asarray([0.05, 0.25, -0.04], dtype=np.float32)

    packet = build_command_packet(
        config=config,
        seq=7,
        obs_seq=6,
        action=action,
        status="ok",
        message="policy_eval",
        runtime_ms=12.3,
    )

    assert np.allclose(packet["base_twist"], np.asarray([0.0, 0.25, 0.0], dtype=np.float32))


def test_build_command_packet_applies_batch_mutual_exclusion_rules() -> None:
    config = SimpleNamespace(
        policy=SimpleNamespace(base_action_dim=3, arm_dof=7),
        command=SimpleNamespace(
            publish_base=True,
            publish_arms=True,
            max_linear_x=1.0,
            max_linear_y=1.0,
            max_angular_z=1.0,
            deadzone_linear_x=0.0,
            deadzone_linear_y=0.0,
            deadzone_angular_z=0.0,
            mutual_exclusion=SimpleNamespace(
                enabled=True,
                rules=[
                    SimpleNamespace(
                        source_index=0,
                        threshold=0.4,
                        target_indices=[4, 10],
                        mask_value=0.0,
                    ),
                    SimpleNamespace(
                        source_index=2,
                        threshold=0.2,
                        target_indices=[1],
                        mask_value=-1.0,
                    ),
                ],
            ),
        ),
    )

    action = np.arange(17, dtype=np.float32) / 10.0
    action[:3] = np.asarray([0.5, 0.25, 0.3], dtype=np.float32)

    packet = build_command_packet(
        config=config,
        seq=7,
        obs_seq=6,
        action=action,
        status="ok",
        message="policy_eval",
        runtime_ms=12.3,
    )

    assert np.allclose(
        packet["base_twist"],
        np.asarray([0.5, -1.0, 0.3], dtype=np.float32),
    )
    assert packet["left_joint_positions"][1] == 0.0
    assert packet["right_joint_positions"][0] == 0.0
