from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.bridge.bridge_core import BridgeConfig, BridgeRuntime  # noqa: E402
from deploy.bridge.protocol import SensorPacket  # noqa: E402


def _make_config(policy_type: str) -> BridgeConfig:
    return BridgeConfig(
        sensor_bind="tcp://*:5556",
        command_bind="tcp://*:5557",
        control_bind="tcp://*:5559",
        policy_endpoint="tcp://127.0.0.1:5555",
        policy_control_endpoint="tcp://127.0.0.1:5558",
        policy_type=policy_type,
        control_rate_hz=30.0,
        policy_request_timeout_ms=80,
        signature_backend="simple",
        signature_depth=3,
        state_streams={
            "base": "odom",
            "left_arm": "left_arm_joint_state",
            "right_arm": "right_arm_joint_state",
        },
        image_streams={
            "observation.images.realsense_top": "realsense_top",
        },
        freshness_ms={},
        max_skew_ms=150,
        left_arm_dim=7,
        right_arm_dim=7,
        enable_base_action=True,
        initial_episode_id="episode-0000",
        initial_mode="auto",
    )


def _make_samples() -> dict[str, SensorPacket]:
    return {
        "odom": SensorPacket(
            stream="odom",
            seq=1,
            stamp_ns=123,
            payload_type="base_velocity",
            array=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        ),
        "left_arm_joint_state": SensorPacket(
            stream="left_arm_joint_state",
            seq=2,
            stamp_ns=123,
            payload_type="joint_positions",
            array=np.arange(7, dtype=np.float32),
        ),
        "right_arm_joint_state": SensorPacket(
            stream="right_arm_joint_state",
            seq=3,
            stamp_ns=123,
            payload_type="joint_positions",
            array=np.arange(7, dtype=np.float32) + 10.0,
        ),
        "realsense_top": SensorPacket(
            stream="realsense_top",
            seq=4,
            stamp_ns=123,
            payload_type="image_rgb8",
            array=np.zeros((4, 5, 3), dtype=np.uint8),
        ),
    }


class BridgeRuntimeObservationTest(unittest.TestCase):
    def test_act_observation_omits_signatures(self) -> None:
        runtime = BridgeRuntime(config=_make_config("act"))
        observation = runtime._build_observation_packet(
            stamp_ns=123,
            samples=_make_samples(),
        )
        self.assertEqual(observation.policy_type, "act")
        self.assertIsNone(observation.path_signature)
        self.assertIsNone(observation.delta_signature)

    def test_streaming_act_observation_includes_signatures(self) -> None:
        runtime = BridgeRuntime(config=_make_config("streaming_act"))
        observation = runtime._build_observation_packet(
            stamp_ns=123,
            samples=_make_samples(),
        )
        self.assertEqual(observation.policy_type, "streaming_act")
        self.assertIsNotNone(observation.path_signature)
        self.assertIsNotNone(observation.delta_signature)
        assert observation.path_signature is not None
        assert observation.delta_signature is not None
        self.assertEqual(observation.path_signature.shape[0], 51)
        self.assertEqual(observation.delta_signature.shape[0], 51)


if __name__ == "__main__":
    unittest.main()
