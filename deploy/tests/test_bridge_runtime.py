from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.bridge.bridge_core import BridgeConfig, BridgeRuntime  # noqa: E402
from deploy.bridge.protocol import SensorPacket  # noqa: E402
from deploy.bridge.sync import StreamRequirement  # noqa: E402


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

    def test_snapshot_fingerprint_changes_only_when_seq_or_stamp_changes(self) -> None:
        runtime = BridgeRuntime(config=_make_config("act"))
        samples = _make_samples()
        fingerprint = runtime._snapshot_fingerprint(samples)
        repeated_fingerprint = runtime._snapshot_fingerprint(_make_samples())
        self.assertEqual(fingerprint, repeated_fingerprint)

        advanced_samples = _make_samples()
        advanced_samples["odom"] = SensorPacket(
            stream="odom",
            seq=99,
            stamp_ns=456,
            payload_type="base_velocity",
            array=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        )
        advanced_fingerprint = runtime._snapshot_fingerprint(advanced_samples)
        self.assertNotEqual(fingerprint, advanced_fingerprint)

    def test_time_rewind_triggers_automatic_reset_and_cache_clear(self) -> None:
        runtime = BridgeRuntime(config=_make_config("streaming_act"))
        runtime._pending_reset = False
        runtime._last_snapshot_fingerprint = (("odom", 1, 123),)
        runtime._sensor_cache.update(
            SensorPacket(
                stream="odom",
                seq=5,
                stamp_ns=500,
                payload_type="base_velocity",
                array=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        runtime._last_sensor_stamp_by_stream["odom"] = 500
        runtime._last_sensor_seq_by_stream["odom"] = 5

        runtime._handle_sensor_packet(
            SensorPacket(
                stream="odom",
                seq=1,
                stamp_ns=100,
                payload_type="base_velocity",
                array=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            )
        )

        self.assertTrue(runtime._pending_reset)
        self.assertIsNone(runtime._last_snapshot_fingerprint)
        self.assertEqual(runtime._auto_reset_count, 1)
        snapshot, failure = runtime._sensor_cache.snapshot([StreamRequirement("odom", max_age_ms=50)])
        self.assertIsNone(failure)
        assert snapshot is not None
        self.assertEqual(int(snapshot.samples["odom"].stamp_ns), 100)

    def test_policy_health_summary(self) -> None:
        from deploy.bridge.protocol import ControlPacket

        summary = BridgeRuntime._summarize_policy_health(
            ControlPacket(
                command="health_check_ack",
                params={
                    "ok": True,
                    "policy_type": "act",
                    "policy_dir": "/tmp/policy",
                    "device": "cuda:0",
                    "paused": False,
                },
            )
        )

        self.assertIn("command=health_check_ack", summary)
        self.assertIn("ok=True", summary)
        self.assertIn("policy_type=act", summary)
        self.assertIn("policy_dir=/tmp/policy", summary)
        self.assertIn("device=cuda:0", summary)
        self.assertIn("paused=False", summary)


if __name__ == "__main__":
    unittest.main()
