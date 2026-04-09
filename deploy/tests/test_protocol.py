from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.bridge.protocol import (  # noqa: E402
    ActionPacket,
    ObservationPacket,
    RobotCommandPacket,
    SensorPacket,
    decode_packet,
)


class ProtocolRoundTripTest(unittest.TestCase):
    def test_sensor_packet_round_trip(self) -> None:
        packet = SensorPacket(
            stream="odom",
            seq=3,
            stamp_ns=123,
            payload_type="base_velocity",
            array=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            metadata={"topic": "/odom"},
        )
        decoded = decode_packet(packet.to_multipart())
        self.assertIsInstance(decoded, SensorPacket)
        assert isinstance(decoded, SensorPacket)
        np.testing.assert_allclose(decoded.array, packet.array)
        self.assertEqual(decoded.stream, "odom")

    def test_observation_packet_round_trip(self) -> None:
        packet = ObservationPacket(
            seq=1,
            episode_id="episode-1",
            stamp_ns=456,
            reset=True,
            policy_type="streaming_act",
            mode="auto",
            state=np.arange(17, dtype=np.float32),
            images={
                "observation.images.realsense_top": np.zeros((4, 5, 3), dtype=np.uint8),
                "observation.images.realsense_left": np.ones((4, 5, 3), dtype=np.uint8),
            },
            path_signature=np.arange(6, dtype=np.float32),
            delta_signature=np.ones((6,), dtype=np.float32),
        )
        decoded = decode_packet(packet.to_multipart())
        self.assertIsInstance(decoded, ObservationPacket)
        assert isinstance(decoded, ObservationPacket)
        np.testing.assert_allclose(decoded.state, packet.state)
        np.testing.assert_allclose(
            decoded.images["observation.images.realsense_left"],
            packet.images["observation.images.realsense_left"],
        )
        np.testing.assert_allclose(decoded.path_signature, packet.path_signature)

    def test_robot_command_round_trip(self) -> None:
        packet = RobotCommandPacket(
            seq=7,
            obs_seq=5,
            stamp_ns=789,
            mode="auto",
            status="ok",
            raw_action=np.arange(17, dtype=np.float32),
            left_arm=np.arange(7, dtype=np.float32),
            right_arm=np.arange(7, dtype=np.float32) + 10,
            base=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        )
        decoded = decode_packet(packet.to_multipart())
        self.assertIsInstance(decoded, RobotCommandPacket)
        assert isinstance(decoded, RobotCommandPacket)
        np.testing.assert_allclose(decoded.raw_action, packet.raw_action)

    def test_action_packet_round_trip(self) -> None:
        packet = ActionPacket(
            seq=1,
            obs_seq=2,
            stamp_ns=3,
            runtime_ms=4.5,
            action=np.arange(17, dtype=np.float32),
            status="ok",
            message="",
        )
        decoded = decode_packet(packet.to_multipart())
        self.assertIsInstance(decoded, ActionPacket)
        assert isinstance(decoded, ActionPacket)
        np.testing.assert_allclose(decoded.action, packet.action)


if __name__ == "__main__":
    unittest.main()

