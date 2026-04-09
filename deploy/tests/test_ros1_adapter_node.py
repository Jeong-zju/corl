from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.bridge.protocol import RobotCommandPacket  # noqa: E402
from deploy.ros1_adapter.ros1_adapter_node import _summarize_robot_command  # noqa: E402


class Ros1AdapterCommandSummaryTest(unittest.TestCase):
    def test_summary_includes_policy_metadata(self) -> None:
        packet = RobotCommandPacket(
            seq=3,
            obs_seq=8,
            stamp_ns=123,
            mode="auto",
            status="ok",
            raw_action=np.zeros((17,), dtype=np.float32),
            left_arm=np.zeros((7,), dtype=np.float32),
            right_arm=np.zeros((7,), dtype=np.float32),
            base=np.zeros((3,), dtype=np.float32),
            metadata={
                "policy_seq": 5,
                "policy_status": "ok",
                "policy_runtime_ms": 12.345,
                "policy_message": "ready",
            },
        )

        summary = _summarize_robot_command(packet)

        self.assertIn("mode=auto", summary)
        self.assertIn("status=ok", summary)
        self.assertIn("obs_seq=8", summary)
        self.assertIn("hold_reason=-", summary)
        self.assertIn("policy_seq=5", summary)
        self.assertIn("policy_status=ok", summary)
        self.assertIn("policy_runtime_ms=12.3", summary)
        self.assertIn("policy_message=ready", summary)

    def test_summary_uses_fallback_message_key(self) -> None:
        packet = RobotCommandPacket.hold(
            seq=4,
            obs_seq=9,
            stamp_ns=123,
            raw_action_dim=17,
            left_arm_dim=7,
            right_arm_dim=7,
            hold_reason="policy_error:timeout",
            metadata={
                "policy_runtime_ms": 80.0,
                "policy_status": "error",
                "message": "timed out",
            },
        )

        summary = _summarize_robot_command(packet)

        self.assertIn("mode=hold", summary)
        self.assertIn("policy_status=error", summary)
        self.assertIn("policy_runtime_ms=80.0", summary)
        self.assertIn("policy_message=timed out", summary)


if __name__ == "__main__":
    unittest.main()
