from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.bridge.protocol import SensorPacket  # noqa: E402
from deploy.bridge.sync import LatestSensorCache, StreamRequirement  # noqa: E402


class LatestSensorCacheTest(unittest.TestCase):
    def test_snapshot_returns_samples_when_fresh(self) -> None:
        cache = LatestSensorCache(max_skew_ms=10)
        cache.update(
            SensorPacket(
                stream="left",
                seq=1,
                stamp_ns=1_000_000_000,
                payload_type="joint_positions",
                array=np.asarray([1.0, 2.0], dtype=np.float32),
            ),
            received_mono_ns=5_000_000_000,
        )
        cache.update(
            SensorPacket(
                stream="right",
                seq=2,
                stamp_ns=1_005_000_000,
                payload_type="joint_positions",
                array=np.asarray([3.0, 4.0], dtype=np.float32),
            ),
            received_mono_ns=5_001_000_000,
        )
        snapshot, failure = cache.snapshot(
            [
                StreamRequirement("left", max_age_ms=50),
                StreamRequirement("right", max_age_ms=50),
            ],
            now_mono_ns=5_010_000_000,
        )
        self.assertIsNone(failure)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(set(snapshot.samples), {"left", "right"})

    def test_snapshot_reports_stale_stream(self) -> None:
        cache = LatestSensorCache(max_skew_ms=10)
        cache.update(
            SensorPacket(
                stream="left",
                seq=1,
                stamp_ns=1_000_000_000,
                payload_type="joint_positions",
                array=np.asarray([1.0, 2.0], dtype=np.float32),
            ),
            received_mono_ns=5_000_000_000,
        )
        snapshot, failure = cache.snapshot(
            [StreamRequirement("left", max_age_ms=5)],
            now_mono_ns=5_020_000_000,
        )
        self.assertIsNone(snapshot)
        self.assertEqual(failure, "stale:left")


if __name__ == "__main__":
    unittest.main()

