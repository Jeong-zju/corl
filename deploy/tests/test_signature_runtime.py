from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.bridge.signature_runtime import StreamingSignatureTracker  # noqa: E402


class StreamingSignatureTrackerTest(unittest.TestCase):
    def test_tracker_updates_and_resets(self) -> None:
        tracker = StreamingSignatureTracker(enabled=True, depth=3, backend="simple")
        first = tracker.update(np.asarray([0.0, 1.0], dtype=np.float32))
        self.assertIsNotNone(first)
        assert first is not None
        np.testing.assert_allclose(first.delta_signature, np.zeros_like(first.path_signature))

        second = tracker.update(np.asarray([1.0, 3.0], dtype=np.float32))
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second.path_signature.shape[0], 6)
        self.assertTrue(np.any(second.delta_signature != 0))

        tracker.reset()
        reset_step = tracker.update(np.asarray([2.0, 5.0], dtype=np.float32))
        self.assertIsNotNone(reset_step)
        assert reset_step is not None
        np.testing.assert_allclose(
            reset_step.delta_signature,
            np.zeros_like(reset_step.path_signature),
        )

    def test_disabled_tracker_returns_none(self) -> None:
        tracker = StreamingSignatureTracker(enabled=False, depth=3, backend="simple")
        self.assertIsNone(tracker.update(np.asarray([1.0, 2.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()

