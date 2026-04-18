from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SOURCE_IMAGE_LEFT = "image_left"
SOURCE_IMAGE_RIGHT = "image_right"
SOURCE_IMAGE_TOP = "image_top"
SOURCE_JOINT_LEFT = "joint_state_left"
SOURCE_JOINT_RIGHT = "joint_state_right"
SOURCE_ODOM = "odom"


@dataclass
class TimedSample:
    stamp_ns: int
    received_ns: int
    value: np.ndarray | bytes


class SensorCache:
    def __init__(self) -> None:
        self._samples: dict[str, TimedSample] = {}

    def update(self, source: str, sample: TimedSample) -> None:
        self._samples[source] = sample

    def get(self, source: str) -> TimedSample | None:
        return self._samples.get(source)

    def latest_state_vector(self, *, arm_dof: int) -> np.ndarray | None:
        odom = self.get(SOURCE_ODOM)
        left = self.get(SOURCE_JOINT_LEFT)
        right = self.get(SOURCE_JOINT_RIGHT)
        if odom is None or left is None or right is None:
            return None

        state = np.empty((3 + arm_dof * 2,), dtype=np.float32)
        split = 3 + arm_dof
        state[:3] = np.asarray(odom.value, dtype=np.float32).reshape(-1)[:3]
        state[3:split] = np.asarray(left.value, dtype=np.float32).reshape(-1)[:arm_dof]
        state[split:] = np.asarray(right.value, dtype=np.float32).reshape(-1)[:arm_dof]
        return state

    def all_sources_present(self, sources: list[str]) -> bool:
        return all(source in self._samples for source in sources)

    def max_age_ms(self, sources: list[str], *, now_ns: int) -> float:
        ages = [
            max(0.0, (now_ns - self._samples[source].received_ns) / 1_000_000.0)
            for source in sources
            if source in self._samples
        ]
        return max(ages) if ages else float("inf")

    def stamp_span_ms(self, sources: list[str]) -> float:
        stamps = [
            self._samples[source].stamp_ns
            for source in sources
            if source in self._samples
        ]
        if not stamps:
            return float("inf")
        return max(stamps) / 1_000_000.0 - min(stamps) / 1_000_000.0


def reorder_joint_positions(
    *,
    positions: list[float] | tuple[float, ...],
    names: list[str] | tuple[str, ...] | None,
    expected_names: list[str] | tuple[str, ...] | None,
    dof: int,
) -> np.ndarray:
    if dof <= 0:
        raise ValueError(f"`dof` must be positive, got {dof}.")

    raw_positions = list(positions)
    expected = [str(name) for name in list(expected_names or [])[:dof]]
    observed_names = [str(name) for name in list(names or [])]
    if expected and observed_names and len(observed_names) == len(raw_positions):
        index_by_name = {name: idx for idx, name in enumerate(observed_names)}
        if all(name in index_by_name for name in expected):
            reordered = [raw_positions[index_by_name[name]] for name in expected]
            return np.asarray(reordered, dtype=np.float32)

    truncated = raw_positions[:dof]
    if len(truncated) < dof:
        truncated = truncated + [0.0] * (dof - len(truncated))
    return np.asarray(truncated, dtype=np.float32)
