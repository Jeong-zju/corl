from __future__ import annotations

import time
from dataclasses import dataclass

from deploy.bridge.protocol import SensorPacket


@dataclass(slots=True)
class StreamRequirement:
    stream: str
    max_age_ms: int
    required: bool = True


@dataclass(slots=True)
class BufferedSample:
    packet: SensorPacket
    received_mono_ns: int


@dataclass(slots=True)
class SyncSnapshot:
    stamp_ns: int
    samples: dict[str, SensorPacket]


class LatestSensorCache:
    def __init__(self, *, max_skew_ms: int) -> None:
        self._max_skew_ns = int(max_skew_ms) * 1_000_000
        self._samples: dict[str, BufferedSample] = {}

    def clear(self) -> None:
        self._samples.clear()

    def update(self, packet: SensorPacket, *, received_mono_ns: int | None = None) -> None:
        self._samples[packet.stream] = BufferedSample(
            packet=packet,
            received_mono_ns=time.monotonic_ns() if received_mono_ns is None else int(received_mono_ns),
        )

    def snapshot(
        self,
        requirements: list[StreamRequirement],
        *,
        now_mono_ns: int | None = None,
    ) -> tuple[SyncSnapshot | None, str | None]:
        now_ns = time.monotonic_ns() if now_mono_ns is None else int(now_mono_ns)
        selected: dict[str, SensorPacket] = {}
        latest_stamp_ns = 0

        for requirement in requirements:
            buffered = self._samples.get(requirement.stream)
            if buffered is None:
                if requirement.required:
                    return None, f"missing:{requirement.stream}"
                continue

            age_ns = now_ns - buffered.received_mono_ns
            max_age_ns = int(requirement.max_age_ms) * 1_000_000
            if age_ns > max_age_ns:
                if requirement.required:
                    return None, f"stale:{requirement.stream}"
                continue

            selected[requirement.stream] = buffered.packet
            latest_stamp_ns = max(latest_stamp_ns, int(buffered.packet.stamp_ns))

        if latest_stamp_ns <= 0:
            return None, "no_fresh_streams"

        for requirement in requirements:
            if requirement.stream not in selected:
                continue
            packet = selected[requirement.stream]
            skew_ns = latest_stamp_ns - int(packet.stamp_ns)
            if skew_ns > self._max_skew_ns:
                return None, f"skew:{requirement.stream}"

        return SyncSnapshot(stamp_ns=latest_stamp_ns, samples=selected), None
