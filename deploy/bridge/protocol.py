from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1

KIND_SENSOR = "sensor"
KIND_OBSERVATION = "observation"
KIND_ACTION = "action"
KIND_CONTROL = "control"
KIND_ROBOT_COMMAND = "robot_command"


def _ensure_array(array: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    normalized = np.asarray(array)
    if not normalized.flags.c_contiguous:
        normalized = np.ascontiguousarray(normalized)
    return normalized


def _encode_envelope(
    *,
    kind: str,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> list[bytes]:
    ordered_names = sorted(arrays)
    payload_specs = []
    payload_frames: list[bytes] = []
    for name in ordered_names:
        array = _ensure_array(arrays[name])
        payload_specs.append(
            {
                "name": name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
        )
        payload_frames.append(array.tobytes(order="C"))

    envelope = {
        "version": PROTOCOL_VERSION,
        "kind": kind,
        "metadata": metadata,
        "arrays": payload_specs,
    }
    header = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return [header, *payload_frames]


def _decode_envelope(frames: list[bytes]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not frames:
        raise ValueError("Cannot decode an empty multipart message.")
    header = json.loads(frames[0].decode("utf-8"))
    if header.get("version") != PROTOCOL_VERSION:
        raise ValueError(
            f"Protocol version mismatch: expected={PROTOCOL_VERSION}, got={header.get('version')}."
        )
    specs = header.get("arrays", [])
    if len(frames) != len(specs) + 1:
        raise ValueError(
            "Multipart frame count mismatch when decoding packet: "
            f"frames={len(frames)}, arrays={len(specs)}."
        )

    arrays: dict[str, np.ndarray] = {}
    for spec, frame in zip(specs, frames[1:], strict=True):
        name = str(spec["name"])
        dtype = np.dtype(spec["dtype"])
        shape = tuple(int(dim) for dim in spec["shape"])
        array = np.frombuffer(frame, dtype=dtype)
        if shape:
            array = array.reshape(shape)
        arrays[name] = np.array(array, copy=True)
    return header, arrays


@dataclass(slots=True)
class SensorPacket:
    stream: str
    seq: int
    stamp_ns: int
    payload_type: str
    array: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_multipart(self) -> list[bytes]:
        metadata = {
            "stream": self.stream,
            "seq": int(self.seq),
            "stamp_ns": int(self.stamp_ns),
            "payload_type": self.payload_type,
            "metadata": self.metadata,
        }
        arrays = {"payload": _ensure_array(self.array)} if self.array is not None else {}
        return _encode_envelope(kind=KIND_SENSOR, metadata=metadata, arrays=arrays)

    @classmethod
    def from_multipart(cls, frames: list[bytes]) -> "SensorPacket":
        header, arrays = _decode_envelope(frames)
        if header["kind"] != KIND_SENSOR:
            raise ValueError(f"Expected `{KIND_SENSOR}` packet, got `{header['kind']}`.")
        metadata = header["metadata"]
        return cls(
            stream=str(metadata["stream"]),
            seq=int(metadata["seq"]),
            stamp_ns=int(metadata["stamp_ns"]),
            payload_type=str(metadata["payload_type"]),
            array=arrays.get("payload"),
            metadata=dict(metadata.get("metadata", {})),
        )


@dataclass(slots=True)
class ObservationPacket:
    seq: int
    episode_id: str
    stamp_ns: int
    reset: bool
    policy_type: str
    mode: str
    state: np.ndarray
    images: dict[str, np.ndarray]
    path_signature: np.ndarray | None = None
    delta_signature: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_multipart(self) -> list[bytes]:
        arrays = {"state": _ensure_array(self.state)}
        for key, image in sorted(self.images.items()):
            arrays[f"image::{key}"] = _ensure_array(image)
        if self.path_signature is not None:
            arrays["path_signature"] = _ensure_array(self.path_signature)
        if self.delta_signature is not None:
            arrays["delta_signature"] = _ensure_array(self.delta_signature)
        metadata = {
            "seq": int(self.seq),
            "episode_id": self.episode_id,
            "stamp_ns": int(self.stamp_ns),
            "reset": bool(self.reset),
            "policy_type": self.policy_type,
            "mode": self.mode,
            "metadata": self.metadata,
        }
        return _encode_envelope(kind=KIND_OBSERVATION, metadata=metadata, arrays=arrays)

    @classmethod
    def from_multipart(cls, frames: list[bytes]) -> "ObservationPacket":
        header, arrays = _decode_envelope(frames)
        if header["kind"] != KIND_OBSERVATION:
            raise ValueError(f"Expected `{KIND_OBSERVATION}` packet, got `{header['kind']}`.")
        metadata = header["metadata"]
        images = {
            key.split("::", 1)[1]: value
            for key, value in arrays.items()
            if key.startswith("image::")
        }
        return cls(
            seq=int(metadata["seq"]),
            episode_id=str(metadata["episode_id"]),
            stamp_ns=int(metadata["stamp_ns"]),
            reset=bool(metadata["reset"]),
            policy_type=str(metadata["policy_type"]),
            mode=str(metadata["mode"]),
            state=arrays["state"],
            images=images,
            path_signature=arrays.get("path_signature"),
            delta_signature=arrays.get("delta_signature"),
            metadata=dict(metadata.get("metadata", {})),
        )


@dataclass(slots=True)
class ActionPacket:
    seq: int
    obs_seq: int
    stamp_ns: int
    runtime_ms: float
    action: np.ndarray
    status: str
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_multipart(self) -> list[bytes]:
        metadata = {
            "seq": int(self.seq),
            "obs_seq": int(self.obs_seq),
            "stamp_ns": int(self.stamp_ns),
            "runtime_ms": float(self.runtime_ms),
            "status": self.status,
            "message": self.message,
            "metadata": self.metadata,
        }
        arrays = {"action": _ensure_array(self.action)}
        return _encode_envelope(kind=KIND_ACTION, metadata=metadata, arrays=arrays)

    @classmethod
    def from_multipart(cls, frames: list[bytes]) -> "ActionPacket":
        header, arrays = _decode_envelope(frames)
        if header["kind"] != KIND_ACTION:
            raise ValueError(f"Expected `{KIND_ACTION}` packet, got `{header['kind']}`.")
        metadata = header["metadata"]
        return cls(
            seq=int(metadata["seq"]),
            obs_seq=int(metadata["obs_seq"]),
            stamp_ns=int(metadata["stamp_ns"]),
            runtime_ms=float(metadata["runtime_ms"]),
            action=arrays["action"],
            status=str(metadata["status"]),
            message=str(metadata.get("message", "")),
            metadata=dict(metadata.get("metadata", {})),
        )


@dataclass(slots=True)
class ControlPacket:
    command: str
    stamp_ns: int = field(default_factory=time.time_ns)
    params: dict[str, Any] = field(default_factory=dict)

    def to_multipart(self) -> list[bytes]:
        metadata = {
            "command": self.command,
            "stamp_ns": int(self.stamp_ns),
            "params": self.params,
        }
        return _encode_envelope(kind=KIND_CONTROL, metadata=metadata, arrays={})

    @classmethod
    def from_multipart(cls, frames: list[bytes]) -> "ControlPacket":
        header, _ = _decode_envelope(frames)
        if header["kind"] != KIND_CONTROL:
            raise ValueError(f"Expected `{KIND_CONTROL}` packet, got `{header['kind']}`.")
        metadata = header["metadata"]
        return cls(
            command=str(metadata["command"]),
            stamp_ns=int(metadata["stamp_ns"]),
            params=dict(metadata.get("params", {})),
        )


@dataclass(slots=True)
class RobotCommandPacket:
    seq: int
    obs_seq: int
    stamp_ns: int
    mode: str
    status: str
    raw_action: np.ndarray
    left_arm: np.ndarray
    right_arm: np.ndarray
    base: np.ndarray
    hold_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_multipart(self) -> list[bytes]:
        metadata = {
            "seq": int(self.seq),
            "obs_seq": int(self.obs_seq),
            "stamp_ns": int(self.stamp_ns),
            "mode": self.mode,
            "status": self.status,
            "hold_reason": self.hold_reason,
            "metadata": self.metadata,
        }
        arrays = {
            "raw_action": _ensure_array(self.raw_action),
            "left_arm": _ensure_array(self.left_arm),
            "right_arm": _ensure_array(self.right_arm),
            "base": _ensure_array(self.base),
        }
        return _encode_envelope(kind=KIND_ROBOT_COMMAND, metadata=metadata, arrays=arrays)

    @classmethod
    def hold(
        cls,
        *,
        seq: int,
        obs_seq: int,
        stamp_ns: int,
        raw_action_dim: int,
        left_arm_dim: int,
        right_arm_dim: int,
        hold_reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> "RobotCommandPacket":
        return cls(
            seq=int(seq),
            obs_seq=int(obs_seq),
            stamp_ns=int(stamp_ns),
            mode="hold",
            status="hold",
            raw_action=np.zeros((raw_action_dim,), dtype=np.float32),
            left_arm=np.zeros((left_arm_dim,), dtype=np.float32),
            right_arm=np.zeros((right_arm_dim,), dtype=np.float32),
            base=np.zeros((3,), dtype=np.float32),
            hold_reason=hold_reason,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_multipart(cls, frames: list[bytes]) -> "RobotCommandPacket":
        header, arrays = _decode_envelope(frames)
        if header["kind"] != KIND_ROBOT_COMMAND:
            raise ValueError(
                f"Expected `{KIND_ROBOT_COMMAND}` packet, got `{header['kind']}`."
            )
        metadata = header["metadata"]
        return cls(
            seq=int(metadata["seq"]),
            obs_seq=int(metadata["obs_seq"]),
            stamp_ns=int(metadata["stamp_ns"]),
            mode=str(metadata["mode"]),
            status=str(metadata["status"]),
            raw_action=arrays["raw_action"],
            left_arm=arrays["left_arm"],
            right_arm=arrays["right_arm"],
            base=arrays["base"],
            hold_reason=str(metadata.get("hold_reason", "")),
            metadata=dict(metadata.get("metadata", {})),
        )


def decode_packet(frames: list[bytes]) -> SensorPacket | ObservationPacket | ActionPacket | ControlPacket | RobotCommandPacket:
    if not frames:
        raise ValueError("Cannot decode an empty multipart message.")
    header = json.loads(frames[0].decode("utf-8"))
    kind = header.get("kind")
    if kind == KIND_SENSOR:
        return SensorPacket.from_multipart(frames)
    if kind == KIND_OBSERVATION:
        return ObservationPacket.from_multipart(frames)
    if kind == KIND_ACTION:
        return ActionPacket.from_multipart(frames)
    if kind == KIND_CONTROL:
        return ControlPacket.from_multipart(frames)
    if kind == KIND_ROBOT_COMMAND:
        return RobotCommandPacket.from_multipart(frames)
    raise ValueError(f"Unsupported packet kind: {kind!r}.")

