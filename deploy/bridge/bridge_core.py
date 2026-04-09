from __future__ import annotations

if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from deploy.bridge.protocol import (
    ActionPacket,
    ControlPacket,
    ObservationPacket,
    RobotCommandPacket,
    SensorPacket,
    decode_packet,
)
from deploy.bridge.signature_runtime import StreamingSignatureTracker
from deploy.bridge.sync import LatestSensorCache, StreamRequirement
from deploy.transport import close_socket, make_socket, require_zmq
from deploy.utils import bootstrap_main_pythonpath, load_mapping_file, nested_mapping_get


@dataclass(slots=True)
class BridgeConfig:
    sensor_bind: str
    command_bind: str
    control_bind: str | None
    policy_endpoint: str
    policy_control_endpoint: str | None
    policy_type: str
    control_rate_hz: float
    policy_request_timeout_ms: int
    signature_backend: str
    signature_depth: int
    state_streams: dict[str, str]
    image_streams: dict[str, str]
    freshness_ms: dict[str, int]
    max_skew_ms: int
    left_arm_dim: int
    right_arm_dim: int
    enable_base_action: bool
    initial_episode_id: str
    initial_mode: str

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "BridgeConfig":
        root = nested_mapping_get(mapping, "bridge", default=mapping)
        state_streams = nested_mapping_get(root, "state_streams")
        image_streams = nested_mapping_get(root, "image_streams")
        freshness_ms = nested_mapping_get(root, "freshness_ms")
        return cls(
            sensor_bind=str(root.get("sensor_bind", "tcp://*:5556")),
            command_bind=str(root.get("command_bind", "tcp://*:5557")),
            control_bind=root.get("control_bind"),
            policy_endpoint=str(root.get("policy_endpoint", "tcp://127.0.0.1:5555")),
            policy_control_endpoint=root.get("policy_control_endpoint"),
            policy_type=str(root.get("policy_type", "streaming_act")),
            control_rate_hz=float(root.get("control_rate_hz", 30.0)),
            policy_request_timeout_ms=int(root.get("policy_request_timeout_ms", 80)),
            signature_backend=str(root.get("signature_backend", "simple")),
            signature_depth=int(root.get("signature_depth", 3)),
            state_streams={str(key): str(value) for key, value in state_streams.items()},
            image_streams={str(key): str(value) for key, value in image_streams.items()},
            freshness_ms={str(key): int(value) for key, value in freshness_ms.items()},
            max_skew_ms=int(root.get("max_skew_ms", 40)),
            left_arm_dim=int(root.get("left_arm_dim", 7)),
            right_arm_dim=int(root.get("right_arm_dim", 7)),
            enable_base_action=bool(root.get("enable_base_action", False)),
            initial_episode_id=str(root.get("initial_episode_id", "episode-0000")),
            initial_mode=str(root.get("initial_mode", "auto")),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "BridgeConfig":
        return cls.from_mapping(load_mapping_file(path))


class PolicyClient:
    def __init__(
        self,
        *,
        endpoint: str,
        timeout_ms: int,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)
        self._ctx = require_zmq().Context.instance()
        self._socket = self._make_socket()

    def _make_socket(self):
        zmq = require_zmq()
        return make_socket(
            self._ctx,
            zmq.REQ,
            self.endpoint,
            bind=False,
            snd_hwm=2,
            rcv_hwm=2,
        )

    def request_action(self, packet: ObservationPacket) -> ActionPacket:
        zmq = require_zmq()
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        self._socket.send_multipart(packet.to_multipart())
        ready = dict(poller.poll(self.timeout_ms))
        if self._socket not in ready:
            close_socket(self._socket)
            self._socket = self._make_socket()
            raise TimeoutError(
                f"Timed out waiting for policy runtime response after {self.timeout_ms} ms."
            )
        response = decode_packet(self._socket.recv_multipart())
        if not isinstance(response, ActionPacket):
            raise RuntimeError(
                f"Expected ActionPacket from policy runtime, got {type(response).__name__}."
            )
        return response

    def close(self) -> None:
        close_socket(self._socket)


class BridgeRuntime:
    def __init__(self, *, config: BridgeConfig) -> None:
        self.config = config
        self._sensor_cache = LatestSensorCache(max_skew_ms=self.config.max_skew_ms)
        self._signature_tracker = StreamingSignatureTracker(
            enabled=self.config.policy_type == "streaming_act",
            depth=self.config.signature_depth,
            backend=self.config.signature_backend,
        )
        self._obs_seq = 0
        self._command_seq = 0
        self._episode_id = self.config.initial_episode_id
        self._mode = self.config.initial_mode
        self._pending_reset = True
        self._running = True
        self._last_obs_seq = -1
        self._last_hold_reason = ""
        self._last_reported_status = "init"

    def _log(self, level: str, message: str) -> None:
        print(f"[{level}] {message}", flush=True)

    def _report_status_transition(self, *, status: str, detail: str) -> None:
        marker = f"{status}:{detail}"
        if marker == self._last_reported_status:
            return
        self._last_reported_status = marker
        if status == "hold":
            self._log("status", f"Bridge entering hold: {detail}")
            return
        if status == "auto":
            self._log("status", f"Bridge resumed auto execution: {detail}")
            return
        self._log("status", f"{status}: {detail}")

    def _build_requirements(self) -> list[StreamRequirement]:
        requirements = []
        for stream in self.config.state_streams.values():
            max_age_ms = self.config.freshness_ms.get(stream, 20)
            requirements.append(StreamRequirement(stream=stream, max_age_ms=max_age_ms))
        for stream in self.config.image_streams.values():
            max_age_ms = self.config.freshness_ms.get(stream, 50)
            requirements.append(StreamRequirement(stream=stream, max_age_ms=max_age_ms))
        return requirements

    def _build_state_vector(self, samples: dict[str, SensorPacket]) -> np.ndarray:
        base_stream = self.config.state_streams["base"]
        left_stream = self.config.state_streams["left_arm"]
        right_stream = self.config.state_streams["right_arm"]

        base = np.asarray(samples[base_stream].array, dtype=np.float32).reshape(-1)
        left = np.asarray(samples[left_stream].array, dtype=np.float32).reshape(-1)
        right = np.asarray(samples[right_stream].array, dtype=np.float32).reshape(-1)

        if base.shape[0] < 3:
            raise ValueError(f"Base stream `{base_stream}` must provide 3 values, got {base.shape[0]}.")
        if left.shape[0] < self.config.left_arm_dim:
            raise ValueError(
                f"Left arm stream `{left_stream}` must provide at least {self.config.left_arm_dim} values, "
                f"got {left.shape[0]}."
            )
        if right.shape[0] < self.config.right_arm_dim:
            raise ValueError(
                f"Right arm stream `{right_stream}` must provide at least {self.config.right_arm_dim} values, "
                f"got {right.shape[0]}."
            )

        return np.concatenate(
            [
                base[:3],
                left[: self.config.left_arm_dim],
                right[: self.config.right_arm_dim],
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    def _build_observation_packet(
        self,
        *,
        stamp_ns: int,
        samples: dict[str, SensorPacket],
    ) -> ObservationPacket:
        self._obs_seq += 1
        state = self._build_state_vector(samples)
        images = {}
        for policy_image_key, stream_name in self.config.image_streams.items():
            image = samples[stream_name].array
            if image is None:
                raise ValueError(f"Image stream `{stream_name}` has no payload.")
            images[policy_image_key] = np.asarray(image, dtype=np.uint8)

        signature_step = self._signature_tracker.update(state)
        return ObservationPacket(
            seq=self._obs_seq,
            episode_id=self._episode_id,
            stamp_ns=stamp_ns,
            reset=self._pending_reset,
            policy_type=self.config.policy_type,
            mode=self._mode,
            state=state,
            images=images,
            path_signature=None if signature_step is None else signature_step.path_signature,
            delta_signature=None if signature_step is None else signature_step.delta_signature,
            metadata={"signature_step": None if signature_step is None else signature_step.step_index},
        )

    def _build_command_packet(
        self,
        action_packet: ActionPacket,
    ) -> RobotCommandPacket:
        self._command_seq += 1
        raw_action = np.asarray(action_packet.action, dtype=np.float32).reshape(-1)
        expected_dim = 3 + self.config.left_arm_dim + self.config.right_arm_dim
        if raw_action.shape[0] != expected_dim:
            return RobotCommandPacket.hold(
                seq=self._command_seq,
                obs_seq=action_packet.obs_seq,
                stamp_ns=time.time_ns(),
                raw_action_dim=max(expected_dim, int(raw_action.shape[0])),
                left_arm_dim=self.config.left_arm_dim,
                right_arm_dim=self.config.right_arm_dim,
                hold_reason=(
                    "action_dim_mismatch:"
                    f"expected={expected_dim},got={raw_action.shape[0]}"
                ),
                metadata={"status": action_packet.status, "message": action_packet.message},
            )

        left_start = 3
        left_end = left_start + self.config.left_arm_dim
        right_end = left_end + self.config.right_arm_dim
        base = raw_action[:3] if self.config.enable_base_action else np.zeros((3,), dtype=np.float32)
        mode = "auto" if action_packet.status == "ok" and self._mode == "auto" else "hold"
        hold_reason = "" if mode == "auto" else (action_packet.message or action_packet.status)
        return RobotCommandPacket(
            seq=self._command_seq,
            obs_seq=action_packet.obs_seq,
            stamp_ns=time.time_ns(),
            mode=mode,
            status=action_packet.status,
            raw_action=raw_action,
            left_arm=raw_action[left_start:left_end],
            right_arm=raw_action[left_end:right_end],
            base=base,
            hold_reason=hold_reason,
            metadata={
                "policy_runtime_ms": action_packet.runtime_ms,
                "policy_status": action_packet.status,
            },
        )

    def _make_hold_command(self, *, reason: str, obs_seq: int) -> RobotCommandPacket:
        self._command_seq += 1
        self._last_hold_reason = reason
        return RobotCommandPacket.hold(
            seq=self._command_seq,
            obs_seq=obs_seq,
            stamp_ns=time.time_ns(),
            raw_action_dim=3 + self.config.left_arm_dim + self.config.right_arm_dim,
            left_arm_dim=self.config.left_arm_dim,
            right_arm_dim=self.config.right_arm_dim,
            hold_reason=reason,
        )

    def _handle_control(self, packet: ControlPacket) -> ControlPacket:
        command = packet.command
        if command == "health_check":
            return ControlPacket(
                command="health_check_ack",
                params={
                    "ok": True,
                    "episode_id": self._episode_id,
                    "mode": self._mode,
                    "pending_reset": self._pending_reset,
                    "last_obs_seq": self._last_obs_seq,
                    "last_hold_reason": self._last_hold_reason,
                },
            )
        if command == "reset_episode":
            new_episode_id = str(packet.params.get("episode_id", self._episode_id))
            self._episode_id = new_episode_id
            self._pending_reset = True
            self._signature_tracker.reset()
            return ControlPacket(command="ack", params={"ok": True, "episode_id": self._episode_id})
        if command == "set_mode":
            mode = str(packet.params.get("mode", self._mode))
            if mode not in {"auto", "hold", "teleop"}:
                return ControlPacket(
                    command="error",
                    params={"ok": False, "message": f"Unsupported mode: {mode!r}."},
                )
            self._mode = mode
            if mode != "auto":
                self._pending_reset = True
            return ControlPacket(command="ack", params={"ok": True, "mode": self._mode})
        if command == "shutdown":
            self._running = False
            return ControlPacket(command="ack", params={"ok": True})
        return ControlPacket(
            command="error",
            params={"ok": False, "message": f"Unsupported bridge control command: {command!r}."},
        )

    def run(self) -> None:
        zmq = require_zmq()
        ctx = zmq.Context.instance()
        sensor_socket = make_socket(ctx, zmq.PULL, self.config.sensor_bind, bind=True, snd_hwm=64, rcv_hwm=64)
        command_socket = make_socket(ctx, zmq.PUSH, self.config.command_bind, bind=True, snd_hwm=16, rcv_hwm=16)
        control_socket = None
        if self.config.control_bind:
            control_socket = make_socket(
                ctx,
                zmq.REP,
                str(self.config.control_bind),
                bind=True,
                snd_hwm=4,
                rcv_hwm=4,
            )

        poller = zmq.Poller()
        poller.register(sensor_socket, zmq.POLLIN)
        if control_socket is not None:
            poller.register(control_socket, zmq.POLLIN)

        policy_client = PolicyClient(
            endpoint=self.config.policy_endpoint,
            timeout_ms=self.config.policy_request_timeout_ms,
        )
        requirements = self._build_requirements()
        period_s = 1.0 / max(self.config.control_rate_hz, 1e-6)
        next_tick_s = time.monotonic()
        self._log(
            "ready",
            "Bridge is listening: "
            f"sensors={self.config.sensor_bind}, commands={self.config.command_bind}, "
            f"control={self.config.control_bind or 'disabled'}, "
            f"policy={self.config.policy_endpoint}, mode={self._mode}",
        )

        try:
            while self._running:
                timeout_ms = max(0, int((next_tick_s - time.monotonic()) * 1000.0))
                ready = dict(poller.poll(timeout_ms))

                if sensor_socket in ready:
                    while True:
                        try:
                            packet = decode_packet(sensor_socket.recv_multipart(flags=zmq.NOBLOCK))
                        except zmq.Again:
                            break
                        if isinstance(packet, SensorPacket):
                            self._sensor_cache.update(packet)
                        elif isinstance(packet, ControlPacket):
                            response = self._handle_control(packet)
                            if response.command == "ack" and packet.command == "reset_episode":
                                # External reset arrived through the sensor stream.
                                pass

                if control_socket is not None and control_socket in ready:
                    packet = decode_packet(control_socket.recv_multipart())
                    if isinstance(packet, ControlPacket):
                        response = self._handle_control(packet)
                    else:
                        response = ControlPacket(
                            command="error",
                            params={"ok": False, "message": "Expected control packet."},
                        )
                    control_socket.send_multipart(response.to_multipart())

                now_s = time.monotonic()
                if now_s < next_tick_s:
                    continue

                next_tick_s += period_s
                if self._mode != "auto":
                    self._report_status_transition(status="hold", detail=f"mode:{self._mode}")
                    command_socket.send_multipart(
                        self._make_hold_command(reason=f"mode:{self._mode}", obs_seq=self._last_obs_seq).to_multipart()
                    )
                    continue

                snapshot, failure = self._sensor_cache.snapshot(requirements)
                if snapshot is None:
                    self._report_status_transition(
                        status="hold",
                        detail=f"observation_unavailable:{failure}",
                    )
                    command_socket.send_multipart(
                        self._make_hold_command(reason=f"observation_unavailable:{failure}", obs_seq=self._last_obs_seq).to_multipart()
                    )
                    continue

                observation = self._build_observation_packet(
                    stamp_ns=snapshot.stamp_ns,
                    samples=snapshot.samples,
                )
                try:
                    action_packet = policy_client.request_action(observation)
                except Exception as exc:
                    self._report_status_transition(
                        status="hold",
                        detail=f"policy_request_failed:{exc}",
                    )
                    command_socket.send_multipart(
                        self._make_hold_command(
                            reason=f"policy_request_failed:{exc}",
                            obs_seq=observation.seq,
                        ).to_multipart()
                    )
                    continue

                self._pending_reset = False
                self._last_obs_seq = observation.seq
                command_packet = self._build_command_packet(action_packet)
                if command_packet.mode == "auto":
                    self._report_status_transition(
                        status="auto",
                        detail=f"obs_seq={observation.seq}",
                    )
                else:
                    self._report_status_transition(
                        status="hold",
                        detail=command_packet.hold_reason or command_packet.status,
                    )
                command_socket.send_multipart(command_packet.to_multipart())
        finally:
            policy_client.close()
            close_socket(sensor_socket)
            close_socket(command_socket)
            if control_socket is not None:
                close_socket(control_socket)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bridge between ROS1 sensor streams and the main/ policy runtime."
    )
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    bootstrap_main_pythonpath(__file__)
    args = parse_args(argv)
    config = BridgeConfig.from_file(args.config)
    runtime = BridgeRuntime(config=config)
    runtime.run()


if __name__ == "__main__":
    main()
