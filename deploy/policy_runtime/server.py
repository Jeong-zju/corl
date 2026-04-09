from __future__ import annotations

if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deploy.bridge.protocol import ActionPacket, ControlPacket, ObservationPacket, decode_packet
from deploy.policy_runtime.loader import PolicyBundle, load_policy_bundle
from deploy.policy_runtime.preprocess import preprocess_packet
from deploy.transport import close_socket, make_socket, require_zmq
from deploy.utils import bootstrap_main_pythonpath


@dataclass(slots=True)
class ServerArgs:
    policy_type: str
    policy_path: str
    device: str
    bind: str
    control_bind: str
    n_action_steps: int | None


class PolicyRuntimeServer:
    def __init__(self, *, bundle: PolicyBundle, inference_bind: str, control_bind: str) -> None:
        self.bundle = bundle
        self.inference_bind = inference_bind
        self.control_bind = control_bind
        self._action_seq = 0
        self._last_episode_id: str | None = None
        self._paused = False
        self._running = True

    def _handle_observation(self, packet: ObservationPacket) -> ActionPacket:
        self._action_seq += 1
        if self._paused:
            return ActionPacket(
                seq=self._action_seq,
                obs_seq=packet.seq,
                stamp_ns=time.time_ns(),
                runtime_ms=0.0,
                action=np.zeros((0,), dtype=np.float32),
                status="stale",
                message="policy runtime is paused",
            )

        if packet.reset or packet.episode_id != self._last_episode_id:
            self.bundle.reset()
            self._last_episode_id = packet.episode_id

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Policy inference requires `torch`.") from exc

        start_s = time.perf_counter()
        try:
            obs = preprocess_packet(self.bundle, packet)
            with torch.no_grad():
                predicted_action = self.bundle.policy.select_action(obs)
            predicted_action = self.bundle.postprocessor(predicted_action)
            action_np = (
                predicted_action.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            )
            return ActionPacket(
                seq=self._action_seq,
                obs_seq=packet.seq,
                stamp_ns=time.time_ns(),
                runtime_ms=(time.perf_counter() - start_s) * 1000.0,
                action=np.array(action_np, copy=True),
                status="ok",
                metadata={"episode_id": packet.episode_id},
            )
        except Exception as exc:  # pragma: no cover - exercised in runtime
            return ActionPacket(
                seq=self._action_seq,
                obs_seq=packet.seq,
                stamp_ns=time.time_ns(),
                runtime_ms=(time.perf_counter() - start_s) * 1000.0,
                action=np.zeros((0,), dtype=np.float32),
                status="error",
                message=str(exc),
                metadata={"episode_id": packet.episode_id},
            )

    def _handle_control(self, packet: ControlPacket) -> ControlPacket:
        command = packet.command
        if command == "health_check":
            return ControlPacket(
                command="health_check_ack",
                params={
                    "ok": True,
                    "policy_type": self.bundle.policy_type,
                    "policy_dir": str(self.bundle.policy_dir),
                    "device": self.bundle.device,
                    "paused": self._paused,
                },
            )
        if command == "reset_episode":
            self.bundle.reset()
            self._last_episode_id = None
            return ControlPacket(command="ack", params={"ok": True, "command": command})
        if command == "pause":
            self._paused = True
            return ControlPacket(command="ack", params={"ok": True, "command": command})
        if command == "resume":
            self._paused = False
            return ControlPacket(command="ack", params={"ok": True, "command": command})
        if command == "load_checkpoint":
            new_policy_path = packet.params.get("policy_path")
            new_device = packet.params.get("device", self.bundle.device)
            new_n_action_steps = packet.params.get("n_action_steps")
            if not new_policy_path:
                return ControlPacket(
                    command="error",
                    params={"ok": False, "message": "`policy_path` is required for load_checkpoint."},
                )
            main_root = Path(__file__).resolve().parents[2]
            self.bundle = load_policy_bundle(
                main_root=main_root,
                policy_path=str(new_policy_path),
                policy_type=self.bundle.policy_type,
                device=str(new_device),
                n_action_steps=None if new_n_action_steps is None else int(new_n_action_steps),
            )
            self._last_episode_id = None
            self._paused = False
            return ControlPacket(
                command="ack",
                params={
                    "ok": True,
                    "command": command,
                    "policy_dir": str(self.bundle.policy_dir),
                },
            )
        if command == "shutdown":
            self._running = False
            return ControlPacket(command="ack", params={"ok": True, "command": command})
        return ControlPacket(
            command="error",
            params={"ok": False, "message": f"Unsupported control command: {command!r}."},
        )

    def run(self) -> None:
        zmq = require_zmq()
        ctx = zmq.Context.instance()
        inference_socket = make_socket(ctx, zmq.REP, self.inference_bind, bind=True, snd_hwm=2, rcv_hwm=2)
        control_socket = make_socket(ctx, zmq.REP, self.control_bind, bind=True, snd_hwm=2, rcv_hwm=2)
        poller = zmq.Poller()
        poller.register(inference_socket, zmq.POLLIN)
        poller.register(control_socket, zmq.POLLIN)

        try:
            while self._running:
                ready = dict(poller.poll(100))
                if inference_socket in ready:
                    packet = decode_packet(inference_socket.recv_multipart())
                    if not isinstance(packet, ObservationPacket):
                        response = ActionPacket(
                            seq=self._action_seq + 1,
                            obs_seq=-1,
                            stamp_ns=time.time_ns(),
                            runtime_ms=0.0,
                            action=np.zeros((0,), dtype=np.float32),
                            status="error",
                            message=f"Expected observation packet, got {type(packet).__name__}.",
                        )
                    else:
                        response = self._handle_observation(packet)
                    inference_socket.send_multipart(response.to_multipart())

                if control_socket in ready:
                    packet = decode_packet(control_socket.recv_multipart())
                    if not isinstance(packet, ControlPacket):
                        response = ControlPacket(
                            command="error",
                            params={"ok": False, "message": "Expected control packet."},
                        )
                    else:
                        response = self._handle_control(packet)
                    control_socket.send_multipart(response.to_multipart())
        finally:
            close_socket(inference_socket)
            close_socket(control_socket)


def parse_args(argv: list[str] | None = None) -> ServerArgs:
    parser = argparse.ArgumentParser(
        description="Run a persistent policy runtime service for `main/` checkpoints."
    )
    parser.add_argument("--policy-type", choices=["act", "streaming_act"], required=True)
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bind", type=str, default="tcp://*:5555")
    parser.add_argument("--control-bind", type=str, default="tcp://*:5558")
    parser.add_argument("--n-action-steps", type=int, default=None)
    args = parser.parse_args(argv)
    return ServerArgs(
        policy_type=str(args.policy_type),
        policy_path=str(args.policy_path),
        device=str(args.device),
        bind=str(args.bind),
        control_bind=str(args.control_bind),
        n_action_steps=None if args.n_action_steps is None else int(args.n_action_steps),
    )


def main(argv: list[str] | None = None) -> None:
    main_root = bootstrap_main_pythonpath(__file__)
    args = parse_args(argv)
    bundle = load_policy_bundle(
        main_root=main_root,
        policy_path=args.policy_path,
        policy_type=args.policy_type,
        device=args.device,
        n_action_steps=args.n_action_steps,
    )
    server = PolicyRuntimeServer(
        bundle=bundle,
        inference_bind=args.bind,
        control_bind=args.control_bind,
    )
    server.run()


if __name__ == "__main__":
    main()

