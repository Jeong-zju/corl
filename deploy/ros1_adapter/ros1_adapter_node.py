from __future__ import annotations

if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import queue
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from deploy.bridge.protocol import RobotCommandPacket, SensorPacket, decode_packet
from deploy.ros1_adapter.ros_publishers import BaseTwistPublisher, JointTrajectoryCommandPublisher
from deploy.ros1_adapter.ros_topics import (
    ImageTopicConfig,
    JointStateTopicConfig,
    OdomTopicConfig,
    Ros1AdapterConfig,
)
from deploy.transport import close_socket, make_socket, require_zmq
from deploy.utils import bootstrap_main_pythonpath


def _stamp_to_ns(stamp: Any) -> int:
    try:
        value = int(stamp.to_nsec())
    except Exception:
        value = 0
    return value if value > 0 else time.time_ns()


def _reorder_joint_positions(msg: Any, joint_names: tuple[str, ...]) -> np.ndarray:
    positions = np.asarray(msg.position, dtype=np.float32)
    if not joint_names:
        return positions
    index_by_name = {str(name): idx for idx, name in enumerate(msg.name)}
    reordered = []
    missing = [name for name in joint_names if name not in index_by_name]
    if missing:
        raise KeyError(f"JointState message is missing joints: {missing}.")
    for name in joint_names:
        reordered.append(float(positions[index_by_name[name]]))
    return np.asarray(reordered, dtype=np.float32)


def _decode_compressed_rgb(msg: Any) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Compressed image decoding requires `opencv-python` in the ROS1 environment."
        ) from exc

    encoded = np.frombuffer(msg.data, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Failed to decode compressed image payload.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@dataclass(slots=True)
class OutgoingPacket:
    packet: SensorPacket


class Ros1BridgeAdapter:
    def __init__(self, *, config: Ros1AdapterConfig) -> None:
        self.config = config
        self._seq = 0
        self._queue: queue.Queue[OutgoingPacket] = queue.Queue(maxsize=self.config.outgoing_queue_size)
        self._stream_counts: dict[str, int] = {}
        self._stream_last_recv_ns: dict[str, int] = {}
        self._stream_last_stamp_ns: dict[str, int] = {}
        self._last_status_log_ns = 0
        self._last_command_summary = "none"

        import rospy
        import nav_msgs.msg
        import sensor_msgs.msg

        self._rospy = rospy
        self._sensor_msgs = sensor_msgs.msg
        self._nav_msgs = nav_msgs.msg
        rospy.init_node(self.config.node_name, anonymous=False)

        zmq = require_zmq()
        self._ctx = zmq.Context.instance()
        self._sensor_socket = make_socket(
            self._ctx,
            zmq.PUSH,
            self.config.sensor_endpoint,
            bind=False,
            snd_hwm=self.config.outgoing_queue_size,
            rcv_hwm=8,
        )
        self._command_socket = make_socket(
            self._ctx,
            zmq.PULL,
            self.config.command_endpoint,
            bind=False,
            snd_hwm=8,
            rcv_hwm=16,
        )

        self._left_arm_publisher = JointTrajectoryCommandPublisher(self.config.left_arm_command)
        self._right_arm_publisher = JointTrajectoryCommandPublisher(self.config.right_arm_command)
        self._base_publisher = (
            BaseTwistPublisher(self.config.base_command)
            if self.config.base_command.enabled
            else None
        )

        self._register_subscribers()
        self._rospy.loginfo(
            "ROS1 adapter ready: sensor_endpoint=%s command_endpoint=%s poll_hz=%.1f",
            self.config.sensor_endpoint,
            self.config.command_endpoint,
            self.config.command_poll_hz,
        )

    def _note_received(self, packet: SensorPacket) -> None:
        stream = packet.stream
        self._stream_counts[stream] = self._stream_counts.get(stream, 0) + 1
        self._stream_last_recv_ns[stream] = time.monotonic_ns()
        self._stream_last_stamp_ns[stream] = int(packet.stamp_ns)
        if self._stream_counts[stream] == 1:
            self._rospy.loginfo(
                "First packet received for stream=%s payload_type=%s topic=%s",
                packet.stream,
                packet.payload_type,
                packet.metadata.get("topic", "?"),
            )

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _enqueue(self, packet: SensorPacket) -> None:
        self._note_received(packet)
        outgoing = OutgoingPacket(packet=packet)
        try:
            self._queue.put_nowait(outgoing)
        except queue.Full:
            self._rospy.logwarn_throttle(
                5.0,
                "ROS1 adapter outgoing queue full; dropping oldest sensor packet.",
            )
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(outgoing)

    def _register_subscribers(self) -> None:
        for image_topic in self.config.image_topics:
            msg_type = self._sensor_msgs.CompressedImage if image_topic.compressed else self._sensor_msgs.Image
            self._rospy.loginfo(
                "Subscribing image stream=%s topic=%s compressed=%s",
                image_topic.stream,
                image_topic.topic,
                image_topic.compressed,
            )
            self._rospy.Subscriber(
                image_topic.topic,
                msg_type,
                self._make_image_callback(image_topic),
                queue_size=1,
                buff_size=2**24,
            )

        self._rospy.loginfo(
            "Subscribing left joint state stream=%s topic=%s",
            self.config.left_arm_state_topic.stream,
            self.config.left_arm_state_topic.topic,
        )
        self._rospy.Subscriber(
            self.config.left_arm_state_topic.topic,
            self._sensor_msgs.JointState,
            self._make_joint_state_callback(self.config.left_arm_state_topic),
            queue_size=10,
        )
        self._rospy.loginfo(
            "Subscribing right joint state stream=%s topic=%s",
            self.config.right_arm_state_topic.stream,
            self.config.right_arm_state_topic.topic,
        )
        self._rospy.Subscriber(
            self.config.right_arm_state_topic.topic,
            self._sensor_msgs.JointState,
            self._make_joint_state_callback(self.config.right_arm_state_topic),
            queue_size=10,
        )
        self._rospy.loginfo(
            "Subscribing odom stream=%s topic=%s",
            self.config.odom_topic.stream,
            self.config.odom_topic.topic,
        )
        self._rospy.Subscriber(
            self.config.odom_topic.topic,
            self._nav_msgs.Odometry,
            self._make_odom_callback(self.config.odom_topic),
            queue_size=20,
        )

    def _make_image_callback(self, config: ImageTopicConfig):
        def _callback(msg: Any) -> None:
            try:
                if config.compressed:
                    image = _decode_compressed_rgb(msg)
                else:
                    raise NotImplementedError(
                        "Raw ROS images are not wired yet. Use a compressed image topic first."
                    )
                self._enqueue(
                    SensorPacket(
                        stream=config.stream,
                        seq=self._next_seq(),
                        stamp_ns=_stamp_to_ns(msg.header.stamp),
                        payload_type="image_rgb8",
                        array=image,
                        metadata={"topic": config.topic},
                    )
                )
            except Exception as exc:
                self._rospy.logerr_throttle(
                    5.0,
                    "Image callback failed for stream=%s topic=%s: %s",
                    config.stream,
                    config.topic,
                    exc,
                )

        return _callback

    def _make_joint_state_callback(self, config: JointStateTopicConfig):
        def _callback(msg: Any) -> None:
            try:
                positions = _reorder_joint_positions(msg, config.joint_names)
                self._enqueue(
                    SensorPacket(
                        stream=config.stream,
                        seq=self._next_seq(),
                        stamp_ns=_stamp_to_ns(msg.header.stamp),
                        payload_type="joint_positions",
                        array=positions,
                        metadata={"topic": config.topic},
                    )
                )
            except Exception as exc:
                self._rospy.logerr_throttle(
                    5.0,
                    "JointState callback failed for stream=%s topic=%s: %s",
                    config.stream,
                    config.topic,
                    exc,
                )

        return _callback

    def _make_odom_callback(self, config: OdomTopicConfig):
        def _callback(msg: Any) -> None:
            try:
                base_velocity = np.asarray(
                    [
                        float(msg.twist.twist.linear.x),
                        float(msg.twist.twist.linear.y),
                        float(msg.twist.twist.angular.z),
                    ],
                    dtype=np.float32,
                )
                self._enqueue(
                    SensorPacket(
                        stream=config.stream,
                        seq=self._next_seq(),
                        stamp_ns=_stamp_to_ns(msg.header.stamp),
                        payload_type="base_velocity",
                        array=base_velocity,
                        metadata={"topic": config.topic},
                    )
                )
            except Exception as exc:
                self._rospy.logerr_throttle(
                    5.0,
                    "Odom callback failed for stream=%s topic=%s: %s",
                    config.stream,
                    config.topic,
                    exc,
                )

        return _callback

    def _drain_sensor_queue(self) -> None:
        while True:
            try:
                outgoing = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._sensor_socket.send_multipart(outgoing.packet.to_multipart())
            except Exception as exc:
                self._rospy.logerr_throttle(
                    5.0,
                    "Failed to forward sensor packet for stream=%s: %s",
                    outgoing.packet.stream,
                    exc,
                )

    def _handle_robot_command(self, packet: RobotCommandPacket) -> None:
        self._last_command_summary = (
            f"mode={packet.mode}, status={packet.status}, obs_seq={packet.obs_seq}, "
            f"hold_reason={packet.hold_reason or '-'}"
        )
        if packet.mode != "auto":
            if self._base_publisher is not None and self.config.publish_zero_base_on_hold:
                self._base_publisher.publish_zero()
            return

        self._left_arm_publisher.publish_positions(packet.left_arm)
        self._right_arm_publisher.publish_positions(packet.right_arm)
        if self._base_publisher is not None:
            self._base_publisher.publish_twist(packet.base)

    def _drain_command_socket(self) -> None:
        zmq = require_zmq()
        while True:
            try:
                packet = decode_packet(self._command_socket.recv_multipart(flags=zmq.NOBLOCK))
            except zmq.Again:
                break
            if not isinstance(packet, RobotCommandPacket):
                self._rospy.logwarn(
                    "ROS1 adapter expected RobotCommandPacket, got %s",
                    type(packet).__name__,
                )
                continue
            self._handle_robot_command(packet)

    def _log_periodic_status(self) -> None:
        interval_ns = int(self.config.status_log_interval_s * 1_000_000_000)
        if interval_ns <= 0:
            return
        now_ns = time.monotonic_ns()
        if now_ns - self._last_status_log_ns < interval_ns:
            return
        self._last_status_log_ns = now_ns
        if not self._stream_counts:
            self._rospy.logwarn("ROS1 adapter has not received any sensor packets yet.")
            return
        parts = []
        for stream in sorted(self._stream_counts):
            age_ms = (now_ns - self._stream_last_recv_ns[stream]) / 1_000_000.0
            parts.append(
                f"{stream}:count={self._stream_counts[stream]},age_ms={age_ms:.1f}"
            )
        self._rospy.loginfo(
            "ROS1 adapter status | sensors=[%s] | last_command=%s | queue=%d",
            "; ".join(parts),
            self._last_command_summary,
            self._queue.qsize(),
        )

    def spin(self) -> None:
        rate = self._rospy.Rate(self.config.command_poll_hz)
        try:
            while not self._rospy.is_shutdown():
                self._drain_sensor_queue()
                self._drain_command_socket()
                self._log_periodic_status()
                rate.sleep()
        finally:
            close_socket(self._sensor_socket)
            close_socket(self._command_socket)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROS1 adapter that forwards topics into the deployment bridge."
    )
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    bootstrap_main_pythonpath(__file__)
    args = parse_args(argv)
    config = Ros1AdapterConfig.from_file(args.config)
    adapter = Ros1BridgeAdapter(config=config)
    adapter.spin()


if __name__ == "__main__":
    main()
