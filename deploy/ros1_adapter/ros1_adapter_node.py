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

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _enqueue(self, packet: SensorPacket) -> None:
        outgoing = OutgoingPacket(packet=packet)
        try:
            self._queue.put_nowait(outgoing)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(outgoing)

    def _register_subscribers(self) -> None:
        for image_topic in self.config.image_topics:
            msg_type = self._sensor_msgs.CompressedImage if image_topic.compressed else self._sensor_msgs.Image
            self._rospy.Subscriber(
                image_topic.topic,
                msg_type,
                self._make_image_callback(image_topic),
                queue_size=1,
                buff_size=2**24,
            )

        self._rospy.Subscriber(
            self.config.left_arm_state_topic.topic,
            self._sensor_msgs.JointState,
            self._make_joint_state_callback(self.config.left_arm_state_topic),
            queue_size=10,
        )
        self._rospy.Subscriber(
            self.config.right_arm_state_topic.topic,
            self._sensor_msgs.JointState,
            self._make_joint_state_callback(self.config.right_arm_state_topic),
            queue_size=10,
        )
        self._rospy.Subscriber(
            self.config.odom_topic.topic,
            self._nav_msgs.Odometry,
            self._make_odom_callback(self.config.odom_topic),
            queue_size=20,
        )

    def _make_image_callback(self, config: ImageTopicConfig):
        def _callback(msg: Any) -> None:
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

        return _callback

    def _make_joint_state_callback(self, config: JointStateTopicConfig):
        def _callback(msg: Any) -> None:
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

        return _callback

    def _make_odom_callback(self, config: OdomTopicConfig):
        def _callback(msg: Any) -> None:
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

        return _callback

    def _drain_sensor_queue(self) -> None:
        while True:
            try:
                outgoing = self._queue.get_nowait()
            except queue.Empty:
                break
            self._sensor_socket.send_multipart(outgoing.packet.to_multipart())

    def _handle_robot_command(self, packet: RobotCommandPacket) -> None:
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

    def spin(self) -> None:
        rate = self._rospy.Rate(self.config.command_poll_hz)
        try:
            while not self._rospy.is_shutdown():
                self._drain_sensor_queue()
                self._drain_command_socket()
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

