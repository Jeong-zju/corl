from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
import sys

import cv2
import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from bridge.protocol import build_command_packet, build_hold_action_from_state
from bridge.signature_runtime import OnlineSignatureRuntime
from bridge.sync import (
    SOURCE_IMAGE_LEFT,
    SOURCE_IMAGE_RIGHT,
    SOURCE_IMAGE_TOP,
    SOURCE_JOINT_LEFT,
    SOURCE_JOINT_RIGHT,
    SOURCE_ODOM,
    SensorCache,
    TimedSample,
)
from config import DeployConfig, load_deploy_config
from policy_runtime.loader import PolicyRuntime
from ros1_adapter.ros_publishers import CommandPublishers
from ros1_adapter.ros_topics import ALL_REQUIRED_SOURCES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-process ROS1 deployment node for ACT/Streaming ACT policies."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the deploy YAML config.",
    )
    return parser.parse_args(argv)


def ros_stamp_to_ns(stamp) -> int:
    if stamp is None:
        return time.monotonic_ns()
    if hasattr(stamp, "to_nsec"):
        value = int(stamp.to_nsec())
        if value > 0:
            return value
    secs = getattr(stamp, "secs", 0)
    nsecs = getattr(stamp, "nsecs", 0)
    value = int(secs) * 1_000_000_000 + int(nsecs)
    return value if value > 0 else time.monotonic_ns()


class DeployRosNode:
    def __init__(self, config: DeployConfig) -> None:
        import rospy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CompressedImage, JointState

        self.rospy = rospy
        self.config = config
        self.cache = SensorCache()
        self.signature_runtime = OnlineSignatureRuntime(config.policy)
        self.policy_runtime = PolicyRuntime(config)
        self.policy_runtime.load()
        self.publishers = CommandPublishers(config)
        self.lock = threading.Lock()
        self.seq = 0
        self.pending_reset = True

        rospy.init_node(config.ros.node_name, anonymous=False)

        self._subscribers = [
            rospy.Subscriber(
                config.ros.topics.image_left,
                CompressedImage,
                self._on_image_left,
                queue_size=config.ros.queue_size,
                buff_size=2**24,
            ),
            rospy.Subscriber(
                config.ros.topics.image_right,
                CompressedImage,
                self._on_image_right,
                queue_size=config.ros.queue_size,
                buff_size=2**24,
            ),
            rospy.Subscriber(
                config.ros.topics.image_top,
                CompressedImage,
                self._on_image_top,
                queue_size=config.ros.queue_size,
                buff_size=2**24,
            ),
            rospy.Subscriber(
                config.ros.topics.joint_state_left,
                JointState,
                self._on_joint_left,
                queue_size=config.ros.queue_size,
            ),
            rospy.Subscriber(
                config.ros.topics.joint_state_right,
                JointState,
                self._on_joint_right,
                queue_size=config.ros.queue_size,
            ),
            rospy.Subscriber(
                config.ros.topics.odom,
                Odometry,
                self._on_odom,
                queue_size=config.ros.queue_size,
            ),
        ]
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / float(config.runtime.control_hz)),
            self._control_step,
        )

    def _topic_name_by_source(self, source: str) -> str:
        topic_map = {
            SOURCE_IMAGE_LEFT: self.config.ros.topics.image_left,
            SOURCE_IMAGE_RIGHT: self.config.ros.topics.image_right,
            SOURCE_IMAGE_TOP: self.config.ros.topics.image_top,
            SOURCE_JOINT_LEFT: self.config.ros.topics.joint_state_left,
            SOURCE_JOINT_RIGHT: self.config.ros.topics.joint_state_right,
            SOURCE_ODOM: self.config.ros.topics.odom,
        }
        return topic_map.get(source, source)

    def _missing_required_topics_locked(self) -> list[str]:
        missing_sources = [
            source
            for source in ALL_REQUIRED_SOURCES
            if self.cache.get(source) is None
        ]
        return [self._topic_name_by_source(source) for source in missing_sources]

    def _store_sample(self, source: str, stamp_ns: int, value: np.ndarray) -> None:
        with self.lock:
            self.cache.update(
                source,
                TimedSample(
                    stamp_ns=int(stamp_ns),
                    received_ns=time.monotonic_ns(),
                    value=np.asarray(value),
                ),
            )

    def _decode_image(self, msg) -> np.ndarray | None:
        encoded_array = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
        if frame is None:
            self.rospy.logwarn_throttle(2.0, "Failed to decode compressed image.")
            return None
        if frame.shape[1] != self.config.image.width or frame.shape[0] != self.config.image.height:
            frame = cv2.resize(
                frame,
                (self.config.image.width, self.config.image.height),
                interpolation=cv2.INTER_LINEAR,
            )
        if self.config.image.color_order == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(frame)

    def _extract_joint_positions(self, msg) -> np.ndarray:
        positions = list(msg.position)
        dof = self.config.policy.arm_dof
        if len(positions) >= dof:
            positions = positions[:dof]
        else:
            positions = positions + [0.0] * (dof - len(positions))
        return np.asarray(positions, dtype=np.float32)

    def _extract_base_velocity(self, msg) -> np.ndarray:
        twist = msg.twist.twist
        return np.asarray(
            [twist.linear.x, twist.linear.y, twist.angular.z],
            dtype=np.float32,
        )

    def _on_image_left(self, msg) -> None:
        frame = self._decode_image(msg)
        if frame is not None:
            self._store_sample(SOURCE_IMAGE_LEFT, ros_stamp_to_ns(msg.header.stamp), frame)

    def _on_image_right(self, msg) -> None:
        frame = self._decode_image(msg)
        if frame is not None:
            self._store_sample(SOURCE_IMAGE_RIGHT, ros_stamp_to_ns(msg.header.stamp), frame)

    def _on_image_top(self, msg) -> None:
        frame = self._decode_image(msg)
        if frame is not None:
            self._store_sample(SOURCE_IMAGE_TOP, ros_stamp_to_ns(msg.header.stamp), frame)

    def _on_joint_left(self, msg) -> None:
        self._store_sample(
            SOURCE_JOINT_LEFT,
            ros_stamp_to_ns(msg.header.stamp),
            self._extract_joint_positions(msg),
        )

    def _on_joint_right(self, msg) -> None:
        self._store_sample(
            SOURCE_JOINT_RIGHT,
            ros_stamp_to_ns(msg.header.stamp),
            self._extract_joint_positions(msg),
        )

    def _on_odom(self, msg) -> None:
        self._store_sample(
            SOURCE_ODOM,
            ros_stamp_to_ns(msg.header.stamp),
            self._extract_base_velocity(msg),
        )

    def _reset_runtime(self) -> None:
        self.signature_runtime.reset()
        self.policy_runtime.reset()
        self.pending_reset = True

    def _build_observation_locked(self) -> tuple[dict[str, object] | None, str]:
        if not self.cache.all_sources_present(list(ALL_REQUIRED_SOURCES)):
            missing_topics = self._missing_required_topics_locked()
            return None, "Waiting for ROS topics: " + ", ".join(missing_topics)

        state = self.cache.latest_state_vector(arm_dof=self.config.policy.arm_dof)
        if state is None:
            return None, "Missing low-dimensional state for inference."

        images = {
            self.config.policy.image_keys["left"]: self.cache.get(SOURCE_IMAGE_LEFT).value,
            self.config.policy.image_keys["right"]: self.cache.get(SOURCE_IMAGE_RIGHT).value,
            self.config.policy.image_keys["top"]: self.cache.get(SOURCE_IMAGE_TOP).value,
        }
        path_signature, delta_signature = self.signature_runtime.update(state)

        return (
            {
                "seq": int(self.seq),
                "stamp_ns": max(
                    self.cache.get(SOURCE_IMAGE_LEFT).stamp_ns,
                    self.cache.get(SOURCE_IMAGE_RIGHT).stamp_ns,
                    self.cache.get(SOURCE_IMAGE_TOP).stamp_ns,
                    self.cache.get(SOURCE_JOINT_LEFT).stamp_ns,
                    self.cache.get(SOURCE_JOINT_RIGHT).stamp_ns,
                    self.cache.get(SOURCE_ODOM).stamp_ns,
                ),
                "reset": bool(self.pending_reset),
                "state": state.astype(np.float32, copy=False),
                "images": images,
                "path_signature": path_signature,
                "delta_signature": delta_signature,
            },
            "",
        )

    def _publish_hold_locked(self, reason: str) -> None:
        state = self.cache.latest_state_vector(arm_dof=self.config.policy.arm_dof)
        hold_action = build_hold_action_from_state(
            state,
            action_dim=self.config.policy.action_dim,
            base_action_dim=self.config.policy.base_action_dim,
        )
        command_packet = build_command_packet(
            config=self.config,
            seq=self.seq,
            obs_seq=self.seq,
            action=hold_action,
            status="hold",
            message=reason,
            runtime_ms=None,
        )
        self.publishers.publish(command_packet)

    def _control_step(self, _event) -> None:
        with self.lock:
            observation, error = self._build_observation_locked()
            if observation is None:
                self.rospy.logwarn_throttle(2.0, error)
                self._publish_hold_locked(error)
                return

            try:
                result = self.policy_runtime.infer(observation)
            except Exception as exc:
                self.rospy.logerr_throttle(2.0, f"Policy inference failed: {exc}")
                self._reset_runtime()
                self._publish_hold_locked(f"Policy inference failed: {exc}")
                return

            action = np.asarray(result["action"], dtype=np.float32).reshape(-1)
            command_packet = build_command_packet(
                config=self.config,
                seq=self.seq,
                obs_seq=int(observation["seq"]),
                action=action,
                status="ok",
                message="policy_eval",
                runtime_ms=float(result["runtime_ms"]),
            )
            self.publishers.publish(command_packet)
            self.pending_reset = False
            self.seq += 1

    def spin(self) -> None:
        self.rospy.loginfo(
            "Deploy node started: policy=%s path=%s hz=%.2f",
            self.config.policy.type,
            self.config.policy.path,
            self.config.runtime.control_hz,
        )
        self.rospy.spin()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_deploy_config(args.config)
    node = DeployRosNode(config)
    node.spin()


if __name__ == "__main__":
    main()
