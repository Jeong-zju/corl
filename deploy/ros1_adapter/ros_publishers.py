from __future__ import annotations

import numpy as np

from deploy.ros1_adapter.ros_topics import ArmCommandConfig, BaseCommandConfig


class ArmCommandPublisher:
    def __init__(self, config: ArmCommandConfig, *, queue_size: int = 1) -> None:
        import rospy
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory

        self._rospy = rospy
        self._joint_state_cls = JointState
        self._joint_trajectory_cls = JointTrajectory
        self._joint_names = list(config.joint_names)
        self._message_type = str(config.message_type)
        self._duration_s = float(config.duration_s)
        if self._message_type == "joint_state":
            publisher_type = JointState
        elif self._message_type == "joint_trajectory":
            publisher_type = JointTrajectory
        else:
            raise ValueError(
                f"Unsupported arm command message_type={config.message_type!r}. "
                "Expected `joint_state` or `joint_trajectory`."
            )
        self._publisher = rospy.Publisher(config.topic, publisher_type, queue_size=queue_size)

    def publish_positions(self, positions: np.ndarray) -> None:
        command = [float(value) for value in np.asarray(positions, dtype=np.float32).reshape(-1)]
        if self._message_type == "joint_state":
            msg = self._joint_state_cls()
            msg.header.stamp = self._rospy.Time.now()
            msg.name = list(self._joint_names)
            msg.position = command
            self._publisher.publish(msg)
            return

        from trajectory_msgs.msg import JointTrajectoryPoint

        msg = self._joint_trajectory_cls()
        msg.header.stamp = self._rospy.Time.now()
        msg.joint_names = list(self._joint_names)
        point = JointTrajectoryPoint()
        point.positions = command
        point.time_from_start = self._rospy.Duration.from_sec(self._duration_s)
        msg.points = [point]
        self._publisher.publish(msg)


class BaseTwistPublisher:
    def __init__(self, config: BaseCommandConfig, *, queue_size: int = 1) -> None:
        import rospy
        from geometry_msgs.msg import Twist

        self._twist_cls = Twist
        self._publisher = rospy.Publisher(config.topic, Twist, queue_size=queue_size)

    def publish_twist(self, base_command: np.ndarray) -> None:
        command = np.asarray(base_command, dtype=np.float32).reshape(-1)
        msg = self._twist_cls()
        if command.shape[0] > 0:
            msg.linear.x = float(command[0])
        if command.shape[0] > 1:
            msg.linear.y = float(command[1])
        if command.shape[0] > 2:
            msg.angular.z = float(command[2])
        self._publisher.publish(msg)

    def publish_zero(self) -> None:
        self.publish_twist(np.zeros((3,), dtype=np.float32))
