from __future__ import annotations

from config import DeployConfig


class CommandPublishers:
    def __init__(self, config: DeployConfig) -> None:
        import rospy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import JointState

        self._rospy = rospy
        self._Twist = Twist
        self._JointState = JointState
        self._config = config
        self._base_pub = rospy.Publisher(
            config.ros.topics.cmd_vel,
            Twist,
            queue_size=config.ros.queue_size,
        )
        self._left_pub = rospy.Publisher(
            config.ros.topics.cmd_joint_left,
            JointState,
            queue_size=config.ros.queue_size,
        )
        self._right_pub = rospy.Publisher(
            config.ros.topics.cmd_joint_right,
            JointState,
            queue_size=config.ros.queue_size,
        )

    def _make_joint_state(self, names: list[str], positions) -> object:
        msg = self._JointState()
        msg.header.stamp = self._rospy.Time.now()
        msg.name = list(names)
        msg.position = [float(value) for value in positions]
        return msg

    def publish(self, command_packet: dict[str, object]) -> None:
        if command_packet.get("publish_base", False):
            twist = self._Twist()
            values = command_packet.get("base_twist")
            if values is not None:
                twist.linear.x = float(values[0]) if len(values) > 0 else 0.0
                twist.linear.y = float(values[1]) if len(values) > 1 else 0.0
                twist.angular.z = float(values[2]) if len(values) > 2 else 0.0
            self._base_pub.publish(twist)

        if command_packet.get("publish_arms", False):
            left_msg = self._make_joint_state(
                self._config.ros.joint_names_left.name,
                command_packet.get("left_joint_positions", []),
            )
            right_msg = self._make_joint_state(
                self._config.ros.joint_names_right.name,
                command_packet.get("right_joint_positions", []),
            )
            self._left_pub.publish(left_msg)
            self._right_pub.publish(right_msg)
