from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deploy.utils import load_mapping_file, nested_mapping_get


@dataclass(slots=True)
class ImageTopicConfig:
    stream: str
    topic: str
    compressed: bool = True


@dataclass(slots=True)
class JointStateTopicConfig:
    stream: str
    topic: str
    joint_names: tuple[str, ...] = ()


@dataclass(slots=True)
class OdomTopicConfig:
    stream: str
    topic: str


@dataclass(slots=True)
class ArmCommandConfig:
    topic: str
    joint_names: tuple[str, ...]
    duration_s: float = 0.2


@dataclass(slots=True)
class BaseCommandConfig:
    topic: str
    enabled: bool = False


@dataclass(slots=True)
class Ros1AdapterConfig:
    node_name: str
    sensor_endpoint: str
    command_endpoint: str
    command_poll_hz: float
    outgoing_queue_size: int
    image_topics: tuple[ImageTopicConfig, ...]
    left_arm_state_topic: JointStateTopicConfig
    right_arm_state_topic: JointStateTopicConfig
    odom_topic: OdomTopicConfig
    left_arm_command: ArmCommandConfig
    right_arm_command: ArmCommandConfig
    base_command: BaseCommandConfig
    publish_zero_base_on_hold: bool

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "Ros1AdapterConfig":
        root = nested_mapping_get(mapping, "ros1", default=mapping)
        image_topics_raw = root.get("image_topics", [])
        if not isinstance(image_topics_raw, list):
            raise ValueError("`ros1.image_topics` must be a list.")
        image_topics = tuple(
            ImageTopicConfig(
                stream=str(item["stream"]),
                topic=str(item["topic"]),
                compressed=bool(item.get("compressed", True)),
            )
            for item in image_topics_raw
        )
        left_state_raw = nested_mapping_get(root, "left_arm_state_topic")
        right_state_raw = nested_mapping_get(root, "right_arm_state_topic")
        odom_raw = nested_mapping_get(root, "odom_topic")
        left_cmd_raw = nested_mapping_get(root, "left_arm_command")
        right_cmd_raw = nested_mapping_get(root, "right_arm_command")
        base_cmd_raw = nested_mapping_get(root, "base_command", default={})
        return cls(
            node_name=str(root.get("node_name", "policy_bridge_ros1_adapter")),
            sensor_endpoint=str(root.get("sensor_endpoint", "tcp://127.0.0.1:5556")),
            command_endpoint=str(root.get("command_endpoint", "tcp://127.0.0.1:5557")),
            command_poll_hz=float(root.get("command_poll_hz", 100.0)),
            outgoing_queue_size=int(root.get("outgoing_queue_size", 256)),
            image_topics=image_topics,
            left_arm_state_topic=JointStateTopicConfig(
                stream=str(left_state_raw["stream"]),
                topic=str(left_state_raw["topic"]),
                joint_names=tuple(str(name) for name in left_state_raw.get("joint_names", [])),
            ),
            right_arm_state_topic=JointStateTopicConfig(
                stream=str(right_state_raw["stream"]),
                topic=str(right_state_raw["topic"]),
                joint_names=tuple(str(name) for name in right_state_raw.get("joint_names", [])),
            ),
            odom_topic=OdomTopicConfig(
                stream=str(odom_raw["stream"]),
                topic=str(odom_raw["topic"]),
            ),
            left_arm_command=ArmCommandConfig(
                topic=str(left_cmd_raw["topic"]),
                joint_names=tuple(str(name) for name in left_cmd_raw["joint_names"]),
                duration_s=float(left_cmd_raw.get("duration_s", 0.2)),
            ),
            right_arm_command=ArmCommandConfig(
                topic=str(right_cmd_raw["topic"]),
                joint_names=tuple(str(name) for name in right_cmd_raw["joint_names"]),
                duration_s=float(right_cmd_raw.get("duration_s", 0.2)),
            ),
            base_command=BaseCommandConfig(
                topic=str(base_cmd_raw.get("topic", "/cmd_vel")),
                enabled=bool(base_cmd_raw.get("enabled", False)),
            ),
            publish_zero_base_on_hold=bool(root.get("publish_zero_base_on_hold", True)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Ros1AdapterConfig":
        return cls.from_mapping(load_mapping_file(path))

