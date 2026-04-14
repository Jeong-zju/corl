from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common import load_yaml_mapping, resolve_path


@dataclass(frozen=True)
class PolicyConfig:
    type: str
    path: Path | None
    device: str
    load_device: str | None
    n_action_steps: int | None
    state_dim: int
    action_dim: int
    arm_dof: int
    base_action_dim: int
    state_key: str
    action_key: str
    image_keys: dict[str, str]
    use_path_signature: bool
    use_delta_signature: bool
    signature_depth: int
    signature_dim: int | None
    signature_backend: str


@dataclass(frozen=True)
class RuntimeConfig:
    control_hz: float


@dataclass(frozen=True)
class ImageConfig:
    width: int
    height: int
    color_order: str


@dataclass(frozen=True)
class TopicConfig:
    image_left: str
    image_right: str
    image_top: str
    joint_state_left: str
    joint_state_right: str
    odom: str
    cmd_vel: str
    cmd_joint_left: str
    cmd_joint_right: str


@dataclass(frozen=True)
class JointNameConfig:
    name: list[str]


@dataclass(frozen=True)
class RosConfig:
    node_name: str
    queue_size: int
    topics: TopicConfig
    joint_names_left: JointNameConfig
    joint_names_right: JointNameConfig


@dataclass(frozen=True)
class CommandConfig:
    publish_base: bool
    publish_arms: bool
    max_linear_x: float
    max_linear_y: float
    max_angular_z: float


@dataclass(frozen=True)
class DeployConfig:
    path: Path
    policy: PolicyConfig
    runtime: RuntimeConfig
    image: ImageConfig
    ros: RosConfig
    command: CommandConfig


def _as_mapping(data: dict, key: str) -> dict:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping for `{key}`, got {type(value).__name__}.")
    return value


def _as_str_map(data: dict[str, object], *, key: str) -> dict[str, str]:
    raw = _as_mapping(data, key)
    return {str(name): str(value) for name, value in raw.items()}


def _parse_joint_name_config(value: object, *, key: str) -> JointNameConfig:
    if isinstance(value, dict):
        names = value.get("name", [])
    elif isinstance(value, list):
        names = value
    else:
        raise TypeError(
            f"Expected `{key}` to be a mapping or list, got {type(value).__name__}."
        )
    return JointNameConfig(name=[str(item) for item in list(names)])


def load_deploy_config(config_path: str | Path) -> DeployConfig:
    path = Path(config_path).expanduser().resolve()
    raw = load_yaml_mapping(path)

    policy_raw = _as_mapping(raw, "policy")
    policy_type = str(policy_raw.get("type", "act"))
    use_streaming_signatures = policy_type == "streaming_act"
    runtime_raw = _as_mapping(raw, "runtime")
    image_raw = _as_mapping(raw, "image")
    ros_raw = _as_mapping(raw, "ros")
    topics_raw = _as_mapping(ros_raw, "topics")
    command_raw = _as_mapping(raw, "command")

    policy = PolicyConfig(
        type=policy_type,
        path=resolve_path(policy_raw.get("path"), config_path=path, must_exist=False),
        device=str(policy_raw.get("device", "cuda")),
        load_device=(
            None
            if policy_raw.get("load_device") in {None, "", "null"}
            else str(policy_raw.get("load_device"))
        ),
        n_action_steps=(
            None if policy_raw.get("n_action_steps") is None else int(policy_raw["n_action_steps"])
        ),
        state_dim=int(policy_raw.get("state_dim", 17)),
        action_dim=int(policy_raw.get("action_dim", 17)),
        arm_dof=int(policy_raw.get("arm_dof", 7)),
        base_action_dim=int(policy_raw.get("base_action_dim", 3)),
        state_key=str(policy_raw.get("state_key", "observation.state")),
        action_key=str(policy_raw.get("action_key", "action")),
        image_keys=_as_str_map(
            policy_raw,
            key="image_keys",
        )
        or {
            "left": "observation.images.realsense_left",
            "right": "observation.images.realsense_right",
            "top": "observation.images.realsense_top",
        },
        use_path_signature=(
            bool(policy_raw.get("use_path_signature", False))
            if use_streaming_signatures
            else False
        ),
        use_delta_signature=(
            bool(policy_raw.get("use_delta_signature", False))
            if use_streaming_signatures
            else False
        ),
        signature_depth=(
            int(policy_raw.get("signature_depth", 3))
            if use_streaming_signatures
            else 0
        ),
        signature_dim=(
            None
            if (not use_streaming_signatures or policy_raw.get("signature_dim") is None)
            else int(policy_raw["signature_dim"])
        ),
        signature_backend=(
            str(policy_raw.get("signature_backend", "auto"))
            if use_streaming_signatures
            else "disabled"
        ),
    )

    runtime = RuntimeConfig(
        control_hz=float(runtime_raw.get("control_hz", 20.0)),
    )

    image = ImageConfig(
        width=int(image_raw.get("width", 224)),
        height=int(image_raw.get("height", 224)),
        color_order=str(image_raw.get("color_order", "rgb")).lower(),
    )
    if image.color_order not in {"rgb", "bgr"}:
        raise ValueError(
            f"`image.color_order` must be `rgb` or `bgr`, got {image.color_order!r}."
        )

    ros = RosConfig(
        node_name=str(ros_raw.get("node_name", "deploy_policy_bridge")),
        queue_size=int(ros_raw.get("queue_size", 1)),
        topics=TopicConfig(
            image_left=str(
                topics_raw.get("image_left", "/realsense_left/color/image_raw/compressed")
            ),
            image_right=str(
                topics_raw.get("image_right", "/realsense_right/color/image_raw/compressed")
            ),
            image_top=str(
                topics_raw.get("image_top", "/realsense_top/color/image_raw/compressed")
            ),
            joint_state_left=str(
                topics_raw.get("joint_state_left", "/robot/arm_left/joint_states_single")
            ),
            joint_state_right=str(
                topics_raw.get("joint_state_right", "/robot/arm_right/joint_states_single")
            ),
            odom=str(topics_raw.get("odom", "/ranger_base_node/odom")),
            cmd_vel=str(topics_raw.get("cmd_vel", "/cmd_vel")),
            cmd_joint_left=str(
                topics_raw.get("cmd_joint_left", "/deploy/arm_left/joint_states")
            ),
            cmd_joint_right=str(
                topics_raw.get("cmd_joint_right", "/deploy/arm_right/joint_states")
            ),
        ),
        joint_names_left=_parse_joint_name_config(
            ros_raw.get("joint_names_left", []),
            key="ros.joint_names_left",
        ),
        joint_names_right=_parse_joint_name_config(
            ros_raw.get("joint_names_right", []),
            key="ros.joint_names_right",
        ),
    )

    command = CommandConfig(
        publish_base=bool(command_raw.get("publish_base", True)),
        publish_arms=bool(command_raw.get("publish_arms", True)),
        max_linear_x=float(command_raw.get("max_linear_x", 0.3)),
        max_linear_y=float(command_raw.get("max_linear_y", 0.3)),
        max_angular_z=float(command_raw.get("max_angular_z", 0.5)),
    )

    missing_image_keys = {"left", "right", "top"} - set(policy.image_keys)
    if missing_image_keys:
        raise ValueError(
            "Missing required `policy.image_keys` entries: "
            + ", ".join(sorted(missing_image_keys))
        )

    return DeployConfig(
        path=path,
        policy=policy,
        runtime=runtime,
        image=image,
        ros=ros,
        command=command,
    )
