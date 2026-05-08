from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

from common import load_yaml_mapping, resolve_path
from gripper_hysteresis import (
    GripperHysteresisConfig,
    parse_int_sequence,
    parse_gripper_hysteresis_config,
)


@dataclass(frozen=True)
class PolicyRTCConfig:
    enabled: bool = False
    prefix_attention_schedule: str = "linear"
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10
    inference_delay_steps: int | None = None
    debug: bool = False
    debug_maxlen: int = 100


@dataclass(frozen=True)
class PolicyConfig:
    type: str
    path: Path | None
    device: str
    load_device: str | None
    task: str
    n_action_steps: int | None
    temporal_ensemble_coeff: float
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
    rtc: PolicyRTCConfig = field(default_factory=PolicyRTCConfig)


@dataclass(frozen=True)
class RuntimeConfig:
    control_hz: float


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool
    publish_hz: float
    attention_topic: str
    slot_memory_topic: str
    signature_topic: str
    overlay_alpha: float
    attention_query_step: int


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
    deadzone_linear_x: float = 0.0
    deadzone_linear_y: float = 0.0
    deadzone_angular_z: float = 0.0
    mutual_exclusion: "CommandMutualExclusionConfig" = field(
        default_factory=lambda: CommandMutualExclusionConfig()
    )


@dataclass(frozen=True)
class CommandMutualExclusionRuleConfig:
    source_index: int
    threshold: float
    target_indices: tuple[int, ...]
    mask_value: float = 0.0


@dataclass(frozen=True)
class CommandMutualExclusionConfig:
    enabled: bool = False
    rules: tuple[CommandMutualExclusionRuleConfig, ...] = ()


@dataclass(frozen=True)
class DeployConfig:
    path: Path
    policy: PolicyConfig
    runtime: RuntimeConfig
    gripper_hysteresis: GripperHysteresisConfig
    debug: DebugConfig
    image: ImageConfig
    ros: RosConfig
    command: CommandConfig


_POLICY_COMMON_KEYS = {
    "type",
    "path",
    "device",
    "load_device",
    "task",
    "instruction",
    "n_action_steps",
    "state_dim",
    "action_dim",
    "arm_dof",
    "base_action_dim",
    "state_key",
    "action_key",
    "image_keys",
}

_POLICY_STREAMING_SIGNATURE_KEYS = {
    "temporal_ensemble_coeff",
    "use_path_signature",
    "use_delta_signature",
    "signature_depth",
    "signature_dim",
    "signature_backend",
}

_POLICY_ALLOWED_KEYS_BY_TYPE = {
    "act": _POLICY_COMMON_KEYS | {"temporal_ensemble_coeff"},
    "streaming_act": _POLICY_COMMON_KEYS | _POLICY_STREAMING_SIGNATURE_KEYS,
    "pi0": _POLICY_COMMON_KEYS,
    "pi05": _POLICY_COMMON_KEYS | {"rtc"},
    "smolvla": _POLICY_COMMON_KEYS | {"rtc"},
}


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


def _parse_non_negative_float(
    data: dict[str, object],
    *,
    key: str,
    default: float = 0.0,
) -> float:
    raw_value = data.get(key, default)
    value = float(default if raw_value is None or raw_value == "" else raw_value)
    if value < 0.0:
        raise ValueError(f"`{key}` must be >= 0, got {value}.")
    return value


def _parse_finite_float(data: dict[str, object], *, key: str) -> float:
    if key not in data or data[key] in {None, ""}:
        raise ValueError(f"`{key}` is required.")
    value = float(data[key])
    if not math.isfinite(value):
        raise ValueError(f"`{key}` must be finite, got {value}.")
    return value


def _parse_non_negative_int(data: dict[str, object], *, key: str) -> int:
    if key not in data or data[key] in {None, ""}:
        raise ValueError(f"`{key}` is required.")
    value = int(data[key])
    if value < 0:
        raise ValueError(f"`{key}` must be >= 0, got {value}.")
    return value


def _parse_optional_non_negative_int(
    data: dict[str, object],
    *,
    key: str,
) -> int | None:
    if key not in data or data[key] in {None, "", "null"}:
        return None
    value = int(data[key])
    if value < 0:
        raise ValueError(f"`{key}` must be >= 0, got {value}.")
    return value


def parse_policy_rtc_config(
    raw: dict[str, object] | None,
    *,
    policy_type: str,
) -> PolicyRTCConfig:
    data = dict(raw or {})
    enabled = bool(data.get("enabled", False))
    if enabled and policy_type not in {"pi05", "smolvla"}:
        raise ValueError(
            f"`policy.rtc.enabled` is only supported for 'pi05' and 'smolvla', got {policy_type!r}."
        )

    execution_horizon = int(data.get("execution_horizon", 10))
    if execution_horizon <= 0:
        raise ValueError(
            f"`policy.rtc.execution_horizon` must be > 0, got {execution_horizon}."
        )
    max_guidance_weight = float(data.get("max_guidance_weight", 10.0))
    if max_guidance_weight <= 0.0 or not math.isfinite(max_guidance_weight):
        raise ValueError(
            "`policy.rtc.max_guidance_weight` must be finite and > 0, "
            f"got {max_guidance_weight}."
        )
    debug_maxlen = int(data.get("debug_maxlen", 100))
    if debug_maxlen <= 0:
        raise ValueError(f"`policy.rtc.debug_maxlen` must be > 0, got {debug_maxlen}.")

    schedule = str(data.get("prefix_attention_schedule", "linear")).strip().lower()
    if schedule not in {"linear", "exp", "zeros", "ones"}:
        raise ValueError(
            "`policy.rtc.prefix_attention_schedule` must be one of "
            "`linear`, `exp`, `zeros`, `ones`, got "
            f"{schedule!r}."
        )

    return PolicyRTCConfig(
        enabled=enabled,
        prefix_attention_schedule=schedule,
        max_guidance_weight=max_guidance_weight,
        execution_horizon=execution_horizon,
        inference_delay_steps=_parse_optional_non_negative_int(
            data,
            key="inference_delay_steps",
        ),
        debug=bool(data.get("debug", False)),
        debug_maxlen=debug_maxlen,
    )


def _validate_policy_config_keys(policy_type: str, policy_raw: dict[str, object]) -> None:
    allowed_keys = _POLICY_ALLOWED_KEYS_BY_TYPE.get(policy_type)
    if allowed_keys is None:
        raise ValueError(f"Unsupported deploy policy type: {policy_type!r}.")

    unknown_keys = sorted(str(key) for key in policy_raw if str(key) not in allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unsupported deploy policy keys for {policy_type!r}: "
            + ", ".join(unknown_keys)
        )


def parse_command_mutual_exclusion_config(
    raw: dict[str, object] | None,
) -> CommandMutualExclusionConfig:
    data = dict(raw or {})
    enabled = bool(data.get("enabled", False))
    rules_raw = data.get("rules", ())
    if rules_raw is None:
        rules_raw = ()
    if not isinstance(rules_raw, (list, tuple)):
        raise TypeError(
            "Expected `command.mutual_exclusion.rules` to be a list of mappings."
        )

    rules: list[CommandMutualExclusionRuleConfig] = []
    for rule_index, rule_raw in enumerate(rules_raw):
        if not isinstance(rule_raw, dict):
            raise TypeError(
                "Expected each `command.mutual_exclusion.rules` entry to be a mapping."
            )
        source_index = _parse_non_negative_int(rule_raw, key="source_index")
        target_indices = parse_int_sequence(
            rule_raw.get("target_indices"),
            key="command.mutual_exclusion.rules[].target_indices",
        )
        if not target_indices:
            raise ValueError(
                f"`command.mutual_exclusion.rules[{rule_index}].target_indices` "
                "must not be empty."
            )
        if any(target_index < 0 for target_index in target_indices):
            raise ValueError(
                f"`command.mutual_exclusion.rules[{rule_index}].target_indices` "
                "must contain non-negative indices."
            )
        mask_value_raw = rule_raw.get("mask_value", 0.0)
        mask_value = 0.0 if mask_value_raw in {None, ""} else float(mask_value_raw)
        if not math.isfinite(mask_value):
            raise ValueError(
                f"`command.mutual_exclusion.rules[{rule_index}].mask_value` must be finite."
            )
        rules.append(
            CommandMutualExclusionRuleConfig(
                source_index=source_index,
                threshold=_parse_finite_float(rule_raw, key="threshold"),
                target_indices=tuple(target_indices),
                mask_value=mask_value,
            )
        )

    return CommandMutualExclusionConfig(enabled=enabled, rules=tuple(rules))


def _validate_command_mutual_exclusion_config(
    config: CommandMutualExclusionConfig,
    *,
    action_dim: int,
) -> None:
    if not config.enabled:
        return
    for rule_index, rule in enumerate(config.rules):
        if rule.source_index >= action_dim:
            raise ValueError(
                f"`command.mutual_exclusion.rules[{rule_index}].source_index` "
                f"must be < action_dim ({action_dim}), got {rule.source_index}."
            )
        for target_offset, target_index in enumerate(rule.target_indices):
            if target_index >= action_dim:
                raise ValueError(
                    f"`command.mutual_exclusion.rules[{rule_index}].target_indices[{target_offset}]` "
                    f"must be < action_dim ({action_dim}), got {target_index}."
                )


def load_deploy_config(config_path: str | Path) -> DeployConfig:
    path = Path(config_path).expanduser().resolve()
    raw = load_yaml_mapping(path)

    policy_raw = _as_mapping(raw, "policy")
    policy_type = str(policy_raw.get("type", "act"))
    _validate_policy_config_keys(policy_type, policy_raw)
    rtc = parse_policy_rtc_config(_as_mapping(policy_raw, "rtc"), policy_type=policy_type)
    use_streaming_signatures = policy_type == "streaming_act"
    runtime_raw = _as_mapping(raw, "runtime")
    debug_raw = _as_mapping(raw, "debug")
    image_raw = _as_mapping(raw, "image")
    ros_raw = _as_mapping(raw, "ros")
    topics_raw = _as_mapping(ros_raw, "topics")
    command_raw = _as_mapping(raw, "command")
    gripper_hysteresis_raw = _as_mapping(raw, "gripper_hysteresis")

    policy = PolicyConfig(
        type=policy_type,
        path=resolve_path(policy_raw.get("path"), config_path=path, must_exist=False),
        device=str(policy_raw.get("device", "cuda")),
        load_device=(
            None
            if policy_raw.get("load_device") in {None, "", "null"}
            else str(policy_raw.get("load_device"))
        ),
        task=str(policy_raw.get("task", policy_raw.get("instruction", ""))).strip(),
        n_action_steps=(
            None if policy_raw.get("n_action_steps") is None else int(policy_raw["n_action_steps"])
        ),
        temporal_ensemble_coeff=float(policy_raw.get("temporal_ensemble_coeff", 0.0)),
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
        rtc=rtc,
    )

    runtime = RuntimeConfig(
        control_hz=float(runtime_raw.get("control_hz", 20.0)),
    )

    gripper_hysteresis = parse_gripper_hysteresis_config(
        gripper_hysteresis_raw,
        action_dim=policy.action_dim,
        base_action_dim=policy.base_action_dim,
        arm_dof=policy.arm_dof,
    )

    debug = DebugConfig(
        enabled=bool(debug_raw.get("enabled", False)),
        publish_hz=float(debug_raw.get("publish_hz", 5.0)),
        attention_topic=str(debug_raw.get("attention_topic", "/deploy/debug/attention")),
        slot_memory_topic=str(debug_raw.get("slot_memory_topic", "/deploy/debug/slot_memory")),
        signature_topic=str(debug_raw.get("signature_topic", "/deploy/debug/signature")),
        overlay_alpha=float(debug_raw.get("overlay_alpha", 0.45)),
        attention_query_step=int(debug_raw.get("attention_query_step", 0)),
    )
    if debug.publish_hz < 0.0:
        raise ValueError(f"`debug.publish_hz` must be >= 0, got {debug.publish_hz}.")
    if not (0.0 <= debug.overlay_alpha <= 1.0):
        raise ValueError(
            f"`debug.overlay_alpha` must be in [0, 1], got {debug.overlay_alpha}."
        )
    if debug.attention_query_step < 0:
        raise ValueError(
            f"`debug.attention_query_step` must be >= 0, got {debug.attention_query_step}."
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

    command_mutual_exclusion = parse_command_mutual_exclusion_config(
        _as_mapping(command_raw, "mutual_exclusion")
    )
    _validate_command_mutual_exclusion_config(
        command_mutual_exclusion,
        action_dim=int(policy.action_dim),
    )

    command = CommandConfig(
        publish_base=bool(command_raw.get("publish_base", True)),
        publish_arms=bool(command_raw.get("publish_arms", True)),
        max_linear_x=float(command_raw.get("max_linear_x", 0.3)),
        max_linear_y=float(command_raw.get("max_linear_y", 0.3)),
        max_angular_z=float(command_raw.get("max_angular_z", 0.5)),
        deadzone_linear_x=_parse_non_negative_float(
            command_raw,
            key="deadzone_linear_x",
        ),
        deadzone_linear_y=_parse_non_negative_float(
            command_raw,
            key="deadzone_linear_y",
        ),
        deadzone_angular_z=_parse_non_negative_float(
            command_raw,
            key="deadzone_angular_z",
        ),
        mutual_exclusion=command_mutual_exclusion,
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
        gripper_hysteresis=gripper_hysteresis,
        debug=debug,
        image=image,
        ros=ros,
        command=command,
    )
