from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import DeployConfig


@dataclass(frozen=True)
class SplitAction:
    base: np.ndarray
    left: np.ndarray
    right: np.ndarray


def split_action_vector(
    action: np.ndarray,
    *,
    base_action_dim: int,
    arm_dof: int,
) -> SplitAction:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    expected_dim = base_action_dim + arm_dof * 2
    if vector.shape[0] != expected_dim:
        raise ValueError(
            f"Expected action dim {expected_dim}, got {vector.shape[0]}."
        )

    split = base_action_dim + arm_dof
    return SplitAction(
        base=vector[:base_action_dim].astype(np.float32, copy=False),
        left=vector[base_action_dim:split].astype(np.float32, copy=False),
        right=vector[split:].astype(np.float32, copy=False),
    )


def _apply_deadzone(value: float, deadzone: float) -> float:
    if deadzone <= 0.0:
        return float(value)
    return 0.0 if abs(float(value)) < deadzone else float(value)


def _validate_action_index(index: int, *, size: int, role: str, rule_index: int) -> None:
    if index < 0 or index >= size:
        raise ValueError(
            f"Mutual exclusion rule {rule_index} has {role} index {index}, "
            f"but action dim is {size}."
        )


def apply_mutual_exclusion_rules(
    action: np.ndarray,
    config: DeployConfig,
) -> np.ndarray:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    command_cfg = getattr(config, "command", None)
    mutual_exclusion = getattr(command_cfg, "mutual_exclusion", None)
    if command_cfg is None or mutual_exclusion is None:
        return vector.copy()
    if not getattr(mutual_exclusion, "enabled", False):
        return vector.copy()

    original = vector.copy()
    filtered = vector.copy()
    size = original.shape[0]
    rules = tuple(getattr(mutual_exclusion, "rules", ()) or ())
    for rule_index, rule in enumerate(rules):
        source_index = int(getattr(rule, "source_index"))
        _validate_action_index(source_index, size=size, role="source", rule_index=rule_index)
        if original[source_index] <= float(getattr(rule, "threshold")):
            continue
        mask_value = float(getattr(rule, "mask_value", 0.0))
        target_indices = tuple(getattr(rule, "target_indices", ()) or ())
        for target_index in target_indices:
            target_index = int(target_index)
            _validate_action_index(
                target_index,
                size=size,
                role="target",
                rule_index=rule_index,
            )
            filtered[target_index] = mask_value
    return filtered


def build_hold_action_from_state(
    state: np.ndarray | None,
    *,
    action_dim: int,
    base_action_dim: int,
) -> np.ndarray:
    if state is None:
        return np.zeros((action_dim,), dtype=np.float32)

    state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
    hold = np.zeros((action_dim,), dtype=np.float32)
    copy_n = min(action_dim, state_vec.shape[0])
    hold[:copy_n] = state_vec[:copy_n]
    hold[:base_action_dim] = 0.0
    return hold


def clamp_base_action(base: np.ndarray, config: DeployConfig) -> np.ndarray:
    limited = np.asarray(base, dtype=np.float32).copy()
    if limited.shape[0] >= 1:
        limited[0] = float(
            np.clip(
                limited[0],
                -config.command.max_linear_x,
                config.command.max_linear_x,
            )
        )
        limited[0] = _apply_deadzone(
            limited[0],
            config.command.deadzone_linear_x,
        )
    if limited.shape[0] >= 2:
        limited[1] = float(
            np.clip(
                limited[1],
                -config.command.max_linear_y,
                config.command.max_linear_y,
            )
        )
        limited[1] = _apply_deadzone(
            limited[1],
            config.command.deadzone_linear_y,
        )
    if limited.shape[0] >= 3:
        limited[2] = float(
            np.clip(
                limited[2],
                -config.command.max_angular_z,
                config.command.max_angular_z,
            )
        )
        limited[2] = _apply_deadzone(
            limited[2],
            config.command.deadzone_angular_z,
        )
    return limited


def build_command_packet(
    *,
    config: DeployConfig,
    seq: int,
    obs_seq: int,
    action: np.ndarray,
    status: str,
    message: str,
    runtime_ms: float | None,
) -> dict[str, object]:
    filtered_action = apply_mutual_exclusion_rules(action, config)
    split = split_action_vector(
        filtered_action,
        base_action_dim=config.policy.base_action_dim,
        arm_dof=config.policy.arm_dof,
    )
    base = clamp_base_action(split.base, config)
    return {
        "seq": int(seq),
        "obs_seq": int(obs_seq),
        "status": str(status),
        "message": str(message),
        "runtime_ms": None if runtime_ms is None else float(runtime_ms),
        "publish_base": bool(config.command.publish_base),
        "publish_arms": bool(config.command.publish_arms),
        "base_twist": base.astype(np.float32, copy=False),
        "left_joint_positions": split.left.astype(np.float32, copy=False),
        "right_joint_positions": split.right.astype(np.float32, copy=False),
    }
