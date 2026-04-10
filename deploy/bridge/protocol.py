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
    if limited.shape[0] >= 2:
        limited[1] = float(
            np.clip(
                limited[1],
                -config.command.max_linear_y,
                config.command.max_linear_y,
            )
        )
    if limited.shape[0] >= 3:
        limited[2] = float(
            np.clip(
                limited[2],
                -config.command.max_angular_z,
                config.command.max_angular_z,
            )
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
    split = split_action_vector(
        action,
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
