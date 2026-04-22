from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GripperHysteresisConfig:
    enabled: bool = False
    action_indices: tuple[int, ...] = ()
    closed_values: tuple[float, ...] = ()
    open_values: tuple[float, ...] = ()
    close_thresholds: tuple[float, ...] = ()
    open_thresholds: tuple[float, ...] = ()
    hysteresis_ratio: float = 0.25
    initial_state: str = "current"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_sequence(value: object, *, cast, key: str) -> tuple:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise TypeError(f"Expected `{key}` to be a list or comma-separated string.")
    return tuple(cast(item) for item in items)


def parse_int_sequence(value: object, *, key: str) -> tuple[int, ...]:
    return _as_sequence(value, cast=int, key=key)


def parse_float_sequence(value: object, *, key: str) -> tuple[float, ...]:
    return _as_sequence(value, cast=float, key=key)


def default_gripper_action_indices(
    *,
    action_dim: int,
    base_action_dim: int = 3,
    arm_dof: int | None = None,
) -> tuple[int, ...]:
    if arm_dof is None:
        remaining = int(action_dim) - int(base_action_dim)
        if remaining <= 0 or remaining % 2 != 0:
            return ()
        arm_dof = remaining // 2
    left_idx = int(base_action_dim) + int(arm_dof) - 1
    right_idx = int(base_action_dim) + int(arm_dof) * 2 - 1
    if left_idx < 0 or right_idx >= int(action_dim):
        return ()
    return (left_idx, right_idx)


def parse_gripper_hysteresis_config(
    raw: dict[str, object] | None,
    *,
    action_dim: int | None = None,
    base_action_dim: int = 3,
    arm_dof: int | None = None,
) -> GripperHysteresisConfig:
    data = dict(raw or {})
    enabled = bool(data.get("enabled", False))
    action_indices = parse_int_sequence(
        data.get("action_indices", data.get("indices")),
        key="gripper_hysteresis.action_indices",
    )
    if not action_indices and enabled and action_dim is not None:
        action_indices = default_gripper_action_indices(
            action_dim=int(action_dim),
            base_action_dim=int(base_action_dim),
            arm_dof=arm_dof,
        )
    return GripperHysteresisConfig(
        enabled=enabled,
        action_indices=action_indices,
        closed_values=parse_float_sequence(
            data.get("closed_values"),
            key="gripper_hysteresis.closed_values",
        ),
        open_values=parse_float_sequence(
            data.get("open_values"),
            key="gripper_hysteresis.open_values",
        ),
        close_thresholds=parse_float_sequence(
            data.get("close_thresholds"),
            key="gripper_hysteresis.close_thresholds",
        ),
        open_thresholds=parse_float_sequence(
            data.get("open_thresholds"),
            key="gripper_hysteresis.open_thresholds",
        ),
        hysteresis_ratio=float(data.get("hysteresis_ratio", 0.25)),
        initial_state=str(data.get("initial_state", "current")).lower(),
    )


def _load_action_stats(dataset_root: Path, action_key: str) -> dict[str, list[float]]:
    stats_path = Path(dataset_root) / "meta" / "stats.json"
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    if action_key not in data:
        raise KeyError(f"Dataset stats do not contain action key `{action_key}`: {stats_path}")
    stats = data[action_key]
    if not isinstance(stats, dict):
        raise TypeError(f"Expected stats for `{action_key}` to be a mapping in {stats_path}.")
    return stats


def _derive_thresholds(
    *,
    closed_values: tuple[float, ...],
    open_values: tuple[float, ...],
    hysteresis_ratio: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    close_thresholds: list[float] = []
    open_thresholds: list[float] = []
    ratio = float(np.clip(hysteresis_ratio, 0.0, 0.95))
    for closed, opened in zip(closed_values, open_values):
        closed_f = float(closed)
        opened_f = float(opened)
        direction = 1.0 if opened_f >= closed_f else -1.0
        low = direction * closed_f
        high = direction * opened_f
        midpoint = (low + high) * 0.5
        half_gap = abs(high - low) * ratio * 0.5
        close_thresholds.append(direction * (midpoint - half_gap))
        open_thresholds.append(direction * (midpoint + half_gap))
    return tuple(close_thresholds), tuple(open_thresholds)


def complete_gripper_hysteresis_config_from_dataset_stats(
    config: GripperHysteresisConfig,
    *,
    dataset_root: Path,
    action_key: str = "action",
) -> GripperHysteresisConfig:
    if not config.enabled:
        return config
    if not config.action_indices:
        stats = _load_action_stats(dataset_root, action_key)
        min_values = tuple(float(value) for value in stats.get("min", ()))
        action_dim = len(min_values)
        indices = default_gripper_action_indices(action_dim=action_dim)
        if not indices:
            raise ValueError(
                "`gripper_hysteresis.action_indices` is required because the action "
                f"dimension ({action_dim}) does not match the default zeno layout."
            )
        config = replace(config, action_indices=indices)

    closed_values = config.closed_values
    open_values = config.open_values
    if not closed_values or not open_values:
        stats = _load_action_stats(dataset_root, action_key)
        min_values = tuple(float(value) for value in stats["min"])
        max_values = tuple(float(value) for value in stats["max"])
        closed_values = closed_values or tuple(min_values[idx] for idx in config.action_indices)
        open_values = open_values or tuple(max_values[idx] for idx in config.action_indices)

    close_thresholds = config.close_thresholds
    open_thresholds = config.open_thresholds
    if not close_thresholds or not open_thresholds:
        close_thresholds, open_thresholds = _derive_thresholds(
            closed_values=closed_values,
            open_values=open_values,
            hysteresis_ratio=config.hysteresis_ratio,
        )

    return replace(
        config,
        closed_values=tuple(float(value) for value in closed_values),
        open_values=tuple(float(value) for value in open_values),
        close_thresholds=tuple(float(value) for value in close_thresholds),
        open_thresholds=tuple(float(value) for value in open_thresholds),
    )


class GripperHysteresis:
    def __init__(self, config: GripperHysteresisConfig) -> None:
        self.config = config
        self._validate_config()
        self._is_open: list[bool | None] = [None for _ in self.config.action_indices]

    def _validate_config(self) -> None:
        cfg = self.config
        if not cfg.enabled:
            return
        n = len(cfg.action_indices)
        if n == 0:
            raise ValueError("`gripper_hysteresis.action_indices` must not be empty.")
        for key, values in (
            ("closed_values", cfg.closed_values),
            ("open_values", cfg.open_values),
            ("close_thresholds", cfg.close_thresholds),
            ("open_thresholds", cfg.open_thresholds),
        ):
            if len(values) != n:
                raise ValueError(
                    f"`gripper_hysteresis.{key}` must have {n} values, got {len(values)}."
                )
        if cfg.initial_state not in {"current", "prediction", "closed", "open"}:
            raise ValueError(
                "`gripper_hysteresis.initial_state` must be one of "
                "'current', 'prediction', 'closed', or 'open'."
            )
        for i, (closed, opened, close_threshold, open_threshold) in enumerate(
            zip(
                cfg.closed_values,
                cfg.open_values,
                cfg.close_thresholds,
                cfg.open_thresholds,
            )
        ):
            direction = 1.0 if opened >= closed else -1.0
            if direction * close_threshold > direction * open_threshold:
                raise ValueError(
                    "Expected close_threshold <= open_threshold in gripper state space "
                    f"for index {cfg.action_indices[i]}."
                )

    def reset(self) -> None:
        self._is_open = [None for _ in self.config.action_indices]

    def apply(
        self,
        action: np.ndarray,
        *,
        current_state: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.config.enabled:
            return np.asarray(action, dtype=np.float32)

        vector = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        state_vec = (
            None
            if current_state is None
            else np.asarray(current_state, dtype=np.float32).reshape(-1)
        )
        for i, action_idx in enumerate(self.config.action_indices):
            if action_idx < 0 or action_idx >= vector.shape[0]:
                raise IndexError(
                    f"Gripper action index {action_idx} is outside action dim {vector.shape[0]}."
                )

            raw_value = float(vector[action_idx])
            state_value = (
                None
                if state_vec is None or action_idx >= state_vec.shape[0]
                else float(state_vec[action_idx])
            )
            if self._is_open[i] is None:
                if self.config.initial_state == "open":
                    self._is_open[i] = True
                elif self.config.initial_state == "closed":
                    self._is_open[i] = False
                else:
                    source = raw_value
                    if self.config.initial_state == "current" and state_value is not None:
                        source = state_value
                    self._is_open[i] = abs(source - self.config.open_values[i]) <= abs(
                        source - self.config.closed_values[i]
                    )

            opened = bool(self._is_open[i])
            closed_value = float(self.config.closed_values[i])
            open_value = float(self.config.open_values[i])
            direction = 1.0 if open_value >= closed_value else -1.0
            score = direction * raw_value
            close_boundary = direction * float(self.config.close_thresholds[i])
            open_boundary = direction * float(self.config.open_thresholds[i])

            if opened and score <= close_boundary:
                opened = False
            elif not opened and score >= open_boundary:
                opened = True

            self._is_open[i] = opened
            vector[action_idx] = open_value if opened else closed_value

        return vector

    def describe(self) -> dict[str, Any]:
        return self.config.to_dict()
