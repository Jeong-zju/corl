from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_ROBOCASA_TASK = "PickPlaceCounterToSink"
ROBOCASA_ACTION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("action.gripper_close", 1),
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.base_motion", 4),
    ("action.control_mode", 1),
)
ROBOCASA_IMAGE_KEY_PREFIX = "video."
ROBOCASA_STATE_KEY_PREFIX = "state."
ROBOCASA_TASK_DESCRIPTION_KEY = "annotation.human.task_description"


@dataclass
class RoboCasaRolloutStep:
    step_index: int
    action: np.ndarray
    reward: float
    success: bool
    success_details: dict[str, bool]
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    observation: dict[str, Any] | None = None


@dataclass
class RoboCasaRolloutResult:
    task: str
    seed: int | None
    max_steps: int
    num_steps: int
    total_reward: float
    success: bool
    success_details: dict[str, bool]
    terminated: bool
    truncated: bool
    done_reason: str
    initial_info: dict[str, Any]
    final_info: dict[str, Any]
    initial_observation: dict[str, Any] | None = None
    final_observation: dict[str, Any] | None = None
    trajectory: list[RoboCasaRolloutStep] | None = None
    video_path: str | None = None
    details_path: str | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "num_steps": self.num_steps,
            "total_reward": self.total_reward,
            "success": self.success,
            "success_details": dict(self.success_details),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done_reason": self.done_reason,
            "task_description": self.final_info.get("task_description"),
            "video_path": self.video_path,
            "details_path": self.details_path,
        }


@dataclass
class RoboCasaEvaluationResult:
    task: str
    num_rollouts: int
    max_steps: int
    success_count: int
    success_rate: float
    average_reward: float
    average_steps: float
    rollout_results: list[RoboCasaRolloutResult]
    output_dir: str | None = None
    summary_path: str | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "num_rollouts": self.num_rollouts,
            "max_steps": self.max_steps,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "average_reward": self.average_reward,
            "average_steps": self.average_steps,
            "output_dir": self.output_dir,
            "summary_path": self.summary_path,
            "rollouts": [result.to_summary_dict() for result in self.rollout_results],
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_robocasa_importable() -> None:
    for package_dir in ("robocasa", "robosuite"):
        package_path = _repo_root() / package_dir
        if package_path.exists():
            package_path_str = str(package_path)
            if package_path_str not in sys.path:
                sys.path.insert(0, package_path_str)


def _normalize_success(raw_success: Any) -> tuple[bool, dict[str, bool]]:
    if isinstance(raw_success, Mapping):
        details = {str(key): bool(value) for key, value in raw_success.items()}
        overall = bool(details.get("task", all(details.values()) if details else False))
        if "task" not in details:
            details["task"] = overall
        return overall, details
    overall = bool(raw_success)
    return overall, {"task": overall}


def _safe_copy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            copied[str(key)] = value.copy()
        elif np.isscalar(value):
            copied[str(key)] = value.item() if isinstance(value, np.generic) else value
        else:
            copied[str(key)] = value
    return copied


def _safe_copy_info(info: Mapping[str, Any] | None) -> dict[str, Any]:
    if info is None:
        return {}
    copied: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            copied[str(key)] = value.copy()
        elif isinstance(value, Mapping):
            copied[str(key)] = _safe_copy_info(value)
        elif np.isscalar(value):
            copied[str(key)] = value.item() if isinstance(value, np.generic) else value
        else:
            copied[str(key)] = value
    return copied


def _summarize_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            summary[str(key)] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            summary[str(key)] = value
    return summary


def _transport_encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__transport_type__": "ndarray",
            "dtype": str(value.dtype),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _transport_encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_transport_encode(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _transport_decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_transport_decode(item) for item in value]
    if isinstance(value, dict):
        if value.get("__transport_type__") == "ndarray":
            return np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        return {str(key): _transport_decode(item) for key, item in value.items()}
    return value


def _rollout_step_to_dict(step: RoboCasaRolloutStep) -> dict[str, Any]:
    return {
        "step_index": step.step_index,
        "action": step.action,
        "reward": step.reward,
        "success": step.success,
        "success_details": dict(step.success_details),
        "terminated": step.terminated,
        "truncated": step.truncated,
        "info": dict(step.info),
        "observation": step.observation,
    }


def _rollout_step_from_dict(payload: Mapping[str, Any]) -> RoboCasaRolloutStep:
    return RoboCasaRolloutStep(
        step_index=int(payload["step_index"]),
        action=np.asarray(payload["action"], dtype=np.float32).reshape(-1),
        reward=float(payload["reward"]),
        success=bool(payload["success"]),
        success_details={
            str(key): bool(value)
            for key, value in dict(payload["success_details"]).items()
        },
        terminated=bool(payload["terminated"]),
        truncated=bool(payload["truncated"]),
        info=_safe_copy_info(dict(payload["info"])),
        observation=(
            None
            if payload.get("observation") is None
            else _safe_copy_observation(dict(payload["observation"]))
        ),
    )


def _rollout_result_to_dict(result: RoboCasaRolloutResult) -> dict[str, Any]:
    return {
        "task": result.task,
        "seed": result.seed,
        "max_steps": result.max_steps,
        "num_steps": result.num_steps,
        "total_reward": result.total_reward,
        "success": result.success,
        "success_details": dict(result.success_details),
        "terminated": result.terminated,
        "truncated": result.truncated,
        "done_reason": result.done_reason,
        "initial_info": dict(result.initial_info),
        "final_info": dict(result.final_info),
        "initial_observation": result.initial_observation,
        "final_observation": result.final_observation,
        "trajectory": (
            None
            if result.trajectory is None
            else [_rollout_step_to_dict(step) for step in result.trajectory]
        ),
        "video_path": result.video_path,
        "details_path": result.details_path,
    }


def _rollout_result_from_dict(payload: Mapping[str, Any]) -> RoboCasaRolloutResult:
    trajectory_payload = payload.get("trajectory")
    return RoboCasaRolloutResult(
        task=str(payload["task"]),
        seed=None if payload.get("seed") is None else int(payload["seed"]),
        max_steps=int(payload["max_steps"]),
        num_steps=int(payload["num_steps"]),
        total_reward=float(payload["total_reward"]),
        success=bool(payload["success"]),
        success_details={
            str(key): bool(value)
            for key, value in dict(payload["success_details"]).items()
        },
        terminated=bool(payload["terminated"]),
        truncated=bool(payload["truncated"]),
        done_reason=str(payload["done_reason"]),
        initial_info=_safe_copy_info(dict(payload["initial_info"])),
        final_info=_safe_copy_info(dict(payload["final_info"])),
        initial_observation=(
            None
            if payload.get("initial_observation") is None
            else _safe_copy_observation(dict(payload["initial_observation"]))
        ),
        final_observation=(
            None
            if payload.get("final_observation") is None
            else _safe_copy_observation(dict(payload["final_observation"]))
        ),
        trajectory=(
            None
            if trajectory_payload is None
            else [
                _rollout_step_from_dict(dict(step_payload))
                for step_payload in trajectory_payload
            ]
        ),
        video_path=payload.get("video_path"),
        details_path=payload.get("details_path"),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(output_path: Path, payload: Mapping[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return output_path


def start_ffmpeg_raw_writer(output_path: Path, width: int, height: int, fps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("ffmpeg is required to export RoboCasa playback videos.") from exc


def _normalize_video_frame(frame: np.ndarray | Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"Expected an RGB frame with shape (H, W, 3), got shape={tuple(array.shape)}"
        )
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _close_video_writer(writer, *, output_path: Path) -> None:
    if writer is None:
        return
    if writer.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin for RoboCasa video writing.")
    writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {return_code} while writing {output_path}"
        )


def _run_rollout_with_env_api(
    env: Any,
    *,
    policy: Any | None,
    max_steps: int,
    seed: int | None,
    stop_on_success: bool,
    record_trajectory: bool,
    record_observations: bool,
    video_path: str | Path | None = None,
    details_path: str | Path | None = None,
    video_fps: int = 20,
    video_image_key: str | None = None,
) -> RoboCasaRolloutResult:
    if max_steps <= 0:
        raise ValueError("`max_steps` must be positive.")

    rollout_policy = policy or RandomRoboCasaPolicy()
    if hasattr(rollout_policy, "reset"):
        rollout_policy.reset()

    capture_video = video_path is not None
    if capture_video and not bool(getattr(env, "enable_render", True)) and video_image_key is None:
        raise ValueError(
            "Video recording requires `enable_render=True` unless you explicitly "
            "record from an image key via `video_image_key`."
        )

    writer = None
    video_path_obj = None if video_path is None else Path(video_path)
    details_path_obj = None if details_path is None else Path(details_path)

    def write_frame() -> None:
        nonlocal writer
        if video_path_obj is None:
            return
        frame = env.render_frame(image_key=video_image_key)
        frame = _normalize_video_frame(frame)
        if writer is None:
            writer = start_ffmpeg_raw_writer(
                video_path_obj,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                fps=int(video_fps),
            )
            if writer.stdin is None:
                raise RuntimeError(
                    "Failed to open ffmpeg stdin for RoboCasa rollout video writing."
                )
        writer.stdin.write(frame.tobytes())

    try:
        initial_observation, initial_info = env.reset(
            seed=seed,
            record_observation=record_observations,
        )
        if capture_video:
            write_frame()

        current_observation = initial_observation
        total_reward = 0.0
        executed_steps = 0
        success = bool(initial_info.get("success", False))
        success_details = dict(initial_info.get("success_details", {"task": success}))
        terminated = False
        truncated = False
        done_reason = "reset_success" if success and stop_on_success else "horizon"
        trajectory: list[RoboCasaRolloutStep] = []
        final_info = dict(initial_info)

        if not (success and stop_on_success):
            for step_index in range(int(max_steps)):
                if not hasattr(rollout_policy, "act"):
                    raise TypeError(
                        "Rollout policy must define `act(observation, *, env=...)`."
                    )
                action = rollout_policy.act(current_observation, env=env)
                next_observation, reward, terminated, truncated, step_info = env.step(
                    action,
                    record_observation=record_observations,
                )
                if capture_video:
                    write_frame()
                executed_steps += 1
                total_reward += float(reward)
                success = bool(step_info.get("success", False))
                success_details = dict(
                    step_info.get("success_details", {"task": success})
                )
                final_info = dict(step_info)

                if record_trajectory:
                    trajectory.append(
                        RoboCasaRolloutStep(
                            step_index=step_index,
                            action=np.asarray(action, dtype=np.float32).reshape(-1).copy(),
                            reward=float(reward),
                            success=success,
                            success_details=success_details,
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                            info=dict(step_info),
                            observation=next_observation if record_observations else None,
                        )
                    )

                current_observation = next_observation
                if success and stop_on_success:
                    done_reason = "success"
                    break
                if terminated:
                    done_reason = "terminated"
                    break
                if truncated:
                    done_reason = "truncated"
                    break
            else:
                done_reason = "horizon"

        result = RoboCasaRolloutResult(
            task=str(getattr(env, "task", DEFAULT_ROBOCASA_TASK)),
            seed=None if seed is None else int(seed),
            max_steps=int(max_steps),
            num_steps=int(executed_steps),
            total_reward=float(total_reward),
            success=bool(success),
            success_details=success_details,
            terminated=bool(terminated),
            truncated=bool(truncated),
            done_reason=done_reason,
            initial_info=dict(initial_info),
            final_info=dict(final_info),
            initial_observation=initial_observation if record_observations else None,
            final_observation=current_observation if record_observations else None,
            trajectory=trajectory if record_trajectory else None,
            video_path=None if video_path_obj is None else str(video_path_obj),
            details_path=None if details_path_obj is None else str(details_path_obj),
        )
        if details_path_obj is not None:
            _write_json(
                details_path_obj,
                _rollout_result_to_dict(result),
            )
        return result
    finally:
        _close_video_writer(writer, output_path=video_path_obj) if writer is not None and video_path_obj is not None else None


def evaluate_policy_rollouts(
    env: Any,
    *,
    policy: Any | None = None,
    num_rollouts: int,
    max_steps: int,
    seed: int | None = None,
    stop_on_success: bool = True,
    record_trajectory: bool = False,
    record_observations: bool = False,
    output_dir: str | Path | None = None,
    save_videos: bool = False,
    video_fps: int = 20,
    video_image_key: str | None = None,
) -> RoboCasaEvaluationResult:
    if num_rollouts <= 0:
        raise ValueError("`num_rollouts` must be positive.")

    output_dir_path = None if output_dir is None else Path(output_dir)
    rollout_results: list[RoboCasaRolloutResult] = []
    success_count = 0
    total_reward = 0.0
    total_steps = 0

    for rollout_index in range(int(num_rollouts)):
        rollout_seed = None if seed is None else int(seed) + rollout_index
        rollout_video_path = None
        rollout_details_path = None
        if output_dir_path is not None:
            rollout_details_path = output_dir_path / f"rollout_{rollout_index:03d}.json"
            if save_videos:
                rollout_video_path = output_dir_path / f"rollout_{rollout_index:03d}.mp4"

        result = _run_rollout_with_env_api(
            env,
            policy=policy,
            max_steps=max_steps,
            seed=rollout_seed,
            stop_on_success=stop_on_success,
            record_trajectory=record_trajectory,
            record_observations=record_observations,
            video_path=rollout_video_path,
            details_path=rollout_details_path,
            video_fps=video_fps,
            video_image_key=video_image_key,
        )
        rollout_results.append(result)
        success_count += int(result.success)
        total_reward += float(result.total_reward)
        total_steps += int(result.num_steps)

    summary = RoboCasaEvaluationResult(
        task=str(getattr(env, "task", DEFAULT_ROBOCASA_TASK)),
        num_rollouts=int(num_rollouts),
        max_steps=int(max_steps),
        success_count=int(success_count),
        success_rate=float(success_count / max(1, num_rollouts)),
        average_reward=float(total_reward / max(1, num_rollouts)),
        average_steps=float(total_steps / max(1, num_rollouts)),
        rollout_results=rollout_results,
        output_dir=None if output_dir_path is None else str(output_dir_path),
        summary_path=(
            None if output_dir_path is None else str(output_dir_path / "summary.json")
        ),
    )
    if output_dir_path is not None:
        _write_json(output_dir_path / "summary.json", summary.to_summary_dict())
    return summary


def get_robocasa_action_dim() -> int:
    return sum(size for _, size in ROBOCASA_ACTION_COMPONENTS)


def flatten_robocasa_action(action: Mapping[str, Any]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for key, size in ROBOCASA_ACTION_COMPONENTS:
        if key not in action:
            raise KeyError(f"Missing RoboCasa action component: {key}")
        value = np.asarray(action[key], dtype=np.float32).reshape(-1)
        if value.shape[0] != size:
            raise ValueError(
                f"RoboCasa action component {key!r} expected size={size}, "
                f"got shape={tuple(value.shape)}"
            )
        chunks.append(value)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def unflatten_robocasa_action(action: Sequence[float] | np.ndarray) -> dict[str, np.ndarray]:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    expected_dim = get_robocasa_action_dim()
    if vector.shape[0] != expected_dim:
        raise ValueError(
            f"RoboCasa action vector expected dim={expected_dim}, "
            f"got shape={tuple(vector.shape)}"
        )

    action_dict: dict[str, np.ndarray] = {}
    start = 0
    for key, size in ROBOCASA_ACTION_COMPONENTS:
        end = start + size
        action_dict[key] = vector[start:end].copy()
        start = end
    return action_dict


def flatten_robocasa_state_observation(
    observation: Mapping[str, Any],
    *,
    state_keys: Sequence[str] | None = None,
) -> np.ndarray:
    if state_keys is None:
        state_keys = sorted(
            key
            for key, value in observation.items()
            if str(key).startswith(ROBOCASA_STATE_KEY_PREFIX)
            and isinstance(value, np.ndarray)
        )
    if not state_keys:
        raise ValueError("No RoboCasa state observation keys were found to flatten.")
    parts = [
        np.asarray(observation[key], dtype=np.float32).reshape(-1)
        for key in state_keys
    ]
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _list_available_robocasa_tasks_local() -> list[str]:
    _ensure_robocasa_importable()
    from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS

    return sorted(str(task_name) for task_name in REGISTERED_KITCHEN_ENVS)


def list_available_robocasa_tasks(
    *,
    conda_env: str | None = None,
) -> list[str]:
    if conda_env is None:
        return _list_available_robocasa_tasks_local()
    remote = _RoboCasaRemoteProcess(conda_env=conda_env)
    try:
        tasks = remote.request("list_tasks")
        return [str(task_name) for task_name in list(tasks)]
    finally:
        remote.close()


class RandomRoboCasaPolicy:
    def __init__(
        self,
        *,
        freeze_base_motion: bool = True,
        control_mode: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.freeze_base_motion = bool(freeze_base_motion)
        self.control_mode = control_mode
        self._rng = rng

    def reset(self) -> None:
        return None

    def act(
        self,
        observation: Mapping[str, Any],
        *,
        env: "RoboCasaBenchmarkEnv",
    ) -> np.ndarray:
        action = env.sample_random_action(
            freeze_base_motion=self.freeze_base_motion,
            control_mode=self.control_mode,
            rng=self._rng,
        )
        return action


class RoboCasaBenchmarkEnv:
    def __init__(
        self,
        task: str = DEFAULT_ROBOCASA_TASK,
        *,
        seed: int | None = None,
        camera_names: Sequence[str] | None = None,
        camera_width: int = 256,
        camera_height: int = 256,
        enable_render: bool = True,
        split: str = "all",
        robots: str = "PandaOmron",
        env_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.task = str(task)
        self.seed = seed
        self.camera_names = tuple(camera_names) if camera_names is not None else None
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.enable_render = bool(enable_render)
        self.split = str(split)
        self.robots = str(robots)
        self.env_kwargs = dict(env_kwargs or {})
        self._last_observation: dict[str, Any] | None = None
        self._env = self._make_env()

    def _make_env(self):
        _ensure_robocasa_importable()
        try:
            from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Failed to import RoboCasa dependencies. Make sure `mujoco`, "
                "`gymnasium`, OpenCV, and the local `robocasa` / `robosuite` "
                "packages are available."
            ) from exc

        kwargs = dict(self.env_kwargs)
        if self.camera_names is not None:
            kwargs["camera_names"] = list(self.camera_names)

        return RoboCasaGymEnv(
            env_name=self.task,
            camera_widths=self.camera_width,
            camera_heights=self.camera_height,
            enable_render=self.enable_render,
            split=self.split,
            robots=self.robots,
            **kwargs,
        )

    @property
    def env(self):
        return self._env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_dim(self) -> int:
        return get_robocasa_action_dim()

    @property
    def state_keys(self) -> list[str]:
        return sorted(
            key
            for key in self.observation_space.keys()
            if str(key).startswith(ROBOCASA_STATE_KEY_PREFIX)
        )

    @property
    def image_keys(self) -> list[str]:
        return sorted(
            key
            for key in self.observation_space.keys()
            if str(key).startswith(ROBOCASA_IMAGE_KEY_PREFIX)
        )

    def close(self) -> None:
        self._env.close()

    def render_frame(self, *, image_key: str | None = None) -> np.ndarray:
        if image_key is None:
            return np.asarray(self._env.render(), dtype=np.uint8).copy()
        if self._last_observation is None:
            raise RuntimeError("Call reset() before requesting a RoboCasa render frame.")
        if image_key not in self._last_observation:
            raise KeyError(
                f"Image key {image_key!r} is not available in the latest observation."
            )
        return np.asarray(self._last_observation[image_key], dtype=np.uint8).copy()

    def _compute_success(self) -> tuple[bool, dict[str, bool]]:
        raw_success = self._env.env._check_success()
        return _normalize_success(raw_success)

    def _augment_info(
        self,
        info: Mapping[str, Any] | None,
        *,
        observation: Mapping[str, Any] | None = None,
        reward: float | None = None,
        action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        success, success_details = self._compute_success()
        enriched = _safe_copy_info(info)
        enriched["success"] = success
        enriched["success_details"] = dict(success_details)
        if reward is not None:
            enriched["reward"] = float(reward)
        if observation is not None:
            enriched["task_description"] = observation.get(ROBOCASA_TASK_DESCRIPTION_KEY)
        if action is not None:
            enriched["action"] = np.asarray(action, dtype=np.float32).copy()
        return enriched

    def reset(
        self,
        *,
        seed: int | None = None,
        record_observation: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effective_seed = self.seed if seed is None else int(seed)
        observation, info = self._env.reset(seed=effective_seed)
        observation_copy = _safe_copy_observation(observation)
        self._last_observation = observation_copy
        return (
            observation_copy if record_observation else _summarize_observation(observation_copy),
            self._augment_info(info, observation=observation_copy),
        )

    def sample_random_action(
        self,
        *,
        freeze_base_motion: bool = True,
        control_mode: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        sampled = self.action_space.sample()
        if rng is not None:
            for key, space in self.action_space.spaces.items():
                if hasattr(space, "low") and hasattr(space, "high"):
                    low = np.asarray(space.low, dtype=np.float32)
                    high = np.asarray(space.high, dtype=np.float32)
                    sampled[key] = rng.uniform(low=low, high=high).astype(
                        np.float32, copy=False
                    )
                elif hasattr(space, "n"):
                    sampled[key] = np.asarray(
                        [rng.integers(int(space.n))], dtype=np.float32
                    )
        if freeze_base_motion and "action.base_motion" in sampled:
            sampled["action.base_motion"] = np.zeros_like(
                sampled["action.base_motion"], dtype=np.float32
            )
        if control_mode is not None and "action.control_mode" in sampled:
            sampled["action.control_mode"] = np.asarray(
                [control_mode], dtype=np.float32
            )
        return flatten_robocasa_action(sampled)

    def step(
        self,
        action: Sequence[float] | Mapping[str, Any] | np.ndarray,
        *,
        record_observation: bool = True,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if isinstance(action, Mapping):
            action_dict = {
                str(key): np.asarray(value, dtype=np.float32).reshape(-1)
                for key, value in action.items()
            }
            action_vector = flatten_robocasa_action(action_dict)
        else:
            action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
            action_dict = unflatten_robocasa_action(action_vector)

        observation, _, terminated, truncated, info = self._env.step(action_dict)
        reward = float(self._env.env.reward())
        observation_copy = _safe_copy_observation(observation)
        self._last_observation = observation_copy
        enriched_info = self._augment_info(
            info,
            observation=observation_copy,
            reward=reward,
            action=action_vector,
        )
        return (
            observation_copy if record_observation else _summarize_observation(observation_copy),
            reward,
            bool(terminated),
            bool(truncated),
            enriched_info,
        )

    def rollout(
        self,
        *,
        policy: Any | None = None,
        max_steps: int = 200,
        seed: int | None = None,
        stop_on_success: bool = True,
        record_trajectory: bool = True,
        record_observations: bool = True,
        video_path: str | Path | None = None,
        details_path: str | Path | None = None,
        video_fps: int = 20,
        video_image_key: str | None = None,
    ) -> RoboCasaRolloutResult:
        return _run_rollout_with_env_api(
            self,
            policy=policy,
            max_steps=max_steps,
            seed=seed,
            stop_on_success=stop_on_success,
            record_trajectory=record_trajectory,
            record_observations=record_observations,
            video_path=video_path,
            details_path=details_path,
            video_fps=video_fps,
            video_image_key=video_image_key,
        )

    def evaluate_policy(
        self,
        *,
        policy: Any | None = None,
        num_rollouts: int,
        max_steps: int,
        seed: int | None = None,
        stop_on_success: bool = True,
        record_trajectory: bool = False,
        record_observations: bool = False,
        output_dir: str | Path | None = None,
        save_videos: bool = False,
        video_fps: int = 20,
        video_image_key: str | None = None,
    ) -> RoboCasaEvaluationResult:
        return evaluate_policy_rollouts(
            self,
            policy=policy,
            num_rollouts=num_rollouts,
            max_steps=max_steps,
            seed=seed,
            stop_on_success=stop_on_success,
            record_trajectory=record_trajectory,
            record_observations=record_observations,
            output_dir=output_dir,
            save_videos=save_videos,
            video_fps=video_fps,
            video_image_key=video_image_key,
        )


def _describe_env(env: RoboCasaBenchmarkEnv) -> dict[str, Any]:
    return {
        "task": env.task,
        "seed": env.seed,
        "action_dim": env.action_dim,
        "state_keys": list(env.state_keys),
        "image_keys": list(env.image_keys),
        "action_components": [
            {"key": key, "size": size}
            for key, size in ROBOCASA_ACTION_COMPONENTS
        ],
    }


class _RoboCasaRemoteProcess:
    def __init__(
        self,
        *,
        conda_env: str = "robocasa",
        python_executable: str = "python",
    ) -> None:
        self.conda_env = str(conda_env)
        self.python_executable = str(python_executable)
        self._stderr_lines: list[str] = []
        self._closed = False
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            self.conda_env,
            self.python_executable,
            "-u",
            str(Path(__file__).resolve()),
            "--serve",
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(_repo_root()),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
        )
        self._stderr_thread.start()
        ready = self._read_response()
        if ready.get("type") != "ready":
            raise RuntimeError(
                "RoboCasa remote worker did not send a ready message. "
                + self._format_stderr_tail()
            )

    def _drain_stderr(self) -> None:
        if self._process.stderr is None:
            return
        for line in self._process.stderr:
            self._stderr_lines.append(line.rstrip())
            if len(self._stderr_lines) > 200:
                self._stderr_lines = self._stderr_lines[-200:]

    def _format_stderr_tail(self, limit: int = 40) -> str:
        if not self._stderr_lines:
            return ""
        tail = "\n".join(self._stderr_lines[-limit:])
        return f"\n[remote stderr]\n{tail}"

    def _read_response(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RuntimeError("RoboCasa remote worker stdout is unavailable.")
        line = self._process.stdout.readline()
        if line == "":
            raise RuntimeError(
                "RoboCasa remote worker exited unexpectedly."
                + self._format_stderr_tail()
            )
        payload = json.loads(line)
        return _transport_decode(payload)

    def request(self, command: str, **kwargs) -> Any:
        if self._closed:
            raise RuntimeError("RoboCasa remote worker is already closed.")
        if self._process.stdin is None:
            raise RuntimeError("RoboCasa remote worker stdin is unavailable.")
        message = {
            "command": str(command),
            "kwargs": kwargs,
        }
        self._process.stdin.write(json.dumps(_transport_encode(message)) + "\n")
        self._process.stdin.flush()
        response = self._read_response()
        if not bool(response.get("ok", False)):
            error = response.get("error", {})
            error_type = error.get("type", "RuntimeError")
            error_message = error.get("message", "Unknown remote RoboCasa error.")
            error_traceback = error.get("traceback")
            details = f"{error_type}: {error_message}"
            if error_traceback:
                details = f"{details}\n{error_traceback}"
            raise RuntimeError(details + self._format_stderr_tail())
        return response.get("result")

    def close(self) -> None:
        if self._closed:
            return
        try:
            try:
                self.request("close")
            except Exception:
                pass
            if self._process.stdin is not None:
                self._process.stdin.close()
        finally:
            self._closed = True
            self._process.wait(timeout=10)


class RemoteRoboCasaBenchmarkEnv:
    def __init__(
        self,
        task: str = DEFAULT_ROBOCASA_TASK,
        *,
        conda_env: str = "robocasa",
        python_executable: str = "python",
        seed: int | None = None,
        camera_names: Sequence[str] | None = None,
        camera_width: int = 256,
        camera_height: int = 256,
        enable_render: bool = True,
        split: str = "all",
        robots: str = "PandaOmron",
        env_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.task = str(task)
        self.seed = seed
        self.conda_env = str(conda_env)
        self.camera_names = tuple(camera_names) if camera_names is not None else None
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.enable_render = bool(enable_render)
        self.split = str(split)
        self.robots = str(robots)
        self.env_kwargs = dict(env_kwargs or {})
        self._remote = _RoboCasaRemoteProcess(
            conda_env=self.conda_env,
            python_executable=python_executable,
        )
        description = self._remote.request(
            "create",
            task=self.task,
            seed=self.seed,
            camera_names=self.camera_names,
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            enable_render=self.enable_render,
            split=self.split,
            robots=self.robots,
            env_kwargs=self.env_kwargs,
        )
        self._action_dim = int(description["action_dim"])
        self._state_keys = [str(key) for key in list(description["state_keys"])]
        self._image_keys = [str(key) for key in list(description["image_keys"])]

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_keys(self) -> list[str]:
        return list(self._state_keys)

    @property
    def image_keys(self) -> list[str]:
        return list(self._image_keys)

    def close(self) -> None:
        self._remote.close()

    def render_frame(self, *, image_key: str | None = None) -> np.ndarray:
        result = self._remote.request("render", image_key=image_key)
        return np.asarray(result, dtype=np.uint8).copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        record_observation: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self._remote.request(
            "reset",
            seed=seed,
            record_observation=record_observation,
        )
        return dict(result["observation"]), dict(result["info"])

    def sample_random_action(
        self,
        *,
        freeze_base_motion: bool = True,
        control_mode: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        result = self._remote.request(
            "sample_random_action",
            freeze_base_motion=freeze_base_motion,
            control_mode=control_mode,
        )
        return np.asarray(result, dtype=np.float32).reshape(-1)

    def step(
        self,
        action: Sequence[float] | Mapping[str, Any] | np.ndarray,
        *,
        record_observation: bool = True,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if isinstance(action, Mapping):
            action_payload: Any = {
                str(key): np.asarray(value, dtype=np.float32).reshape(-1)
                for key, value in action.items()
            }
        else:
            action_payload = np.asarray(action, dtype=np.float32).reshape(-1)
        result = self._remote.request(
            "step",
            action=action_payload,
            record_observation=record_observation,
        )
        return (
            dict(result["observation"]),
            float(result["reward"]),
            bool(result["terminated"]),
            bool(result["truncated"]),
            dict(result["info"]),
        )

    def rollout(
        self,
        *,
        policy: Any | None = None,
        max_steps: int = 200,
        seed: int | None = None,
        stop_on_success: bool = True,
        record_trajectory: bool = True,
        record_observations: bool = True,
        video_path: str | Path | None = None,
        details_path: str | Path | None = None,
        video_fps: int = 20,
        video_image_key: str | None = None,
    ) -> RoboCasaRolloutResult:
        rollout_policy = policy or RandomRoboCasaPolicy()
        if (
            isinstance(rollout_policy, RandomRoboCasaPolicy)
            and video_path is None
            and details_path is None
            and not record_trajectory
            and not record_observations
        ):
            result = self._remote.request(
                "rollout_random",
                max_steps=max_steps,
                seed=seed,
                stop_on_success=stop_on_success,
                record_trajectory=record_trajectory,
                record_observations=record_observations,
                freeze_base_motion=rollout_policy.freeze_base_motion,
                control_mode=rollout_policy.control_mode,
            )
            return _rollout_result_from_dict(dict(result))
        return _run_rollout_with_env_api(
            self,
            policy=policy,
            max_steps=max_steps,
            seed=seed,
            stop_on_success=stop_on_success,
            record_trajectory=record_trajectory,
            record_observations=record_observations,
            video_path=video_path,
            details_path=details_path,
            video_fps=video_fps,
            video_image_key=video_image_key,
        )

    def evaluate_policy(
        self,
        *,
        policy: Any | None = None,
        num_rollouts: int,
        max_steps: int,
        seed: int | None = None,
        stop_on_success: bool = True,
        record_trajectory: bool = False,
        record_observations: bool = False,
        output_dir: str | Path | None = None,
        save_videos: bool = False,
        video_fps: int = 20,
        video_image_key: str | None = None,
    ) -> RoboCasaEvaluationResult:
        return evaluate_policy_rollouts(
            self,
            policy=policy,
            num_rollouts=num_rollouts,
            max_steps=max_steps,
            seed=seed,
            stop_on_success=stop_on_success,
            record_trajectory=record_trajectory,
            record_observations=record_observations,
            output_dir=output_dir,
            save_videos=save_videos,
            video_fps=video_fps,
            video_image_key=video_image_key,
        )


def create_robocasa_benchmark_env(
    *,
    conda_env: str | None = None,
    **kwargs,
) -> RoboCasaBenchmarkEnv | RemoteRoboCasaBenchmarkEnv:
    if conda_env is None:
        return RoboCasaBenchmarkEnv(**kwargs)
    return RemoteRoboCasaBenchmarkEnv(conda_env=conda_env, **kwargs)


def _run_server() -> int:
    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr
    env: RoboCasaBenchmarkEnv | None = None

    def send(payload: Mapping[str, Any]) -> None:
        protocol_stdout.write(json.dumps(_transport_encode(dict(payload))) + "\n")
        protocol_stdout.flush()

    send({"ok": True, "type": "ready"})

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            request = _transport_decode(json.loads(raw))
            command = str(request["command"])
            kwargs = dict(request.get("kwargs", {}))

            if command == "list_tasks":
                result: Any = _list_available_robocasa_tasks_local()
            elif command == "create":
                if env is not None:
                    env.close()
                env = RoboCasaBenchmarkEnv(**kwargs)
                result = _describe_env(env)
            elif command == "reset":
                if env is None:
                    raise RuntimeError("Remote RoboCasa env has not been created yet.")
                observation, info = env.reset(**kwargs)
                result = {
                    "observation": observation,
                    "info": info,
                }
            elif command == "sample_random_action":
                if env is None:
                    raise RuntimeError("Remote RoboCasa env has not been created yet.")
                result = env.sample_random_action(**kwargs)
            elif command == "step":
                if env is None:
                    raise RuntimeError("Remote RoboCasa env has not been created yet.")
                observation, reward, terminated, truncated, info = env.step(**kwargs)
                result = {
                    "observation": observation,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                }
            elif command == "render":
                if env is None:
                    raise RuntimeError("Remote RoboCasa env has not been created yet.")
                result = env.render_frame(**kwargs)
            elif command == "rollout_random":
                if env is None:
                    raise RuntimeError("Remote RoboCasa env has not been created yet.")
                policy = RandomRoboCasaPolicy(
                    freeze_base_motion=bool(kwargs.pop("freeze_base_motion", True)),
                    control_mode=kwargs.pop("control_mode", None),
                )
                result = _rollout_result_to_dict(env.rollout(policy=policy, **kwargs))
            elif command == "close":
                if env is not None:
                    env.close()
                    env = None
                send({"ok": True, "result": None})
                return 0
            else:
                raise ValueError(f"Unknown RoboCasa remote command: {command}")

            send({"ok": True, "result": result})
        except Exception as exc:
            send(
                {
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            )

    if env is not None:
        env.close()
    return 0


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a RoboCasa task, run a random-policy rollout, and report "
            "reward / success metrics."
        )
    )
    parser.add_argument("--task", type=str, default=DEFAULT_ROBOCASA_TASK)
    parser.add_argument(
        "--conda-env",
        type=str,
        default=None,
        help=(
            "Optional conda environment name used to launch RoboCasa remotely. "
            "Use this from another environment such as `corl-py312`."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory used to save rollout / eval summaries and playback videos.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save rollout playback as mp4 files under --output-dir.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=20,
        help="Playback video fps.",
    )
    parser.add_argument(
        "--video-image-key",
        type=str,
        default=None,
        help="Optional image observation key to record instead of env.render().",
    )
    parser.add_argument(
        "--disable-render",
        action="store_true",
        help="Disable image observations and replace them with black frames.",
    )
    parser.add_argument(
        "--allow-base-motion",
        action="store_true",
        help="Let the random policy sample base motion instead of clamping it to zero.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List registered RoboCasa task names and exit.",
    )
    parser.add_argument(
        "--print-observation-summary",
        action="store_true",
        help="Print the reset observation structure before rollout.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.serve:
        return _run_server()

    if args.list_tasks:
        for task_name in list_available_robocasa_tasks(conda_env=args.conda_env):
            print(task_name)
        return 0

    if args.save_video and args.output_dir is None:
        raise ValueError("`--save-video` requires `--output-dir`.")

    env = create_robocasa_benchmark_env(
        conda_env=args.conda_env,
        task=args.task,
        seed=args.seed,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        enable_render=not args.disable_render,
        split=args.split,
    )
    try:
        output_dir = None if args.output_dir is None else Path(args.output_dir)
        if args.print_observation_summary:
            observation, info = env.reset(seed=args.seed)
            print(json.dumps(_summarize_observation(observation), indent=2))
            print(json.dumps(_safe_copy_info(info), indent=2, default=str))

        policy = RandomRoboCasaPolicy(
            freeze_base_motion=not args.allow_base_motion
        )
        if int(args.num_rollouts) <= 1:
            video_path = None
            details_path = None
            if output_dir is not None:
                details_path = output_dir / "rollout_000.json"
                if args.save_video:
                    video_path = output_dir / "rollout_000.mp4"
            result = env.rollout(
                policy=policy,
                max_steps=args.max_steps,
                seed=args.seed,
                stop_on_success=True,
                record_trajectory=False,
                record_observations=False,
                video_path=video_path,
                details_path=details_path,
                video_fps=args.video_fps,
                video_image_key=args.video_image_key,
            )
            if output_dir is not None:
                _write_json(output_dir / "summary.json", result.to_summary_dict())
            print(json.dumps(result.to_summary_dict(), indent=2, default=str))
        else:
            evaluation = env.evaluate_policy(
                policy=policy,
                num_rollouts=args.num_rollouts,
                max_steps=args.max_steps,
                seed=args.seed,
                stop_on_success=True,
                record_trajectory=False,
                record_observations=False,
                output_dir=output_dir,
                save_videos=args.save_video,
                video_fps=args.video_fps,
                video_image_key=args.video_image_key,
            )
            print(json.dumps(evaluation.to_summary_dict(), indent=2, default=str))
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
