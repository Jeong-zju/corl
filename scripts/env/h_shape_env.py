from __future__ import annotations

from collections import deque
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from eval_helpers import (
    build_eval_observation,
    build_prefix_sequence_eval_inputs,
    compute_delta_signature_step_np,
    compute_signatory_signature_np,
    compute_simple_signature_np,
    ensure_prefix_sequence_batch_dims,
    resolve_signature_backend,
    resolve_single_visual_observation_feature,
    write_summary,
)
from policy_defaults import load_policy_mode_defaults


ENV_NAME = "h_shape"
DEFAULT_DATASET_ROOT = Path("data/zeno-ai/rrt_connect_h_v30")
DEFAULT_DATASET_REPO_ID = "zeno-ai/rrt_connect_h_v30"
DEFAULT_NUM_EPISODES = 100
DEFAULT_NUM_ROLLOUTS = 20
DEFAULT_MAX_STEPS = 120
DEFAULT_FPS = 20
DEFAULT_IMAGE_SIZE = 128
DEFAULT_SUCCESS_THRESHOLD = 0.20
DEFAULT_MAX_ACTION_STEP = 0.30
DEFAULT_INCLUDE_PATH_SIGNATURES = True
DEFAULT_PATH_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_DELTA_SIGNATURE_KEY = "observation.delta_signature"
DEFAULT_SIGNATURE_WINDOW_SIZE = 0
DEFAULT_SIGNATURE_DEPTH = 3
DEFAULT_SIGNATURE_BACKEND = "auto"

TASK_ID_TO_NAME = {
    0: "upper_bridge",
    1: "lower_bridge",
}
TASK_ID_TO_DESCRIPTION = {
    0: "Navigate through the upper bridge of the H-maze from left to right.",
    1: "Navigate through the lower bridge of the H-maze from left to right.",
}

_TRAIN_DEFAULTS = {
    "act": {
        "output_root": Path("outputs/train/h_shape_act"),
        "job_name": "act_h_shape",
        "wandb_project": "lerobot-rrt-act",
        "eval_output_dir": Path("outputs/eval/h_shape_act"),
    },
    "streaming_act": {
        "output_root": Path("outputs/train/h_shape_streaming_act"),
        "job_name": "streaming_act_h_shape",
        "wandb_project": "lerobot-rrt-streaming-act",
        "eval_output_dir": Path("outputs/eval/h_shape_streaming_act"),
    },
}


def create_h_shape_grid(
    grid_size: tuple[int, int] = (200, 200),
    total_width: float = 6.0,
    total_height: float = 8.0,
    bar_thickness: float = 1.5,
    mid_bar_height: float = 1.0,
    wall_thickness: float = 0.3,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    rows, cols = grid_size
    half_width = total_width / 2
    half_height = total_height / 2
    middle_half_height = mid_bar_height / 2

    xmin, xmax = -half_width - 1.0, half_width + 1.0
    ymin, ymax = -half_height - 1.0, half_height + 1.0

    xs = np.linspace(xmin, xmax, cols)
    ys = np.linspace(ymin, ymax, rows)
    grid_x, grid_y = np.meshgrid(xs, ys)

    left_inner = (
        (grid_x > -half_width + wall_thickness)
        & (grid_x < -half_width + bar_thickness - wall_thickness)
        & (grid_y > -half_height + wall_thickness)
        & (grid_y < half_height - wall_thickness)
    )
    right_inner = (
        (grid_x > half_width - bar_thickness + wall_thickness)
        & (grid_x < half_width - wall_thickness)
        & (grid_y > -half_height + wall_thickness)
        & (grid_y < half_height - wall_thickness)
    )
    middle_inner = (
        (grid_x > -half_width + bar_thickness - wall_thickness)
        & (grid_x < half_width - bar_thickness + wall_thickness)
        & (grid_y > -middle_half_height + wall_thickness)
        & (grid_y < middle_half_height - wall_thickness)
    )

    grid = np.ones((rows, cols), dtype=np.float64)
    grid[left_inner | right_inner | middle_inner] = 0.0
    return grid, (xmin, xmax, ymin, ymax)


def world_to_grid(
    point: tuple[float, float],
    extent: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> tuple[int, int]:
    xmin, xmax, ymin, ymax = extent
    rows, cols = shape
    x, y = point
    col = int(round((x - xmin) / (xmax - xmin) * (cols - 1)))
    row = int(round((y - ymin) / (ymax - ymin) * (rows - 1)))
    row = int(np.clip(row, 0, rows - 1))
    col = int(np.clip(col, 0, cols - 1))
    return row, col


def grid_to_world(
    row: int,
    col: int,
    extent: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = extent
    rows, cols = shape
    x = np.interp(col, [0, cols - 1], [xmin, xmax])
    y = np.interp(row, [0, rows - 1], [ymin, ymax])
    return float(x), float(y)


def is_collision_free(
    point: tuple[float, float],
    grid: np.ndarray,
    extent: tuple[float, float, float, float],
) -> bool:
    row, col = world_to_grid(point, extent, grid.shape)
    return bool(grid[row, col] == 0.0)


def segment_collision_free(
    start: tuple[float, float],
    end: tuple[float, float],
    grid: np.ndarray,
    extent: tuple[float, float, float, float],
    num_checks: int = 24,
) -> bool:
    for t in np.linspace(0.0, 1.0, num_checks):
        point = (
            float(start[0] * (1.0 - t) + end[0] * t),
            float(start[1] * (1.0 - t) + end[1] * t),
        )
        if not is_collision_free(point, grid, extent):
            return False
    return True


def find_fixed_h_corners(
    grid: np.ndarray,
    extent: tuple[float, float, float, float],
) -> dict[str, tuple[float, float]]:
    free = np.argwhere(grid == 0)
    if free.shape[0] == 0:
        raise RuntimeError("No free cells found in the H-map.")

    upper_left = upper_right = lower_left = lower_right = None
    ul_score = ur_score = ll_score = lr_score = None

    for row, col in free:
        x, y = grid_to_world(int(row), int(col), extent, grid.shape)
        if x < 0.0 and y > 0.0:
            score = (int(col), -int(row))
            if ul_score is None or score < ul_score:
                upper_left = (int(row), int(col))
                ul_score = score
        if x > 0.0 and y > 0.0:
            score = (-int(col), -int(row))
            if ur_score is None or score < ur_score:
                upper_right = (int(row), int(col))
                ur_score = score
        if x < 0.0 and y < 0.0:
            score = (int(col), int(row))
            if ll_score is None or score < ll_score:
                lower_left = (int(row), int(col))
                ll_score = score
        if x > 0.0 and y < 0.0:
            score = (-int(col), int(row))
            if lr_score is None or score < lr_score:
                lower_right = (int(row), int(col))
                lr_score = score

    if any(value is None for value in (upper_left, upper_right, lower_left, lower_right)):
        raise RuntimeError("Could not locate all four deterministic H corners.")

    return {
        "upper_left": grid_to_world(*upper_left, extent, grid.shape),
        "upper_right": grid_to_world(*upper_right, extent, grid.shape),
        "lower_left": grid_to_world(*lower_left, extent, grid.shape),
        "lower_right": grid_to_world(*lower_right, extent, grid.shape),
    }


def world_to_pixel(
    point: tuple[float, float],
    extent: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[int, int]:
    xmin, xmax, ymin, ymax = extent
    height, width = size
    x, y = point
    px = int(round((x - xmin) / (xmax - xmin) * (width - 1)))
    py = int(round((y - ymin) / (ymax - ymin) * (height - 1)))
    px = int(np.clip(px, 0, width - 1))
    py = int(np.clip(py, 0, height - 1))
    return px, py


def draw_disk(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    cx, cy = center
    height, width = image.shape[:2]
    y_coords, x_coords = np.ogrid[:height, :width]
    mask = (x_coords - cx) ** 2 + (y_coords - cy) ** 2 <= radius * radius
    image[mask] = np.asarray(color, dtype=np.uint8)


def make_base_image(grid: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    rows, cols = grid.shape
    height, width = image_hw
    row_indices = np.linspace(0, rows - 1, height).astype(int)
    col_indices = np.linspace(0, cols - 1, width).astype(int)
    sampled = grid[np.ix_(row_indices, col_indices)]
    gray = np.where(sampled == 0, np.uint8(240), np.uint8(55)).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def render_frame(
    base_img: np.ndarray,
    extent: tuple[float, float, float, float],
    agent_pos: tuple[float, float],
) -> np.ndarray:
    frame = base_img.copy()
    height, width = frame.shape[:2]
    agent_px = world_to_pixel(agent_pos, extent, (height, width))
    draw_disk(frame, agent_px, max(2, int(min(height, width) * 0.025)), (240, 70, 70))
    return frame


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
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def get_dataset_defaults() -> dict[str, Any]:
    return {
        "dataset_root": DEFAULT_DATASET_ROOT,
        "dataset_repo_id": DEFAULT_DATASET_REPO_ID,
        "output_dir": DEFAULT_DATASET_ROOT,
        "num_episodes": DEFAULT_NUM_EPISODES,
        "fps": DEFAULT_FPS,
        "image_size": DEFAULT_IMAGE_SIZE,
    }


def get_train_defaults(policy_type: str) -> dict[str, Any]:
    return load_policy_mode_defaults("train", ENV_NAME, policy_type)


def get_eval_defaults(policy_type: str) -> dict[str, Any]:
    return load_policy_mode_defaults("eval", ENV_NAME, policy_type)


class HShape2DEnv:
    def __init__(
        self,
        seed: int,
        success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    ) -> None:
        self.grid, self.extent = create_h_shape_grid()
        self.corners = find_fixed_h_corners(self.grid, self.extent)
        self.success_threshold = float(success_threshold)
        self.rng = random.Random(seed)

        self.state: tuple[float, float] | None = None
        self.goal: tuple[float, float] | None = None
        self.trajectory: list[tuple[float, float]] = []
        self.step_count = 0
        self.task_id: int | None = None
        self.start_region_name: str | None = None
        self.target_goal_name: str | None = None
        self.last_info: dict[str, Any] = {}
        self.done = False

    def sample_task_id(self) -> int:
        return int(self.rng.choice(tuple(sorted(TASK_ID_TO_NAME))))

    def reset(self, task_id: int | None = None) -> tuple[float, float]:
        chosen_task_id = self.sample_task_id() if task_id is None else int(task_id)
        if chosen_task_id == 0:
            start = self.corners["upper_left"]
            goal = self.corners["upper_right"]
            self.start_region_name = "upper_left"
            self.target_goal_name = "upper_right"
        elif chosen_task_id == 1:
            start = self.corners["lower_left"]
            goal = self.corners["lower_right"]
            self.start_region_name = "lower_left"
            self.target_goal_name = "lower_right"
        else:
            raise ValueError(f"Unsupported h_shape task_id={chosen_task_id}")

        self.state = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))
        self.trajectory = [self.state]
        self.step_count = 0
        self.task_id = chosen_task_id
        self.done = False
        self.last_info = {
            "task_id": chosen_task_id,
            "task_name": TASK_ID_TO_NAME[chosen_task_id],
            "start_region_name": self.start_region_name,
            "target_goal_name": self.target_goal_name,
            "success": False,
            "collision_rejected": False,
            "reached_goal": False,
        }
        return self.state

    def get_phase_name(self, state: tuple[float, float]) -> str:
        x, y = state
        if self.goal is not None and np.linalg.norm(np.asarray(state) - np.asarray(self.goal)) < self.success_threshold:
            return "goal_region"
        if x < 0.0 and y >= 0.0:
            return "upper_left_corridor"
        if x > 0.0 and y >= 0.0:
            return "upper_right_corridor"
        if x < 0.0 and y < 0.0:
            return "lower_left_corridor"
        if x > 0.0 and y < 0.0:
            return "lower_right_corridor"
        return "middle_corridor"

    def render_base_image(self, image_hw: tuple[int, int]) -> np.ndarray:
        return make_base_image(self.grid, image_hw)

    def render_frame(
        self,
        base_img: np.ndarray,
        state_xy: tuple[float, float],
    ) -> np.ndarray:
        return render_frame(base_img, self.extent, state_xy)

    def step(
        self,
        action: tuple[float, float],
    ) -> tuple[tuple[float, float], float, bool, dict[str, Any]]:
        if self.state is None or self.goal is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() to start a new one.")

        dx, dy = (float(action[0]), float(action[1]))
        proposed = (self.state[0] + dx, self.state[1] + dy)
        collision_rejected = False
        used_half_step = False

        if segment_collision_free(
            self.state,
            proposed,
            self.grid,
            self.extent,
        ):
            next_state = proposed
        else:
            proposed_half = (self.state[0] + dx * 0.5, self.state[1] + dy * 0.5)
            if segment_collision_free(
                self.state,
                proposed_half,
                self.grid,
                self.extent,
            ):
                next_state = proposed_half
                used_half_step = True
            else:
                next_state = self.state
                collision_rejected = True

        distance_to_goal = float(
            np.linalg.norm(np.asarray(next_state) - np.asarray(self.goal))
        )
        success = distance_to_goal < self.success_threshold
        reward = 1.0 if success else -distance_to_goal

        self.state = (float(next_state[0]), float(next_state[1]))
        self.trajectory.append(self.state)
        self.step_count += 1
        self.done = success
        info = {
            "task_id": self.task_id,
            "task_name": None if self.task_id is None else TASK_ID_TO_NAME[self.task_id],
            "start_region_name": self.start_region_name,
            "target_goal_name": self.target_goal_name,
            "step_count": self.step_count,
            "collision_rejected": collision_rejected,
            "used_half_step": used_half_step,
            "applied_state": self.state,
            "distance_to_goal": distance_to_goal,
            "phase_name": self.get_phase_name(self.state),
            "reached_goal": success,
            "success": success,
        }
        self.last_info = info
        return self.state, reward, self.done, info

    @staticmethod
    def build_task_schedule(num_rollouts: int, _seed: int) -> list[int]:
        return [int(rollout_index % 2) for rollout_index in range(num_rollouts)]


def _compute_online_signature(
    state_history: deque[np.ndarray],
    history_length: int,
    sig_depth: int,
    signature_backend: str,
) -> np.ndarray:
    if history_length <= 0:
        raise ValueError(f"history_length must be > 0, got {history_length}")
    if len(state_history) == 0:
        raise ValueError("state_history is empty; cannot compute path signature.")

    window = np.stack(list(state_history), axis=0).astype(np.float32, copy=False)
    if window.shape[0] < history_length:
        pad_len = history_length - window.shape[0]
        pad = np.repeat(window[:1], pad_len, axis=0)
        window = np.concatenate([pad, window], axis=0)

    if signature_backend == "signatory":
        return compute_signatory_signature_np(window, sig_depth)
    return compute_simple_signature_np(window, sig_depth)


def evaluate_policy(
    *,
    policy_type: str,
    args,
    policy,
    cfg,
    preprocessor,
    postprocessor,
    policy_dir: Path,
) -> None:
    import torch

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_key, image_shape = resolve_single_visual_observation_feature(cfg)
    image_hw = (int(image_shape[1]), int(image_shape[2]))
    if bool(getattr(cfg, "use_first_frame_anchor", False)):
        raise NotImplementedError(
            "First-frame anchor evaluation is not implemented for `h_shape` yet. "
            "Only `braidedhub` currently supports the shared-backbone raw-anchor path."
        )

    if cfg.robot_state_feature is None:
        raise RuntimeError("Policy has no observation.state feature.")
    state_key = "observation.state"
    state_dim = int(cfg.robot_state_feature.shape[0])

    use_path_signature = bool(
        policy_type == "streaming_act" and getattr(cfg, "use_path_signature", False)
    )
    use_prefix_sequence_training = bool(
        policy_type == "streaming_act"
        and getattr(cfg, "use_prefix_sequence_training", False)
    )
    use_visual_prefix_memory = bool(
        policy_type == "streaming_act"
        and getattr(cfg, "use_visual_prefix_memory", False)
    )
    use_delta_signature = bool(
        policy_type == "streaming_act" and getattr(cfg, "use_delta_signature", False)
    )
    build_explicit_prefix_eval_inputs = (
        use_prefix_sequence_training and not use_visual_prefix_memory
    )
    signature_key = DEFAULT_PATH_SIGNATURE_KEY
    signature_backend = None
    if use_path_signature:
        signature_backend = resolve_signature_backend(args.signature_backend)
        if int(cfg.history_length) <= 0:
            raise ValueError(f"Invalid cfg.history_length={cfg.history_length}. Must be > 0.")
        if int(cfg.signature_depth) <= 0:
            raise ValueError(f"Invalid cfg.signature_depth={cfg.signature_depth}. Must be > 0.")
        if int(cfg.signature_dim) <= 0:
            raise ValueError(
                f"Invalid cfg.signature_dim={cfg.signature_dim}. "
                "Expected positive signature dimension when use_path_signature=True."
            )
        print(
            "[info] online path-signature enabled: "
            f"backend={signature_backend}, history={cfg.history_length}, "
            f"depth={cfg.signature_depth}, dim={cfg.signature_dim}"
        )
    if use_delta_signature:
        print(
            "[info] online delta-signature enabled: "
            f"key={DEFAULT_DELTA_SIGNATURE_KEY}, rule=g_t-g_(t-1), first_step=zeros"
        )
    if build_explicit_prefix_eval_inputs:
        print(
            "[info] online prefix-sequence enabled: "
            f"max_steps={cfg.prefix_train_max_steps}, stride={cfg.prefix_frame_stride}, "
            f"pad_value={cfg.prefix_pad_value}"
        )
    elif use_visual_prefix_memory:
        print(
            "[info] visual prefix memory online update enabled: "
            "rollout uses fixed-size recurrent memory without rebuilding "
            "explicit prefix-sequence tensors each step"
        )

    env = HShape2DEnv(seed=args.seed, success_threshold=args.success_threshold)
    base_img = env.render_base_image(image_hw)

    results = []
    success_count = 0

    for ep_idx, task_id in enumerate(env.build_task_schedule(args.num_rollouts, args.seed)):
        state_xy = tuple(float(v) for v in env.reset(task_id=task_id))
        if hasattr(policy, "reset"):
            policy.reset()

        episode_reward = 0.0
        video_path = output_dir / f"rollout_{ep_idx:03d}.mp4"
        writer = start_ffmpeg_raw_writer(
            video_path,
            image_hw[1],
            image_hw[0],
            args.fps,
        )
        if writer.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin for rollout video writing.")

        trajectory = [state_xy]
        state_history = (
            deque(maxlen=int(cfg.history_length)) if use_path_signature else None
        )
        prefix_state_history = [] if build_explicit_prefix_eval_inputs else None
        prefix_signature_history = (
            [] if build_explicit_prefix_eval_inputs and use_path_signature else None
        )
        prefix_delta_signature_history = (
            [] if build_explicit_prefix_eval_inputs and use_delta_signature else None
        )
        prefix_image_history = [] if build_explicit_prefix_eval_inputs else None
        previous_signature_vec = None
        success = False

        for _step_idx in range(args.max_steps):
            frame = env.render_frame(base_img, state_xy)
            writer.stdin.write(frame.astype(np.uint8).tobytes())

            obs = build_eval_observation(
                state_xy=state_xy,
                rgb_frame=frame,
                state_key=state_key,
                image_key=image_key,
                state_dim=state_dim,
            )

            if use_path_signature:
                assert state_history is not None
                state_now = (
                    obs[state_key].detach().cpu().numpy().astype(np.float32, copy=False)
                )
                state_history.append(state_now.copy())
                signature_vec = _compute_online_signature(
                    state_history=state_history,
                    history_length=int(cfg.history_length),
                    sig_depth=int(cfg.signature_depth),
                    signature_backend=str(signature_backend),
                )
                if signature_vec.shape[0] != int(cfg.signature_dim):
                    raise RuntimeError(
                        "Online signature dimension mismatch: "
                        f"got {signature_vec.shape[0]}, "
                        f"expected cfg.signature_dim={cfg.signature_dim}."
                    )
                obs[signature_key] = torch.from_numpy(
                    signature_vec.astype(np.float32, copy=False)
                )
                if use_delta_signature:
                    delta_signature_vec = compute_delta_signature_step_np(
                        signature_vec,
                        previous_signature_vec,
                    )
                    obs[DEFAULT_DELTA_SIGNATURE_KEY] = torch.from_numpy(
                        delta_signature_vec.astype(np.float32, copy=False)
                    )
                    previous_signature_vec = signature_vec.astype(np.float32, copy=True)

            if build_explicit_prefix_eval_inputs:
                assert prefix_state_history is not None
                assert prefix_image_history is not None
                if use_path_signature:
                    assert prefix_signature_history is not None
                if use_delta_signature:
                    assert prefix_delta_signature_history is not None
                build_prefix_sequence_eval_inputs(
                    obs=obs,
                    cfg=cfg,
                    state_key=state_key,
                    image_key=image_key,
                    signature_key=signature_key if use_path_signature else None,
                    delta_signature_key=(
                        DEFAULT_DELTA_SIGNATURE_KEY if use_delta_signature else None
                    ),
                    prefix_state_history=prefix_state_history,
                    prefix_signature_history=prefix_signature_history,
                    prefix_delta_signature_history=prefix_delta_signature_history,
                    prefix_image_history=prefix_image_history,
                )

            obs = preprocessor(obs)
            if use_path_signature:
                if signature_key not in obs:
                    raise KeyError(
                        f"`{signature_key}` missing after preprocessor; "
                        "cannot run policy with use_path_signature=True."
                    )
                path_signature = obs[signature_key]
                if path_signature.ndim == 1:
                    path_signature = path_signature.unsqueeze(0)
                elif path_signature.ndim != 2:
                    raise RuntimeError(
                        f"`{signature_key}` must be 1D/2D after preprocessing, "
                        f"got shape={tuple(path_signature.shape)}"
                    )
                obs[signature_key] = path_signature.to(
                    device=obs[state_key].device,
                    dtype=obs[state_key].dtype,
                )
            if use_delta_signature:
                if DEFAULT_DELTA_SIGNATURE_KEY not in obs:
                    raise KeyError(
                        f"`{DEFAULT_DELTA_SIGNATURE_KEY}` missing after preprocessor; "
                        "cannot run policy with use_delta_signature=True."
                    )
                delta_signature = obs[DEFAULT_DELTA_SIGNATURE_KEY]
                if delta_signature.ndim == 1:
                    delta_signature = delta_signature.unsqueeze(0)
                elif delta_signature.ndim != 2:
                    raise RuntimeError(
                        f"`{DEFAULT_DELTA_SIGNATURE_KEY}` must be 1D/2D after preprocessing, "
                        f"got shape={tuple(delta_signature.shape)}"
                    )
                obs[DEFAULT_DELTA_SIGNATURE_KEY] = delta_signature.to(
                    device=obs[state_key].device,
                    dtype=obs[state_key].dtype,
                )
            if build_explicit_prefix_eval_inputs:
                ensure_prefix_sequence_batch_dims(
                    obs=obs,
                    state_key=state_key,
                    image_key=image_key,
                    use_path_signature=use_path_signature,
                    use_delta_signature=use_delta_signature,
                )

            with torch.no_grad():
                action = policy.select_action(obs)
            action = postprocessor(action)
            action_np = action.squeeze(0).detach().cpu().numpy()

            dx = float(action_np[0]) if action_np.shape[0] >= 1 else 0.0
            dy = float(action_np[1]) if action_np.shape[0] >= 2 else 0.0
            norm = math.sqrt(dx * dx + dy * dy)
            if norm > args.max_action_step and norm > 1e-8:
                scale = args.max_action_step / norm
                dx *= scale
                dy *= scale

            next_state, reward, done, info = env.step((dx, dy))
            state_xy = (float(next_state[0]), float(next_state[1]))
            trajectory.append(state_xy)
            episode_reward += float(reward)
            success = bool(info.get("success", False))

            if done:
                final_frame = env.render_frame(base_img, state_xy)
                writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                break

        writer.stdin.close()
        code = writer.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed on rollout {ep_idx} with exit code {code}")

        if success:
            success_count += 1

        final_distance = float(
            np.linalg.norm(np.asarray(state_xy) - np.asarray(env.goal, dtype=np.float64))
        )
        result = {
            "episode_index": ep_idx,
            "task_id": task_id,
            "task_name": TASK_ID_TO_NAME[task_id],
            "video_path": str(video_path),
            "start": [float(trajectory[0][0]), float(trajectory[0][1])],
            "goal": None
            if env.goal is None
            else [float(env.goal[0]), float(env.goal[1])],
            "final_position": [float(state_xy[0]), float(state_xy[1])],
            "final_distance": final_distance,
            "success": bool(success),
            "steps": int(len(trajectory) - 1),
            "sum_reward": float(episode_reward),
        }
        results.append(result)
        print(
            f"[{ep_idx + 1:03d}/{args.num_rollouts:03d}] "
            f"task={TASK_ID_TO_NAME[task_id]} success={success} "
            f"steps={result['steps']} final_dist={final_distance:.3f} "
            f"video={video_path.name}"
        )

    summary = {
        "env": ENV_NAME,
        "policy_type": policy_type,
        "num_rollouts": args.num_rollouts,
        "success_count": success_count,
        "success_rate": float(success_count / max(1, args.num_rollouts)),
        "seed": args.seed,
        "fps": args.fps,
        "max_steps": args.max_steps,
        "success_threshold": args.success_threshold,
        "policy_dir": str(policy_dir),
        "results": results,
    }
    summary_path = write_summary(output_dir, summary)

    print(f"\nSaved {args.num_rollouts} rollout videos to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(
        f"Success rate: {summary['success_rate']:.3f} ({success_count}/{args.num_rollouts})"
    )
