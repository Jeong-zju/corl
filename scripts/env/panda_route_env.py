from __future__ import annotations

from collections import deque
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from xml.etree import ElementTree as ET

import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover - optional dependency guard
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised only when mujoco is installed
    _MUJOCO_IMPORT_ERROR = None

from eval_helpers import (
    build_eval_observation,
    compute_signatory_signature_np,
    compute_simple_signature_np,
    resolve_signature_backend,
    write_summary,
)


ENV_NAME = "panda_route"
VIDEO_KEY = "observation.images.front"

DEFAULT_DATASET_ROOT = Path("data/zeno-ai/panda_route_onebit_cue_v30")
DEFAULT_DATASET_REPO_ID = "zeno-ai/panda_route_onebit_cue_v30"
DEFAULT_NUM_PER_TASK = 25
DEFAULT_NUM_ROLLOUTS = 20
DEFAULT_MAX_STEPS = 180
DEFAULT_FPS = 20
DEFAULT_IMAGE_SIZE = 128
DEFAULT_MAX_ACTION_STEP = 0.035
DEFAULT_RANDOM_SEED = 17
DEFAULT_T_FIXED = 100
DEFAULT_LAST_ACTION_MODE = "zero"
DEFAULT_INCLUDE_PATH_SIGNATURES = True
DEFAULT_PATH_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_SIGNATURE_WINDOW_SIZE = 0
DEFAULT_SIGNATURE_DEPTH = 3
DEFAULT_SIGNATURE_BACKEND = "auto"
DEFAULT_VIDEO_FPS = 20
DEFAULT_VIDEO_IMAGE_SIZE = 128
DEFAULT_LEROBOT_EPISODES_PER_CHUNK = 1000
DEFAULT_RAW_OUTPUT = None
DEFAULT_PROCESSED_OUTPUT = None
DEFAULT_SOLVE_TIME = 1.0
DEFAULT_STEP_SIZE = 0.045
DEFAULT_CONNECT_TOLERANCE = 0.025
DEFAULT_COLLISION_CHECK_RESOLUTION = 0.01
DEFAULT_GOAL_SAMPLE_PROBABILITY = 0.20
DEFAULT_MAX_ITERATIONS = 10_000
DEFAULT_RETRIES_PER_DEMO = 5
DEFAULT_LOW_SUCCESS_WARNING_THRESHOLD = 0.80
DEFAULT_STEP_PENALTY = -0.01
DEFAULT_GOAL_REWARD = 1.0
DEFAULT_START_RANDOMIZE = False
DEFAULT_PANDA_DESCRIPTION_NAME = "panda_mj_description"
DEFAULT_PANDA_DESCRIPTION_VARIANT = "panda_nohand"

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MUJOCO_MENAGERIE_ROOT = WORKSPACE_ROOT / "mujoco_menagerie"
LOCAL_FRANKA_PANDA_ROOT = LOCAL_MUJOCO_MENAGERIE_ROOT / "franka_emika_panda"
ROBOT_DESCRIPTIONS_CACHE_ROOT = WORKSPACE_ROOT / ".robot_descriptions_cache"
FRANKA_ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
LEGACY_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
FRANKA_FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
FRANKA_HOME_QPOS = np.asarray(
    [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
    dtype=np.float64,
)
LEGACY_HOME_QPOS = np.asarray(
    [0.0, 0.4, 0.0, -1.8, 0.0, 1.6, 0.0],
    dtype=np.float64,
)
FRANKA_OPEN_GRIPPER_QPOS = 0.04

TASK_ID_TO_START_NAME = {
    0: "S1",
    1: "S2",
}
TASK_ID_TO_BRANCH_NAME = {
    0: "H1",
    1: "L1",
}
TASK_ID_TO_GOAL_NAME = {
    0: "G1",
    1: "G2",
}
TASK_DESCRIPTION_BY_ID = {
    0: "Start from the upper-left region S1, merge through M1, pass the upper hole H1, merge through M2, and finish in the upper-right goal G1.",
    1: "Start from the lower-left region S2, merge through M1, pass the lower hole L1, merge through M2, and finish in the lower-right goal G2.",
}
TASK_ID_VALUES = tuple(sorted(TASK_ID_TO_GOAL_NAME))
TASK_INDEX_BY_ID = {task_id: index for index, task_id in enumerate(TASK_ID_VALUES)}
TASK_COLOR_BY_ID = {
    0: "#1b9e77",
    1: "#d95f02",
}
PHASE_LABEL_VOCAB = (
    "S1",
    "S2",
    "start_room",
    "M1",
    "middle_shared_region",
    "H1",
    "L1",
    "post_branch_shared_region",
    "M2",
    "G1",
    "G2",
    "free_space_other",
    "obstacle",
    "out_of_bounds",
)
PHASE_NAME_TO_ID = {
    phase_name: phase_index for phase_index, phase_name in enumerate(PHASE_LABEL_VOCAB)
}

SCHEMATIC_BACKGROUND_COLOR = (244, 239, 228)
SCHEMATIC_OBSTACLE_COLOR = (68, 68, 68)
SCHEMATIC_START_COLOR = (76, 120, 168)
SCHEMATIC_SHARED_COLOR = (120, 120, 120)
SCHEMATIC_H1_COLOR = (44, 160, 44)
SCHEMATIC_L1_COLOR = (148, 103, 189)
SCHEMATIC_GOAL_COLOR_BY_NAME = {
    "G1": (27, 158, 119),
    "G2": (217, 95, 2),
}
SCHEMATIC_CURSOR_OUTER = (255, 255, 255)
SCHEMATIC_CURSOR_INNER = (240, 70, 70)

_TRAIN_DEFAULTS = {
    "act": {
        "output_root": Path("outputs/train/panda_route_act"),
        "job_name": "act_panda_route",
        "wandb_project": "lerobot-panda-route-act",
        "eval_output_dir": Path("outputs/eval/panda_route_act"),
    },
    "streaming_act": {
        "output_root": Path("outputs/train/panda_route_streaming_act"),
        "job_name": "streaming_act_panda_route",
        "wandb_project": "lerobot-panda-route-streaming-act",
        "eval_output_dir": Path("outputs/eval/panda_route_streaming_act"),
    },
}


@dataclass(frozen=True)
class RectangleRegion:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmin >= self.xmax:
            raise ValueError(f"{self.name}: xmin must be smaller than xmax.")
        if self.ymin >= self.ymax:
            raise ValueError(f"{self.name}: ymin must be smaller than ymax.")

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass(frozen=True)
class RectangleObstacle(RectangleRegion):
    pass


@dataclass(frozen=True)
class GoalRegion(RectangleRegion):
    pass


@dataclass(frozen=True)
class SemanticRegion:
    name: str
    rectangles: tuple[RectangleRegion, ...]

    def contains_point(self, x: float, y: float) -> bool:
        return any(rectangle.contains_point(x, y) for rectangle in self.rectangles)


@dataclass(frozen=True)
class MapConfig:
    workspace: RectangleRegion
    start_room: SemanticRegion
    task_start_regions: tuple[RectangleRegion, ...]
    merge_region_m1: SemanticRegion
    middle_shared_region: SemanticRegion
    upper_hole_region: SemanticRegion
    lower_hole_region: SemanticRegion
    post_branch_shared_region: SemanticRegion
    merge_region_m2: SemanticRegion
    goal_regions: tuple[GoalRegion, ...]
    terminal_regions: tuple[GoalRegion, ...]
    free_space_rectangles: tuple[RectangleRegion, ...]
    obstacle_rectangles: tuple[RectangleObstacle, ...]


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    task_code: str
    start_region_name: str
    branch_region_name: str
    target_goal_name: str


@dataclass(frozen=True)
class TrajectoryPhaseAnnotation:
    phase_labels: tuple[str, ...]
    first_m1_index: int | None
    first_branch_index: int | None
    first_m2_index: int | None
    first_terminal_index: int | None
    first_terminal_phase: str | None


@dataclass(slots=True)
class DemonstrationEpisode:
    episode_id: int
    task_id: int
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    target_goal_name: str
    path_xy: np.ndarray
    path_length: int
    success: bool


@dataclass(slots=True)
class DemonstrationDataset:
    episodes: list[DemonstrationEpisode]
    seed: int
    num_per_task_requested: int
    success_counts_by_task: dict[int, int]
    attempt_counts_by_task: dict[int, int]
    skipped_counts_by_task: dict[int, int]
    solve_time: float
    retries_per_demo: int

    def __len__(self) -> int:
        return len(self.episodes)


@dataclass(slots=True)
class ProcessedDemonstrationDataset:
    observations: np.ndarray
    actions: np.ndarray
    path_signatures: np.ndarray | None
    task_ids: np.ndarray
    task_code_bits: np.ndarray
    goal_onehot: np.ndarray
    phase_labels: np.ndarray | None
    episode_ids: np.ndarray
    target_goal_names: np.ndarray
    start_xy: np.ndarray
    goal_xy: np.ndarray
    raw_path_lengths: np.ndarray
    raw_path_distances: np.ndarray
    success: np.ndarray
    seed: int
    t_fixed: int
    action_padding_mode: str
    path_signature_key: str | None
    path_signature_window_size: int
    path_signature_depth: int
    path_signature_backend: str | None
    phase_label_vocab: tuple[str, ...]
    source_num_per_task_requested: int
    source_solve_time: float
    source_retries_per_demo: int

    def __len__(self) -> int:
        return int(self.observations.shape[0])


@dataclass(slots=True)
class ReplayEpisode:
    replay_index: int
    source_episode_id: int
    task_id: int
    target_goal_name: str
    states_xy: np.ndarray
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    success: bool
    source_format: str
    phase_labels: tuple[str, ...] | None = None
    raw_path_length: int | None = None

    def __len__(self) -> int:
        return int(self.states_xy.shape[0])


@dataclass(slots=True)
class ReplayDataset:
    dataset_path: Path
    format_version: str
    source_kind: str
    episodes: list[ReplayEpisode]

    def __len__(self) -> int:
        return len(self.episodes)


class SupportsUniform(Protocol):
    def uniform(self, a: float, b: float) -> float: ...


def build_obstacles_from_free_rectangles(
    workspace: RectangleRegion,
    free_rectangles: tuple[RectangleRegion, ...],
) -> tuple[RectangleObstacle, ...]:
    x_coords = sorted(
        {
            workspace.xmin,
            workspace.xmax,
            *(rect.xmin for rect in free_rectangles),
            *(rect.xmax for rect in free_rectangles),
        }
    )
    y_coords = sorted(
        {
            workspace.ymin,
            workspace.ymax,
            *(rect.ymin for rect in free_rectangles),
            *(rect.ymax for rect in free_rectangles),
        }
    )

    cell_free: list[tuple[float, float, float, float]] = []
    cell_obstacles: list[tuple[float, float, float, float]] = []
    for x0, x1 in zip(x_coords[:-1], x_coords[1:], strict=False):
        for y0, y1 in zip(y_coords[:-1], y_coords[1:], strict=False):
            if x1 <= x0 or y1 <= y0:
                continue
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            if any(rect.contains_point(cx, cy) for rect in free_rectangles):
                cell_free.append((x0, x1, y0, y1))
            else:
                cell_obstacles.append((x0, x1, y0, y1))

    merged: list[list[float]] = []
    for x0, x1, y0, y1 in cell_obstacles:
        merged_existing = False
        for rect in merged:
            same_x = math.isclose(rect[0], x0) and math.isclose(rect[1], x1)
            touching_y = math.isclose(rect[3], y0) or math.isclose(rect[2], y1)
            overlap_y = not (y1 < rect[2] or y0 > rect[3])
            if same_x and (touching_y or overlap_y):
                rect[2] = min(rect[2], y0)
                rect[3] = max(rect[3], y1)
                merged_existing = True
                break
        if not merged_existing:
            merged.append([x0, x1, y0, y1])

    merged_second_pass: list[list[float]] = []
    for x0, x1, y0, y1 in merged:
        merged_existing = False
        for rect in merged_second_pass:
            same_y = math.isclose(rect[2], y0) and math.isclose(rect[3], y1)
            touching_x = math.isclose(rect[1], x0) or math.isclose(rect[0], x1)
            overlap_x = not (x1 < rect[0] or x0 > rect[1])
            if same_y and (touching_x or overlap_x):
                rect[0] = min(rect[0], x0)
                rect[1] = max(rect[1], x1)
                merged_existing = True
                break
        if not merged_existing:
            merged_second_pass.append([x0, x1, y0, y1])

    obstacles = tuple(
        RectangleObstacle(
            name=f"obstacle_{index:02d}",
            xmin=float(x0),
            xmax=float(x1),
            ymin=float(y0),
            ymax=float(y1),
        )
        for index, (x0, x1, y0, y1) in enumerate(merged_second_pass)
    )
    return obstacles


def build_task_spec(task_id: int) -> TaskSpec:
    if task_id not in TASK_ID_TO_GOAL_NAME:
        raise ValueError(f"Unsupported task_id={task_id}.")
    return TaskSpec(
        task_id=int(task_id),
        task_code=f"T{task_id}",
        start_region_name=TASK_ID_TO_START_NAME[task_id],
        branch_region_name=TASK_ID_TO_BRANCH_NAME[task_id],
        target_goal_name=TASK_ID_TO_GOAL_NAME[task_id],
    )


def build_default_map_config() -> MapConfig:
    workspace = RectangleRegion("workspace", 0.12, 0.78, -0.28, 0.28)
    start_room_rect = RectangleRegion("start_room_rect", 0.18, 0.31, -0.24, 0.24)
    s1 = RectangleRegion("S1", 0.18, 0.25, 0.10, 0.20)
    s2 = RectangleRegion("S2", 0.18, 0.25, -0.20, -0.10)
    m1 = RectangleRegion("M1", 0.31, 0.39, -0.05, 0.05)
    middle_shared = RectangleRegion("middle_shared_region_rect", 0.39, 0.43, -0.24, 0.24)
    h1 = RectangleRegion("H1", 0.43, 0.51, 0.10, 0.18)
    l1 = RectangleRegion("L1", 0.43, 0.51, -0.18, -0.10)
    post_branch_shared = RectangleRegion(
        "post_branch_shared_region_rect",
        0.51,
        0.55,
        -0.24,
        0.24,
    )
    m2 = RectangleRegion("M2", 0.55, 0.63, -0.05, 0.05)
    goal_room = RectangleRegion("goal_room_rect", 0.63, 0.74, -0.24, 0.24)
    g1 = GoalRegion("G1", 0.66, 0.74, 0.10, 0.20)
    g2 = GoalRegion("G2", 0.66, 0.74, -0.20, -0.10)

    free_space_rectangles = (
        start_room_rect,
        m1,
        middle_shared,
        h1,
        l1,
        post_branch_shared,
        m2,
        goal_room,
    )
    obstacle_rectangles = build_obstacles_from_free_rectangles(
        workspace=workspace,
        free_rectangles=free_space_rectangles,
    )
    return MapConfig(
        workspace=workspace,
        start_room=SemanticRegion("start_room", (start_room_rect,)),
        task_start_regions=(s1, s2),
        merge_region_m1=SemanticRegion("M1", (m1,)),
        middle_shared_region=SemanticRegion("middle_shared_region", (middle_shared,)),
        upper_hole_region=SemanticRegion("H1", (h1,)),
        lower_hole_region=SemanticRegion("L1", (l1,)),
        post_branch_shared_region=SemanticRegion(
            "post_branch_shared_region",
            (post_branch_shared,),
        ),
        merge_region_m2=SemanticRegion("M2", (m2,)),
        goal_regions=(g1, g2),
        terminal_regions=(g1, g2),
        free_space_rectangles=free_space_rectangles,
        obstacle_rectangles=obstacle_rectangles,
    )


def get_task_start_region(task_id: int, config: MapConfig | None = None) -> RectangleRegion:
    resolved_config = build_default_map_config() if config is None else config
    if task_id == 0:
        return resolved_config.task_start_regions[0]
    if task_id == 1:
        return resolved_config.task_start_regions[1]
    raise ValueError(f"Unsupported task_id={task_id}.")


def get_branch_region(task_id: int, config: MapConfig | None = None) -> SemanticRegion:
    resolved_config = build_default_map_config() if config is None else config
    if task_id == 0:
        return resolved_config.upper_hole_region
    if task_id == 1:
        return resolved_config.lower_hole_region
    raise ValueError(f"Unsupported task_id={task_id}.")


def get_goal_region_for_task(task_id: int, config: MapConfig | None = None) -> GoalRegion:
    resolved_config = build_default_map_config() if config is None else config
    goal_name = TASK_ID_TO_GOAL_NAME[task_id]
    for goal_region in resolved_config.goal_regions:
        if goal_region.name == goal_name:
            return goal_region
    raise RuntimeError(f"Goal region {goal_name} not found.")


def get_goal_region_for_state(
    x: float,
    y: float,
    config: MapConfig | None = None,
) -> GoalRegion | None:
    resolved_config = build_default_map_config() if config is None else config
    for goal_region in resolved_config.goal_regions:
        if goal_region.contains_point(x, y):
            return goal_region
    return None


def is_state_valid(
    x: float,
    y: float,
    config: MapConfig | None = None,
) -> bool:
    resolved_config = build_default_map_config() if config is None else config
    if not resolved_config.workspace.contains_point(x, y):
        return False
    return any(rect.contains_point(x, y) for rect in resolved_config.free_space_rectangles)


def sample_valid_state_in_region(
    region: RectangleRegion,
    rng: SupportsUniform,
    config: MapConfig | None = None,
    max_attempts: int = 2000,
) -> tuple[float, float]:
    resolved_config = build_default_map_config() if config is None else config
    for _ in range(max_attempts):
        x = float(rng.uniform(region.xmin, region.xmax))
        y = float(rng.uniform(region.ymin, region.ymax))
        if is_state_valid(x, y, config=resolved_config):
            return (x, y)
    raise RuntimeError(f"Could not sample a valid state in region {region.name}.")


def sample_free_state(
    rng: SupportsUniform,
    config: MapConfig | None = None,
    max_attempts: int = 5000,
) -> tuple[float, float]:
    resolved_config = build_default_map_config() if config is None else config
    for _ in range(max_attempts):
        x = float(rng.uniform(resolved_config.workspace.xmin, resolved_config.workspace.xmax))
        y = float(rng.uniform(resolved_config.workspace.ymin, resolved_config.workspace.ymax))
        if is_state_valid(x, y, config=resolved_config):
            return (x, y)
    raise RuntimeError("Could not sample a free state in the workspace.")


def get_phase_name(
    x: float,
    y: float,
    config: MapConfig | None = None,
) -> str:
    resolved_config = build_default_map_config() if config is None else config
    if not resolved_config.workspace.contains_point(x, y):
        return "out_of_bounds"
    if not is_state_valid(x, y, config=resolved_config):
        return "obstacle"
    if resolved_config.task_start_regions[0].contains_point(x, y):
        return "S1"
    if resolved_config.task_start_regions[1].contains_point(x, y):
        return "S2"
    for goal_region in resolved_config.goal_regions:
        if goal_region.contains_point(x, y):
            return goal_region.name
    if resolved_config.merge_region_m1.contains_point(x, y):
        return "M1"
    if resolved_config.upper_hole_region.contains_point(x, y):
        return "H1"
    if resolved_config.lower_hole_region.contains_point(x, y):
        return "L1"
    if resolved_config.merge_region_m2.contains_point(x, y):
        return "M2"
    if resolved_config.middle_shared_region.contains_point(x, y):
        return "middle_shared_region"
    if resolved_config.post_branch_shared_region.contains_point(x, y):
        return "post_branch_shared_region"
    if resolved_config.start_room.contains_point(x, y):
        return "start_room"
    return "free_space_other"


def _first_index_of_phase(
    phase_labels: tuple[str, ...],
    target_phase: str,
) -> int | None:
    for index, phase_label in enumerate(phase_labels):
        if phase_label == target_phase:
            return index
    return None


def annotate_trajectory_phases(
    path: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    config: MapConfig | None = None,
) -> TrajectoryPhaseAnnotation:
    resolved_config = build_default_map_config() if config is None else config
    phase_labels = tuple(get_phase_name(x, y, config=resolved_config) for x, y in path)
    terminal_phase_names = {goal_region.name for goal_region in resolved_config.goal_regions}
    first_terminal_index = None
    first_terminal_phase = None
    for index, phase_label in enumerate(phase_labels):
        if phase_label in terminal_phase_names:
            first_terminal_index = index
            first_terminal_phase = phase_label
            break
    branch_index = _first_index_of_phase(phase_labels, "H1")
    if branch_index is None:
        branch_index = _first_index_of_phase(phase_labels, "L1")
    return TrajectoryPhaseAnnotation(
        phase_labels=phase_labels,
        first_m1_index=_first_index_of_phase(phase_labels, "M1"),
        first_branch_index=branch_index,
        first_m2_index=_first_index_of_phase(phase_labels, "M2"),
        first_terminal_index=first_terminal_index,
        first_terminal_phase=first_terminal_phase,
    )


def _workspace_to_pixel(
    point_xy: tuple[float, float],
    config: MapConfig,
    image_size: tuple[int, int],
) -> tuple[int, int]:
    x, y = point_xy
    height, width = image_size
    workspace = config.workspace
    px = int(
        round(
            (x - workspace.xmin)
            / max(workspace.xmax - workspace.xmin, 1e-8)
            * (width - 1)
        )
    )
    py = int(
        round(
            (workspace.ymax - y)
            / max(workspace.ymax - workspace.ymin, 1e-8)
            * (height - 1)
        )
    )
    return int(np.clip(px, 0, width - 1)), int(np.clip(py, 0, height - 1))


def _fill_rectangle(
    image: np.ndarray,
    region: RectangleRegion | SemanticRegion,
    color: tuple[int, int, int],
    config: MapConfig,
) -> None:
    if isinstance(region, SemanticRegion):
        for rectangle in region.rectangles:
            _fill_rectangle(image, rectangle, color, config=config)
        return
    top_left = _workspace_to_pixel(
        (region.xmin, region.ymax),
        config=config,
        image_size=image.shape[:2],
    )
    bottom_right = _workspace_to_pixel(
        (region.xmax, region.ymin),
        config=config,
        image_size=image.shape[:2],
    )
    x0, x1 = sorted((top_left[0], bottom_right[0]))
    y0, y1 = sorted((top_left[1], bottom_right[1]))
    image[y0 : y1 + 1, x0 : x1 + 1] = np.asarray(color, dtype=np.uint8)


def _draw_disk(
    image: np.ndarray,
    center_px: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    cx, cy = center_px
    height, width = image.shape[:2]
    y_coords, x_coords = np.ogrid[:height, :width]
    mask = (x_coords - cx) ** 2 + (y_coords - cy) ** 2 <= radius * radius
    image[mask] = np.asarray(color, dtype=np.uint8)


def make_schematic_base_image(
    config: MapConfig,
    image_size: int = DEFAULT_VIDEO_IMAGE_SIZE,
) -> np.ndarray:
    base_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    base_image[:] = np.asarray(SCHEMATIC_BACKGROUND_COLOR, dtype=np.uint8)
    for obstacle in config.obstacle_rectangles:
        _fill_rectangle(base_image, obstacle, SCHEMATIC_OBSTACLE_COLOR, config=config)
    _fill_rectangle(base_image, config.start_room, SCHEMATIC_START_COLOR, config=config)
    _fill_rectangle(base_image, config.merge_region_m1, SCHEMATIC_SHARED_COLOR, config=config)
    _fill_rectangle(base_image, config.middle_shared_region, SCHEMATIC_SHARED_COLOR, config=config)
    _fill_rectangle(
        base_image,
        config.post_branch_shared_region,
        SCHEMATIC_SHARED_COLOR,
        config=config,
    )
    _fill_rectangle(base_image, config.merge_region_m2, SCHEMATIC_SHARED_COLOR, config=config)
    _fill_rectangle(base_image, config.upper_hole_region, SCHEMATIC_H1_COLOR, config=config)
    _fill_rectangle(base_image, config.lower_hole_region, SCHEMATIC_L1_COLOR, config=config)
    for goal_region in config.goal_regions:
        _fill_rectangle(
            base_image,
            goal_region,
            SCHEMATIC_GOAL_COLOR_BY_NAME[goal_region.name],
            config=config,
        )
    return base_image


def render_schematic_frame(
    base_image: np.ndarray,
    config: MapConfig,
    probe_xy: tuple[float, float],
) -> np.ndarray:
    frame = base_image.copy()
    height, width = frame.shape[:2]
    probe_px = _workspace_to_pixel(probe_xy, config=config, image_size=(height, width))
    outer_radius = max(3, int(min(height, width) * 0.03))
    inner_radius = max(2, int(outer_radius * 0.65))
    _draw_disk(frame, probe_px, outer_radius, SCHEMATIC_CURSOR_OUTER)
    _draw_disk(frame, probe_px, inner_radius, SCHEMATIC_CURSOR_INNER)
    return frame


class PandaRouteSemanticEnv:
    def __init__(
        self,
        map_config: MapConfig | None = None,
        rng_seed: int = DEFAULT_RANDOM_SEED,
        enable_randomize: bool = DEFAULT_START_RANDOMIZE,
        step_penalty: float = DEFAULT_STEP_PENALTY,
        goal_reward: float = DEFAULT_GOAL_REWARD,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.map_config = build_default_map_config() if map_config is None else map_config
        self.rng = random.Random(rng_seed)
        self.enable_randomize = bool(enable_randomize)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.image_size = int(image_size)
        self.state: tuple[float, float] | None = None
        self.step_count = 0
        self.done = False
        self.task_spec: TaskSpec | None = None
        self.start_region_name: str | None = None
        self.target_goal_name: str | None = None
        self.trajectory: list[tuple[float, float]] = []
        self.last_info: dict[str, Any] = {}
        self._base_image = make_schematic_base_image(
            self.map_config,
            image_size=max(self.image_size, 32),
        )

    def sample_task_id(self) -> int:
        return int(self.rng.choice(TASK_ID_VALUES))

    def _resolve_start_state(
        self,
        task_start_region: RectangleRegion,
        enable_randomize: bool,
        explicit_start_state: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        if explicit_start_state is not None:
            x = float(explicit_start_state[0])
            y = float(explicit_start_state[1])
            if not task_start_region.contains_point(x, y):
                raise ValueError(
                    f"Explicit start_state {explicit_start_state} is outside task start region "
                    f"{task_start_region.name}."
                )
            if not is_state_valid(x, y, config=self.map_config):
                raise ValueError(f"Explicit start_state {explicit_start_state} is not valid.")
            return (x, y)
        if enable_randomize:
            return sample_valid_state_in_region(
                task_start_region,
                self.rng,
                config=self.map_config,
            )
        center_state = task_start_region.center
        if not is_state_valid(*center_state, config=self.map_config):
            raise RuntimeError(
                "The deterministic start-region center is not valid free space. "
                f"region={task_start_region.name}, center={center_state}"
            )
        return center_state

    def reset(
        self,
        task_id: int | None = None,
        enable_randomize: bool | None = None,
        start_state: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        chosen_task_id = self.sample_task_id() if task_id is None else int(task_id)
        self.task_spec = build_task_spec(chosen_task_id)
        self.start_region_name = self.task_spec.start_region_name
        self.target_goal_name = self.task_spec.target_goal_name
        task_start_region = get_task_start_region(chosen_task_id, self.map_config)
        use_randomized_start = (
            self.enable_randomize if enable_randomize is None else bool(enable_randomize)
        )
        self.state = self._resolve_start_state(
            task_start_region=task_start_region,
            enable_randomize=use_randomized_start,
            explicit_start_state=start_state,
        )
        self.trajectory = [self.state]
        self.step_count = 0
        self.done = False
        self.last_info = {
            "task_id": self.task_spec.task_id,
            "task_code": self.task_spec.task_code,
            "start_region_name": self.start_region_name,
            "branch_region_name": self.task_spec.branch_region_name,
            "target_goal_name": self.target_goal_name,
            "reached_goal": None,
            "success": False,
            "collision_rejected": False,
            "start_randomized": use_randomized_start,
        }
        return self.state

    def get_policy_state(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Call reset() before requesting policy state.")
        return np.asarray([self.state[0], self.state[1]], dtype=np.float32)

    def sync_to_state(
        self,
        state: tuple[float, float],
        step_count: int | None = None,
    ) -> tuple[float, float]:
        if self.task_spec is None or self.target_goal_name is None:
            raise RuntimeError("Call reset() before syncing state.")
        synced_state = (float(state[0]), float(state[1]))
        self.state = synced_state
        if step_count is not None:
            self.step_count = int(step_count)
        if self.trajectory:
            if step_count is not None:
                self.trajectory = self.trajectory[: max(0, int(step_count))]
            self.trajectory.append(synced_state)
        else:
            self.trajectory = [synced_state]
        reached_goal = get_goal_region_for_state(*synced_state, config=self.map_config)
        success = reached_goal is not None and reached_goal.name == self.target_goal_name
        self.done = reached_goal is not None
        self.last_info = {
            "task_id": self.task_spec.task_id,
            "task_code": self.task_spec.task_code,
            "start_region_name": self.start_region_name,
            "branch_region_name": self.task_spec.branch_region_name,
            "target_goal_name": self.target_goal_name,
            "step_count": self.step_count,
            "action": None,
            "proposed_state": synced_state,
            "applied_state": synced_state,
            "proposed_state_valid": is_state_valid(*synced_state, config=self.map_config),
            "collision_rejected": False,
            "reached_goal": None if reached_goal is None else reached_goal.name,
            "success": success,
            "partial_observation": self.get_partial_observation(),
            "full_observation": self.get_full_observation(),
        }
        return synced_state

    def get_partial_observation(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Call reset() before requesting observations.")
        return {
            "x": float(self.state[0]),
            "y": float(self.state[1]),
            "position": (float(self.state[0]), float(self.state[1])),
            "step_count": int(self.step_count),
            "phase_name": get_phase_name(*self.state, config=self.map_config),
        }

    def get_full_observation(self) -> dict[str, Any]:
        if self.state is None or self.task_spec is None:
            raise RuntimeError("Call reset() before requesting observations.")
        return {
            **self.get_partial_observation(),
            "task_id": self.task_spec.task_id,
            "task_code": self.task_spec.task_code,
            "start_region_name": self.task_spec.start_region_name,
            "branch_region_name": self.task_spec.branch_region_name,
            "target_goal_name": self.task_spec.target_goal_name,
        }

    def is_state_valid(self, state: tuple[float, float]) -> bool:
        return is_state_valid(*state, config=self.map_config)

    def get_goal_region(self, state: tuple[float, float]) -> GoalRegion | None:
        return get_goal_region_for_state(*state, config=self.map_config)

    def get_phase_name(self, state: tuple[float, float]) -> str:
        return get_phase_name(*state, config=self.map_config)

    def annotate_trajectory_phases(
        self,
        path: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    ) -> TrajectoryPhaseAnnotation:
        return annotate_trajectory_phases(path, config=self.map_config)

    def step(
        self,
        action: tuple[float, float],
    ) -> tuple[tuple[float, float], float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() to start a new one.")
        if self.task_spec is None or self.target_goal_name is None:
            raise RuntimeError("Call reset() before step().")

        dx = float(action[0])
        dy = float(action[1])
        proposed_state = (float(self.state[0] + dx), float(self.state[1] + dy))
        proposed_valid = is_state_valid(*proposed_state, config=self.map_config)
        collision_rejected = not proposed_valid
        next_state = self.state if collision_rejected else proposed_state
        reached_goal = get_goal_region_for_state(*next_state, config=self.map_config)
        success = reached_goal is not None and reached_goal.name == self.target_goal_name

        self.state = next_state
        self.trajectory.append(next_state)
        self.step_count += 1
        self.done = reached_goal is not None

        reward = self.step_penalty
        if success:
            reward += self.goal_reward

        info = {
            "task_id": self.task_spec.task_id,
            "task_code": self.task_spec.task_code,
            "start_region_name": self.start_region_name,
            "branch_region_name": self.task_spec.branch_region_name,
            "target_goal_name": self.target_goal_name,
            "step_count": self.step_count,
            "action": (dx, dy),
            "proposed_state": proposed_state,
            "applied_state": next_state,
            "proposed_state_valid": proposed_valid,
            "collision_rejected": collision_rejected,
            "reached_goal": None if reached_goal is None else reached_goal.name,
            "success": success,
            "partial_observation": self.get_partial_observation(),
            "full_observation": self.get_full_observation(),
        }
        self.last_info = info
        return next_state, reward, self.done, info

    def render_frame(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Call reset() before render_frame().")
        return render_schematic_frame(
            self._base_image,
            config=self.map_config,
            probe_xy=(float(self.state[0]), float(self.state[1])),
        )

    def render(self) -> np.ndarray:
        return self.render_frame()

    def close(self) -> None:
        return None


def _format_geom_rgba(color: tuple[float, float, float, float]) -> str:
    return " ".join(f"{float(channel):.6f}" for channel in color)


def _rectangle_to_mjcf_geom(
    rectangle: RectangleRegion,
    *,
    height: float,
    rgba: tuple[float, float, float, float],
    geom_type: str = "box",
    mass: float = 0.0,
    contype: int = 1,
    conaffinity: int = 1,
) -> str:
    center_x, center_y = rectangle.center
    size_x = (rectangle.xmax - rectangle.xmin) * 0.5
    size_y = (rectangle.ymax - rectangle.ymin) * 0.5
    return (
        f'<geom name="{rectangle.name}" type="{geom_type}" '
        f'pos="{center_x:.6f} {center_y:.6f} {height * 0.5:.6f}" '
        f'size="{size_x:.6f} {size_y:.6f} {height * 0.5:.6f}" '
        f'rgba="{_format_geom_rgba(rgba)}" mass="{mass:.6f}" '
        f'contype="{contype}" conaffinity="{conaffinity}"/>'
    )


def _normalize_panda_description_variant(variant: str | None) -> str:
    normalized = (
        DEFAULT_PANDA_DESCRIPTION_VARIANT
        if variant is None
        else str(variant).strip()
    )
    if normalized.endswith(".xml"):
        normalized = normalized[:-4]
    if not normalized:
        normalized = DEFAULT_PANDA_DESCRIPTION_VARIANT
    return normalized


def _ensure_local_robot_descriptions_cache() -> None:
    if "ROBOT_DESCRIPTIONS_CACHE" in os.environ:
        return
    if not LOCAL_MUJOCO_MENAGERIE_ROOT.is_dir():
        return
    cache_repo_path = ROBOT_DESCRIPTIONS_CACHE_ROOT / "mujoco_menagerie"
    ROBOT_DESCRIPTIONS_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not cache_repo_path.exists():
        try:
            cache_repo_path.symlink_to(
                LOCAL_MUJOCO_MENAGERIE_ROOT,
                target_is_directory=True,
            )
        except OSError:
            shutil.copytree(
                LOCAL_MUJOCO_MENAGERIE_ROOT,
                cache_repo_path,
                dirs_exist_ok=True,
            )
    os.environ["ROBOT_DESCRIPTIONS_CACHE"] = str(ROBOT_DESCRIPTIONS_CACHE_ROOT)


def get_franka_panda_mjcf_path(
    variant: str | None = DEFAULT_PANDA_DESCRIPTION_VARIANT,
) -> Path:
    normalized_variant = _normalize_panda_description_variant(variant)
    filename = "panda.xml" if normalized_variant == "panda" else f"{normalized_variant}.xml"
    _ensure_local_robot_descriptions_cache()
    try:
        from robot_descriptions import panda_mj_description
    except Exception:
        panda_package_path = None
    else:
        panda_package_path = Path(panda_mj_description.PACKAGE_PATH)
        candidate_path = panda_package_path / filename
        if candidate_path.is_file():
            return candidate_path.resolve()
    fallback_path = LOCAL_FRANKA_PANDA_ROOT / filename
    if fallback_path.is_file():
        return fallback_path.resolve()
    raise FileNotFoundError(
        "Could not locate the Franka Emika Panda MJCF description. "
        f"Tried variant={normalized_variant!r} via robot_descriptions and {fallback_path}."
    )


def load_franka_panda_model(
    variant: str | None = DEFAULT_PANDA_DESCRIPTION_VARIANT,
) -> "mujoco.MjModel":
    if mujoco is None:  # pragma: no cover
        raise RuntimeError(
            "mujoco is required to load the Franka Emika Panda model."
        ) from _MUJOCO_IMPORT_ERROR
    return mujoco.MjModel.from_xml_path(str(get_franka_panda_mjcf_path(variant)))


def _find_named_mjcf_element(
    root: ET.Element,
    tag: str,
    name: str,
) -> ET.Element | None:
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    return None


def _append_route_scene_assets(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if _find_named_mjcf_element(asset, "texture", "ground_tex") is None:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "ground_tex",
                "type": "2d",
                "builtin": "checker",
                "rgb1": "0.92 0.92 0.90",
                "rgb2": "0.84 0.84 0.82",
                "width": "512",
                "height": "512",
            },
        )
    if _find_named_mjcf_element(asset, "material", "ground_mat") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "ground_mat",
                "texture": "ground_tex",
                "texrepeat": "8 8",
                "specular": "0.0",
                "shininess": "0.0",
                "reflectance": "0.0",
            },
        )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    if visual.find("headlight") is None:
        ET.SubElement(
            visual,
            "headlight",
            {
                "ambient": "0.35 0.35 0.35",
                "diffuse": "0.85 0.85 0.85",
                "specular": "0.15 0.15 0.15",
            },
        )
    if visual.find("map") is None:
        ET.SubElement(visual, "map", {"znear": "0.001"})


def _append_route_probe_site(root: ET.Element) -> None:
    if _find_named_mjcf_element(root, "site", "probe_tip") is not None:
        return
    attachment_body = _find_named_mjcf_element(root, "body", "attachment")
    if attachment_body is not None:
        ET.SubElement(
            attachment_body,
            "site",
            {
                "name": "probe_tip",
                "type": "sphere",
                "pos": "0 0 0",
                "size": "0.010",
                "rgba": "0.90 0.15 0.15 1.0",
            },
        )
        return
    hand_body = _find_named_mjcf_element(root, "body", "hand")
    if hand_body is not None:
        ET.SubElement(
            hand_body,
            "site",
            {
                "name": "probe_tip",
                "type": "sphere",
                "pos": "0 0 0.103",
                "size": "0.010",
                "rgba": "0.90 0.15 0.15 1.0",
            },
        )
        return
    raise RuntimeError("Could not attach a probe_tip site to the Franka Panda MJCF.")


def _append_route_world_geoms(root: ET.Element, config: MapConfig) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Franka Panda MJCF does not define a worldbody.")
    if _find_named_mjcf_element(worldbody, "geom", "ground") is None:
        worldbody.insert(
            0,
            ET.Element(
                "geom",
                {
                    "name": "ground",
                    "type": "plane",
                    "size": "2 2 0.05",
                    "material": "ground_mat",
                    "contype": "1",
                    "conaffinity": "1",
                },
            ),
        )

    wall_height = 0.035
    marker_height = 0.002
    geoms = [
        _rectangle_to_mjcf_geom(
            config.task_start_regions[0],
            height=marker_height,
            rgba=(0.30, 0.47, 0.66, 0.55),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.task_start_regions[1],
            height=marker_height,
            rgba=(0.30, 0.47, 0.66, 0.55),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.merge_region_m1.rectangles[0],
            height=marker_height,
            rgba=(0.55, 0.55, 0.55, 0.45),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.upper_hole_region.rectangles[0],
            height=marker_height,
            rgba=(0.17, 0.62, 0.17, 0.55),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.lower_hole_region.rectangles[0],
            height=marker_height,
            rgba=(0.58, 0.40, 0.74, 0.55),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.merge_region_m2.rectangles[0],
            height=marker_height,
            rgba=(0.55, 0.55, 0.55, 0.45),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.goal_regions[0],
            height=marker_height,
            rgba=(0.10, 0.62, 0.47, 0.60),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.goal_regions[1],
            height=marker_height,
            rgba=(0.85, 0.37, 0.08, 0.60),
            contype=0,
            conaffinity=0,
        ),
    ]
    geoms.extend(
        _rectangle_to_mjcf_geom(
            obstacle,
            height=wall_height,
            rgba=(0.28, 0.28, 0.30, 1.0),
        )
        for obstacle in config.obstacle_rectangles
    )
    for geom_xml in geoms:
        geom = ET.fromstring(geom_xml)
        if _find_named_mjcf_element(worldbody, "geom", geom.get("name", "")) is None:
            worldbody.append(geom)


def build_panda_route_mjcf(
    config: MapConfig,
    robot_variant: str | None = DEFAULT_PANDA_DESCRIPTION_VARIANT,
) -> str:
    base_xml_path = get_franka_panda_mjcf_path(robot_variant)
    root = ET.fromstring(base_xml_path.read_text())
    root.set("model", "panda_route")
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((base_xml_path.parent / "assets").resolve()))
    _append_route_scene_assets(root)
    _append_route_probe_site(root)
    _append_route_world_geoms(root, config)
    return ET.tostring(root, encoding="unicode")


def _resolve_arm_joint_names(model: "mujoco.MjModel") -> tuple[str, ...]:
    for joint_names in (FRANKA_ARM_JOINT_NAMES, LEGACY_ARM_JOINT_NAMES):
        joint_ids = [
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            for joint_name in joint_names
        ]
        if all(joint_id >= 0 for joint_id in joint_ids):
            return joint_names
    raise RuntimeError("Could not resolve the Panda arm joints in the MuJoCo model.")


def _resolve_optional_qpos_indices(
    model: "mujoco.MjModel",
    joint_names: tuple[str, ...],
) -> np.ndarray:
    indices: list[int] = []
    for joint_name in joint_names:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if joint_id < 0:
            continue
        indices.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(indices, dtype=np.int64)


def _default_home_qpos_for_joint_names(joint_names: tuple[str, ...]) -> np.ndarray:
    if joint_names == FRANKA_ARM_JOINT_NAMES:
        return FRANKA_HOME_QPOS.copy()
    return LEGACY_HOME_QPOS.copy()


def _resolve_probe_site_name(model: "mujoco.MjModel") -> str:
    for site_name in ("probe_tip", "attachment_site"):
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name))
        if site_id >= 0:
            return site_name
    raise RuntimeError("Could not resolve a probe site in the MuJoCo model.")


class PandaRouteMjEnv(PandaRouteSemanticEnv):
    PROBE_TARGET_Z = 0.100
    IK_TOLERANCE = 5e-4
    IK_ACCEPT_RESIDUAL = 1e-2
    IK_MAX_ITERS = 240
    IK_DAMPING = 2e-2
    IK_REGULARIZATION = 1e-2
    RENDER_LOOKAT = np.asarray([0.45, 0.0, 0.24], dtype=np.float64)
    RENDER_DISTANCE = 1.35
    RENDER_AZIMUTH = 135.0
    RENDER_ELEVATION = -28.0

    def __init__(
        self,
        map_config: MapConfig | None = None,
        rng_seed: int = DEFAULT_RANDOM_SEED,
        enable_randomize: bool = DEFAULT_START_RANDOMIZE,
        step_penalty: float = DEFAULT_STEP_PENALTY,
        goal_reward: float = DEFAULT_GOAL_REWARD,
        image_size: int = DEFAULT_IMAGE_SIZE,
        robot_variant: str = DEFAULT_PANDA_DESCRIPTION_VARIANT,
    ) -> None:
        if mujoco is None:  # pragma: no cover
            raise RuntimeError(
                "mujoco is required for PandaRouteMjEnv. Install it first."
            ) from _MUJOCO_IMPORT_ERROR
        super().__init__(
            map_config=map_config,
            rng_seed=rng_seed,
            enable_randomize=enable_randomize,
            step_penalty=step_penalty,
            goal_reward=goal_reward,
            image_size=image_size,
        )
        self.robot_variant = _normalize_panda_description_variant(robot_variant)
        self.model = mujoco.MjModel.from_xml_string(
            build_panda_route_mjcf(
                self.map_config,
                robot_variant=self.robot_variant,
            )
        )
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = self.RENDER_LOOKAT
        self.camera.distance = float(self.RENDER_DISTANCE)
        self.camera.azimuth = float(self.RENDER_AZIMUTH)
        self.camera.elevation = float(self.RENDER_ELEVATION)
        self.joint_names = _resolve_arm_joint_names(self.model)
        self.joint_ids = tuple(
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            for joint_name in self.joint_names
        )
        if any(joint_id < 0 for joint_id in self.joint_ids):
            raise RuntimeError("Could not resolve Panda joint ids in MuJoCo model.")
        self.qpos_indices = np.asarray(
            [int(self.model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids],
            dtype=np.int64,
        )
        self.dof_indices = np.asarray(
            [int(self.model.jnt_dofadr[joint_id]) for joint_id in self.joint_ids],
            dtype=np.int64,
        )
        self.joint_range = self.model.jnt_range[
            np.asarray(self.joint_ids, dtype=np.intp)
        ].astype(np.float64)
        self.gripper_qpos_indices = _resolve_optional_qpos_indices(
            self.model,
            FRANKA_FINGER_JOINT_NAMES,
        )
        self.home_keyframe_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        )
        self.site_name = _resolve_probe_site_name(self.model)
        self.site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        )
        if self.site_id < 0:
            raise RuntimeError("Could not resolve a probe site in the MuJoCo model.")
        self.home_qpos = _default_home_qpos_for_joint_names(self.joint_names)
        self._reset_robot_pose()
        self.home_qpos = self.data.qpos[self.qpos_indices].astype(np.float64, copy=True)

    def _reset_robot_pose(self) -> None:
        if self.home_keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(
                self.model,
                self.data,
                self.home_keyframe_id,
            )
        else:
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            self.data.qpos[self.qpos_indices] = self.home_qpos
            if self.gripper_qpos_indices.size > 0:
                self.data.qpos[self.gripper_qpos_indices] = FRANKA_OPEN_GRIPPER_QPOS
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_indices].astype(np.float32, copy=True)

    def get_probe_position_3d(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].astype(np.float32, copy=True)

    def get_full_observation(self) -> dict[str, Any]:
        base = super().get_full_observation()
        base.update(
            {
                "joint_positions": self.get_joint_positions().tolist(),
                "probe_position_3d": self.get_probe_position_3d().tolist(),
            }
        )
        return base

    def _solve_inverse_kinematics(
        self,
        target_xyz: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> np.ndarray:
        q = (
            self.home_qpos.copy()
            if q_init is None
            else np.asarray(q_init, dtype=np.float64).copy()
        )
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        eye = np.eye(3, dtype=np.float64)

        for _ in range(self.IK_MAX_ITERS):
            self.data.qpos[self.qpos_indices] = q
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            current_xyz = self.data.site_xpos[self.site_id]
            error = np.asarray(target_xyz - current_xyz, dtype=np.float64)
            if float(np.linalg.norm(error)) <= self.IK_TOLERANCE:
                return q

            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jac = jacp[:, self.dof_indices]
            system = jac @ jac.T + (self.IK_DAMPING**2) * eye
            dq = jac.T @ np.linalg.solve(system, error)
            dq += self.IK_REGULARIZATION * (self.home_qpos - q)
            q = q + dq
            q = np.clip(q, self.joint_range[:, 0], self.joint_range[:, 1])

        self.data.qpos[self.qpos_indices] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        residual = float(np.linalg.norm(self.data.site_xpos[self.site_id] - target_xyz))
        if residual > self.IK_ACCEPT_RESIDUAL:
            raise RuntimeError(
                f"IK did not converge for target_xyz={target_xyz.tolist()}, residual={residual:.6f}"
            )
        return q

    def _sync_robot_to_probe_state(self, probe_state: tuple[float, float]) -> None:
        target_xyz = np.asarray(
            [float(probe_state[0]), float(probe_state[1]), self.PROBE_TARGET_Z],
            dtype=np.float64,
        )
        q_now = self.data.qpos[self.qpos_indices].astype(np.float64, copy=True)
        solved_q = self._solve_inverse_kinematics(target_xyz=target_xyz, q_init=q_now)
        self.data.qpos[self.qpos_indices] = solved_q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        task_id: int | None = None,
        enable_randomize: bool | None = None,
        start_state: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        state = super().reset(
            task_id=task_id,
            enable_randomize=enable_randomize,
            start_state=start_state,
        )
        self._reset_robot_pose()
        self._sync_robot_to_probe_state(state)
        return state

    def step(
        self,
        action: tuple[float, float],
    ) -> tuple[tuple[float, float], float, bool, dict[str, Any]]:
        next_state, reward, done, info = super().step(action)
        self._sync_robot_to_probe_state(next_state)
        info = {
            **info,
            "joint_positions": self.get_joint_positions().tolist(),
            "probe_position_3d": self.get_probe_position_3d().tolist(),
        }
        self.last_info = info
        return next_state, reward, done, info

    def sync_to_state(
        self,
        state: tuple[float, float],
        step_count: int | None = None,
    ) -> tuple[float, float]:
        synced_state = super().sync_to_state(state=state, step_count=step_count)
        self._sync_robot_to_probe_state(synced_state)
        self.last_info = {
            **self.last_info,
            "joint_positions": self.get_joint_positions().tolist(),
            "probe_position_3d": self.get_probe_position_3d().tolist(),
        }
        return synced_state

    def render_frame(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Call reset() before render_frame().")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self.image_size,
                width=self.image_size,
            )
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()


@dataclass(slots=True)
class _Node:
    point: tuple[float, float]
    parent: int | None


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return math.hypot(dx, dy)


def _mix_seed(state: int, value: int) -> int:
    return (
        state
        ^ (
            value
            + 0x9E3779B9
            + ((state << 6) & 0xFFFFFFFF)
            + (state >> 2)
        )
    ) & 0xFFFFFFFF


def _derive_seed(*values: int | float) -> int:
    state = 0xA5A5A5A5
    for value in values:
        normalized = int(round(float(value) * 10_000.0)) if isinstance(value, float) else int(value)
        state = _mix_seed(state, normalized & 0xFFFFFFFF)
    return state & 0xFFFFFFFF


def _nearest_node_index(tree: list[_Node], point: tuple[float, float]) -> int:
    return min(
        range(len(tree)),
        key=lambda node_index: _distance(tree[node_index].point, point),
    )


def _steer(
    from_point: tuple[float, float],
    to_point: tuple[float, float],
    max_distance: float,
) -> tuple[float, float]:
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    distance = math.hypot(dx, dy)
    if distance <= max_distance:
        return (float(to_point[0]), float(to_point[1]))
    scale = max_distance / max(distance, 1e-8)
    return (float(from_point[0] + dx * scale), float(from_point[1] + dy * scale))


def _sample_workspace_state(
    rng: random.Random,
    config: MapConfig,
) -> tuple[float, float]:
    return (
        float(rng.uniform(config.workspace.xmin, config.workspace.xmax)),
        float(rng.uniform(config.workspace.ymin, config.workspace.ymax)),
    )


def _is_segment_valid(
    from_point: tuple[float, float],
    to_point: tuple[float, float],
    config: MapConfig,
    resolution: float,
) -> bool:
    if resolution <= 0.0:
        raise ValueError("resolution must be positive.")
    distance = _distance(from_point, to_point)
    num_checks = max(1, int(math.ceil(distance / resolution)))
    for check_index in range(num_checks + 1):
        t = check_index / num_checks
        point = (
            float(from_point[0] * (1.0 - t) + to_point[0] * t),
            float(from_point[1] * (1.0 - t) + to_point[1] * t),
        )
        if not is_state_valid(*point, config=config):
            return False
    return True


def _extend_tree_towards(
    tree: list[_Node],
    target_point: tuple[float, float],
    config: MapConfig,
    step_size: float,
    collision_check_resolution: float,
) -> int | None:
    nearest_index = _nearest_node_index(tree, target_point)
    nearest_point = tree[nearest_index].point
    new_point = _steer(nearest_point, target_point, max_distance=step_size)
    if not _is_segment_valid(
        nearest_point,
        new_point,
        config=config,
        resolution=collision_check_resolution,
    ):
        return None
    tree.append(_Node(point=new_point, parent=nearest_index))
    return len(tree) - 1


def _connect_trees(
    tree: list[_Node],
    target_point: tuple[float, float],
    config: MapConfig,
    step_size: float,
    collision_check_resolution: float,
    connect_tolerance: float,
) -> int | None:
    nearest_index = _nearest_node_index(tree, target_point)
    current_index = nearest_index
    while True:
        current_point = tree[current_index].point
        if _distance(current_point, target_point) <= connect_tolerance:
            if _is_segment_valid(
                current_point,
                target_point,
                config=config,
                resolution=collision_check_resolution,
            ):
                tree.append(_Node(point=target_point, parent=current_index))
                return len(tree) - 1
            return None
        new_point = _steer(current_point, target_point, max_distance=step_size)
        if not _is_segment_valid(
            current_point,
            new_point,
            config=config,
            resolution=collision_check_resolution,
        ):
            return None
        tree.append(_Node(point=new_point, parent=current_index))
        current_index = len(tree) - 1


def _path_to_root(tree: list[_Node], node_index: int) -> list[tuple[float, float]]:
    path: list[tuple[float, float]] = []
    current_index: int | None = node_index
    while current_index is not None:
        node = tree[current_index]
        path.append(node.point)
        current_index = node.parent
    return path


def _reconstruct_bidirectional_path(
    tree_start: list[_Node],
    tree_goal: list[_Node],
    connect_from_start: int,
    connect_from_goal: int,
) -> list[tuple[float, float]]:
    path_start = list(reversed(_path_to_root(tree_start, connect_from_start)))
    path_goal = _path_to_root(tree_goal, connect_from_goal)
    return path_start + path_goal[1:]


def _densify_path(
    path: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    resolution: float,
) -> list[tuple[float, float]]:
    if resolution <= 0.0:
        raise ValueError("resolution must be positive.")
    if len(path) <= 1:
        return list(path)
    dense_path = [path[0]]
    for start_point, end_point in zip(path[:-1], path[1:], strict=False):
        segment_length = _distance(start_point, end_point)
        num_steps = max(1, int(math.ceil(segment_length / resolution)))
        for step_index in range(1, num_steps + 1):
            t = step_index / num_steps
            dense_path.append(
                (
                    float(start_point[0] * (1.0 - t) + end_point[0] * t),
                    float(start_point[1] * (1.0 - t) + end_point[1] * t),
                )
            )
    return dense_path


def _shortcut_path(
    path: list[tuple[float, float]],
    config: MapConfig,
    collision_check_resolution: float,
    rng: random.Random,
    num_trials: int = 100,
) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    shortened = path[:]
    for _ in range(num_trials):
        if len(shortened) <= 2:
            break
        index_a, index_b = sorted(rng.sample(range(len(shortened)), 2))
        if index_b - index_a <= 1:
            continue
        point_a = shortened[index_a]
        point_b = shortened[index_b]
        if _is_segment_valid(
            point_a,
            point_b,
            config=config,
            resolution=collision_check_resolution,
        ):
            shortened = shortened[: index_a + 1] + shortened[index_b:]
    return shortened


def plan_path_rrtconnect(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    solve_time: float = DEFAULT_SOLVE_TIME,
    config: MapConfig | None = None,
    interpolate_solution: bool = True,
    rng_seed: int | None = None,
) -> list[tuple[float, float]] | None:
    resolved_config = build_default_map_config() if config is None else config
    if solve_time <= 0.0:
        raise ValueError("solve_time must be positive.")
    if not is_state_valid(*start_xy, config=resolved_config):
        raise ValueError(f"Start state {start_xy} is not valid.")
    if not is_state_valid(*goal_xy, config=resolved_config):
        raise ValueError(f"Goal state {goal_xy} is not valid.")
    if _is_segment_valid(
        start_xy,
        goal_xy,
        config=resolved_config,
        resolution=DEFAULT_COLLISION_CHECK_RESOLUTION,
    ):
        path = [start_xy, goal_xy]
        return (
            _densify_path(path, resolution=DEFAULT_COLLISION_CHECK_RESOLUTION)
            if interpolate_solution
            else path
        )

    seed = (
        _derive_seed(start_xy[0], start_xy[1], goal_xy[0], goal_xy[1])
        if rng_seed is None
        else int(rng_seed) & 0xFFFFFFFF
    )
    rng = random.Random(seed)
    tree_start = [_Node(point=(float(start_xy[0]), float(start_xy[1])), parent=None)]
    tree_goal = [_Node(point=(float(goal_xy[0]), float(goal_xy[1])), parent=None)]
    start_time = time.monotonic()
    iteration_count = 0
    trees_swapped = False

    while iteration_count < DEFAULT_MAX_ITERATIONS:
        if time.monotonic() - start_time >= solve_time:
            break
        iteration_count += 1
        if rng.random() < DEFAULT_GOAL_SAMPLE_PROBABILITY:
            sample_point = tree_goal[0].point if not trees_swapped else tree_start[0].point
        else:
            sample_point = _sample_workspace_state(rng, resolved_config)
        extend_index = _extend_tree_towards(
            tree_start,
            sample_point,
            config=resolved_config,
            step_size=DEFAULT_STEP_SIZE,
            collision_check_resolution=DEFAULT_COLLISION_CHECK_RESOLUTION,
        )
        if extend_index is not None:
            new_point = tree_start[extend_index].point
            connect_index = _connect_trees(
                tree_goal,
                new_point,
                config=resolved_config,
                step_size=DEFAULT_STEP_SIZE,
                collision_check_resolution=DEFAULT_COLLISION_CHECK_RESOLUTION,
                connect_tolerance=DEFAULT_CONNECT_TOLERANCE,
            )
            if connect_index is not None:
                if trees_swapped:
                    path = _reconstruct_bidirectional_path(
                        tree_goal,
                        tree_start,
                        connect_index,
                        extend_index,
                    )
                else:
                    path = _reconstruct_bidirectional_path(
                        tree_start,
                        tree_goal,
                        extend_index,
                        connect_index,
                    )
                path = _shortcut_path(
                    path,
                    config=resolved_config,
                    collision_check_resolution=DEFAULT_COLLISION_CHECK_RESOLUTION,
                    rng=rng,
                )
                return (
                    _densify_path(path, resolution=DEFAULT_COLLISION_CHECK_RESOLUTION)
                    if interpolate_solution
                    else path
                )
        tree_start, tree_goal = tree_goal, tree_start
        trees_swapped = not trees_swapped
    return None


def plan_path_rrtconnect_via_waypoints(
    waypoints_xy: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    solve_time: float = DEFAULT_SOLVE_TIME,
    config: MapConfig | None = None,
    rng_seed: int | None = None,
) -> list[tuple[float, float]] | None:
    if len(waypoints_xy) < 2:
        raise ValueError("waypoints_xy must contain at least 2 waypoints.")
    resolved_config = build_default_map_config() if config is None else config
    full_path: list[tuple[float, float]] = []
    for segment_index, (segment_start, segment_goal) in enumerate(
        zip(waypoints_xy[:-1], waypoints_xy[1:], strict=False)
    ):
        segment_seed = None if rng_seed is None else _derive_seed(rng_seed, segment_index)
        segment_path = plan_path_rrtconnect(
            start_xy=segment_start,
            goal_xy=segment_goal,
            solve_time=solve_time,
            config=resolved_config,
            interpolate_solution=True,
            rng_seed=segment_seed,
        )
        if segment_path is None:
            return None
        if segment_index == 0:
            full_path.extend(segment_path)
        else:
            full_path.extend(segment_path[1:])
    return full_path


def _validate_task_conditioned_path(
    task_id: int,
    path_xy: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    config: MapConfig,
) -> tuple[bool, str | None]:
    task_spec = build_task_spec(task_id)
    saw_expected_start = False
    saw_m1 = False
    saw_expected_branch = False
    saw_m2 = False
    saw_expected_goal = False
    for point in path_xy:
        phase_name = get_phase_name(float(point[0]), float(point[1]), config=config)
        if phase_name == task_spec.start_region_name:
            saw_expected_start = True
        if phase_name == "M1":
            saw_m1 = True
        if phase_name in {"H1", "L1"}:
            if phase_name != task_spec.branch_region_name:
                return (
                    False,
                    f"branch mismatch: expected {task_spec.branch_region_name}, observed {phase_name}",
                )
            saw_expected_branch = True
        if phase_name == "M2":
            saw_m2 = True
        if phase_name in {"G1", "G2"}:
            if phase_name != task_spec.target_goal_name:
                return (
                    False,
                    f"goal mismatch: expected {task_spec.target_goal_name}, observed {phase_name}",
                )
            saw_expected_goal = True
    if not saw_expected_start:
        return False, f"path never entered expected start region {task_spec.start_region_name}"
    if not saw_m1:
        return False, "path never entered M1"
    if not saw_expected_branch:
        return False, f"path never entered expected branch {task_spec.branch_region_name}"
    if not saw_m2:
        return False, "path never entered M2"
    if not saw_expected_goal:
        return False, f"path never entered expected goal {task_spec.target_goal_name}"
    return True, None


def _build_task_route_waypoints(
    task_id: int,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    config: MapConfig,
) -> tuple[tuple[float, float], ...]:
    branch_center = get_branch_region(task_id, config=config).rectangles[0].center
    return (
        (float(start_xy[0]), float(start_xy[1])),
        config.merge_region_m1.rectangles[0].center,
        branch_center,
        config.merge_region_m2.rectangles[0].center,
        (float(goal_xy[0]), float(goal_xy[1])),
    )


def _make_episode(
    episode_id: int,
    task_id: int,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    target_goal_name: str,
    path_xy: list[tuple[float, float]],
) -> DemonstrationEpisode:
    path_array = np.asarray(path_xy, dtype=np.float64)
    return DemonstrationEpisode(
        episode_id=episode_id,
        task_id=task_id,
        start_xy=(float(start_xy[0]), float(start_xy[1])),
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        target_goal_name=target_goal_name,
        path_xy=path_array,
        path_length=int(path_array.shape[0]),
        success=True,
    )


def generate_demonstrations(
    num_per_task: int,
    seed: int,
    solve_time: float = DEFAULT_SOLVE_TIME,
    max_retries_per_demo: int = DEFAULT_RETRIES_PER_DEMO,
    low_success_warning_threshold: float = DEFAULT_LOW_SUCCESS_WARNING_THRESHOLD,
    enable_randomize: bool = False,
    config: MapConfig | None = None,
) -> DemonstrationDataset:
    if num_per_task <= 0:
        raise ValueError("num_per_task must be positive.")
    if max_retries_per_demo <= 0:
        raise ValueError("max_retries_per_demo must be positive.")

    resolved_config = build_default_map_config() if config is None else config
    env = PandaRouteSemanticEnv(
        map_config=resolved_config,
        rng_seed=seed,
        enable_randomize=enable_randomize,
    )
    episodes: list[DemonstrationEpisode] = []
    success_counts_by_task = {task_id: 0 for task_id in TASK_ID_VALUES}
    attempt_counts_by_task = {task_id: 0 for task_id in TASK_ID_VALUES}
    skipped_counts_by_task = {task_id: 0 for task_id in TASK_ID_VALUES}
    episode_id = 0

    for task_id in TASK_ID_VALUES:
        task_spec = build_task_spec(task_id)
        goal_xy = get_goal_region_for_task(task_id, config=resolved_config).center
        print(
            f"[task {task_id}] target={task_spec.target_goal_name}, "
            f"requested={num_per_task}, solve_time={solve_time:.2f}s"
        )
        for sample_index in range(num_per_task):
            success_path: list[tuple[float, float]] | None = None
            start_xy: tuple[float, float] | None = None
            for retry_index in range(max_retries_per_demo):
                attempt_counts_by_task[task_id] += 1
                start_xy = env.reset(task_id=task_id)
                if env.start_region_name != task_spec.start_region_name:
                    raise RuntimeError(
                        "Environment reset returned a start region that does not match the task. "
                        f"task_id={task_id}, expected={task_spec.start_region_name}, got={env.start_region_name}"
                    )
                route_waypoints = _build_task_route_waypoints(
                    task_id=task_id,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    config=resolved_config,
                )
                planning_seed = _derive_seed(seed, task_id, sample_index, retry_index)
                success_path = plan_path_rrtconnect_via_waypoints(
                    waypoints_xy=route_waypoints,
                    solve_time=solve_time,
                    config=resolved_config,
                    rng_seed=planning_seed,
                )
                if success_path is not None:
                    path_ok, reject_reason = _validate_task_conditioned_path(
                        task_id=task_id,
                        path_xy=success_path,
                        config=resolved_config,
                    )
                    if not path_ok:
                        success_path = None
                        print(
                            f"  retry {retry_index + 1}/{max_retries_per_demo} rejected "
                            f"for sample {sample_index + 1}/{num_per_task}: {reject_reason}"
                        )
                        continue
                    break
                print(
                    f"  retry {retry_index + 1}/{max_retries_per_demo} failed "
                    f"for sample {sample_index + 1}/{num_per_task}"
                )
            if success_path is None or start_xy is None:
                skipped_counts_by_task[task_id] += 1
                print(
                    f"  skipped sample {sample_index + 1}/{num_per_task} "
                    f"after {max_retries_per_demo} failed attempts"
                )
                continue
            episodes.append(
                _make_episode(
                    episode_id=episode_id,
                    task_id=task_id,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    target_goal_name=task_spec.target_goal_name,
                    path_xy=success_path,
                )
            )
            success_counts_by_task[task_id] += 1
            episode_id += 1
            print(
                f"  success {success_counts_by_task[task_id]}/{num_per_task} "
                f"(path_length={len(success_path)})"
            )
        attempts = attempt_counts_by_task[task_id]
        successes = success_counts_by_task[task_id]
        success_rate = successes / max(1, attempts)
        print(
            f"[task {task_id}] completed with {successes}/{num_per_task} saved "
            f"episodes across {attempts} planning attempts "
            f"(attempt success rate={success_rate:.3f})"
        )
        if success_rate < low_success_warning_threshold:
            print(
                f"WARNING: task {task_id} success rate {success_rate:.3f} is below "
                f"threshold {low_success_warning_threshold:.3f}"
            )
    return DemonstrationDataset(
        episodes=episodes,
        seed=seed,
        num_per_task_requested=num_per_task,
        success_counts_by_task=success_counts_by_task,
        attempt_counts_by_task=attempt_counts_by_task,
        skipped_counts_by_task=skipped_counts_by_task,
        solve_time=solve_time,
        retries_per_demo=max_retries_per_demo,
    )


def save_demonstrations(
    dataset: DemonstrationDataset,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_episodes = len(dataset.episodes)
    max_path_length = max((episode.path_length for episode in dataset.episodes), default=0)
    padded_paths = np.zeros((num_episodes, max_path_length, 2), dtype=np.float64)
    episode_ids = np.zeros(num_episodes, dtype=np.int64)
    task_ids = np.zeros(num_episodes, dtype=np.int64)
    target_goal_names = np.empty(num_episodes, dtype="<U8")
    start_xy = np.zeros((num_episodes, 2), dtype=np.float64)
    goal_xy = np.zeros((num_episodes, 2), dtype=np.float64)
    path_length = np.zeros(num_episodes, dtype=np.int64)
    success = np.zeros(num_episodes, dtype=bool)

    for episode_index, episode in enumerate(dataset.episodes):
        current_length = episode.path_length
        padded_paths[episode_index, :current_length] = episode.path_xy
        episode_ids[episode_index] = episode.episode_id
        task_ids[episode_index] = episode.task_id
        target_goal_names[episode_index] = episode.target_goal_name
        start_xy[episode_index] = np.asarray(episode.start_xy, dtype=np.float64)
        goal_xy[episode_index] = np.asarray(episode.goal_xy, dtype=np.float64)
        path_length[episode_index] = current_length
        success[episode_index] = episode.success

    np.savez_compressed(
        output_path,
        format_version=np.asarray("panda_route_rrtconnect_onebit_v1"),
        seed=np.asarray(dataset.seed, dtype=np.int64),
        num_per_task_requested=np.asarray(dataset.num_per_task_requested, dtype=np.int64),
        solve_time=np.asarray(dataset.solve_time, dtype=np.float64),
        retries_per_demo=np.asarray(dataset.retries_per_demo, dtype=np.int64),
        episode_id=episode_ids,
        task_id=task_ids,
        target_goal_name=target_goal_names,
        start_xy=start_xy,
        goal_xy=goal_xy,
        path_xy=padded_paths,
        path_length=path_length,
        success=success,
        success_counts_by_task=np.asarray(
            [dataset.success_counts_by_task[task_id] for task_id in TASK_ID_VALUES],
            dtype=np.int64,
        ),
        attempt_counts_by_task=np.asarray(
            [dataset.attempt_counts_by_task[task_id] for task_id in TASK_ID_VALUES],
            dtype=np.int64,
        ),
        skipped_counts_by_task=np.asarray(
            [dataset.skipped_counts_by_task[task_id] for task_id in TASK_ID_VALUES],
            dtype=np.int64,
        ),
    )
    print(f"Saved {num_episodes} demonstrations to {output_path}")
    return output_path


def _as_path_array(path_xy: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
    path_array = np.asarray(path_xy, dtype=np.float64)
    if path_array.ndim != 2 or path_array.shape[1] != 2:
        raise ValueError("path_xy must have shape [T, 2].")
    if path_array.shape[0] == 0:
        raise ValueError("path_xy must contain at least one point.")
    return path_array


def compute_path_distance(path_xy: np.ndarray | list[tuple[float, float]]) -> float:
    path_array = _as_path_array(path_xy)
    if path_array.shape[0] <= 1:
        return 0.0
    deltas = np.diff(path_array, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def resample_path_fixed_length(
    path_xy: np.ndarray | list[tuple[float, float]],
    t_fixed: int,
) -> np.ndarray:
    if t_fixed <= 0:
        raise ValueError("t_fixed must be positive.")
    path_array = _as_path_array(path_xy)
    if path_array.shape[0] == 1:
        return np.repeat(path_array.astype(np.float32), t_fixed, axis=0)

    segment_lengths = np.linalg.norm(np.diff(path_array, axis=0), axis=1)
    cumulative_lengths = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths, dtype=np.float64)]
    )
    total_length = float(cumulative_lengths[-1])
    if total_length <= 1e-8:
        return np.repeat(path_array[:1].astype(np.float32), t_fixed, axis=0)

    target_lengths = np.linspace(0.0, total_length, num=t_fixed, dtype=np.float64)
    resampled_x = np.interp(target_lengths, cumulative_lengths, path_array[:, 0])
    resampled_y = np.interp(target_lengths, cumulative_lengths, path_array[:, 1])
    return np.stack([resampled_x, resampled_y], axis=1).astype(np.float32)


def build_actions_from_states(
    states_xy: np.ndarray,
    last_action_mode: str = DEFAULT_LAST_ACTION_MODE,
) -> np.ndarray:
    state_array = _as_path_array(states_xy).astype(np.float32, copy=False)
    if last_action_mode != "zero":
        raise ValueError(
            f"Unsupported last_action_mode={last_action_mode!r}. Expected 'zero'."
        )
    actions = np.zeros_like(state_array, dtype=np.float32)
    if state_array.shape[0] >= 2:
        actions[:-1] = state_array[1:] - state_array[:-1]
    return actions


def build_goal_onehot(task_id: int) -> np.ndarray:
    if task_id not in TASK_INDEX_BY_ID:
        raise ValueError(f"Unsupported task_id={task_id}.")
    goal_onehot = np.zeros(len(TASK_ID_VALUES), dtype=np.float32)
    goal_onehot[TASK_INDEX_BY_ID[task_id]] = 1.0
    return goal_onehot


def build_task_code_bits(task_id: int) -> np.ndarray:
    if task_id not in TASK_INDEX_BY_ID:
        raise ValueError(f"Unsupported task_id={task_id}.")
    return np.asarray([int(task_id)], dtype=np.int64)


def compute_path_signature_sequence(
    states_xy: np.ndarray,
    window_size: int = DEFAULT_SIGNATURE_WINDOW_SIZE,
    sig_depth: int = DEFAULT_SIGNATURE_DEPTH,
    signature_backend: str = DEFAULT_SIGNATURE_BACKEND,
) -> np.ndarray:
    if sig_depth <= 0:
        raise ValueError("sig_depth must be positive.")
    state_array = _as_path_array(states_xy).astype(np.float32, copy=False)
    resolved_backend = (
        signature_backend
        if signature_backend in {"signatory", "simple"}
        else resolve_signature_backend(signature_backend)
    )
    signatures: list[np.ndarray] = []
    for time_index in range(state_array.shape[0]):
        if window_size <= 0:
            window = state_array[: time_index + 1]
        else:
            start = max(0, time_index - window_size + 1)
            window = state_array[start : time_index + 1]
            if window.shape[0] < window_size:
                pad = np.repeat(state_array[:1], window_size - window.shape[0], axis=0)
                window = np.concatenate([pad, window], axis=0)
        if resolved_backend == "signatory":
            signature = compute_signatory_signature_np(window, sig_depth)
        else:
            signature = compute_simple_signature_np(window, sig_depth)
        signatures.append(signature)
    return np.stack(signatures, axis=0).astype(np.float32)


def encode_phase_labels(phase_names: tuple[str, ...] | list[str]) -> np.ndarray:
    encoded = np.zeros(len(phase_names), dtype=np.int64)
    fallback_id = PHASE_NAME_TO_ID["free_space_other"]
    for phase_index, phase_name in enumerate(phase_names):
        encoded[phase_index] = PHASE_NAME_TO_ID.get(phase_name, fallback_id)
    return encoded


def validate_task_coverage(task_ids: np.ndarray | list[int]) -> dict[int, int]:
    task_array = np.asarray(task_ids, dtype=np.int64)
    task_counts = {task_id: int(np.sum(task_array == task_id)) for task_id in TASK_ID_VALUES}
    missing_tasks = [
        f"{task_id}:{TASK_ID_TO_GOAL_NAME[task_id]}"
        for task_id, count in task_counts.items()
        if count <= 0
    ]
    if missing_tasks:
        raise ValueError(
            "Dataset is missing at least one target class. "
            f"Missing tasks: {', '.join(missing_tasks)}"
        )
    return task_counts


def build_balanced_episode_order(
    task_ids: np.ndarray | list[int],
    seed: int,
) -> np.ndarray:
    task_array = np.asarray(task_ids, dtype=np.int64)
    validate_task_coverage(task_array)
    rng = np.random.default_rng(seed)
    indices_by_task = {
        task_id: np.flatnonzero(task_array == task_id).tolist()
        for task_id in TASK_ID_VALUES
    }
    for indices in indices_by_task.values():
        rng.shuffle(indices)

    mixed_order: list[int] = []
    while True:
        available_tasks = [task_id for task_id, indices in indices_by_task.items() if indices]
        if not available_tasks:
            break
        rng.shuffle(available_tasks)
        for task_id in available_tasks:
            mixed_order.append(indices_by_task[task_id].pop())
    return np.asarray(mixed_order, dtype=np.int64)


def process_demonstration_dataset(
    raw_dataset: DemonstrationDataset,
    t_fixed: int = DEFAULT_T_FIXED,
    include_phase_labels: bool = True,
    include_path_signatures: bool = DEFAULT_INCLUDE_PATH_SIGNATURES,
    path_signature_key: str = DEFAULT_PATH_SIGNATURE_KEY,
    path_signature_window_size: int = DEFAULT_SIGNATURE_WINDOW_SIZE,
    path_signature_depth: int = DEFAULT_SIGNATURE_DEPTH,
    path_signature_backend: str = DEFAULT_SIGNATURE_BACKEND,
    last_action_mode: str = DEFAULT_LAST_ACTION_MODE,
    config: MapConfig | None = None,
) -> ProcessedDemonstrationDataset:
    if len(raw_dataset) == 0:
        raise ValueError("raw_dataset is empty.")
    if t_fixed <= 0:
        raise ValueError("t_fixed must be positive.")

    map_config = build_default_map_config() if config is None else config
    num_episodes = len(raw_dataset.episodes)
    observations = np.zeros((num_episodes, t_fixed, 2), dtype=np.float32)
    actions = np.zeros((num_episodes, t_fixed, 2), dtype=np.float32)
    path_signatures: np.ndarray | None = None
    task_ids = np.zeros(num_episodes, dtype=np.int64)
    task_code_bits = np.zeros((num_episodes, 1), dtype=np.int64)
    goal_onehot = np.zeros((num_episodes, len(TASK_ID_VALUES)), dtype=np.float32)
    phase_labels = (
        np.zeros((num_episodes, t_fixed), dtype=np.int64)
        if include_phase_labels
        else None
    )
    episode_ids = np.zeros(num_episodes, dtype=np.int64)
    target_goal_names = np.empty(num_episodes, dtype="<U8")
    start_xy = np.zeros((num_episodes, 2), dtype=np.float32)
    goal_xy = np.zeros((num_episodes, 2), dtype=np.float32)
    raw_path_lengths = np.zeros(num_episodes, dtype=np.int64)
    raw_path_distances = np.zeros(num_episodes, dtype=np.float32)
    success = np.zeros(num_episodes, dtype=bool)

    raw_task_ids = np.asarray([episode.task_id for episode in raw_dataset.episodes], dtype=np.int64)
    task_counts = validate_task_coverage(raw_task_ids)
    episode_order = build_balanced_episode_order(raw_task_ids, raw_dataset.seed)
    resolved_signature_backend: str | None = None
    if include_path_signatures:
        resolved_signature_backend = resolve_signature_backend(path_signature_backend)
        window_label = (
            "all_prefix" if path_signature_window_size <= 0 else str(path_signature_window_size)
        )
        print(
            "Processing demonstrations with path signatures: "
            f"key={path_signature_key}, window={window_label}, "
            f"depth={path_signature_depth}, backend={resolved_signature_backend}"
        )
    print(
        "Processing demonstrations with balanced shuffled order: "
        + ", ".join(
            f"{TASK_ID_TO_GOAL_NAME[task_id]}={task_counts[task_id]}"
            for task_id in TASK_ID_VALUES
        )
    )

    for episode_index, source_index in enumerate(episode_order.tolist()):
        episode = raw_dataset.episodes[source_index]
        resampled_path = resample_path_fixed_length(episode.path_xy, t_fixed=t_fixed)
        observations[episode_index] = resampled_path
        actions[episode_index] = build_actions_from_states(
            resampled_path,
            last_action_mode=last_action_mode,
        )
        if include_path_signatures:
            signature_sequence = compute_path_signature_sequence(
                resampled_path,
                window_size=path_signature_window_size,
                sig_depth=path_signature_depth,
                signature_backend=(
                    DEFAULT_SIGNATURE_BACKEND
                    if resolved_signature_backend is None
                    else resolved_signature_backend
                ),
            )
            if path_signatures is None:
                path_signatures = np.zeros(
                    (num_episodes, t_fixed, int(signature_sequence.shape[1])),
                    dtype=np.float32,
                )
            path_signatures[episode_index] = signature_sequence
        task_ids[episode_index] = episode.task_id
        task_code_bits[episode_index] = build_task_code_bits(episode.task_id)
        goal_onehot[episode_index] = build_goal_onehot(episode.task_id)
        if phase_labels is not None:
            phase_annotation = annotate_trajectory_phases(
                [tuple(map(float, point_xy)) for point_xy in resampled_path.tolist()],
                config=map_config,
            )
            phase_labels[episode_index] = encode_phase_labels(phase_annotation.phase_labels)
        episode_ids[episode_index] = episode.episode_id
        target_goal_names[episode_index] = episode.target_goal_name
        start_xy[episode_index] = np.asarray(episode.start_xy, dtype=np.float32)
        goal_xy[episode_index] = np.asarray(episode.goal_xy, dtype=np.float32)
        raw_path_lengths[episode_index] = int(episode.path_length)
        raw_path_distances[episode_index] = compute_path_distance(episode.path_xy)
        success[episode_index] = bool(episode.success)

    return ProcessedDemonstrationDataset(
        observations=observations,
        actions=actions,
        path_signatures=path_signatures,
        task_ids=task_ids,
        task_code_bits=task_code_bits,
        goal_onehot=goal_onehot,
        phase_labels=phase_labels,
        episode_ids=episode_ids,
        target_goal_names=target_goal_names,
        start_xy=start_xy,
        goal_xy=goal_xy,
        raw_path_lengths=raw_path_lengths,
        raw_path_distances=raw_path_distances,
        success=success,
        seed=raw_dataset.seed,
        t_fixed=t_fixed,
        action_padding_mode=last_action_mode,
        path_signature_key=None if path_signatures is None else str(path_signature_key),
        path_signature_window_size=(0 if path_signatures is None else int(path_signature_window_size)),
        path_signature_depth=(0 if path_signatures is None else int(path_signature_depth)),
        path_signature_backend=(None if path_signatures is None else resolved_signature_backend),
        phase_label_vocab=PHASE_LABEL_VOCAB,
        source_num_per_task_requested=raw_dataset.num_per_task_requested,
        source_solve_time=raw_dataset.solve_time,
        source_retries_per_demo=raw_dataset.retries_per_demo,
    )


def save_processed_dataset(
    dataset: ProcessedDemonstrationDataset,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {
        "format_version": np.asarray("panda_route_act_ready_sig_v1"),
        "observations": dataset.observations,
        "actions": dataset.actions,
        "task_ids": dataset.task_ids,
        "task_code_bits": dataset.task_code_bits,
        "goal_onehot": dataset.goal_onehot,
        "episode_ids": dataset.episode_ids,
        "target_goal_names": dataset.target_goal_names,
        "start_xy": dataset.start_xy,
        "goal_xy": dataset.goal_xy,
        "raw_path_lengths": dataset.raw_path_lengths,
        "raw_path_distances": dataset.raw_path_distances,
        "success": dataset.success,
        "seed": np.asarray(dataset.seed, dtype=np.int64),
        "t_fixed": np.asarray(dataset.t_fixed, dtype=np.int64),
        "action_padding_mode": np.asarray(dataset.action_padding_mode),
        "path_signature_key": np.asarray("" if dataset.path_signature_key is None else dataset.path_signature_key),
        "path_signature_window_size": np.asarray(dataset.path_signature_window_size, dtype=np.int64),
        "path_signature_depth": np.asarray(dataset.path_signature_depth, dtype=np.int64),
        "path_signature_backend": np.asarray("" if dataset.path_signature_backend is None else dataset.path_signature_backend),
        "phase_label_vocab": np.asarray(dataset.phase_label_vocab, dtype="<U32"),
        "source_num_per_task_requested": np.asarray(dataset.source_num_per_task_requested, dtype=np.int64),
        "source_solve_time": np.asarray(dataset.source_solve_time, dtype=np.float32),
        "source_retries_per_demo": np.asarray(dataset.source_retries_per_demo, dtype=np.int64),
    }
    if dataset.phase_labels is not None:
        save_kwargs["phase_labels"] = dataset.phase_labels
    if dataset.path_signatures is not None:
        save_kwargs["path_signatures"] = dataset.path_signatures
    np.savez_compressed(output_path, **save_kwargs)
    print(f"Saved processed dataset with {len(dataset)} samples to {output_path}")
    return output_path


def _np_scalar_to_str(value: Any) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(()).item())
    return str(value)


def _decode_phase_labels_from_ids(
    phase_ids: np.ndarray,
    phase_vocab: tuple[str, ...],
) -> tuple[str, ...]:
    fallback_name = "free_space_other"
    decoded: list[str] = []
    for phase_id in np.asarray(phase_ids, dtype=np.int64):
        index = int(phase_id)
        if 0 <= index < len(phase_vocab):
            decoded.append(phase_vocab[index])
        else:
            decoded.append(fallback_name)
    return tuple(decoded)


def _load_replay_dataset_from_raw_npz(
    dataset_path: Path,
    payload: np.lib.npyio.NpzFile,
    format_version: str,
) -> ReplayDataset:
    padded_paths = np.asarray(payload["path_xy"], dtype=np.float32)
    path_lengths = np.asarray(payload["path_length"], dtype=np.int64)
    task_ids = np.asarray(payload["task_id"], dtype=np.int64)
    source_episode_ids = np.asarray(payload["episode_id"], dtype=np.int64)
    target_goal_names = np.asarray(payload["target_goal_name"])
    start_xy = np.asarray(payload["start_xy"], dtype=np.float32)
    goal_xy = np.asarray(payload["goal_xy"], dtype=np.float32)
    success = np.asarray(payload["success"], dtype=bool)

    if padded_paths.ndim != 3 or padded_paths.shape[-1] != 2:
        raise ValueError(
            f"Raw replay dataset {dataset_path} has invalid path_xy shape {tuple(padded_paths.shape)}."
        )

    episodes: list[ReplayEpisode] = []
    for replay_index in range(int(padded_paths.shape[0])):
        path_length = int(path_lengths[replay_index])
        if path_length <= 0:
            raise ValueError(
                f"Episode {replay_index} in {dataset_path} has non-positive path_length={path_length}."
            )
        states_xy = np.asarray(
            padded_paths[replay_index, :path_length],
            dtype=np.float32,
        ).copy()
        episodes.append(
            ReplayEpisode(
                replay_index=replay_index,
                source_episode_id=int(source_episode_ids[replay_index]),
                task_id=int(task_ids[replay_index]),
                target_goal_name=str(target_goal_names[replay_index]),
                states_xy=states_xy,
                start_xy=(float(start_xy[replay_index, 0]), float(start_xy[replay_index, 1])),
                goal_xy=(float(goal_xy[replay_index, 0]), float(goal_xy[replay_index, 1])),
                success=bool(success[replay_index]),
                source_format="raw_npz",
                raw_path_length=path_length,
            )
        )
    return ReplayDataset(
        dataset_path=dataset_path,
        format_version=format_version,
        source_kind="raw_npz",
        episodes=episodes,
    )


def _load_replay_dataset_from_processed_npz(
    dataset_path: Path,
    payload: np.lib.npyio.NpzFile,
    format_version: str,
) -> ReplayDataset:
    observations = np.asarray(payload["observations"], dtype=np.float32)
    task_ids = np.asarray(payload["task_ids"], dtype=np.int64)
    source_episode_ids = np.asarray(payload["episode_ids"], dtype=np.int64)
    target_goal_names = np.asarray(payload["target_goal_names"])
    start_xy = np.asarray(payload["start_xy"], dtype=np.float32)
    goal_xy = np.asarray(payload["goal_xy"], dtype=np.float32)
    success = np.asarray(payload["success"], dtype=bool)
    raw_path_lengths = np.asarray(payload["raw_path_lengths"], dtype=np.int64)
    phase_ids = (
        None if "phase_labels" not in payload.files else np.asarray(payload["phase_labels"], dtype=np.int64)
    )
    phase_vocab = (
        PHASE_LABEL_VOCAB
        if "phase_label_vocab" not in payload.files
        else tuple(str(name) for name in np.asarray(payload["phase_label_vocab"]).tolist())
    )

    if observations.ndim != 3 or observations.shape[-1] != 2:
        raise ValueError(
            f"Processed replay dataset {dataset_path} has invalid observations shape {tuple(observations.shape)}."
        )

    episodes: list[ReplayEpisode] = []
    for replay_index in range(int(observations.shape[0])):
        states_xy = np.asarray(observations[replay_index], dtype=np.float32).copy()
        phase_labels = (
            None
            if phase_ids is None
            else _decode_phase_labels_from_ids(phase_ids[replay_index], phase_vocab)
        )
        episodes.append(
            ReplayEpisode(
                replay_index=replay_index,
                source_episode_id=int(source_episode_ids[replay_index]),
                task_id=int(task_ids[replay_index]),
                target_goal_name=str(target_goal_names[replay_index]),
                states_xy=states_xy,
                start_xy=(float(start_xy[replay_index, 0]), float(start_xy[replay_index, 1])),
                goal_xy=(float(goal_xy[replay_index, 0]), float(goal_xy[replay_index, 1])),
                success=bool(success[replay_index]),
                source_format="processed_npz",
                phase_labels=phase_labels,
                raw_path_length=int(raw_path_lengths[replay_index]),
            )
        )
    return ReplayDataset(
        dataset_path=dataset_path,
        format_version=format_version,
        source_kind="processed_npz",
        episodes=episodes,
    )


def load_replay_dataset(dataset_path: str | Path) -> ReplayDataset:
    dataset_path = Path(dataset_path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Replay dataset path does not exist: {dataset_path}")
    if dataset_path.suffix != ".npz":
        raise ValueError(
            f"Unsupported replay dataset format for {dataset_path}. Expected a .npz file."
        )

    with np.load(dataset_path, allow_pickle=False) as payload:
        format_version = (
            "unknown"
            if "format_version" not in payload.files
            else _np_scalar_to_str(payload["format_version"])
        )
        if "path_xy" in payload.files and "path_length" in payload.files:
            return _load_replay_dataset_from_raw_npz(dataset_path, payload, format_version)
        if "observations" in payload.files:
            return _load_replay_dataset_from_processed_npz(dataset_path, payload, format_version)

    raise ValueError(
        f"Replay dataset {dataset_path} does not look like a supported panda_route .npz export."
    )


def filter_replay_episodes(
    dataset: ReplayDataset,
    *,
    task_id: int | None = None,
    success_only: bool = False,
) -> list[ReplayEpisode]:
    episodes = list(dataset.episodes)
    if task_id is not None:
        episodes = [episode for episode in episodes if episode.task_id == int(task_id)]
    if success_only:
        episodes = [episode for episode in episodes if episode.success]
    return episodes


def print_replay_dataset_summary(
    dataset: ReplayDataset,
    *,
    episodes: list[ReplayEpisode] | None = None,
    max_episodes: int = 20,
) -> None:
    listed_episodes = dataset.episodes if episodes is None else episodes
    print(
        f"Replay dataset: {dataset.dataset_path} "
        f"(format={dataset.format_version}, kind={dataset.source_kind}, "
        f"episodes={len(listed_episodes)}/{len(dataset)})"
    )
    if not listed_episodes:
        print("  No episodes matched the requested filter.")
        return
    for listed_index, episode in enumerate(listed_episodes[: max(0, int(max_episodes))]):
        print(
            f"  [{listed_index:03d}] replay_index={episode.replay_index:03d} "
            f"episode_id={episode.source_episode_id:03d} "
            f"task={episode.task_id}->{episode.target_goal_name} "
            f"len={len(episode):03d} success={episode.success}"
        )
    if len(listed_episodes) > max_episodes:
        print(f"  ... {len(listed_episodes) - max_episodes} more episodes omitted")


def _select_replay_episode_index_interactively(
    episodes: list[ReplayEpisode],
) -> int:
    if not episodes:
        raise ValueError("Cannot select an episode from an empty replay set.")
    upper_bound = len(episodes) - 1
    while True:
        try:
            raw_value = input(
                f"Select replay episode index [0-{upper_bound}] "
                "(press Enter for 0): "
            ).strip()
        except EOFError:
            return 0
        if raw_value == "":
            return 0
        try:
            selected_index = int(raw_value)
        except ValueError:
            print(f"Invalid selection {raw_value!r}; expected an integer.")
            continue
        if 0 <= selected_index <= upper_bound:
            return selected_index
        print(f"Selection {selected_index} is out of range [0, {upper_bound}].")


def _print_replay_controls() -> None:
    print("MuJoCo replay controls:")
    print("  mouse / trackpad: move camera")
    print("  space: pause or resume")
    print("  n / p: next or previous episode")
    print("  r: restart current episode")
    print("  l: toggle loop mode")
    print("  s: print current episode metadata")
    print("  q: quit replay")


def _print_replay_episode_header(
    episode: ReplayEpisode,
    *,
    list_index: int,
    total_episodes: int,
    paused: bool,
    loop: bool,
) -> None:
    print(
        f"[viewer] episode {list_index + 1}/{total_episodes} "
        f"(replay_index={episode.replay_index}, episode_id={episode.source_episode_id}, "
        f"task={episode.task_id}->{episode.target_goal_name}, len={len(episode)}, "
        f"success={episode.success}, paused={paused}, loop={loop})"
    )


def launch_replay_viewer(
    episodes: list[ReplayEpisode],
    *,
    initial_episode_index: int = 0,
    fps: int = DEFAULT_FPS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    loop: bool = False,
    start_paused: bool = False,
) -> None:
    if mujoco is None:  # pragma: no cover
        raise RuntimeError(
            "mujoco is required for interactive replay. Install it first."
        ) from _MUJOCO_IMPORT_ERROR
    try:
        import mujoco.viewer as mujoco_viewer
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "mujoco.viewer is required for the interactive 3D replay window."
        ) from exc

    if not episodes:
        raise ValueError("No replay episodes were provided.")
    if fps <= 0:
        raise ValueError("fps must be positive.")

    env = PandaRouteMjEnv(
        map_config=build_default_map_config(),
        rng_seed=DEFAULT_RANDOM_SEED,
        enable_randomize=False,
        image_size=image_size,
    )
    controller = {
        "paused": bool(start_paused),
        "loop": bool(loop),
        "episode_delta": 0,
        "restart": False,
        "quit": False,
        "print_info": False,
    }

    current_episode_index = int(np.clip(initial_episode_index, 0, len(episodes) - 1))
    current_frame_index = 0
    frame_dt = 1.0 / float(fps)

    def _load_episode(episode_index: int, viewer_handle: Any | None) -> None:
        nonlocal current_episode_index, current_frame_index
        current_episode_index = int(np.clip(episode_index, 0, len(episodes) - 1))
        current_frame_index = 0
        episode = episodes[current_episode_index]
        env.reset(
            task_id=episode.task_id,
            start_state=(float(episode.states_xy[0, 0]), float(episode.states_xy[0, 1])),
        )
        env.sync_to_state(
            (float(episode.states_xy[0, 0]), float(episode.states_xy[0, 1])),
            step_count=0,
        )
        if viewer_handle is not None:
            viewer_handle.cam.lookat[:] = env.RENDER_LOOKAT
            viewer_handle.cam.distance = float(env.RENDER_DISTANCE)
            viewer_handle.cam.azimuth = float(env.RENDER_AZIMUTH)
            viewer_handle.cam.elevation = float(env.RENDER_ELEVATION)
        _print_replay_episode_header(
            episode,
            list_index=current_episode_index,
            total_episodes=len(episodes),
            paused=bool(controller["paused"]),
            loop=bool(controller["loop"]),
        )

    def _key_callback(keycode: int) -> None:
        if keycode in (ord(" "),):
            controller["paused"] = not bool(controller["paused"])
        elif keycode in (ord("n"), ord("N")):
            controller["episode_delta"] = 1
        elif keycode in (ord("p"), ord("P")):
            controller["episode_delta"] = -1
        elif keycode in (ord("r"), ord("R")):
            controller["restart"] = True
        elif keycode in (ord("l"), ord("L")):
            controller["loop"] = not bool(controller["loop"])
        elif keycode in (ord("s"), ord("S")):
            controller["print_info"] = True
        elif keycode in (ord("q"), ord("Q")):
            controller["quit"] = True

    _print_replay_controls()
    try:
        with mujoco_viewer.launch_passive(
            env.model,
            env.data,
            key_callback=_key_callback,
        ) as viewer_handle:
            _load_episode(current_episode_index, viewer_handle)
            while viewer_handle.is_running() and not bool(controller["quit"]):
                loop_start = time.monotonic()
                if int(controller["episode_delta"]) != 0:
                    next_episode_index = (
                        current_episode_index + int(controller["episode_delta"])
                    ) % len(episodes)
                    controller["episode_delta"] = 0
                    _load_episode(next_episode_index, viewer_handle)
                if bool(controller["restart"]):
                    controller["restart"] = False
                    _load_episode(current_episode_index, viewer_handle)
                if bool(controller["print_info"]):
                    controller["print_info"] = False
                    _print_replay_episode_header(
                        episodes[current_episode_index],
                        list_index=current_episode_index,
                        total_episodes=len(episodes),
                        paused=bool(controller["paused"]),
                        loop=bool(controller["loop"]),
                    )

                episode = episodes[current_episode_index]
                if not bool(controller["paused"]):
                    env.sync_to_state(
                        (
                            float(episode.states_xy[current_frame_index, 0]),
                            float(episode.states_xy[current_frame_index, 1]),
                        ),
                        step_count=current_frame_index,
                    )
                    current_frame_index += 1
                    if current_frame_index >= len(episode):
                        if bool(controller["loop"]):
                            _load_episode(current_episode_index, viewer_handle)
                        else:
                            controller["paused"] = True
                            current_frame_index = len(episode) - 1
                viewer_handle.sync()
                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, frame_dt - elapsed) if not bool(controller["paused"]) else 0.01)
    finally:
        env.close()


def get_replay_defaults() -> dict[str, Any]:
    return {
        "fps": DEFAULT_FPS,
        "image_size": DEFAULT_IMAGE_SIZE,
        "max_list_episodes": 30,
    }


def replay_dataset_with_viewer(args) -> None:
    dataset = load_replay_dataset(args.dataset_path)
    episodes = filter_replay_episodes(
        dataset,
        task_id=getattr(args, "task_id", None),
        success_only=bool(getattr(args, "success_only", False)),
    )
    if not episodes:
        raise RuntimeError("No replay episodes matched the requested filters.")

    max_list_episodes = int(getattr(args, "max_list_episodes", 30))
    print_replay_dataset_summary(
        dataset,
        episodes=episodes,
        max_episodes=max_list_episodes,
    )
    if bool(getattr(args, "list_episodes", False)):
        return

    selected_episode_index = 0
    requested_episode_id = getattr(args, "episode_id", None)
    if requested_episode_id is not None:
        matching_indices = [
            index
            for index, episode in enumerate(episodes)
            if episode.source_episode_id == int(requested_episode_id)
        ]
        if not matching_indices:
            raise ValueError(
                f"No replay episode with episode_id={requested_episode_id} matched the current filters."
            )
        selected_episode_index = matching_indices[0]
    elif getattr(args, "episode_index", None) is not None:
        selected_episode_index = int(args.episode_index)
        if not (0 <= selected_episode_index < len(episodes)):
            raise ValueError(
                f"episode_index={selected_episode_index} is out of range for {len(episodes)} filtered episodes."
            )
    elif not bool(getattr(args, "no_prompt", False)) and len(episodes) > 1 and sys.stdin.isatty():
        selected_episode_index = _select_replay_episode_index_interactively(episodes)

    launch_replay_viewer(
        episodes,
        initial_episode_index=selected_episode_index,
        fps=int(args.fps),
        image_size=int(args.image_size),
        loop=bool(getattr(args, "loop", False)),
        start_paused=bool(getattr(args, "start_paused", False)),
    )


def dataset_summary(dataset: ProcessedDemonstrationDataset) -> dict[str, Any]:
    num_samples = len(dataset)
    task_counts = {task_id: int(np.sum(dataset.task_ids == task_id)) for task_id in TASK_ID_VALUES}
    avg_raw_steps = float(dataset.raw_path_lengths.mean()) if num_samples > 0 else 0.0
    avg_raw_distance = float(dataset.raw_path_distances.mean()) if num_samples > 0 else 0.0
    print("Processed dataset summary")
    print(f"  num_samples={num_samples}")
    print(f"  fixed_horizon={dataset.t_fixed}")
    print(f"  observation_shape={tuple(dataset.observations.shape)}")
    print(f"  action_shape={tuple(dataset.actions.shape)}")
    if dataset.path_signatures is not None:
        window_label = (
            "all_prefix" if dataset.path_signature_window_size <= 0 else str(dataset.path_signature_window_size)
        )
        print(f"  path_signature_shape={tuple(dataset.path_signatures.shape)}")
        print(
            "  path_signature="
            f"{dataset.path_signature_key}, window={window_label}, "
            f"depth={dataset.path_signature_depth}, backend={dataset.path_signature_backend}"
        )
    print(f"  avg_raw_path_steps={avg_raw_steps:.2f}")
    print(f"  avg_raw_path_distance={avg_raw_distance:.4f}")
    for task_id in TASK_ID_VALUES:
        task_mask = dataset.task_ids == task_id
        task_count = task_counts[task_id]
        task_name = TASK_ID_TO_GOAL_NAME[task_id]
        if task_count == 0:
            print(f"  task_{task_id}_{task_name}: count=0")
            continue
        task_avg_steps = float(dataset.raw_path_lengths[task_mask].mean())
        task_avg_distance = float(dataset.raw_path_distances[task_mask].mean())
        print(
            f"  task_{task_id}_{task_name}: count={task_count}, "
            f"avg_steps={task_avg_steps:.2f}, avg_distance={task_avg_distance:.4f}"
        )
    return {
        "num_samples": num_samples,
        "fixed_horizon": dataset.t_fixed,
        "task_counts": task_counts,
        "avg_raw_path_steps": avg_raw_steps,
        "avg_raw_path_distance": avg_raw_distance,
    }


def _require_lerobot_export_dependencies():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to export LeRobot v3.0 datasets. "
            "Install it first or rerun with --skip-lerobot-export."
        ) from exc
    return pa, pq


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
        raise RuntimeError("ffmpeg is required to export LeRobot videos.") from exc


def ffprobe_video(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        raw_output = subprocess.check_output(cmd, text=True)
    except FileNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("ffprobe is required to validate exported videos.") from exc
    payload = json.loads(raw_output)
    stream = payload["streams"][0]
    frame_rate = stream.get("avg_frame_rate", "0/1")
    if "/" in frame_rate:
        numerator, denominator = frame_rate.split("/", 1)
        fps = float(numerator) / max(float(denominator), 1.0)
    else:
        fps = float(frame_rate)
    frame_count = stream.get("nb_frames")
    if frame_count in (None, "N/A"):
        duration = float(stream.get("duration", 0.0))
        frame_count = int(round(duration * fps))
    else:
        frame_count = int(frame_count)
    return {
        "codec": stream.get("codec_name", "unknown"),
        "pix_fmt": stream.get("pix_fmt", "unknown"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": int(round(fps)),
        "frames": frame_count,
    }


def build_stats(values: np.ndarray | list[float]) -> dict[str, Any]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": arr.min(axis=0).astype(np.float64).tolist(),
        "max": arr.max(axis=0).astype(np.float64).tolist(),
        "mean": arr.mean(axis=0).astype(np.float64).tolist(),
        "std": arr.std(axis=0).astype(np.float64).tolist(),
        "count": [int(arr.shape[0])],
    }


def fixed_size_list_array(pa: Any, values: np.ndarray, width: int):
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def build_episode_data_table(pa: Any, frame_records: dict[str, list[Any]]) -> Any:
    state_array = np.asarray(frame_records["observation.state"], dtype=np.float32)
    action_array = np.asarray(frame_records["action"], dtype=np.float32)
    arrays = [fixed_size_list_array(pa, state_array, 2)]
    names = ["observation.state"]
    if DEFAULT_PATH_SIGNATURE_KEY in frame_records:
        signature_array = np.asarray(frame_records[DEFAULT_PATH_SIGNATURE_KEY], dtype=np.float32)
        arrays.append(
            fixed_size_list_array(pa, signature_array, int(signature_array.shape[1]))
        )
        names.append(DEFAULT_PATH_SIGNATURE_KEY)
    arrays.extend(
        [
            fixed_size_list_array(pa, action_array, 2),
            pa.array(frame_records["next.reward"], type=pa.float32()),
            pa.array(frame_records["next.done"], type=pa.bool_()),
            pa.array(frame_records["next.success"], type=pa.bool_()),
            pa.array(frame_records["timestamp"], type=pa.float32()),
            pa.array(frame_records["frame_index"], type=pa.int64()),
            pa.array(frame_records["episode_index"], type=pa.int64()),
            pa.array(frame_records["index"], type=pa.int64()),
            pa.array(frame_records["task_index"], type=pa.int64()),
        ]
    )
    names.extend(
        [
            "action",
            "next.reward",
            "next.done",
            "next.success",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]
    )
    return pa.Table.from_arrays(arrays, names=names)


def get_chunk_and_file_index(
    episode_index: int,
    episodes_per_chunk: int,
) -> tuple[int, int]:
    return int(episode_index // episodes_per_chunk), int(episode_index % episodes_per_chunk)


def estimate_total_size_mb(paths: list[Path]) -> int:
    if not paths:
        return 0
    total_bytes = sum(path.stat().st_size for path in paths if path.exists())
    return int(round(total_bytes / (1024 * 1024)))


def validate_lerobot_v30_consistency(
    records: dict[str, list[Any]],
    episodes_meta: list[dict[str, Any]],
    splits: dict[str, str],
    total_frames: int,
    total_episodes: int,
    video_frame_counts: dict[int, int],
) -> None:
    if total_frames != len(records["index"]):
        raise ValueError("total_frames mismatch with frame table")
    expected_indices = np.arange(total_frames, dtype=np.int64)
    if not np.array_equal(np.asarray(records["index"], dtype=np.int64), expected_indices):
        raise ValueError("global index must be continuous and monotonic")
    total_video_frames = int(sum(video_frame_counts.values()))
    if total_video_frames != total_frames:
        raise ValueError(
            f"video frame count mismatch: video={total_video_frames}, parquet={total_frames}"
        )
    total_length = sum(episode["length"] for episode in episodes_meta)
    if total_length != total_frames:
        raise ValueError("sum(episode.length) must equal total_frames")
    for episode_meta in episodes_meta:
        dataset_from = episode_meta["dataset_from_index"]
        dataset_to = episode_meta["dataset_to_index"]
        if not (0 <= dataset_from < dataset_to <= total_frames):
            raise ValueError(
                f"invalid dataset index range in episode {episode_meta['episode_index']}"
            )
        if (dataset_to - dataset_from) != episode_meta["length"]:
            raise ValueError(
                f"episode length mismatch in episode {episode_meta['episode_index']}"
            )
        video_frames = video_frame_counts.get(int(episode_meta["episode_index"]))
        if video_frames != int(episode_meta["length"]):
            raise ValueError(
                "per-episode video frame count mismatch in "
                f"episode {episode_meta['episode_index']}: "
                f"video={video_frames}, episode_length={episode_meta['length']}"
            )
    for split_name, split_spec in splits.items():
        split_start, split_end = split_spec.split(":", 1)
        split_from = int(split_start)
        split_to = int(split_end)
        if not (0 <= split_from <= split_to <= total_episodes):
            raise ValueError(f"invalid split range for {split_name}: {split_spec}")


def _make_replay_env(
    map_config: MapConfig,
    image_size: int,
    seed: int,
) -> PandaRouteSemanticEnv:
    if mujoco is None:
        raise RuntimeError(
            "mujoco is required to export Panda route videos. "
            "Install it first to generate the simulation dataset."
        ) from _MUJOCO_IMPORT_ERROR
    return PandaRouteMjEnv(
        map_config=map_config,
        rng_seed=seed,
        enable_randomize=False,
        image_size=image_size,
    )


def generate_lerobot_v30_dataset(
    processed_dataset: ProcessedDemonstrationDataset,
    output_dir: str | Path = DEFAULT_DATASET_ROOT,
    fps: int = DEFAULT_VIDEO_FPS,
    image_size: int = DEFAULT_VIDEO_IMAGE_SIZE,
    config: MapConfig | None = None,
    episodes_per_chunk: int = DEFAULT_LEROBOT_EPISODES_PER_CHUNK,
) -> Path:
    pa, pq = _require_lerobot_export_dependencies()
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if episodes_per_chunk <= 0:
        raise ValueError("episodes_per_chunk must be positive.")

    map_config = build_default_map_config() if config is None else config
    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)

    episodes_file = root / "meta/episodes/chunk-000/file-000.parquet"
    info_file = root / "meta/info.json"
    stats_file = root / "meta/stats.json"
    tasks_jsonl_file = root / "meta/tasks.jsonl"
    tasks_parquet_file = root / "meta/tasks.parquet"
    episodes_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_jsonl_file.parent.mkdir(parents=True, exist_ok=True)

    task_counts = validate_task_coverage(processed_dataset.task_ids)
    episode_order = build_balanced_episode_order(
        processed_dataset.task_ids,
        processed_dataset.seed,
    )
    print(
        "Exporting LeRobot dataset with balanced shuffled episodes: "
        + ", ".join(
            f"{TASK_ID_TO_GOAL_NAME[task_id]}={task_counts[task_id]}"
            for task_id in TASK_ID_VALUES
        )
    )

    env = _make_replay_env(map_config=map_config, image_size=image_size, seed=processed_dataset.seed)
    records: dict[str, list[Any]] = {
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
        "observation.state": [],
        "action": [],
        "next.reward": [],
        "next.done": [],
        "next.success": [],
    }
    if processed_dataset.path_signatures is not None:
        records[DEFAULT_PATH_SIGNATURE_KEY] = []
    episodes_meta: list[dict[str, Any]] = []
    global_index = 0
    video_frame_counts: dict[int, int] = {}
    data_files: list[Path] = []
    video_files: list[Path] = []
    first_video_info: dict[str, Any] | None = None

    try:
        for episode_idx, source_idx in enumerate(episode_order.tolist()):
            observations = processed_dataset.observations[source_idx]
            actions = processed_dataset.actions[source_idx]
            goal_xy = processed_dataset.goal_xy[source_idx]
            task_id = int(processed_dataset.task_ids[source_idx])
            task_text = TASK_DESCRIPTION_BY_ID[task_id]
            episode_length = int(observations.shape[0])
            episode_from_index = global_index
            chunk_index, file_index = get_chunk_and_file_index(
                episode_idx,
                episodes_per_chunk,
            )
            data_file = root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
            video_file = (
                root / f"videos/{VIDEO_KEY}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            )
            data_file.parent.mkdir(parents=True, exist_ok=True)
            video_file.parent.mkdir(parents=True, exist_ok=True)

            ffmpeg_proc = start_ffmpeg_raw_writer(video_file, image_size, image_size, fps)
            if ffmpeg_proc.stdin is None:
                raise RuntimeError("Failed to open ffmpeg stdin for raw video writing.")
            episode_records: dict[str, list[Any]] = {
                "timestamp": [],
                "frame_index": [],
                "episode_index": [],
                "index": [],
                "task_index": [],
                "observation.state": [],
                "action": [],
                "next.reward": [],
                "next.done": [],
                "next.success": [],
            }
            if processed_dataset.path_signatures is not None:
                episode_records[DEFAULT_PATH_SIGNATURE_KEY] = []

            env.reset(task_id=task_id, start_state=(float(observations[0, 0]), float(observations[0, 1])))
            for frame_idx in range(episode_length):
                state_xy = observations[frame_idx]
                action_xy = actions[frame_idx]
                signature_xy = (
                    None
                    if processed_dataset.path_signatures is None
                    else processed_dataset.path_signatures[source_idx, frame_idx]
                )
                # Render directly from the processed trajectory so export stays aligned
                # with the saved observations even if replay dynamics diverge slightly.
                env.sync_to_state(
                    (float(state_xy[0]), float(state_xy[1])),
                    step_count=frame_idx,
                )
                timestamp = frame_idx / fps
                done = frame_idx == episode_length - 1
                success = bool(done and processed_dataset.success[source_idx])
                distance_to_goal = float(
                    np.linalg.norm(state_xy.astype(np.float64) - goal_xy.astype(np.float64))
                )
                reward = 1.0 if success else -distance_to_goal
                frame_state = [float(state_xy[0]), float(state_xy[1])]
                frame_action = [float(action_xy[0]), float(action_xy[1])]
                for target_records in (records, episode_records):
                    target_records["timestamp"].append(float(timestamp))
                    target_records["frame_index"].append(int(frame_idx))
                    target_records["episode_index"].append(int(episode_idx))
                    target_records["index"].append(int(global_index))
                    target_records["task_index"].append(task_id)
                    target_records["observation.state"].append(frame_state)
                    target_records["action"].append(frame_action)
                    target_records["next.reward"].append(float(reward))
                    target_records["next.done"].append(bool(done))
                    target_records["next.success"].append(bool(success))
                    if signature_xy is not None:
                        target_records[DEFAULT_PATH_SIGNATURE_KEY].append(
                            signature_xy.astype(np.float32).tolist()
                        )
                frame = env.render_frame()
                ffmpeg_proc.stdin.write(frame.astype(np.uint8).tobytes())
                global_index += 1

            episode_to_index = global_index
            ffmpeg_proc.stdin.close()
            return_code = ffmpeg_proc.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"ffmpeg failed with code {return_code} for episode {episode_idx}"
                )
            episode_table = build_episode_data_table(pa, episode_records)
            pq.write_table(episode_table, data_file, compression="snappy")

            video_info = ffprobe_video(video_file)
            if first_video_info is None:
                first_video_info = video_info
            video_frame_counts[episode_idx] = int(video_info["frames"])
            data_files.append(data_file)
            video_files.append(video_file)

            episodes_meta.append(
                {
                    "episode_index": episode_idx,
                    "tasks": [task_text],
                    "length": episode_length,
                    "data/chunk_index": chunk_index,
                    "data/file_index": file_index,
                    "dataset_from_index": episode_from_index,
                    "dataset_to_index": episode_to_index,
                    f"videos/{VIDEO_KEY}/chunk_index": chunk_index,
                    f"videos/{VIDEO_KEY}/file_index": file_index,
                    f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
                    f"videos/{VIDEO_KEY}/to_timestamp": float(episode_length / fps),
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )
    finally:
        env.close()

    total_frames = len(records["index"])
    state_array = np.asarray(records["observation.state"], dtype=np.float32)
    action_array = np.asarray(records["action"], dtype=np.float32)
    signature_array = (
        None
        if processed_dataset.path_signatures is None
        else np.asarray(records[DEFAULT_PATH_SIGNATURE_KEY], dtype=np.float32)
    )

    episodes_table = pa.Table.from_arrays(
        [
            pa.array([episode["episode_index"] for episode in episodes_meta], type=pa.int64()),
            pa.array([episode["tasks"] for episode in episodes_meta], type=pa.list_(pa.string())),
            pa.array([episode["length"] for episode in episodes_meta], type=pa.int64()),
            pa.array([episode["data/chunk_index"] for episode in episodes_meta], type=pa.int64()),
            pa.array([episode["data/file_index"] for episode in episodes_meta], type=pa.int64()),
            pa.array([episode["dataset_from_index"] for episode in episodes_meta], type=pa.int64()),
            pa.array([episode["dataset_to_index"] for episode in episodes_meta], type=pa.int64()),
            pa.array([episode[f"videos/{VIDEO_KEY}/chunk_index"] for episode in episodes_meta], type=pa.int64()),
            pa.array([episode[f"videos/{VIDEO_KEY}/file_index"] for episode in episodes_meta], type=pa.int64()),
            pa.array([episode[f"videos/{VIDEO_KEY}/from_timestamp"] for episode in episodes_meta], type=pa.float32()),
            pa.array([episode[f"videos/{VIDEO_KEY}/to_timestamp"] for episode in episodes_meta], type=pa.float32()),
            pa.array([episode["meta/episodes/chunk_index"] for episode in episodes_meta], type=pa.int64()),
            pa.array([episode["meta/episodes/file_index"] for episode in episodes_meta], type=pa.int64()),
        ],
        names=[
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            f"videos/{VIDEO_KEY}/chunk_index",
            f"videos/{VIDEO_KEY}/file_index",
            f"videos/{VIDEO_KEY}/from_timestamp",
            f"videos/{VIDEO_KEY}/to_timestamp",
            "meta/episodes/chunk_index",
            "meta/episodes/file_index",
        ],
    )
    pq.write_table(episodes_table, episodes_file, compression="snappy")

    with tasks_jsonl_file.open("w", encoding="utf-8") as task_file:
        for task_id in TASK_ID_VALUES:
            task_file.write(
                json.dumps(
                    {"task_index": task_id, "task": TASK_DESCRIPTION_BY_ID[task_id]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    tasks_table = pa.Table.from_arrays(
        [
            pa.array(list(TASK_ID_VALUES), type=pa.int64()),
            pa.array(
                [TASK_DESCRIPTION_BY_ID[task_id] for task_id in TASK_ID_VALUES],
                type=pa.string(),
            ),
        ],
        names=["task_index", "task"],
    )
    pq.write_table(tasks_table, tasks_parquet_file, compression="snappy")

    if first_video_info is None:
        raise RuntimeError("No episodes were exported, so no video metadata was created.")

    total_episodes = len(processed_dataset)
    val_start = int(round(total_episodes * 0.8))
    splits = {
        "train": f"0:{val_start}",
        "val": f"{val_start}:{total_episodes}",
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "mujoco_panda_route_onebit",
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(len(TASK_ID_VALUES)),
        "chunks_size": int(episodes_per_chunk),
        "data_files_size_in_mb": estimate_total_size_mb(data_files),
        "video_files_size_in_mb": estimate_total_size_mb(video_files),
        "fps": int(fps),
        "splits": splits,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["probe_x", "probe_y"],
            },
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["delta_probe_x", "delta_probe_y"],
            },
            "next.reward": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "next.done": {
                "dtype": "bool",
                "shape": [1],
                "names": None,
            },
            "next.success": {
                "dtype": "bool",
                "shape": [1],
                "names": None,
            },
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [first_video_info["height"], first_video_info["width"], 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": first_video_info["height"],
                    "video.width": first_video_info["width"],
                    "video.codec": first_video_info["codec"],
                    "video.pix_fmt": first_video_info["pix_fmt"],
                    "video.is_depth_map": False,
                    "video.fps": int(fps),
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    if (
        processed_dataset.path_signatures is not None
        and processed_dataset.path_signature_key is not None
        and signature_array is not None
    ):
        info["features"][processed_dataset.path_signature_key] = {
            "dtype": "float32",
            "shape": [int(signature_array.shape[1])],
            "names": [f"path_sig_{index}" for index in range(int(signature_array.shape[1]))],
        }
        info["path_signature"] = {
            "key": processed_dataset.path_signature_key,
            "window_size": int(processed_dataset.path_signature_window_size),
            "window_mode": (
                "full_prefix"
                if int(processed_dataset.path_signature_window_size) <= 0
                else "sliding_window"
            ),
            "sig_depth": int(processed_dataset.path_signature_depth),
            "signature_dim": int(signature_array.shape[1]),
            "kind": (
                "signature"
                if processed_dataset.path_signature_backend == "signatory"
                else "simple_signature"
            ),
            "backend": processed_dataset.path_signature_backend,
        }

    stats = {
        "observation.state": build_stats(state_array),
        "action": build_stats(action_array),
        "next.reward": build_stats(np.asarray(records["next.reward"], dtype=np.float32)),
        "timestamp": build_stats(np.asarray(records["timestamp"], dtype=np.float32)),
    }
    if (
        processed_dataset.path_signatures is not None
        and processed_dataset.path_signature_key is not None
        and signature_array is not None
    ):
        stats[processed_dataset.path_signature_key] = build_stats(signature_array)

    with info_file.open("w", encoding="utf-8") as info_handle:
        json.dump(info, info_handle, indent=4, ensure_ascii=False)
    with stats_file.open("w", encoding="utf-8") as stats_handle:
        json.dump(stats, stats_handle, indent=4, ensure_ascii=False)

    validate_lerobot_v30_consistency(
        records=records,
        episodes_meta=episodes_meta,
        splits=splits,
        total_frames=total_frames,
        total_episodes=total_episodes,
        video_frame_counts=video_frame_counts,
    )
    print(f"Generated LeRobotDataset v3.0 at: {root.resolve()}")
    print(
        f"Episodes: {total_episodes}, Frames: {total_frames}, "
        f"Video frames: {sum(video_frame_counts.values())}, "
        f"Data files: {len(data_files)}, Video files: {len(video_files)}"
    )
    return root


def get_dataset_defaults() -> dict[str, Any]:
    return {
        "dataset_root": DEFAULT_DATASET_ROOT,
        "dataset_repo_id": DEFAULT_DATASET_REPO_ID,
        "output_dir": DEFAULT_DATASET_ROOT,
        "raw_output": DEFAULT_RAW_OUTPUT,
        "processed_output": DEFAULT_PROCESSED_OUTPUT,
        "num_per_task": DEFAULT_NUM_PER_TASK,
        "fps": DEFAULT_FPS,
        "image_size": DEFAULT_IMAGE_SIZE,
        "t_fixed": DEFAULT_T_FIXED,
        "episodes_per_chunk": DEFAULT_LEROBOT_EPISODES_PER_CHUNK,
    }


def get_train_defaults(policy_type: str) -> dict[str, Any]:
    defaults = dict(_TRAIN_DEFAULTS[policy_type])
    defaults.update(
        {
            "dataset_root": DEFAULT_DATASET_ROOT,
            "dataset_repo_id": DEFAULT_DATASET_REPO_ID,
        }
    )
    return defaults


def get_eval_defaults(policy_type: str) -> dict[str, Any]:
    return {
        "output_dir": _TRAIN_DEFAULTS[policy_type]["eval_output_dir"],
        "num_rollouts": DEFAULT_NUM_ROLLOUTS,
        "max_steps": DEFAULT_MAX_STEPS,
        "fps": DEFAULT_FPS,
        "max_action_step": DEFAULT_MAX_ACTION_STEP,
        "success_threshold": 0.0,
    }


def collect_dataset(args) -> Path:
    map_config = build_default_map_config()
    enable_randomize = bool(getattr(args, "enable_randomize", False))
    print(
        "[info] panda_route dataset collection reset mode: "
        + ("randomized starts" if enable_randomize else "deterministic center starts")
    )
    raw_dataset = generate_demonstrations(
        num_per_task=args.num_per_task,
        seed=args.seed,
        solve_time=args.solve_time,
        max_retries_per_demo=args.max_retries_per_demo,
        low_success_warning_threshold=args.low_success_warning_threshold,
        enable_randomize=enable_randomize,
        config=map_config,
    )

    raw_output = getattr(args, "raw_output", None)
    if raw_output is not None:
        save_demonstrations(raw_dataset, raw_output)

    processed_dataset = process_demonstration_dataset(
        raw_dataset,
        t_fixed=args.t_fixed,
        include_phase_labels=not args.disable_phase_labels,
        include_path_signatures=not args.disable_path_signature,
        path_signature_key=DEFAULT_PATH_SIGNATURE_KEY,
        path_signature_window_size=args.path_signature_window_size,
        path_signature_depth=args.path_signature_depth,
        path_signature_backend=args.signature_backend,
        last_action_mode=DEFAULT_LAST_ACTION_MODE,
        config=map_config,
    )
    dataset_summary(processed_dataset)

    processed_output = getattr(args, "processed_output", None)
    if processed_output is not None:
        save_processed_dataset(processed_dataset, processed_output)

    output_path = Path(args.output_dir)
    if args.skip_lerobot_export:
        return output_path

    generate_lerobot_v30_dataset(
        processed_dataset=processed_dataset,
        output_dir=output_path,
        fps=args.fps,
        image_size=args.image_size,
        config=map_config,
        episodes_per_chunk=args.episodes_per_chunk,
    )
    return output_path


def _compute_online_signature_prefix(
    state_history: deque[np.ndarray],
    sig_depth: int,
    signature_backend: str,
) -> np.ndarray:
    if len(state_history) == 0:
        raise ValueError("state_history is empty; cannot compute path signature.")
    window = np.stack(list(state_history), axis=0).astype(np.float32, copy=False)
    if signature_backend == "signatory":
        return compute_signatory_signature_np(window, sig_depth)
    return compute_simple_signature_np(window, sig_depth)


def build_balanced_task_schedule(num_rollouts: int, seed: int) -> list[int]:
    repeated = [TASK_ID_VALUES[index % len(TASK_ID_VALUES)] for index in range(num_rollouts)]
    rng = np.random.default_rng(seed)
    rng.shuffle(repeated)
    return [int(task_id) for task_id in repeated]


def detect_route_mismatch(task_spec: TaskSpec, phase_name: str) -> dict[str, str] | None:
    if phase_name in {"H1", "L1"} and phase_name != task_spec.branch_region_name:
        return {
            "stage": "branch",
            "expected_phase": task_spec.branch_region_name,
            "observed_phase": phase_name,
        }
    if phase_name in {"G1", "G2"} and phase_name != task_spec.target_goal_name:
        return {
            "stage": "goal",
            "expected_phase": task_spec.target_goal_name,
            "observed_phase": phase_name,
        }
    return None


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
    if len(cfg.image_features) == 0:
        raise RuntimeError("Policy has no image input feature; visual eval assumes image input.")
    image_key = list(cfg.image_features.keys())[0]
    image_shape = tuple(cfg.image_features[image_key].shape)
    image_hw = (int(image_shape[1]), int(image_shape[2]))
    if image_hw[0] != image_hw[1]:
        raise RuntimeError("panda_route evaluation expects square image inputs.")
    if cfg.robot_state_feature is None:
        raise RuntimeError("Policy has no observation.state feature.")
    state_key = "observation.state"
    state_dim = int(cfg.robot_state_feature.shape[0])

    use_path_signature = bool(
        policy_type == "streaming_act" and getattr(cfg, "use_path_signature", False)
    )
    signature_key = DEFAULT_PATH_SIGNATURE_KEY
    signature_backend = None
    if use_path_signature:
        signature_backend = resolve_signature_backend(args.signature_backend)
        if int(cfg.signature_depth) <= 0:
            raise ValueError(f"Invalid cfg.signature_depth={cfg.signature_depth}. Must be > 0.")
        if int(cfg.signature_dim) <= 0:
            raise ValueError(
                f"Invalid cfg.signature_dim={cfg.signature_dim}. "
                "Expected positive signature dimension when use_path_signature=True."
            )
        print(
            "[info] online path-signature enabled: "
            f"backend={signature_backend}, history=full_prefix, "
            f"depth={cfg.signature_depth}, dim={cfg.signature_dim}"
        )

    map_config = build_default_map_config()
    enable_randomize = bool(getattr(args, "enable_randomize", False))
    print(
        "[info] panda_route eval reset mode: "
        + ("randomized starts" if enable_randomize else "deterministic center starts")
    )
    env = PandaRouteMjEnv(
        map_config=map_config,
        rng_seed=args.seed,
        enable_randomize=enable_randomize,
        image_size=image_hw[0],
    )

    results = []
    success_count = 0
    route_failure_count = 0
    task_success_counts = {int(task_id): 0 for task_id in TASK_ID_VALUES}
    task_rollout_counts = {int(task_id): 0 for task_id in TASK_ID_VALUES}
    task_route_failure_counts = {int(task_id): 0 for task_id in TASK_ID_VALUES}
    task_schedule = build_balanced_task_schedule(args.num_rollouts, args.seed)

    try:
        for ep_idx, task_id in enumerate(task_schedule):
            task_spec = build_task_spec(task_id)
            task_rollout_counts[task_id] += 1
            state_xy = tuple(float(v) for v in env.reset(task_id=task_id))
            if env.start_region_name != task_spec.start_region_name:
                raise RuntimeError(
                    "Environment reset returned a start region that does not match the task. "
                    f"task={task_spec.task_code}, expected_start={task_spec.start_region_name}, "
                    f"got={env.start_region_name}"
                )
            if hasattr(policy, "reset"):
                policy.reset()

            episode_reward = 0.0
            video_path = output_dir / f"rollout_{ep_idx:03d}_task_{task_spec.task_code}.mp4"
            writer = start_ffmpeg_raw_writer(video_path, image_hw[1], image_hw[0], args.fps)
            if writer.stdin is None:
                raise RuntimeError("Failed to open ffmpeg stdin for rollout video writing.")

            trajectory = [state_xy]
            state_history = deque() if use_path_signature else None
            last_info = {
                **env.last_info,
                "phase_name": env.get_phase_name(state_xy),
                "route_mismatch": False,
                "route_mismatch_stage": None,
                "expected_phase": None,
                "observed_phase": None,
                "failure_reason": None,
            }

            for _step_idx in range(args.max_steps):
                frame = env.render_frame()
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
                    signature_vec = _compute_online_signature_prefix(
                        state_history=state_history,
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
                phase_name = env.get_phase_name(state_xy)
                mismatch = detect_route_mismatch(task_spec=task_spec, phase_name=phase_name)
                last_info = {
                    **info,
                    "phase_name": phase_name,
                    "route_mismatch": False,
                    "route_mismatch_stage": None,
                    "expected_phase": None,
                    "observed_phase": None,
                    "failure_reason": None,
                }
                if mismatch is not None:
                    last_info = {
                        **last_info,
                        "route_mismatch": True,
                        "route_mismatch_stage": mismatch["stage"],
                        "expected_phase": mismatch["expected_phase"],
                        "observed_phase": mismatch["observed_phase"],
                        "failure_reason": "wrong_route",
                        "success": False,
                    }
                    final_frame = env.render_frame()
                    writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                    break
                if done:
                    final_frame = env.render_frame()
                    writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                    break

            writer.stdin.close()
            code = writer.wait()
            if code != 0:
                raise RuntimeError(f"ffmpeg failed on rollout {ep_idx} with exit code {code}")

            success = bool(last_info.get("success", False)) and not bool(last_info.get("route_mismatch", False))
            if success:
                success_count += 1
                task_success_counts[task_id] += 1
            if last_info.get("route_mismatch", False):
                route_failure_count += 1
                task_route_failure_counts[task_id] += 1

            result = {
                "episode_index": ep_idx,
                "task_id": task_id,
                "task_code": task_spec.task_code,
                "start_region_name": str(last_info.get("start_region_name")),
                "target_goal_name": str(last_info.get("target_goal_name")),
                "video_path": str(video_path),
                "final_position": [float(state_xy[0]), float(state_xy[1])],
                "final_phase_name": str(last_info.get("phase_name")),
                "reached_goal": last_info.get("reached_goal"),
                "route_mismatch": bool(last_info.get("route_mismatch", False)),
                "route_mismatch_stage": last_info.get("route_mismatch_stage"),
                "expected_phase": last_info.get("expected_phase"),
                "observed_phase": last_info.get("observed_phase"),
                "failure_reason": last_info.get("failure_reason"),
                "success": success,
                "steps": int(len(trajectory) - 1),
                "sum_reward": float(episode_reward),
                "collision_rejections": int(
                    sum(
                        1
                        for index in range(1, len(env.trajectory))
                        if env.trajectory[index] == env.trajectory[index - 1]
                    )
                ),
            }
            results.append(result)
            print(
                f"[{ep_idx + 1:03d}/{args.num_rollouts:03d}] "
                f"task={task_spec.task_code}->{task_spec.target_goal_name} "
                f"success={success} steps={result['steps']} "
                f"reached={result['reached_goal']} "
                f"route_mismatch={result['route_mismatch']} "
                f"video={video_path.name}"
            )
    finally:
        env.close()

    per_task = {
        str(task_id): {
            "goal_name": TASK_ID_TO_GOAL_NAME[task_id],
            "rollouts": int(task_rollout_counts[task_id]),
            "success_count": int(task_success_counts[task_id]),
            "wrong_route_failures": int(task_route_failure_counts[task_id]),
            "success_rate": float(
                task_success_counts[task_id] / max(1, task_rollout_counts[task_id])
            ),
        }
        for task_id in TASK_ID_VALUES
    }
    summary = {
        "env": ENV_NAME,
        "policy_type": policy_type,
        "num_rollouts": args.num_rollouts,
        "success_count": success_count,
        "success_rate": float(success_count / max(1, args.num_rollouts)),
        "wrong_route_failures": route_failure_count,
        "seed": args.seed,
        "fps": args.fps,
        "max_steps": args.max_steps,
        "max_action_step": args.max_action_step,
        "start_randomized": enable_randomize,
        "policy_dir": str(policy_dir),
        "per_task": per_task,
        "results": results,
    }
    summary_path = write_summary(output_dir, summary)
    print(f"\nSaved {args.num_rollouts} rollout videos to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Success rate: {summary['success_rate']:.3f} ({success_count}/{args.num_rollouts})")
    print(f"Wrong-route failures: {route_failure_count}/{args.num_rollouts}")
    for task_id in TASK_ID_VALUES:
        task_summary = per_task[str(task_id)]
        print(
            f"  task {task_id} ({task_summary['goal_name']}): "
            f"{task_summary['success_count']}/{task_summary['rollouts']} "
            f"= {task_summary['success_rate']:.3f}, "
            f"wrong_route={task_summary['wrong_route_failures']}"
        )
