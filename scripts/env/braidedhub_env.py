from __future__ import annotations

from collections import deque
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
except Exception as exc:  # pragma: no cover - dependency guard
    plt = None
    Line2D = None
    Patch = None
    Rectangle = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

from eval_helpers import (
    build_eval_observation,
    build_prefix_sequence_eval_inputs,
    compute_delta_signature_sequence_np,
    compute_delta_signature_step_np,
    compute_signatory_signature_np,
    compute_simple_signature_np,
    ensure_prefix_sequence_batch_dims,
    resolve_signature_backend,
    resolve_single_visual_observation_feature,
    write_summary,
)
from policy_capabilities import (
    get_visual_memory_debug_stats,
    resolve_policy_capability_flags,
)
from policy_defaults import load_policy_mode_defaults


DEFAULT_RANDOM_SEED = 7
DEFAULT_NUM_FREE_SAMPLES = 500
DEFAULT_NUM_RAW_SAMPLES = 2000
DEFAULT_MAX_SAMPLE_ATTEMPTS = 10000
DEFAULT_DT = 1.0
DEFAULT_STEP_PENALTY = -0.01
DEFAULT_GOAL_REWARD = 1.0
DEFAULT_RANDOM_ROLLOUT_STEPS = 120
DEFAULT_RANDOM_ACTION_SCALE = 2.5
DEFAULT_RENDER_EVERY_N_STEPS = 5
DEFAULT_RENDER_PAUSE_SECONDS = 0.05
DEFAULT_MANUAL_TEST_RESETS = 4
DEFAULT_MANUAL_TEST_STEPS = 7
DEFAULT_MANUAL_PHASE_STEP_SIZE = 2.0
VALID_SAMPLE_COLOR = "#1f77b4"
RAW_VALID_SAMPLE_COLOR = "#9ecae1"
INVALID_SAMPLE_COLOR = "#d9d9d9"
RAW_SAMPLE_MARKER_SIZE = 10
FREE_SAMPLE_MARKER_SIZE = 18
ROBOT_MARKER_SIZE = 90
TRAJECTORY_LINEWIDTH = 2.0
TASK_ID_TO_GOAL_NAME = {
    0: "G00",
    1: "G01",
    2: "G10",
    3: "G11",
}
PHASE_COLOR_BY_NAME = {
    "start_region": "#4c78a8",
    "shared_corridor_region": "#72b7b2",
    "decision_region_H1": "#c0392b",
    "branch1_upper_region": "#2ca02c",
    "branch1_lower_region": "#9467bd",
    "merge_region_1": "#7f7f7f",
    "middle_corridor_region": "#5ab4ac",
    "decision_region_H2": "#ff7f0e",
    "branch2_upper_region": "#1b9e77",
    "branch2_lower_region": "#d95f02",
    "merge_region_2": "#8c8c8c",
    "final_corridor_region": "#80b1d3",
    "G00": "#1b9e77",
    "G01": "#66a61e",
    "G10": "#d95f02",
    "G11": "#e6ab02",
    "free_space_other": "#9ecae1",
    "obstacle": "#7f7f7f",
    "out_of_bounds": "#111111",
}
ENV_NAME = "braidedhub"
DEFAULT_DATASET_ROOT = Path("data/zeno-ai/braidedhub_fourstart_implicit_cue_v30")
DEFAULT_DATASET_REPO_ID = "zeno-ai/braidedhub_fourstart_implicit_cue_v30"
DEFAULT_NUM_PER_TASK = 25
DEFAULT_NUM_ROLLOUTS = 20
DEFAULT_MAX_STEPS = 240
DEFAULT_FPS = 20
DEFAULT_IMAGE_SIZE = 128
DEFAULT_MAX_ACTION_STEP = 2.5
DEFAULT_COLLISION_MODE = "reject"
DEFAULT_RAW_OUTPUT = None
DEFAULT_PROCESSED_OUTPUT = None
BRANCH1_PHASE_BY_BIT = {0: "branch1_upper_region", 1: "branch1_lower_region"}
BRANCH2_PHASE_BY_BIT = {0: "branch2_upper_region", 1: "branch2_lower_region"}
BRANCH1_PHASES = frozenset(BRANCH1_PHASE_BY_BIT.values())
BRANCH2_PHASES = frozenset(BRANCH2_PHASE_BY_BIT.values())
_TRAIN_DEFAULTS = {
    "act": {
        "output_root": Path("outputs/train/braidedhub/act-baseline"),
        "job_name": "act-baseline",
        "wandb_project": "braidedhub",
        "eval_output_dir": Path("outputs/eval/braidedhub/act-baseline"),
    },
    "streaming_act": {
        "output_root": Path("outputs/train/braidedhub/streaming-act-sipm"),
        "job_name": "streaming-act-sipm",
        "wandb_project": "braidedhub",
        "eval_output_dir": Path("outputs/eval/braidedhub/streaming-act-sipm"),
    },
}
DEFAULT_T_FIXED = 100
DEFAULT_LAST_ACTION_MODE = "zero"
DEFAULT_INCLUDE_PATH_SIGNATURES = True
DEFAULT_PATH_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_DELTA_SIGNATURE_KEY = "observation.delta_signature"
DEFAULT_SIGNATURE_WINDOW_SIZE = 0
DEFAULT_SIGNATURE_DEPTH = 3
DEFAULT_SIGNATURE_BACKEND = "auto"
DEFAULT_VIDEO_FPS = 20
DEFAULT_VIDEO_IMAGE_SIZE = 128
DEFAULT_LEROBOT_EPISODES_PER_CHUNK = 1000
DEFAULT_RESAMPLED_SEGMENT_CHECK_RESOLUTION = 0.1
VIDEO_KEY = "observation.images.front"
MAP_BACKGROUND_COLOR = (248, 244, 227)
OBSTACLE_COLOR = (74, 74, 74)
START_COLOR = (76, 120, 168)
GOAL_COLOR_BY_NAME = {
    "G00": (27, 158, 119),
    "G01": (102, 166, 30),
    "G10": (217, 95, 2),
    "G11": (230, 171, 2),
}
TARGET_GOAL_OUTLINE_COLOR = (255, 255, 255)
TARGET_GOAL_MARKER_OUTER_COLOR = (24, 24, 24)
TARGET_GOAL_MARKER_INNER_COLOR = (255, 255, 255)
TRAJECTORY_TRAIL_COLOR = (20, 20, 20)
ROBOT_OUTER_COLOR = (255, 255, 255)
ROBOT_INNER_COLOR = (240, 70, 70)
TASK_ID_VALUES = tuple(sorted(TASK_ID_TO_GOAL_NAME))
TASK_INDEX_BY_ID = {task_id: index for index, task_id in enumerate(TASK_ID_VALUES)}
TASK_DESCRIPTION_BY_ID = {
    0: "Start from S00, merge into the shared trunk, then take the upper branch at H1 and the upper branch at H2 to reach G00.",
    1: "Start from S01, merge into the shared trunk, then take the upper branch at H1 and the lower branch at H2 to reach G01.",
    2: "Start from S10, merge into the shared trunk, then take the lower branch at H1 and the upper branch at H2 to reach G10.",
    3: "Start from S11, merge into the shared trunk, then take the lower branch at H1 and the lower branch at H2 to reach G11.",
}
PHASE_LABEL_VOCAB = (
    "start_region",
    "shared_corridor_region",
    "decision_region_H1",
    "branch1_upper_region",
    "branch1_lower_region",
    "merge_region_1",
    "middle_corridor_region",
    "decision_region_H2",
    "branch2_upper_region",
    "branch2_lower_region",
    "merge_region_2",
    "final_corridor_region",
    "G00",
    "G01",
    "G10",
    "G11",
    "free_space_other",
    "obstacle",
    "out_of_bounds",
)
PHASE_NAME_TO_ID = {
    phase_name: phase_index for phase_index, phase_name in enumerate(PHASE_LABEL_VOCAB)
}
VALID_COLLISION_MODES = ("reject", "detect")


@dataclass(frozen=True)
class RectangleRegion:
    """Axis-aligned rectangle in world coordinates.

    Boundary convention:
    - Rectangle membership is closed: points on the rectangle boundary count
      as inside the rectangle.
    """

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
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def as_patch(self, **kwargs) -> Rectangle:
        if Rectangle is None:
            raise RuntimeError(
                "matplotlib is required for plotting. "
                "Install it first, for example: pip install matplotlib"
            ) from _MATPLOTLIB_IMPORT_ERROR
        return Rectangle((self.xmin, self.ymin), self.width, self.height, **kwargs)


@dataclass(frozen=True)
class RectangleObstacle(RectangleRegion):
    """Obstacle rectangle used by planners or collision checkers later."""


@dataclass(frozen=True)
class GoalRegion(RectangleRegion):
    """Terminal region on the right side of the map."""


@dataclass(frozen=True)
class DecisionPoint:
    """Named navigation decision point."""

    name: str
    x: float
    y: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class MapConfig:
    """Container for all geometry needed by the current static map."""

    workspace: RectangleRegion
    start_region: "SemanticRegion"
    task_start_regions: tuple[RectangleRegion, ...]
    shared_corridor_region: "SemanticRegion"
    decision_region_h1: "SemanticRegion"
    branch1_upper_region: "SemanticRegion"
    branch1_lower_region: "SemanticRegion"
    branch1_upper_detection_region: RectangleRegion
    branch1_lower_detection_region: RectangleRegion
    merge_region_1: "SemanticRegion"
    middle_corridor_region: "SemanticRegion"
    decision_region_h2: "SemanticRegion"
    branch2_upper_region: "SemanticRegion"
    branch2_lower_region: "SemanticRegion"
    branch2_upper_detection_region: RectangleRegion
    branch2_lower_detection_region: RectangleRegion
    merge_region_2: "SemanticRegion"
    final_corridor_region: "SemanticRegion"
    free_space_rectangles: tuple[RectangleRegion, ...]
    obstacle_rectangles: tuple[RectangleObstacle, ...]
    goal_regions: tuple[GoalRegion, ...]
    terminal_regions: tuple[GoalRegion, ...]
    decision_points: tuple[DecisionPoint, ...]
    path_labels: tuple[tuple[str, tuple[float, float]], ...]


@dataclass(frozen=True)
class TaskSpec:
    """Task definition for the 4-goal 2-bit implicit-cue setting."""

    task_id: int
    task_code: str
    task_bits: tuple[int, int]
    start_region_name: str
    target_goal_name: str


@dataclass(frozen=True)
class SemanticRegion:
    """Named semantic region approximated by a union of rectangles."""

    name: str
    rectangles: tuple[RectangleRegion, ...]

    def contains_point(self, x: float, y: float) -> bool:
        return any(rectangle.contains_point(x, y) for rectangle in self.rectangles)


@dataclass(frozen=True)
class TrajectoryPhaseAnnotation:
    """Phase labels and first-entry events for a 2D trajectory."""

    phase_labels: tuple[str, ...]
    first_h1_index: int | None
    first_h2_index: int | None
    first_terminal_index: int | None
    first_terminal_phase: str | None


class SupportsUniform(Protocol):
    """Minimal RNG protocol for workspace and free-space sampling."""

    def uniform(self, a: float, b: float) -> float: ...


def build_obstacles_from_free_rectangles(
    workspace: RectangleRegion,
    free_rectangles: tuple[RectangleRegion, ...],
) -> tuple[RectangleObstacle, ...]:
    """Convert workspace minus free rectangles into merged obstacle rectangles.

    The decomposition uses only the rectangle boundary coordinates already
    present in the map config. This keeps the representation simple and
    avoids pulling in a full boolean geometry library.
    """

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

    cell_is_obstacle: list[list[bool]] = []
    for y0, y1 in zip(y_coords[:-1], y_coords[1:]):
        row: list[bool] = []
        cy = (y0 + y1) / 2.0
        for x0, x1 in zip(x_coords[:-1], x_coords[1:]):
            cx = (x0 + x1) / 2.0
            is_free = any(rect.contains_point(cx, cy) for rect in free_rectangles)
            row.append(not is_free)
        cell_is_obstacle.append(row)

    merged_cells = [row[:] for row in cell_is_obstacle]
    obstacles: list[RectangleObstacle] = []
    obstacle_index = 0

    for row_idx in range(len(merged_cells)):
        for col_idx in range(len(merged_cells[row_idx])):
            if not merged_cells[row_idx][col_idx]:
                continue

            end_col = col_idx + 1
            while (
                end_col < len(merged_cells[row_idx]) and merged_cells[row_idx][end_col]
            ):
                end_col += 1

            end_row = row_idx + 1
            while end_row < len(merged_cells):
                if not all(
                    merged_cells[end_row][col] for col in range(col_idx, end_col)
                ):
                    break
                end_row += 1

            for clear_row in range(row_idx, end_row):
                for clear_col in range(col_idx, end_col):
                    merged_cells[clear_row][clear_col] = False

            obstacles.append(
                RectangleObstacle(
                    name=f"obs_{obstacle_index:02d}",
                    xmin=x_coords[col_idx],
                    xmax=x_coords[end_col],
                    ymin=y_coords[row_idx],
                    ymax=y_coords[end_row],
                )
            )
            obstacle_index += 1

    return tuple(obstacles)


def inset_rectangle_region(
    region: RectangleRegion,
    *,
    name: str,
    x_margin: float,
    y_margin: float,
) -> RectangleRegion:
    """Create a smaller inner rectangle used for event-style region checks."""

    if x_margin < 0.0 or y_margin < 0.0:
        raise ValueError("x_margin and y_margin must be non-negative.")
    if x_margin * 2.0 >= region.width or y_margin * 2.0 >= region.height:
        raise ValueError(
            f"Inset margins are too large for region {region.name}: "
            f"width={region.width}, height={region.height}, "
            f"x_margin={x_margin}, y_margin={y_margin}."
        )
    return RectangleRegion(
        name=name,
        xmin=region.xmin + x_margin,
        xmax=region.xmax - x_margin,
        ymin=region.ymin + y_margin,
        ymax=region.ymax - y_margin,
    )


def build_default_map_config() -> MapConfig:
    """Centralized geometry for the four-start implicit-cue braided map.

    Topology:
    - Four task-specific start rooms on the left encode the task implicitly.
    - The starts merge into one long shared snake corridor.
    - Two serial binary decisions (H1/H2) are separated by long shared segments.
    - The final corridor fans out to four goal rooms on the right.
    """

    workspace = RectangleRegion("workspace", xmin=0.0, xmax=100.0, ymin=0.0, ymax=60.0)

    start_s00 = RectangleRegion("S00", xmin=2.0, xmax=8.0, ymin=44.0, ymax=50.0)
    start_s01 = RectangleRegion("S01", xmin=2.0, xmax=8.0, ymin=34.0, ymax=40.0)
    start_s10 = RectangleRegion("S10", xmin=2.0, xmax=8.0, ymin=20.0, ymax=26.0)
    start_s11 = RectangleRegion("S11", xmin=2.0, xmax=8.0, ymin=10.0, ymax=16.0)
    task_start_regions = (start_s00, start_s01, start_s10, start_s11)

    start_00_prefix = RectangleRegion(
        "start_00_prefix", xmin=8.0, xmax=12.0, ymin=45.0, ymax=49.0
    )
    start_01_prefix = RectangleRegion(
        "start_01_prefix", xmin=8.0, xmax=12.0, ymin=35.0, ymax=39.0
    )
    start_10_prefix = RectangleRegion(
        "start_10_prefix", xmin=8.0, xmax=12.0, ymin=21.0, ymax=25.0
    )
    start_11_prefix = RectangleRegion(
        "start_11_prefix", xmin=8.0, xmax=12.0, ymin=11.0, ymax=15.0
    )
    start_collector = RectangleRegion(
        "start_collector", xmin=11.0, xmax=15.0, ymin=11.0, ymax=49.0
    )
    shared_entry = RectangleRegion(
        "shared_entry", xmin=14.0, xmax=18.0, ymin=27.0, ymax=33.0
    )

    shared_snake_1 = RectangleRegion(
        "shared_snake_1", xmin=18.0, xmax=24.0, ymin=27.0, ymax=33.0
    )
    shared_snake_2 = RectangleRegion(
        "shared_snake_2", xmin=21.0, xmax=25.0, ymin=27.0, ymax=43.0
    )
    shared_snake_3 = RectangleRegion(
        "shared_snake_3", xmin=24.0, xmax=30.0, ymin=37.0, ymax=43.0
    )
    shared_snake_4 = RectangleRegion(
        "shared_snake_4", xmin=29.0, xmax=33.0, ymin=23.0, ymax=43.0
    )
    shared_snake_5 = RectangleRegion(
        "shared_snake_5", xmin=31.0, xmax=35.0, ymin=23.0, ymax=29.0
    )

    h1_hub = RectangleRegion("H1_hub", xmin=33.0, xmax=39.0, ymin=19.0, ymax=37.0)
    branch1_upper_top = RectangleRegion(
        "branch1_upper_top", xmin=40.0, xmax=48.0, ymin=41.0, ymax=47.0
    )
    branch1_upper_rise = RectangleRegion(
        "branch1_upper_rise", xmin=37.0, xmax=41.0, ymin=33.0, ymax=47.0
    )
    branch1_upper_drop = RectangleRegion(
        "branch1_upper_drop", xmin=47.0, xmax=51.0, ymin=31.0, ymax=47.0
    )
    branch1_lower_bottom = RectangleRegion(
        "branch1_lower_bottom", xmin=40.0, xmax=48.0, ymin=9.0, ymax=15.0
    )
    branch1_lower_drop = RectangleRegion(
        "branch1_lower_drop", xmin=37.0, xmax=41.0, ymin=9.0, ymax=23.0
    )
    branch1_lower_rise = RectangleRegion(
        "branch1_lower_rise", xmin=47.0, xmax=51.0, ymin=9.0, ymax=29.0
    )
    merge1_hub = RectangleRegion("merge1_hub", xmin=49.0, xmax=55.0, ymin=23.0, ymax=37.0)

    middle_snake_1 = RectangleRegion(
        "middle_snake_1", xmin=54.0, xmax=60.0, ymin=27.0, ymax=33.0
    )
    middle_snake_2 = RectangleRegion(
        "middle_snake_2", xmin=59.0, xmax=63.0, ymin=27.0, ymax=41.0
    )
    middle_snake_3 = RectangleRegion(
        "middle_snake_3", xmin=62.0, xmax=68.0, ymin=35.0, ymax=41.0
    )
    middle_snake_4 = RectangleRegion(
        "middle_snake_4", xmin=67.0, xmax=71.0, ymin=17.0, ymax=41.0
    )
    middle_snake_5 = RectangleRegion(
        "middle_snake_5", xmin=69.0, xmax=73.0, ymin=17.0, ymax=23.0
    )

    h2_hub = RectangleRegion("H2_hub", xmin=71.0, xmax=77.0, ymin=13.0, ymax=37.0)
    branch2_upper_top = RectangleRegion(
        "branch2_upper_top", xmin=78.0, xmax=86.0, ymin=39.0, ymax=45.0
    )
    branch2_upper_rise = RectangleRegion(
        "branch2_upper_rise", xmin=75.0, xmax=79.0, ymin=31.0, ymax=45.0
    )
    branch2_upper_drop = RectangleRegion(
        "branch2_upper_drop", xmin=85.0, xmax=89.0, ymin=30.0, ymax=45.0
    )
    branch2_lower_bottom = RectangleRegion(
        "branch2_lower_bottom", xmin=78.0, xmax=86.0, ymin=7.0, ymax=13.0
    )
    branch2_lower_drop = RectangleRegion(
        "branch2_lower_drop", xmin=75.0, xmax=79.0, ymin=7.0, ymax=17.0
    )
    branch2_lower_rise = RectangleRegion(
        "branch2_lower_rise", xmin=85.0, xmax=89.0, ymin=7.0, ymax=29.0
    )
    merge2_hub = RectangleRegion("merge2_hub", xmin=87.0, xmax=93.0, ymin=23.0, ymax=37.0)
    final_corridor = RectangleRegion(
        "final_corridor", xmin=91.0, xmax=97.0, ymin=27.0, ymax=33.0
    )
    terminal_hub = RectangleRegion("terminal_hub", xmin=95.0, xmax=99.0, ymin=9.0, ymax=51.0)

    goal_regions = (
        GoalRegion("G00", xmin=98.0, xmax=100.0, ymin=42.0, ymax=50.0),
        GoalRegion("G01", xmin=98.0, xmax=100.0, ymin=32.0, ymax=38.0),
        GoalRegion("G10", xmin=98.0, xmax=100.0, ymin=22.0, ymax=28.0),
        GoalRegion("G11", xmin=98.0, xmax=100.0, ymin=10.0, ymax=16.0),
    )

    decision_points = (
        DecisionPoint("H1", x=36.0, y=28.0),
        DecisionPoint("H2", x=74.0, y=25.0),
    )

    start_region = SemanticRegion(
        "start_region",
        rectangles=(
            *task_start_regions,
            start_00_prefix,
            start_01_prefix,
            start_10_prefix,
            start_11_prefix,
            start_collector,
        ),
    )
    shared_corridor_region = SemanticRegion(
        "shared_corridor_region",
        rectangles=(
            shared_entry,
            shared_snake_1,
            shared_snake_2,
            shared_snake_3,
            shared_snake_4,
            shared_snake_5,
        ),
    )
    decision_region_h1 = SemanticRegion("decision_region_H1", rectangles=(h1_hub,))
    branch1_upper_region = SemanticRegion(
        "branch1_upper_region",
        rectangles=(branch1_upper_top, branch1_upper_rise, branch1_upper_drop),
    )
    branch1_lower_region = SemanticRegion(
        "branch1_lower_region",
        rectangles=(branch1_lower_bottom, branch1_lower_drop, branch1_lower_rise),
    )
    branch1_upper_detection_region = inset_rectangle_region(
        branch1_upper_top,
        name="branch1_upper_detection_region",
        x_margin=2.0,
        y_margin=1.0,
    )
    branch1_lower_detection_region = inset_rectangle_region(
        branch1_lower_bottom,
        name="branch1_lower_detection_region",
        x_margin=2.0,
        y_margin=1.0,
    )
    merge_region_1 = SemanticRegion("merge_region_1", rectangles=(merge1_hub,))
    middle_corridor_region = SemanticRegion(
        "middle_corridor_region",
        rectangles=(
            middle_snake_1,
            middle_snake_2,
            middle_snake_3,
            middle_snake_4,
            middle_snake_5,
        ),
    )
    decision_region_h2 = SemanticRegion("decision_region_H2", rectangles=(h2_hub,))
    branch2_upper_region = SemanticRegion(
        "branch2_upper_region",
        rectangles=(branch2_upper_top, branch2_upper_rise, branch2_upper_drop),
    )
    branch2_lower_region = SemanticRegion(
        "branch2_lower_region",
        rectangles=(branch2_lower_bottom, branch2_lower_drop, branch2_lower_rise),
    )
    branch2_upper_detection_region = inset_rectangle_region(
        branch2_upper_top,
        name="branch2_upper_detection_region",
        x_margin=2.0,
        y_margin=1.0,
    )
    branch2_lower_detection_region = inset_rectangle_region(
        branch2_lower_bottom,
        name="branch2_lower_detection_region",
        x_margin=2.0,
        y_margin=1.0,
    )
    merge_region_2 = SemanticRegion("merge_region_2", rectangles=(merge2_hub,))
    final_corridor_region = SemanticRegion(
        "final_corridor_region",
        rectangles=(final_corridor, terminal_hub),
    )

    free_space_rectangles = (
        *task_start_regions,
        start_00_prefix,
        start_01_prefix,
        start_10_prefix,
        start_11_prefix,
        start_collector,
        shared_entry,
        shared_snake_1,
        shared_snake_2,
        shared_snake_3,
        shared_snake_4,
        shared_snake_5,
        h1_hub,
        branch1_upper_rise,
        branch1_upper_top,
        branch1_upper_drop,
        branch1_lower_drop,
        branch1_lower_bottom,
        branch1_lower_rise,
        merge1_hub,
        middle_snake_1,
        middle_snake_2,
        middle_snake_3,
        middle_snake_4,
        middle_snake_5,
        h2_hub,
        branch2_upper_rise,
        branch2_upper_top,
        branch2_upper_drop,
        branch2_lower_drop,
        branch2_lower_bottom,
        branch2_lower_rise,
        merge2_hub,
        final_corridor,
        terminal_hub,
        *goal_regions,
    )
    obstacle_rectangles = build_obstacles_from_free_rectangles(
        workspace, free_space_rectangles
    )

    path_labels = (
        ("start cue", (8.0, 53.0)),
        ("shared trunk 1", (25.0, 46.0)),
        ("decision 1", (39.0, 50.0)),
        ("shared trunk 2", (63.0, 44.0)),
        ("decision 2", (82.0, 48.0)),
        ("goal fan-out", (97.0, 54.0)),
    )

    return MapConfig(
        workspace=workspace,
        start_region=start_region,
        task_start_regions=task_start_regions,
        shared_corridor_region=shared_corridor_region,
        decision_region_h1=decision_region_h1,
        branch1_upper_region=branch1_upper_region,
        branch1_lower_region=branch1_lower_region,
        branch1_upper_detection_region=branch1_upper_detection_region,
        branch1_lower_detection_region=branch1_lower_detection_region,
        merge_region_1=merge_region_1,
        middle_corridor_region=middle_corridor_region,
        decision_region_h2=decision_region_h2,
        branch2_upper_region=branch2_upper_region,
        branch2_lower_region=branch2_lower_region,
        branch2_upper_detection_region=branch2_upper_detection_region,
        branch2_lower_detection_region=branch2_lower_detection_region,
        merge_region_2=merge_region_2,
        final_corridor_region=final_corridor_region,
        free_space_rectangles=free_space_rectangles,
        obstacle_rectangles=obstacle_rectangles,
        goal_regions=goal_regions,
        terminal_regions=goal_regions,
        decision_points=decision_points,
        path_labels=path_labels,
    )


@lru_cache(maxsize=1)
def get_default_map_config() -> MapConfig:
    """Return a cached default map config for helper functions."""

    return build_default_map_config()


def _resolve_config(config: MapConfig | None) -> MapConfig:
    return get_default_map_config() if config is None else config


def build_task_spec(task_id: int) -> TaskSpec:
    """Map a task id in {0,1,2,3} to a 2-bit code and goal name."""

    if task_id not in TASK_ID_TO_GOAL_NAME:
        raise ValueError(
            f"Unsupported task_id={task_id}. Expected one of {sorted(TASK_ID_TO_GOAL_NAME)}."
        )
    task_code = format(task_id, "02b")
    return TaskSpec(
        task_id=task_id,
        task_code=task_code,
        task_bits=(int(task_code[0]), int(task_code[1])),
        start_region_name=f"S{task_code}",
        target_goal_name=TASK_ID_TO_GOAL_NAME[task_id],
    )


def get_task_start_region(
    task_id: int,
    config: MapConfig | None = None,
) -> RectangleRegion:
    """Return the task-specific implicit-cue start room for a task id."""

    resolved_config = _resolve_config(config)
    start_region_name = build_task_spec(task_id).start_region_name
    for start_region in resolved_config.task_start_regions:
        if start_region.name == start_region_name:
            return start_region
    raise RuntimeError(
        f"Start region {start_region_name} for task_id={task_id} was not found."
    )


def get_task_waypoint_centers(
    task_id: int,
    config: MapConfig | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the two branch waypoints that encode the task's implicit cue."""

    resolved_config = _resolve_config(config)
    task_bits = build_task_spec(task_id).task_bits

    branch1_center = (
        resolved_config.branch1_lower_region.rectangles[0].center
        if task_bits[0] == 1
        else resolved_config.branch1_upper_region.rectangles[0].center
    )
    branch2_center = (
        resolved_config.branch2_lower_region.rectangles[0].center
        if task_bits[1] == 1
        else resolved_config.branch2_upper_region.rectangles[0].center
    )
    return (branch1_center, branch2_center)


def is_in_bounds(x: float, y: float, config: MapConfig | None = None) -> bool:
    """Check whether a point lies inside the global workspace.

    Boundary convention:
    - The workspace is treated as a closed set, so boundary points are in-bounds.
    """

    resolved_config = _resolve_config(config)
    return resolved_config.workspace.contains_point(x, y)


def is_in_obstacle(x: float, y: float, config: MapConfig | None = None) -> bool:
    """Check whether a point lies inside any obstacle rectangle.

    Boundary convention:
    - Obstacles are treated as closed sets, so points on obstacle edges count
      as collisions. This makes corridor walls conservative and unambiguous.
    """

    resolved_config = _resolve_config(config)
    return any(
        obstacle.contains_point(x, y)
        for obstacle in resolved_config.obstacle_rectangles
    )


def is_state_valid(x: float, y: float, config: MapConfig | None = None) -> bool:
    """Check whether a point-robot state is valid in the current map."""

    return is_in_bounds(x, y, config=config) and not is_in_obstacle(x, y, config=config)


def is_line_segment_valid(
    from_point: tuple[float, float],
    to_point: tuple[float, float],
    config: MapConfig | None = None,
    resolution: float = DEFAULT_RESAMPLED_SEGMENT_CHECK_RESOLUTION,
) -> bool:
    """Check whether a straight-line sweep between two states stays in free space."""

    if resolution <= 0.0:
        raise ValueError("resolution must be positive.")

    resolved_config = _resolve_config(config)
    distance = math.dist(from_point, to_point)
    num_checks = max(1, int(math.ceil(distance / resolution)))
    for check_index in range(num_checks + 1):
        t = check_index / num_checks
        point = (
            float(from_point[0] * (1.0 - t) + to_point[0] * t),
            float(from_point[1] * (1.0 - t) + to_point[1] * t),
        )
        if not is_state_valid(*point, config=resolved_config):
            return False
    return True


def sample_workspace_state(
    rng: SupportsUniform,
    config: MapConfig | None = None,
) -> tuple[float, float]:
    """Sample a point uniformly from the workspace bounding box."""

    resolved_config = _resolve_config(config)
    workspace = resolved_config.workspace
    return (
        float(rng.uniform(workspace.xmin, workspace.xmax)),
        float(rng.uniform(workspace.ymin, workspace.ymax)),
    )


def sample_free_state(
    rng: SupportsUniform,
    config: MapConfig | None = None,
    max_attempts: int = DEFAULT_MAX_SAMPLE_ATTEMPTS,
) -> tuple[float, float]:
    """Rejection-sample one valid point-robot state from free space."""

    resolved_config = _resolve_config(config)
    for _ in range(max_attempts):
        x, y = sample_workspace_state(rng, config=resolved_config)
        if is_state_valid(x, y, config=resolved_config):
            return (x, y)

    raise RuntimeError(
        "Failed to sample a valid free state within the attempt budget. "
        "Check the map geometry or increase max_attempts."
    )


def sample_valid_state_in_region(
    region: RectangleRegion,
    rng: SupportsUniform,
    config: MapConfig | None = None,
    max_attempts: int = DEFAULT_MAX_SAMPLE_ATTEMPTS,
) -> tuple[float, float]:
    """Sample a valid state constrained to a specific region."""

    resolved_config = _resolve_config(config)
    for _ in range(max_attempts):
        x = float(rng.uniform(region.xmin, region.xmax))
        y = float(rng.uniform(region.ymin, region.ymax))
        if is_state_valid(x, y, config=resolved_config):
            return (x, y)

    raise RuntimeError(
        f"Failed to sample a valid state from region {region.name}. "
        "Check whether the region overlaps free space."
    )


def get_goal_region_for_state(
    x: float,
    y: float,
    config: MapConfig | None = None,
) -> GoalRegion | None:
    """Return the goal region containing the current point, if any."""

    resolved_config = _resolve_config(config)
    for goal_region in resolved_config.goal_regions:
        if goal_region.contains_point(x, y):
            return goal_region
    return None


def get_phase_name(x: float, y: float, config: MapConfig | None = None) -> str:
    """Return the semantic phase label for a 2D point.

    Priority is chosen so branch and merge events remain stable near region
    boundaries: terminal > final > merge2 > branch2 > H2 > middle > merge1 >
    branch1 > H1 > start > shared corridor.
    """

    resolved_config = _resolve_config(config)

    if not is_in_bounds(x, y, config=resolved_config):
        return "out_of_bounds"
    if is_in_obstacle(x, y, config=resolved_config):
        return "obstacle"

    terminal_region = get_goal_region_for_state(x, y, config=resolved_config)
    if terminal_region is not None:
        return terminal_region.name
    if resolved_config.final_corridor_region.contains_point(x, y):
        return resolved_config.final_corridor_region.name
    if resolved_config.merge_region_2.contains_point(x, y):
        return resolved_config.merge_region_2.name
    if resolved_config.branch2_upper_region.contains_point(x, y):
        return resolved_config.branch2_upper_region.name
    if resolved_config.branch2_lower_region.contains_point(x, y):
        return resolved_config.branch2_lower_region.name
    if resolved_config.decision_region_h2.contains_point(x, y):
        return resolved_config.decision_region_h2.name
    if resolved_config.middle_corridor_region.contains_point(x, y):
        return resolved_config.middle_corridor_region.name
    if resolved_config.merge_region_1.contains_point(x, y):
        return resolved_config.merge_region_1.name
    if resolved_config.branch1_upper_region.contains_point(x, y):
        return resolved_config.branch1_upper_region.name
    if resolved_config.branch1_lower_region.contains_point(x, y):
        return resolved_config.branch1_lower_region.name
    if resolved_config.decision_region_h1.contains_point(x, y):
        return resolved_config.decision_region_h1.name
    if resolved_config.start_region.contains_point(x, y):
        return "start_region"
    if resolved_config.shared_corridor_region.contains_point(x, y):
        return resolved_config.shared_corridor_region.name
    return "free_space_other"


def get_branch_detection_event(
    x: float,
    y: float,
    config: MapConfig | None = None,
) -> tuple[str, str] | None:
    """Return a stage/phase tuple only inside the narrowed branch detection windows."""

    resolved_config = _resolve_config(config)
    detection_regions = (
        ("H1", "branch1_upper_region", resolved_config.branch1_upper_detection_region),
        ("H1", "branch1_lower_region", resolved_config.branch1_lower_detection_region),
        ("H2", "branch2_upper_region", resolved_config.branch2_upper_detection_region),
        ("H2", "branch2_lower_region", resolved_config.branch2_lower_detection_region),
    )
    for stage_name, phase_name, detection_region in detection_regions:
        if detection_region.contains_point(x, y):
            return stage_name, phase_name
    return None


def _first_index_of_phase(
    phase_labels: tuple[str, ...], target_phase: str
) -> int | None:
    for index, phase_label in enumerate(phase_labels):
        if phase_label == target_phase:
            return index
    return None


def annotate_trajectory_phases(
    path: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    config: MapConfig | None = None,
) -> TrajectoryPhaseAnnotation:
    """Annotate each trajectory point with a semantic phase label."""

    phase_labels = tuple(get_phase_name(x, y, config=config) for x, y in path)
    first_h1_index = _first_index_of_phase(phase_labels, "decision_region_H1")
    first_h2_index = _first_index_of_phase(phase_labels, "decision_region_H2")

    first_terminal_index = None
    first_terminal_phase = None
    resolved_config = _resolve_config(config)
    terminal_phase_names = {
        goal_region.name for goal_region in resolved_config.terminal_regions
    }
    for index, phase_label in enumerate(phase_labels):
        if phase_label in terminal_phase_names:
            first_terminal_index = index
            first_terminal_phase = phase_label
            break

    return TrajectoryPhaseAnnotation(
        phase_labels=phase_labels,
        first_h1_index=first_h1_index,
        first_h2_index=first_h2_index,
        first_terminal_index=first_terminal_index,
        first_terminal_phase=first_terminal_phase,
    )


def densify_polyline_path(
    waypoints: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    step_size: float = DEFAULT_MANUAL_PHASE_STEP_SIZE,
) -> list[tuple[float, float]]:
    """Linearly densify a waypoint polyline for trajectory analysis."""

    if not waypoints:
        return []
    if len(waypoints) == 1:
        return [tuple(map(float, waypoints[0]))]
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")

    dense_path: list[tuple[float, float]] = [tuple(map(float, waypoints[0]))]
    for start, end in zip(waypoints[:-1], waypoints[1:], strict=False):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length = (dx * dx + dy * dy) ** 0.5
        num_steps = max(1, int(segment_length / step_size))
        for step_idx in range(1, num_steps + 1):
            t = step_idx / num_steps
            dense_path.append(
                (
                    float(start[0] * (1.0 - t) + end[0] * t),
                    float(start[1] * (1.0 - t) + end[1] * t),
                )
            )
    return dense_path


def build_manual_upper_goal_test_path(
    step_size: float = DEFAULT_MANUAL_PHASE_STEP_SIZE,
) -> list[tuple[float, float]]:
    """Construct a hand-made S00-to-G00 trajectory through both upper loops."""

    pre_bend_path = densify_polyline_path(
        [
            (5.0, 47.0),
            (10.0, 47.0),
            (13.0, 47.0),
            (13.0, 30.0),
            (21.0, 30.0),
            (23.0, 30.0),
            (23.0, 40.0),
            (27.0, 40.0),
            (31.0, 40.0),
            (31.0, 26.0),
            (35.0, 26.0),
            (36.0, 35.0),
            (39.0, 35.0),
            (39.0, 41.0),
            (44.0, 44.0),
            (49.0, 44.0),
            (49.0, 34.0),
            (52.0, 30.0),
            (58.0, 30.0),
            (61.0, 30.0),
            (61.0, 38.0),
            (65.0, 38.0),
            (69.0, 38.0),
            (69.0, 20.0),
            (73.0, 20.0),
            (74.0, 33.0),
            (77.0, 33.0),
            (77.0, 42.0),
            (82.0, 42.0),
            (87.0, 42.0),
            (87.0, 34.0),
            (90.0, 34.0),
            (90.0, 30.0),
            (94.0, 30.0),
        ],
        step_size=step_size,
    )
    post_bend_path = densify_polyline_path(
        [
            (97.0, 30.0),
            (97.0, 46.0),
            (99.0, 46.0),
        ],
        step_size=step_size,
    )
    return pre_bend_path + post_bend_path


def plot_map(config: MapConfig, show: bool = True, ax=None):
    """Plot the rectangle-based 2D map."""

    if plt is None or Line2D is None or Patch is None:
        raise RuntimeError(
            "matplotlib is required to plot this map. "
            "Install it first, for example: pip install matplotlib"
        ) from _MATPLOTLIB_IMPORT_ERROR

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 7))
    else:
        fig = ax.figure
        ax.clear()

    workspace_patch = config.workspace.as_patch(
        facecolor="#f8f4e3",
        edgecolor="black",
        linewidth=2.5,
        zorder=0,
    )
    ax.add_patch(workspace_patch)

    for obstacle in config.obstacle_rectangles:
        ax.add_patch(
            obstacle.as_patch(
                facecolor="#4a4a4a",
                edgecolor="#2f2f2f",
                linewidth=0.8,
                zorder=1,
            )
        )

    for start_region in config.task_start_regions:
        ax.add_patch(
            start_region.as_patch(
                facecolor="#4c78a8",
                edgecolor="#1f3552",
                linewidth=1.2,
                alpha=0.95,
                zorder=3,
            )
        )
        start_x, start_y = start_region.center
        ax.text(
            start_x,
            start_y,
            start_region.name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    goal_colors = {
        "G00": "#1b9e77",
        "G01": "#66a61e",
        "G10": "#d95f02",
        "G11": "#e6ab02",
    }
    for goal in config.goal_regions:
        ax.add_patch(
            goal.as_patch(
                facecolor=goal_colors[goal.name],
                edgecolor="black",
                linewidth=1.2,
                alpha=0.9,
                zorder=3,
            )
        )
        goal_x, goal_y = goal.center
        ax.text(
            goal_x,
            goal_y,
            goal.name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    for label, (x, y) in config.path_labels:
        ax.text(
            x,
            y,
            label,
            fontsize=10,
            color="#5b5b5b",
            ha="center",
            va="center",
            zorder=2,
        )

    legend_handles = build_base_legend_handles()
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)

    ax.set_xlim(config.workspace.xmin - 1.0, config.workspace.xmax + 1.0)
    ax.set_ylim(config.workspace.ymin - 1.0, config.workspace.ymax + 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Four-Start Implicit-Cue Map with Shared Snake Corridor and Two Serial Binary Branches")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.4)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def build_base_legend_handles():
    """Legend entries shared by the static map plot and debug plots."""

    if Line2D is None or Patch is None:
        raise RuntimeError(
            "matplotlib is required to build legend handles. "
            "Install it first, for example: pip install matplotlib"
        ) from _MATPLOTLIB_IMPORT_ERROR

    return [
        Patch(facecolor="#4a4a4a", edgecolor="#2f2f2f", label="Obstacle"),
        Patch(facecolor="#4c78a8", edgecolor="#1f3552", label="Task Start Room"),
        Patch(facecolor="#1b9e77", edgecolor="black", label="Goal Region"),
    ]


def debug_plot_samples(
    config: MapConfig,
    rng: SupportsUniform,
    num_free_samples: int = DEFAULT_NUM_FREE_SAMPLES,
    num_raw_samples: int = DEFAULT_NUM_RAW_SAMPLES,
    show: bool = True,
):
    """Overlay valid and invalid point samples on top of the static map."""

    fig, ax = plot_map(config, show=False)

    raw_valid_points: list[tuple[float, float]] = []
    raw_invalid_points: list[tuple[float, float]] = []
    for _ in range(num_raw_samples):
        x, y = sample_workspace_state(rng, config=config)
        if is_state_valid(x, y, config=config):
            raw_valid_points.append((x, y))
        else:
            raw_invalid_points.append((x, y))

    free_points = [
        sample_free_state(rng, config=config) for _ in range(num_free_samples)
    ]

    if raw_invalid_points:
        invalid_xs, invalid_ys = zip(*raw_invalid_points, strict=False)
        ax.scatter(
            invalid_xs,
            invalid_ys,
            s=RAW_SAMPLE_MARKER_SIZE,
            c=INVALID_SAMPLE_COLOR,
            alpha=0.55,
            linewidths=0.0,
            zorder=2,
        )

    if raw_valid_points:
        raw_valid_xs, raw_valid_ys = zip(*raw_valid_points, strict=False)
        ax.scatter(
            raw_valid_xs,
            raw_valid_ys,
            s=RAW_SAMPLE_MARKER_SIZE,
            c=RAW_VALID_SAMPLE_COLOR,
            alpha=0.35,
            linewidths=0.0,
            zorder=2,
        )

    if free_points:
        free_xs, free_ys = zip(*free_points, strict=False)
        ax.scatter(
            free_xs,
            free_ys,
            s=FREE_SAMPLE_MARKER_SIZE,
            c=VALID_SAMPLE_COLOR,
            alpha=0.9,
            linewidths=0.0,
            zorder=4,
        )

    ax.set_title(
        "Long-Strip Implicit-Cue Map with Point-State Validity Sampling\n"
        f"{num_free_samples} accepted free samples, {num_raw_samples} raw workspace samples"
    )
    legend_handles = build_base_legend_handles()
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=INVALID_SAMPLE_COLOR,
                markeredgecolor=INVALID_SAMPLE_COLOR,
                markersize=6,
                label="Invalid Samples",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=RAW_VALID_SAMPLE_COLOR,
                markeredgecolor=RAW_VALID_SAMPLE_COLOR,
                markersize=6,
                label="Raw Valid Samples",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=VALID_SAMPLE_COLOR,
                markeredgecolor=VALID_SAMPLE_COLOR,
                markersize=7,
                label="Accepted Free Samples",
            ),
        ]
    )
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_trajectory_with_phases(
    path: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    config: MapConfig | None = None,
    show: bool = True,
):
    """Plot a trajectory and highlight the semantic phases it traverses."""

    if plt is None or Line2D is None:
        raise RuntimeError(
            "matplotlib is required to plot trajectory phases. "
            "Install it first, for example: pip install matplotlib"
        ) from _MATPLOTLIB_IMPORT_ERROR

    resolved_config = _resolve_config(config)
    annotation = annotate_trajectory_phases(path, config=resolved_config)
    fig, ax = plot_map(resolved_config, show=False)

    if path:
        path_xs = [point[0] for point in path]
        path_ys = [point[1] for point in path]
        ax.plot(
            path_xs,
            path_ys,
            color="#333333",
            linewidth=1.8,
            alpha=0.65,
            zorder=5,
        )

    phase_points: dict[str, list[tuple[float, float]]] = {}
    for point, phase_label in zip(path, annotation.phase_labels, strict=False):
        phase_points.setdefault(phase_label, []).append(point)

    for phase_label, points in phase_points.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.scatter(
            xs,
            ys,
            s=28,
            c=PHASE_COLOR_BY_NAME.get(phase_label, "#666666"),
            alpha=0.9,
            linewidths=0.0,
            zorder=6,
        )

    event_specs = [
        ("H1", annotation.first_h1_index, "#c0392b"),
        ("H2", annotation.first_h2_index, "#ff7f0e"),
        ("T", annotation.first_terminal_index, "#111111"),
    ]
    for event_name, event_index, color in event_specs:
        if event_index is None:
            continue
        event_x, event_y = path[event_index]
        ax.scatter(
            event_x,
            event_y,
            s=120,
            c=color,
            marker="X",
            edgecolors="white",
            linewidths=1.0,
            zorder=8,
        )
        ax.text(
            event_x + 1.2,
            event_y + 1.2,
            f"{event_name}@{event_index}",
            fontsize=9,
            color=color,
            fontweight="bold",
            zorder=9,
        )

    phase_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PHASE_COLOR_BY_NAME.get(phase_label, "#666666"),
            markeredgecolor=PHASE_COLOR_BY_NAME.get(phase_label, "#666666"),
            markersize=7,
            label=phase_label,
        )
        for phase_label in phase_points
    ]
    event_handles = [
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="#c0392b",
            markeredgecolor="white",
            markersize=8,
            label="First H1 Entry",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="#ff7f0e",
            markeredgecolor="white",
            markersize=8,
            label="First H2 Entry",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="#111111",
            markeredgecolor="white",
            markersize=8,
            label="First Terminal Entry",
        ),
    ]
    ax.legend(
        handles=build_base_legend_handles() + phase_handles + event_handles,
        loc="upper left",
        frameon=True,
    )
    ax.set_title(
        "Trajectory Phase Annotation\n"
        f"first_h1={annotation.first_h1_index}, "
        f"first_h2={annotation.first_h2_index}, "
        f"first_terminal={annotation.first_terminal_index} ({annotation.first_terminal_phase})"
    )

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax, annotation


def manual_test_phase_annotation(
    config: MapConfig | None = None,
    step_size: float = DEFAULT_MANUAL_PHASE_STEP_SIZE,
    show_plot: bool = True,
) -> dict[str, Any]:
    """Run a simple hand-crafted upper-goal path through the phase annotator."""

    resolved_config = _resolve_config(config)
    path = build_manual_upper_goal_test_path(step_size=step_size)
    annotation = annotate_trajectory_phases(path, config=resolved_config)

    print("Manual trajectory phase test:")
    print(
        f"  first_h1_index={annotation.first_h1_index}, "
        f"first_h2_index={annotation.first_h2_index}, "
        f"first_terminal_index={annotation.first_terminal_index}, "
        f"terminal={annotation.first_terminal_phase}"
    )
    print(f"  unique_phase_sequence={list(dict.fromkeys(annotation.phase_labels))}")

    if show_plot and plt is not None:
        plot_trajectory_with_phases(path, config=resolved_config, show=True)

    return {
        "path": path,
        "annotation": annotation,
    }


class BraidedHub2DEnv:
    """Lightweight 2D point-robot environment for planning and data generation."""

    def __init__(
        self,
        map_config: MapConfig | None = None,
        rng_seed: int = DEFAULT_RANDOM_SEED,
        dt: float = DEFAULT_DT,
        step_penalty: float = DEFAULT_STEP_PENALTY,
        goal_reward: float = DEFAULT_GOAL_REWARD,
        enable_randomize: bool = False,
        collision_mode: str = DEFAULT_COLLISION_MODE,
    ) -> None:
        self.map_config = _resolve_config(map_config)
        self.start_region = self.map_config.start_region
        self.goal_regions = self.map_config.goal_regions
        self.goal_region_by_name = {
            goal_region.name: goal_region for goal_region in self.goal_regions
        }
        self.dt = float(dt)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.enable_randomize = bool(enable_randomize)
        self.collision_mode = str(collision_mode).lower()
        if self.collision_mode not in VALID_COLLISION_MODES:
            raise ValueError(
                f"Unsupported collision_mode={collision_mode!r}. "
                f"Expected one of {VALID_COLLISION_MODES}."
            )
        self.rng = random.Random(rng_seed)

        self.state: tuple[float, float] | None = None
        self.trajectory: list[tuple[float, float]] = []
        self.step_count = 0
        self.episode_task_id: int | None = None
        self.task_spec: TaskSpec | None = None
        self.start_region_name: str | None = None
        self.target_goal_name: str | None = None
        self.target_goal_region: GoalRegion | None = None
        self.last_info: dict[str, Any] = {}
        self.done = False

        self._render_fig = None
        self._render_ax = None

    def _resolve_start_state(
        self,
        task_start_region: RectangleRegion,
        enable_randomize: bool,
    ) -> tuple[float, float]:
        """Resolve the start state for one reset.

        By default the start is the region center for reproducibility.
        Randomized start sampling remains available behind an explicit flag.
        """

        if enable_randomize:
            return sample_valid_state_in_region(
                task_start_region,
                rng=self.rng,
                config=self.map_config,
            )

        center_state = (
            float(task_start_region.center[0]),
            float(task_start_region.center[1]),
        )
        if not is_state_valid(*center_state, config=self.map_config):
            raise RuntimeError(
                "The deterministic start-region center is not a valid free-space state. "
                f"region={task_start_region.name}, center={center_state}. "
                "Fix the map geometry or call reset(enable_randomize=True)."
            )
        return center_state

    def reset(
        self,
        task_id: int | None = None,
        enable_randomize: bool | None = None,
    ) -> tuple[float, float]:
        """Start a new episode from the task start-region center by default."""

        chosen_task_id = self.sample_task_id() if task_id is None else int(task_id)
        self.task_spec = build_task_spec(chosen_task_id)
        self.episode_task_id = self.task_spec.task_id
        self.start_region_name = self.task_spec.start_region_name
        self.target_goal_name = self.task_spec.target_goal_name
        self.target_goal_region = self.goal_region_by_name[self.target_goal_name]
        task_start_region = get_task_start_region(chosen_task_id, self.map_config)
        use_randomized_start = (
            self.enable_randomize
            if enable_randomize is None
            else bool(enable_randomize)
        )

        self.state = self._resolve_start_state(
            task_start_region=task_start_region,
            enable_randomize=use_randomized_start,
        )
        self.trajectory = [self.state]
        self.step_count = 0
        self.done = False
        self.last_info = {
            "task_id": self.episode_task_id,
            "task_code": self.task_spec.task_code,
            "task_bits": self.task_spec.task_bits,
            "start_region_name": self.start_region_name,
            "target_goal_name": self.target_goal_name,
            "reached_goal": None,
            "success": False,
            "collision_mode": self.collision_mode,
            "collision_detected": False,
            "collision_rejected": False,
            "start_randomized": use_randomized_start,
        }
        return self.state

    def sample_task_id(self) -> int:
        """Uniformly sample one of the four task ids."""

        return int(self.rng.choice(tuple(TASK_ID_TO_GOAL_NAME)))

    def is_cue_visible(self) -> bool:
        """The explicit cue channel is disabled in the implicit-cue map."""

        return False

    def get_cue_payload(self) -> dict[str, Any] | None:
        """Explicit cue payload is disabled; task information is carried by path history."""

        return None

    def get_full_observation(self) -> dict[str, Any]:
        """Return full observation including internal task truth."""

        if (
            self.state is None
            or self.task_spec is None
            or self.target_goal_region is None
        ):
            raise RuntimeError("Call reset() before requesting observations.")

        return {
            "x": self.state[0],
            "y": self.state[1],
            "position": self.state,
            "step_count": self.step_count,
            "task_id": self.task_spec.task_id,
            "task_code": self.task_spec.task_code,
            "task_bits": self.task_spec.task_bits,
            "start_region_name": self.start_region_name,
            "target_goal_name": self.task_spec.target_goal_name,
            "target_goal_center": self.target_goal_region.center,
        }

    def get_partial_observation(self) -> dict[str, Any]:
        """Return the policy-visible observation without any explicit cue channel."""

        if self.state is None:
            raise RuntimeError("Call reset() before requesting observations.")

        return {
            "x": self.state[0],
            "y": self.state[1],
            "position": self.state,
            "step_count": self.step_count,
        }

    def is_state_valid(self, state: tuple[float, float]) -> bool:
        """Planner-friendly state validity wrapper bound to this map instance."""

        return is_state_valid(*state, config=self.map_config)

    def sample_free_state(self) -> tuple[float, float]:
        """Planner-friendly free-space sampler bound to this map instance."""

        return sample_free_state(self.rng, config=self.map_config)

    def get_goal_region(self, state: tuple[float, float]) -> GoalRegion | None:
        """Return any terminal goal region containing the queried state."""

        return get_goal_region_for_state(*state, config=self.map_config)

    def get_phase_name(self, state: tuple[float, float]) -> str:
        """Return the semantic phase label for a queried state."""

        return get_phase_name(*state, config=self.map_config)

    def annotate_trajectory_phases(
        self,
        path: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    ) -> TrajectoryPhaseAnnotation:
        """Annotate a path using this environment's semantic map."""

        return annotate_trajectory_phases(path, config=self.map_config)

    def plot_trajectory_with_phases(
        self,
        path: list[tuple[float, float]] | tuple[tuple[float, float], ...],
        show: bool = True,
    ):
        """Convenience wrapper around the module-level phase plotter."""

        return plot_trajectory_with_phases(path, config=self.map_config, show=show)

    def step(
        self,
        action: tuple[float, float],
    ) -> tuple[tuple[float, float], float, bool, dict[str, Any]]:
        """Advance the point robot with simple Euler integration.

        Invalid next states are handled according to collision_mode:
        - `reject`: if the proposed state is outside bounds or inside an
          obstacle, the robot stays at the previous valid state.
        - `detect`: the invalid transition is recorded but still applied,
          which allows penetration through walls for debugging/eval ablations.
        Terminal handling:
        - Entering any goal region ends the episode.
        - Only entering the task-selected target goal counts as success and
          yields the terminal goal reward.
        """

        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError(
                "Episode already finished. Call reset() to start a new one."
            )
        if (
            self.task_spec is None
            or self.target_goal_region is None
            or self.target_goal_name is None
        ):
            raise RuntimeError("Call reset() before step().")

        dx, dy = action
        proposed_state = (
            float(self.state[0] + dx * self.dt),
            float(self.state[1] + dy * self.dt),
        )

        proposed_valid = is_state_valid(*proposed_state, config=self.map_config)
        collision_detected = not proposed_valid
        if proposed_valid:
            collision_rejected = False
            next_state = proposed_state
        elif self.collision_mode == "reject":
            collision_rejected = True
            next_state = self.state
        else:
            collision_rejected = False
            next_state = proposed_state
        reached_goal = get_goal_region_for_state(*next_state, config=self.map_config)
        success = (
            reached_goal is not None and reached_goal.name == self.target_goal_name
        )

        self.state = next_state
        self.trajectory.append(next_state)
        self.step_count += 1
        self.done = reached_goal is not None

        reward = self.step_penalty
        if success:
            reward += self.goal_reward

        info = {
            "task_id": self.episode_task_id,
            "task_code": self.task_spec.task_code,
            "task_bits": self.task_spec.task_bits,
            "start_region_name": self.start_region_name,
            "target_goal_name": self.target_goal_name,
            "step_count": self.step_count,
            "action": (float(dx), float(dy)),
            "collision_mode": self.collision_mode,
            "proposed_state": proposed_state,
            "applied_state": next_state,
            "proposed_state_valid": proposed_valid,
            "collision_detected": collision_detected,
            "collision_rejected": collision_rejected,
            "reached_goal": None if reached_goal is None else reached_goal.name,
            "success": success,
            "partial_observation": self.get_partial_observation(),
            "full_observation": self.get_full_observation(),
        }
        self.last_info = info
        return next_state, reward, self.done, info

    def render(
        self,
        show_trajectory: bool = True,
        show: bool = True,
        refresh_every_n_steps: int = 1,
        pause_seconds: float = DEFAULT_RENDER_PAUSE_SECONDS,
        force: bool = False,
    ):
        """Render the environment and current point robot state.

        Set refresh_every_n_steps > 1 to only redraw every N environment steps.
        This gives a simple animation mode without introducing a separate viewer.
        """

        if self.state is None:
            raise RuntimeError("Call reset() before render().")
        if plt is None:
            raise RuntimeError(
                "matplotlib is required to render this environment. "
                "Install it first, for example: pip install matplotlib"
            ) from _MATPLOTLIB_IMPORT_ERROR

        should_refresh = force or self.step_count == 0 or refresh_every_n_steps <= 1
        if not should_refresh:
            should_refresh = (self.step_count % refresh_every_n_steps) == 0
        if not should_refresh:
            return self._render_fig, self._render_ax

        if self._render_fig is None or self._render_ax is None:
            self._render_fig, self._render_ax = plt.subplots(figsize=(13, 7))
        elif not plt.fignum_exists(self._render_fig.number):
            self._render_fig, self._render_ax = plt.subplots(figsize=(13, 7))

        fig, ax = plot_map(self.map_config, show=False, ax=self._render_ax)

        if self.target_goal_region is not None:
            ax.add_patch(
                self.target_goal_region.as_patch(
                    facecolor="none",
                    edgecolor="#111111",
                    linewidth=2.5,
                    linestyle="--",
                    zorder=6,
                )
            )

        if show_trajectory and len(self.trajectory) >= 2:
            traj_xs = [point[0] for point in self.trajectory]
            traj_ys = [point[1] for point in self.trajectory]
            ax.plot(
                traj_xs,
                traj_ys,
                color=VALID_SAMPLE_COLOR,
                linewidth=TRAJECTORY_LINEWIDTH,
                alpha=0.85,
                zorder=6,
            )

        ax.scatter(
            self.state[0],
            self.state[1],
            s=ROBOT_MARKER_SIZE,
            c="#111111",
            edgecolors="white",
            linewidths=1.2,
            zorder=7,
        )
        ax.set_title(
            "BraidedHub2DEnv\n"
            f"step={self.step_count}, task_id={self.episode_task_id}, "
            f"code={None if self.task_spec is None else self.task_spec.task_code}, "
            f"start={self.start_region_name}, "
            f"target={self.target_goal_name}, "
            f"state=({self.state[0]:.2f}, {self.state[1]:.2f})"
        )

        fig.tight_layout()

        if show:
            plt.show(block=False)
            plt.pause(pause_seconds)

        return fig, ax

    def rollout_with_random_actions(
        self,
        num_steps: int = DEFAULT_RANDOM_ROLLOUT_STEPS,
        action_scale: float = DEFAULT_RANDOM_ACTION_SCALE,
        render_every_n_steps: int = DEFAULT_RENDER_EVERY_N_STEPS,
        pause_seconds: float = DEFAULT_RENDER_PAUSE_SECONDS,
        show: bool = True,
        task_id: int | None = None,
    ) -> dict[str, Any]:
        """Run one random rollout for quick collision and wall-debug checks."""

        initial_state = self.reset(task_id=task_id)
        total_reward = 0.0
        collision_count = 0
        actions: list[tuple[float, float]] = []
        rewards: list[float] = []

        if show:
            self.render(
                show_trajectory=True,
                show=True,
                refresh_every_n_steps=1,
                pause_seconds=pause_seconds,
                force=True,
            )

        for _ in range(num_steps):
            action = (
                float(self.rng.uniform(-action_scale, action_scale)),
                float(self.rng.uniform(-action_scale, action_scale)),
            )
            next_state, reward, done, info = self.step(action)
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            if info["collision_rejected"]:
                collision_count += 1

            if show:
                self.render(
                    show_trajectory=True,
                    show=True,
                    refresh_every_n_steps=render_every_n_steps,
                    pause_seconds=pause_seconds,
                )

            if done:
                break

        if show:
            self.render(
                show_trajectory=True,
                show=True,
                refresh_every_n_steps=1,
                pause_seconds=pause_seconds,
                force=True,
            )

        return {
            "initial_state": initial_state,
            "final_state": self.state,
            "trajectory": list(self.trajectory),
            "actions": actions,
            "rewards": rewards,
            "total_reward": total_reward,
            "num_steps": self.step_count,
            "collision_count": collision_count,
            "done": self.done,
            "task_id": self.episode_task_id,
            "task_code": None if self.task_spec is None else self.task_spec.task_code,
            "target_goal_name": self.target_goal_name,
            "reached_goal": self.last_info.get("reached_goal"),
            "success": self.last_info.get("success"),
        }

    def manual_test_task_routes(
        self,
        num_resets: int = DEFAULT_MANUAL_TEST_RESETS,
        steps_per_reset: int = DEFAULT_MANUAL_TEST_STEPS,
        action_scale: float = 1.0,
    ) -> None:
        """Print task assignments and policy-visible observations without explicit cues."""

        for reset_idx in range(num_resets):
            state = self.reset()
            print(
                f"[reset {reset_idx:02d}] "
                f"task_id={self.episode_task_id}, "
                f"task_code={self.task_spec.task_code if self.task_spec else None}, "
                f"task_bits={self.task_spec.task_bits if self.task_spec else None}, "
                f"start={self.start_region_name}, "
                f"target={self.target_goal_name}, "
                f"state=({state[0]:.2f}, {state[1]:.2f})"
            )
            print(f"  partial_obs@reset: {self.get_partial_observation()}")

            for _ in range(steps_per_reset):
                action = (
                    float(self.rng.uniform(-action_scale, action_scale)),
                    float(self.rng.uniform(-action_scale, action_scale)),
                )
                _, reward, done, info = self.step(action)
                print(
                    f"  step={self.step_count:02d}, "
                    f"reward={reward:.2f}, "
                    f"done={done}, "
                    f"obs={info['partial_observation']}"
                )
                if done:
                    break

    def manual_test_task_cues(
        self,
        num_resets: int = DEFAULT_MANUAL_TEST_RESETS,
        steps_per_reset: int = DEFAULT_MANUAL_TEST_STEPS,
        action_scale: float = 1.0,
    ) -> None:
        """Backward-compatible wrapper for the old explicit-cue test entry point."""

        self.manual_test_task_routes(
            num_resets=num_resets,
            steps_per_reset=steps_per_reset,
            action_scale=action_scale,
        )

    def close(self) -> None:
        """Close any cached matplotlib figure created by render()."""

        if plt is not None and self._render_fig is not None:
            plt.close(self._render_fig)
        self._render_fig = None
        self._render_ax = None


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
    return load_policy_mode_defaults("train", ENV_NAME, policy_type)


def get_eval_defaults(policy_type: str) -> dict[str, Any]:
    return load_policy_mode_defaults("eval", ENV_NAME, policy_type)


def collect_dataset(args) -> Path:
    import planner_utils

    map_config = build_default_map_config()
    enable_randomize = bool(getattr(args, "enable_randomize", False))
    print(
        "[info] braidedhub dataset collection reset mode: "
        + ("randomized starts" if enable_randomize else "deterministic center starts")
    )
    raw_dataset = planner_utils.generate_demonstrations(
        num_per_task=args.num_per_task,
        seed=args.seed,
        solve_time=args.solve_time,
        max_retries_per_demo=args.max_retries_per_demo,
        low_success_warning_threshold=args.low_success_warning_threshold,
        enable_randomize=enable_randomize,
        config=map_config,
    )

    if args.raw_output is not None or args.processed_output is not None:
        print(
            "[info] NPZ export is disabled. "
            "Ignoring --raw-output/--processed-output."
        )

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
        use_delta_signature=bool(getattr(args, "enable_delta_signature", False)),
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
    task_ids = sorted(TASK_ID_TO_GOAL_NAME)
    repeated = [task_ids[index % len(task_ids)] for index in range(num_rollouts)]
    rng = np.random.default_rng(seed)
    rng.shuffle(repeated)
    return [int(task_id) for task_id in repeated]


def detect_branch_mismatch(
    task_spec,
    state_xy: tuple[float, float],
    config: MapConfig | None = None,
) -> dict[str, str] | None:
    expected_branch1 = BRANCH1_PHASE_BY_BIT[int(task_spec.task_bits[0])]
    expected_branch2 = BRANCH2_PHASE_BY_BIT[int(task_spec.task_bits[1])]
    detection_event = get_branch_detection_event(
        float(state_xy[0]),
        float(state_xy[1]),
        config=config,
    )
    if detection_event is None:
        return None

    stage_name, observed_phase = detection_event
    expected_phase = expected_branch1 if stage_name == "H1" else expected_branch2
    if observed_phase != expected_phase:
        return {
            "stage": stage_name,
            "expected_phase": expected_phase,
            "observed_phase": observed_phase,
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

    eval_num_rollouts = int(getattr(args, "eval_num_rollouts", args.num_rollouts))
    eval_max_steps = int(getattr(args, "eval_max_steps", args.max_steps))
    eval_fps = int(getattr(args, "eval_fps", args.fps))
    eval_seed = int(getattr(args, "eval_seed", args.seed))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_key, image_shape = resolve_single_visual_observation_feature(cfg)
    image_hw = (int(image_shape[1]), int(image_shape[2]))

    if cfg.robot_state_feature is None:
        raise RuntimeError("Policy has no observation.state feature.")
    state_key = "observation.state"
    state_dim = int(cfg.robot_state_feature.shape[0])

    capability_flags = resolve_policy_capability_flags(cfg)
    use_path_signature = capability_flags.use_path_signature
    use_prefix_sequence_training = capability_flags.use_prefix_sequence_training
    use_visual_prefix_memory = capability_flags.use_visual_prefix_memory
    use_signature_indexed_slot_memory = (
        capability_flags.use_signature_indexed_slot_memory
    )
    use_delta_signature = capability_flags.use_delta_signature
    build_explicit_prefix_eval_inputs = (
        capability_flags.build_explicit_prefix_eval_inputs
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
        initial_memory_debug = get_visual_memory_debug_stats(policy)
        print(
            "[info] visual prefix memory online update enabled: "
            + (
                "rollout uses signature-indexed slot memory without rebuilding "
                "explicit prefix-sequence tensors each step"
                if use_signature_indexed_slot_memory
                else "rollout uses fixed-size recurrent memory without rebuilding "
                "explicit prefix-sequence tensors each step"
            )
        )
        if initial_memory_debug is not None:
            print(
                "[info] visual prefix memory debug: "
                f"enabled={bool(initial_memory_debug.get('enabled', False))}, "
                f"num_slots={int(initial_memory_debug.get('num_slots', 0))}, "
                f"updates={int(initial_memory_debug.get('update_count', 0))}"
            )

    map_config = build_default_map_config()
    enable_randomize = bool(getattr(args, "enable_randomize", False))
    print(
        "[info] braidedhub eval reset mode: "
        + ("randomized starts" if enable_randomize else "deterministic center starts")
    )
    print(
        "[info] braidedhub collision mode: "
        f"{args.collision_mode} "
        + (
            "(invalid moves are rejected)"
            if args.collision_mode == "reject"
            else "(invalid moves are detected but still applied)"
        )
    )
    env = BraidedHub2DEnv(
        map_config=map_config,
        rng_seed=eval_seed,
        enable_randomize=enable_randomize,
        collision_mode=args.collision_mode,
    )
    base_img = make_lerobot_base_image(
        map_config,
        image_size=max(image_hw),
    )
    if base_img.shape[0] != image_hw[0] or base_img.shape[1] != image_hw[1]:
        raise RuntimeError(
            "Rendered image size mismatch between dataset image features and eval map image. "
            f"Policy expects HxW={image_hw}, base_img={base_img.shape[:2]}."
        )

    results = []
    success_count = 0
    branch_failure_count = 0
    task_success_counts = {int(task_id): 0 for task_id in sorted(TASK_ID_TO_GOAL_NAME)}
    task_rollout_counts = {int(task_id): 0 for task_id in sorted(TASK_ID_TO_GOAL_NAME)}
    task_branch_failure_counts = {
        int(task_id): 0 for task_id in sorted(TASK_ID_TO_GOAL_NAME)
    }
    task_schedule = build_balanced_task_schedule(eval_num_rollouts, eval_seed)

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
        episode_collision_detection_count = 0
        episode_collision_rejection_count = 0
        video_path = output_dir / f"rollout_{ep_idx:03d}_task_{task_spec.task_code}.mp4"
        writer = start_ffmpeg_raw_writer(
            video_path,
            image_hw[1],
            image_hw[0],
            eval_fps,
        )
        if writer.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin for rollout video writing.")

        trajectory = [state_xy]
        state_history = deque() if use_path_signature else None
        prefix_state_history = [] if build_explicit_prefix_eval_inputs else None
        prefix_signature_history = (
            [] if build_explicit_prefix_eval_inputs and use_path_signature else None
        )
        prefix_delta_signature_history = (
            [] if build_explicit_prefix_eval_inputs and use_delta_signature else None
        )
        prefix_image_history = [] if build_explicit_prefix_eval_inputs else None
        previous_signature_vec = None
        last_info = {
            **env.last_info,
            "phase_name": env.get_phase_name(state_xy),
            "branch_mismatch": False,
            "branch_mismatch_stage": None,
            "expected_branch_phase": None,
            "observed_branch_phase": None,
            "failure_reason": None,
        }

        for _step_idx in range(eval_max_steps):
            frame = render_lerobot_frame(
                base_img,
                config=map_config,
                robot_xy=state_xy,
            )
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
            if info.get("collision_detected", False):
                episode_collision_detection_count += 1
            if info.get("collision_rejected", False):
                episode_collision_rejection_count += 1
            phase_name = env.get_phase_name(state_xy)
            mismatch = detect_branch_mismatch(
                task_spec=task_spec,
                state_xy=state_xy,
                config=map_config,
            )
            last_info = {
                **info,
                "phase_name": phase_name,
                "branch_mismatch": False,
                "branch_mismatch_stage": None,
                "expected_branch_phase": None,
                "observed_branch_phase": None,
                "failure_reason": None,
            }
            if mismatch is not None:
                last_info = {
                    **last_info,
                    "branch_mismatch": True,
                    "branch_mismatch_stage": mismatch["stage"],
                    "expected_branch_phase": mismatch["expected_phase"],
                    "observed_branch_phase": mismatch["observed_phase"],
                    "failure_reason": "wrong_branch",
                    "success": False,
                }
                final_frame = render_lerobot_frame(
                    base_img,
                    config=map_config,
                    robot_xy=state_xy,
                )
                writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                break

            if done:
                final_frame = render_lerobot_frame(
                    base_img,
                    config=map_config,
                    robot_xy=state_xy,
                )
                writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                break

        writer.stdin.close()
        code = writer.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed on rollout {ep_idx} with exit code {code}")

        success = bool(last_info.get("success", False)) and not bool(
            last_info.get("branch_mismatch", False)
        )
        if success:
            success_count += 1
            task_success_counts[task_id] += 1
        if last_info.get("branch_mismatch", False):
            branch_failure_count += 1
            task_branch_failure_counts[task_id] += 1

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
            "branch_mismatch": bool(last_info.get("branch_mismatch", False)),
            "branch_mismatch_stage": last_info.get("branch_mismatch_stage"),
            "expected_branch_phase": last_info.get("expected_branch_phase"),
            "observed_branch_phase": last_info.get("observed_branch_phase"),
            "failure_reason": last_info.get("failure_reason"),
            "success": success,
            "steps": int(len(trajectory) - 1),
            "sum_reward": float(episode_reward),
            "collision_mode": args.collision_mode,
            "collision_detections": int(episode_collision_detection_count),
            "collision_rejections": int(episode_collision_rejection_count),
        }
        memory_debug_stats = (
            get_visual_memory_debug_stats(policy) if use_visual_prefix_memory else None
        )
        if memory_debug_stats is not None:
            result["visual_memory_debug"] = memory_debug_stats
        results.append(result)
        memory_debug_text = ""
        if memory_debug_stats is not None:
            memory_debug_text = (
                " "
                f"memory_updates={int(memory_debug_stats.get('update_count', 0))} "
                f"memory_norm={float(memory_debug_stats.get('state_norm', 0.0)):.4f}"
            )
        print(
            f"[{ep_idx + 1:03d}/{eval_num_rollouts:03d}] "
            f"task={task_spec.task_code}->{task_spec.target_goal_name} "
            f"success={success} steps={result['steps']} "
            f"reached={result['reached_goal']} "
            f"branch_mismatch={result['branch_mismatch']} "
            f"video={video_path.name}{memory_debug_text}"
        )

    per_task = {
        str(task_id): {
            "goal_name": TASK_ID_TO_GOAL_NAME[task_id],
            "rollouts": int(task_rollout_counts[task_id]),
            "success_count": int(task_success_counts[task_id]),
            "wrong_branch_failures": int(task_branch_failure_counts[task_id]),
            "success_rate": float(
                task_success_counts[task_id] / max(1, task_rollout_counts[task_id])
            ),
        }
        for task_id in sorted(TASK_ID_TO_GOAL_NAME)
    }
    summary = {
        "env": ENV_NAME,
        "policy_type": policy_type,
        "num_rollouts": eval_num_rollouts,
        "success_count": success_count,
        "success_rate": float(success_count / max(1, eval_num_rollouts)),
        "wrong_branch_failures": branch_failure_count,
        "seed": eval_seed,
        "fps": eval_fps,
        "max_steps": eval_max_steps,
        "max_action_step": args.max_action_step,
        "collision_mode": args.collision_mode,
        "collision_detections": int(
            sum(result["collision_detections"] for result in results)
        ),
        "collision_rejections": int(
            sum(result["collision_rejections"] for result in results)
        ),
        "start_randomized": enable_randomize,
        "policy_dir": str(policy_dir),
        "per_task": per_task,
        "results": results,
    }
    if use_visual_prefix_memory:
        memory_debug_stats = get_visual_memory_debug_stats(policy)
        if memory_debug_stats is not None:
            summary["visual_memory_debug"] = memory_debug_stats
    summary_path = write_summary(output_dir, summary)

    print(f"\nSaved {eval_num_rollouts} rollout videos to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(
        f"Success rate: {summary['success_rate']:.3f} ({success_count}/{eval_num_rollouts})"
    )
    print(
        "Collision mode: "
        f"{summary['collision_mode']}, "
        f"detections={summary['collision_detections']}, "
        f"rejections={summary['collision_rejections']}"
    )
    print(f"Wrong-branch failures: {branch_failure_count}/{eval_num_rollouts}")
    for task_id in sorted(TASK_ID_TO_GOAL_NAME):
        task_summary = per_task[str(task_id)]
        print(
            f"  task {task_id} ({task_summary['goal_name']}): "
            f"{task_summary['success_count']}/{task_summary['rollouts']} "
            f"= {task_summary['success_rate']:.3f}, "
            f"wrong_branch={task_summary['wrong_branch_failures']}"
        )


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


def compress_dense_path_to_collision_free_indices(
    path_xy: np.ndarray | list[tuple[float, float]],
    config: MapConfig | None = None,
    resolution: float = DEFAULT_RESAMPLED_SEGMENT_CHECK_RESOLUTION,
) -> list[int]:
    """Greedily compress a dense safe path into a collision-free polyline index set.

    The output keeps path order, uses only points from the original dense path, and ensures
    every consecutive straight segment is collision-free under the configured checker.
    """

    # Match the dtype used by the exported dataset so compression remains valid
    # for the exact straight segments we will later serialize and re-check.
    path_array = _as_path_array(path_xy).astype(np.float32, copy=False)
    if path_array.shape[0] <= 1:
        return [0]

    resolved_config = _resolve_config(config)
    selected_indices = [0]
    current_index = 0
    last_index = int(path_array.shape[0] - 1)

    while current_index < last_index:
        next_index = last_index
        current_point = (
            float(path_array[current_index, 0]),
            float(path_array[current_index, 1]),
        )
        while next_index > current_index + 1:
            next_point = (
                float(path_array[next_index, 0]),
                float(path_array[next_index, 1]),
            )
            if is_line_segment_valid(
                current_point,
                next_point,
                config=resolved_config,
                resolution=resolution,
            ):
                break
            next_index -= 1

        if next_index == current_index:
            raise RuntimeError(
                "Failed to advance collision-free path compression. "
                f"current_index={current_index}, path_length={path_array.shape[0]}"
            )

        if next_index == current_index + 1:
            next_point = (
                float(path_array[next_index, 0]),
                float(path_array[next_index, 1]),
            )
            if not is_line_segment_valid(
                current_point,
                next_point,
                config=resolved_config,
                resolution=resolution,
            ):
                raise ValueError(
                    "Dense raw path contains an invalid consecutive segment. "
                    f"segment=({current_index}, {next_index})"
                )

        selected_indices.append(next_index)
        current_index = next_index

    return selected_indices


def compress_dense_path_to_collision_free_polyline(
    path_xy: np.ndarray | list[tuple[float, float]],
    config: MapConfig | None = None,
    resolution: float = DEFAULT_RESAMPLED_SEGMENT_CHECK_RESOLUTION,
) -> np.ndarray:
    path_array = _as_path_array(path_xy)
    indices = compress_dense_path_to_collision_free_indices(
        path_array,
        config=config,
        resolution=resolution,
    )
    return path_array[np.asarray(indices, dtype=np.int64)].astype(np.float32)


def count_invalid_resampled_segments(
    path_xy: np.ndarray | list[tuple[float, float]],
    config: MapConfig | None = None,
    resolution: float = DEFAULT_RESAMPLED_SEGMENT_CHECK_RESOLUTION,
) -> tuple[int, int | None]:
    path_array = _as_path_array(path_xy)
    invalid_segment_count = 0
    first_invalid_segment_index: int | None = None

    for segment_index in range(path_array.shape[0] - 1):
        from_point = (
            float(path_array[segment_index, 0]),
            float(path_array[segment_index, 1]),
        )
        to_point = (
            float(path_array[segment_index + 1, 0]),
            float(path_array[segment_index + 1, 1]),
        )
        if is_line_segment_valid(
            from_point,
            to_point,
            config=config,
            resolution=resolution,
        ):
            continue
        invalid_segment_count += 1
        if first_invalid_segment_index is None:
            first_invalid_segment_index = segment_index

    return invalid_segment_count, first_invalid_segment_index


def resolve_resampled_horizon(
    compressed_index_paths: list[list[int]],
    requested_t_fixed: int,
) -> int:
    if not compressed_index_paths:
        raise ValueError("raw_dataset is empty.")

    compressed_lengths = np.asarray(
        [len(index_path) for index_path in compressed_index_paths],
        dtype=np.int64,
    )
    required_t_fixed = int(compressed_lengths.max())
    avg_compressed_steps = float(compressed_lengths.mean())

    if requested_t_fixed > 0:
        if requested_t_fixed < required_t_fixed:
            raise ValueError(
                "Requested t_fixed is too small to preserve collision-free polylines "
                f"for every episode. requested={requested_t_fixed}, "
                f"required_minimum={required_t_fixed}."
            )
        return int(requested_t_fixed)

    print(
        "[info] auto-resolving t_fixed from collision-free compressed paths: "
        f"avg_compressed_steps={avg_compressed_steps:.2f}, "
        f"required_minimum={required_t_fixed}"
    )
    return required_t_fixed


def expand_collision_free_polyline_fixed_length(
    path_xy: np.ndarray | list[tuple[float, float]],
    compressed_indices: list[int],
    t_fixed: int,
    config: MapConfig | None = None,
    resolution: float = DEFAULT_RESAMPLED_SEGMENT_CHECK_RESOLUTION,
) -> np.ndarray:
    if t_fixed <= 0:
        raise ValueError("t_fixed must be positive.")

    path_array = _as_path_array(path_xy).astype(np.float32, copy=False)
    if not compressed_indices:
        raise ValueError("compressed_indices must not be empty.")

    current_indices = [int(index) for index in compressed_indices]
    current_length = len(current_indices)
    if current_length > t_fixed:
        raise ValueError(
            f"Cannot expand path of length {current_length} into smaller t_fixed={t_fixed}."
        )
    if current_length == 1:
        repeated = path_array[np.asarray(current_indices[:1], dtype=np.int64)]
        return np.repeat(repeated, t_fixed, axis=0).astype(np.float32, copy=False)

    resolved_config = _resolve_config(config)
    total_intervals = int(t_fixed - 1)

    def try_find_split_index(left_index: int, right_index: int) -> int | None:
        if right_index - left_index <= 1:
            return None

        midpoint = (left_index + right_index) // 2
        max_offset = max(midpoint - left_index, right_index - midpoint)
        candidate_indices: list[int] = []
        for offset in range(max_offset + 1):
            left_candidate = midpoint - offset
            right_candidate = midpoint + offset
            if left_index < left_candidate < right_index:
                candidate_indices.append(left_candidate)
            if (
                right_candidate != left_candidate
                and left_index < right_candidate < right_index
            ):
                candidate_indices.append(right_candidate)

        left_point = (
            float(path_array[left_index, 0]),
            float(path_array[left_index, 1]),
        )
        right_point = (
            float(path_array[right_index, 0]),
            float(path_array[right_index, 1]),
        )
        for candidate_index in candidate_indices:
            middle_point = (
                float(path_array[candidate_index, 0]),
                float(path_array[candidate_index, 1]),
            )
            if not is_line_segment_valid(
                left_point,
                middle_point,
                config=resolved_config,
                resolution=resolution,
            ):
                continue
            if not is_line_segment_valid(
                middle_point,
                right_point,
                config=resolved_config,
                resolution=resolution,
            ):
                continue
            return candidate_index
        return None

    def allocate_interval_counts(polyline_xy: np.ndarray) -> np.ndarray:
        num_segments = int(polyline_xy.shape[0] - 1)
        if total_intervals < num_segments:
            raise ValueError(
                "t_fixed is too small to preserve every collision-free polyline vertex. "
                f"t_fixed={t_fixed}, required_minimum={num_segments + 1}."
            )

        segment_vectors = np.diff(polyline_xy.astype(np.float64), axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        interval_counts = np.ones(num_segments, dtype=np.int64)
        remaining_intervals = int(total_intervals - num_segments)
        for _ in range(remaining_intervals):
            best_segment_index = max(
                range(num_segments),
                key=lambda idx: (
                    float(segment_lengths[idx] / max(1, int(interval_counts[idx]))),
                    float(segment_lengths[idx]),
                ),
            )
            interval_counts[best_segment_index] += 1
        return interval_counts

    def build_uniform_dense_path(
        polyline_xy: np.ndarray,
        interval_counts: np.ndarray,
    ) -> tuple[np.ndarray | None, int | None]:
        expanded = np.empty((t_fixed, 2), dtype=np.float32)
        expanded[0] = polyline_xy[0]
        write_index = 1

        for segment_index in range(polyline_xy.shape[0] - 1):
            start_point = polyline_xy[segment_index].astype(np.float64)
            end_point = polyline_xy[segment_index + 1].astype(np.float64)
            previous_point = polyline_xy[segment_index]
            interval_count = int(interval_counts[segment_index])
            for step_index in range(1, interval_count + 1):
                alpha = float(step_index / interval_count)
                next_point = (
                    start_point * (1.0 - alpha) + end_point * alpha
                ).astype(np.float32)
                if not is_line_segment_valid(
                    (
                        float(previous_point[0]),
                        float(previous_point[1]),
                    ),
                    (
                        float(next_point[0]),
                        float(next_point[1]),
                    ),
                    config=resolved_config,
                    resolution=resolution,
                ):
                    return None, int(segment_index)
                expanded[write_index] = next_point
                write_index += 1
                previous_point = next_point

        if write_index != t_fixed:
            raise RuntimeError(
                "Uniform collision-free densification produced an unexpected number "
                f"of states. expected={t_fixed}, got={write_index}"
            )
        return expanded, None

    while True:
        polyline = path_array[np.asarray(current_indices, dtype=np.int64)].astype(
            np.float32,
            copy=False,
        )
        if polyline.shape[0] == t_fixed:
            expanded = polyline.astype(np.float32, copy=True)
        else:
            interval_counts = allocate_interval_counts(polyline)
            expanded, failing_segment_index = build_uniform_dense_path(
                polyline,
                interval_counts,
            )
            if expanded is None:
                if failing_segment_index is None:
                    raise RuntimeError(
                        "Uniform collision-free densification failed without reporting "
                        "the offending segment."
                    )

                left_index = current_indices[failing_segment_index]
                right_index = current_indices[failing_segment_index + 1]
                split_index = try_find_split_index(left_index, right_index)
                if split_index is None:
                    if right_index - left_index <= 1:
                        raise ValueError(
                            "Unable to refine a collision-free segment for uniform dense "
                            f"resampling. left_index={left_index}, right_index={right_index}, "
                            f"t_fixed={t_fixed}"
                        )
                    split_index = (left_index + right_index) // 2
                current_indices.insert(failing_segment_index + 1, int(split_index))
                continue

        invalid_segment_count, first_invalid_segment_index = (
            count_invalid_resampled_segments(
                expanded,
                config=resolved_config,
                resolution=resolution,
            )
        )
        if invalid_segment_count > 0:
            raise ValueError(
                "Uniform collision-free densification produced an invalid straight-line "
                f"segment. invalid_segments={invalid_segment_count}, "
                f"first_invalid_segment={first_invalid_segment_index}"
            )
        return expanded


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
    task_code = format(task_id, "02b")
    return np.asarray([int(task_code[0]), int(task_code[1])], dtype=np.int64)


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


def _require_lerobot_export_dependencies():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to export LeRobot v3.0 datasets. "
            "Install it first, for example: pip install pyarrow. "
            "If you only need preprocessing, rerun with --skip-lerobot-export."
        ) from exc
    return pa, pq


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
    px = int(np.clip(px, 0, width - 1))
    py = int(np.clip(py, 0, height - 1))
    return px, py


def _fill_region(
    image: np.ndarray,
    region,
    color: tuple[int, int, int],
    config: MapConfig,
) -> None:
    if hasattr(region, "rectangles"):
        for rectangle in region.rectangles:
            _fill_region(image, rectangle, color, config=config)
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


def _draw_region_outline(
    image: np.ndarray,
    region: RectangleRegion | SemanticRegion,
    color: tuple[int, int, int],
    thickness: int,
    config: MapConfig,
) -> None:
    if hasattr(region, "rectangles"):
        for rectangle in region.rectangles:
            _draw_region_outline(
                image,
                rectangle,
                color,
                thickness,
                config=config,
            )
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
    height, width = image.shape[:2]
    x0 = max(0, min(width - 1, x0))
    x1 = max(0, min(width - 1, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(0, min(height - 1, y1))
    t = max(1, int(thickness))
    outline_color = np.asarray(color, dtype=np.uint8)

    image[y0 : min(height, y0 + t), x0 : x1 + 1] = outline_color
    image[max(0, y1 - t + 1) : y1 + 1, x0 : x1 + 1] = outline_color
    image[y0 : y1 + 1, x0 : min(width, x0 + t)] = outline_color
    image[y0 : y1 + 1, max(0, x1 - t + 1) : x1 + 1] = outline_color


def _draw_workspace_segment(
    image: np.ndarray,
    from_xy: tuple[float, float],
    to_xy: tuple[float, float],
    color: tuple[int, int, int],
    radius: int,
    config: MapConfig,
) -> None:
    start_px = _workspace_to_pixel(
        from_xy,
        config=config,
        image_size=image.shape[:2],
    )
    end_px = _workspace_to_pixel(
        to_xy,
        config=config,
        image_size=image.shape[:2],
    )
    dx = end_px[0] - start_px[0]
    dy = end_px[1] - start_px[1]
    num_steps = max(1, int(max(abs(dx), abs(dy))))
    for step_index in range(num_steps + 1):
        alpha = step_index / num_steps
        point_px = (
            int(round(start_px[0] * (1.0 - alpha) + end_px[0] * alpha)),
            int(round(start_px[1] * (1.0 - alpha) + end_px[1] * alpha)),
        )
        _draw_disk(image, point_px, radius, color)


def make_lerobot_base_image(
    config: MapConfig,
    image_size: int = DEFAULT_VIDEO_IMAGE_SIZE,
) -> np.ndarray:
    base_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    base_image[:, :] = np.asarray(MAP_BACKGROUND_COLOR, dtype=np.uint8)

    for obstacle in config.obstacle_rectangles:
        _fill_region(base_image, obstacle, OBSTACLE_COLOR, config=config)
    for start_region in config.task_start_regions:
        _fill_region(base_image, start_region, START_COLOR, config=config)
    for goal_region in config.goal_regions:
        _fill_region(
            base_image,
            goal_region,
            GOAL_COLOR_BY_NAME.get(goal_region.name, (0, 0, 0)),
            config=config,
        )
    return base_image


def make_lerobot_episode_base_image(
    base_image: np.ndarray,
    config: MapConfig,
    target_goal_name: str | None = None,
    goal_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    del config, target_goal_name, goal_xy
    return base_image.copy()


def draw_lerobot_trail_segment(
    image: np.ndarray,
    config: MapConfig,
    from_xy: tuple[float, float],
    to_xy: tuple[float, float],
) -> None:
    trail_radius = max(1, int(min(image.shape[:2]) * 0.01))
    _draw_workspace_segment(
        image,
        from_xy=from_xy,
        to_xy=to_xy,
        color=TRAJECTORY_TRAIL_COLOR,
        radius=trail_radius,
        config=config,
    )


def render_lerobot_frame(
    base_image: np.ndarray,
    config: MapConfig,
    robot_xy: tuple[float, float],
) -> np.ndarray:
    frame = base_image.copy()
    height, width = frame.shape[:2]
    robot_px = _workspace_to_pixel(robot_xy, config=config, image_size=(height, width))
    outer_radius = max(3, int(min(height, width) * 0.03))
    inner_radius = max(2, int(outer_radius * 0.65))
    _draw_disk(frame, robot_px, outer_radius, ROBOT_OUTER_COLOR)
    _draw_disk(frame, robot_px, inner_radius, ROBOT_INNER_COLOR)
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


def init_visual_feature_stats() -> dict[str, Any]:
    return {
        "min": np.full((3,), np.inf, dtype=np.float64),
        "max": np.full((3,), -np.inf, dtype=np.float64),
        "sum": np.zeros((3,), dtype=np.float64),
        "sum_sq": np.zeros((3,), dtype=np.float64),
        "count": 0,
    }


def update_visual_feature_stats(stats: dict[str, Any], frame: np.ndarray, weight: int = 1) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}.")
    if weight <= 0:
        raise ValueError(f"weight must be positive, got {weight}.")

    frame_arr = frame.astype(np.float64, copy=False) / 255.0
    frame_chw = np.transpose(frame_arr, (2, 0, 1)).reshape(3, -1)
    stats["min"] = np.minimum(stats["min"], frame_chw.min(axis=1))
    stats["max"] = np.maximum(stats["max"], frame_chw.max(axis=1))
    stats["sum"] += frame_chw.sum(axis=1) * weight
    stats["sum_sq"] += np.square(frame_chw).sum(axis=1) * weight
    stats["count"] += int(frame_chw.shape[1] * weight)


def finalize_visual_feature_stats(stats: dict[str, Any]) -> dict[str, Any]:
    count = int(stats["count"])
    if count <= 0:
        raise ValueError("Cannot finalize visual stats with zero pixels.")

    mean = stats["sum"] / count
    variance = np.maximum(stats["sum_sq"] / count - mean**2, 0.0)

    def _reshape(value: np.ndarray) -> list:
        return np.asarray(value, dtype=np.float64).reshape(3, 1, 1).tolist()

    return {
        "min": _reshape(stats["min"]),
        "max": _reshape(stats["max"]),
        "mean": _reshape(mean),
        "std": _reshape(np.sqrt(variance)),
        "count": [count],
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
        signature_array = np.asarray(
            frame_records[DEFAULT_PATH_SIGNATURE_KEY],
            dtype=np.float32,
        )
        arrays.append(
            fixed_size_list_array(pa, signature_array, int(signature_array.shape[1]))
        )
        names.append(DEFAULT_PATH_SIGNATURE_KEY)
    if DEFAULT_DELTA_SIGNATURE_KEY in frame_records:
        delta_signature_array = np.asarray(
            frame_records[DEFAULT_DELTA_SIGNATURE_KEY],
            dtype=np.float32,
        )
        arrays.append(
            fixed_size_list_array(
                pa,
                delta_signature_array,
                int(delta_signature_array.shape[1]),
            )
        )
        names.append(DEFAULT_DELTA_SIGNATURE_KEY)

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
    return (
        int(episode_index // episodes_per_chunk),
        int(episode_index % episodes_per_chunk),
    )


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
    video_frame_counts_by_key: dict[str, dict[int, int]],
) -> None:
    if total_frames != len(records["index"]):
        raise ValueError("total_frames mismatch with frame table")

    expected_indices = np.arange(total_frames, dtype=np.int64)
    if not np.array_equal(
        np.asarray(records["index"], dtype=np.int64), expected_indices
    ):
        raise ValueError("global index must be continuous and monotonic")

    for video_key, video_frame_counts in video_frame_counts_by_key.items():
        total_video_frames = int(sum(video_frame_counts.values()))
        if total_video_frames != total_frames:
            raise ValueError(
                f"video frame count mismatch for {video_key}: "
                f"video={total_video_frames}, parquet={total_frames}"
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
        for video_key, video_frame_counts in video_frame_counts_by_key.items():
            video_frames = video_frame_counts.get(int(episode_meta["episode_index"]))
            if video_frames != int(episode_meta["length"]):
                raise ValueError(
                    "per-episode video frame count mismatch in "
                    f"episode {episode_meta['episode_index']} for {video_key}: "
                    f"video={video_frames}, episode_length={episode_meta['length']}"
                )

    for split_name, split_spec in splits.items():
        split_start, split_end = split_spec.split(":", 1)
        split_from = int(split_start)
        split_to = int(split_end)
        if not (0 <= split_from <= split_to <= total_episodes):
            raise ValueError(f"invalid split range for {split_name}: {split_spec}")


def validate_task_coverage(task_ids: np.ndarray | list[int]) -> dict[int, int]:
    task_array = np.asarray(task_ids, dtype=np.int64)
    task_counts = {
        task_id: int(np.sum(task_array == task_id)) for task_id in TASK_ID_VALUES
    }
    missing_tasks = [
        f"{task_id}:{TASK_ID_TO_GOAL_NAME[task_id]}"
        for task_id, count in task_counts.items()
        if count <= 0
    ]
    if missing_tasks:
        missing_str = ", ".join(missing_tasks)
        raise ValueError(
            "Dataset is missing at least one target class. "
            f"Missing tasks: {missing_str}"
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
        available_tasks = [
            task_id for task_id, indices in indices_by_task.items() if indices
        ]
        if not available_tasks:
            break
        rng.shuffle(available_tasks)
        for task_id in available_tasks:
            mixed_order.append(indices_by_task[task_id].pop())

    return np.asarray(mixed_order, dtype=np.int64)


def process_demonstration_dataset(
    raw_dataset,
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

    map_config = build_default_map_config() if config is None else config
    raw_paths = [
        _as_path_array(episode.path_xy).astype(np.float32, copy=False)
        for episode in raw_dataset.episodes
    ]
    compressed_index_paths = [
        compress_dense_path_to_collision_free_indices(
            episode.path_xy,
            config=map_config,
            resolution=DEFAULT_RESAMPLED_SEGMENT_CHECK_RESOLUTION,
        )
        for episode in raw_dataset.episodes
    ]
    compressed_lengths = np.asarray(
        [len(index_path) for index_path in compressed_index_paths],
        dtype=np.int64,
    )
    print(
        "[info] collision-free compression stats: "
        f"min={int(compressed_lengths.min())}, "
        f"avg={float(compressed_lengths.mean()):.2f}, "
        f"max={int(compressed_lengths.max())}"
    )
    resolved_t_fixed = resolve_resampled_horizon(
        compressed_index_paths=compressed_index_paths,
        requested_t_fixed=int(t_fixed),
    )
    num_episodes = len(raw_dataset.episodes)
    observations = np.zeros((num_episodes, resolved_t_fixed, 2), dtype=np.float32)
    actions = np.zeros((num_episodes, resolved_t_fixed, 2), dtype=np.float32)
    path_signatures: np.ndarray | None = None
    task_ids = np.zeros(num_episodes, dtype=np.int64)
    task_code_bits = np.zeros((num_episodes, 2), dtype=np.int64)
    goal_onehot = np.zeros((num_episodes, len(TASK_ID_VALUES)), dtype=np.float32)
    phase_labels = (
        np.zeros((num_episodes, resolved_t_fixed), dtype=np.int64)
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

    raw_task_ids = np.asarray(
        [episode.task_id for episode in raw_dataset.episodes],
        dtype=np.int64,
    )
    task_counts = validate_task_coverage(raw_task_ids)
    episode_order = build_balanced_episode_order(raw_task_ids, raw_dataset.seed)
    resolved_signature_backend: str | None = None
    if include_path_signatures:
        resolved_signature_backend = resolve_signature_backend(path_signature_backend)
        window_label = (
            "all_prefix"
            if path_signature_window_size <= 0
            else str(path_signature_window_size)
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
    progress_interval = max(1, min(25, num_episodes))

    for episode_index, source_index in enumerate(episode_order.tolist()):
        episode = raw_dataset.episodes[source_index]
        resampled_path = expand_collision_free_polyline_fixed_length(
            raw_paths[source_index],
            compressed_indices=compressed_index_paths[source_index],
            t_fixed=resolved_t_fixed,
            config=map_config,
            resolution=DEFAULT_RESAMPLED_SEGMENT_CHECK_RESOLUTION,
        )
        invalid_segment_count, first_invalid_segment_index = (
            count_invalid_resampled_segments(
                resampled_path,
                config=map_config,
                resolution=DEFAULT_RESAMPLED_SEGMENT_CHECK_RESOLUTION,
            )
        )
        if invalid_segment_count > 0:
            raise ValueError(
                "Resampled path contains invalid straight-line segments. "
                f"episode_id={episode.episode_id}, task_id={episode.task_id}, "
                f"t_fixed={resolved_t_fixed}, invalid_segments={invalid_segment_count}, "
                f"first_invalid_segment={first_invalid_segment_index}. "
                "This should not happen after collision-free compression and "
                "uniform dense segment resampling."
            )
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
                    (num_episodes, resolved_t_fixed, int(signature_sequence.shape[1])),
                    dtype=np.float32,
                )
            path_signatures[episode_index] = signature_sequence
        if (
            episode_index == 0
            or episode_index + 1 == num_episodes
            or (episode_index + 1) % progress_interval == 0
        ):
            print(
                "[info] processed demonstrations: "
                f"{episode_index + 1}/{num_episodes}"
            )
        task_ids[episode_index] = episode.task_id
        task_code_bits[episode_index] = build_task_code_bits(episode.task_id)
        goal_onehot[episode_index] = build_goal_onehot(episode.task_id)
        if phase_labels is not None:
            phase_annotation = annotate_trajectory_phases(
                [tuple(map(float, point_xy)) for point_xy in resampled_path.tolist()],
                config=map_config,
            )
            phase_labels[episode_index] = encode_phase_labels(
                phase_annotation.phase_labels
            )

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
        t_fixed=resolved_t_fixed,
        action_padding_mode=last_action_mode,
        path_signature_key=None if path_signatures is None else str(path_signature_key),
        path_signature_window_size=(
            0 if path_signatures is None else int(path_signature_window_size)
        ),
        path_signature_depth=0 if path_signatures is None else int(path_signature_depth),
        path_signature_backend=(
            None if path_signatures is None else resolved_signature_backend
        ),
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
        "format_version": np.asarray("braidedhub_act_ready_fourstart_implicitcue_sig_v4"),
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
        "path_signature_key": np.asarray(
            "" if dataset.path_signature_key is None else dataset.path_signature_key
        ),
        "path_signature_window_size": np.asarray(
            dataset.path_signature_window_size,
            dtype=np.int64,
        ),
        "path_signature_depth": np.asarray(
            dataset.path_signature_depth,
            dtype=np.int64,
        ),
        "path_signature_backend": np.asarray(
            "" if dataset.path_signature_backend is None else dataset.path_signature_backend
        ),
        "phase_label_vocab": np.asarray(dataset.phase_label_vocab, dtype="<U32"),
        "source_num_per_task_requested": np.asarray(
            dataset.source_num_per_task_requested,
            dtype=np.int64,
        ),
        "source_solve_time": np.asarray(dataset.source_solve_time, dtype=np.float32),
        "source_retries_per_demo": np.asarray(
            dataset.source_retries_per_demo,
            dtype=np.int64,
        ),
    }
    if dataset.phase_labels is not None:
        save_kwargs["phase_labels"] = dataset.phase_labels
    if dataset.path_signatures is not None:
        save_kwargs["path_signatures"] = dataset.path_signatures

    np.savez_compressed(output_path, **save_kwargs)
    print(f"Saved processed dataset with {len(dataset)} samples to {output_path}")
    return output_path


def dataset_summary(dataset: ProcessedDemonstrationDataset) -> dict[str, Any]:
    num_samples = len(dataset)
    task_counts = {
        task_id: int(np.sum(dataset.task_ids == task_id)) for task_id in TASK_ID_VALUES
    }
    avg_raw_steps = float(dataset.raw_path_lengths.mean()) if num_samples > 0 else 0.0
    avg_raw_distance = (
        float(dataset.raw_path_distances.mean()) if num_samples > 0 else 0.0
    )

    print("Processed dataset summary")
    print(f"  num_samples={num_samples}")
    print(f"  fixed_horizon={dataset.t_fixed}")
    print(f"  observation_shape={tuple(dataset.observations.shape)}")
    print(f"  action_shape={tuple(dataset.actions.shape)}")
    if dataset.path_signatures is not None:
        window_label = (
            "all_prefix"
            if dataset.path_signature_window_size <= 0
            else str(dataset.path_signature_window_size)
        )
        print(f"  path_signature_shape={tuple(dataset.path_signatures.shape)}")
        print(
            "  path_signature="
            f"{dataset.path_signature_key}, "
            f"window={window_label}, "
            f"depth={dataset.path_signature_depth}, "
            f"backend={dataset.path_signature_backend}"
        )
    print(f"  avg_raw_path_steps={avg_raw_steps:.2f}")
    print(f"  avg_raw_path_distance={avg_raw_distance:.2f}")
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
            f"avg_steps={task_avg_steps:.2f}, avg_distance={task_avg_distance:.2f}"
        )

    return {
        "num_samples": num_samples,
        "fixed_horizon": dataset.t_fixed,
        "task_counts": task_counts,
        "avg_raw_path_steps": avg_raw_steps,
        "avg_raw_path_distance": avg_raw_distance,
    }


def generate_lerobot_v30_dataset(
    processed_dataset: ProcessedDemonstrationDataset,
    output_dir: str | Path = DEFAULT_DATASET_ROOT,
    fps: int = DEFAULT_VIDEO_FPS,
    image_size: int = DEFAULT_VIDEO_IMAGE_SIZE,
    config: MapConfig | None = None,
    episodes_per_chunk: int = DEFAULT_LEROBOT_EPISODES_PER_CHUNK,
    use_delta_signature: bool = False,
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

    base_image = make_lerobot_base_image(map_config, image_size=image_size)
    video_keys = [VIDEO_KEY]
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
    if use_delta_signature:
        if processed_dataset.path_signatures is None:
            raise ValueError(
                "`use_delta_signature=True` requires processed_dataset.path_signatures."
            )
        records[DEFAULT_DELTA_SIGNATURE_KEY] = []
    episodes_meta: list[dict[str, Any]] = []
    global_index = 0
    video_frame_counts_by_key: dict[str, dict[int, int]] = {
        video_key: {} for video_key in video_keys
    }
    visual_stats_by_key: dict[str, dict[str, Any]] = {
        video_key: init_visual_feature_stats() for video_key in video_keys
    }
    data_files: list[Path] = []
    video_files: list[Path] = []
    first_video_info_by_key: dict[str, dict[str, Any]] = {}

    for episode_idx, source_idx in enumerate(episode_order.tolist()):
        observations = processed_dataset.observations[source_idx]
        actions = processed_dataset.actions[source_idx]
        goal_xy = processed_dataset.goal_xy[source_idx]
        target_goal_name = str(processed_dataset.target_goal_names[source_idx])
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
            root
            / f"videos/{VIDEO_KEY}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        data_file.parent.mkdir(parents=True, exist_ok=True)
        video_file.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_proc = start_ffmpeg_raw_writer(video_file, image_size, image_size, fps)
        if ffmpeg_proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin for raw video writing.")
        episode_base_image = make_lerobot_episode_base_image(
            base_image,
            config=map_config,
            target_goal_name=target_goal_name,
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        )

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
        if use_delta_signature:
            episode_records[DEFAULT_DELTA_SIGNATURE_KEY] = []
        episode_delta_signatures = (
            None
            if not use_delta_signature
            else compute_delta_signature_sequence_np(
                processed_dataset.path_signatures[source_idx]
            )
        )

        for frame_idx in range(episode_length):
            state_xy = observations[frame_idx]
            action_xy = actions[frame_idx]
            signature_xy = (
                None
                if processed_dataset.path_signatures is None
                else processed_dataset.path_signatures[source_idx, frame_idx]
            )
            delta_signature_xy = (
                None
                if episode_delta_signatures is None
                else episode_delta_signatures[frame_idx]
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
                if delta_signature_xy is not None:
                    target_records[DEFAULT_DELTA_SIGNATURE_KEY].append(
                        delta_signature_xy.astype(np.float32).tolist()
                    )

            frame = render_lerobot_frame(
                episode_base_image,
                config=map_config,
                robot_xy=(float(state_xy[0]), float(state_xy[1])),
            )
            update_visual_feature_stats(visual_stats_by_key[VIDEO_KEY], frame)
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
        if VIDEO_KEY not in first_video_info_by_key:
            first_video_info_by_key[VIDEO_KEY] = video_info
        video_frame_counts_by_key[VIDEO_KEY][episode_idx] = int(video_info["frames"])
        data_files.append(data_file)
        video_files.append(video_file)

        episode_meta = {
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
        episodes_meta.append(episode_meta)

    total_frames = len(records["index"])
    state_array = np.asarray(records["observation.state"], dtype=np.float32)
    action_array = np.asarray(records["action"], dtype=np.float32)
    signature_array = (
        None
        if processed_dataset.path_signatures is None
        else np.asarray(records[DEFAULT_PATH_SIGNATURE_KEY], dtype=np.float32)
    )
    delta_signature_array = (
        None
        if not use_delta_signature
        else np.asarray(records[DEFAULT_DELTA_SIGNATURE_KEY], dtype=np.float32)
    )

    episode_arrays = [
        pa.array([episode["episode_index"] for episode in episodes_meta], type=pa.int64()),
        pa.array([episode["tasks"] for episode in episodes_meta], type=pa.list_(pa.string())),
        pa.array([episode["length"] for episode in episodes_meta], type=pa.int64()),
        pa.array([episode["data/chunk_index"] for episode in episodes_meta], type=pa.int64()),
        pa.array([episode["data/file_index"] for episode in episodes_meta], type=pa.int64()),
        pa.array([episode["dataset_from_index"] for episode in episodes_meta], type=pa.int64()),
        pa.array([episode["dataset_to_index"] for episode in episodes_meta], type=pa.int64()),
    ]
    episode_names = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for video_key in video_keys:
        episode_arrays.extend(
            [
                pa.array([episode[f"videos/{video_key}/chunk_index"] for episode in episodes_meta], type=pa.int64()),
                pa.array([episode[f"videos/{video_key}/file_index"] for episode in episodes_meta], type=pa.int64()),
                pa.array([episode[f"videos/{video_key}/from_timestamp"] for episode in episodes_meta], type=pa.float32()),
                pa.array([episode[f"videos/{video_key}/to_timestamp"] for episode in episodes_meta], type=pa.float32()),
            ]
        )
        episode_names.extend(
            [
                f"videos/{video_key}/chunk_index",
                f"videos/{video_key}/file_index",
                f"videos/{video_key}/from_timestamp",
                f"videos/{video_key}/to_timestamp",
            ]
        )
    episode_arrays.extend(
        [
            pa.array([episode["meta/episodes/chunk_index"] for episode in episodes_meta], type=pa.int64()),
            pa.array([episode["meta/episodes/file_index"] for episode in episodes_meta], type=pa.int64()),
        ]
    )
    episode_names.extend(
        [
            "meta/episodes/chunk_index",
            "meta/episodes/file_index",
        ]
    )
    episodes_table = pa.Table.from_arrays(episode_arrays, names=episode_names)
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

    if VIDEO_KEY not in first_video_info_by_key:
        raise RuntimeError("No episodes were exported, so no video metadata was created.")
    front_video_info = first_video_info_by_key[VIDEO_KEY]

    total_episodes = len(processed_dataset)
    val_start = int(round(total_episodes * 0.8))
    splits = {
        "train": f"0:{val_start}",
        "val": f"{val_start}:{total_episodes}",
    }

    info = {
        "codebase_version": "v3.0",
        "robot_type": "point_mass_2d_braidedhub_fourstart_implicit_cue",
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
                "names": ["agent_x", "agent_y"],
            },
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["delta_x", "delta_y"],
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
                "shape": [front_video_info["height"], front_video_info["width"], 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": front_video_info["height"],
                    "video.width": front_video_info["width"],
                    "video.codec": front_video_info["codec"],
                    "video.pix_fmt": front_video_info["pix_fmt"],
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
            "names": [
                f"path_sig_{index}" for index in range(int(signature_array.shape[1]))
            ],
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
    if delta_signature_array is not None:
        info["features"][DEFAULT_DELTA_SIGNATURE_KEY] = {
            "dtype": "float32",
            "shape": [int(delta_signature_array.shape[1])],
            "names": [
                f"delta_path_sig_{index}"
                for index in range(int(delta_signature_array.shape[1]))
            ],
        }
        info["delta_signature"] = {
            "key": DEFAULT_DELTA_SIGNATURE_KEY,
            "signature_key": DEFAULT_PATH_SIGNATURE_KEY,
            "definition": "path_signature_t - path_signature_{t-1}",
            "first_step_rule": "zeros",
            "signature_dim": int(delta_signature_array.shape[1]),
        }

    stats = {
        "observation.state": build_stats(state_array),
        "action": build_stats(action_array),
        "next.reward": build_stats(np.asarray(records["next.reward"], dtype=np.float32)),
        "timestamp": build_stats(np.asarray(records["timestamp"], dtype=np.float32)),
        VIDEO_KEY: finalize_visual_feature_stats(visual_stats_by_key[VIDEO_KEY]),
    }
    if (
        processed_dataset.path_signatures is not None
        and processed_dataset.path_signature_key is not None
        and signature_array is not None
    ):
        stats[processed_dataset.path_signature_key] = build_stats(signature_array)
    if delta_signature_array is not None:
        stats[DEFAULT_DELTA_SIGNATURE_KEY] = build_stats(delta_signature_array)

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
        video_frame_counts_by_key=video_frame_counts_by_key,
    )

    front_video_frames = sum(video_frame_counts_by_key[VIDEO_KEY].values())
    print(f"Generated LeRobotDataset v3.0 at: {root.resolve()}")
    print(
        f"Episodes: {total_episodes}, Frames: {total_frames}, "
        f"Video frames per stream: {front_video_frames}, "
        f"Video streams: {len(video_keys)}, "
        f"Data files: {len(data_files)}, Video files: {len(video_files)}"
    )
    return root
