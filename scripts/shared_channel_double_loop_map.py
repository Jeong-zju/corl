from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

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
    merge_region_1: "SemanticRegion"
    middle_corridor_region: "SemanticRegion"
    decision_region_h2: "SemanticRegion"
    branch2_upper_region: "SemanticRegion"
    branch2_lower_region: "SemanticRegion"
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
        "start_00_prefix", xmin=8.0, xmax=12.0, ymin=46.0, ymax=48.0
    )
    start_01_prefix = RectangleRegion(
        "start_01_prefix", xmin=8.0, xmax=12.0, ymin=36.0, ymax=38.0
    )
    start_10_prefix = RectangleRegion(
        "start_10_prefix", xmin=8.0, xmax=12.0, ymin=22.0, ymax=24.0
    )
    start_11_prefix = RectangleRegion(
        "start_11_prefix", xmin=8.0, xmax=12.0, ymin=12.0, ymax=14.0
    )
    start_collector = RectangleRegion(
        "start_collector", xmin=12.0, xmax=14.0, ymin=12.0, ymax=48.0
    )
    shared_entry = RectangleRegion(
        "shared_entry", xmin=14.0, xmax=18.0, ymin=28.0, ymax=32.0
    )

    shared_snake_1 = RectangleRegion(
        "shared_snake_1", xmin=18.0, xmax=24.0, ymin=28.0, ymax=32.0
    )
    shared_snake_2 = RectangleRegion(
        "shared_snake_2", xmin=22.0, xmax=24.0, ymin=28.0, ymax=42.0
    )
    shared_snake_3 = RectangleRegion(
        "shared_snake_3", xmin=24.0, xmax=30.0, ymin=38.0, ymax=42.0
    )
    shared_snake_4 = RectangleRegion(
        "shared_snake_4", xmin=30.0, xmax=32.0, ymin=24.0, ymax=42.0
    )
    shared_snake_5 = RectangleRegion(
        "shared_snake_5", xmin=32.0, xmax=34.0, ymin=24.0, ymax=28.0
    )

    h1_hub = RectangleRegion("H1_hub", xmin=34.0, xmax=38.0, ymin=20.0, ymax=36.0)
    branch1_upper_top = RectangleRegion(
        "branch1_upper_top", xmin=40.0, xmax=48.0, ymin=42.0, ymax=46.0
    )
    branch1_upper_rise = RectangleRegion(
        "branch1_upper_rise", xmin=38.0, xmax=40.0, ymin=34.0, ymax=46.0
    )
    branch1_upper_drop = RectangleRegion(
        "branch1_upper_drop", xmin=48.0, xmax=50.0, ymin=32.0, ymax=46.0
    )
    branch1_lower_bottom = RectangleRegion(
        "branch1_lower_bottom", xmin=40.0, xmax=48.0, ymin=10.0, ymax=14.0
    )
    branch1_lower_drop = RectangleRegion(
        "branch1_lower_drop", xmin=38.0, xmax=40.0, ymin=10.0, ymax=22.0
    )
    branch1_lower_rise = RectangleRegion(
        "branch1_lower_rise", xmin=48.0, xmax=50.0, ymin=10.0, ymax=28.0
    )
    merge1_hub = RectangleRegion("merge1_hub", xmin=50.0, xmax=54.0, ymin=24.0, ymax=36.0)

    middle_snake_1 = RectangleRegion(
        "middle_snake_1", xmin=54.0, xmax=60.0, ymin=28.0, ymax=32.0
    )
    middle_snake_2 = RectangleRegion(
        "middle_snake_2", xmin=60.0, xmax=62.0, ymin=28.0, ymax=40.0
    )
    middle_snake_3 = RectangleRegion(
        "middle_snake_3", xmin=62.0, xmax=68.0, ymin=36.0, ymax=40.0
    )
    middle_snake_4 = RectangleRegion(
        "middle_snake_4", xmin=68.0, xmax=70.0, ymin=18.0, ymax=40.0
    )
    middle_snake_5 = RectangleRegion(
        "middle_snake_5", xmin=70.0, xmax=72.0, ymin=18.0, ymax=22.0
    )

    h2_hub = RectangleRegion("H2_hub", xmin=72.0, xmax=76.0, ymin=14.0, ymax=36.0)
    branch2_upper_top = RectangleRegion(
        "branch2_upper_top", xmin=78.0, xmax=86.0, ymin=40.0, ymax=44.0
    )
    branch2_upper_rise = RectangleRegion(
        "branch2_upper_rise", xmin=76.0, xmax=78.0, ymin=32.0, ymax=44.0
    )
    branch2_upper_drop = RectangleRegion(
        "branch2_upper_drop", xmin=86.0, xmax=88.0, ymin=30.0, ymax=44.0
    )
    branch2_lower_bottom = RectangleRegion(
        "branch2_lower_bottom", xmin=78.0, xmax=86.0, ymin=8.0, ymax=12.0
    )
    branch2_lower_drop = RectangleRegion(
        "branch2_lower_drop", xmin=76.0, xmax=78.0, ymin=8.0, ymax=16.0
    )
    branch2_lower_rise = RectangleRegion(
        "branch2_lower_rise", xmin=86.0, xmax=88.0, ymin=8.0, ymax=28.0
    )
    merge2_hub = RectangleRegion("merge2_hub", xmin=88.0, xmax=92.0, ymin=24.0, ymax=36.0)
    final_corridor = RectangleRegion(
        "final_corridor", xmin=92.0, xmax=96.0, ymin=28.0, ymax=32.0
    )
    terminal_hub = RectangleRegion("terminal_hub", xmin=96.0, xmax=98.0, ymin=10.0, ymax=50.0)

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
        merge_region_1=merge_region_1,
        middle_corridor_region=middle_corridor_region,
        decision_region_h2=decision_region_h2,
        branch2_upper_region=branch2_upper_region,
        branch2_lower_region=branch2_lower_region,
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

    def reset(self, task_id: int | None = None) -> tuple[float, float]:
        """Start a new episode from a valid point inside the task's start room."""

        chosen_task_id = self.sample_task_id() if task_id is None else int(task_id)
        self.task_spec = build_task_spec(chosen_task_id)
        self.episode_task_id = self.task_spec.task_id
        self.start_region_name = self.task_spec.start_region_name
        self.target_goal_name = self.task_spec.target_goal_name
        self.target_goal_region = self.goal_region_by_name[self.target_goal_name]
        task_start_region = get_task_start_region(chosen_task_id, self.map_config)

        self.state = sample_valid_state_in_region(
            task_start_region,
            rng=self.rng,
            config=self.map_config,
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
            "collision_rejected": False,
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

        Invalid next states are handled by rejection:
        - If the proposed state is outside bounds or inside an obstacle,
          the robot stays at the previous valid state.
        - No clipping or wall sliding is applied.
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
        collision_rejected = not proposed_valid
        next_state = self.state if collision_rejected else proposed_state
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


def main() -> None:
    env = BraidedHub2DEnv()
    env.manual_test_task_routes()
    manual_test_phase_annotation(show_plot=plt is not None)
    if plt is not None:
        plot_map(env.map_config, show=True)
        env.rollout_with_random_actions(show=True)


if __name__ == "__main__":
    main()
