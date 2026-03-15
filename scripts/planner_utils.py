from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .shared_channel_double_loop_map import (
        DEFAULT_RANDOM_SEED,
        BraidedHub2DEnv,
        MapConfig,
        TASK_ID_TO_GOAL_NAME,
        build_task_spec,
        build_default_map_config,
        get_phase_name,
        get_task_start_region,
        get_task_waypoint_centers,
        is_state_valid,
        plot_map,
        sample_valid_state_in_region,
    )
except ImportError:
    from shared_channel_double_loop_map import (
        DEFAULT_RANDOM_SEED,
        BraidedHub2DEnv,
        MapConfig,
        TASK_ID_TO_GOAL_NAME,
        build_task_spec,
        build_default_map_config,
        get_phase_name,
        get_task_start_region,
        get_task_waypoint_centers,
        is_state_valid,
        plot_map,
        sample_valid_state_in_region,
    )


DEFAULT_SOLVE_TIME = 1.0
DEFAULT_GOAL_NAME = "G00"
DEFAULT_STEP_SIZE = 2.5
DEFAULT_CONNECT_TOLERANCE = 1.0
DEFAULT_COLLISION_CHECK_RESOLUTION = 0.5
DEFAULT_GOAL_SAMPLE_PROBABILITY = 0.15
DEFAULT_MAX_ITERATIONS = 20000
DEFAULT_RETRIES_PER_DEMO = 5
DEFAULT_LOW_SUCCESS_WARNING_THRESHOLD = 0.8
DEFAULT_DATASET_VIS_SAMPLES = 12
DEFAULT_DATASET_OUTPUT = (
    "/home/jeong/zeno/corl/main/scripts/braidedhub_fourstart_implicit_cue_rrtconnect_demos.npz"
)
TASK_COLOR_BY_ID = {
    0: "#1b9e77",
    1: "#66a61e",
    2: "#d95f02",
    3: "#7570b3",
}
BRANCH1_PHASE_BY_BIT = {0: "branch1_upper_region", 1: "branch1_lower_region"}
BRANCH2_PHASE_BY_BIT = {0: "branch2_upper_region", 1: "branch2_lower_region"}
BRANCH1_PHASES = frozenset(BRANCH1_PHASE_BY_BIT.values())
BRANCH2_PHASES = frozenset(BRANCH2_PHASE_BY_BIT.values())


@dataclass(slots=True)
class _Node:
    point: tuple[float, float]
    parent: int | None


@dataclass(slots=True)
class DemonstrationEpisode:
    """One successful task-conditioned planning trajectory."""

    episode_id: int
    task_id: int
    target_goal_name: str
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    path_xy: np.ndarray
    path_length: int
    success: bool


@dataclass(slots=True)
class DemonstrationDataset:
    """Batch of generated demonstrations plus generation metadata."""

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


def _resolve_config(config: MapConfig | None) -> MapConfig:
    return build_default_map_config() if config is None else config


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return math.hypot(dx, dy)


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
    scale = max_distance / distance
    return (
        float(from_point[0] + dx * scale),
        float(from_point[1] + dy * scale),
    )


def _nearest_node_index(tree: list[_Node], point: tuple[float, float]) -> int:
    return min(
        range(len(tree)),
        key=lambda node_index: _distance(tree[node_index].point, point),
    )


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
    path.reverse()
    return path


def _reconstruct_bidirectional_path(
    tree_a: list[_Node],
    tree_b: list[_Node],
    connect_index_a: int,
    connect_index_b: int,
) -> list[tuple[float, float]]:
    path_a = _path_to_root(tree_a, connect_index_a)
    path_b = _path_to_root(tree_b, connect_index_b)
    path_b.reverse()
    if path_a[-1] == path_b[0]:
        return path_a + path_b[1:]
    return path_a + path_b


def _shortcut_path(
    path: list[tuple[float, float]],
    config: MapConfig,
    collision_check_resolution: float,
    rng: random.Random,
    num_attempts: int = 100,
) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path

    shortened_path = list(path)
    for _ in range(num_attempts):
        if len(shortened_path) <= 2:
            break
        index_a, index_b = sorted(rng.sample(range(len(shortened_path)), 2))
        if index_b - index_a <= 1:
            continue
        point_a = shortened_path[index_a]
        point_b = shortened_path[index_b]
        if _is_segment_valid(
            point_a,
            point_b,
            config=config,
            resolution=collision_check_resolution,
        ):
            shortened_path = shortened_path[: index_a + 1] + shortened_path[index_b:]
    return shortened_path


def _densify_path(
    path: list[tuple[float, float]],
    resolution: float,
) -> list[tuple[float, float]]:
    if len(path) <= 1:
        return list(path)
    if resolution <= 0.0:
        raise ValueError("resolution must be positive.")

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


def _get_goal_center_for_task(
    task_id: int,
    config: MapConfig,
) -> tuple[str, tuple[float, float]]:
    goal_name = TASK_ID_TO_GOAL_NAME[task_id]
    goal_region = next(
        (goal for goal in config.goal_regions if goal.name == goal_name),
        None,
    )
    if goal_region is None:
        raise RuntimeError(f"Goal region for task_id={task_id} was not found.")
    return goal_name, goal_region.center


def _validate_task_conditioned_path(
    task_id: int,
    path_xy: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    config: MapConfig,
) -> tuple[bool, str | None]:
    """Check that a demonstration path follows the branch choices encoded by task_id.

    Validation rule:
    - The path must enter the task-matching branch at H1 and H2.
    - The path must not enter the opposite branch at either decision stage.
    """

    task_spec = build_task_spec(task_id)
    expected_branch1 = BRANCH1_PHASE_BY_BIT[int(task_spec.task_bits[0])]
    expected_branch2 = BRANCH2_PHASE_BY_BIT[int(task_spec.task_bits[1])]
    saw_expected_branch1 = False
    saw_expected_branch2 = False

    for point in path_xy:
        phase_name = get_phase_name(float(point[0]), float(point[1]), config=config)
        if phase_name in BRANCH1_PHASES:
            if phase_name != expected_branch1:
                return (
                    False,
                    f"H1 branch mismatch: expected {expected_branch1}, observed {phase_name}",
                )
            saw_expected_branch1 = True
        if phase_name in BRANCH2_PHASES:
            if phase_name != expected_branch2:
                return (
                    False,
                    f"H2 branch mismatch: expected {expected_branch2}, observed {phase_name}",
                )
            saw_expected_branch2 = True

    if not saw_expected_branch1:
        return False, f"Path never entered expected H1 branch {expected_branch1}"
    if not saw_expected_branch2:
        return False, f"Path never entered expected H2 branch {expected_branch2}"
    return True, None


def _build_task_route_waypoints(
    task_id: int,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    config: MapConfig,
) -> tuple[tuple[float, float], ...]:
    """Build a task-aligned waypoint chain through the correct start and branches."""

    branch1_center, branch2_center = get_task_waypoint_centers(task_id, config)
    return (
        (float(start_xy[0]), float(start_xy[1])),
        config.decision_region_h1.rectangles[0].center,
        branch1_center,
        config.merge_region_1.rectangles[0].center,
        config.decision_region_h2.rectangles[0].center,
        branch2_center,
        config.merge_region_2.rectangles[0].center,
        (float(goal_xy[0]), float(goal_xy[1])),
    )


def plan_path_rrtconnect(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    solve_time: float = DEFAULT_SOLVE_TIME,
    config: MapConfig | None = None,
    interpolate_solution: bool = True,
) -> list[tuple[float, float]] | None:
    """Plan a single 2D geometric path with a self-contained RRTConnect."""

    resolved_config = _resolve_config(config)
    if solve_time <= 0.0:
        raise ValueError("solve_time must be positive.")
    if not is_state_valid(*start_xy, config=resolved_config):
        raise ValueError(f"Start state {start_xy} is not in free space.")
    if not is_state_valid(*goal_xy, config=resolved_config):
        raise ValueError(f"Goal state {goal_xy} is not in free space.")

    if _is_segment_valid(
        start_xy,
        goal_xy,
        config=resolved_config,
        resolution=DEFAULT_COLLISION_CHECK_RESOLUTION,
    ):
        straight_path = [start_xy, goal_xy]
        if interpolate_solution:
            return _densify_path(
                straight_path,
                resolution=DEFAULT_COLLISION_CHECK_RESOLUTION,
            )
        return straight_path

    seed = (
        hash(
            (
                round(start_xy[0], 4),
                round(start_xy[1], 4),
                round(goal_xy[0], 4),
                round(goal_xy[1], 4),
            )
        )
        & 0xFFFFFFFF
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
            sample_point = (
                tree_goal[0].point if not trees_swapped else tree_start[0].point
            )
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
                if interpolate_solution:
                    return _densify_path(
                        path,
                        resolution=DEFAULT_COLLISION_CHECK_RESOLUTION,
                    )
                return path

        tree_start, tree_goal = tree_goal, tree_start
        trees_swapped = not trees_swapped

    return None


def plan_path_rrtconnect_via_waypoints(
    waypoints_xy: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    solve_time: float = DEFAULT_SOLVE_TIME,
    config: MapConfig | None = None,
    interpolate_solution: bool = True,
) -> list[tuple[float, float]] | None:
    """Plan a piecewise path that must pass through intermediate task waypoints."""

    if len(waypoints_xy) < 2:
        raise ValueError("waypoints_xy must contain at least a start and a goal.")

    resolved_config = _resolve_config(config)
    concatenated_path: list[tuple[float, float]] = []
    for start_point, goal_point in zip(waypoints_xy[:-1], waypoints_xy[1:], strict=False):
        segment_path = plan_path_rrtconnect(
            start_xy=start_point,
            goal_xy=goal_point,
            solve_time=solve_time,
            config=resolved_config,
            interpolate_solution=interpolate_solution,
        )
        if segment_path is None:
            return None
        if concatenated_path:
            concatenated_path.extend(segment_path[1:])
        else:
            concatenated_path.extend(segment_path)
    return concatenated_path


def plot_planned_path(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    path: list[tuple[float, float]] | None,
    config: MapConfig | None = None,
    show: bool = True,
):
    """Overlay a planned path on top of the existing static map."""

    resolved_config = _resolve_config(config)
    fig, ax = plot_map(resolved_config, show=False)

    ax.scatter(
        start_xy[0],
        start_xy[1],
        s=90,
        c="#1f77b4",
        edgecolors="white",
        linewidths=1.2,
        zorder=7,
        label="Start",
    )
    ax.scatter(
        goal_xy[0],
        goal_xy[1],
        s=90,
        c="#d62728",
        edgecolors="white",
        linewidths=1.2,
        zorder=7,
        label="Goal",
    )

    if path:
        path_xs = [point[0] for point in path]
        path_ys = [point[1] for point in path]
        ax.plot(
            path_xs,
            path_ys,
            color="#111111",
            linewidth=2.4,
            alpha=0.9,
            zorder=6,
            label="RRTConnect Path",
        )

    ax.set_title("Custom RRTConnect Planning on BraidedHub2D Map")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    if show:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "matplotlib is required to visualize the planned path."
            ) from exc
        plt.show()

    return fig, ax


def _make_episode(
    episode_id: int,
    task_id: int,
    target_goal_name: str,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    path_xy: list[tuple[float, float]],
) -> DemonstrationEpisode:
    path_array = np.asarray(path_xy, dtype=np.float64)
    return DemonstrationEpisode(
        episode_id=episode_id,
        task_id=task_id,
        target_goal_name=target_goal_name,
        start_xy=(float(start_xy[0]), float(start_xy[1])),
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
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
    config: MapConfig | None = None,
) -> DemonstrationDataset:
    """Generate task-conditioned RRTConnect demonstrations for all four tasks.

    Goal selection policy:
    - This generator uses the target goal-region center, not random goal samples.
    - The center is deterministic, always well inside the goal rectangle, and
      therefore more stable for reproducible dataset generation.
    Route selection policy:
    - The path is forced through two intermediate branch waypoints derived from
      the task bits, so the trajectory itself carries the implicit 2-bit code.
    - Generated paths are rejected unless their semantic branch phases strictly
      match the task id at both H1 and H2.
    """

    if num_per_task <= 0:
        raise ValueError("num_per_task must be positive.")
    if max_retries_per_demo <= 0:
        raise ValueError("max_retries_per_demo must be positive.")

    resolved_config = _resolve_config(config)
    env = BraidedHub2DEnv(map_config=resolved_config, rng_seed=seed)

    episodes: list[DemonstrationEpisode] = []
    success_counts_by_task = {task_id: 0 for task_id in TASK_ID_TO_GOAL_NAME}
    attempt_counts_by_task = {task_id: 0 for task_id in TASK_ID_TO_GOAL_NAME}
    skipped_counts_by_task = {task_id: 0 for task_id in TASK_ID_TO_GOAL_NAME}

    episode_id = 0
    for task_id in sorted(TASK_ID_TO_GOAL_NAME):
        task_spec = build_task_spec(task_id)
        target_goal_name, goal_xy = _get_goal_center_for_task(task_id, resolved_config)
        print(
            f"[task {task_id}] target={target_goal_name}, "
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
                        f"task_id={task_id}, expected={task_spec.start_region_name}, "
                        f"got={env.start_region_name}"
                    )
                route_waypoints = _build_task_route_waypoints(
                    task_id=task_id,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    config=resolved_config,
                )
                success_path = plan_path_rrtconnect_via_waypoints(
                    waypoints_xy=route_waypoints,
                    solve_time=solve_time,
                    config=resolved_config,
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
                    target_goal_name=target_goal_name,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
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
    """Save demonstrations to a compressed NPZ with padded variable-length paths."""

    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_episodes = len(dataset.episodes)
    max_path_length = max(
        (episode.path_length for episode in dataset.episodes),
        default=0,
    )
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
        format_version=np.asarray("braidedhub_rrtconnect_fourstart_implicitcue_v3"),
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
            [dataset.success_counts_by_task[task_id] for task_id in sorted(TASK_ID_TO_GOAL_NAME)],
            dtype=np.int64,
        ),
        attempt_counts_by_task=np.asarray(
            [dataset.attempt_counts_by_task[task_id] for task_id in sorted(TASK_ID_TO_GOAL_NAME)],
            dtype=np.int64,
        ),
        skipped_counts_by_task=np.asarray(
            [dataset.skipped_counts_by_task[task_id] for task_id in sorted(TASK_ID_TO_GOAL_NAME)],
            dtype=np.int64,
        ),
    )
    print(f"Saved {num_episodes} demonstrations to {output_path}")
    return output_path


def visualize_dataset_samples(
    dataset: DemonstrationDataset,
    num_samples: int = DEFAULT_DATASET_VIS_SAMPLES,
    seed: int | None = None,
    config: MapConfig | None = None,
    show: bool = True,
):
    """Overlay a random subset of dataset trajectories, colored by task id."""

    if len(dataset.episodes) == 0:
        raise ValueError("Dataset is empty. Nothing to visualize.")

    resolved_config = _resolve_config(config)
    fig, ax = plot_map(resolved_config, show=False)

    rng = random.Random(dataset.seed if seed is None else seed)
    sampled_episodes = rng.sample(
        dataset.episodes,
        k=min(num_samples, len(dataset.episodes)),
    )

    used_task_ids: set[int] = set()
    for episode in sampled_episodes:
        color = TASK_COLOR_BY_ID[episode.task_id]
        label = None
        if episode.task_id not in used_task_ids:
            label = f"task {episode.task_id} -> {episode.target_goal_name}"
            used_task_ids.add(episode.task_id)

        ax.plot(
            episode.path_xy[:, 0],
            episode.path_xy[:, 1],
            color=color,
            linewidth=1.8,
            alpha=0.65,
            zorder=6,
            label=label,
        )
        ax.scatter(
            episode.start_xy[0],
            episode.start_xy[1],
            s=18,
            c=color,
            alpha=0.8,
            linewidths=0.0,
            zorder=7,
        )

    ax.set_title(
        f"Dataset Samples: {len(sampled_episodes)} trajectories from {len(dataset.episodes)} saved episodes"
    )
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    if show:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "matplotlib is required to visualize dataset samples."
            ) from exc
        plt.show()

    return fig, ax


def run_single_rrtconnect_demo(
    solve_time: float = DEFAULT_SOLVE_TIME,
    goal_name: str = DEFAULT_GOAL_NAME,
    rng_seed: int = DEFAULT_RANDOM_SEED,
    show_plot: bool = True,
) -> dict[str, Any]:
    """Sample one start state, plan to one goal center, and visualize the result."""

    resolved_config = build_default_map_config()
    goal_to_task_id = {
        mapped_goal_name: task_id for task_id, mapped_goal_name in TASK_ID_TO_GOAL_NAME.items()
    }
    goal_region = next(
        (goal for goal in resolved_config.goal_regions if goal.name == goal_name),
        None,
    )
    if goal_region is None:
        raise ValueError(f"Unknown goal name: {goal_name}")
    if goal_name not in goal_to_task_id:
        raise ValueError(f"Goal {goal_name} is not associated with a task id.")

    rng = random.Random(rng_seed)
    start_xy = sample_valid_state_in_region(
        get_task_start_region(goal_to_task_id[goal_name], resolved_config),
        rng=rng,
        config=resolved_config,
    )
    goal_xy = goal_region.center
    route_waypoints = _build_task_route_waypoints(
        task_id=goal_to_task_id[goal_name],
        start_xy=start_xy,
        goal_xy=goal_xy,
        config=resolved_config,
    )
    path = plan_path_rrtconnect_via_waypoints(
        waypoints_xy=route_waypoints,
        solve_time=solve_time,
        config=resolved_config,
    )

    if path is not None and show_plot:
        plot_planned_path(start_xy, goal_xy, path, config=resolved_config, show=True)

    return {
        "start_xy": start_xy,
        "goal_xy": goal_xy,
        "goal_name": goal_name,
        "path": path,
    }


def main() -> None:
    dataset = generate_demonstrations(
        num_per_task=20,
        seed=DEFAULT_RANDOM_SEED,
    )

    try:
        visualize_dataset_samples(
            dataset,
            num_samples=DEFAULT_DATASET_VIS_SAMPLES,
            seed=DEFAULT_RANDOM_SEED,
            show=True,
        )
    except RuntimeError as exc:
        print(f"Visualization skipped: {exc}")

    print(f"Dataset generation complete: {len(dataset)} episodes prepared in memory.")


if __name__ == "__main__":
    main()
