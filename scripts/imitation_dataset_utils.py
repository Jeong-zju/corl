from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .planner_utils import (
        DEFAULT_DATASET_OUTPUT,
        TASK_COLOR_BY_ID,
        DemonstrationDataset,
        DemonstrationEpisode,
        generate_demonstrations,
        save_demonstrations,
    )
    from .shared_channel_double_loop_map import (
        DEFAULT_RANDOM_SEED,
        MapConfig,
        TASK_ID_TO_GOAL_NAME,
        annotate_trajectory_phases,
        build_default_map_config,
        plot_map,
    )
except ImportError:
    from planner_utils import (
        DEFAULT_DATASET_OUTPUT,
        TASK_COLOR_BY_ID,
        DemonstrationDataset,
        DemonstrationEpisode,
        generate_demonstrations,
        save_demonstrations,
    )
    from shared_channel_double_loop_map import (
        DEFAULT_RANDOM_SEED,
        MapConfig,
        TASK_ID_TO_GOAL_NAME,
        annotate_trajectory_phases,
        build_default_map_config,
        plot_map,
    )


DEFAULT_T_FIXED = 100
DEFAULT_PROCESSED_SAMPLE_INDEX = 0
DEFAULT_PROCESSED_OUTPUT = "/home/jeong/zeno/corl/main/scripts/generated/braidedhub_implicit_cue_act_ready_t100.npz"
DEFAULT_LAST_ACTION_MODE = "zero"
DEFAULT_LEROBOT_V30_OUTPUT = (
    "/home/jeong/zeno/corl/data/zeno-ai/braidedhub_implicit_cue_v30"
)
DEFAULT_VIDEO_FPS = 20
DEFAULT_VIDEO_IMAGE_SIZE = 128
DEFAULT_LEROBOT_EPISODES_PER_CHUNK = 1000
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
ROBOT_OUTER_COLOR = (255, 255, 255)
ROBOT_INNER_COLOR = (240, 70, 70)

TASK_ID_VALUES = tuple(sorted(TASK_ID_TO_GOAL_NAME))
TASK_INDEX_BY_ID = {task_id: index for index, task_id in enumerate(TASK_ID_VALUES)}
TASK_DESCRIPTION_BY_ID = {
    0: "Reach G00 by taking the upper branch at the first split and the upper branch at the second split.",
    1: "Reach G01 by taking the upper branch at the first split and the lower branch at the second split.",
    2: "Reach G10 by taking the lower branch at the first split and the upper branch at the second split.",
    3: "Reach G11 by taking the lower branch at the first split and the lower branch at the second split.",
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


@dataclass(slots=True)
class ProcessedDemonstrationDataset:
    """Fixed-length training tensors derived from raw planning trajectories."""

    observations: np.ndarray
    actions: np.ndarray
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
    """Return the geometric polyline length of a 2D path."""

    path_array = _as_path_array(path_xy)
    if path_array.shape[0] <= 1:
        return 0.0
    deltas = np.diff(path_array, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def resample_path_fixed_length(
    path_xy: np.ndarray | list[tuple[float, float]],
    t_fixed: int,
) -> np.ndarray:
    """Resample a variable-length polyline to a fixed number of states."""

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
    """Construct point-robot delta actions from fixed-length states.

    The final action is padded with zeros so the terminal frame does not encode
    a fictitious motion beyond the last state.
    """

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
    """Encode the target goal as a one-hot vector of length four."""

    if task_id not in TASK_INDEX_BY_ID:
        raise ValueError(f"Unsupported task_id={task_id}.")
    goal_onehot = np.zeros(len(TASK_ID_VALUES), dtype=np.float32)
    goal_onehot[TASK_INDEX_BY_ID[task_id]] = 1.0
    return goal_onehot


def build_task_code_bits(task_id: int) -> np.ndarray:
    """Encode the task id as the implicit two-branch binary code."""

    if task_id not in TASK_INDEX_BY_ID:
        raise ValueError(f"Unsupported task_id={task_id}.")
    task_code = format(task_id, "02b")
    return np.asarray([int(task_code[0]), int(task_code[1])], dtype=np.int64)


def encode_phase_labels(phase_names: tuple[str, ...] | list[str]) -> np.ndarray:
    """Map semantic phase names to integer ids."""

    encoded = np.zeros(len(phase_names), dtype=np.int64)
    fallback_id = PHASE_NAME_TO_ID["free_space_other"]
    for phase_index, phase_name in enumerate(phase_names):
        encoded[phase_index] = PHASE_NAME_TO_ID.get(phase_name, fallback_id)
    return encoded


def decode_phase_labels(phase_ids: np.ndarray) -> tuple[str, ...]:
    """Map integer phase ids back to semantic labels."""

    return tuple(PHASE_LABEL_VOCAB[int(phase_id)] for phase_id in phase_ids.tolist())


def load_processed_dataset(input_path: str | Path) -> ProcessedDemonstrationDataset:
    """Load fixed-length ACT/IL tensors from a saved processed NPZ."""

    input_path = Path(input_path)
    loaded = np.load(input_path, allow_pickle=False)

    required_keys = {
        "observations",
        "actions",
        "task_ids",
        "task_code_bits",
        "goal_onehot",
        "episode_ids",
        "target_goal_names",
        "start_xy",
        "goal_xy",
        "raw_path_lengths",
        "raw_path_distances",
        "success",
        "seed",
        "t_fixed",
        "action_padding_mode",
        "phase_label_vocab",
        "source_num_per_task_requested",
        "source_solve_time",
        "source_retries_per_demo",
    }
    missing_keys = required_keys.difference(loaded.files)
    if missing_keys:
        missing_str = ", ".join(sorted(missing_keys))
        raise ValueError(f"Processed dataset file is missing keys: {missing_str}")

    phase_labels = loaded["phase_labels"] if "phase_labels" in loaded.files else None
    return ProcessedDemonstrationDataset(
        observations=np.asarray(loaded["observations"], dtype=np.float32),
        actions=np.asarray(loaded["actions"], dtype=np.float32),
        task_ids=np.asarray(loaded["task_ids"], dtype=np.int64),
        task_code_bits=np.asarray(loaded["task_code_bits"], dtype=np.int64),
        goal_onehot=np.asarray(loaded["goal_onehot"], dtype=np.float32),
        phase_labels=(
            None if phase_labels is None else np.asarray(phase_labels, dtype=np.int64)
        ),
        episode_ids=np.asarray(loaded["episode_ids"], dtype=np.int64),
        target_goal_names=np.asarray(loaded["target_goal_names"]),
        start_xy=np.asarray(loaded["start_xy"], dtype=np.float32),
        goal_xy=np.asarray(loaded["goal_xy"], dtype=np.float32),
        raw_path_lengths=np.asarray(loaded["raw_path_lengths"], dtype=np.int64),
        raw_path_distances=np.asarray(loaded["raw_path_distances"], dtype=np.float32),
        success=np.asarray(loaded["success"], dtype=bool),
        seed=int(np.asarray(loaded["seed"]).item()),
        t_fixed=int(np.asarray(loaded["t_fixed"]).item()),
        action_padding_mode=str(np.asarray(loaded["action_padding_mode"]).item()),
        phase_label_vocab=tuple(
            str(item) for item in np.asarray(loaded["phase_label_vocab"]).tolist()
        ),
        source_num_per_task_requested=int(
            np.asarray(loaded["source_num_per_task_requested"]).item()
        ),
        source_solve_time=float(np.asarray(loaded["source_solve_time"]).item()),
        source_retries_per_demo=int(
            np.asarray(loaded["source_retries_per_demo"]).item()
        ),
    )


def _require_lerobot_export_dependencies():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pyarrow is required to export LeRobot v3.0 datasets. "
            "Install it first, for example: pip install pyarrow"
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


def make_lerobot_base_image(
    config: MapConfig,
    image_size: int = DEFAULT_VIDEO_IMAGE_SIZE,
) -> np.ndarray:
    """Render a simple RGB top-down map image without matplotlib."""

    base_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    base_image[:, :] = np.asarray(MAP_BACKGROUND_COLOR, dtype=np.uint8)

    for obstacle in config.obstacle_rectangles:
        _fill_region(base_image, obstacle, OBSTACLE_COLOR, config=config)

    _fill_region(base_image, config.start_region, START_COLOR, config=config)
    for goal_region in config.goal_regions:
        _fill_region(
            base_image,
            goal_region,
            GOAL_COLOR_BY_NAME.get(goal_region.name, (0, 0, 0)),
            config=config,
        )

    return base_image


def render_lerobot_frame(
    base_image: np.ndarray,
    config: MapConfig,
    robot_xy: tuple[float, float],
) -> np.ndarray:
    """Overlay the current robot point on top of the static RGB map."""

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
    except FileNotFoundError as exc:  # pragma: no cover - external dependency
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
    except FileNotFoundError as exc:  # pragma: no cover - external dependency
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
    """Build a PyArrow fixed-size list array from a dense numpy array."""

    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def build_episode_data_table(pa: Any, frame_records: dict[str, list[Any]]) -> Any:
    """Convert one episode worth of frame records into a parquet-ready table."""

    state_array = np.asarray(frame_records["observation.state"], dtype=np.float32)
    action_array = np.asarray(frame_records["action"], dtype=np.float32)
    return pa.Table.from_arrays(
        [
            fixed_size_list_array(pa, state_array, 2),
            fixed_size_list_array(pa, action_array, 2),
            pa.array(frame_records["next.reward"], type=pa.float32()),
            pa.array(frame_records["next.done"], type=pa.bool_()),
            pa.array(frame_records["next.success"], type=pa.bool_()),
            pa.array(frame_records["timestamp"], type=pa.float32()),
            pa.array(frame_records["frame_index"], type=pa.int64()),
            pa.array(frame_records["episode_index"], type=pa.int64()),
            pa.array(frame_records["index"], type=pa.int64()),
            pa.array(frame_records["task_index"], type=pa.int64()),
        ],
        names=[
            "observation.state",
            "action",
            "next.reward",
            "next.done",
            "next.success",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    )


def get_chunk_and_file_index(
    episode_index: int,
    episodes_per_chunk: int,
) -> tuple[int, int]:
    """Map an episode index to LeRobot chunk and file indices."""

    return (
        int(episode_index // episodes_per_chunk),
        int(episode_index % episodes_per_chunk),
    )


def estimate_total_size_mb(paths: list[Path]) -> int:
    """Return an integer MB estimate for a collection of files."""

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
    if not np.array_equal(
        np.asarray(records["index"], dtype=np.int64), expected_indices
    ):
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


def validate_task_coverage(task_ids: np.ndarray | list[int]) -> dict[int, int]:
    """Ensure that every target task appears at least once in the dataset."""

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
    """Return a deterministic shuffled order that mixes tasks across the dataset."""

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


def load_demonstrations(input_path: str | Path) -> DemonstrationDataset:
    """Load raw variable-length demonstration trajectories from a saved NPZ."""

    input_path = Path(input_path)
    loaded = np.load(input_path, allow_pickle=False)

    required_keys = {
        "seed",
        "num_per_task_requested",
        "solve_time",
        "retries_per_demo",
        "episode_id",
        "task_id",
        "target_goal_name",
        "start_xy",
        "goal_xy",
        "path_xy",
        "path_length",
        "success",
        "success_counts_by_task",
        "attempt_counts_by_task",
        "skipped_counts_by_task",
    }
    missing_keys = required_keys.difference(loaded.files)
    if missing_keys:
        missing_str = ", ".join(sorted(missing_keys))
        raise ValueError(f"Raw demonstration file is missing keys: {missing_str}")

    task_ids = loaded["task_id"]
    sorted_task_ids = list(TASK_ID_VALUES)
    success_counts = loaded["success_counts_by_task"]
    attempt_counts = loaded["attempt_counts_by_task"]
    skipped_counts = loaded["skipped_counts_by_task"]

    episodes: list[DemonstrationEpisode] = []
    for episode_index in range(task_ids.shape[0]):
        current_length = int(loaded["path_length"][episode_index])
        path_xy = np.asarray(
            loaded["path_xy"][episode_index, :current_length],
            dtype=np.float64,
        )
        episodes.append(
            DemonstrationEpisode(
                episode_id=int(loaded["episode_id"][episode_index]),
                task_id=int(task_ids[episode_index]),
                target_goal_name=str(loaded["target_goal_name"][episode_index]),
                start_xy=tuple(map(float, loaded["start_xy"][episode_index])),
                goal_xy=tuple(map(float, loaded["goal_xy"][episode_index])),
                path_xy=path_xy,
                path_length=current_length,
                success=bool(loaded["success"][episode_index]),
            )
        )

    return DemonstrationDataset(
        episodes=episodes,
        seed=int(np.asarray(loaded["seed"]).item()),
        num_per_task_requested=int(np.asarray(loaded["num_per_task_requested"]).item()),
        success_counts_by_task={
            task_id: int(success_counts[index])
            for index, task_id in enumerate(sorted_task_ids)
        },
        attempt_counts_by_task={
            task_id: int(attempt_counts[index])
            for index, task_id in enumerate(sorted_task_ids)
        },
        skipped_counts_by_task={
            task_id: int(skipped_counts[index])
            for index, task_id in enumerate(sorted_task_ids)
        },
        solve_time=float(np.asarray(loaded["solve_time"]).item()),
        retries_per_demo=int(np.asarray(loaded["retries_per_demo"]).item()),
    )


def process_demonstration_dataset(
    raw_dataset: DemonstrationDataset,
    t_fixed: int = DEFAULT_T_FIXED,
    include_phase_labels: bool = True,
    last_action_mode: str = DEFAULT_LAST_ACTION_MODE,
    config: MapConfig | None = None,
) -> ProcessedDemonstrationDataset:
    """Convert raw variable-length trajectories into fixed-shape training tensors."""

    if len(raw_dataset) == 0:
        raise ValueError("raw_dataset is empty.")
    if t_fixed <= 0:
        raise ValueError("t_fixed must be positive.")

    num_episodes = len(raw_dataset.episodes)
    observations = np.zeros((num_episodes, t_fixed, 2), dtype=np.float32)
    actions = np.zeros((num_episodes, t_fixed, 2), dtype=np.float32)
    task_ids = np.zeros(num_episodes, dtype=np.int64)
    task_code_bits = np.zeros((num_episodes, 2), dtype=np.int64)
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

    map_config = build_default_map_config() if config is None else config
    raw_task_ids = np.asarray(
        [episode.task_id for episode in raw_dataset.episodes],
        dtype=np.int64,
    )
    task_counts = validate_task_coverage(raw_task_ids)
    episode_order = build_balanced_episode_order(raw_task_ids, raw_dataset.seed)
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
        phase_label_vocab=PHASE_LABEL_VOCAB,
        source_num_per_task_requested=raw_dataset.num_per_task_requested,
        source_solve_time=raw_dataset.solve_time,
        source_retries_per_demo=raw_dataset.retries_per_demo,
    )


def save_processed_dataset(
    dataset: ProcessedDemonstrationDataset,
    output_path: str | Path,
) -> Path:
    """Save processed fixed-length tensors to a compressed NPZ."""

    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict[str, Any] = {
        "format_version": np.asarray("braidedhub_act_ready_implicitcue_v2"),
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

    np.savez_compressed(output_path, **save_kwargs)
    print(f"Saved processed dataset with {len(dataset)} samples to {output_path}")
    return output_path


def dataset_summary(dataset: ProcessedDemonstrationDataset) -> dict[str, Any]:
    """Print and return a compact summary of the processed dataset."""

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


def _compress_phase_sequence(phase_names: tuple[str, ...]) -> tuple[str, ...]:
    compressed: list[str] = []
    for phase_name in phase_names:
        if not compressed or compressed[-1] != phase_name:
            compressed.append(phase_name)
    return tuple(compressed)


def _first_phase_index(phase_names: tuple[str, ...], target_phase: str) -> int | None:
    for phase_index, phase_name in enumerate(phase_names):
        if phase_name == target_phase:
            return phase_index
    return None


def plot_processed_sample(
    dataset: ProcessedDemonstrationDataset,
    sample_index: int,
    show: bool = True,
    arrow_stride: int | None = None,
    config: MapConfig | None = None,
):
    """Visualize one processed sample with trajectory, actions, and phase summary."""

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={sample_index} is out of range for {len(dataset)} samples."
        )

    map_config = build_default_map_config() if config is None else config
    fig, ax = plot_map(map_config, show=False)

    observations = dataset.observations[sample_index]
    actions = dataset.actions[sample_index]
    task_id = int(dataset.task_ids[sample_index])
    task_code = "".join(
        str(int(bit)) for bit in dataset.task_code_bits[sample_index].tolist()
    )
    target_goal_name = str(dataset.target_goal_names[sample_index])
    color = TASK_COLOR_BY_ID.get(task_id, "#111111")

    ax.plot(
        observations[:, 0],
        observations[:, 1],
        color=color,
        linewidth=2.2,
        alpha=0.95,
        zorder=6,
        label=f"task {task_id} -> {target_goal_name}",
    )
    ax.scatter(
        observations[0, 0],
        observations[0, 1],
        s=70,
        c=color,
        edgecolors="white",
        linewidths=1.0,
        zorder=7,
        label="resampled start",
    )
    ax.scatter(
        observations[-1, 0],
        observations[-1, 1],
        s=70,
        c="#111111",
        edgecolors="white",
        linewidths=1.0,
        zorder=7,
        label="resampled end",
    )

    current_arrow_stride = (
        max(1, dataset.t_fixed // 20) if arrow_stride is None else arrow_stride
    )
    if current_arrow_stride <= 0:
        raise ValueError("arrow_stride must be positive.")
    arrow_indices = np.arange(0, dataset.t_fixed, current_arrow_stride, dtype=np.int64)
    ax.quiver(
        observations[arrow_indices, 0],
        observations[arrow_indices, 1],
        actions[arrow_indices, 0],
        actions[arrow_indices, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0035,
        color=color,
        alpha=0.55,
        zorder=5,
    )

    if dataset.phase_labels is not None:
        phase_names = decode_phase_labels(dataset.phase_labels[sample_index])
    else:
        phase_annotation = annotate_trajectory_phases(
            [tuple(map(float, point_xy)) for point_xy in observations.tolist()],
            config=map_config,
        )
        phase_names = phase_annotation.phase_labels

    first_h1_index = _first_phase_index(phase_names, "decision_region_H1")
    first_h2_index = _first_phase_index(phase_names, "decision_region_H2")
    first_terminal_index = None
    for terminal_name in TASK_ID_TO_GOAL_NAME.values():
        terminal_index = _first_phase_index(phase_names, terminal_name)
        if terminal_index is None:
            continue
        if first_terminal_index is None or terminal_index < first_terminal_index:
            first_terminal_index = terminal_index

    for event_index, label, marker_color in (
        (first_h1_index, "H1", "#ffcc00"),
        (first_h2_index, "H2", "#00c2ff"),
        (first_terminal_index, "Terminal", "#ff4d6d"),
    ):
        if event_index is None:
            continue
        ax.scatter(
            observations[event_index, 0],
            observations[event_index, 1],
            s=90,
            c=marker_color,
            edgecolors="black",
            linewidths=0.9,
            zorder=8,
        )
        ax.text(
            observations[event_index, 0] + 1.0,
            observations[event_index, 1] + 0.8,
            label,
            fontsize=9,
            color=marker_color,
            zorder=9,
        )

    compressed_phases = _compress_phase_sequence(phase_names)
    summary_lines = [
        f"task_id={task_id}",
        f"implicit_code={task_code}",
        f"target_goal={target_goal_name}",
        f"raw_steps={int(dataset.raw_path_lengths[sample_index])}",
        f"phase_flow={' -> '.join(compressed_phases[:6])}",
    ]
    if len(compressed_phases) > 6:
        summary_lines[-1] += " -> ..."
    ax.text(
        0.02,
        0.02,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#444444",
        },
    )

    ax.set_title("Processed ACT/IL Sample")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    if show:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "matplotlib is required to visualize processed samples."
            ) from exc
        plt.show()

    return fig, ax


def generate_lerobot_v30_dataset(
    processed_dataset: ProcessedDemonstrationDataset | None = None,
    output_dir: str | Path = DEFAULT_LEROBOT_V30_OUTPUT,
    fps: int = DEFAULT_VIDEO_FPS,
    image_size: int = DEFAULT_VIDEO_IMAGE_SIZE,
    processed_input_path: str | Path | None = None,
    config: MapConfig | None = None,
    episodes_per_chunk: int = DEFAULT_LEROBOT_EPISODES_PER_CHUNK,
) -> Path:
    """Export processed demonstrations to a LeRobotDataset v3.0-style layout."""

    pa, pq = _require_lerobot_export_dependencies()

    if fps <= 0:
        raise ValueError("fps must be positive.")
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if episodes_per_chunk <= 0:
        raise ValueError("episodes_per_chunk must be positive.")

    if processed_dataset is None:
        dataset_path = (
            Path(DEFAULT_PROCESSED_OUTPUT)
            if processed_input_path is None
            else Path(processed_input_path)
        )
        processed_dataset = load_processed_dataset(dataset_path)

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
    episodes_meta: list[dict[str, Any]] = []
    global_index = 0
    video_frame_counts: dict[int, int] = {}
    data_files: list[Path] = []
    video_files: list[Path] = []
    first_video_info: dict[str, Any] | None = None

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
            root
            / f"videos/{VIDEO_KEY}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
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

        for frame_idx in range(episode_length):
            state_xy = observations[frame_idx]
            action_xy = actions[frame_idx]
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

            frame = render_lerobot_frame(
                base_image,
                config=map_config,
                robot_xy=(float(state_xy[0]), float(state_xy[1])),
            )
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

    total_frames = len(records["index"])
    state_array = np.asarray(records["observation.state"], dtype=np.float32)
    action_array = np.asarray(records["action"], dtype=np.float32)

    episodes_table = pa.Table.from_arrays(
        [
            pa.array(
                [episode["episode_index"] for episode in episodes_meta], type=pa.int64()
            ),
            pa.array(
                [episode["tasks"] for episode in episodes_meta],
                type=pa.list_(pa.string()),
            ),
            pa.array([episode["length"] for episode in episodes_meta], type=pa.int64()),
            pa.array(
                [episode["data/chunk_index"] for episode in episodes_meta],
                type=pa.int64(),
            ),
            pa.array(
                [episode["data/file_index"] for episode in episodes_meta],
                type=pa.int64(),
            ),
            pa.array(
                [episode["dataset_from_index"] for episode in episodes_meta],
                type=pa.int64(),
            ),
            pa.array(
                [episode["dataset_to_index"] for episode in episodes_meta],
                type=pa.int64(),
            ),
            pa.array(
                [
                    episode[f"videos/{VIDEO_KEY}/chunk_index"]
                    for episode in episodes_meta
                ],
                type=pa.int64(),
            ),
            pa.array(
                [
                    episode[f"videos/{VIDEO_KEY}/file_index"]
                    for episode in episodes_meta
                ],
                type=pa.int64(),
            ),
            pa.array(
                [
                    episode[f"videos/{VIDEO_KEY}/from_timestamp"]
                    for episode in episodes_meta
                ],
                type=pa.float32(),
            ),
            pa.array(
                [
                    episode[f"videos/{VIDEO_KEY}/to_timestamp"]
                    for episode in episodes_meta
                ],
                type=pa.float32(),
            ),
            pa.array(
                [episode["meta/episodes/chunk_index"] for episode in episodes_meta],
                type=pa.int64(),
            ),
            pa.array(
                [episode["meta/episodes/file_index"] for episode in episodes_meta],
                type=pa.int64(),
            ),
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
        raise RuntimeError(
            "No episodes were exported, so no video metadata was created."
        )

    total_episodes = len(processed_dataset)
    val_start = int(round(total_episodes * 0.8))
    splits = {
        "train": f"0:{val_start}",
        "val": f"{val_start}:{total_episodes}",
    }

    info = {
        "codebase_version": "v3.0",
        "robot_type": "point_mass_2d_braidedhub_implicit_cue",
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
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
    }

    stats = {
        "observation.state": build_stats(state_array),
        "action": build_stats(action_array),
        "next.reward": build_stats(
            np.asarray(records["next.reward"], dtype=np.float32)
        ),
        "timestamp": build_stats(np.asarray(records["timestamp"], dtype=np.float32)),
    }

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


def main() -> None:
    raw_output_path = Path(DEFAULT_DATASET_OUTPUT)
    if raw_output_path.exists():
        raw_dataset = load_demonstrations(raw_output_path)
        print(f"Loaded raw demonstrations from {raw_output_path}")
    else:
        raw_dataset = generate_demonstrations(
            num_per_task=25,
            seed=DEFAULT_RANDOM_SEED,
        )
        save_demonstrations(raw_dataset, raw_output_path)

    processed_dataset = process_demonstration_dataset(
        raw_dataset,
        t_fixed=DEFAULT_T_FIXED,
        include_phase_labels=True,
    )
    processed_output_path = save_processed_dataset(
        processed_dataset,
        DEFAULT_PROCESSED_OUTPUT,
    )
    dataset_summary(processed_dataset)
    lerobot_output_path: Path | None = None
    try:
        lerobot_output_path = generate_lerobot_v30_dataset(
            processed_dataset=processed_dataset,
            output_dir=DEFAULT_LEROBOT_V30_OUTPUT,
            fps=DEFAULT_VIDEO_FPS,
            image_size=DEFAULT_VIDEO_IMAGE_SIZE,
        )
    except RuntimeError as exc:
        print(f"LeRobot export skipped: {exc}")

    try:
        plot_processed_sample(
            processed_dataset,
            sample_index=DEFAULT_PROCESSED_SAMPLE_INDEX,
            show=True,
        )
    except RuntimeError as exc:
        print(f"Visualization skipped: {exc}")

    print(f"Processed dataset ready at {processed_output_path}")
    if lerobot_output_path is not None:
        print(f"LeRobot v3.0 dataset ready at {lerobot_output_path}")


if __name__ == "__main__":
    main()
