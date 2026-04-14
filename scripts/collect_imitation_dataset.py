import argparse
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np

from env import get_env_choices, get_env_module
from eval_helpers import (
    compute_delta_signature_sequence_np,
    compute_signatory_signature_np,
    compute_simple_signature_np,
    resolve_signature_backend,
)


VIDEO_KEY = "observation.images.front"
DEFAULT_PATH_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_DELTA_SIGNATURE_KEY = "observation.delta_signature"
DEFAULT_INCLUDE_PATH_SIGNATURES = True
DEFAULT_SIGNATURE_WINDOW_SIZE = 0
DEFAULT_SIGNATURE_DEPTH = 3
DEFAULT_SIGNATURE_BACKEND = "auto"
TASKS = [
    "Navigate through the upper bridge of the H-maze from left to right.",
    "Navigate through the lower bridge of the H-maze from left to right.",
]


def _require_h_shape_export_dependencies():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "h_shape dataset export requires `pyarrow` and `pandas`. "
            "Install them first before exporting LeRobot files."
        ) from exc
    return pa, pq, pd


def create_h_shape_grid(
    grid_size=(100, 100),
    total_width=6.0,
    total_height=8.0,
    bar_thickness=1.5,
    mid_bar_height=2.0,
    wall_thickness=0.3,
):
    rows, cols = grid_size
    hw = total_width / 2
    hh = total_height / 2
    bt = bar_thickness
    mh = mid_bar_height / 2
    wt = wall_thickness

    xmin, xmax = -hw - 1.0, hw + 1.0
    ymin, ymax = -hh - 1.0, hh + 1.0

    xs = np.linspace(xmin, xmax, cols)
    ys = np.linspace(ymin, ymax, rows)
    X, Y = np.meshgrid(xs, ys)

    left_inner = (
        (X > -hw + wt)
        & (X < -hw + bt - wt)
        & (Y > -hh + wt)
        & (Y < hh - wt)
    )
    right_inner = (
        (X > hw - bt + wt)
        & (X < hw - wt)
        & (Y > -hh + wt)
        & (Y < hh - wt)
    )
    mid_inner = (
        (X > -hw + bt - wt)
        & (X < hw - bt + wt)
        & (Y > -mh + wt)
        & (Y < mh - wt)
    )

    free = left_inner | right_inner | mid_inner
    grid = np.ones((rows, cols), dtype=np.float64)
    grid[free] = 0.0
    extent = (xmin, xmax, ymin, ymax)
    return grid, extent


def distance(p1, p2):
    return float(np.linalg.norm(np.subtract(p1, p2)))


def world_to_grid(point, extent, shape):
    xmin, xmax, ymin, ymax = extent
    rows, cols = shape
    x, y = point
    col = int(round((x - xmin) / (xmax - xmin) * (cols - 1)))
    row = int(round((y - ymin) / (ymax - ymin) * (rows - 1)))
    row = int(np.clip(row, 0, rows - 1))
    col = int(np.clip(col, 0, cols - 1))
    return row, col


def grid_to_world(row, col, extent, shape):
    xmin, xmax, ymin, ymax = extent
    rows, cols = shape
    x = np.interp(col, [0, cols - 1], [xmin, xmax])
    y = np.interp(row, [0, rows - 1], [ymin, ymax])
    return (float(x), float(y))


def is_collision_free(point, grid, extent):
    row, col = world_to_grid(point, extent, grid.shape)
    return bool(grid[row, col] == 0)


def steer(from_pt, to_pt, max_dist=0.5):
    vec = np.subtract(to_pt, from_pt)
    dist = np.linalg.norm(vec)
    if dist <= max_dist:
        return (float(to_pt[0]), float(to_pt[1]))
    scale = max_dist / dist
    new_pt = np.add(from_pt, vec * scale)
    return (float(new_pt[0]), float(new_pt[1]))


def rrt_connect(start, goal, grid, extent, rng, max_iters=5000, step_size=0.35, connect_dist=0.35):
    class Node:
        __slots__ = ("pt", "parent")

        def __init__(self, pt, parent=None):
            self.pt = pt
            self.parent = parent

    tree_a = [Node(start)]
    tree_b = [Node(goal)]

    def nearest(tree, pt):
        dists = [distance(node.pt, pt) for node in tree]
        return tree[int(np.argmin(dists))]

    def path_to_root(node):
        path = []
        while node is not None:
            path.append(node.pt)
            node = node.parent
        return list(reversed(path))

    for _ in range(max_iters):
        while True:
            rand_pt = (
                rng.uniform(extent[0], extent[1]),
                rng.uniform(extent[2], extent[3]),
            )
            if is_collision_free(rand_pt, grid, extent):
                break

        near_a = nearest(tree_a, rand_pt)
        new_pt_a = steer(near_a.pt, rand_pt, max_dist=step_size)
        if is_collision_free(new_pt_a, grid, extent):
            new_node_a = Node(new_pt_a, parent=near_a)
            tree_a.append(new_node_a)
            near_b = nearest(tree_b, new_pt_a)
            new_pt_b = steer(near_b.pt, new_pt_a, max_dist=step_size)

            while is_collision_free(new_pt_b, grid, extent):
                new_node_b = Node(new_pt_b, parent=near_b)
                tree_b.append(new_node_b)
                if distance(new_pt_b, new_pt_a) < connect_dist:
                    path_a = path_to_root(new_node_a)
                    path_b = path_to_root(new_node_b)
                    path_b.reverse()
                    return path_a + path_b[1:]
                near_b = new_node_b
                new_pt_b = steer(near_b.pt, new_pt_a, max_dist=step_size)

        tree_a, tree_b = tree_b, tree_a

    return None


def find_fixed_h_corners(grid, extent):
    rows, cols = grid.shape
    free = np.argwhere(grid == 0)
    if free.shape[0] == 0:
        raise RuntimeError("No free cells found in the H-map.")

    ul = ur = ll = lr = None
    ul_score = ur_score = ll_score = lr_score = None

    for row, col in free:
        x, y = grid_to_world(int(row), int(col), extent, grid.shape)
        if x < 0.0 and y > 0.0:
            score = (int(col), -int(row))
            if ul_score is None or score < ul_score:
                ul = (int(row), int(col))
                ul_score = score
        if x > 0.0 and y > 0.0:
            score = (-int(col), -int(row))
            if ur_score is None or score < ur_score:
                ur = (int(row), int(col))
                ur_score = score
        if x < 0.0 and y < 0.0:
            score = (int(col), int(row))
            if ll_score is None or score < ll_score:
                ll = (int(row), int(col))
                ll_score = score
        if x > 0.0 and y < 0.0:
            score = (-int(col), int(row))
            if lr_score is None or score < lr_score:
                lr = (int(row), int(col))
                lr_score = score

    if any(v is None for v in (ul, ur, ll, lr)):
        raise RuntimeError("Could not locate all four deterministic H corners.")

    return {
        "upper_left": grid_to_world(*ul, extent, grid.shape),
        "upper_right": grid_to_world(*ur, extent, grid.shape),
        "lower_left": grid_to_world(*ll, extent, grid.shape),
        "lower_right": grid_to_world(*lr, extent, grid.shape),
    }


def build_corner_to_corner_path(start, goal, connector_y=0.0):
    return [
        (float(start[0]), float(start[1])),
        (float(start[0]), float(connector_y)),
        (float(goal[0]), float(connector_y)),
        (float(goal[0]), float(goal[1])),
    ]


def densify_path(path, step=0.12, min_points=12, max_points=42):
    if path is None or len(path) < 2:
        return []

    dense = [path[0]]
    for a, b in zip(path[:-1], path[1:], strict=False):
        seg = np.subtract(b, a)
        seg_len = float(np.linalg.norm(seg))
        n = max(1, int(math.ceil(seg_len / step)))
        for i in range(1, n + 1):
            t = i / n
            p = (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)
            dense.append((float(p[0]), float(p[1])))

    if len(dense) > max_points:
        keep = np.linspace(0, len(dense) - 1, max_points).astype(int)
        dense = [dense[i] for i in keep]

    while len(dense) < min_points:
        dense.append(dense[-1])

    return dense


def make_base_image(grid, size):
    rows, cols = grid.shape
    h, w = size
    row_idx = np.linspace(0, rows - 1, h).astype(int)
    col_idx = np.linspace(0, cols - 1, w).astype(int)
    sampled = grid[np.ix_(row_idx, col_idx)]

    free_color = np.uint8(240)
    obs_color = np.uint8(55)
    gray = np.where(sampled == 0, free_color, obs_color).astype(np.uint8)
    frame = np.stack([gray, gray, gray], axis=-1)
    return frame


def world_to_pixel(point, extent, size):
    xmin, xmax, ymin, ymax = extent
    h, w = size
    x, y = point
    px = int(round((x - xmin) / (xmax - xmin) * (w - 1)))
    py = int(round((y - ymin) / (ymax - ymin) * (h - 1)))
    px = int(np.clip(px, 0, w - 1))
    py = int(np.clip(py, 0, h - 1))
    return px, py


def draw_disk(img, center, radius, color):
    cx, cy = center
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius
    img[mask] = np.array(color, dtype=np.uint8)


def render_frame(base_img, extent, agent_pos):
    frame = base_img.copy()
    h, w = frame.shape[:2]
    r_agent = max(2, int(min(h, w) * 0.025))

    agent_px = world_to_pixel(agent_pos, extent, (h, w))

    draw_disk(frame, agent_px, r_agent, (240, 70, 70))

    return frame


def start_ffmpeg_raw_writer(output_path, width, height, fps):
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


def ffprobe_video(video_path):
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
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    stream = payload["streams"][0]

    rate = stream.get("avg_frame_rate", "0/1")
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / max(float(den), 1.0)
    else:
        fps = float(rate)

    nb_frames = stream.get("nb_frames")
    if nb_frames in (None, "N/A"):
        duration = float(stream.get("duration", 0.0))
        nb_frames = int(round(duration * fps))
    else:
        nb_frames = int(nb_frames)

    return {
        "codec": stream.get("codec_name", "unknown"),
        "pix_fmt": stream.get("pix_fmt", "unknown"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": int(round(fps)),
        "frames": nb_frames,
    }


def build_stats(values):
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


def init_image_stats_accumulator(num_channels):
    if num_channels <= 0:
        raise ValueError(f"`num_channels` must be positive, got {num_channels}.")
    channel_shape = (num_channels, 1, 1)
    return {
        "sum": np.zeros(channel_shape, dtype=np.float64),
        "sumsq": np.zeros(channel_shape, dtype=np.float64),
        "min": np.full(channel_shape, np.inf, dtype=np.float64),
        "max": np.full(channel_shape, -np.inf, dtype=np.float64),
        "pixel_count": 0,
        "frame_count": 0,
    }


def update_image_stats_accumulator(accumulator, frame_hwc_uint8):
    frame = np.asarray(frame_hwc_uint8, dtype=np.float32)
    if frame.ndim != 3 or frame.shape[2] <= 0:
        raise ValueError(
            "Expected an HWC image array with a positive channel dimension. "
            f"Got shape={tuple(frame.shape)}."
        )
    chw = np.transpose(frame / 255.0, (2, 0, 1))
    channel_min = chw.min(axis=(1, 2), keepdims=True).astype(np.float64)
    channel_max = chw.max(axis=(1, 2), keepdims=True).astype(np.float64)
    accumulator["sum"] += chw.sum(axis=(1, 2), keepdims=True, dtype=np.float64)
    accumulator["sumsq"] += np.square(chw, dtype=np.float64).sum(
        axis=(1, 2), keepdims=True, dtype=np.float64
    )
    accumulator["min"] = np.minimum(accumulator["min"], channel_min)
    accumulator["max"] = np.maximum(accumulator["max"], channel_max)
    accumulator["pixel_count"] += int(chw.shape[1] * chw.shape[2])
    accumulator["frame_count"] += 1


def finalize_image_stats(accumulator):
    pixel_count = int(accumulator["pixel_count"])
    frame_count = int(accumulator["frame_count"])
    if pixel_count <= 0 or frame_count <= 0:
        raise ValueError("Image stats accumulator is empty.")
    mean = accumulator["sum"] / pixel_count
    variance = accumulator["sumsq"] / pixel_count - np.square(mean, dtype=np.float64)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance, dtype=np.float64)
    return {
        "min": accumulator["min"].tolist(),
        "max": accumulator["max"].tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": [frame_count],
    }


def compute_path_signature_sequence(
    states,
    *,
    window_size,
    sig_depth,
    signature_backend,
):
    states_array = np.asarray(states, dtype=np.float32)
    if states_array.ndim != 2:
        raise ValueError(
            "Expected state trajectory with shape (T, state_dim). "
            f"Got {states_array.shape}."
        )
    if states_array.shape[0] == 0:
        raise ValueError("State trajectory must contain at least one step.")
    if sig_depth <= 0:
        raise ValueError(f"`sig_depth` must be positive, got {sig_depth}.")

    signatures = []
    signature_dim = None
    for end_idx in range(int(states_array.shape[0])):
        start_idx = 0 if window_size <= 0 else max(0, end_idx + 1 - int(window_size))
        window = states_array[start_idx : end_idx + 1]
        if signature_backend == "signatory":
            signature = compute_signatory_signature_np(window, sig_depth)
        else:
            signature = compute_simple_signature_np(window, sig_depth)
        if signature_dim is None:
            signature_dim = int(signature.shape[0])
        elif int(signature.shape[0]) != signature_dim:
            raise RuntimeError(
                "Path-signature dimension changed across timesteps. "
                f"Expected {signature_dim}, got {int(signature.shape[0])}."
            )
        signatures.append(signature.astype(np.float32, copy=False))
    return np.stack(signatures, axis=0)


def validate_consistency(records, episodes_meta, splits, total_frames, total_episodes, video_frame_count):
    if total_frames != len(records["index"]):
        raise ValueError("total_frames mismatch with frame table")

    expected_indices = np.arange(total_frames, dtype=np.int64)
    if not np.array_equal(np.asarray(records["index"], dtype=np.int64), expected_indices):
        raise ValueError("global index must be continuous and monotonic")

    if video_frame_count != total_frames:
        raise ValueError(
            f"video frame count mismatch: video={video_frame_count}, parquet={total_frames}"
        )

    lengths_sum = sum(ep["length"] for ep in episodes_meta)
    if lengths_sum != total_frames:
        raise ValueError("sum(episode.length) must equal total_frames")

    for ep in episodes_meta:
        ep_idx = ep["episode_index"]
        from_i = ep["dataset_from_index"]
        to_i = ep["dataset_to_index"]
        if not (0 <= from_i < to_i <= total_frames):
            raise ValueError(f"invalid dataset index range in episode {ep_idx}")
        if (to_i - from_i) != ep["length"]:
            raise ValueError(f"episode length mismatch in episode {ep_idx}")

    for split_name, split_spec in splits.items():
        start_s, end_s = split_spec.split(":", 1)
        start_i, end_i = int(start_s), int(end_s)
        if not (0 <= start_i <= end_i <= total_episodes):
            raise ValueError(f"invalid split range for {split_name}: {split_spec}")


def generate_lerobot_v30_dataset(
    num_episodes=100,
    output_dir="data/zeno-ai/rrt_connect_h_v30",
    seed=42,
    fps=20,
    image_size=128,
    include_path_signatures=DEFAULT_INCLUDE_PATH_SIGNATURES,
    include_delta_signatures=False,
    path_signature_key=DEFAULT_PATH_SIGNATURE_KEY,
    delta_signature_key=DEFAULT_DELTA_SIGNATURE_KEY,
    path_signature_window_size=DEFAULT_SIGNATURE_WINDOW_SIZE,
    path_signature_depth=DEFAULT_SIGNATURE_DEPTH,
    path_signature_backend=DEFAULT_SIGNATURE_BACKEND,
):
    pa, pq, pd = _require_h_shape_export_dependencies()
    rng = random.Random(seed)
    np.random.seed(seed)

    grid, extent = create_h_shape_grid(
        grid_size=(200, 200),
        total_width=6.0,
        total_height=8.0,
        bar_thickness=1.5,
        mid_bar_height=1.0,
        wall_thickness=0.3,
    )

    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)

    data_file = root / "data/chunk-000/file-000.parquet"
    video_file = root / f"videos/{VIDEO_KEY}/chunk-000/file-000.mp4"
    episodes_file = root / "meta/episodes/chunk-000/file-000.parquet"
    info_file = root / "meta/info.json"
    stats_file = root / "meta/stats.json"
    tasks_jsonl_file = root / "meta/tasks.jsonl"
    tasks_parquet_file = root / "meta/tasks.parquet"

    data_file.parent.mkdir(parents=True, exist_ok=True)
    video_file.parent.mkdir(parents=True, exist_ok=True)
    episodes_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_jsonl_file.parent.mkdir(parents=True, exist_ok=True)

    base_img = make_base_image(grid, (image_size, image_size))
    ffmpeg_proc = start_ffmpeg_raw_writer(video_file, image_size, image_size, fps)

    records = {
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
    if include_path_signatures:
        records[path_signature_key] = []
    if include_delta_signatures:
        records[delta_signature_key] = []

    episodes_meta = []
    global_index = 0
    cumulative_video_frames = 0
    corners = find_fixed_h_corners(grid, extent)
    image_stats_accumulator = init_image_stats_accumulator(num_channels=3)
    resolved_signature_backend = None
    if include_path_signatures:
        resolved_signature_backend = resolve_signature_backend(path_signature_backend)
        window_label = (
            "full_prefix"
            if int(path_signature_window_size) <= 0
            else str(int(path_signature_window_size))
        )
        print(
            "Collecting h_shape path signatures: "
            f"key={path_signature_key}, window={window_label}, "
            f"depth={path_signature_depth}, backend={resolved_signature_backend}"
        )

    for ep_idx in range(num_episodes):
        use_upper = (ep_idx % 2 == 0)
        if use_upper:
            start = corners["upper_left"]
            goal = corners["upper_right"]
        else:
            start = corners["lower_left"]
            goal = corners["lower_right"]

        path = None
        for _ in range(10):
            path = rrt_connect(start, goal, grid, extent, rng=rng)
            if path is not None and len(path) >= 2:
                break
        if path is None or len(path) < 2:
            # Fallback for rare planning failure: keep fixed endpoints.
            path = build_corner_to_corner_path(start, goal, connector_y=0.0)
        elif distance(path[0], start) > distance(path[-1], start):
            path = list(reversed(path))

        trajectory = densify_path(path, step=0.12, min_points=12, max_points=42)
        signature_sequence = None
        delta_signature_sequence = None
        if include_path_signatures:
            signature_sequence = compute_path_signature_sequence(
                trajectory,
                window_size=path_signature_window_size,
                sig_depth=path_signature_depth,
                signature_backend=str(resolved_signature_backend),
            )
            if include_delta_signatures:
                delta_signature_sequence = compute_delta_signature_sequence_np(signature_sequence)
        ep_len = len(trajectory)
        task_index = 0 if use_upper else 1

        ep_from_idx = global_index
        video_from_ts = cumulative_video_frames / fps

        for frame_idx, pos in enumerate(trajectory):
            ts = frame_idx / fps
            nxt = trajectory[min(frame_idx + 1, ep_len - 1)]
            act = [float(nxt[0] - pos[0]), float(nxt[1] - pos[1])]
            # Do not expose goal as privileged state; policy should infer from visual observation.
            obs = [float(pos[0]), float(pos[1])]
            dist_to_goal = distance(pos, goal)
            done = frame_idx == ep_len - 1
            success = done and (dist_to_goal < 0.2)
            reward = 1.0 if success else -dist_to_goal

            records["timestamp"].append(float(ts))
            records["frame_index"].append(int(frame_idx))
            records["episode_index"].append(int(ep_idx))
            records["index"].append(int(global_index))
            records["task_index"].append(int(task_index))
            records["observation.state"].append(obs)
            records["action"].append(act)
            records["next.reward"].append(float(reward))
            records["next.done"].append(bool(done))
            records["next.success"].append(bool(success))
            if signature_sequence is not None:
                records[path_signature_key].append(
                    signature_sequence[frame_idx].astype(np.float32).tolist()
                )
            if delta_signature_sequence is not None:
                records[delta_signature_key].append(
                    delta_signature_sequence[frame_idx].astype(np.float32).tolist()
                )

            frame = render_frame(base_img, extent, pos)
            update_image_stats_accumulator(image_stats_accumulator, frame)
            ffmpeg_proc.stdin.write(frame.astype(np.uint8).tobytes())

            global_index += 1
            cumulative_video_frames += 1

        ep_to_idx = global_index
        video_to_ts = cumulative_video_frames / fps

        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "tasks": [TASKS[task_index]],
                "length": ep_len,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": ep_from_idx,
                "dataset_to_index": ep_to_idx,
                f"videos/{VIDEO_KEY}/chunk_index": 0,
                f"videos/{VIDEO_KEY}/file_index": 0,
                f"videos/{VIDEO_KEY}/from_timestamp": float(video_from_ts),
                f"videos/{VIDEO_KEY}/to_timestamp": float(video_to_ts),
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )

    ffmpeg_proc.stdin.close()
    ret = ffmpeg_proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with code {ret}")

    total_frames = len(records["index"])

    state_arr = np.asarray(records["observation.state"], dtype=np.float32)
    action_arr = np.asarray(records["action"], dtype=np.float32)
    signature_arr = (
        None
        if path_signature_key not in records
        else np.asarray(records[path_signature_key], dtype=np.float32)
    )
    delta_signature_arr = (
        None
        if delta_signature_key not in records
        else np.asarray(records[delta_signature_key], dtype=np.float32)
    )

    def fixed_size_list(values, width):
        flat = pa.array(values.reshape(-1), type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, width)

    data_arrays = [fixed_size_list(state_arr, 2)]
    data_names = ["observation.state"]
    if signature_arr is not None:
        data_arrays.append(fixed_size_list(signature_arr, int(signature_arr.shape[1])))
        data_names.append(path_signature_key)
    if delta_signature_arr is not None:
        data_arrays.append(
            fixed_size_list(delta_signature_arr, int(delta_signature_arr.shape[1]))
        )
        data_names.append(delta_signature_key)
    data_arrays.extend(
        [
            fixed_size_list(action_arr, 2),
            pa.array(records["next.reward"], type=pa.float32()),
            pa.array(records["next.done"], type=pa.bool_()),
            pa.array(records["next.success"], type=pa.bool_()),
            pa.array(records["timestamp"], type=pa.float32()),
            pa.array(records["frame_index"], type=pa.int64()),
            pa.array(records["episode_index"], type=pa.int64()),
            pa.array(records["index"], type=pa.int64()),
            pa.array(records["task_index"], type=pa.int64()),
        ]
    )
    data_names.extend(
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
    data_table = pa.Table.from_arrays(data_arrays, names=data_names)
    pq.write_table(data_table, data_file, compression="snappy")

    episodes_table = pa.Table.from_arrays(
        [
            pa.array([ep["episode_index"] for ep in episodes_meta], type=pa.int64()),
            pa.array([ep["tasks"] for ep in episodes_meta], type=pa.list_(pa.string())),
            pa.array([ep["length"] for ep in episodes_meta], type=pa.int64()),
            pa.array([ep["data/chunk_index"] for ep in episodes_meta], type=pa.int64()),
            pa.array([ep["data/file_index"] for ep in episodes_meta], type=pa.int64()),
            pa.array([ep["dataset_from_index"] for ep in episodes_meta], type=pa.int64()),
            pa.array([ep["dataset_to_index"] for ep in episodes_meta], type=pa.int64()),
            pa.array([ep[f"videos/{VIDEO_KEY}/chunk_index"] for ep in episodes_meta], type=pa.int64()),
            pa.array([ep[f"videos/{VIDEO_KEY}/file_index"] for ep in episodes_meta], type=pa.int64()),
            pa.array([ep[f"videos/{VIDEO_KEY}/from_timestamp"] for ep in episodes_meta], type=pa.float32()),
            pa.array([ep[f"videos/{VIDEO_KEY}/to_timestamp"] for ep in episodes_meta], type=pa.float32()),
            pa.array([ep["meta/episodes/chunk_index"] for ep in episodes_meta], type=pa.int64()),
            pa.array([ep["meta/episodes/file_index"] for ep in episodes_meta], type=pa.int64()),
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

    with tasks_jsonl_file.open("w", encoding="utf-8") as f:
        for task_idx, task in enumerate(TASKS):
            f.write(json.dumps({"task_index": task_idx, "task": task}, ensure_ascii=False) + "\n")

    tasks_df = pd.DataFrame({"task_index": [0, 1]}, index=TASKS)
    tasks_df.to_parquet(tasks_parquet_file)

    video_info = ffprobe_video(video_file)

    total_tasks = len(TASKS)
    val_start = int(round(num_episodes * 0.8))
    splits = {
        "train": f"0:{val_start}",
        "val": f"{val_start}:{num_episodes}",
    }

    info = {
        "codebase_version": "v3.0",
        "robot_type": "point_mass_2d_rrt",
        "total_episodes": int(num_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
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
                "shape": [video_info["height"], video_info["width"], 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": video_info["height"],
                    "video.width": video_info["width"],
                    "video.codec": video_info["codec"],
                    "video.pix_fmt": video_info["pix_fmt"],
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
    if signature_arr is not None:
        info["features"][path_signature_key] = {
            "dtype": "float32",
            "shape": [int(signature_arr.shape[1])],
            "names": [f"path_sig_{index}" for index in range(int(signature_arr.shape[1]))],
        }
        info["path_signature"] = {
            "key": path_signature_key,
            "window_size": int(path_signature_window_size),
            "window_mode": (
                "full_prefix"
                if int(path_signature_window_size) <= 0
                else "sliding_window"
            ),
            "sig_depth": int(path_signature_depth),
            "signature_dim": int(signature_arr.shape[1]),
            "kind": (
                "signature"
                if resolved_signature_backend == "signatory"
                else "simple_signature"
            ),
            "backend": resolved_signature_backend,
        }
    if delta_signature_arr is not None:
        info["features"][delta_signature_key] = {
            "dtype": "float32",
            "shape": [int(delta_signature_arr.shape[1])],
            "names": [f"delta_path_sig_{index}" for index in range(int(delta_signature_arr.shape[1]))],
        }
        info["delta_signature"] = {
            "key": delta_signature_key,
            "signature_key": path_signature_key,
            "definition": "path_signature_t - path_signature_{t-1}",
            "first_step_rule": "zeros",
            "signature_dim": int(delta_signature_arr.shape[1]),
        }

    stats = {
        "observation.state": build_stats(state_arr),
        "action": build_stats(action_arr),
        "next.reward": build_stats(np.asarray(records["next.reward"], dtype=np.float32)),
        "timestamp": build_stats(np.asarray(records["timestamp"], dtype=np.float32)),
        VIDEO_KEY: finalize_image_stats(image_stats_accumulator),
    }
    if signature_arr is not None:
        stats[path_signature_key] = build_stats(signature_arr)
    if delta_signature_arr is not None:
        stats[delta_signature_key] = build_stats(delta_signature_arr)

    with info_file.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    validate_consistency(
        records=records,
        episodes_meta=episodes_meta,
        splits=splits,
        total_frames=total_frames,
        total_episodes=num_episodes,
        video_frame_count=video_info["frames"],
    )

    print(f"Generated LeRobotDataset v3.0 at: {root.resolve()}")
    print(f"Episodes: {num_episodes}, Frames: {total_frames}, Video frames: {video_info['frames']}")
    return root


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env", choices=get_env_choices(), default="h_shape")
    known_args, _ = bootstrap.parse_known_args(argv)
    env_module = get_env_module(known_args.env)
    defaults = env_module.get_dataset_defaults()

    parser = argparse.ArgumentParser(
        description="Collect imitation datasets for supported environments."
    )
    parser.add_argument("--env", choices=get_env_choices(), default=known_args.env)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--fps", type=int, default=defaults["fps"])
    parser.add_argument("--image-size", type=int, default=defaults["image_size"])
    if known_args.env == "h_shape":
        parser.add_argument(
            "--num-episodes",
            type=int,
            default=defaults["num_episodes"],
            help="Total number of synthetic H-shape episodes to export.",
        )
    else:
        import planner_utils

        solve_time_default = getattr(
            env_module,
            "DEFAULT_SOLVE_TIME",
            planner_utils.DEFAULT_SOLVE_TIME,
        )
        retries_default = getattr(
            env_module,
            "DEFAULT_RETRIES_PER_DEMO",
            planner_utils.DEFAULT_RETRIES_PER_DEMO,
        )
        low_success_warning_default = getattr(
            env_module,
            "DEFAULT_LOW_SUCCESS_WARNING_THRESHOLD",
            planner_utils.DEFAULT_LOW_SUCCESS_WARNING_THRESHOLD,
        )
        parser.add_argument(
            "--num-per-task",
            type=int,
            default=defaults["num_per_task"],
            help="Number of successful demonstrations to keep for each task.",
        )
        parser.add_argument(
            "--solve-time",
            type=float,
            default=solve_time_default,
            help="Planner budget per RRTConnect segment.",
        )
        parser.add_argument(
            "--max-retries-per-demo",
            type=int,
            default=retries_default,
            help="Maximum retries before skipping one demonstration.",
        )
        parser.add_argument(
            "--low-success-warning-threshold",
            type=float,
            default=low_success_warning_default,
        )
        parser.add_argument(
            "--enable-randomize",
            action="store_true",
            help=(
                "Randomize the reset start state within each task's "
                "start region. Default behavior uses the region center."
            ),
        )
        parser.add_argument(
            "--t-fixed",
            type=int,
            default=defaults["t_fixed"],
            help=(
                "Fixed resampled horizon used before LeRobot export. "
                "Use 0 to auto-resolve the smallest common collision-free horizon."
            ),
        )
        parser.add_argument(
            "--episodes-per-chunk",
            type=int,
            default=defaults["episodes_per_chunk"],
        )
        parser.add_argument(
            "--raw-output",
            type=Path,
            default=defaults["raw_output"],
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--processed-output",
            type=Path,
            default=defaults["processed_output"],
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--disable-phase-labels",
            action="store_true",
            help="Skip writing phase labels into the processed dataset.",
        )
        parser.add_argument(
            "--skip-lerobot-export",
            action="store_true",
            help="Process demonstrations in memory but skip LeRobot dataset export.",
        )

    parser.add_argument(
        "--disable-path-signature",
        action="store_true",
        help="Skip path-signature preprocessing.",
    )
    delta_signature_group = parser.add_mutually_exclusive_group()
    delta_signature_group.add_argument(
        "--enable-delta-signature",
        dest="enable_delta_signature",
        action="store_true",
        help=(
            "Export observation.delta_signature defined as "
            "path_signature_t - path_signature_{t-1} with a zero first step."
        ),
    )
    delta_signature_group.add_argument(
        "--disable-delta-signature",
        dest="enable_delta_signature",
        action="store_false",
        help="Do not export observation.delta_signature.",
    )
    parser.set_defaults(enable_delta_signature=False)
    parser.add_argument(
        "--path-signature-window-size",
        type=int,
        default=getattr(env_module, "DEFAULT_SIGNATURE_WINDOW_SIZE", 0),
        help="0 means full-prefix signatures; positive values use sliding windows.",
    )
    parser.add_argument(
        "--path-signature-depth",
        type=int,
        default=getattr(env_module, "DEFAULT_SIGNATURE_DEPTH", DEFAULT_SIGNATURE_DEPTH),
    )
    parser.add_argument(
        "--signature-backend",
        type=str,
        default=getattr(env_module, "DEFAULT_SIGNATURE_BACKEND", DEFAULT_SIGNATURE_BACKEND),
        choices=["auto", "signatory", "simple"],
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(argv)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.enable_delta_signature and args.disable_path_signature:
        raise ValueError(
            "`--enable-delta-signature` requires path signatures. "
            "Remove `--disable-path-signature`."
        )
    if args.env == "h_shape":
        output_path = generate_lerobot_v30_dataset(
            num_episodes=args.num_episodes,
            output_dir=args.output_dir,
            seed=args.seed,
            fps=args.fps,
            image_size=args.image_size,
            include_path_signatures=not args.disable_path_signature,
            include_delta_signatures=bool(args.enable_delta_signature),
            path_signature_key=DEFAULT_PATH_SIGNATURE_KEY,
            delta_signature_key=DEFAULT_DELTA_SIGNATURE_KEY,
            path_signature_window_size=args.path_signature_window_size,
            path_signature_depth=args.path_signature_depth,
            path_signature_backend=args.signature_backend,
        )
    else:
        env_module = get_env_module(args.env)
        output_path = env_module.collect_dataset(args)
    print(f"Dataset collection complete: {output_path}")


if __name__ == "__main__":
    main()
