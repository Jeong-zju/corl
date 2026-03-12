import argparse
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "pandas is required to write meta/tasks.parquet. Run with: "
        "uv run --with numpy --with pyarrow --with pandas main/scripts/rrt_connect_h_env.py"
    ) from exc


VIDEO_KEY = "observation.images.front"
TASKS = [
    "Navigate through the upper bridge of the H-maze from left to right.",
    "Navigate through the lower bridge of the H-maze from left to right.",
]


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
):
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

    episodes_meta = []
    global_index = 0
    cumulative_video_frames = 0
    corners = find_fixed_h_corners(grid, extent)

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

            frame = render_frame(base_img, extent, pos)
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

    def fixed_size_list(values, width):
        flat = pa.array(values.reshape(-1), type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, width)

    data_table = pa.Table.from_arrays(
        [
            fixed_size_list(state_arr, 2),
            fixed_size_list(action_arr, 2),
            pa.array(records["next.reward"], type=pa.float32()),
            pa.array(records["next.done"], type=pa.bool_()),
            pa.array(records["next.success"], type=pa.bool_()),
            pa.array(records["timestamp"], type=pa.float32()),
            pa.array(records["frame_index"], type=pa.int64()),
            pa.array(records["episode_index"], type=pa.int64()),
            pa.array(records["index"], type=pa.int64()),
            pa.array(records["task_index"], type=pa.int64()),
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

    stats = {
        "observation.state": build_stats(state_arr),
        "action": build_stats(action_arr),
        "next.reward": build_stats(np.asarray(records["next.reward"], dtype=np.float32)),
        "timestamp": build_stats(np.asarray(records["timestamp"], dtype=np.float32)),
    }

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an H-maze synthetic LeRobotDataset v3.0")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="data/zeno-ai/rrt_connect_h_v30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    generate_lerobot_v30_dataset(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
        fps=args.fps,
        image_size=args.image_size,
    )
