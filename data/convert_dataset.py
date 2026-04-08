#!/usr/bin/env python3
"""Convert non-LeRobot datasets into local LeRobot v3.0 datasets.

Examples:
    python3 main/data/convert_dataset.py \
        /home/jeong/zeno/corl/RMBench/data \
        --convert rmbench \
        --dataset-id rmbench
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import queue
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"
DEFAULT_FPS = 50
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_WORKERS = "auto"
DEFAULT_VIDEO_ENCODER = "auto"
DEFAULT_GPU_VIDEO_ENCODERS = ("h264_nvenc", "hevc_nvenc")
DEFAULT_CPU_VIDEO_ENCODER = "libx264"
SUPPORTED_CONVERTERS = ("rmbench",)
RMbench_RAW_CAMERA_TO_LEROBOT = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
MOTOR_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]
EPISODE_FILE_RE = re.compile(r"episode(\d+)$")


@dataclass(frozen=True)
class EpisodeSource:
    source_task: str
    episode_id: int
    hdf5_path: Path
    instruction_path: Path
    primary_task: str
    task_candidates: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeConversionResult:
    episode_index: int
    source_task: str
    source_episode_id: int
    primary_task: str
    task_candidates: tuple[str, ...]
    frame_count: int
    state_sequence: np.ndarray
    action_sequence: np.ndarray
    video_paths: dict[str, str]
    image_stats_by_key: dict[str, dict[str, Any]]


def maybe_import_tqdm():
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm


def require_binary(binary_name: str) -> None:
    if shutil.which(binary_name) is None:
        raise RuntimeError(
            f"Required binary `{binary_name}` was not found in PATH. "
            "Install it before running dataset conversion."
        )


def require_rmbench_dependencies():
    missing: list[str] = []
    imported: dict[str, Any] = {}

    try:
        import cv2  # type: ignore
    except Exception:
        missing.append("opencv-python")
    else:
        imported["cv2"] = cv2

    try:
        import h5py  # type: ignore
    except Exception:
        missing.append("h5py")
    else:
        imported["h5py"] = h5py

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        missing.append("pyarrow")
    else:
        imported["pa"] = pa
        imported["pq"] = pq

    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            "RMBench conversion requires extra Python packages: "
            f"{missing_text}. Install them in the current environment first."
        )

    return imported["cv2"], imported["h5py"], imported["pa"], imported["pq"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_episode_id(path: Path) -> int:
    match = EPISODE_FILE_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Expected an episode file like `episode12.*`, got {path.name}")
    return int(match.group(1))


def dedupe_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def humanize_task_name(task_name: str) -> str:
    return task_name.replace("_", " ").strip() or task_name


def is_rmbench_demo_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "data").is_dir()
        and (path / "instructions").is_dir()
    )


def discover_rmbench_demo_roots(source_root: Path) -> list[tuple[str, Path]]:
    resolved = source_root.expanduser().resolve()

    if is_rmbench_demo_root(resolved):
        return [(resolved.parent.name, resolved)]

    direct_demo = resolved / "demo_clean"
    if is_rmbench_demo_root(direct_demo):
        return [(resolved.name, direct_demo)]

    discovered: list[tuple[str, Path]] = []
    for child in sorted(resolved.iterdir()):
        if not child.is_dir() or child.name.startswith(".") or child.name == "__pycache__":
            continue
        demo_root = child / "demo_clean"
        if is_rmbench_demo_root(demo_root):
            discovered.append((child.name, demo_root))

    if not discovered:
        raise FileNotFoundError(
            "Could not find any RMBench task directories under "
            f"{resolved}. Expected child folders containing `demo_clean/data` "
            "and `demo_clean/instructions`."
        )
    return discovered


def resolve_episode_tasks(instruction_path: Path, *, fallback_task_name: str) -> tuple[str, tuple[str, ...]]:
    payload = load_json(instruction_path)
    seen_values = payload.get("seen", []) if isinstance(payload, dict) else []
    unseen_values = payload.get("unseen", []) if isinstance(payload, dict) else []
    task_candidates = dedupe_texts(
        [str(value) for value in seen_values] + [str(value) for value in unseen_values]
    )
    if not task_candidates:
        task_candidates = [humanize_task_name(fallback_task_name)]
    return task_candidates[0], tuple(task_candidates)


def collect_rmbench_task_groups(source_root: Path) -> dict[str, list[EpisodeSource]]:
    grouped: dict[str, list[EpisodeSource]] = {}
    for task_name, demo_root in discover_rmbench_demo_roots(source_root):
        data_dir = demo_root / "data"
        instruction_dir = demo_root / "instructions"
        data_files = sorted(data_dir.glob("episode*.hdf5"), key=extract_episode_id)
        if not data_files:
            raise FileNotFoundError(f"No HDF5 episodes were found under {data_dir}.")

        task_episodes: list[EpisodeSource] = []
        for hdf5_path in data_files:
            episode_id = extract_episode_id(hdf5_path)
            instruction_path = instruction_dir / f"episode{episode_id}.json"
            if not instruction_path.exists():
                raise FileNotFoundError(
                    "Instruction file is missing for RMBench episode: "
                    f"{instruction_path}"
                )
            primary_task, task_candidates = resolve_episode_tasks(
                instruction_path,
                fallback_task_name=task_name,
            )
            task_episodes.append(
                EpisodeSource(
                    source_task=task_name,
                    episode_id=episode_id,
                    hdf5_path=hdf5_path.resolve(),
                    instruction_path=instruction_path.resolve(),
                    primary_task=primary_task,
                    task_candidates=task_candidates,
                )
            )

        grouped[task_name] = task_episodes
    return grouped


def list_ffmpeg_encoders() -> set[str]:
    output = subprocess.check_output(
        ["ffmpeg", "-hide_banner", "-encoders"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    encoders: set[str] = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(("V", "A", "S")):
            encoders.add(parts[1])
    return encoders


def is_gpu_video_encoder(encoder_name: str) -> bool:
    lowered = encoder_name.strip().lower()
    return any(
        token in lowered
        for token in ("_nvenc", "_qsv", "_vaapi", "_amf", "_videotoolbox")
    )


def probe_video_encoder_runtime(encoder_name: str) -> tuple[bool, str]:
    if not is_gpu_video_encoder(encoder_name):
        return True, ""

    probe_path: Path | None = None
    temp_dir: str | None = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="rmbench_ffmpeg_probe_")
        probe_path = Path(temp_dir) / "probe.mp4"
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "16x16",
            "-r",
            "1",
            "-i",
            "-",
            "-frames:v",
            "1",
            "-an",
            "-c:v",
            encoder_name,
        ]
        if encoder_name == "h264_nvenc":
            command.extend(["-preset", "p4", "-tune", "hq", "-cq", "23"])
        elif encoder_name == "hevc_nvenc":
            command.extend(["-preset", "p4", "-tune", "hq", "-cq", "28"])
        command.extend(["-pix_fmt", "yuv420p", str(probe_path)])

        completed = subprocess.run(
            command,
            input=frame.tobytes(),
            capture_output=True,
            check=False,
        )
        stderr_text = completed.stderr.decode("utf-8", errors="replace").strip()
        stdout_text = completed.stdout.decode("utf-8", errors="replace").strip()
        detail = stderr_text or stdout_text
        return completed.returncode == 0, detail
    finally:
        if probe_path is not None and probe_path.exists():
            probe_path.unlink()
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def resolve_video_encoder(requested: str) -> tuple[str, str | None]:
    requested_text = str(requested).strip().lower()
    available_encoders = list_ffmpeg_encoders()

    if requested_text == "auto":
        for encoder_name in DEFAULT_GPU_VIDEO_ENCODERS:
            if encoder_name in available_encoders:
                is_usable, detail = probe_video_encoder_runtime(encoder_name)
                if is_usable:
                    return encoder_name, None
                message = (
                    f"GPU encoder `{encoder_name}` is present in ffmpeg but not usable at runtime. "
                    f"Falling back to `{DEFAULT_CPU_VIDEO_ENCODER}`."
                )
                if detail:
                    message += f" Detail: {detail}"
                return DEFAULT_CPU_VIDEO_ENCODER, message
        return DEFAULT_CPU_VIDEO_ENCODER, None

    if requested_text == "gpu":
        for encoder_name in DEFAULT_GPU_VIDEO_ENCODERS:
            if encoder_name in available_encoders:
                is_usable, detail = probe_video_encoder_runtime(encoder_name)
                if is_usable:
                    return encoder_name, None
                raise RuntimeError(
                    f"Requested GPU video encoding via `{encoder_name}`, but runtime probing failed. "
                    f"Detail: {detail or 'unknown error'}"
                )
        available_gpu = sorted(
            encoder_name
            for encoder_name in available_encoders
            if is_gpu_video_encoder(encoder_name)
        )
        available_text = ", ".join(available_gpu) if available_gpu else "none"
        raise RuntimeError(
            "Requested GPU video encoding, but no supported GPU encoder was found. "
            f"Available GPU-like encoders: {available_text}"
        )

    if requested_text == "cpu":
        return DEFAULT_CPU_VIDEO_ENCODER, None

    if requested_text not in available_encoders:
        raise RuntimeError(
            f"Requested ffmpeg video encoder `{requested_text}` is not available."
        )
    if is_gpu_video_encoder(requested_text):
        is_usable, detail = probe_video_encoder_runtime(requested_text)
        if not is_usable:
            raise RuntimeError(
                f"Requested GPU video encoder `{requested_text}` failed runtime probing. "
                f"Detail: {detail or 'unknown error'}"
            )
    return requested_text, None


def resolve_worker_count(
    requested: int | str,
    *,
    total_episodes: int,
    video_encoder: str,
) -> int:
    if total_episodes <= 0:
        return 1

    if isinstance(requested, int):
        return max(1, min(int(requested), total_episodes))

    requested_text = str(requested).strip().lower()
    if requested_text != DEFAULT_WORKERS:
        return max(1, min(int(requested_text), total_episodes))

    cpu_count = max(1, os.cpu_count() or 1)
    if is_gpu_video_encoder(video_encoder):
        # Each episode launches three camera encoders, so keep the default
        # conservative to avoid exhausting NVENC session limits.
        return max(1, min(1, total_episodes))
    return max(1, min(4, cpu_count, total_episodes))


def start_ffmpeg_raw_writer(
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    video_encoder: str,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        video_encoder,
    ]
    if video_encoder == "libx264":
        command.extend(["-preset", "medium", "-crf", "23"])
    elif video_encoder == "h264_nvenc":
        command.extend(["-preset", "p4", "-tune", "hq", "-cq", "23"])
    elif video_encoder == "hevc_nvenc":
        command.extend(["-preset", "p4", "-tune", "hq", "-cq", "28"])

    command.extend(["-pix_fmt", "yuv420p", str(output_path)])
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def ffprobe_video(video_path: Path) -> dict[str, Any]:
    command = [
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
    output = subprocess.check_output(command, text=True)
    payload = json.loads(output)
    stream = payload["streams"][0]

    rate = stream.get("avg_frame_rate", "0/1")
    if "/" in rate:
        numerator_text, denominator_text = rate.split("/", 1)
        fps = float(numerator_text) / max(float(denominator_text), 1.0)
    else:
        fps = float(rate)

    frame_count = stream.get("nb_frames")
    if frame_count in (None, "N/A"):
        duration = float(stream.get("duration", 0.0))
        frame_count = int(round(duration * fps))
    else:
        frame_count = int(frame_count)

    return {
        "codec": str(stream.get("codec_name", "unknown")),
        "pix_fmt": str(stream.get("pix_fmt", "unknown")),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": int(round(fps)),
        "frames": frame_count,
    }


def build_stats(values: np.ndarray | list[float]) -> dict[str, Any]:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "min": array.min(axis=0).astype(np.float64).tolist(),
        "max": array.max(axis=0).astype(np.float64).tolist(),
        "mean": array.mean(axis=0).astype(np.float64).tolist(),
        "std": array.std(axis=0).astype(np.float64).tolist(),
        "count": [int(array.shape[0])],
    }


def init_image_stats_accumulator(num_channels: int) -> dict[str, Any]:
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


def merge_image_stats_accumulator(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["sum"] += source["sum"]
    target["sumsq"] += source["sumsq"]
    target["min"] = np.minimum(target["min"], source["min"])
    target["max"] = np.maximum(target["max"], source["max"])
    target["pixel_count"] += int(source["pixel_count"])
    target["frame_count"] += int(source["frame_count"])


def update_image_stats_accumulator(accumulator: dict[str, Any], frame_hwc_uint8: np.ndarray) -> None:
    frame = np.asarray(frame_hwc_uint8, dtype=np.float32)
    if frame.ndim != 3 or frame.shape[2] <= 0:
        raise ValueError(
            "Expected an image with shape (height, width, channels). "
            f"Got {tuple(frame.shape)}."
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


def finalize_image_stats(accumulator: dict[str, Any]) -> dict[str, Any]:
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


def fixed_size_float_list_array(pa, values: np.ndarray):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={array.shape}.")
    return pa.FixedSizeListArray.from_arrays(
        pa.array(array.reshape(-1), type=pa.float32()),
        int(array.shape[1]),
    )


def scalar_feature(dtype: str) -> dict[str, Any]:
    return {"dtype": dtype, "shape": [1], "names": None}


def decode_rmbench_image(cv2, encoded: Any) -> np.ndarray:
    if isinstance(encoded, np.ndarray) and encoded.dtype == np.uint8:
        encoded_array = encoded.reshape(-1)
    elif isinstance(encoded, (bytes, bytearray, np.bytes_)):
        encoded_array = np.frombuffer(encoded, dtype=np.uint8)
    else:
        encoded_array = np.frombuffer(bytes(encoded), dtype=np.uint8)

    frame = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode an RMBench JPEG frame from HDF5.")
    if frame.shape[0] != DEFAULT_IMAGE_HEIGHT or frame.shape[1] != DEFAULT_IMAGE_WIDTH:
        frame = cv2.resize(
            frame,
            (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(frame)


def build_joint_state_matrix(h5_file) -> np.ndarray:
    joint_group = h5_file["/joint_action"]
    if "vector" in joint_group:
        state_matrix = np.asarray(joint_group["vector"][()], dtype=np.float32)
    else:
        left_arm = np.asarray(joint_group["left_arm"][()], dtype=np.float32)
        left_gripper = np.asarray(joint_group["left_gripper"][()], dtype=np.float32)[:, None]
        right_arm = np.asarray(joint_group["right_arm"][()], dtype=np.float32)
        right_gripper = np.asarray(joint_group["right_gripper"][()], dtype=np.float32)[:, None]
        state_matrix = np.concatenate(
            [left_arm, left_gripper, right_arm, right_gripper],
            axis=1,
        ).astype(np.float32, copy=False)

    if state_matrix.ndim != 2:
        raise RuntimeError(
            "Expected RMBench joint_action data with shape (T, state_dim). "
            f"Got shape={state_matrix.shape}."
        )
    if state_matrix.shape[0] < 2:
        raise RuntimeError(
            "Each RMBench episode must contain at least 2 timesteps to derive actions. "
            f"Got shape={state_matrix.shape}."
        )
    return state_matrix


def write_tasks_tables(pa, pq, *, output_root: Path, task_index_by_text: dict[str, int]) -> None:
    ordered_tasks = sorted(task_index_by_text.items(), key=lambda item: item[1])
    jsonl_path = output_root / "meta/tasks.jsonl"
    parquet_path = output_root / "meta/tasks.parquet"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for task_text, task_index in ordered_tasks:
            handle.write(
                json.dumps(
                    {"task_index": int(task_index), "task": task_text},
                    ensure_ascii=False,
                )
                + "\n"
            )

    table = pa.Table.from_arrays(
        [
            pa.array([task_index for _, task_index in ordered_tasks], type=pa.int64()),
            pa.array([task_text for task_text, _ in ordered_tasks], type=pa.string()),
        ],
        names=["task_index", "task"],
    )
    pq.write_table(table, parquet_path, compression="snappy")


def write_episodes_table(pa, pq, *, output_root: Path, episodes_meta: list[dict[str, Any]], video_keys: list[str]) -> None:
    output_path = output_root / DEFAULT_EPISODES_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = [
        pa.array([episode["episode_index"] for episode in episodes_meta], type=pa.int64()),
        pa.array([episode["tasks"] for episode in episodes_meta], type=pa.list_(pa.string())),
        pa.array([episode["length"] for episode in episodes_meta], type=pa.int64()),
        pa.array([episode["data/chunk_index"] for episode in episodes_meta], type=pa.int64()),
        pa.array([episode["data/file_index"] for episode in episodes_meta], type=pa.int64()),
        pa.array([episode["dataset_from_index"] for episode in episodes_meta], type=pa.int64()),
        pa.array([episode["dataset_to_index"] for episode in episodes_meta], type=pa.int64()),
    ]
    names = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]

    for video_key in video_keys:
        arrays.extend(
            [
                pa.array(
                    [episode[f"videos/{video_key}/chunk_index"] for episode in episodes_meta],
                    type=pa.int64(),
                ),
                pa.array(
                    [episode[f"videos/{video_key}/file_index"] for episode in episodes_meta],
                    type=pa.int64(),
                ),
                pa.array(
                    [episode[f"videos/{video_key}/from_timestamp"] for episode in episodes_meta],
                    type=pa.float32(),
                ),
                pa.array(
                    [episode[f"videos/{video_key}/to_timestamp"] for episode in episodes_meta],
                    type=pa.float32(),
                ),
            ]
        )
        names.extend(
            [
                f"videos/{video_key}/chunk_index",
                f"videos/{video_key}/file_index",
                f"videos/{video_key}/from_timestamp",
                f"videos/{video_key}/to_timestamp",
            ]
        )

    arrays.extend(
        [
            pa.array([episode["meta/episodes/chunk_index"] for episode in episodes_meta], type=pa.int64()),
            pa.array([episode["meta/episodes/file_index"] for episode in episodes_meta], type=pa.int64()),
        ]
    )
    names.extend(["meta/episodes/chunk_index", "meta/episodes/file_index"])

    table = pa.Table.from_arrays(arrays, names=names)
    pq.write_table(table, output_path, compression="snappy")


def validate_episode_ranges(*, total_frames: int, episodes_meta: list[dict[str, Any]]) -> None:
    expected_index = 0
    total_length = 0
    for episode in episodes_meta:
        from_index = int(episode["dataset_from_index"])
        to_index = int(episode["dataset_to_index"])
        length = int(episode["length"])
        if from_index != expected_index:
            raise RuntimeError(
                "Episode data ranges are not contiguous. "
                f"Expected start index {expected_index}, got {from_index}."
            )
        if to_index - from_index != length:
            raise RuntimeError(
                "Episode range length does not match episode metadata. "
                f"episode_index={episode['episode_index']}, range=({from_index}, {to_index}), "
                f"length={length}"
            )
        total_length += length
        expected_index = to_index

    if total_length != total_frames:
        raise RuntimeError(
            "The sum of all episode lengths does not match total frame count. "
            f"sum(lengths)={total_length}, total_frames={total_frames}"
        )


def build_split_spec(total_episodes: int, train_ratio: float) -> dict[str, str]:
    if total_episodes <= 0:
        raise ValueError("`total_episodes` must be positive.")
    if total_episodes == 1:
        return {"train": "0:1"}
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"`train_ratio` must lie in (0, 1), got {train_ratio}.")
    train_count = int(round(total_episodes * train_ratio))
    train_count = min(max(1, train_count), total_episodes - 1)
    return {
        "train": f"0:{train_count}",
        "val": f"{train_count}:{total_episodes}",
    }


def convert_one_rmbench_episode(
    *,
    episode_index: int,
    episode: EpisodeSource,
    output_root: Path,
    fps: int,
    video_encoder: str,
    progress_slots: queue.SimpleQueue[int] | None,
    tqdm_cls,
) -> EpisodeConversionResult:
    cv2, h5py, _, _ = require_rmbench_dependencies()
    video_keys = [
        f"observation.images.{camera_name}"
        for camera_name in RMbench_RAW_CAMERA_TO_LEROBOT.values()
    ]
    image_stats_by_key = {
        video_key: init_image_stats_accumulator(num_channels=3) for video_key in video_keys
    }
    progress_slot: int | None = None
    progress_bar = None

    try:
        if progress_slots is not None:
            progress_slot = progress_slots.get()

        with h5py.File(episode.hdf5_path, "r") as h5_file:
            state_matrix = build_joint_state_matrix(h5_file)
            frame_count = int(state_matrix.shape[0]) - 1
            state_sequence = state_matrix[:-1].astype(np.float32, copy=False)
            action_sequence = state_matrix[1:].astype(np.float32, copy=False)

            if tqdm_cls is not None and progress_slot is not None:
                progress_bar = tqdm_cls(
                    total=frame_count,
                    desc=f"ep {episode_index:03d} {episode.source_task}",
                    position=progress_slot,
                    leave=False,
                    dynamic_ncols=True,
                )

            video_paths: dict[str, str] = {}
            writer_by_key: dict[str, subprocess.Popen] = {}
            try:
                for raw_camera_name, lerobot_camera_name in RMbench_RAW_CAMERA_TO_LEROBOT.items():
                    video_key = f"observation.images.{lerobot_camera_name}"
                    video_path = output_root / DEFAULT_VIDEO_PATH.format(
                        video_key=video_key,
                        chunk_index=0,
                        file_index=episode_index,
                    )
                    video_paths[video_key] = str(video_path)
                    writer_by_key[video_key] = start_ffmpeg_raw_writer(
                        video_path,
                        width=DEFAULT_IMAGE_WIDTH,
                        height=DEFAULT_IMAGE_HEIGHT,
                        fps=fps,
                        video_encoder=video_encoder,
                    )

                for frame_index in range(frame_count):
                    for raw_camera_name, lerobot_camera_name in RMbench_RAW_CAMERA_TO_LEROBOT.items():
                        video_key = f"observation.images.{lerobot_camera_name}"
                        encoded = h5_file[f"/observation/{raw_camera_name}/rgb"][frame_index]
                        frame = decode_rmbench_image(cv2, encoded)
                        writer = writer_by_key[video_key]
                        if writer.stdin is None:
                            raise RuntimeError(f"ffmpeg stdin closed unexpectedly for {video_key}.")
                        writer.stdin.write(frame.tobytes())
                        update_image_stats_accumulator(image_stats_by_key[video_key], frame)
                    if progress_bar is not None:
                        progress_bar.update(1)
            finally:
                for video_key, writer in writer_by_key.items():
                    if writer.stdin is not None and not writer.stdin.closed:
                        writer.stdin.close()
                    return_code = writer.wait()
                    if return_code != 0:
                        raise RuntimeError(
                            f"ffmpeg failed while encoding {video_key} for episode {episode_index} "
                            f"with exit code {return_code}."
                        )

        return EpisodeConversionResult(
            episode_index=int(episode_index),
            source_task=episode.source_task,
            source_episode_id=int(episode.episode_id),
            primary_task=episode.primary_task,
            task_candidates=episode.task_candidates,
            frame_count=int(frame_count),
            state_sequence=state_sequence,
            action_sequence=action_sequence,
            video_paths=video_paths,
            image_stats_by_key=image_stats_by_key,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if progress_slot is not None and progress_slots is not None:
            progress_slots.put(progress_slot)


def write_rmbench_collection_manifest(
    *,
    output_root: Path,
    source_root: Path,
    dataset_id: str,
    task_groups: dict[str, list[EpisodeSource]],
) -> None:
    tasks_payload: list[dict[str, Any]] = []
    total_episodes = 0
    for task_name, episodes in sorted(task_groups.items()):
        total_episodes += len(episodes)
        tasks_payload.append(
            {
                "task_name": task_name,
                "dataset_id": f"{dataset_id}/{task_name}",
                "episode_count": int(len(episodes)),
                "path": task_name,
            }
        )

    payload = {
        "source_format": "rmbench",
        "source_path": str(source_root),
        "dataset_id": str(dataset_id),
        "storage": "per_task",
        "task_datasets": tasks_payload,
        "total_task_datasets": int(len(tasks_payload)),
        "total_episodes": int(total_episodes),
    }
    (output_root / "collection.json").write_text(
        json.dumps(payload, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def convert_rmbench_task_dataset(
    *,
    source_root: Path,
    task_name: str,
    episodes: list[EpisodeSource],
    dataset_id: str,
    output_root: Path,
    fps: int,
    train_ratio: float,
    overwrite: bool,
    workers: int | str,
    video_encoder: str,
) -> Path:
    require_binary("ffmpeg")
    require_binary("ffprobe")
    _, _, pa, pq = require_rmbench_dependencies()
    tqdm_cls = maybe_import_tqdm()

    if not episodes:
        raise RuntimeError(
            f"No RMBench episodes were discovered for task `{task_name}` under {source_root}."
        )
    resolved_video_encoder, encoder_notice = resolve_video_encoder(video_encoder)
    resolved_workers = resolve_worker_count(
        workers,
        total_episodes=len(episodes),
        video_encoder=resolved_video_encoder,
    )

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Pass `--overwrite` to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    video_keys = [
        f"observation.images.{camera_name}"
        for camera_name in RMbench_RAW_CAMERA_TO_LEROBOT.values()
    ]
    image_stats_by_key = {
        video_key: init_image_stats_accumulator(num_channels=3) for video_key in video_keys
    }
    sample_video_path_by_key: dict[str, Path] = {}
    task_index_by_text: dict[str, int] = {}
    for episode in episodes:
        task_index_by_text.setdefault(episode.primary_task, len(task_index_by_text))
    records: dict[str, list[Any]] = {
        "next.reward": [],
        "next.done": [],
        "next.success": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    episodes_meta: list[dict[str, Any]] = []
    state_dim: int | None = None
    global_frame_index = 0

    source_task_names = sorted({episode.source_task for episode in episodes})
    print(
        "Converting RMBench task dataset: "
        f"source={source_root}, task={task_name}, episodes={len(episodes)}, "
        f"dataset_id={dataset_id}, output={output_root}, "
        f"video_encoder={resolved_video_encoder}, workers={resolved_workers}"
    )
    if encoder_notice:
        print(f"[info] {encoder_notice}")
    progress_slots: queue.SimpleQueue[int] | None = None
    if tqdm_cls is not None:
        progress_slots = queue.SimpleQueue()
        for position in range(1, resolved_workers + 1):
            progress_slots.put(position)

    overall_progress = (
        None
        if tqdm_cls is None
        else tqdm_cls(
            total=len(episodes),
            desc=f"episodes {task_name}",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )
    )

    results_by_episode_index: dict[int, EpisodeConversionResult] = {}
    try:
        with cf.ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            futures = [
                executor.submit(
                    convert_one_rmbench_episode,
                    episode_index=episode_index,
                    episode=episode,
                    output_root=output_root,
                    fps=fps,
                    video_encoder=resolved_video_encoder,
                    progress_slots=progress_slots,
                    tqdm_cls=tqdm_cls,
                )
                for episode_index, episode in enumerate(episodes)
            ]
            for future in cf.as_completed(futures):
                result = future.result()
                results_by_episode_index[result.episode_index] = result
                if overall_progress is not None:
                    overall_progress.update(1)
    finally:
        if overall_progress is not None:
            overall_progress.close()

    for episode_index, episode in enumerate(episodes):
        result = results_by_episode_index[episode_index]
        task_index = task_index_by_text[result.primary_task]
        episode_from_index = global_frame_index

        if state_dim is None:
            state_dim = int(result.state_sequence.shape[1])
        elif int(result.state_sequence.shape[1]) != state_dim:
            raise RuntimeError(
                "All RMBench episodes must share the same state dimension. "
                f"Expected {state_dim}, got {int(result.state_sequence.shape[1])} "
                f"in episode_index={episode_index}."
            )

        state_rows.append(result.state_sequence)
        action_rows.append(result.action_sequence)
        episode_length = int(result.frame_count)
        for frame_index in range(episode_length):
            done = frame_index == episode_length - 1
            success = done
            records["next.reward"].append(float(1.0 if success else 0.0))
            records["next.done"].append(bool(done))
            records["next.success"].append(bool(success))
            records["timestamp"].append(float(frame_index / fps))
            records["frame_index"].append(int(frame_index))
            records["episode_index"].append(int(episode_index))
            records["index"].append(int(global_frame_index))
            records["task_index"].append(int(task_index))
            global_frame_index += 1

        for video_key in video_keys:
            merge_image_stats_accumulator(
                image_stats_by_key[video_key],
                result.image_stats_by_key[video_key],
            )
            sample_video_path_by_key.setdefault(video_key, Path(result.video_paths[video_key]))

        episode_to_index = global_frame_index
        episode_length = episode_to_index - episode_from_index
        episode_metadata: dict[str, Any] = {
            "episode_index": int(episode_index),
            "tasks": list(result.task_candidates),
            "length": int(episode_length),
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": int(episode_from_index),
            "dataset_to_index": int(episode_to_index),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for video_key in video_keys:
            episode_metadata[f"videos/{video_key}/chunk_index"] = 0
            episode_metadata[f"videos/{video_key}/file_index"] = int(episode_index)
            episode_metadata[f"videos/{video_key}/from_timestamp"] = float(0.0)
            episode_metadata[f"videos/{video_key}/to_timestamp"] = float(episode_length / fps)
        episodes_meta.append(episode_metadata)

    if state_dim is None:
        raise RuntimeError("No RMBench frames were converted.")

    validate_episode_ranges(
        total_frames=global_frame_index,
        episodes_meta=episodes_meta,
    )

    state_array = np.concatenate(state_rows, axis=0)
    action_array = np.concatenate(action_rows, axis=0)
    total_frames = int(state_array.shape[0])
    total_episodes = int(len(episodes_meta))
    if int(action_array.shape[0]) != total_frames:
        raise RuntimeError(
            "State/action frame counts diverged during conversion. "
            f"state_frames={total_frames}, action_frames={int(action_array.shape[0])}"
        )

    data_table = pa.Table.from_arrays(
        [
            fixed_size_float_list_array(pa, state_array),
            fixed_size_float_list_array(pa, action_array),
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
    data_output_path = output_root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data_table, data_output_path, compression="snappy")

    write_episodes_table(
        pa,
        pq,
        output_root=output_root,
        episodes_meta=episodes_meta,
        video_keys=video_keys,
    )
    write_tasks_tables(
        pa,
        pq,
        output_root=output_root,
        task_index_by_text=task_index_by_text,
    )

    sample_video_info_by_key = {
        video_key: ffprobe_video(sample_video_path_by_key[video_key]) for video_key in video_keys
    }
    splits = build_split_spec(total_episodes, train_ratio)
    state_feature_names = (
        MOTOR_NAMES if state_dim == len(MOTOR_NAMES) else [f"joint_{index}" for index in range(state_dim)]
    )

    info = {
        "codebase_version": "v3.0",
        "robot_type": "aloha",
        "source_format": "rmbench",
        "source_path": str(source_root),
        "source_task_names": source_task_names,
        "dataset_id": str(dataset_id),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": int(len(task_index_by_text)),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": int(fps),
        "video_encoder": resolved_video_encoder,
        "conversion_workers": int(resolved_workers),
        "splits": splits,
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [int(state_dim)],
                "names": state_feature_names,
            },
            "action": {
                "dtype": "float32",
                "shape": [int(state_dim)],
                "names": state_feature_names,
            },
            "next.reward": scalar_feature("float32"),
            "next.done": scalar_feature("bool"),
            "next.success": scalar_feature("bool"),
            "timestamp": scalar_feature("float32"),
            "frame_index": scalar_feature("int64"),
            "episode_index": scalar_feature("int64"),
            "index": scalar_feature("int64"),
            "task_index": scalar_feature("int64"),
        },
    }
    for video_key in video_keys:
        video_info = sample_video_info_by_key[video_key]
        info["features"][video_key] = {
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
        }

    stats = {
        "observation.state": build_stats(state_array),
        "action": build_stats(action_array),
        "next.reward": build_stats(np.asarray(records["next.reward"], dtype=np.float32)),
        "timestamp": build_stats(np.asarray(records["timestamp"], dtype=np.float32)),
    }
    for video_key in video_keys:
        stats[video_key] = finalize_image_stats(image_stats_by_key[video_key])

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(
        json.dumps(info, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    (meta_dir / "stats.json").write_text(
        json.dumps(stats, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "Generated LeRobotDataset v3.0 at: "
        f"{output_root.resolve()} "
        f"(task={task_name}, episodes={total_episodes}, frames={total_frames}, tasks={len(task_index_by_text)})"
    )
    return output_root


def convert_rmbench_dataset(
    *,
    source_root: Path,
    dataset_id: str,
    output_root: Path,
    fps: int,
    train_ratio: float,
    overwrite: bool,
    workers: int | str,
    video_encoder: str,
) -> Path:
    require_binary("ffmpeg")
    require_binary("ffprobe")
    require_rmbench_dependencies()

    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    task_groups = collect_rmbench_task_groups(source_root)
    if not task_groups:
        raise RuntimeError(f"No RMBench task directories were discovered under {source_root}.")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Pass `--overwrite` to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    total_episodes = sum(len(episodes) for episodes in task_groups.values())
    print(
        "Converting RMBench task collection: "
        f"source={source_root}, task_datasets={len(task_groups)}, episodes={total_episodes}, "
        f"dataset_id={dataset_id}, output={output_root}"
    )

    for task_name, episodes in sorted(task_groups.items()):
        task_output_root = output_root / task_name
        task_dataset_id = f"{dataset_id}/{task_name}"
        convert_rmbench_task_dataset(
            source_root=source_root,
            task_name=task_name,
            episodes=episodes,
            dataset_id=task_dataset_id,
            output_root=task_output_root,
            fps=fps,
            train_ratio=train_ratio,
            overwrite=False,
            workers=workers,
            video_encoder=video_encoder,
        )

    write_rmbench_collection_manifest(
        output_root=output_root,
        source_root=source_root,
        dataset_id=dataset_id,
        task_groups=task_groups,
    )
    print(
        "Generated RMBench task collection at: "
        f"{output_root.resolve()} "
        f"(task_datasets={len(task_groups)}, episodes={total_episodes})"
    )
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert external datasets into local LeRobot v3.0 datasets."
    )
    parser.add_argument(
        "source_path",
        type=Path,
        help="Path to the source dataset root, for example `/path/to/RMBench/data`.",
    )
    parser.add_argument(
        "--convert",
        choices=SUPPORTED_CONVERTERS,
        required=True,
        help="Source dataset format to convert.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Logical dataset id used to name the local output dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to `main/data/<dataset-id>`.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Target FPS for the converted LeRobot videos.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Fraction of episodes assigned to the contiguous `train` split.",
    )
    parser.add_argument(
        "--workers",
        type=str,
        default=DEFAULT_WORKERS,
        help='Episode conversion parallelism. Pass a positive integer or "auto".',
    )
    parser.add_argument(
        "--video-encoder",
        type=str,
        default=DEFAULT_VIDEO_ENCODER,
        help='ffmpeg video encoder. Use "auto", "gpu", "cpu", or an exact encoder name.',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / args.dataset_id.replace("/", "_")
    )

    if args.convert == "rmbench":
        convert_rmbench_dataset(
            source_root=args.source_path,
            dataset_id=args.dataset_id,
            output_root=output_dir,
            fps=int(args.fps),
            train_ratio=float(args.train_ratio),
            overwrite=bool(args.overwrite),
            workers=args.workers,
            video_encoder=str(args.video_encoder),
        )
        return

    raise ValueError(f"Unsupported converter: {args.convert}")


if __name__ == "__main__":
    main()
