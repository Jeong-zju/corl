#!/usr/bin/env python3
"""Convert ROS bag datasets into local LeRobot v3 datasets."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
from contextlib import ExitStack
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"
DEFAULT_STAGING_DIRNAME = ".staging"
DEFAULT_FPS = 30
DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_ARM_DOF = 7
DEFAULT_WORKERS = "auto"
DEFAULT_VIDEO_ENCODER = "auto"
DEFAULT_GPU_VIDEO_ENCODERS = (
    "h264_nvenc",
    "hevc_nvenc",
    "h264_qsv",
    "h264_vaapi",
    "h264_videotoolbox",
    "hevc_videotoolbox",
)
DEFAULT_CPU_VIDEO_ENCODER = "libx264"

DEFAULT_CAMERA_TOP_TOPIC = "/realsense_top/color/image_raw/compressed"
DEFAULT_CAMERA_LEFT_TOPIC = "/realsense_left/color/image_raw/compressed"
DEFAULT_CAMERA_RIGHT_TOPIC = "/realsense_right/color/image_raw/compressed"
DEFAULT_STATE_LEFT_TOPIC = "/robot/arm_left/joint_states_single"
DEFAULT_STATE_RIGHT_TOPIC = "/robot/arm_right/joint_states_single"
DEFAULT_ACTION_LEFT_TOPIC = "/teleop/arm_left/joint_states_single"
DEFAULT_ACTION_RIGHT_TOPIC = "/teleop/arm_right/joint_states_single"
DEFAULT_ODOM_TOPIC = "/ranger_base_node/odom"

CAMERA_TOP_KEY = "observation.images.realsense_top"
CAMERA_LEFT_KEY = "observation.images.realsense_left"
CAMERA_RIGHT_KEY = "observation.images.realsense_right"


@dataclass(frozen=True)
class RosbagTopics:
    camera_top_topic: str
    camera_left_topic: str
    camera_right_topic: str
    state_left_topic: str
    state_right_topic: str
    action_left_topic: str
    action_right_topic: str
    odom_topic: str | None

    def camera_topic_pairs(self) -> list[tuple[str, str]]:
        return [
            (self.camera_top_topic, CAMERA_TOP_KEY),
            (self.camera_left_topic, CAMERA_LEFT_KEY),
            (self.camera_right_topic, CAMERA_RIGHT_KEY),
        ]

    def required_topics(self) -> list[str]:
        return [
            self.camera_top_topic,
            self.camera_left_topic,
            self.camera_right_topic,
            self.state_left_topic,
            self.state_right_topic,
            self.action_left_topic,
            self.action_right_topic,
        ]

    def all_topics(self) -> list[str]:
        topics = self.required_topics()
        if self.odom_topic:
            topics.append(self.odom_topic)
        return topics


@dataclass(frozen=True)
class ConversionConfig:
    dataset_id: str
    task_label: str
    robot_type: str
    fps: int
    image_width: int
    image_height: int
    arm_dof: int
    topics: RosbagTopics


@dataclass(frozen=True)
class WorkerTask:
    bag_index: int
    bag_path: str
    staging_root: str
    config: ConversionConfig
    video_encoder: str


@dataclass(frozen=True)
class EpisodeArtifact:
    bag_index: int
    bag_path: str
    status: str
    frame_count: int
    elapsed_s: float
    detail: str | None = None
    staging_dir: str | None = None
    data_path: str | None = None
    video_paths: dict[str, str] | None = None
    state_stats: dict[str, Any] | None = None
    action_stats: dict[str, Any] | None = None
    reward_stats: dict[str, Any] | None = None
    timestamp_stats: dict[str, Any] | None = None
    image_stats_by_key: dict[str, dict[str, Any]] | None = None


@dataclass
class LowDimStreams:
    camera_bounds: dict[str, tuple[int, int]]
    state_left_times: np.ndarray
    state_left_values: np.ndarray
    state_right_times: np.ndarray
    state_right_values: np.ndarray
    action_left_times: np.ndarray
    action_left_values: np.ndarray
    action_right_times: np.ndarray
    action_right_values: np.ndarray
    odom_times: np.ndarray
    odom_values: np.ndarray


@dataclass
class CameraCursor:
    topic: str
    video_key: str
    reader: Any
    connection: Any
    iterator: Iterator[tuple[Any, int, Any]]
    prev_timestamp: int | None = None
    prev_raw: Any | None = None
    curr_timestamp: int | None = None
    curr_raw: Any | None = None
    cached_timestamp: int | None = None
    cached_frame: np.ndarray | None = None


class SkippedBag(RuntimeError):
    """Raised when a bag should be skipped without failing the whole run."""


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


def require_rosbag_decode_dependencies():
    missing: list[str] = []
    imported: dict[str, Any] = {}

    try:
        import cv2  # type: ignore
    except Exception:
        missing.append("opencv-python")
    else:
        imported["cv2"] = cv2

    try:
        from rosbags.highlevel import AnyReader  # type: ignore
    except Exception:
        missing.append("rosbags")
    else:
        imported["AnyReader"] = AnyReader

    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            "ROS bag conversion requires extra Python packages: "
            f"{missing_text}. Install them in the current environment first."
        )

    return imported["cv2"], imported["AnyReader"]


def require_output_dependencies():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Dataset writing requires `pyarrow`. "
            "Install it in the current environment first."
        ) from exc
    return pa, pq


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_get_load_average() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None


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
        temp_dir = tempfile.mkdtemp(prefix="rosbag_ffmpeg_probe_")
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
    aliases = {
        "cpu": "libx264",
        "h264": "libx264",
        "libx264": "libx264",
        "hevc": "libx265",
        "libx265": "libx265",
    }

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

    resolved = aliases.get(requested_text, requested_text)
    if resolved not in available_encoders:
        raise RuntimeError(
            f"Requested ffmpeg video encoder `{requested_text}` is not available."
        )
    if is_gpu_video_encoder(resolved):
        is_usable, detail = probe_video_encoder_runtime(resolved)
        if not is_usable:
            raise RuntimeError(
                f"Requested GPU video encoder `{resolved}` failed runtime probing. "
                f"Detail: {detail or 'unknown error'}"
            )
    return resolved, None


def resolve_worker_count(
    requested: int | str,
    *,
    total_bags: int,
    video_encoder: str,
) -> tuple[int, str | None]:
    if total_bags <= 0:
        return 1, None

    if isinstance(requested, int):
        return max(1, min(int(requested), total_bags)), None

    requested_text = str(requested).strip().lower()
    if requested_text != DEFAULT_WORKERS:
        return max(1, min(int(requested_text), total_bags)), None

    cpu_count = max(1, os.cpu_count() or 1)
    load1 = maybe_get_load_average()
    if load1 is None:
        available_cpu = cpu_count
    else:
        available_cpu = max(1, int(math.floor(cpu_count - load1)))

    divisor = 12 if is_gpu_video_encoder(video_encoder) else 8
    estimated_workers = max(1, available_cpu // divisor)
    if cpu_count >= 16 and estimated_workers < 2:
        estimated_workers = 2
    cap = 2 if is_gpu_video_encoder(video_encoder) else 6
    resolved = max(1, min(total_bags, cap, estimated_workers))
    notice = (
        "Auto worker selection: "
        f"cpu_count={cpu_count}, "
        f"load1={'n/a' if load1 is None else f'{load1:.2f}'}, "
        f"available_cpu={available_cpu} -> workers={resolved}"
    )
    return resolved, notice


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
    elif video_encoder == "libx265":
        command.extend(["-preset", "medium", "-crf", "28"])
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


def init_vector_stats_accumulator(num_features: int) -> dict[str, Any]:
    if num_features <= 0:
        raise ValueError(f"`num_features` must be positive, got {num_features}.")
    return {
        "sum": np.zeros((num_features,), dtype=np.float64),
        "sumsq": np.zeros((num_features,), dtype=np.float64),
        "min": np.full((num_features,), np.inf, dtype=np.float64),
        "max": np.full((num_features,), -np.inf, dtype=np.float64),
        "count": 0,
    }


def update_vector_stats_accumulator(accumulator: dict[str, Any], values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={array.shape}.")
    accumulator["sum"] += array.sum(axis=0, dtype=np.float64)
    accumulator["sumsq"] += np.square(array, dtype=np.float64).sum(axis=0, dtype=np.float64)
    accumulator["min"] = np.minimum(accumulator["min"], array.min(axis=0).astype(np.float64))
    accumulator["max"] = np.maximum(accumulator["max"], array.max(axis=0).astype(np.float64))
    accumulator["count"] += int(array.shape[0])


def merge_vector_stats_accumulator(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["sum"] += source["sum"]
    target["sumsq"] += source["sumsq"]
    target["min"] = np.minimum(target["min"], source["min"])
    target["max"] = np.maximum(target["max"], source["max"])
    target["count"] += int(source["count"])


def finalize_vector_stats(accumulator: dict[str, Any]) -> dict[str, Any]:
    count = int(accumulator["count"])
    if count <= 0:
        raise ValueError("Vector stats accumulator is empty.")
    mean = accumulator["sum"] / count
    variance = accumulator["sumsq"] / count - np.square(mean, dtype=np.float64)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance, dtype=np.float64)
    return {
        "min": accumulator["min"].tolist(),
        "max": accumulator["max"].tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": [count],
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


def scalar_feature(dtype: str, *, fps: int | None = None) -> dict[str, Any]:
    feature: dict[str, Any] = {"dtype": dtype, "shape": [1], "names": None}
    if fps is not None:
        feature["fps"] = int(fps)
    return feature


def vector_feature(
    dtype: str,
    shape: list[int],
    names: list[str] | None,
    *,
    fps: int | None = None,
) -> dict[str, Any]:
    feature: dict[str, Any] = {"dtype": dtype, "shape": shape, "names": names}
    if fps is not None:
        feature["fps"] = int(fps)
    return feature


def video_feature(
    *,
    video_info: dict[str, Any],
    fps: int,
) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [3, int(video_info["height"]), int(video_info["width"])],
        "names": ["channels", "height", "width"],
        "fps": int(fps),
        "info": {
            "video.height": int(video_info["height"]),
            "video.width": int(video_info["width"]),
            "video.codec": str(video_info["codec"]),
            "video.pix_fmt": str(video_info["pix_fmt"]),
            "video.is_depth_map": False,
            "video.fps": int(fps),
            "video.channels": 3,
            "has_audio": False,
        },
    }


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


def normalize_optional_topic(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "disable", "disabled", "false", "off"}:
        return None
    return text


def safe_video_key_filename(video_key: str) -> str:
    return video_key.replace("/", "__").replace(".", "_") + ".mp4"


def collect_bag_files(source_path: Path, bag_glob: str) -> list[Path]:
    resolved = source_path.expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix != ".bag":
            raise ValueError(f"Expected a `.bag` file, got {resolved}.")
        return [resolved]

    if not resolved.is_dir():
        raise FileNotFoundError(f"Source path does not exist: {resolved}")

    bag_files = sorted(path.resolve() for path in resolved.glob(bag_glob) if path.is_file())
    if not bag_files:
        raise FileNotFoundError(
            f"No `.bag` files matched `{bag_glob}` under {resolved}."
        )
    return bag_files


def nearest_indices(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if times.size == 0:
        raise ValueError("`times` must not be empty.")
    right = np.searchsorted(times, targets, side="left")
    right_clipped = np.clip(right, 0, times.size - 1)
    left = np.clip(right - 1, 0, times.size - 1)
    choose_left = (right > 0) & (
        (right == times.size)
        | (np.abs(times[right_clipped] - targets) >= np.abs(targets - times[left]))
    )
    return np.where(choose_left, left, right_clipped).astype(np.int64, copy=False)


def extract_joint_positions(msg: Any, *, dof: int) -> np.ndarray:
    positions = list(msg.position)
    if len(positions) >= dof:
        values = positions[:dof]
    else:
        values = positions + [0.0] * (dof - len(positions))
    return np.asarray(values, dtype=np.float32)


def extract_base_velocity(msg: Any) -> np.ndarray:
    twist = msg.twist.twist
    return np.asarray(
        [twist.linear.x, twist.linear.y, twist.angular.z],
        dtype=np.float32,
    )


def decode_compressed_image(
    cv2,
    msg: Any,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    encoded_array = np.frombuffer(msg.data, dtype=np.uint8)
    frame = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode a compressed ROS image.")
    if frame.shape[1] != image_width or frame.shape[0] != image_height:
        frame = cv2.resize(
            frame,
            (image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.ascontiguousarray(frame)


def build_connection_map(reader: Any, topics: list[str]) -> dict[str, Any]:
    topic_set = set(topics)
    connection_map: dict[str, Any] = {}
    for connection in reader.connections:
        if connection.topic in topic_set and connection.topic not in connection_map:
            connection_map[connection.topic] = connection
    return connection_map


def collect_low_dim_streams(bag_path: Path, config: ConversionConfig) -> LowDimStreams:
    _, AnyReader = require_rosbag_decode_dependencies()
    topics = config.topics

    with AnyReader([bag_path]) as reader:
        connection_map = build_connection_map(reader, topics.all_topics())
        missing_required = [
            topic for topic in topics.required_topics() if topic not in connection_map
        ]
        if missing_required:
            raise SkippedBag(
                "Bag is missing required topics: " + ", ".join(sorted(missing_required))
            )

        camera_bounds: dict[str, list[int | None]] = {
            topic: [None, None] for topic, _ in topics.camera_topic_pairs()
        }
        state_left_times: list[int] = []
        state_left_values: list[np.ndarray] = []
        state_right_times: list[int] = []
        state_right_values: list[np.ndarray] = []
        action_left_times: list[int] = []
        action_left_values: list[np.ndarray] = []
        action_right_times: list[int] = []
        action_right_values: list[np.ndarray] = []
        odom_times: list[int] = []
        odom_values: list[np.ndarray] = []

        selected_connections = list(connection_map.values())
        for connection, timestamp, raw in reader.messages(connections=selected_connections):
            topic = connection.topic
            if topic in camera_bounds:
                if camera_bounds[topic][0] is None:
                    camera_bounds[topic][0] = int(timestamp)
                camera_bounds[topic][1] = int(timestamp)
                continue

            msg = reader.deserialize(raw, connection.msgtype)
            if topic == topics.state_left_topic:
                state_left_times.append(int(timestamp))
                state_left_values.append(extract_joint_positions(msg, dof=config.arm_dof))
            elif topic == topics.state_right_topic:
                state_right_times.append(int(timestamp))
                state_right_values.append(extract_joint_positions(msg, dof=config.arm_dof))
            elif topic == topics.action_left_topic:
                action_left_times.append(int(timestamp))
                action_left_values.append(extract_joint_positions(msg, dof=config.arm_dof))
            elif topic == topics.action_right_topic:
                action_right_times.append(int(timestamp))
                action_right_values.append(extract_joint_positions(msg, dof=config.arm_dof))
            elif topics.odom_topic and topic == topics.odom_topic:
                odom_times.append(int(timestamp))
                odom_values.append(extract_base_velocity(msg))

    missing_camera_bounds = [
        topic
        for topic, bounds in camera_bounds.items()
        if bounds[0] is None or bounds[1] is None
    ]
    if missing_camera_bounds:
        raise SkippedBag(
            "Bag is missing camera data on topics: "
            + ", ".join(sorted(missing_camera_bounds))
        )

    if not state_left_times or not state_right_times or not action_left_times or not action_right_times:
        raise SkippedBag("Bag is missing required state/action messages.")

    return LowDimStreams(
        camera_bounds={
            topic: (int(bounds[0]), int(bounds[1]))  # type: ignore[arg-type]
            for topic, bounds in camera_bounds.items()
        },
        state_left_times=np.asarray(state_left_times, dtype=np.int64),
        state_left_values=np.asarray(state_left_values, dtype=np.float32),
        state_right_times=np.asarray(state_right_times, dtype=np.int64),
        state_right_values=np.asarray(state_right_values, dtype=np.float32),
        action_left_times=np.asarray(action_left_times, dtype=np.int64),
        action_left_values=np.asarray(action_left_values, dtype=np.float32),
        action_right_times=np.asarray(action_right_times, dtype=np.int64),
        action_right_values=np.asarray(action_right_values, dtype=np.float32),
        odom_times=np.asarray(odom_times, dtype=np.int64),
        odom_values=np.asarray(odom_values, dtype=np.float32),
    )


def build_sample_times(streams: LowDimStreams, config: ConversionConfig) -> np.ndarray:
    starts = [
        streams.camera_bounds[config.topics.camera_top_topic][0],
        streams.camera_bounds[config.topics.camera_left_topic][0],
        streams.camera_bounds[config.topics.camera_right_topic][0],
        int(streams.state_left_times[0]),
        int(streams.state_right_times[0]),
        int(streams.action_left_times[0]),
        int(streams.action_right_times[0]),
    ]
    ends = [
        streams.camera_bounds[config.topics.camera_top_topic][1],
        streams.camera_bounds[config.topics.camera_left_topic][1],
        streams.camera_bounds[config.topics.camera_right_topic][1],
        int(streams.state_left_times[-1]),
        int(streams.state_right_times[-1]),
        int(streams.action_left_times[-1]),
        int(streams.action_right_times[-1]),
    ]
    if streams.odom_times.size > 0:
        starts.append(int(streams.odom_times[0]))
        ends.append(int(streams.odom_times[-1]))

    t_start = max(starts)
    t_end = min(ends)
    if t_end <= t_start:
        raise SkippedBag(
            f"Bag has no overlapping time window across required topics. start={t_start}, end={t_end}"
        )

    duration_s = (t_end - t_start) / 1e9
    frame_count = int(duration_s * config.fps)
    if frame_count < 2:
        raise SkippedBag(
            f"Bag is too short after alignment ({duration_s:.3f}s, {frame_count} frames)."
        )
    return np.linspace(t_start, t_end, frame_count, dtype=np.int64)


def open_camera_cursor(
    *,
    stack: ExitStack,
    bag_path: Path,
    AnyReader,
    topic: str,
    video_key: str,
) -> CameraCursor:
    reader = stack.enter_context(AnyReader([bag_path]))
    connection_map = build_connection_map(reader, [topic])
    connection = connection_map.get(topic)
    if connection is None:
        raise SkippedBag(f"Bag is missing camera topic `{topic}`.")
    iterator = reader.messages(connections=[connection])
    cursor = CameraCursor(
        topic=topic,
        video_key=video_key,
        reader=reader,
        connection=connection,
        iterator=iterator,
    )
    advance_camera_cursor(cursor)
    if cursor.curr_timestamp is None:
        raise SkippedBag(f"Bag camera topic `{topic}` does not contain any messages.")
    return cursor


def advance_camera_cursor(cursor: CameraCursor) -> bool:
    try:
        _, timestamp, raw = next(cursor.iterator)
    except StopIteration:
        cursor.curr_timestamp = None
        cursor.curr_raw = None
        return False
    cursor.curr_timestamp = int(timestamp)
    cursor.curr_raw = raw
    return True


def select_camera_message(cursor: CameraCursor, sample_time: int) -> tuple[int, Any]:
    while cursor.curr_timestamp is not None and cursor.curr_timestamp < sample_time:
        cursor.prev_timestamp = cursor.curr_timestamp
        cursor.prev_raw = cursor.curr_raw
        advance_camera_cursor(cursor)

    if cursor.prev_timestamp is None:
        if cursor.curr_timestamp is None or cursor.curr_raw is None:
            raise RuntimeError(f"Camera topic `{cursor.topic}` has no message near {sample_time}.")
        return cursor.curr_timestamp, cursor.curr_raw

    if cursor.curr_timestamp is None or cursor.curr_raw is None:
        if cursor.prev_raw is None:
            raise RuntimeError(f"Camera topic `{cursor.topic}` exhausted unexpectedly.")
        return cursor.prev_timestamp, cursor.prev_raw

    if abs(cursor.curr_timestamp - sample_time) < abs(sample_time - cursor.prev_timestamp):
        return cursor.curr_timestamp, cursor.curr_raw
    if cursor.prev_raw is None:
        raise RuntimeError(f"Camera topic `{cursor.topic}` lost previous frame cache.")
    return cursor.prev_timestamp, cursor.prev_raw


def decode_camera_frame_for_sample(
    *,
    cursor: CameraCursor,
    sample_time: int,
    cv2,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    chosen_timestamp, chosen_raw = select_camera_message(cursor, sample_time)
    if cursor.cached_timestamp != chosen_timestamp:
        msg = cursor.reader.deserialize(chosen_raw, cursor.connection.msgtype)
        cursor.cached_frame = decode_compressed_image(
            cv2,
            msg,
            image_width=image_width,
            image_height=image_height,
        )
        cursor.cached_timestamp = chosen_timestamp
    if cursor.cached_frame is None:
        raise RuntimeError(f"Camera topic `{cursor.topic}` failed to decode frame cache.")
    return cursor.cached_frame


def stage_episode_table(
    *,
    pa,
    pq,
    output_path: Path,
    state_array: np.ndarray,
    action_array: np.ndarray,
    reward_array: np.ndarray,
    done_array: np.ndarray,
    success_array: np.ndarray,
    timestamp_array: np.ndarray,
    frame_index_array: np.ndarray,
) -> None:
    table = pa.Table.from_arrays(
        [
            fixed_size_float_list_array(pa, state_array),
            fixed_size_float_list_array(pa, action_array),
            pa.array(reward_array, type=pa.float32()),
            pa.array(done_array, type=pa.bool_()),
            pa.array(success_array, type=pa.bool_()),
            pa.array(timestamp_array, type=pa.float32()),
            pa.array(frame_index_array, type=pa.int64()),
        ],
        names=[
            "observation.state",
            "action",
            "next.reward",
            "next.done",
            "next.success",
            "timestamp",
            "frame_index",
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")


def convert_one_bag(task: WorkerTask) -> EpisodeArtifact:
    start_time = time.time()
    bag_path = Path(task.bag_path)
    staging_root = Path(task.staging_root)
    staging_dir = staging_root / f"bag-{task.bag_index:06d}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        cv2, AnyReader = require_rosbag_decode_dependencies()
        pa, pq = require_output_dependencies()
        require_binary("ffmpeg")

        streams = collect_low_dim_streams(bag_path, task.config)
        sample_times = build_sample_times(streams, task.config)
        frame_count = int(sample_times.shape[0])
        state_dim = 3 + task.config.arm_dof * 2

        base_velocity = np.zeros((frame_count, 3), dtype=np.float32)
        if streams.odom_times.size > 0:
            odom_indices = nearest_indices(streams.odom_times, sample_times)
            base_velocity = streams.odom_values[odom_indices]

        state_left_indices = nearest_indices(streams.state_left_times, sample_times)
        state_right_indices = nearest_indices(streams.state_right_times, sample_times)
        action_left_indices = nearest_indices(streams.action_left_times, sample_times)
        action_right_indices = nearest_indices(streams.action_right_times, sample_times)

        state_array = np.empty((frame_count, state_dim), dtype=np.float32)
        action_array = np.empty((frame_count, state_dim), dtype=np.float32)
        split = 3 + task.config.arm_dof
        state_array[:, :3] = base_velocity
        state_array[:, 3:split] = streams.state_left_values[state_left_indices]
        state_array[:, split:] = streams.state_right_values[state_right_indices]
        action_array[:, :3] = base_velocity
        action_array[:, 3:split] = streams.action_left_values[action_left_indices]
        action_array[:, split:] = streams.action_right_values[action_right_indices]

        reward_array = np.zeros((frame_count,), dtype=np.float32)
        reward_array[-1] = 1.0
        done_array = np.zeros((frame_count,), dtype=bool)
        done_array[-1] = True
        success_array = done_array.copy()
        timestamp_array = np.arange(frame_count, dtype=np.float32) / float(task.config.fps)
        frame_index_array = np.arange(frame_count, dtype=np.int64)

        state_stats = init_vector_stats_accumulator(state_dim)
        action_stats = init_vector_stats_accumulator(state_dim)
        reward_stats = init_vector_stats_accumulator(1)
        timestamp_stats = init_vector_stats_accumulator(1)
        update_vector_stats_accumulator(state_stats, state_array)
        update_vector_stats_accumulator(action_stats, action_array)
        update_vector_stats_accumulator(reward_stats, reward_array)
        update_vector_stats_accumulator(timestamp_stats, timestamp_array)

        image_stats_by_key = {
            video_key: init_image_stats_accumulator(num_channels=3)
            for _, video_key in task.config.topics.camera_topic_pairs()
        }
        video_paths: dict[str, str] = {}

        with ExitStack() as stack:
            camera_cursors = [
                open_camera_cursor(
                    stack=stack,
                    bag_path=bag_path,
                    AnyReader=AnyReader,
                    topic=topic,
                    video_key=video_key,
                )
                for topic, video_key in task.config.topics.camera_topic_pairs()
            ]
            writers: dict[str, subprocess.Popen] = {}
            for cursor in camera_cursors:
                video_path = staging_dir / safe_video_key_filename(cursor.video_key)
                video_paths[cursor.video_key] = str(video_path)
                writers[cursor.video_key] = start_ffmpeg_raw_writer(
                    video_path,
                    width=task.config.image_width,
                    height=task.config.image_height,
                    fps=task.config.fps,
                    video_encoder=task.video_encoder,
                )

            try:
                for sample_time in sample_times:
                    for cursor in camera_cursors:
                        frame = decode_camera_frame_for_sample(
                            cursor=cursor,
                            sample_time=int(sample_time),
                            cv2=cv2,
                            image_width=task.config.image_width,
                            image_height=task.config.image_height,
                        )
                        writer = writers[cursor.video_key]
                        if writer.stdin is None:
                            raise RuntimeError(
                                f"ffmpeg stdin closed unexpectedly for {cursor.video_key}."
                            )
                        writer.stdin.write(frame.tobytes())
                        update_image_stats_accumulator(
                            image_stats_by_key[cursor.video_key],
                            frame,
                        )
            finally:
                for video_key, writer in writers.items():
                    if writer.stdin is not None and not writer.stdin.closed:
                        writer.stdin.close()
                    return_code = writer.wait()
                    if return_code != 0:
                        raise RuntimeError(
                            f"ffmpeg failed while encoding {video_key} for {bag_path.name} "
                            f"with exit code {return_code}."
                        )

        data_path = staging_dir / "episode.parquet"
        stage_episode_table(
            pa=pa,
            pq=pq,
            output_path=data_path,
            state_array=state_array,
            action_array=action_array,
            reward_array=reward_array,
            done_array=done_array,
            success_array=success_array,
            timestamp_array=timestamp_array,
            frame_index_array=frame_index_array,
        )

        elapsed = time.time() - start_time
        return EpisodeArtifact(
            bag_index=task.bag_index,
            bag_path=str(bag_path),
            status="ready",
            frame_count=frame_count,
            elapsed_s=elapsed,
            staging_dir=str(staging_dir),
            data_path=str(data_path),
            video_paths=video_paths,
            state_stats=state_stats,
            action_stats=action_stats,
            reward_stats=reward_stats,
            timestamp_stats=timestamp_stats,
            image_stats_by_key=image_stats_by_key,
        )
    except SkippedBag as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        return EpisodeArtifact(
            bag_index=task.bag_index,
            bag_path=str(bag_path),
            status="skipped",
            frame_count=0,
            elapsed_s=time.time() - start_time,
            detail=str(exc),
        )
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        return EpisodeArtifact(
            bag_index=task.bag_index,
            bag_path=str(bag_path),
            status="error",
            frame_count=0,
            elapsed_s=time.time() - start_time,
            detail=f"{type(exc).__name__}: {exc}",
        )


def finalize_episode_artifact(
    *,
    pa,
    pq,
    artifact: EpisodeArtifact,
    output_root: Path,
    final_episode_index: int,
    global_frame_offset: int,
    fps: int,
    task_label: str,
) -> tuple[int, dict[str, Any], dict[str, Path]]:
    if artifact.data_path is None or artifact.video_paths is None or artifact.staging_dir is None:
        raise RuntimeError("Episode artifact is incomplete and cannot be finalized.")

    episode_table = pq.read_table(artifact.data_path)
    frame_count = int(episode_table.num_rows)
    if frame_count != artifact.frame_count:
        raise RuntimeError(
            "Episode parquet row count does not match artifact metadata. "
            f"rows={frame_count}, artifact.frame_count={artifact.frame_count}"
        )

    final_table = episode_table
    final_table = final_table.append_column(
        "episode_index",
        pa.array([final_episode_index] * frame_count, type=pa.int64()),
    )
    final_table = final_table.append_column(
        "index",
        pa.array(
            list(range(global_frame_offset, global_frame_offset + frame_count)),
            type=pa.int64(),
        ),
    )
    final_table = final_table.append_column(
        "task_index",
        pa.array([0] * frame_count, type=pa.int64()),
    )

    data_output_path = output_root / DEFAULT_DATA_PATH.format(
        chunk_index=0,
        file_index=final_episode_index,
    )
    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(final_table, data_output_path, compression="snappy")

    moved_video_paths: dict[str, Path] = {}
    for video_key, staging_video_path_text in artifact.video_paths.items():
        staging_video_path = Path(staging_video_path_text)
        final_video_path = output_root / DEFAULT_VIDEO_PATH.format(
            video_key=video_key,
            chunk_index=0,
            file_index=final_episode_index,
        )
        final_video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_video_path), str(final_video_path))
        moved_video_paths[video_key] = final_video_path

    episode_metadata: dict[str, Any] = {
        "episode_index": int(final_episode_index),
        "tasks": [task_label],
        "length": int(frame_count),
        "data/chunk_index": 0,
        "data/file_index": int(final_episode_index),
        "dataset_from_index": int(global_frame_offset),
        "dataset_to_index": int(global_frame_offset + frame_count),
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": 0,
    }
    for video_key in moved_video_paths:
        episode_metadata[f"videos/{video_key}/chunk_index"] = 0
        episode_metadata[f"videos/{video_key}/file_index"] = int(final_episode_index)
        episode_metadata[f"videos/{video_key}/from_timestamp"] = float(0.0)
        episode_metadata[f"videos/{video_key}/to_timestamp"] = float(frame_count / fps)

    shutil.rmtree(Path(artifact.staging_dir), ignore_errors=True)
    return frame_count, episode_metadata, moved_video_paths


def directory_size_in_mb(paths: list[Path]) -> int:
    total_bytes = 0
    for path in paths:
        if path.exists():
            total_bytes += int(path.stat().st_size)
    if total_bytes <= 0:
        return 0
    return int(math.ceil(total_bytes / (1024 * 1024)))


def build_state_feature_names(arm_dof: int) -> list[str]:
    return (
        ["base_vx", "base_vy", "base_omega"]
        + [f"left_joint_{index}" for index in range(arm_dof)]
        + [f"right_joint_{index}" for index in range(arm_dof)]
    )


def convert_dataset(
    *,
    source_path: Path,
    dataset_id: str,
    output_root: Path,
    task_label: str,
    robot_type: str,
    fps: int,
    image_width: int,
    image_height: int,
    arm_dof: int,
    bag_glob: str,
    workers: int | str,
    video_encoder: str,
    overwrite: bool,
    topics: RosbagTopics,
) -> Path:
    require_binary("ffmpeg")
    require_binary("ffprobe")
    require_output_dependencies()
    require_rosbag_decode_dependencies()

    source_path = source_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    bag_files = collect_bag_files(source_path, bag_glob)
    resolved_video_encoder, encoder_notice = resolve_video_encoder(video_encoder)
    resolved_workers, worker_notice = resolve_worker_count(
        workers,
        total_bags=len(bag_files),
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
    staging_root = output_root / DEFAULT_STAGING_DIRNAME
    staging_root.mkdir(parents=True, exist_ok=True)

    config = ConversionConfig(
        dataset_id=dataset_id,
        task_label=task_label,
        robot_type=robot_type,
        fps=fps,
        image_width=image_width,
        image_height=image_height,
        arm_dof=arm_dof,
        topics=topics,
    )
    video_keys = [video_key for _, video_key in topics.camera_topic_pairs()]
    state_dim = 3 + arm_dof * 2

    pa, pq = require_output_dependencies()
    tqdm_cls = maybe_import_tqdm()

    print(
        "Converting ROS bag dataset: "
        f"source={source_path}, bags={len(bag_files)}, dataset_id={dataset_id}, "
        f"output={output_root}, fps={fps}, encoder={resolved_video_encoder}"
    )
    if encoder_notice:
        print(f"[info] {encoder_notice}")
    if worker_notice:
        print(f"[info] {worker_notice}")

    task_index_by_text = {task_label: 0}
    episodes_meta: list[dict[str, Any]] = []
    sample_video_path_by_key: dict[str, Path] = {}
    finalized_data_paths: list[Path] = []
    finalized_video_paths: list[Path] = []
    global_frame_index = 0
    saved_episodes = 0
    skipped_episodes = 0
    error_episodes = 0
    error_messages: list[str] = []

    state_stats = init_vector_stats_accumulator(state_dim)
    action_stats = init_vector_stats_accumulator(state_dim)
    reward_stats = init_vector_stats_accumulator(1)
    timestamp_stats = init_vector_stats_accumulator(1)
    image_stats_by_key = {
        video_key: init_image_stats_accumulator(num_channels=3) for video_key in video_keys
    }

    overall_progress = (
        None
        if tqdm_cls is None
        else tqdm_cls(
            total=len(bag_files),
            desc="bags",
            leave=True,
            dynamic_ncols=True,
        )
    )

    pending_results: dict[int, EpisodeArtifact] = {}
    next_bag_index_to_flush = 0
    worker_tasks = [
        WorkerTask(
            bag_index=bag_index,
            bag_path=str(bag_path),
            staging_root=str(staging_root),
            config=config,
            video_encoder=resolved_video_encoder,
        )
        for bag_index, bag_path in enumerate(bag_files)
    ]

    def update_progress() -> None:
        if overall_progress is not None:
            overall_progress.set_postfix(
                saved=saved_episodes,
                skipped=skipped_episodes,
                errors=error_episodes,
                frames=global_frame_index,
            )

    try:
        with cf.ProcessPoolExecutor(
            max_workers=resolved_workers,
            mp_context=mp.get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            future_to_task = {
                executor.submit(convert_one_bag, task): task for task in worker_tasks
            }
            for future in cf.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    artifact = future.result()
                except Exception as exc:
                    artifact = EpisodeArtifact(
                        bag_index=task.bag_index,
                        bag_path=task.bag_path,
                        status="error",
                        frame_count=0,
                        elapsed_s=0.0,
                        detail=f"{type(exc).__name__}: {exc}",
                    )

                pending_results[artifact.bag_index] = artifact

                while next_bag_index_to_flush in pending_results:
                    ready_artifact = pending_results.pop(next_bag_index_to_flush)
                    if ready_artifact.status == "ready":
                        if (
                            ready_artifact.state_stats is None
                            or ready_artifact.action_stats is None
                            or ready_artifact.reward_stats is None
                            or ready_artifact.timestamp_stats is None
                            or ready_artifact.image_stats_by_key is None
                        ):
                            raise RuntimeError(
                                f"Ready artifact for {ready_artifact.bag_path} is missing stats."
                            )

                        frame_count, episode_metadata, moved_video_paths = finalize_episode_artifact(
                            pa=pa,
                            pq=pq,
                            artifact=ready_artifact,
                            output_root=output_root,
                            final_episode_index=saved_episodes,
                            global_frame_offset=global_frame_index,
                            fps=fps,
                            task_label=task_label,
                        )
                        episodes_meta.append(episode_metadata)
                        merge_vector_stats_accumulator(state_stats, ready_artifact.state_stats)
                        merge_vector_stats_accumulator(action_stats, ready_artifact.action_stats)
                        merge_vector_stats_accumulator(reward_stats, ready_artifact.reward_stats)
                        merge_vector_stats_accumulator(timestamp_stats, ready_artifact.timestamp_stats)
                        for video_key in video_keys:
                            merge_image_stats_accumulator(
                                image_stats_by_key[video_key],
                                ready_artifact.image_stats_by_key[video_key],
                            )
                            sample_video_path_by_key.setdefault(video_key, moved_video_paths[video_key])
                            finalized_video_paths.append(moved_video_paths[video_key])
                        finalized_data_paths.append(
                            output_root
                            / DEFAULT_DATA_PATH.format(
                                chunk_index=0,
                                file_index=saved_episodes,
                            )
                        )
                        saved_episodes += 1
                        global_frame_index += frame_count
                    elif ready_artifact.status == "skipped":
                        skipped_episodes += 1
                        if ready_artifact.detail:
                            error_messages.append(
                                f"[skip] {Path(ready_artifact.bag_path).name}: {ready_artifact.detail}"
                            )
                    else:
                        error_episodes += 1
                        if ready_artifact.detail:
                            error_messages.append(
                                f"[error] {Path(ready_artifact.bag_path).name}: {ready_artifact.detail}"
                            )
                        if ready_artifact.staging_dir:
                            shutil.rmtree(Path(ready_artifact.staging_dir), ignore_errors=True)

                    next_bag_index_to_flush += 1
                    if overall_progress is not None:
                        overall_progress.update(1)
                    update_progress()
    finally:
        if overall_progress is not None:
            overall_progress.close()

    shutil.rmtree(staging_root, ignore_errors=True)

    if saved_episodes <= 0:
        issue_summary = "\n".join(error_messages[:10])
        raise RuntimeError(
            "No episodes were converted successfully."
            + (f"\nFirst issues:\n{issue_summary}" if issue_summary else "")
        )

    validate_episode_ranges(total_frames=global_frame_index, episodes_meta=episodes_meta)
    write_tasks_tables(pa, pq, output_root=output_root, task_index_by_text=task_index_by_text)
    write_episodes_table(pa, pq, output_root=output_root, episodes_meta=episodes_meta, video_keys=video_keys)

    sample_video_info_by_key = {
        video_key: ffprobe_video(sample_video_path_by_key[video_key])
        for video_key in video_keys
    }

    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "source_format": "rosbag",
        "source_path": str(source_path),
        "dataset_id": dataset_id,
        "task_label": task_label,
        "total_episodes": int(saved_episodes),
        "total_frames": int(global_frame_index),
        "total_tasks": int(len(task_index_by_text)),
        "chunks_size": int(max(saved_episodes, 1)),
        "data_files_size_in_mb": directory_size_in_mb(finalized_data_paths),
        "video_files_size_in_mb": directory_size_in_mb(finalized_video_paths),
        "fps": int(fps),
        "video_encoder": resolved_video_encoder,
        "conversion_workers": int(resolved_workers),
        "splits": {"train": f"0:{saved_episodes}"},
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features": {
            "observation.state": vector_feature(
                "float32",
                [state_dim],
                build_state_feature_names(arm_dof),
                fps=fps,
            ),
            "action": vector_feature(
                "float32",
                [state_dim],
                build_state_feature_names(arm_dof),
                fps=fps,
            ),
            "next.reward": scalar_feature("float32", fps=fps),
            "next.done": scalar_feature("bool", fps=fps),
            "next.success": scalar_feature("bool", fps=fps),
            "timestamp": scalar_feature("float32"),
            "frame_index": scalar_feature("int64"),
            "episode_index": scalar_feature("int64"),
            "index": scalar_feature("int64"),
            "task_index": scalar_feature("int64"),
        },
        "conversion_summary": {
            "input_bags": int(len(bag_files)),
            "saved_episodes": int(saved_episodes),
            "skipped_episodes": int(skipped_episodes),
            "error_episodes": int(error_episodes),
        },
    }
    for video_key in video_keys:
        info["features"][video_key] = video_feature(
            video_info=sample_video_info_by_key[video_key],
            fps=fps,
        )

    stats = {
        "observation.state": finalize_vector_stats(state_stats),
        "action": finalize_vector_stats(action_stats),
        "next.reward": finalize_vector_stats(reward_stats),
        "timestamp": finalize_vector_stats(timestamp_stats),
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

    if error_messages:
        (meta_dir / "conversion_issues.log").write_text(
            "\n".join(error_messages) + "\n",
            encoding="utf-8",
        )

    print(
        "Generated LeRobot v3 dataset at: "
        f"{output_root.resolve()} "
        f"(episodes={saved_episodes}, frames={global_frame_index}, "
        f"skipped={skipped_episodes}, errors={error_episodes})"
    )
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ROS `.bag` files into a local LeRobot v3 dataset."
    )
    parser.add_argument(
        "source_path",
        type=Path,
        help="Path to a `.bag` file or a directory containing bag files.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Logical dataset id written into LeRobot metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to `main/data/<dataset-id>`.",
    )
    parser.add_argument(
        "--task-label",
        type=str,
        default="pick",
        help="Task text written into the dataset task table.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="zeno",
        help="Robot type written into LeRobot metadata.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Target FPS for resampling and encoded videos.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Target image width for encoded videos.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help="Target image height for encoded videos.",
    )
    parser.add_argument(
        "--arm-dof",
        type=int,
        default=DEFAULT_ARM_DOF,
        help="Arm degrees of freedom used for left/right state and action vectors.",
    )
    parser.add_argument(
        "--bag-glob",
        type=str,
        default="*.bag",
        help="Glob used when `source_path` is a directory.",
    )
    parser.add_argument(
        "--workers",
        type=str,
        default=DEFAULT_WORKERS,
        help='Parallel worker count. Pass a positive integer or "auto".',
    )
    parser.add_argument(
        "--video-encoder",
        type=str,
        default=DEFAULT_VIDEO_ENCODER,
        help='ffmpeg encoder. Use "auto", "gpu", "cpu", or an exact encoder name.',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    parser.add_argument(
        "--camera-top-topic",
        type=str,
        default=DEFAULT_CAMERA_TOP_TOPIC,
        help="ROS topic for the top camera compressed image stream.",
    )
    parser.add_argument(
        "--camera-left-topic",
        type=str,
        default=DEFAULT_CAMERA_LEFT_TOPIC,
        help="ROS topic for the left camera compressed image stream.",
    )
    parser.add_argument(
        "--camera-right-topic",
        type=str,
        default=DEFAULT_CAMERA_RIGHT_TOPIC,
        help="ROS topic for the right camera compressed image stream.",
    )
    parser.add_argument(
        "--state-left-topic",
        type=str,
        default=DEFAULT_STATE_LEFT_TOPIC,
        help="ROS topic for the left arm state stream.",
    )
    parser.add_argument(
        "--state-right-topic",
        type=str,
        default=DEFAULT_STATE_RIGHT_TOPIC,
        help="ROS topic for the right arm state stream.",
    )
    parser.add_argument(
        "--action-left-topic",
        type=str,
        default=DEFAULT_ACTION_LEFT_TOPIC,
        help="ROS topic for the left arm teleop action stream.",
    )
    parser.add_argument(
        "--action-right-topic",
        type=str,
        default=DEFAULT_ACTION_RIGHT_TOPIC,
        help="ROS topic for the right arm teleop action stream.",
    )
    parser.add_argument(
        "--odom-topic",
        type=str,
        default=DEFAULT_ODOM_TOPIC,
        help='ROS topic for base odometry. Pass "none" to disable odom ingestion.',
    )
    return parser


def parse_workers_arg(value: str) -> int | str:
    text = str(value).strip().lower()
    if text == DEFAULT_WORKERS:
        return text
    parsed = int(text)
    if parsed <= 0:
        raise ValueError("`--workers` must be a positive integer or `auto`.")
    return parsed


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / args.dataset_id.replace("/", "_")
    )

    topics = RosbagTopics(
        camera_top_topic=str(args.camera_top_topic),
        camera_left_topic=str(args.camera_left_topic),
        camera_right_topic=str(args.camera_right_topic),
        state_left_topic=str(args.state_left_topic),
        state_right_topic=str(args.state_right_topic),
        action_left_topic=str(args.action_left_topic),
        action_right_topic=str(args.action_right_topic),
        odom_topic=normalize_optional_topic(args.odom_topic),
    )

    convert_dataset(
        source_path=args.source_path,
        dataset_id=str(args.dataset_id),
        output_root=output_dir,
        task_label=str(args.task_label),
        robot_type=str(args.robot_type),
        fps=int(args.fps),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        arm_dof=int(args.arm_dof),
        bag_glob=str(args.bag_glob),
        workers=parse_workers_arg(args.workers),
        video_encoder=str(args.video_encoder),
        overwrite=bool(args.overwrite),
        topics=topics,
    )


if __name__ == "__main__":
    main()
