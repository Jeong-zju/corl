#!/usr/bin/env python3
"""Config-driven ROS bag to LeRobot v3.0 dataset converter.

This script generalizes the behavior of `convert_hanger_stage3.py` by moving
dataset parameters, image topic mappings, and vector feature composition into a
YAML config file.

Example:
    python data/convert_rosbag_to_lerobot.py \
        --config data/configs/zeno_conversion.yaml
"""

from __future__ import annotations

import argparse
import gc
import shutil
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ImageTopicSpec:
    name: str
    topic: str
    key: str


@dataclass(frozen=True, slots=True)
class VectorExtractorSpec:
    type: str
    field: str | None
    fields: tuple[str, ...]
    length: int | None
    pad_value: float
    default: tuple[float, ...] | None
    names: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class VectorSourceSpec:
    name: str
    topic: str
    required: bool
    sync: bool
    extractor: VectorExtractorSpec


@dataclass(frozen=True, slots=True)
class VectorFeatureSpec:
    key: str
    dtype: str
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConverterConfig:
    path: Path
    data_root: Path
    output_dir: Path
    output_name: str
    task_label: str | None
    robot_type: str
    bag_glob: str
    overwrite_output: bool
    fps: int
    n_workers: int
    image_width: int
    image_height: int
    image_color_order: str
    use_videos: bool
    vcodec: str
    image_writer_threads: int
    image_writer_processes: int
    images: tuple[ImageTopicSpec, ...]
    vector_sources: dict[str, VectorSourceSpec]
    vector_features: tuple[VectorFeatureSpec, ...]

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.output_name


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(data).__name__}.")
    return data


def resolve_path(
    raw_path: str | None,
    *,
    field_name: str,
    project_root: Path = PROJECT_ROOT,
    must_exist: bool = False,
) -> Path | None:
    if raw_path is None:
        return None

    text = str(raw_path).strip()
    if not text:
        return None

    candidate = Path(text)
    if candidate.is_absolute():
        raise ValueError(
            f"`{field_name}` must be a path relative to the project root "
            f"({project_root}), got absolute path {raw_path!r}."
        )

    resolved = (project_root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"`{field_name}` must stay within the project root ({project_root}), "
            f"got {raw_path!r} -> {resolved}."
        ) from exc

    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Could not resolve an existing path for `{field_name}` from project root "
            f"{project_root}: {raw_path!r} -> {resolved}"
        )
    return resolved


def _as_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping for `{key}`, got {type(value).__name__}.")
    return value


def _as_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Expected list for `{key}`, got {type(value).__name__}.")
    return value


def _normalize_names(raw: Any, *, source_name: str, expected_length: int) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError(
            f"Expected `names` for vector source `{source_name}` to be a list, "
            f"got {type(raw).__name__}."
        )
    names = tuple(str(item) for item in raw)
    if len(names) != expected_length:
        raise ValueError(
            f"Vector source `{source_name}` declares {len(names)} names, "
            f"but extractor output length is {expected_length}."
        )
    return names


def _parse_image_specs(raw: list[Any]) -> tuple[ImageTopicSpec, ...]:
    specs: list[ImageTopicSpec] = []
    seen_keys: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected image item #{index} to be a mapping, got {type(item).__name__}."
            )
        name = str(item.get("name", f"image_{index}"))
        topic = str(item.get("topic", "")).strip()
        key = str(item.get("key", "")).strip()
        if not topic:
            raise ValueError(f"Image item `{name}` is missing `topic`.")
        if not key:
            raise ValueError(f"Image item `{name}` is missing `key`.")
        if key in seen_keys:
            raise ValueError(f"Duplicate image feature key: {key}")
        seen_keys.add(key)
        specs.append(ImageTopicSpec(name=name, topic=topic, key=key))
    if not specs:
        raise ValueError("Config must declare at least one image topic in `topics.images`.")
    return tuple(specs)


def _parse_vector_source(name: str, raw: Any) -> VectorSourceSpec:
    if not isinstance(raw, dict):
        raise TypeError(
            f"Expected vector source `{name}` to be a mapping, got {type(raw).__name__}."
        )
    topic = str(raw.get("topic", "")).strip()
    if not topic:
        raise ValueError(f"Vector source `{name}` is missing `topic`.")

    extractor_raw = raw.get("extractor")
    if not isinstance(extractor_raw, dict):
        raise TypeError(f"Vector source `{name}` is missing a mapping `extractor` section.")

    extractor_type = str(extractor_raw.get("type", "")).strip()
    if extractor_type not in {"fields", "sequence_field"}:
        raise ValueError(
            f"Vector source `{name}` has unsupported extractor type `{extractor_type}`. "
            "Supported types: `fields`, `sequence_field`."
        )

    if extractor_type == "fields":
        fields_raw = extractor_raw.get("fields", [])
        if not isinstance(fields_raw, list) or not fields_raw:
            raise ValueError(f"Vector source `{name}` with `fields` extractor must define `fields`.")
        fields = tuple(str(field) for field in fields_raw)
        length = len(fields)
        field = None
    else:
        field = str(extractor_raw.get("field", "")).strip()
        length_raw = extractor_raw.get("length")
        if not field:
            raise ValueError(
                f"Vector source `{name}` with `sequence_field` extractor must define `field`."
            )
        if length_raw is None:
            raise ValueError(
                f"Vector source `{name}` with `sequence_field` extractor must define `length`."
            )
        length = int(length_raw)
        if length <= 0:
            raise ValueError(f"Vector source `{name}` has invalid `length={length}`.")
        fields = ()

    default_raw = extractor_raw.get("default")
    default: tuple[float, ...] | None = None
    if default_raw is not None:
        if not isinstance(default_raw, list):
            raise TypeError(
                f"Expected `default` for vector source `{name}` to be a list, "
                f"got {type(default_raw).__name__}."
            )
        default = tuple(float(item) for item in default_raw)
        if len(default) != length:
            raise ValueError(
                f"Vector source `{name}` default length is {len(default)}, expected {length}."
            )

    names = _normalize_names(extractor_raw.get("names"), source_name=name, expected_length=length)

    return VectorSourceSpec(
        name=name,
        topic=topic,
        required=bool(raw.get("required", True)),
        sync=bool(raw.get("sync", raw.get("required", True))),
        extractor=VectorExtractorSpec(
            type=extractor_type,
            field=field,
            fields=fields,
            length=length,
            pad_value=float(extractor_raw.get("pad_value", 0.0)),
            default=default,
            names=names,
        ),
    )


def _parse_vector_features(
    raw: list[Any],
    *,
    vector_sources: dict[str, VectorSourceSpec],
) -> tuple[VectorFeatureSpec, ...]:
    features: list[VectorFeatureSpec] = []
    seen_keys: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected vector feature #{index} to be a mapping, got {type(item).__name__}."
            )
        key = str(item.get("key", "")).strip()
        if not key:
            raise ValueError(f"Vector feature #{index} is missing `key`.")
        if key in seen_keys:
            raise ValueError(f"Duplicate vector feature key: {key}")
        seen_keys.add(key)

        sources_raw = item.get("sources", [])
        if not isinstance(sources_raw, list) or not sources_raw:
            raise ValueError(f"Vector feature `{key}` must define a non-empty `sources` list.")
        sources = tuple(str(name) for name in sources_raw)
        missing = [name for name in sources if name not in vector_sources]
        if missing:
            raise ValueError(
                f"Vector feature `{key}` references unknown vector sources: {missing}"
            )
        features.append(
            VectorFeatureSpec(
                key=key,
                dtype=str(item.get("dtype", "float32")),
                sources=sources,
            )
        )

    if not features:
        raise ValueError("Config must declare at least one vector feature in `features.vectors`.")
    return tuple(features)


def load_converter_config(
    config_path: str | Path,
    *,
    validate_paths: bool = True,
) -> ConverterConfig:
    path = Path(config_path).expanduser().resolve()
    raw = load_yaml_mapping(path)

    dataset_raw = _as_mapping(raw, "dataset")
    processing_raw = _as_mapping(raw, "processing")
    image_raw = _as_mapping(raw, "image")
    topics_raw = _as_mapping(raw, "topics")
    features_raw = _as_mapping(raw, "features")

    data_root = resolve_path(
        str(dataset_raw.get("data_root", "")),
        field_name="dataset.data_root",
        must_exist=validate_paths,
    )
    output_dir = resolve_path(
        str(dataset_raw.get("output_dir", "")),
        field_name="dataset.output_dir",
        must_exist=False,
    )
    if data_root is None:
        raise ValueError("Config is missing `dataset.data_root`.")
    if output_dir is None:
        raise ValueError("Config is missing `dataset.output_dir`.")

    output_name = str(dataset_raw.get("output_name", "")).strip()
    if not output_name:
        raise ValueError("Config is missing `dataset.output_name`.")

    image_color_order = str(image_raw.get("color_order", "rgb")).lower()
    if image_color_order not in {"rgb", "bgr"}:
        raise ValueError(
            f"`image.color_order` must be `rgb` or `bgr`, got {image_color_order!r}."
        )

    images = _parse_image_specs(_as_list(topics_raw, "images"))
    vector_sources_raw = _as_mapping(topics_raw, "vectors")
    vector_sources = {
        str(name): _parse_vector_source(str(name), value)
        for name, value in vector_sources_raw.items()
    }
    vector_features = _parse_vector_features(
        _as_list(features_raw, "vectors"),
        vector_sources=vector_sources,
    )

    return ConverterConfig(
        path=path,
        data_root=data_root,
        output_dir=output_dir,
        output_name=output_name,
        task_label=(
            None
            if dataset_raw.get("task_label") in {None, "", "null"}
            else str(dataset_raw.get("task_label"))
        ),
        robot_type=str(dataset_raw.get("robot_type", "unknown")),
        bag_glob=str(dataset_raw.get("bag_glob", "*.bag")),
        overwrite_output=bool(dataset_raw.get("overwrite_output", False)),
        fps=int(processing_raw.get("fps", 30)),
        n_workers=int(processing_raw.get("n_workers", 1)),
        image_width=int(image_raw.get("width", 224)),
        image_height=int(image_raw.get("height", 224)),
        image_color_order=image_color_order,
        use_videos=bool(image_raw.get("use_videos", True)),
        vcodec=str(image_raw.get("vcodec", "h264")),
        image_writer_threads=int(image_raw.get("image_writer_threads", 4)),
        image_writer_processes=int(image_raw.get("image_writer_processes", 2)),
        images=images,
        vector_sources=vector_sources,
        vector_features=vector_features,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ROS bag files into a LeRobot v3.0 dataset using a YAML config "
            "that defines image mappings and vector feature composition."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the converter YAML config.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Only parse and validate the config, then exit.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Override `dataset.overwrite_output=true` without editing the config.",
    )
    return parser


def nearest_idx(times: np.ndarray, t: np.int64) -> int:
    idx = np.searchsorted(times, t)
    if idx == 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    return idx if abs(times[idx] - t) < abs(t - times[idx - 1]) else idx - 1


def get_attr_by_path(message: Any, path: str) -> Any:
    current = message
    for part in path.split("."):
        if not part:
            raise ValueError(f"Invalid empty segment in field path `{path}`.")
        current = getattr(current, part)
    return current


def decode_compressed_image(msg: Any, *, width: int, height: int, color_order: str) -> np.ndarray | None:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    if color_order == "rgb":
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = img_bgr
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


def extract_vector_from_msg(source: VectorSourceSpec, msg: Any) -> np.ndarray:
    extractor = source.extractor
    if extractor.type == "fields":
        values = [float(get_attr_by_path(msg, field_path)) for field_path in extractor.fields]
        return np.asarray(values, dtype=np.float32)

    sequence = list(get_attr_by_path(msg, extractor.field or ""))
    values = [float(item) for item in sequence[: extractor.length]]
    if len(values) < int(extractor.length or 0):
        values.extend([extractor.pad_value] * (int(extractor.length or 0) - len(values)))
    return np.asarray(values, dtype=np.float32)


def default_vector_value(source: VectorSourceSpec) -> np.ndarray:
    if source.extractor.default is None:
        raise ValueError(
            f"Vector source `{source.name}` has no messages in this bag and no default value."
        )
    return np.asarray(source.extractor.default, dtype=np.float32)


def build_vector_feature_names(feature: VectorFeatureSpec, *, config: ConverterConfig) -> list[str] | None:
    names: list[str] = []
    for source_name in feature.sources:
        source_names = config.vector_sources[source_name].extractor.names
        if source_names is None:
            return None
        names.extend(source_names)
    return names


def build_dataset_features(config: ConverterConfig) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for feature in config.vector_features:
        length = sum(int(config.vector_sources[source].extractor.length or 0) for source in feature.sources)
        feature_dict: dict[str, Any] = {
            "dtype": feature.dtype,
            "shape": (length,),
        }
        names = build_vector_feature_names(feature, config=config)
        if names is not None:
            feature_dict["names"] = names
        features[feature.key] = feature_dict

    for image in config.images:
        features[image.key] = {
            "dtype": "video" if config.use_videos else "image",
            "shape": (config.image_height, config.image_width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _collect_required_topics(config: ConverterConfig) -> dict[str, bool]:
    required_topics: dict[str, bool] = {image.topic: True for image in config.images}
    for source in config.vector_sources.values():
        required_topics[source.topic] = required_topics.get(source.topic, False) or source.required
    return required_topics


def _collect_sync_topics(config: ConverterConfig) -> list[str]:
    sync_topics = [image.topic for image in config.images]
    sync_topics.extend(source.topic for source in config.vector_sources.values() if source.sync)
    seen: set[str] = set()
    ordered: list[str] = []
    for topic in sync_topics:
        if topic in seen:
            continue
        seen.add(topic)
        ordered.append(topic)
    return ordered


def nearest_indices(times: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        raise ValueError("Cannot compute nearest indices for an empty timestamp array.")

    right = np.searchsorted(times, sample_times, side="left")
    left = np.clip(right - 1, 0, len(times) - 1)
    right_clipped = np.clip(right, 0, len(times) - 1)

    choose_right = np.abs(times[right_clipped] - sample_times) < np.abs(sample_times - times[left])
    nearest = np.where(right == 0, 0, np.where(right >= len(times), len(times) - 1, left))
    nearest = np.where((right > 0) & (right < len(times)) & choose_right, right_clipped, nearest)
    return nearest.astype(np.int64, copy=False)


def _capture_message_indices(selected_indices: np.ndarray | None) -> np.ndarray:
    if selected_indices is None:
        return np.empty(0, dtype=np.int64)
    return np.unique(selected_indices)


def _collect_topic_timestamps(
    bag_path: Path,
    *,
    all_topics: set[str],
) -> tuple[dict[str, np.ndarray], bool]:
    try:
        from rosbags.highlevel import AnyReader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency `rosbags`. Please run in an environment with rosbags installed."
        ) from exc

    with AnyReader([bag_path]) as reader:
        topic_to_times: dict[str, list[int]] = {topic: [] for topic in all_topics}
        connections = [conn for conn in reader.connections if conn.topic in all_topics]
        if not connections:
            return {topic: np.empty(0, dtype=np.int64) for topic in all_topics}, False

        for conn, timestamp, _ in reader.messages(connections=connections):
            topic_to_times[conn.topic].append(timestamp)

    return {
        topic: np.asarray(timestamps, dtype=np.int64)
        for topic, timestamps in topic_to_times.items()
    }, True


def _build_sample_times(
    *,
    topic_to_times: dict[str, np.ndarray],
    sync_topics: list[str],
    fps: int,
) -> tuple[np.ndarray | None, float]:
    sync_time_arrays = [topic_to_times[topic] for topic in sync_topics if len(topic_to_times[topic]) > 0]
    if not sync_time_arrays:
        return None, 0.0

    t_start = max(int(times[0]) for times in sync_time_arrays)
    t_end = min(int(times[-1]) for times in sync_time_arrays)
    duration_s = (t_end - t_start) / 1e9
    n_frames = int(duration_s * fps)
    if n_frames < 2:
        return None, duration_s
    return np.linspace(t_start, t_end, n_frames, dtype=np.int64), duration_s


def _build_selected_indices_by_topic(
    *,
    all_topics: set[str],
    topic_to_times: dict[str, np.ndarray],
    sample_times: np.ndarray,
) -> dict[str, np.ndarray | None]:
    selected_indices: dict[str, np.ndarray | None] = {}
    for topic in all_topics:
        times = topic_to_times[topic]
        selected_indices[topic] = None if len(times) == 0 else nearest_indices(times, sample_times)
    return selected_indices


def _ensure_topic_value_for_frame(
    *,
    topic: str,
    frame_idx: int,
    selected_indices_by_topic: dict[str, np.ndarray | None],
    active_topic_indices: dict[str, int],
    active_topic_values: dict[str, Any],
    buffered_topic_values: dict[str, deque[tuple[int, Any]]],
) -> bool:
    selected_indices = selected_indices_by_topic[topic]
    if selected_indices is None:
        return True

    needed_index = int(selected_indices[frame_idx])
    if active_topic_indices.get(topic) == needed_index:
        return True

    buffer = buffered_topic_values[topic]
    while buffer and buffer[0][0] < needed_index:
        buffer.popleft()
    if not buffer or buffer[0][0] != needed_index:
        return False

    active_index, active_value = buffer.popleft()
    active_topic_indices[topic] = active_index
    active_topic_values[topic] = active_value
    return True


_FAILED_FRAME_VALUE = object()


def process_single_bag(
    *,
    bag_path: Path,
    bag_idx: int,
    total_bags: int,
    dataset: Any,
    config: ConverterConfig,
) -> int:
    print(f"[Bag {bag_idx}/{total_bags}] Start: {bag_path.name}", flush=True)
    t0 = time.time()

    try:
        from rosbags.highlevel import AnyReader

        all_topics = {image.topic for image in config.images}
        all_topics.update(source.topic for source in config.vector_sources.values())
        required_topics = _collect_required_topics(config)
        sync_topics = _collect_sync_topics(config)

        topic_to_times, has_connections = _collect_topic_timestamps(bag_path, all_topics=all_topics)
        if not has_connections:
            print(f"[Bag {bag_idx}] No configured topics found, skipping.", flush=True)
            return 0

        for topic, required in required_topics.items():
            if required and len(topic_to_times[topic]) == 0:
                print(f"[Bag {bag_idx}] Missing required topic {topic}, skipping.", flush=True)
                return 0

        sample_times, duration_s = _build_sample_times(
            topic_to_times=topic_to_times,
            sync_topics=sync_topics,
            fps=config.fps,
        )
        if sample_times is None:
            if duration_s == 0.0:
                print(f"[Bag {bag_idx}] No sync topics with messages, skipping.", flush=True)
            else:
                print(f"[Bag {bag_idx}] Too short ({duration_s:.1f}s), skipping.", flush=True)
            return 0

        selected_indices_by_topic = _build_selected_indices_by_topic(
            all_topics=all_topics,
            topic_to_times=topic_to_times,
            sample_times=sample_times,
        )
        capture_indices_by_topic = {
            topic: _capture_message_indices(selected_indices)
            for topic, selected_indices in selected_indices_by_topic.items()
        }

        images_by_topic = {image.topic: image for image in config.images}
        vector_sources_by_topic: dict[str, list[VectorSourceSpec]] = {}
        for source in config.vector_sources.values():
            vector_sources_by_topic.setdefault(source.topic, []).append(source)

        active_topic_indices: dict[str, int] = {}
        active_topic_values: dict[str, Any] = {}
        buffered_topic_values = {
            topic: deque()
            for topic in all_topics
            if len(capture_indices_by_topic[topic]) > 0
        }
        topic_capture_positions = {topic: 0 for topic in all_topics}
        topic_message_indices = {topic: -1 for topic in all_topics}

        default_source_values = {
            source.name: default_vector_value(source)
            for source in config.vector_sources.values()
            if selected_indices_by_topic[source.topic] is None
        }

        frame_idx = 0
        written_frames = 0

        with AnyReader([bag_path]) as reader:
            connections = [conn for conn in reader.connections if conn.topic in all_topics]
            for conn, _, raw in reader.messages(connections=connections):
                topic = conn.topic
                capture_indices = capture_indices_by_topic[topic]
                if len(capture_indices) == 0:
                    continue

                topic_message_indices[topic] += 1
                capture_pos = topic_capture_positions[topic]
                if capture_pos >= len(capture_indices):
                    continue

                capture_index = int(capture_indices[capture_pos])
                if topic_message_indices[topic] != capture_index:
                    continue

                msg = reader.deserialize(raw, conn.msgtype)
                if topic in images_by_topic:
                    prepared_value = decode_compressed_image(
                        msg,
                        width=config.image_width,
                        height=config.image_height,
                        color_order=config.image_color_order,
                    )
                    if prepared_value is None:
                        prepared_value = _FAILED_FRAME_VALUE
                else:
                    prepared_value = {
                        source.name: extract_vector_from_msg(source, msg)
                        for source in vector_sources_by_topic.get(topic, [])
                    }

                buffered_topic_values[topic].append((capture_index, prepared_value))
                topic_capture_positions[topic] = capture_pos + 1

                while frame_idx < len(sample_times):
                    ready = True
                    for topic_name in all_topics:
                        if not _ensure_topic_value_for_frame(
                            topic=topic_name,
                            frame_idx=frame_idx,
                            selected_indices_by_topic=selected_indices_by_topic,
                            active_topic_indices=active_topic_indices,
                            active_topic_values=active_topic_values,
                            buffered_topic_values=buffered_topic_values,
                        ):
                            ready = False
                            break
                    if not ready:
                        break

                    frame: dict[str, Any] = {}
                    failed_frame = False

                    for image in config.images:
                        image_value = active_topic_values.get(image.topic)
                        if image_value is _FAILED_FRAME_VALUE:
                            failed_frame = True
                            break
                        frame[image.key] = image_value

                    if not failed_frame:
                        vector_values: dict[str, np.ndarray] = {}
                        for source_name, source in config.vector_sources.items():
                            if selected_indices_by_topic[source.topic] is None:
                                vector_values[source_name] = default_source_values[source_name]
                                continue
                            topic_value = active_topic_values[source.topic]
                            vector_values[source_name] = topic_value[source_name]

                        for feature in config.vector_features:
                            feature_value = np.concatenate(
                                [vector_values[source_name] for source_name in feature.sources],
                                axis=0,
                            )
                            frame[feature.key] = np.asarray(feature_value, dtype=np.dtype(feature.dtype))

                        if config.task_label is not None:
                            frame["task"] = config.task_label
                        dataset.add_frame(frame)
                        written_frames += 1

                    frame_idx += 1

        if frame_idx < len(sample_times):
            raise RuntimeError(
                f"Incremental conversion stalled after {frame_idx}/{len(sample_times)} planned frames."
            )

        if written_frames == 0:
            print(f"[Bag {bag_idx}] No valid frames after decoding, skipping.", flush=True)
            return 0

        dataset.save_episode()
        elapsed = time.time() - t0
        print(
            f"[Bag {bag_idx}] Done: {written_frames}/{len(sample_times)} frames in {elapsed:.1f}s",
            flush=True,
        )
        return written_frames

    except Exception as exc:
        import traceback

        print(f"[Bag {bag_idx}] Error: {exc}", flush=True)
        traceback.print_exc()
        return 0
    finally:
        gc.collect()


def create_dataset(config: ConverterConfig):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency `lerobot`. Please run in an environment with lerobot installed."
        ) from exc

    return LeRobotDataset.create(
        repo_id=config.output_name,
        root=config.output_path,
        robot_type=config.robot_type,
        fps=config.fps,
        features=build_dataset_features(config),
        use_videos=config.use_videos,
        vcodec=config.vcodec,
        image_writer_threads=config.image_writer_threads,
        image_writer_processes=config.image_writer_processes,
    )


def print_config_summary(config: ConverterConfig) -> None:
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Config:        {config.path}")
    print(f"Data root:     {config.data_root}")
    print(f"Output:        {config.output_path}")
    print(f"Bags glob:     {config.bag_glob}")
    print(f"FPS:           {config.fps}")
    print(f"Workers:       {config.n_workers} configured (incremental mode writes sequentially)")
    print(f"Images:        {len(config.images)}")
    print(f"Vector sources:{len(config.vector_sources)}")
    print(f"Vector feats:  {len(config.vector_features)}")
    print("Image keys:")
    for image in config.images:
        print(f"  - {image.name}: {image.topic} -> {image.key}")
    print("Vector features:")
    for feature in config.vector_features:
        print(f"  - {feature.key}: {' + '.join(feature.sources)}")


def run_conversion(config: ConverterConfig) -> None:
    total_start = time.time()

    if not config.data_root.exists():
        raise FileNotFoundError(f"Configured data root does not exist: {config.data_root}")

    bag_files = sorted(config.data_root.glob(config.bag_glob))
    if not bag_files:
        raise FileNotFoundError(
            f"No bag files matched `{config.bag_glob}` under {config.data_root}"
        )

    output_path = config.output_path
    if output_path.exists():
        if not config.overwrite_output:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. "
                "Set `dataset.overwrite_output: true` or pass `--overwrite-output`."
            )
        print(f"Removing existing output: {output_path}")
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = create_dataset(config)

    total_bags = len(bag_files)
    print(f"\n{'=' * 60}")
    print("Converting ROS bags -> LeRobot v3.0")
    print(f"  Bags:    {total_bags}")
    print("  Mode:    incremental (two-pass per bag, sequential write)")
    print(f"  Workers: {config.n_workers} configured (ignored to keep memory bounded)")
    print(f"  Output:  {output_path}")
    print(f"  Config:  {config.path}")
    print(f"{'=' * 60}\n")

    successful = 0
    for index, bag_path in enumerate(bag_files, start=1):
        written_frames = process_single_bag(
            bag_path=bag_path,
            bag_idx=index,
            total_bags=total_bags,
            dataset=dataset,
            config=config,
        )
        if written_frames <= 0:
            continue
        successful += 1
        elapsed = time.time() - total_start
        rate = successful / elapsed * 60 if elapsed > 0 else 0.0
        eta = (total_bags - successful) / (successful / elapsed) if successful and elapsed > 0 else 0.0
        print(
            f"  [Episode {successful}/{total_bags}] saved | "
            f"{rate:.1f} ep/min | ETA {eta/60:.1f}min",
            flush=True,
        )
        gc.collect()

    dataset.finalize()

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"  Episodes: {successful}/{total_bags}")
    print(f"  Time:     {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  Output:   {output_path}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_converter_config(args.config, validate_paths=not args.validate_config)
    if args.overwrite_output:
        config = replace(config, overwrite_output=True)

    print_config_summary(config)
    if args.validate_config:
        print("\nConfig validation passed.")
        return

    run_conversion(config)


if __name__ == "__main__":
    main()
