from __future__ import annotations

from collections import deque
import importlib.util
from pathlib import Path
import sys
import textwrap

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "data" / "convert_rosbag_to_lerobot.py"
SPEC = importlib.util.spec_from_file_location("convert_rosbag_to_lerobot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def build_config_text(data_root: str, output_dir: str) -> str:
    return textwrap.dedent(
        f"""
        dataset:
          data_root: {data_root}
          output_dir: {output_dir}
          output_name: demo_dataset
          robot_type: demo

        processing:
          fps: 30
          n_workers: 1

        image:
          width: 224
          height: 224
          color_order: rgb

        topics:
          images:
            - name: top
              topic: /camera
              key: observation.images.top
          vectors:
            state:
              topic: /joint_states
              extractor:
                type: sequence_field
                field: position
                length: 2

        features:
          vectors:
            - key: observation.state
              dtype: float32
              sources:
                - state
        """
    ).strip()


def test_load_converter_config_resolves_paths_from_project_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        build_config_text(
            data_root="data/demo/input_bags",
            output_dir="data/demo/output_root",
        ),
        encoding="utf-8",
    )

    config = MODULE.load_converter_config(config_path, validate_paths=False)

    assert config.data_root == MODULE.PROJECT_ROOT / "data" / "demo" / "input_bags"
    assert config.output_dir == MODULE.PROJECT_ROOT / "data" / "demo" / "output_root"


def test_load_converter_config_rejects_absolute_dataset_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        build_config_text(
            data_root="/tmp/input_bags",
            output_dir="data/demo/output_root",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset\\.data_root.*relative to the project root"):
        MODULE.load_converter_config(config_path, validate_paths=False)


def test_nearest_indices_prefers_left_value_on_ties() -> None:
    times = MODULE.np.asarray([10, 20, 40, 80], dtype=MODULE.np.int64)
    sample_times = MODULE.np.asarray([0, 15, 30, 60, 99], dtype=MODULE.np.int64)

    result = MODULE.nearest_indices(times, sample_times)

    assert result.tolist() == [0, 0, 1, 2, 3]


def test_capture_message_indices_deduplicates_frame_plan() -> None:
    selected = MODULE.np.asarray([0, 0, 1, 1, 1, 3], dtype=MODULE.np.int64)

    capture = MODULE._capture_message_indices(selected)

    assert capture.tolist() == [0, 1, 3]


def test_ensure_topic_value_for_frame_reuses_active_and_promotes_buffered_values() -> None:
    selected_indices_by_topic = {
        "/camera": MODULE.np.asarray([0, 0, 1, 1, 2], dtype=MODULE.np.int64),
    }
    active_topic_indices: dict[str, int] = {}
    active_topic_values: dict[str, str] = {}
    buffered_topic_values = {
        "/camera": deque([(0, "frame-a"), (1, "frame-b"), (2, "frame-c")]),
    }

    assert MODULE._ensure_topic_value_for_frame(
        topic="/camera",
        frame_idx=0,
        selected_indices_by_topic=selected_indices_by_topic,
        active_topic_indices=active_topic_indices,
        active_topic_values=active_topic_values,
        buffered_topic_values=buffered_topic_values,
    )
    assert active_topic_indices["/camera"] == 0
    assert active_topic_values["/camera"] == "frame-a"

    assert MODULE._ensure_topic_value_for_frame(
        topic="/camera",
        frame_idx=1,
        selected_indices_by_topic=selected_indices_by_topic,
        active_topic_indices=active_topic_indices,
        active_topic_values=active_topic_values,
        buffered_topic_values=buffered_topic_values,
    )
    assert active_topic_indices["/camera"] == 0

    assert MODULE._ensure_topic_value_for_frame(
        topic="/camera",
        frame_idx=2,
        selected_indices_by_topic=selected_indices_by_topic,
        active_topic_indices=active_topic_indices,
        active_topic_values=active_topic_values,
        buffered_topic_values=buffered_topic_values,
    )
    assert active_topic_indices["/camera"] == 1
    assert active_topic_values["/camera"] == "frame-b"

    assert MODULE._ensure_topic_value_for_frame(
        topic="/camera",
        frame_idx=4,
        selected_indices_by_topic=selected_indices_by_topic,
        active_topic_indices=active_topic_indices,
        active_topic_values=active_topic_values,
        buffered_topic_values=buffered_topic_values,
    )
    assert active_topic_indices["/camera"] == 2
    assert active_topic_values["/camera"] == "frame-c"


def test_ensure_topic_value_for_frame_returns_false_when_future_value_not_buffered() -> None:
    selected_indices_by_topic = {
        "/camera": MODULE.np.asarray([0, 1], dtype=MODULE.np.int64),
    }
    active_topic_indices = {"/camera": 0}
    active_topic_values = {"/camera": "frame-a"}
    buffered_topic_values = {"/camera": deque()}

    ready = MODULE._ensure_topic_value_for_frame(
        topic="/camera",
        frame_idx=1,
        selected_indices_by_topic=selected_indices_by_topic,
        active_topic_indices=active_topic_indices,
        active_topic_values=active_topic_values,
        buffered_topic_values=buffered_topic_values,
    )

    assert ready is False
