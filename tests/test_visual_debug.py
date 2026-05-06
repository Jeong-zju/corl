from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main"))

from deploy.visual_debug import (  # noqa: E402
    make_slot_memory_history_sample,
    render_attention_panel,
    render_slot_memory_panel,
)


def test_attention_panel_letterboxes_non_square_video_without_crop() -> None:
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    image[:, :] = (20, 40, 60)

    panel = render_attention_panel(
        images={"observation.images.front": image},
        debug=None,
        color_order="rgb",
        camera_labels={"observation.images.front": "front"},
        overlay_alpha=0.4,
        query_step=0,
        tile_width=160,
        tile_height=160,
    )

    video_y0 = 44
    assert panel.shape == (286, 160, 3)
    assert np.all(panel[video_y0 + 5, 80] == 0)
    assert np.any(panel[video_y0 + 40, 80] != 0)
    assert np.all(panel[video_y0 + 155, 80] == 0)


def test_attention_panel_wraps_all_extra_memory_labels() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    labels = [f"slot_{idx}" for idx in range(5)]
    debug = {
        "decoder_token_layout": {
            "camera_keys": ["observation.images.front"],
            "num_cameras": 1,
            "extra_memory_token_labels": labels,
        },
        "decoder_cross_attention": np.ones((1, len(labels)), dtype=np.float32),
    }

    panel = render_attention_panel(
        images={"observation.images.front": image},
        debug=debug,
        color_order="rgb",
        camera_labels={"observation.images.front": "front"},
        overlay_alpha=0.4,
        query_step=0,
        tile_width=160,
        tile_height=160,
    )

    assert panel.shape[0] == 286 + (len(labels) - 1) * 38


def test_slot_memory_panel_history_expands_for_all_slots() -> None:
    num_slots = 30
    routing = np.linspace(0.0, 1.0, num_slots, dtype=np.float32)[None, :]
    debug = {
        "visual_memory_stats": {
            "enabled": True,
            "initialized": True,
            "num_slots": num_slots,
            "update_count": 7,
            "state_norm": 1.25,
        },
        "slot_memory": {
            "routing_weights": routing,
            "routing_weights_zero_signature": routing * 0.5,
            "routing_delta_from_zero_signature": routing * 0.5,
            "gate": np.ones((1, num_slots, 4), dtype=np.float32) * 0.25,
            "write_strength": routing * 0.25,
            "readout_weights": routing[::-1],
            "memory_next_norm": routing + 1.0,
            "memory_delta_norm": routing * 0.1,
        },
    }
    history = []
    for idx in range(5):
        frame_debug = dict(debug)
        frame_debug["slot_memory"] = dict(debug["slot_memory"])
        frame_debug["slot_memory"]["routing_weights"] = routing * ((idx + 1) / 5.0)
        history.append(make_slot_memory_history_sample(frame_debug))

    panel = render_slot_memory_panel(
        debug=debug,
        history=history,
        current_step=4,
        total_steps=12,
    )

    assert panel.shape[1] >= 260 + num_slots * 28
    assert panel.shape[0] >= 1010
    assert panel.shape[0] % 2 == 0
    assert panel.shape[1] % 2 == 0


def test_slot_memory_panel_can_embed_video_stream() -> None:
    num_slots = 4
    routing = np.array([[0.05, 0.55, 0.25, 0.15]], dtype=np.float32)
    baseline = np.ones((1, num_slots), dtype=np.float32) / float(num_slots)
    debug = {
        "visual_memory_stats": {
            "enabled": True,
            "initialized": True,
            "num_slots": num_slots,
            "update_count": 3,
            "state_norm": 2.0,
        },
        "slot_memory": {
            "routing_weights": routing,
            "routing_weights_zero_signature": baseline,
            "routing_delta_from_zero_signature": routing - baseline,
            "write_strength": routing * 0.25,
            "readout_weights": routing[:, ::-1],
            "memory_next_norm": np.array([[1.0, 1.4, 0.9, 1.2]], dtype=np.float32),
            "memory_delta_norm": routing * 0.1,
        },
    }
    history = [make_slot_memory_history_sample(debug) for _ in range(3)]
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :] = (10, 120, 240)

    panel = render_slot_memory_panel(
        debug=debug,
        images={"observation.images.front": image},
        color_order="rgb",
        camera_labels={"observation.images.front": "front"},
        history=history,
        current_step=2,
        total_steps=5,
    )

    assert panel.shape[1] >= 1280
    assert panel.shape[0] >= 720
    assert np.any(panel[250, 280] != 248)
