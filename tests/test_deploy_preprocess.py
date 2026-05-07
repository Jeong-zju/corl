from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main"))
sys.path.insert(0, str(REPO_ROOT / "main" / "deploy"))

from deploy.policy_runtime.preprocess import (
    build_raw_policy_observation,
    finalize_preprocessed_observation,
    select_visual_observation_keys,
)


def test_build_raw_policy_observation_preserves_vla_task() -> None:
    pytest.importorskip("torch")
    image_key = "observation.images.front"
    cfg = SimpleNamespace(
        input_features={
            "observation.state": SimpleNamespace(shape=(2,)),
            image_key: SimpleNamespace(type="VISUAL", shape=(3, 4, 4)),
        },
        output_features={"action": SimpleNamespace(shape=(2,))},
    )

    obs = build_raw_policy_observation(
        {
            "state": np.asarray([1.0, 2.0], dtype=np.float32),
            "images": {image_key: np.zeros((4, 4, 3), dtype=np.uint8)},
            "task": "Pick up the object.",
        },
        cfg,
    )

    assert obs["task"] == "Pick up the object."
    assert obs["observation.state"].shape == (2,)
    assert obs[image_key].shape == (3, 4, 4)


def test_select_visual_observation_keys_falls_back_to_input_features() -> None:
    cfg = SimpleNamespace(
        input_features={
            "observation.state": SimpleNamespace(type="STATE", shape=(2,)),
            "observation.images.front": SimpleNamespace(type="VISUAL", shape=(3, 4, 4)),
        },
    )

    assert select_visual_observation_keys(cfg) == ["observation.images.front"]


def test_finalize_preprocessed_observation_accepts_packed_state() -> None:
    class FakeTensor:
        device = "cpu"
        dtype = "float32"

    obs = {
        "state": FakeTensor(),
        "state_mask": FakeTensor(),
    }

    result = finalize_preprocessed_observation(
        obs,
        SimpleNamespace(input_features={"observation.state": SimpleNamespace(shape=(17,))}),
    )

    assert result is obs
