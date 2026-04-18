from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))
sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "main"
        / "policy"
        / "lerobot_policy_streaming_act"
        / "src"
    ),
)

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from env.robocasa_env import (
    ROBOCASA_MAX_STEPS_MARGIN_MIN,
    _estimate_conservative_max_steps_from_dataset,
    _validate_supported_input_features,
)
from lerobot_policy_streaming_act.configuration_streaming_act import StreamingACTConfig


def test_validate_supported_input_features_allows_act_signature_placeholders() -> None:
    cfg = ACTConfig(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.main": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 32, 32),
            ),
            "observation.path_signature": PolicyFeature(
                type=FeatureType.STATE,
                shape=(64,),
            ),
            "observation.delta_signature": PolicyFeature(
                type=FeatureType.STATE,
                shape=(64,),
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(12,)),
        },
    )

    _validate_supported_input_features(cfg)


def test_validate_supported_input_features_allows_streaming_prefix_placeholders() -> None:
    cfg = StreamingACTConfig(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.main": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 32, 32),
            ),
            "observation.path_signature": PolicyFeature(
                type=FeatureType.STATE,
                shape=(64,),
            ),
            "observation.delta_signature": PolicyFeature(
                type=FeatureType.STATE,
                shape=(64,),
            ),
            "observation.prefix_state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(4, 16),
            ),
            "observation.prefix_mask": PolicyFeature(
                type=FeatureType.STATE,
                shape=(4,),
            ),
            "observation.prefix_path_signature": PolicyFeature(
                type=FeatureType.STATE,
                shape=(4, 64),
            ),
            "observation.prefix_delta_signature": PolicyFeature(
                type=FeatureType.STATE,
                shape=(4, 64),
            ),
            "observation.prefix_images.main": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(4, 3, 32, 32),
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(12,)),
        },
        use_path_signature=True,
        use_delta_signature=True,
        use_prefix_sequence_training=True,
        prefix_train_max_steps=4,
        prefix_frame_stride=1,
        history_length=4,
        signature_dim=64,
        signature_depth=2,
        signature_hidden_dim=32,
        signature_dropout=0.0,
        chunk_size=2,
        n_action_steps=1,
        dim_model=32,
        dim_feedforward=64,
        n_heads=4,
    )

    _validate_supported_input_features(cfg)


def test_validate_supported_input_features_rejects_unknown_feature() -> None:
    cfg = ACTConfig(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.main": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 32, 32),
            ),
            "observation.unsupported_context": PolicyFeature(
                type=FeatureType.STATE,
                shape=(4,),
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(12,)),
        },
    )

    try:
        _validate_supported_input_features(cfg)
    except NotImplementedError as exc:
        assert "Unsupported input features" in str(exc)
    else:
        raise AssertionError("Expected RoboCasa feature validation to reject unknown inputs.")


def test_estimate_conservative_max_steps_uses_v30_episode_parquet() -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_root = Path(tmp_dir)
        episodes_dir = dataset_root / "meta" / "episodes" / "chunk-000"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table({"length": pa.array([20, 37, 15], type=pa.int64())})
        pq.write_table(table, episodes_dir / "file-000.parquet")

        resolved, longest, margin = _estimate_conservative_max_steps_from_dataset(
            dataset_root
        )

        assert longest == 37
        assert margin == ROBOCASA_MAX_STEPS_MARGIN_MIN
        assert resolved == 37 + ROBOCASA_MAX_STEPS_MARGIN_MIN


def test_estimate_conservative_max_steps_falls_back_to_state_archives() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_root = Path(tmp_dir)
        extras_root = dataset_root / "extras"
        (extras_root / "episode_000000").mkdir(parents=True, exist_ok=True)
        (extras_root / "episode_000001").mkdir(parents=True, exist_ok=True)
        np.savez(extras_root / "episode_000000" / "states.npz", states=np.zeros((12, 8)))
        np.savez(extras_root / "episode_000001" / "states.npz", states=np.zeros((41, 8)))

        resolved, longest, margin = _estimate_conservative_max_steps_from_dataset(
            dataset_root
        )

        assert longest == 41
        assert margin == ROBOCASA_MAX_STEPS_MARGIN_MIN
        assert resolved == 41 + ROBOCASA_MAX_STEPS_MARGIN_MIN


def test_estimate_conservative_max_steps_falls_back_to_info_json() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_root = Path(tmp_dir)
        meta_root = dataset_root / "meta"
        meta_root.mkdir(parents=True, exist_ok=True)
        (meta_root / "info.json").write_text(
            json.dumps({"total_frames": 100, "total_episodes": 4}),
            encoding="utf-8",
        )

        resolved, longest, margin = _estimate_conservative_max_steps_from_dataset(
            dataset_root
        )

        assert longest == 25
        assert margin == ROBOCASA_MAX_STEPS_MARGIN_MIN
        assert resolved == 25 + ROBOCASA_MAX_STEPS_MARGIN_MIN
