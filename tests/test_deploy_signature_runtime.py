from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main"))
sys.path.insert(0, str(REPO_ROOT / "main" / "deploy"))
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from deploy.bridge.signature_runtime import OnlineSignatureRuntime
from deploy.config import PolicyConfig
from eval_helpers import compute_simple_signature_np
from eval_helpers import resolve_policy_dir


def _make_policy_config(**overrides) -> PolicyConfig:
    base = dict(
        type="streaming_act",
        path=Path("."),
        device="cpu",
        load_device=None,
        n_action_steps=None,
        state_dim=17,
        action_dim=17,
        arm_dof=7,
        base_action_dim=3,
        state_key="observation.state",
        action_key="action",
        image_keys={
            "left": "observation.images.left",
            "right": "observation.images.right",
            "top": "observation.images.top",
        },
        use_path_signature=False,
        use_delta_signature=False,
        signature_depth=1,
        signature_dim=None,
        signature_backend="simple",
    )
    base.update(overrides)
    return PolicyConfig(**base)


def test_online_signature_runtime_uses_loaded_checkpoint_signature_settings() -> None:
    deploy_cfg = _make_policy_config()
    loaded_cfg = SimpleNamespace(
        use_path_signature=True,
        use_delta_signature=True,
        history_length=3,
        signature_depth=2,
        signature_dim=None,
    )
    runtime = OnlineSignatureRuntime(
        deploy_cfg,
        loaded_policy_cfg=loaded_cfg,
    )

    state_t0 = np.asarray([1.0, 2.0], dtype=np.float32)
    state_t1 = np.asarray([3.0, 5.0], dtype=np.float32)

    signature_t0, delta_t0 = runtime.update(state_t0)
    signature_t1, delta_t1 = runtime.update(state_t1)

    expected_t0 = compute_simple_signature_np(
        np.repeat(state_t0[None, :], 3, axis=0),
        2,
    )
    expected_t1 = compute_simple_signature_np(
        np.asarray(
            [
                state_t0,
                state_t0,
                state_t1,
            ],
            dtype=np.float32,
        ),
        2,
    )

    assert runtime.enabled is True
    assert runtime.history_length == 3
    assert runtime.backend == "simple"
    assert signature_t0 is not None
    assert delta_t0 is not None
    assert signature_t1 is not None
    assert delta_t1 is not None
    assert np.allclose(signature_t0, expected_t0)
    assert np.allclose(delta_t0, np.zeros_like(expected_t0))
    assert np.allclose(signature_t1, expected_t1)
    assert np.allclose(delta_t1, expected_t1 - expected_t0)


def test_online_signature_runtime_normalizes_pre_normalized_signature_features(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data" / "zeno-ai" / "CleanTableTopDelayedToolChoice"
    stats_dir = dataset_root / "meta"
    stats_dir.mkdir(parents=True)

    state_t0 = np.asarray([1.0, 2.0], dtype=np.float32)
    state_t1 = np.asarray([3.0, 5.0], dtype=np.float32)
    raw_signature_t0 = compute_simple_signature_np(
        np.repeat(state_t0[None, :], 3, axis=0),
        2,
    )
    raw_signature_t1 = compute_simple_signature_np(
        np.asarray(
            [
                state_t0,
                state_t0,
                state_t1,
            ],
            dtype=np.float32,
        ),
        2,
    )
    raw_delta_t0 = np.zeros_like(raw_signature_t0)
    raw_delta_t1 = raw_signature_t1 - raw_signature_t0

    path_mean = np.full_like(raw_signature_t0, 1.5, dtype=np.float32)
    path_std = np.full_like(raw_signature_t0, 0.5, dtype=np.float32)
    delta_mean = np.full_like(raw_signature_t0, -0.25, dtype=np.float32)
    delta_std = np.full_like(raw_signature_t0, 2.0, dtype=np.float32)
    stats_payload = {
        "observation.path_signature": {
            "mean": path_mean.tolist(),
            "std": path_std.tolist(),
        },
        "observation.delta_signature": {
            "mean": delta_mean.tolist(),
            "std": delta_std.tolist(),
        },
    }
    (stats_dir / "stats.json").write_text(json.dumps(stats_payload), encoding="utf-8")

    run_root = tmp_path / "outputs" / "train" / "streaming-act-prism" / "20260418_010000"
    policy_dir = run_root / "checkpoints" / "last" / "pretrained_model"
    policy_dir.mkdir(parents=True)
    (run_root / "dataset_split.json").write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "dataset_repo_id": "zeno-ai/CleanTableTopDelayedToolChoice",
            }
        ),
        encoding="utf-8",
    )

    runtime = OnlineSignatureRuntime(
        _make_policy_config(signature_backend="simple"),
        loaded_policy_cfg=SimpleNamespace(
            use_path_signature=True,
            use_delta_signature=True,
            history_length=3,
            signature_depth=2,
            signature_dim=None,
            pre_normalized_observation_keys=(
                "observation.path_signature",
                "observation.delta_signature",
            ),
            normalization_mapping={"STATE": "MEAN_STD"},
        ),
        policy_dir=policy_dir,
    )

    signature_t0, delta_t0 = runtime.update(state_t0)
    signature_t1, delta_t1 = runtime.update(state_t1)

    expected_signature_t0 = (raw_signature_t0 - path_mean) / path_std
    expected_signature_t1 = (raw_signature_t1 - path_mean) / path_std
    expected_delta_t0 = (raw_delta_t0 - delta_mean) / delta_std
    expected_delta_t1 = (raw_delta_t1 - delta_mean) / delta_std

    assert runtime.normalization_summary.startswith("enabled(")
    assert np.allclose(signature_t0, expected_signature_t0)
    assert np.allclose(signature_t1, expected_signature_t1)
    assert np.allclose(delta_t0, expected_delta_t0)
    assert np.allclose(delta_t1, expected_delta_t1)


def test_resolve_policy_dir_accepts_train_output_root_and_finds_latest_run(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "outputs" / "train" / "streaming-act-prism"
    old_run = train_root / "20260417_010000"
    new_run = train_root / "20260418_010000"

    old_model_dir = old_run / "checkpoints" / "last" / "pretrained_model"
    new_model_dir = new_run / "checkpoints" / "last" / "pretrained_model"
    old_model_dir.mkdir(parents=True)
    new_model_dir.mkdir(parents=True)
    (old_model_dir / "model.safetensors").touch()
    (new_model_dir / "model.safetensors").touch()

    os.utime(old_run, (1, 1))
    os.utime(new_run, (2, 2))

    assert resolve_policy_dir(train_root) == new_model_dir.resolve()
