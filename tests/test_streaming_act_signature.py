from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "data"))
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

from process_dataset import build_parser, process_dataset, resolve_dataset_arg
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot_policy_streaming_act.configuration_streaming_act import (
    DELTA_SIGNATURE_KEY,
    PATH_SIGNATURE_KEY,
    StreamingACTConfig,
)
from lerobot_policy_streaming_act.signature_cache import _manifests_are_compatible
from lerobot_policy_streaming_act.modeling_streaming_act import StreamingACTPolicy
from train_policy import resolve_effective_dataset_repo_id


def _fixed_size_list(values: np.ndarray):
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, int(values.shape[1]))


def test_process_dataset_cli_accepts_dataset_flag_alias() -> None:
    parser = build_parser()

    args = parser.parse_args(["--dataset", "zeno-ai/wholebody_hanger_stage3_v30"])
    assert (
        resolve_dataset_arg(
            parser,
            dataset=args.dataset,
            dataset_option=args.dataset_option,
        )
        == "zeno-ai/wholebody_hanger_stage3_v30"
    )

    args = parser.parse_args(["zeno-ai/wholebody_hanger_stage3_v30"])
    assert (
        resolve_dataset_arg(
            parser,
            dataset=args.dataset,
            dataset_option=args.dataset_option,
        )
        == "zeno-ai/wholebody_hanger_stage3_v30"
    )

    with pytest.raises(SystemExit):
        resolve_dataset_arg(
            parser,
            dataset="zeno-ai/wholebody_hanger_stage3_v30",
            dataset_option="zeno-ai/wholebody_hanger_stage3_v30",
        )


def test_streaming_act_signature_affects_output() -> None:
    torch.manual_seed(0)
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(17,)),
        "observation.images.main": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 64, 64)
        ),
        PATH_SIGNATURE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        DELTA_SIGNATURE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(17,)),
    }
    cfg = StreamingACTConfig(
        input_features=input_features,
        output_features=output_features,
        use_path_signature=True,
        use_delta_signature=True,
        signature_dim=8,
        signature_hidden_dim=16,
        signature_dropout=0.0,
        chunk_size=2,
        n_action_steps=1,
        dim_model=32,
        dim_feedforward=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        n_vae_encoder_layers=1,
        dropout=0.0,
        use_vae=False,
        pretrained_backbone_weights=None,
    )
    policy = StreamingACTPolicy(cfg)
    policy.eval()

    batch = {
        "observation.state": torch.randn(2, 17),
        "observation.images.main": torch.randn(2, 3, 64, 64),
        PATH_SIGNATURE_KEY: torch.randn(2, 8),
        DELTA_SIGNATURE_KEY: torch.randn(2, 8),
    }
    out_1 = policy.predict_action_chunk(batch)
    batch_2 = dict(batch)
    batch_2[PATH_SIGNATURE_KEY] = batch[PATH_SIGNATURE_KEY] + 2.0
    out_2 = policy.predict_action_chunk(batch_2)

    assert out_1.shape == (2, 2, 17)
    assert not torch.allclose(out_1, out_2)


def test_process_dataset_writes_signature_cache_without_parquet_columns(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "toy_dataset"
    (dataset_root / "data/chunk-000").mkdir(parents=True)
    (dataset_root / "meta/episodes/chunk-000").mkdir(parents=True)

    states = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    table = pa.table(
        {
            "observation.state": _fixed_size_list(states),
            "action": _fixed_size_list(states.copy()),
            "episode_index": pa.array([0, 0, 0, 1, 1], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2, 0, 1], type=pa.int64()),
            "index": pa.array([0, 1, 2, 3, 4], type=pa.int64()),
            "timestamp": pa.array([0.0, 0.1, 0.2, 0.3, 0.4], type=pa.float32()),
        }
    )
    pq.write_table(table, dataset_root / "data/chunk-000/file-000.parquet")

    episodes = pa.table(
        {
            "episode_index": pa.array([0, 1], type=pa.int64()),
            "dataset_from_index": pa.array([0, 3], type=pa.int64()),
            "dataset_to_index": pa.array([3, 5], type=pa.int64()),
            "length": pa.array([3, 2], type=pa.int64()),
        }
    )
    pq.write_table(episodes, dataset_root / "meta/episodes/chunk-000/file-000.parquet")

    info = {
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["x", "y"],
            },
            "action": {"dtype": "float32", "shape": [2], "names": ["ax", "ay"]},
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["episode_index"],
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["frame_index"],
            },
            "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
        }
    }
    stats = {
        "observation.state": {
            "min": states.min(axis=0).astype(np.float32).tolist(),
            "max": states.max(axis=0).astype(np.float32).tolist(),
            "mean": states.mean(axis=0).astype(np.float32).tolist(),
            "std": states.std(axis=0).astype(np.float32).tolist(),
            "count": [int(len(states))],
        },
        "action": {
            "min": states.min(axis=0).astype(np.float32).tolist(),
            "max": states.max(axis=0).astype(np.float32).tolist(),
            "mean": states.mean(axis=0).astype(np.float32).tolist(),
            "std": states.std(axis=0).astype(np.float32).tolist(),
            "count": [int(len(states))],
        },
    }
    (dataset_root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (dataset_root / "meta/stats.json").write_text(json.dumps(stats), encoding="utf-8")

    process_dataset(
        dataset_root,
        observation_key="observation.state",
        path_signature_key=PATH_SIGNATURE_KEY,
        delta_signature_key=DELTA_SIGNATURE_KEY,
        signature_depth=2,
        signature_cache_dtype="float32",
        signature_cache_root=None,
    )

    schema = pq.read_schema(dataset_root / "data/chunk-000/file-000.parquet")
    assert PATH_SIGNATURE_KEY not in schema.names
    assert DELTA_SIGNATURE_KEY not in schema.names

    updated_info = json.loads((dataset_root / "meta/info.json").read_text())
    updated_stats = json.loads((dataset_root / "meta/stats.json").read_text())
    assert PATH_SIGNATURE_KEY in updated_info["features"]
    assert DELTA_SIGNATURE_KEY in updated_info["features"]
    assert PATH_SIGNATURE_KEY in updated_stats
    assert DELTA_SIGNATURE_KEY in updated_stats

    cache_dir = next(dataset_root.glob(".signature_cache/*/signature_cache_v1"))
    assert (cache_dir / "metadata.json").exists()
    assert list(cache_dir.glob("*.npy"))


def test_resolve_effective_dataset_repo_id_prefers_leaf_dataset_over_broad_defaults(
    tmp_path: Path,
) -> None:
    local_data_root = tmp_path / "data"
    dataset_root = local_data_root / "robocasa" / "composite" / "ArrangeBreadBasket"
    dataset_root.mkdir(parents=True)

    resolved = resolve_effective_dataset_repo_id(
        requested_repo_id="robocasa/composite",
        default_repo_id="robocasa/composite",
        dataset_root=dataset_root,
        local_data_root=local_data_root,
    )

    assert resolved == "robocasa/composite/ArrangeBreadBasket"


def test_resolve_effective_dataset_repo_id_keeps_exact_dataset_defaults(
    tmp_path: Path,
) -> None:
    local_data_root = tmp_path / "data"
    dataset_root = local_data_root / "robocasa" / "composite" / "ArrangeBreadBasket"
    dataset_root.mkdir(parents=True)

    resolved = resolve_effective_dataset_repo_id(
        requested_repo_id="robocasa/composite/ArrangeBreadBasket",
        default_repo_id="robocasa/composite/ArrangeBreadBasket",
        dataset_root=dataset_root,
        local_data_root=local_data_root,
    )

    assert resolved == "robocasa/composite/ArrangeBreadBasket"


def test_signature_cache_manifest_compat_accepts_legacy_info_backed_episode_path() -> None:
    current_manifest = {
        "dataset_root": "/tmp/dataset",
        "info_path": "/tmp/dataset/meta/info.json",
        "stats_path": "/tmp/dataset/meta/stats.json",
        "episodes_path": "/tmp/dataset/meta/episodes/chunk-000/file-000.parquet",
        "total_frames": 5,
        "data_files": [
            {
                "path": "data/chunk-000/file-000.parquet",
                "size": 123,
                "mtime_ns": 1,
            }
        ],
    }
    cached_manifest = {
        "dataset_root": "/tmp/dataset",
        "info_path": "/tmp/dataset/meta/info.json",
        "stats_path": "/tmp/dataset/meta/stats.json",
        "episodes_path": "/tmp/dataset/meta/info.json",
        "episode_metadata_path": "/tmp/dataset/meta/info.json",
        "total_frames": 5,
        "data_files": [
            {
                "path": "data/chunk-000/file-000.parquet",
                "size": 123,
                "mtime_ns": 999,
            }
        ],
    }

    assert _manifests_are_compatible(cached_manifest, current_manifest)
