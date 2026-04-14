from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

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
from lerobot_policy_streaming_act.prefix_image_cache import _resolve_camera_specs
from lerobot_policy_streaming_act.signature_cache import _manifests_are_compatible
from lerobot_policy_streaming_act.modeling_streaming_act import StreamingACTPolicy
from lerobot_policy_streaming_act.prefix_sequence import (
    PREFIX_DELTA_SIGNATURE_KEY,
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
    PrefixSequenceDataset,
    build_prefix_sequence_input_features,
    prefix_image_key_from_camera_key,
)
from train_policy import resolve_effective_dataset_repo_id


def _fixed_size_list(values: np.ndarray):
    if pa is None:
        raise RuntimeError("pyarrow is required for this helper.")
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


def test_streaming_act_signature_indexed_slot_memory_forward_smoke() -> None:
    torch.manual_seed(0)
    base_input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(17,)),
        "observation.images.main": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 32, 32)
        ),
        PATH_SIGNATURE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        DELTA_SIGNATURE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    input_features = build_prefix_sequence_input_features(
        base_input_features=base_input_features,
        prefix_train_max_steps=4,
        use_path_signature=True,
        use_delta_signature=True,
    )
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(17,)),
    }
    cfg = StreamingACTConfig(
        input_features=input_features,
        output_features=output_features,
        use_path_signature=True,
        use_delta_signature=True,
        use_prefix_sequence_training=True,
        prefix_train_max_steps=4,
        prefix_frame_stride=1,
        use_visual_prefix_memory=True,
        use_signature_indexed_slot_memory=True,
        use_memory_conditioned_encoder_film=True,
        slot_memory_num_slots=2,
        slot_memory_use_delta_routing=True,
        slot_memory_balance_loss_coef=0.1,
        slot_memory_consistency_loss_coef=0.1,
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

    batch_size = 2
    prefix_key = prefix_image_key_from_camera_key("observation.images.main")
    batch = {
        "observation.state": torch.randn(batch_size, 17),
        "observation.images.main": torch.randn(batch_size, 3, 32, 32),
        PATH_SIGNATURE_KEY: torch.randn(batch_size, 8),
        DELTA_SIGNATURE_KEY: torch.randn(batch_size, 8),
        PREFIX_STATE_KEY: torch.randn(batch_size, 4, 17),
        PREFIX_PATH_SIGNATURE_KEY: torch.randn(batch_size, 4, 8),
        PREFIX_DELTA_SIGNATURE_KEY: torch.randn(batch_size, 4, 8),
        PREFIX_MASK_KEY: torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        ),
        prefix_key: torch.randn(batch_size, 4, 3, 32, 32),
    }

    actions, _latent = policy.model(policy._prepare_observation_batch(batch))
    aux_losses = policy.model.get_visual_prefix_memory_aux_losses()

    assert actions.shape == (batch_size, 2, 17)
    assert "slot_memory_balance_loss" in aux_losses
    assert "slot_memory_consistency_loss" in aux_losses
    assert torch.isfinite(aux_losses["slot_memory_balance_loss"])
    assert torch.isfinite(aux_losses["slot_memory_consistency_loss"])


def test_prefix_sequence_cache_applies_current_frame_transform() -> None:
    camera_key = "observation.images.main"
    prefix_key = prefix_image_key_from_camera_key(camera_key)

    class _RowAccessor:
        def __getitem__(self, _idx):
            return type("Row", (), {"name": "toy-task"})()

    class _TaskTable:
        iloc = _RowAccessor()

    class _CacheReader:
        def has_key(self, key: str) -> bool:
            return key == camera_key

        def get(self, key: str, absolute_index: int) -> torch.Tensor:
            assert key == camera_key
            return torch.full((3, 2, 2), float(absolute_index), dtype=torch.float32)

        def get_many(self, key: str, absolute_indices: list[int]) -> torch.Tensor:
            assert key == camera_key
            return torch.stack(
                [
                    torch.full((3, 2, 2), float(index), dtype=torch.float32)
                    for index in absolute_indices
                ],
                dim=0,
            )

    class _BaseDataset:
        def __init__(self) -> None:
            self.hf_dataset = [
                {
                    "observation.state": torch.tensor([0.0, 0.0], dtype=torch.float32),
                    "episode_index": torch.tensor(0),
                    "index": torch.tensor(0),
                    "timestamp": torch.tensor(0.0),
                    "task_index": torch.tensor(0),
                },
                {
                    "observation.state": torch.tensor([1.0, 1.0], dtype=torch.float32),
                    "episode_index": torch.tensor(0),
                    "index": torch.tensor(1),
                    "timestamp": torch.tensor(1.0),
                    "task_index": torch.tensor(0),
                },
            ]
            self.image_transforms = lambda image: image + 1.0
            self.features = {}
            self.delta_indices = None
            self.meta = type(
                "Meta",
                (),
                {
                    "camera_keys": (camera_key,),
                    "video_keys": (camera_key,),
                    "stats": {
                        "observation.state": {"mean": [0.0], "std": [1.0]},
                        camera_key: {"mean": [0.0], "std": [1.0]},
                    },
                    "episodes": [{"dataset_from_index": 0}],
                    "tasks": _TaskTable(),
                    "subtasks": None,
                },
            )()

        def __len__(self) -> int:
            return len(self.hf_dataset)

        def __getattr__(self, name: str):
            raise AttributeError(name)

        def _ensure_hf_dataset_loaded(self) -> None:
            return None

        def _query_hf_dataset(self, query_indices):
            result = {}
            if "observation.state" in query_indices:
                result["observation.state"] = torch.stack(
                    [self.hf_dataset[index]["observation.state"] for index in query_indices["observation.state"]],
                    dim=0,
                )
            return result

    dataset = PrefixSequenceDataset(
        _BaseDataset(),
        prefix_train_max_steps=2,
        prefix_frame_stride=1,
        prefix_pad_value=0.0,
        use_path_signature=False,
        use_delta_signature=False,
        prefix_image_cache_reader=_CacheReader(),
    )

    item = dataset[1]

    assert torch.allclose(item[camera_key], torch.full((3, 2, 2), 2.0))
    assert torch.allclose(item[prefix_key][0], torch.full((3, 2, 2), 1.0))
    assert torch.allclose(item[prefix_key][1], torch.full((3, 2, 2), 2.0))


def test_prefix_image_cache_normalizes_hwc_shapes_to_chw() -> None:
    info = {
        "features": {
            "observation.images.main": {
                "dtype": "video",
                "shape": [256, 256, 3],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [3, 128, 128],
            },
        }
    }

    specs = _resolve_camera_specs(info)

    assert specs["observation.images.main"] == (3, 256, 256)
    assert specs["observation.images.wrist"] == (3, 128, 128)


def test_prefix_sequence_fast_path_restores_current_signatures_from_cache() -> None:
    camera_key = "observation.images.main"
    prefix_key = prefix_image_key_from_camera_key(camera_key)

    class _RowAccessor:
        def __getitem__(self, _idx):
            return type("Row", (), {"name": "toy-task"})()

    class _TaskTable:
        iloc = _RowAccessor()

    class _PrefixImageCacheReader:
        def has_key(self, key: str) -> bool:
            return key == camera_key

        def get(self, key: str, absolute_index: int) -> torch.Tensor:
            assert key == camera_key
            return torch.full((3, 2, 2), float(absolute_index), dtype=torch.float32)

        def get_many(self, key: str, absolute_indices: list[int]) -> torch.Tensor:
            assert key == camera_key
            return torch.stack(
                [
                    torch.full((3, 2, 2), float(index), dtype=torch.float32)
                    for index in absolute_indices
                ],
                dim=0,
            )

    class _SignatureCacheReader:
        def has_key(self, key: str) -> bool:
            return key in {PATH_SIGNATURE_KEY, DELTA_SIGNATURE_KEY}

        def get(self, key: str, absolute_index: int) -> torch.Tensor:
            base = float(absolute_index)
            if key == PATH_SIGNATURE_KEY:
                return torch.tensor([base, base + 10.0], dtype=torch.float32)
            if key == DELTA_SIGNATURE_KEY:
                return torch.tensor([base, -base], dtype=torch.float32)
            raise KeyError(key)

        def get_many(self, key: str, absolute_indices: list[int]) -> torch.Tensor:
            return torch.stack([self.get(key, index) for index in absolute_indices], dim=0)

    class _BaseDataset:
        def __init__(self) -> None:
            self.hf_dataset = [
                {
                    "observation.state": torch.tensor([0.0, 0.0], dtype=torch.float32),
                    "episode_index": torch.tensor(0),
                    "index": torch.tensor(0),
                    "timestamp": torch.tensor(0.0),
                    "task_index": torch.tensor(0),
                },
                {
                    "observation.state": torch.tensor([1.0, 1.0], dtype=torch.float32),
                    "episode_index": torch.tensor(0),
                    "index": torch.tensor(1),
                    "timestamp": torch.tensor(1.0),
                    "task_index": torch.tensor(0),
                },
            ]
            self.image_transforms = None
            self.features = {}
            self.delta_indices = None
            self._signature_cache_reader = _SignatureCacheReader()
            self.meta = type(
                "Meta",
                (),
                {
                    "camera_keys": (camera_key,),
                    "video_keys": (camera_key,),
                    "stats": {
                        "observation.state": {"mean": [0.0], "std": [1.0]},
                        PATH_SIGNATURE_KEY: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                        DELTA_SIGNATURE_KEY: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                        camera_key: {"mean": [0.0], "std": [1.0]},
                    },
                    "episodes": [{"dataset_from_index": 0}],
                    "tasks": _TaskTable(),
                    "subtasks": None,
                },
            )()

        def __len__(self) -> int:
            return len(self.hf_dataset)

        def _ensure_hf_dataset_loaded(self) -> None:
            return None

        def _query_hf_dataset(self, query_indices):
            result = {}
            for key, indices in query_indices.items():
                if key == "observation.state":
                    result[key] = torch.stack(
                        [self.hf_dataset[index]["observation.state"] for index in indices],
                        dim=0,
                    )
                elif self._signature_cache_reader.has_key(key):
                    result[key] = self._signature_cache_reader.get_many(key, indices)
            return result

    dataset = PrefixSequenceDataset(
        _BaseDataset(),
        prefix_train_max_steps=2,
        prefix_frame_stride=1,
        prefix_pad_value=0.0,
        use_path_signature=True,
        use_delta_signature=True,
        prefix_image_cache_reader=_PrefixImageCacheReader(),
    )

    item = dataset[1]

    assert torch.allclose(item[PATH_SIGNATURE_KEY], torch.tensor([1.0, 11.0]))
    assert torch.allclose(item[DELTA_SIGNATURE_KEY], torch.tensor([1.0, -1.0]))
    assert torch.allclose(
        item[PREFIX_PATH_SIGNATURE_KEY],
        torch.tensor([[0.0, 10.0], [1.0, 11.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        item[PREFIX_DELTA_SIGNATURE_KEY],
        torch.tensor([[0.0, -0.0], [1.0, -1.0]], dtype=torch.float32),
    )
    assert torch.allclose(item[prefix_key][1], torch.full((3, 2, 2), 1.0))


def test_prefix_sequence_fast_path_preserves_delta_indexed_action_sequence() -> None:
    camera_key = "observation.images.main"
    prefix_key = prefix_image_key_from_camera_key(camera_key)

    class _RowAccessor:
        def __getitem__(self, _idx):
            return type("Row", (), {"name": "toy-task"})()

    class _TaskTable:
        iloc = _RowAccessor()

    class _PrefixImageCacheReader:
        def has_key(self, key: str) -> bool:
            return key == camera_key

        def get(self, key: str, absolute_index: int) -> torch.Tensor:
            assert key == camera_key
            return torch.full((3, 2, 2), float(absolute_index), dtype=torch.float32)

        def get_many(self, key: str, absolute_indices: list[int]) -> torch.Tensor:
            assert key == camera_key
            return torch.stack(
                [
                    torch.full((3, 2, 2), float(index), dtype=torch.float32)
                    for index in absolute_indices
                ],
                dim=0,
            )

    class _BaseDataset:
        def __init__(self) -> None:
            self.hf_dataset = [
                {
                    "observation.state": torch.tensor([0.0, 0.0], dtype=torch.float32),
                    "episode_index": torch.tensor(0),
                    "index": torch.tensor(0),
                    "timestamp": torch.tensor(0.0),
                    "task_index": torch.tensor(0),
                },
                {
                    "observation.state": torch.tensor([1.0, 1.0], dtype=torch.float32),
                    "episode_index": torch.tensor(0),
                    "index": torch.tensor(1),
                    "timestamp": torch.tensor(1.0),
                    "task_index": torch.tensor(0),
                },
                {
                    "observation.state": torch.tensor([2.0, 2.0], dtype=torch.float32),
                    "episode_index": torch.tensor(0),
                    "index": torch.tensor(2),
                    "timestamp": torch.tensor(2.0),
                    "task_index": torch.tensor(0),
                },
            ]
            self.image_transforms = None
            self.features = {}
            self.delta_indices = {"action": [0, 1]}
            self.meta = type(
                "Meta",
                (),
                {
                    "camera_keys": (camera_key,),
                    "video_keys": (camera_key,),
                    "stats": {
                        "observation.state": {"mean": [0.0], "std": [1.0]},
                        camera_key: {"mean": [0.0], "std": [1.0]},
                    },
                    "episodes": [{"dataset_from_index": 0, "dataset_to_index": 3}],
                    "tasks": _TaskTable(),
                    "subtasks": None,
                },
            )()

        def __len__(self) -> int:
            return len(self.hf_dataset)

        def _ensure_hf_dataset_loaded(self) -> None:
            return None

        def _get_query_indices(self, abs_idx: int, _ep_idx: int):
            query = {"action": [min(abs_idx + delta, 2) for delta in self.delta_indices["action"]]}
            padding = {
                "action_is_pad": torch.BoolTensor([False, abs_idx + 1 >= 3]),
            }
            return query, padding

        def _query_hf_dataset(self, query_indices):
            result = {}
            if "observation.state" in query_indices:
                result["observation.state"] = torch.stack(
                    [self.hf_dataset[index]["observation.state"] for index in query_indices["observation.state"]],
                    dim=0,
                )
            if "action" in query_indices:
                result["action"] = torch.stack(
                    [
                        torch.tensor([float(index), float(index) + 0.5], dtype=torch.float32)
                        for index in query_indices["action"]
                    ],
                    dim=0,
                )
            return result

    dataset = PrefixSequenceDataset(
        _BaseDataset(),
        prefix_train_max_steps=2,
        prefix_frame_stride=1,
        prefix_pad_value=0.0,
        use_path_signature=False,
        use_delta_signature=False,
        prefix_image_cache_reader=_PrefixImageCacheReader(),
    )

    item = dataset[1]

    assert item["action"].shape == (2, 2)
    assert torch.equal(item["action_is_pad"], torch.tensor([False, False]))
    assert torch.allclose(
        item["action"],
        torch.tensor([[1.0, 1.5], [2.0, 2.5]], dtype=torch.float32),
    )
    assert torch.allclose(item[prefix_key][1], torch.full((3, 2, 2), 1.0))


def test_process_dataset_writes_signature_cache_without_parquet_columns(
    tmp_path: Path,
) -> None:
    if pa is None or pq is None:
        pytest.skip("pyarrow is not available in this environment")
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
