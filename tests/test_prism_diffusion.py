from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

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
sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "main"
        / "policy"
        / "lerobot_policy_prism_diffusion"
        / "src"
    ),
)

from eval_policy import build_parser as build_eval_parser
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_policy_prism_diffusion.configuration_diffusion import PrismDiffusionConfig
from lerobot_policy_prism_diffusion.modeling_diffusion import PrismDiffusionPolicy
from lerobot_policy_prism_diffusion.prefix_dataset import PrismDiffusionPrefixDataset
from lerobot_policy_streaming_act.prefix_sequence import (
    DELTA_SIGNATURE_KEY,
    PATH_SIGNATURE_KEY,
    PREFIX_DELTA_SIGNATURE_KEY,
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
    build_prefix_sequence_input_features,
    prefix_image_key_from_camera_key,
)

CAMERA_KEY = "observation.images.main"
PREFIX_IMAGE_KEY = prefix_image_key_from_camera_key(CAMERA_KEY)


class _RowAccessor:
    def __getitem__(self, _idx):
        return type("Row", (), {"name": "toy-task"})()


class _TaskTable:
    iloc = _RowAccessor()


class _BaseDiffusionDataset:
    def __init__(self) -> None:
        self.hf_dataset = []
        for index in range(4):
            self.hf_dataset.append(
                {
                    "observation.state": torch.tensor(
                        [10.0 * (index + 1), 10.0 * (index + 1) + 1.0],
                        dtype=torch.float32,
                    ),
                    CAMERA_KEY: torch.full((3, 2, 2), float(index + 1), dtype=torch.float32),
                    PATH_SIGNATURE_KEY: torch.tensor(
                        [100.0 + index, 200.0 + index],
                        dtype=torch.float32,
                    ),
                    DELTA_SIGNATURE_KEY: torch.tensor(
                        [900.0 + index, -900.0 - index],
                        dtype=torch.float32,
                    ),
                    "episode_index": torch.tensor(0),
                    "index": torch.tensor(index),
                    "timestamp": torch.tensor(float(index)),
                    "task_index": torch.tensor(0),
                }
            )

        self.image_transforms = None
        self.features = {}
        self.delta_indices = {
            "observation.state": [-2, -1, 0],
            CAMERA_KEY: [-2, -1, 0],
            "action": [-2, -1, 0, 1],
        }
        self.meta = type(
            "Meta",
            (),
            {
                "camera_keys": (CAMERA_KEY,),
                "video_keys": (),
                "stats": {
                    "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    CAMERA_KEY: {"mean": [0.0], "std": [1.0]},
                    PATH_SIGNATURE_KEY: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    DELTA_SIGNATURE_KEY: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                },
                "episodes": [{"dataset_from_index": 0, "dataset_to_index": 4}],
                "tasks": _TaskTable(),
                "subtasks": None,
            },
        )()

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _ensure_hf_dataset_loaded(self) -> None:
        return None

    def _get_query_indices(self, abs_idx: int, ep_idx: int):
        episode = self.meta.episodes[ep_idx]
        ep_start = int(episode["dataset_from_index"])
        ep_end = int(episode["dataset_to_index"])
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in deltas]
            for key, deltas in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) or (abs_idx + delta >= ep_end) for delta in deltas]
            )
            for key, deltas in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_hf_dataset(self, query_indices):
        result = {}
        for key, indices in query_indices.items():
            if key == "action":
                result[key] = torch.stack(
                    [
                        torch.tensor([float(index), float(index) + 0.25], dtype=torch.float32)
                        for index in indices
                    ],
                    dim=0,
                )
                continue
            result[key] = torch.stack([self.hf_dataset[index][key] for index in indices], dim=0)
        return result

    def __getitem__(self, idx: int):
        item = dict(self.hf_dataset[idx])
        ep_idx = int(item["episode_index"].item())
        abs_idx = int(item["index"].item())
        query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
        item.update(padding)
        item.update(self._query_hf_dataset(query_indices))
        item["task"] = "toy-task"
        return item


def _make_policy(
    *,
    state_dim: int = 4,
    action_dim: int = 2,
    image_hw: int = 16,
    signature_dim: int = 8,
    n_obs_steps: int = 1,
    horizon: int = 4,
    n_action_steps: int = 1,
    use_path_signature: bool = False,
    use_delta_signature: bool = False,
    use_prefix_sequence_training: bool = False,
    use_visual_prefix_memory: bool = False,
    use_signature_indexed_slot_memory: bool = False,
    prism_adapter_zero_init: bool = False,
) -> PrismDiffusionPolicy:
    base_input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        CAMERA_KEY: PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, image_hw, image_hw),
        ),
    }
    if use_path_signature:
        base_input_features[PATH_SIGNATURE_KEY] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(signature_dim,),
        )
    if use_delta_signature:
        base_input_features[DELTA_SIGNATURE_KEY] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(signature_dim,),
        )
    input_features = (
        build_prefix_sequence_input_features(
            base_input_features=base_input_features,
            prefix_train_max_steps=4,
            use_path_signature=use_path_signature,
            use_delta_signature=use_delta_signature,
        )
        if use_prefix_sequence_training
        else base_input_features
    )
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    cfg = PrismDiffusionConfig(
        device="cpu",
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
        down_dims=(16, 32),
        kernel_size=3,
        n_groups=4,
        diffusion_step_embed_dim=32,
        spatial_softmax_num_keypoints=8,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        compile_model=False,
        use_path_signature=use_path_signature,
        use_delta_signature=use_delta_signature,
        history_length=8 if use_path_signature else 0,
        signature_dim=signature_dim if use_path_signature else 0,
        signature_hidden_dim=16 if use_path_signature else 512,
        signature_dropout=0.0,
        use_prefix_sequence_training=use_prefix_sequence_training,
        prefix_train_max_steps=4,
        prefix_frame_stride=1,
        prefix_pad_value=0.0,
        use_visual_prefix_memory=use_visual_prefix_memory,
        use_signature_indexed_slot_memory=use_signature_indexed_slot_memory,
        slot_memory_num_slots=2,
        slot_memory_routing_hidden_dim=16,
        slot_memory_use_delta_routing=use_delta_signature,
        slot_memory_use_softmax_routing=True,
        slot_memory_use_readout_pooling=True,
        prism_adapter_hidden_dim=32,
        prism_adapter_zero_init=prism_adapter_zero_init,
    )
    return PrismDiffusionPolicy(cfg)


def _make_online_batch(
    *,
    state_dim: int = 4,
    image_hw: int = 16,
    signature_dim: int = 8,
    use_path_signature: bool = False,
    use_delta_signature: bool = False,
) -> dict[str, torch.Tensor]:
    batch = {
        "observation.state": torch.randn(1, state_dim),
        CAMERA_KEY: torch.randn(1, 3, image_hw, image_hw),
    }
    if use_path_signature:
        batch[PATH_SIGNATURE_KEY] = torch.randn(1, signature_dim)
    if use_delta_signature:
        batch[DELTA_SIGNATURE_KEY] = torch.randn(1, signature_dim)
    return batch


def test_prism_diffusion_config_and_policy_import_smoke() -> None:
    policy = _make_policy()

    assert isinstance(policy.config, PrismDiffusionConfig)
    assert isinstance(policy, PrismDiffusionPolicy)
    assert policy.name == "prism_diffusion"


def test_prism_diffusion_eval_parser_accepts_signature_backend() -> None:
    parser = build_eval_parser(["--policy", "prism_diffusion"])
    args = parser.parse_args(
        [
            "--policy",
            "prism_diffusion",
            "--signature-backend",
            "simple",
        ]
    )

    assert args.signature_backend == "simple"


def test_prism_diffusion_signature_affects_output() -> None:
    torch.manual_seed(0)
    policy = _make_policy(
        use_path_signature=True,
        prism_adapter_zero_init=False,
    )
    policy.eval()

    noise = torch.randn(1, 4, 2)
    batch = _make_online_batch(use_path_signature=True)
    batch_shifted = {key: value.clone() for key, value in batch.items()}
    batch_shifted[PATH_SIGNATURE_KEY] = batch_shifted[PATH_SIGNATURE_KEY] + 2.0

    policy.reset()
    action_1 = policy.select_action(batch, noise=noise)
    policy.reset()
    action_2 = policy.select_action(batch_shifted, noise=noise)

    assert action_1.shape == (1, 2)
    assert action_2.shape == (1, 2)
    assert not torch.allclose(action_1, action_2)


def test_prism_diffusion_prefix_dataset_aligns_current_step() -> None:
    dataset = PrismDiffusionPrefixDataset(
        _BaseDiffusionDataset(),
        prefix_train_max_steps=4,
        prefix_frame_stride=1,
        prefix_pad_value=-1.0,
        use_path_signature=True,
        use_delta_signature=True,
    )

    item = dataset[2]

    assert item["observation.state"].shape == (3, 2)
    assert item[PREFIX_STATE_KEY].shape == (4, 2)
    assert torch.equal(item[PREFIX_MASK_KEY], torch.tensor([True, True, True, False]))
    assert torch.allclose(item[PREFIX_STATE_KEY][2], item["observation.state"][-1])
    assert torch.allclose(item[PREFIX_IMAGE_KEY][2], item[CAMERA_KEY][-1])
    assert torch.allclose(
        item[PREFIX_PATH_SIGNATURE_KEY][2],
        torch.tensor([102.0, 202.0], dtype=torch.float32),
    )
    assert torch.allclose(
        item[PREFIX_DELTA_SIGNATURE_KEY][0],
        torch.zeros(2, dtype=torch.float32),
    )


def test_prism_diffusion_slot_memory_forward_smoke() -> None:
    torch.manual_seed(0)
    policy = _make_policy(
        state_dim=17,
        action_dim=4,
        image_hw=32,
        n_obs_steps=2,
        horizon=4,
        use_path_signature=True,
        use_delta_signature=True,
        use_prefix_sequence_training=True,
        use_visual_prefix_memory=True,
        use_signature_indexed_slot_memory=True,
        prism_adapter_zero_init=True,
    )
    policy.train()

    batch_size = 2
    batch = {
        "observation.state": torch.randn(batch_size, 2, 17),
        CAMERA_KEY: torch.randn(batch_size, 2, 3, 32, 32),
        PATH_SIGNATURE_KEY: torch.randn(batch_size, 8),
        DELTA_SIGNATURE_KEY: torch.randn(batch_size, 8),
        PREFIX_STATE_KEY: torch.randn(batch_size, 4, 17),
        PREFIX_PATH_SIGNATURE_KEY: torch.randn(batch_size, 4, 8),
        PREFIX_DELTA_SIGNATURE_KEY: torch.randn(batch_size, 4, 8),
        PREFIX_MASK_KEY: torch.tensor([[True, True, True, True], [True, True, False, False]]),
        PREFIX_IMAGE_KEY: torch.randn(batch_size, 4, 3, 32, 32),
        "action": torch.randn(batch_size, 4, 4),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }

    loss, _ = policy(batch)

    assert torch.isfinite(loss)
    assert policy.diffusion.prism_cond_dim == 32


def test_prism_diffusion_online_reset_clears_memory_state() -> None:
    torch.manual_seed(0)
    policy = _make_policy(
        state_dim=17,
        action_dim=4,
        image_hw=32,
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        use_path_signature=True,
        use_delta_signature=True,
        use_prefix_sequence_training=True,
        use_visual_prefix_memory=True,
        use_signature_indexed_slot_memory=True,
        prism_adapter_zero_init=True,
    )
    policy.eval()

    batch_t0 = _make_online_batch(
        state_dim=17,
        image_hw=32,
        use_path_signature=True,
        use_delta_signature=True,
    )
    batch_t1 = _make_online_batch(
        state_dim=17,
        image_hw=32,
        use_path_signature=True,
        use_delta_signature=True,
    )

    with pytest.warns(UserWarning, match="n_action_steps>1"):
        action_t0 = policy.select_action(batch_t0)
    action_t1 = policy.select_action(batch_t1)

    assert action_t0.shape == (1, 4)
    assert action_t1.shape == (1, 4)
    assert policy._prism_memory_state is not None
    assert policy._prism_memory_update_count == 2
    assert policy.get_prism_memory_debug_stats()["initialized"] is True

    policy.reset()

    assert policy._prism_memory_state is None
    assert policy._prism_memory_update_count == 0
    assert policy._prism_memory_last_state_norm == 0.0
    assert len(policy._queues[ACTION]) == 0
    assert len(policy._queues[OBS_STATE]) == 0
    assert policy.get_prism_memory_debug_stats()["initialized"] is False
