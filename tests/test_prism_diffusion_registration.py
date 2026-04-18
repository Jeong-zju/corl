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
        / "lerobot_policy_prism_diffusion"
        / "src"
    ),
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from policy_defaults import load_policy_mode_defaults_for_dataset
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
from lerobot_policy_prism_diffusion.configuration_diffusion import (
    PrismDiffusionConfig,
)
from lerobot_policy_prism_diffusion.modeling_diffusion import PrismDiffusionPolicy


def test_prism_diffusion_config_and_policy_registration() -> None:
    assert "prism_diffusion" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("prism_diffusion") is PrismDiffusionConfig
    assert get_policy_class("prism_diffusion") is PrismDiffusionPolicy


def test_prism_diffusion_processor_factory_dynamic_discovery() -> None:
    cfg = PrismDiffusionConfig(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            "observation.images.main": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 16, 16),
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
    )

    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=cfg)

    assert preprocessor.name == POLICY_PREPROCESSOR_DEFAULT_NAME
    assert postprocessor.name == POLICY_POSTPROCESSOR_DEFAULT_NAME


def test_prism_diffusion_config_round_trips_prism_fields(tmp_path: Path) -> None:
    cfg = PrismDiffusionConfig(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            "observation.images.main": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 16, 16),
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        use_path_signature=True,
        use_delta_signature=True,
        history_length=12,
        signature_dim=24,
        signature_depth=4,
        signature_hidden_dim=96,
        signature_dropout=0.2,
        use_prefix_sequence_training=True,
        prefix_train_max_steps=20,
        prefix_frame_stride=2,
        prefix_pad_value=-1.0,
        use_visual_prefix_memory=True,
        use_signature_indexed_slot_memory=True,
        slot_memory_num_slots=6,
        slot_memory_routing_hidden_dim=128,
        slot_memory_use_delta_routing=True,
        slot_memory_use_softmax_routing=False,
        slot_memory_use_readout_pooling=False,
        slot_memory_balance_loss_coef=0.25,
        slot_memory_consistency_loss_coef=0.5,
        prism_adapter_hidden_dim=192,
        prism_adapter_zero_init=False,
    )

    save_dir = tmp_path / "prism_diffusion_cfg"
    cfg.save_pretrained(save_dir)
    loaded = PreTrainedConfig.from_pretrained(save_dir, local_files_only=True)

    assert isinstance(loaded, PrismDiffusionConfig)
    assert loaded.use_path_signature is True
    assert loaded.use_delta_signature is True
    assert loaded.history_length == 12
    assert loaded.signature_dim == 24
    assert loaded.signature_depth == 4
    assert loaded.signature_hidden_dim == 96
    assert loaded.signature_dropout == 0.2
    assert loaded.use_prefix_sequence_training is True
    assert loaded.prefix_train_max_steps == 20
    assert loaded.prefix_frame_stride == 2
    assert loaded.prefix_pad_value == -1.0
    assert loaded.use_visual_prefix_memory is True
    assert loaded.use_signature_indexed_slot_memory is True
    assert loaded.slot_memory_num_slots == 6
    assert loaded.slot_memory_routing_hidden_dim == 128
    assert loaded.slot_memory_use_delta_routing is True
    assert loaded.slot_memory_use_softmax_routing is False
    assert loaded.slot_memory_use_readout_pooling is False
    assert loaded.slot_memory_balance_loss_coef == 0.25
    assert loaded.slot_memory_consistency_loss_coef == 0.5
    assert loaded.prism_adapter_hidden_dim == 192
    assert loaded.prism_adapter_zero_init is False


def test_prism_diffusion_dataset_defaults_resolve() -> None:
    defaults, path = load_policy_mode_defaults_for_dataset(
        mode="train",
        dataset_selector="zeno-ai/day3_5_Exp1_processed",
        policy_name="prism_diffusion",
    )

    assert path is not None
    assert path.name == "prism_diffusion.yaml"
    assert defaults["output_root"].endswith("prism-diffusion")
    assert defaults["use_path_signature"] is False
    assert defaults["prism_adapter_zero_init"] is True


def _make_prism_diffusion_smoke_policy(*, n_action_steps: int) -> PrismDiffusionPolicy:
    camera_key = "observation.images.main"
    input_features = build_prefix_sequence_input_features(
        base_input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(17,)),
            camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            PATH_SIGNATURE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            DELTA_SIGNATURE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        prefix_train_max_steps=4,
        use_path_signature=True,
        use_delta_signature=True,
    )
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }
    cfg = PrismDiffusionConfig(
        device="cpu",
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=2,
        horizon=8,
        n_action_steps=n_action_steps,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=4,
        diffusion_step_embed_dim=32,
        spatial_softmax_num_keypoints=8,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        compile_model=False,
        use_path_signature=True,
        use_delta_signature=True,
        history_length=8,
        signature_dim=8,
        signature_hidden_dim=16,
        signature_dropout=0.0,
        use_prefix_sequence_training=True,
        prefix_train_max_steps=4,
        prefix_frame_stride=1,
        prefix_pad_value=0.0,
        use_visual_prefix_memory=True,
        use_signature_indexed_slot_memory=True,
        slot_memory_num_slots=2,
        slot_memory_routing_hidden_dim=16,
        slot_memory_use_delta_routing=True,
        slot_memory_use_softmax_routing=True,
        slot_memory_use_readout_pooling=True,
        prism_adapter_hidden_dim=32,
        prism_adapter_zero_init=True,
    )
    return PrismDiffusionPolicy(cfg)


def test_prism_diffusion_prism_forward_prefix_scan_smoke() -> None:
    torch.manual_seed(0)
    policy = _make_prism_diffusion_smoke_policy(n_action_steps=1)
    policy.train()

    batch_size = 2
    camera_key = "observation.images.main"
    prefix_key = prefix_image_key_from_camera_key(camera_key)
    batch = {
        "observation.state": torch.randn(batch_size, 2, 17),
        camera_key: torch.randn(batch_size, 2, 3, 32, 32),
        PATH_SIGNATURE_KEY: torch.randn(batch_size, 8),
        DELTA_SIGNATURE_KEY: torch.randn(batch_size, 8),
        PREFIX_STATE_KEY: torch.randn(batch_size, 4, 17),
        PREFIX_PATH_SIGNATURE_KEY: torch.randn(batch_size, 4, 8),
        PREFIX_DELTA_SIGNATURE_KEY: torch.randn(batch_size, 4, 8),
        PREFIX_MASK_KEY: torch.tensor([[True, True, True, True], [True, True, False, False]]),
        prefix_key: torch.randn(batch_size, 4, 3, 32, 32),
        "action": torch.randn(batch_size, 8, 4),
        "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
    }

    loss, _ = policy(batch)

    assert torch.isfinite(loss)
    assert policy.diffusion.prism_cond_dim == 32


def test_prism_diffusion_online_select_action_smoke_and_reset() -> None:
    torch.manual_seed(0)
    policy = _make_prism_diffusion_smoke_policy(n_action_steps=2)
    policy.eval()

    camera_key = "observation.images.main"
    batch_t0 = {
        "observation.state": torch.randn(1, 17),
        camera_key: torch.randn(1, 3, 32, 32),
        PATH_SIGNATURE_KEY: torch.randn(1, 8),
        DELTA_SIGNATURE_KEY: torch.randn(1, 8),
    }
    batch_t1 = {
        "observation.state": torch.randn(1, 17),
        camera_key: torch.randn(1, 3, 32, 32),
        PATH_SIGNATURE_KEY: torch.randn(1, 8),
        DELTA_SIGNATURE_KEY: torch.randn(1, 8),
    }

    with pytest.warns(UserWarning, match="n_action_steps>1"):
        action_t0 = policy.select_action(batch_t0)
    action_t1 = policy.select_action(batch_t1)

    assert action_t0.shape == (1, 4)
    assert action_t1.shape == (1, 4)
    assert policy._prism_memory_state is not None
    assert tuple(policy._prism_memory_state.shape) == (1, 2, 32)
    assert policy._prism_memory_update_count == 2
    assert policy.get_prism_memory_debug_stats()["initialized"] is True

    policy.reset()

    assert policy._prism_memory_state is None
    assert policy._prism_memory_update_count == 0
    assert policy._prism_memory_last_state_norm == 0.0
    assert len(policy._queues[ACTION]) == 0
    assert len(policy._queues[OBS_STATE]) == 0
