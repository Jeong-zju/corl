from __future__ import annotations

import sys
from pathlib import Path

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
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from policy_defaults import load_policy_mode_defaults_for_dataset
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
