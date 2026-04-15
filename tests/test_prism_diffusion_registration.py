from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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
