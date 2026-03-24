#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

import lerobot.policies  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig

FIRST_FRAME_ANCHOR_KEY = "observation.anchor_image"

existing_act_config = PreTrainedConfig.get_known_choices().get("act")
if existing_act_config is not None and existing_act_config.__module__.startswith("lerobot."):
    PreTrainedConfig.get_known_choices().pop("act")


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        use_path_signature: Whether to enable path-signature-based historical context encoding for
            streaming inference/training.
        history_length: Number of past timesteps to include when computing path signatures.
        signature_dim: Dimension of the computed signature feature vector. This is often determined by
            signature channels and truncation depth; set to 0 to indicate "infer from implementation".
        signature_depth: Truncation level used when computing the path signature.
        signature_hidden_dim: Hidden dimension used by the projection MLP that maps high-dimensional
            signature vectors into model-friendly embeddings.
        signature_dropout: Dropout used in the signature projection MLP for regularization.
        use_first_frame_anchor: Whether to consume an episode-constant first-frame anchor image
            from `observation.anchor_image` and inject it as one extra encoder memory token.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Extra conditioning inputs.
    use_first_frame_anchor: bool = False
    use_path_signature: bool = False

    # Streaming path-signature history module.
    history_length: int = 10
    # Signature output dimensionality is typically a function of channels and truncation depth.
    # Set to 0 to infer automatically in the model implementation.
    signature_dim: int = 0
    signature_depth: int = 3
    signature_hidden_dim: int = 512
    signature_dropout: float = 0.1

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.use_path_signature:
            if self.history_length <= 0:
                raise ValueError(f"`history_length` must be > 0. Got {self.history_length}.")
            if self.signature_dim < 0:
                raise ValueError(f"`signature_dim` must be >= 0. Got {self.signature_dim}.")
            if self.signature_depth <= 0:
                raise ValueError(f"`signature_depth` must be > 0. Got {self.signature_depth}.")
            if self.signature_hidden_dim <= 0:
                raise ValueError(
                    f"`signature_hidden_dim` must be > 0. Got {self.signature_hidden_dim}."
                )
            if not (0.0 <= self.signature_dropout <= 1.0):
                raise ValueError(
                    f"`signature_dropout` must be in [0, 1]. Got {self.signature_dropout}."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.visual_observation_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        if self.use_first_frame_anchor:
            if not self.visual_observation_features:
                raise ValueError(
                    "`use_first_frame_anchor=True` requires at least one regular observation image "
                    "feature in addition to the anchor image."
                )
            anchor_feature = self.first_frame_anchor_feature
            if anchor_feature is None:
                raise ValueError(
                    "`use_first_frame_anchor=True` requires dataset/config input feature "
                    f"`{FIRST_FRAME_ANCHOR_KEY}`."
                )
            first_visual_feature = next(iter(self.visual_observation_features.values()))
            if tuple(anchor_feature.shape) != tuple(first_visual_feature.shape):
                raise ValueError(
                    "First-frame anchor image must match the regular observation image shape. "
                    f"Got anchor={tuple(anchor_feature.shape)} vs "
                    f"observation={tuple(first_visual_feature.shape)}."
                )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def visual_observation_features(self) -> dict:
        return {
            key: ft
            for key, ft in self.image_features.items()
            if key != FIRST_FRAME_ANCHOR_KEY
        }

    @property
    def first_frame_anchor_feature(self):
        return self.image_features.get(FIRST_FRAME_ANCHOR_KEY)


@PreTrainedConfig.register_subclass("streaming_act")
@dataclass
class StreamingACTConfig(ACTConfig):
    use_path_signature: bool = True
