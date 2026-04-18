#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot_policy_streaming_act.prefix_sequence import (
    DELTA_SIGNATURE_KEY,
    PATH_SIGNATURE_KEY,
    PREFIX_DELTA_SIGNATURE_KEY,
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
    is_prefix_image_key,
)


@PreTrainedConfig.register_subclass("prism_diffusion")
@dataclass
class PrismDiffusionConfig(PreTrainedConfig):
    """Configuration class for PrismDiffusionPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Diffusion model action prediction size as detailed in `PrismDiffusionPolicy.select_action`.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `PrismDiffusionPolicy.select_action` for more details.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        resize_shape: (H, W) shape to resize images to as a preprocessing step for the vision
            backbone. If None, no resizing is done and the original image resolution is used.
        crop_ratio: Ratio in (0, 1] used to derive the crop size from resize_shape
            (crop_h = int(resize_shape[0] * crop_ratio), likewise for width).
            Set to 1.0 to disable cropping. Only takes effect when resize_shape is not None.
        crop_shape: (H, W) shape to crop images to. When resize_shape is set and crop_ratio < 1.0,
            this is computed automatically. Can also be set directly for legacy configs that use
            crop-only (without resize). If None and no derivation applies, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center
            crop in eval mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoder_per_camera: Whether to use a separate RGB encoder for each camera view.
        down_dims: Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the diffusion modeling Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim: The Unet is conditioned on the diffusion timestep via a small non-linear
            network. This is the output dimension of that network, i.e., the embedding dimension.
        use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) is used for the Unet conditioning.
            Bias modulation is used be default, while this parameter indicates whether to also use scale
            modulation.
        noise_scheduler_type: Name of the noise scheduler to use. Supported options: ["DDPM", "DDIM"].
        num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
        beta_schedule: Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
        beta_start: Beta value for the first forward-diffusion step.
        beta_end: Beta value for the last forward-diffusion step.
        prediction_type: The type of prediction that the diffusion modeling Unet makes. Choose from "epsilon"
            or "sample". These have equivalent outcomes from a latent variable modeling perspective, but
            "epsilon" has been shown to work better in many deep neural network settings.
        clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
            denoising step at inference time. WARNING: you will need to make sure your action-space is
            normalized to fit within this range.
        clip_sample_range: The magnitude of the clipping range as described above.
        num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
            spaced). If not provided, this defaults to be the same as `num_train_timesteps`.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for more information. Note, this defaults
            to False as the original Diffusion Policy implementation does the same.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Optimization
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Processor/runtime controls
    pre_normalized_observation_keys: tuple[str, ...] = field(default_factory=tuple)

    # Optional PRISM extensions. These default to the baseline diffusion behavior
    # and are serialized so checkpoint configs can round-trip local PRISM settings.
    use_path_signature: bool = False
    use_delta_signature: bool = False
    history_length: int = 0
    signature_dim: int = 0
    signature_depth: int = 3
    signature_hidden_dim: int = 512
    signature_dropout: float = 0.1
    use_prefix_sequence_training: bool = False
    prefix_train_max_steps: int = 32
    prefix_frame_stride: int = 1
    prefix_pad_value: float = 0.0
    use_visual_prefix_memory: bool = False
    use_signature_indexed_slot_memory: bool = False
    slot_memory_num_slots: int = 4
    slot_memory_routing_hidden_dim: int = 512
    slot_memory_use_delta_routing: bool = False
    slot_memory_use_softmax_routing: bool = True
    slot_memory_use_readout_pooling: bool = True
    slot_memory_balance_loss_coef: float = 0.0
    slot_memory_consistency_loss_coef: float = 0.0
    prism_adapter_hidden_dim: int = 512
    prism_adapter_zero_init: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                # Explicitly disable cropping for resize+ratio path when crop_ratio == 1.0.
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )
        if self.use_path_signature:
            if self.history_length <= 0:
                raise ValueError(f"`history_length` must be > 0. Got {self.history_length}.")
            if self.signature_dim <= 0:
                raise ValueError(f"`signature_dim` must be > 0. Got {self.signature_dim}.")
            if self.signature_depth <= 0:
                raise ValueError(f"`signature_depth` must be > 0. Got {self.signature_depth}.")
            if self.signature_hidden_dim <= 0:
                raise ValueError(f"`signature_hidden_dim` must be > 0. Got {self.signature_hidden_dim}.")
            if not (0.0 <= self.signature_dropout <= 1.0):
                raise ValueError(
                    f"`signature_dropout` must be in [0, 1]. Got {self.signature_dropout}."
                )
        if self.use_delta_signature and not self.use_path_signature:
            raise ValueError(
                "`use_delta_signature=True` requires `use_path_signature=True` because "
                "delta signatures are differences between path signatures."
            )
        if self.use_prefix_sequence_training:
            if self.prefix_train_max_steps <= 0:
                raise ValueError(
                    "`prefix_train_max_steps` must be > 0 when "
                    f"`use_prefix_sequence_training=True`. Got {self.prefix_train_max_steps}."
                )
            if self.prefix_frame_stride <= 0:
                raise ValueError(
                    "`prefix_frame_stride` must be > 0 when "
                    f"`use_prefix_sequence_training=True`. Got {self.prefix_frame_stride}."
                )
        if self.use_visual_prefix_memory:
            if not self.use_prefix_sequence_training:
                raise ValueError(
                    "`use_visual_prefix_memory=True` requires `use_prefix_sequence_training=True` "
                    "so training can reconstruct PRISM memory via prefix scan."
                )
            if self.slot_memory_num_slots <= 0:
                raise ValueError(
                    "`slot_memory_num_slots` must be > 0 when PRISM memory is enabled. "
                    f"Got {self.slot_memory_num_slots}."
                )
        if self.use_signature_indexed_slot_memory:
            if not self.use_visual_prefix_memory:
                raise ValueError(
                    "`use_signature_indexed_slot_memory=True` requires `use_visual_prefix_memory=True`."
                )
            if not self.use_path_signature:
                raise ValueError(
                    "`use_signature_indexed_slot_memory=True` requires `use_path_signature=True`."
                )
            if self.slot_memory_routing_hidden_dim <= 0:
                raise ValueError(
                    "`slot_memory_routing_hidden_dim` must be > 0 when slot memory routing is enabled. "
                    f"Got {self.slot_memory_routing_hidden_dim}."
                )
            if self.slot_memory_use_delta_routing and not self.use_delta_signature:
                raise ValueError(
                    "`slot_memory_use_delta_routing=True` requires `use_delta_signature=True`."
                )
            if self.slot_memory_balance_loss_coef < 0.0:
                raise ValueError(
                    "`slot_memory_balance_loss_coef` must be >= 0.0. "
                    f"Got {self.slot_memory_balance_loss_coef}."
                )
            if self.slot_memory_consistency_loss_coef < 0.0:
                raise ValueError(
                    "`slot_memory_consistency_loss_coef` must be >= 0.0. "
                    f"Got {self.slot_memory_consistency_loss_coef}."
                )
        if self.prism_adapter_hidden_dim <= 0:
            raise ValueError(
                f"`prism_adapter_hidden_dim` must be > 0. Got {self.prism_adapter_hidden_dim}."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )
        if self.use_path_signature:
            path_signature_feature = self.path_signature_feature
            if path_signature_feature is None:
                raise ValueError(
                    f"`use_path_signature=True` requires input feature `{PATH_SIGNATURE_KEY}`."
                )
            expected_shape = (self.signature_dim,)
            if tuple(path_signature_feature.shape) != expected_shape:
                raise ValueError(
                    "Path-signature feature shape mismatch. "
                    f"Expected {expected_shape}, got {tuple(path_signature_feature.shape)}."
                )
        if self.use_delta_signature:
            delta_signature_feature = self.delta_signature_feature
            if delta_signature_feature is None:
                raise ValueError(
                    f"`use_delta_signature=True` requires input feature `{DELTA_SIGNATURE_KEY}`."
                )
            expected_shape = (self.signature_dim,)
            if tuple(delta_signature_feature.shape) != expected_shape:
                raise ValueError(
                    "Delta-signature feature shape mismatch. "
                    f"Expected {expected_shape}, got {tuple(delta_signature_feature.shape)}."
                )
        if self.use_prefix_sequence_training:
            if self.robot_state_feature is None:
                raise ValueError(
                    "`use_prefix_sequence_training=True` requires a current `observation.state` feature."
                )
            prefix_state_feature = self.prefix_state_feature
            if prefix_state_feature is None:
                raise ValueError(
                    "`use_prefix_sequence_training=True` requires input feature "
                    f"`{PREFIX_STATE_KEY}`."
                )
            expected_prefix_state_shape = (self.prefix_train_max_steps, self.robot_state_feature.shape[0])
            if tuple(prefix_state_feature.shape) != expected_prefix_state_shape:
                raise ValueError(
                    "Prefix-state feature shape mismatch. "
                    f"Expected {expected_prefix_state_shape}, got {tuple(prefix_state_feature.shape)}."
                )
            prefix_mask_feature = self.prefix_mask_feature
            if prefix_mask_feature is None:
                raise ValueError(
                    "`use_prefix_sequence_training=True` requires input feature "
                    f"`{PREFIX_MASK_KEY}`."
                )
            if tuple(prefix_mask_feature.shape) != (self.prefix_train_max_steps,):
                raise ValueError(
                    "Prefix-mask feature shape mismatch. "
                    f"Expected {(self.prefix_train_max_steps,)}, got {tuple(prefix_mask_feature.shape)}."
                )
            if not self.prefix_image_features:
                raise ValueError(
                    "`use_prefix_sequence_training=True` requires at least one "
                    "`observation.prefix_images.*` feature."
                )
            if len(self.prefix_image_features) != len(self.image_features):
                raise ValueError(
                    "Prefix image feature count must match the number of current observation cameras. "
                    f"Got prefix={len(self.prefix_image_features)} vs current={len(self.image_features)}."
                )

            current_image_shapes: dict[str, tuple[int, ...]] = {}
            for key, feature in self.image_features.items():
                if key.startswith("observation.images."):
                    suffix = key.removeprefix("observation.images.")
                elif key == "observation.image":
                    suffix = "main"
                else:
                    raise ValueError(
                        "Unsupported current image feature name for prefix-sequence mode. "
                        f"Got `{key}`."
                    )
                current_image_shapes[suffix] = tuple(feature.shape)

            for prefix_key, prefix_feature in self.prefix_image_features.items():
                suffix = prefix_key.removeprefix("observation.prefix_images.")
                current_shape = current_image_shapes.get(suffix)
                if current_shape is None:
                    raise ValueError(
                        "Prefix image feature does not map to any regular observation image. "
                        f"Got prefix key `{prefix_key}`."
                    )
                expected_prefix_shape = (self.prefix_train_max_steps, *current_shape)
                if tuple(prefix_feature.shape) != expected_prefix_shape:
                    raise ValueError(
                        "Prefix image feature shape mismatch. "
                        f"Expected {expected_prefix_shape} for `{prefix_key}`, "
                        f"got {tuple(prefix_feature.shape)}."
                    )

            if self.use_path_signature:
                prefix_path_signature_feature = self.prefix_path_signature_feature
                if prefix_path_signature_feature is None:
                    raise ValueError(
                        "`use_prefix_sequence_training=True` requires input feature "
                        f"`{PREFIX_PATH_SIGNATURE_KEY}` when `use_path_signature=True`."
                    )
                expected_shape = (self.prefix_train_max_steps, self.signature_dim)
                if tuple(prefix_path_signature_feature.shape) != expected_shape:
                    raise ValueError(
                        "Prefix path-signature feature shape mismatch. "
                        f"Expected {expected_shape}, got {tuple(prefix_path_signature_feature.shape)}."
                    )
            if self.use_delta_signature:
                prefix_delta_signature_feature = self.prefix_delta_signature_feature
                if prefix_delta_signature_feature is None:
                    raise ValueError(
                        "`use_prefix_sequence_training=True` requires input feature "
                        f"`{PREFIX_DELTA_SIGNATURE_KEY}` when `use_delta_signature=True`."
                    )
                expected_shape = (self.prefix_train_max_steps, self.signature_dim)
                if tuple(prefix_delta_signature_feature.shape) != expected_shape:
                    raise ValueError(
                        "Prefix delta-signature feature shape mismatch. "
                        f"Expected {expected_shape}, got {tuple(prefix_delta_signature_feature.shape)}."
                    )
        if self.use_visual_prefix_memory:
            if not self.image_features:
                raise ValueError(
                    "`use_visual_prefix_memory=True` requires at least one regular observation image feature."
                )
            if self.robot_state_feature is None:
                raise ValueError(
                    "`use_visual_prefix_memory=True` requires `observation.state`."
                )

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if ft.type is FeatureType.VISUAL
            and not key.startswith("observation.prefix_images.")
        }

    @property
    def path_signature_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(PATH_SIGNATURE_KEY)

    @property
    def delta_signature_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(DELTA_SIGNATURE_KEY)

    @property
    def prefix_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_STATE_KEY)

    @property
    def prefix_path_signature_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_PATH_SIGNATURE_KEY)

    @property
    def prefix_delta_signature_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_DELTA_SIGNATURE_KEY)

    @property
    def prefix_mask_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_MASK_KEY)

    @property
    def prefix_image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {
            key: ft for key, ft in self.input_features.items() if is_prefix_image_key(key)
        }

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
