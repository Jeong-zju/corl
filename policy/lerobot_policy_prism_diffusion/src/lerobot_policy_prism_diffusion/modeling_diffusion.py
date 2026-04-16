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
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
import warnings
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot_policy_streaming_act.prefix_sequence import (
    DELTA_SIGNATURE_KEY,
    PATH_SIGNATURE_KEY,
    PREFIX_DELTA_SIGNATURE_KEY,
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
)
from .configuration_diffusion import PrismDiffusionConfig


class PrismDiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://huggingface.co/papers/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = PrismDiffusionConfig
    name = "prism_diffusion"

    def __init__(
        self,
        config: PrismDiffusionConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def _prepare_observation_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.config.image_features:
            return batch
        if OBS_IMAGES in batch and not all(key in batch for key in self.config.image_features):
            return batch
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return batch

    def _warn_if_multi_step_prism_queue(self) -> None:
        if (
            not self.config.use_visual_prefix_memory
            or self.config.n_action_steps <= 1
            or self._prism_multi_step_warning_emitted
        ):
            return
        warnings.warn(
            "`prism_diffusion` is running with `n_action_steps>1`. Online PRISM memory is updated every "
            "observation step, but newly updated memory does not affect actions already queued for "
            "execution. For the most responsive streaming behavior, prefer `n_action_steps=1`.",
            stacklevel=2,
        )
        self._prism_multi_step_warning_emitted = True

    @torch.no_grad()
    def _update_online_prism_memory(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.config.use_visual_prefix_memory:
            return None
        memory_state = self.diffusion.update_online_prism_memory(
            batch,
            previous_state=self._prism_memory_state,
        )
        self._prism_memory_state = memory_state.detach()
        self._prism_memory_update_count += 1
        self._prism_memory_last_state_norm = float(
            self._prism_memory_state.norm(dim=-1).mean().detach().cpu().item()
        )
        return self._prism_memory_state

    def reset(self):
        """Clear observation/action queues and PRISM memory state on `env.reset()`."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        self._prism_memory_state = None
        self._prism_memory_update_count = 0
        self._prism_memory_last_state_norm = 0.0
        self._prism_multi_step_warning_emitted = False

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        *,
        prism_memory_state: Tensor | None = None,
    ) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = self._prepare_observation_batch(batch)
        if self.config.use_visual_prefix_memory and prism_memory_state is None:
            prism_memory_state = self._update_online_prism_memory(batch)

        queued_batch = {
            key: torch.stack(list(self._queues[key]), dim=1)
            for key in (OBS_STATE, OBS_IMAGES, OBS_ENV_STATE)
            if key in self._queues
        }
        for passthrough_key in (PATH_SIGNATURE_KEY, DELTA_SIGNATURE_KEY):
            if passthrough_key in batch:
                queued_batch[passthrough_key] = batch[passthrough_key]

        actions = self.diffusion.generate_actions(
            queued_batch,
            noise=noise,
            prism_memory_state=prism_memory_state,
        )

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        batch = self._prepare_observation_batch(batch)
        self._warn_if_multi_step_prism_queue()
        prism_memory_state = self._update_online_prism_memory(batch)

        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(
                batch,
                noise=noise,
                prism_memory_state=prism_memory_state,
            )
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def get_prism_memory_debug_stats(self) -> dict[str, float | int | bool]:
        state = self._prism_memory_state
        return {
            "enabled": bool(self.config.use_visual_prefix_memory),
            "initialized": state is not None,
            "num_slots": int(self.config.slot_memory_num_slots),
            "signature_indexed_slot_memory": bool(self.config.use_signature_indexed_slot_memory),
            "uses_path_signature": bool(self.config.use_path_signature),
            "uses_delta_signature": bool(self.config.use_delta_signature),
            "update_count": int(self._prism_memory_update_count),
            "state_norm": float(self._prism_memory_last_state_norm),
        }

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.diffusion.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: PrismDiffusionConfig):
        super().__init__()
        self.config = config
        self.use_path_signature = bool(config.use_path_signature)
        self.use_delta_signature = bool(config.use_delta_signature)
        self.use_visual_prefix_memory = bool(config.use_visual_prefix_memory)
        self.use_signature_indexed_slot_memory = bool(config.use_signature_indexed_slot_memory)
        self.prism_enabled = (
            self.use_path_signature or self.use_delta_signature or self.use_visual_prefix_memory
        )
        self.prism_dim = int(config.prism_adapter_hidden_dim) if self.prism_enabled else 0
        self.active_prism_num_slots = int(config.slot_memory_num_slots)

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        self.rgb_feature_dim = 0
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                self.rgb_feature_dim = encoders[0].feature_dim
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                self.rgb_feature_dim = self.rgb_encoder.feature_dim
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.native_global_cond_dim = global_cond_dim * config.n_obs_steps
        self.prism_cond_dim = self.prism_dim if self.prism_enabled else 0

        if self.use_path_signature:
            self.signature_proj = nn.Sequential(
                nn.Linear(config.signature_dim, config.signature_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.signature_dropout),
                nn.Linear(config.signature_hidden_dim, self.prism_dim),
            )
        else:
            self.signature_proj = None
        if self.use_delta_signature:
            self.delta_signature_proj = nn.Sequential(
                nn.Linear(config.signature_dim, config.signature_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.signature_dropout),
                nn.Linear(config.signature_hidden_dim, self.prism_dim),
            )
        else:
            self.delta_signature_proj = None

        if self.prism_enabled:
            self.prism_state_proj = nn.Linear(self.config.robot_state_feature.shape[0], self.prism_dim)
            self.prism_visual_proj = (
                nn.Linear(self.rgb_feature_dim, self.prism_dim) if self.config.image_features else None
            )
        else:
            self.prism_state_proj = None
            self.prism_visual_proj = None

        if self.use_visual_prefix_memory:
            step_input_dim = self.prism_dim * (
                2 + int(self.use_path_signature) + int(self.use_delta_signature)
            )
            if self.use_signature_indexed_slot_memory:
                route_input_dim = self.prism_dim * (
                    1 + int(self.config.slot_memory_use_delta_routing)
                )
                slot_state_input_dim = self.prism_dim * 2 + self.config.slot_memory_routing_hidden_dim
                self.slot_memory_route_proj = nn.Sequential(
                    nn.Linear(route_input_dim, self.config.slot_memory_routing_hidden_dim),
                    nn.GELU(),
                    nn.Linear(
                        self.config.slot_memory_routing_hidden_dim,
                        self.config.slot_memory_routing_hidden_dim,
                    ),
                )
                self.slot_memory_route_query_proj = nn.Linear(
                    self.config.slot_memory_routing_hidden_dim,
                    self.config.slot_memory_routing_hidden_dim,
                    bias=False,
                )
                self.slot_memory_route_key_proj = nn.Linear(
                    self.prism_dim,
                    self.config.slot_memory_routing_hidden_dim,
                    bias=False,
                )
                self.slot_memory_write_proj = nn.Sequential(
                    nn.Linear(step_input_dim, self.prism_dim),
                    nn.GELU(),
                    nn.Linear(self.prism_dim, self.prism_dim),
                )
                self.slot_memory_candidate_proj = nn.Sequential(
                    nn.Linear(slot_state_input_dim, self.prism_dim),
                    nn.GELU(),
                    nn.Linear(self.prism_dim, self.prism_dim),
                )
                self.slot_memory_gate_proj = nn.Sequential(
                    nn.Linear(slot_state_input_dim, self.prism_dim),
                    nn.GELU(),
                    nn.Linear(self.prism_dim, 1),
                )
                self.slot_memory_read_query_proj = nn.Sequential(
                    nn.Linear(slot_state_input_dim, self.prism_dim),
                    nn.GELU(),
                    nn.Linear(self.prism_dim, self.prism_dim),
                )
                self.slot_memory_read_key_proj = nn.Linear(self.prism_dim, self.prism_dim, bias=False)
                self.slot_memory_read_value_proj = nn.Linear(self.prism_dim, self.prism_dim, bias=False)
            else:
                self.prism_memory_update = nn.GRUCell(step_input_dim, self.prism_dim)
                self.prism_memory_extra_updates = nn.ModuleList(
                    nn.GRUCell(step_input_dim, self.prism_dim)
                    for _ in range(self.active_prism_num_slots - 1)
                )

        if self.prism_enabled:
            prism_adapter_input_dim = 0
            if self.use_path_signature:
                prism_adapter_input_dim += self.prism_dim
            if self.use_delta_signature:
                prism_adapter_input_dim += self.prism_dim
            if self.use_visual_prefix_memory:
                prism_adapter_input_dim += self.prism_dim * 2
            self.prism_adapter = nn.Sequential(
                nn.Linear(prism_adapter_input_dim, config.prism_adapter_hidden_dim),
                nn.GELU(),
                nn.Linear(config.prism_adapter_hidden_dim, self.prism_cond_dim),
            )
            if config.prism_adapter_zero_init:
                nn.init.zeros_(self.prism_adapter[-1].weight)
                nn.init.zeros_(self.prism_adapter[-1].bias)
        else:
            self.prism_adapter = None

        self.unet = DiffusionConditionalUnet1d(
            config,
            global_cond_dim=self.native_global_cond_dim + self.prism_cond_dim,
        )

        if config.compile_model:
            # Compile the U-Net. "reduce-overhead" is preferred for the small-batch repetitive loops
            # common in diffusion inference.
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def _select_current_step_feature(self, tensor: Tensor, *, context: str) -> Tensor:
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim == 3:
            return tensor[:, -1]
        raise ValueError(f"{context} must have shape (B, D) or (B, T, D). Got {tuple(tensor.shape)}.")

    def _extract_current_camera_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        if OBS_IMAGES not in batch:
            raise KeyError("PRISM conditioning requires `batch[OBS_IMAGES]`.")
        images = batch[OBS_IMAGES]
        if images.ndim == 6:
            images = images[:, -1]
        elif images.ndim != 5:
            raise ValueError(
                "Current observation images must have shape (B, N, C, H, W) or (B, T, N, C, H, W). "
                f"Got {tuple(images.shape)}."
            )
        return [images[:, camera_idx] for camera_idx in range(images.shape[1])]

    def _project_prism_state_tensor(self, state: Tensor, *, context: str) -> Tensor:
        if self.prism_state_proj is None:
            raise RuntimeError(f"{context} requires `self.prism_state_proj` to be initialized.")
        if state.ndim not in {2, 3}:
            raise ValueError(f"{context} must have shape (B, D) or (B, T, D). Got {tuple(state.shape)}.")
        weight_dtype = self.prism_state_proj.weight.dtype
        if state.ndim == 2:
            return self.prism_state_proj(state.to(dtype=weight_dtype))
        batch_size, time_steps, state_dim = state.shape
        projected = self.prism_state_proj(state.reshape(batch_size * time_steps, state_dim).to(dtype=weight_dtype))
        return projected.reshape(batch_size, time_steps, self.prism_dim)

    def _project_prism_visual_tensor(self, visual: Tensor, *, context: str) -> Tensor:
        if self.prism_visual_proj is None:
            raise RuntimeError(f"{context} requires `self.prism_visual_proj` to be initialized.")
        if visual.ndim not in {2, 3}:
            raise ValueError(f"{context} must have shape (B, D) or (B, T, D). Got {tuple(visual.shape)}.")
        weight_dtype = self.prism_visual_proj.weight.dtype
        if visual.ndim == 2:
            return self.prism_visual_proj(visual.to(dtype=weight_dtype))
        batch_size, time_steps, feature_dim = visual.shape
        projected = self.prism_visual_proj(
            visual.reshape(batch_size * time_steps, feature_dim).to(dtype=weight_dtype)
        )
        return projected.reshape(batch_size, time_steps, self.prism_dim)

    def _project_signature_tensor(self, signature: Tensor, *, context: str) -> Tensor:
        if self.signature_proj is None:
            raise RuntimeError(f"{context} requires `self.signature_proj` to be initialized.")
        if signature.ndim not in {2, 3}:
            raise ValueError(f"{context} must have shape (B, D) or (B, T, D). Got {tuple(signature.shape)}.")
        if signature.shape[-1] != self.config.signature_dim:
            raise ValueError(
                f"{context} trailing dim must equal signature_dim={self.config.signature_dim}. "
                f"Got {signature.shape[-1]}."
            )
        weight_dtype = self.signature_proj[0].weight.dtype
        if signature.ndim == 2:
            return self.signature_proj(signature.to(dtype=weight_dtype))
        batch_size, time_steps, signature_dim = signature.shape
        projected = self.signature_proj(
            signature.reshape(batch_size * time_steps, signature_dim).to(dtype=weight_dtype)
        )
        return projected.reshape(batch_size, time_steps, self.prism_dim)

    def _project_delta_signature_tensor(self, delta_signature: Tensor, *, context: str) -> Tensor:
        if self.delta_signature_proj is None:
            raise RuntimeError(f"{context} requires `self.delta_signature_proj` to be initialized.")
        if delta_signature.ndim not in {2, 3}:
            raise ValueError(
                f"{context} must have shape (B, D) or (B, T, D). Got {tuple(delta_signature.shape)}."
            )
        if delta_signature.shape[-1] != self.config.signature_dim:
            raise ValueError(
                f"{context} trailing dim must equal signature_dim={self.config.signature_dim}. "
                f"Got {delta_signature.shape[-1]}."
            )
        weight_dtype = self.delta_signature_proj[0].weight.dtype
        if delta_signature.ndim == 2:
            return self.delta_signature_proj(delta_signature.to(dtype=weight_dtype))
        batch_size, time_steps, signature_dim = delta_signature.shape
        projected = self.delta_signature_proj(
            delta_signature.reshape(batch_size * time_steps, signature_dim).to(dtype=weight_dtype)
        )
        return projected.reshape(batch_size, time_steps, self.prism_dim)

    def _encode_multi_camera_images_for_prism(
        self,
        camera_images: list[Tensor],
        *,
        context: str,
    ) -> Tensor:
        if not camera_images:
            raise ValueError(f"{context} requires at least one camera tensor.")
        if not self.config.image_features:
            raise RuntimeError(f"{context} requires image features to be enabled.")

        ref_shape = tuple(camera_images[0].shape)
        ref_ndim = camera_images[0].ndim
        if ref_ndim not in {4, 5}:
            raise ValueError(
                f"{context} camera tensors must have shape (B, C, H, W) or (B, T, C, H, W). "
                f"Got {ref_shape}."
            )
        for camera_idx, images in enumerate(camera_images[1:], start=1):
            if images.ndim != ref_ndim or tuple(images.shape) != ref_shape:
                raise ValueError(
                    f"{context} camera tensor {camera_idx} shape mismatch. "
                    f"Expected {ref_shape}, got {tuple(images.shape)}."
                )

        num_cameras = len(camera_images)
        if self.config.use_separate_rgb_encoder_per_camera:
            encoded_per_camera: list[Tensor] = []
            for encoder, images in zip(self.rgb_encoder, camera_images, strict=True):
                if ref_ndim == 4:
                    encoded = encoder(images)
                else:
                    batch_size, time_steps = images.shape[:2]
                    encoded = encoder(images.reshape(batch_size * time_steps, *images.shape[2:]))
                    encoded = encoded.reshape(batch_size, time_steps, self.rgb_feature_dim)
                encoded_per_camera.append(encoded)
            encoded = torch.stack(encoded_per_camera, dim=0)
        else:
            if ref_ndim == 4:
                flat_images = torch.cat(camera_images, dim=0)
                encoded = self.rgb_encoder(flat_images)
                encoded = encoded.reshape(num_cameras, camera_images[0].shape[0], self.rgb_feature_dim)
            else:
                batch_size, time_steps = camera_images[0].shape[:2]
                flat_images = torch.cat(
                    [images.reshape(batch_size * time_steps, *images.shape[2:]) for images in camera_images],
                    dim=0,
                )
                encoded = self.rgb_encoder(flat_images)
                encoded = encoded.reshape(num_cameras, batch_size, time_steps, self.rgb_feature_dim)
        return self._project_prism_visual_tensor(encoded.mean(dim=0), context=context)

    def _build_zero_prism_memory_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.zeros(
            (batch_size, self.active_prism_num_slots, self.prism_dim),
            device=device,
            dtype=dtype,
        )

    def _normalize_prism_memory_state(
        self,
        memory_state: Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        context: str,
    ) -> Tensor:
        expected_shape = (batch_size, self.active_prism_num_slots, self.prism_dim)
        if memory_state.ndim == 2:
            if self.active_prism_num_slots != 1:
                raise ValueError(
                    f"{context} expects PRISM memory shape {expected_shape}. "
                    f"Got legacy single-slot shape {tuple(memory_state.shape)}."
                )
            memory_state = memory_state.unsqueeze(1)
        elif memory_state.ndim != 3:
            raise ValueError(f"{context} must have shape {expected_shape}. Got {tuple(memory_state.shape)}.")
        if tuple(memory_state.shape) != expected_shape:
            raise ValueError(
                f"{context} shape mismatch. Expected {expected_shape}, got {tuple(memory_state.shape)}."
            )
        return memory_state.to(device=device, dtype=dtype)

    def _iter_prism_memory_updates(self) -> list[nn.GRUCell]:
        if self.use_signature_indexed_slot_memory:
            raise RuntimeError("Legacy GRU PRISM memory updates are unavailable in slot-memory mode.")
        updates = [self.prism_memory_update, *self.prism_memory_extra_updates]
        if len(updates) != self.active_prism_num_slots:
            raise RuntimeError(
                "PRISM memory updater count must match the configured slot count. "
                f"Got updaters={len(updates)} vs slots={self.active_prism_num_slots}."
            )
        return updates

    def _build_prism_memory_step_input(
        self,
        *,
        visual_t: Tensor,
        state_t: Tensor,
        signature_t: Tensor | None,
        delta_signature_t: Tensor | None,
    ) -> Tensor:
        inputs = [visual_t, state_t]
        if signature_t is not None:
            inputs.append(signature_t)
        if delta_signature_t is not None:
            inputs.append(delta_signature_t)
        return torch.cat(inputs, dim=-1)

    def _build_slot_memory_route_features(
        self,
        *,
        signature_t: Tensor | None,
        delta_signature_t: Tensor | None,
        context: str,
    ) -> Tensor:
        if signature_t is None:
            raise ValueError(f"{context} requires `signature_t` for slot routing.")
        route_features = [signature_t]
        if self.config.slot_memory_use_delta_routing:
            if delta_signature_t is None:
                raise ValueError(
                    f"{context} requires `delta_signature_t` when `slot_memory_use_delta_routing=True`."
                )
            route_features.append(delta_signature_t)
        route_features = torch.cat(route_features, dim=-1)
        expected_dim = self.prism_dim * (1 + int(self.config.slot_memory_use_delta_routing))
        if tuple(route_features.shape[1:]) != (expected_dim,):
            raise ValueError(
                f"{context} route features must have trailing dim {expected_dim}. "
                f"Got {tuple(route_features.shape)}."
            )
        return route_features

    def _compute_signature_indexed_slot_memory_route(
        self,
        *,
        memory_prev: Tensor,
        route_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        route_hidden = self.slot_memory_route_proj(route_features)
        routing_query = self.slot_memory_route_query_proj(route_hidden).unsqueeze(1)
        routing_keys = self.slot_memory_route_key_proj(memory_prev)
        routing_logits = torch.matmul(routing_query, routing_keys.transpose(1, 2)).squeeze(1)
        routing_logits = routing_logits / math.sqrt(self.config.slot_memory_routing_hidden_dim)
        if self.config.slot_memory_use_softmax_routing:
            routing_weights = torch.softmax(routing_logits, dim=-1)
        else:
            routing_weights = torch.sigmoid(routing_logits)
        return route_hidden, routing_logits, routing_weights

    def _read_signature_indexed_slot_memory_context(
        self,
        *,
        memory_state: Tensor,
        visual_t: Tensor,
        state_t: Tensor,
        signature_t: Tensor | None,
        delta_signature_t: Tensor | None,
        route_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if route_hidden is None:
            route_features = self._build_slot_memory_route_features(
                signature_t=signature_t,
                delta_signature_t=delta_signature_t,
                context="PRISM slot-memory readout",
            )
            route_hidden = self.slot_memory_route_proj(route_features)
        read_query_input = torch.cat([visual_t, state_t, route_hidden], dim=-1)
        read_query = self.slot_memory_read_query_proj(read_query_input).unsqueeze(1)
        read_keys = self.slot_memory_read_key_proj(memory_state)
        read_values = self.slot_memory_read_value_proj(memory_state)
        read_logits = torch.matmul(read_query, read_keys.transpose(1, 2)).squeeze(1)
        read_logits = read_logits / math.sqrt(self.prism_dim)
        read_weights = torch.softmax(read_logits, dim=-1)
        readout_context = torch.sum(read_weights.unsqueeze(-1) * read_values, dim=1)
        return readout_context, read_logits, read_weights

    def _update_signature_indexed_slot_memory_step(
        self,
        *,
        memory_prev: Tensor,
        visual_t: Tensor,
        state_t: Tensor,
        signature_t: Tensor | None,
        delta_signature_t: Tensor | None,
        valid_t: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size, num_slots, hidden_dim = memory_prev.shape
        route_features = self._build_slot_memory_route_features(
            signature_t=signature_t,
            delta_signature_t=delta_signature_t,
            context="PRISM slot-memory update",
        )
        route_hidden, routing_logits, routing_weights = self._compute_signature_indexed_slot_memory_route(
            memory_prev=memory_prev,
            route_features=route_features,
        )
        write_base = self.slot_memory_write_proj(
            self._build_prism_memory_step_input(
                visual_t=visual_t,
                state_t=state_t,
                signature_t=signature_t,
                delta_signature_t=delta_signature_t,
            )
        )
        slot_state_input = torch.cat(
            [
                memory_prev,
                write_base.unsqueeze(1).expand(batch_size, num_slots, hidden_dim),
                route_hidden.unsqueeze(1).expand(
                    batch_size,
                    num_slots,
                    self.config.slot_memory_routing_hidden_dim,
                ),
            ],
            dim=-1,
        )
        candidate = self.slot_memory_candidate_proj(slot_state_input)
        gate = torch.sigmoid(self.slot_memory_gate_proj(slot_state_input))
        updated_hidden = memory_prev + routing_weights.unsqueeze(-1) * gate * (candidate - memory_prev)

        if valid_t is not None:
            valid_t = valid_t.to(dtype=torch.bool, device=memory_prev.device)
            if valid_t.shape != (batch_size,):
                raise ValueError(
                    f"PRISM slot-memory valid mask must have shape {(batch_size,)}. "
                    f"Got {tuple(valid_t.shape)}."
                )
            updated_hidden = torch.where(valid_t.view(batch_size, 1, 1), updated_hidden, memory_prev)

        readout_context, _, _ = self._read_signature_indexed_slot_memory_context(
            memory_state=updated_hidden,
            visual_t=visual_t,
            state_t=state_t,
            signature_t=signature_t,
            delta_signature_t=delta_signature_t,
            route_hidden=route_hidden,
        )
        return updated_hidden, {
            "routing_logits": routing_logits,
            "routing_weights": routing_weights,
            "candidate": candidate,
            "gate": gate,
            "readout_context": readout_context,
        }

    def _update_prism_memory_step(
        self,
        *,
        memory_prev: Tensor,
        visual_t: Tensor,
        state_t: Tensor,
        signature_t: Tensor | None,
        delta_signature_t: Tensor | None,
        valid_t: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.use_signature_indexed_slot_memory:
            return self._update_signature_indexed_slot_memory_step(
                memory_prev=memory_prev,
                visual_t=visual_t,
                state_t=state_t,
                signature_t=signature_t,
                delta_signature_t=delta_signature_t,
                valid_t=valid_t,
            )

        batch_size = memory_prev.shape[0]
        step_input = self._build_prism_memory_step_input(
            visual_t=visual_t,
            state_t=state_t,
            signature_t=signature_t,
            delta_signature_t=delta_signature_t,
        )
        updated_hidden = torch.stack(
            [
                update_cell(step_input, memory_prev[:, slot_idx, :])
                for slot_idx, update_cell in enumerate(self._iter_prism_memory_updates())
            ],
            dim=1,
        )
        if valid_t is not None:
            valid_t = valid_t.to(dtype=torch.bool, device=memory_prev.device)
            if valid_t.shape != (batch_size,):
                raise ValueError(
                    f"PRISM memory valid mask must have shape {(batch_size,)}. Got {tuple(valid_t.shape)}."
                )
            updated_hidden = torch.where(valid_t.view(batch_size, 1, 1), updated_hidden, memory_prev)
        return updated_hidden, {}

    def _scan_prism_memory_from_prefix_sequence(self, batch: dict[str, Tensor]) -> Tensor:
        if PREFIX_STATE_KEY not in batch or PREFIX_MASK_KEY not in batch:
            raise KeyError(
                "Training PRISM memory reconstruction requires `observation.prefix_state` "
                "and `observation.prefix_mask`."
            )
        prefix_mask = batch[PREFIX_MASK_KEY].to(dtype=torch.bool)
        prefix_camera_images = [batch[prefix_key] for prefix_key in self.config.prefix_image_features]
        visual_embeddings = self._encode_multi_camera_images_for_prism(
            prefix_camera_images,
            context="PRISM prefix images",
        )
        state_embeddings = self._project_prism_state_tensor(
            batch[PREFIX_STATE_KEY],
            context="PRISM prefix states",
        )
        signature_embeddings = None
        delta_signature_embeddings = None
        if self.use_path_signature:
            if PREFIX_PATH_SIGNATURE_KEY not in batch:
                raise KeyError(
                    f"Training PRISM memory reconstruction requires `{PREFIX_PATH_SIGNATURE_KEY}`."
                )
            signature_embeddings = self._project_signature_tensor(
                batch[PREFIX_PATH_SIGNATURE_KEY],
                context="PRISM prefix path signatures",
            )
        if self.use_delta_signature:
            if PREFIX_DELTA_SIGNATURE_KEY not in batch:
                raise KeyError(
                    f"Training PRISM memory reconstruction requires `{PREFIX_DELTA_SIGNATURE_KEY}`."
                )
            delta_signature_embeddings = self._project_delta_signature_tensor(
                batch[PREFIX_DELTA_SIGNATURE_KEY],
                context="PRISM prefix delta signatures",
            )

        batch_size, time_steps = prefix_mask.shape
        memory_state = self._build_zero_prism_memory_state(
            batch_size=batch_size,
            device=visual_embeddings.device,
            dtype=visual_embeddings.dtype,
        )
        for step_idx in range(time_steps):
            memory_state, _ = self._update_prism_memory_step(
                memory_prev=memory_state,
                visual_t=visual_embeddings[:, step_idx],
                state_t=state_embeddings[:, step_idx],
                signature_t=None if signature_embeddings is None else signature_embeddings[:, step_idx],
                delta_signature_t=(
                    None if delta_signature_embeddings is None else delta_signature_embeddings[:, step_idx]
                ),
                valid_t=prefix_mask[:, step_idx],
            )
        return memory_state

    def _compute_prism_memory_context(
        self,
        *,
        memory_state: Tensor,
        visual_t: Tensor,
        state_t: Tensor,
        signature_t: Tensor | None,
        delta_signature_t: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        pooled_summary = memory_state.mean(dim=1)
        if not self.use_signature_indexed_slot_memory or not self.config.slot_memory_use_readout_pooling:
            return pooled_summary, pooled_summary
        readout_context, _, _ = self._read_signature_indexed_slot_memory_context(
            memory_state=memory_state,
            visual_t=visual_t,
            state_t=state_t,
            signature_t=signature_t,
            delta_signature_t=delta_signature_t,
        )
        return readout_context, pooled_summary

    def _prepare_prism_conditioning(
        self,
        batch: dict[str, Tensor],
        *,
        prism_memory_state: Tensor | None = None,
    ) -> Tensor | None:
        if not self.prism_enabled:
            return None

        adapter_inputs: list[Tensor] = []
        current_state = self._project_prism_state_tensor(
            self._select_current_step_feature(batch[OBS_STATE], context="Current observation.state"),
            context="Current observation.state",
        )

        signature_t = None
        if self.use_path_signature:
            if PATH_SIGNATURE_KEY not in batch:
                raise KeyError(f"PRISM conditioning requires `{PATH_SIGNATURE_KEY}`.")
            signature_t = self._project_signature_tensor(
                self._select_current_step_feature(batch[PATH_SIGNATURE_KEY], context="Current path signature"),
                context="Current path signature",
            )
            adapter_inputs.append(signature_t)

        delta_signature_t = None
        if self.use_delta_signature:
            if DELTA_SIGNATURE_KEY not in batch:
                raise KeyError(f"PRISM conditioning requires `{DELTA_SIGNATURE_KEY}`.")
            delta_signature_t = self._project_delta_signature_tensor(
                self._select_current_step_feature(
                    batch[DELTA_SIGNATURE_KEY],
                    context="Current delta signature",
                ),
                context="Current delta signature",
            )
            adapter_inputs.append(delta_signature_t)

        if self.use_visual_prefix_memory:
            if prism_memory_state is None:
                prism_memory_state = self._scan_prism_memory_from_prefix_sequence(batch)
            else:
                prism_memory_state = self._normalize_prism_memory_state(
                    prism_memory_state,
                    batch_size=current_state.shape[0],
                    device=current_state.device,
                    dtype=current_state.dtype,
                    context="Cached PRISM memory state",
                )
            current_visual = self._encode_multi_camera_images_for_prism(
                self._extract_current_camera_images(batch),
                context="Current PRISM observation images",
            )
            readout_context, pooled_summary = self._compute_prism_memory_context(
                memory_state=prism_memory_state,
                visual_t=current_visual,
                state_t=current_state,
                signature_t=signature_t,
                delta_signature_t=delta_signature_t,
            )
            adapter_inputs.extend([readout_context, pooled_summary])

        if not adapter_inputs:
            return None
        adapter_input = torch.cat(adapter_inputs, dim=-1)
        weight_dtype = self.prism_adapter[0].weight.dtype
        return self.prism_adapter(adapter_input.to(dtype=weight_dtype))

    def _prepare_combined_global_conditioning(
        self,
        batch: dict[str, Tensor],
        *,
        prism_memory_state: Tensor | None = None,
    ) -> Tensor:
        native_global_cond = self._prepare_global_conditioning(batch)
        prism_cond = self._prepare_prism_conditioning(batch, prism_memory_state=prism_memory_state)
        if prism_cond is None:
            return native_global_cond
        return torch.cat([native_global_cond, prism_cond.to(dtype=native_global_cond.dtype)], dim=-1)

    @torch.no_grad()
    def update_online_prism_memory(
        self,
        batch: dict[str, Tensor],
        *,
        previous_state: Tensor | None = None,
    ) -> Tensor:
        if not self.use_visual_prefix_memory:
            raise RuntimeError("`update_online_prism_memory` requires `use_visual_prefix_memory=True`.")

        visual_t = self._encode_multi_camera_images_for_prism(
            self._extract_current_camera_images(batch),
            context="Online PRISM observation images",
        )
        state_t = self._project_prism_state_tensor(
            self._select_current_step_feature(batch[OBS_STATE], context="Online observation.state"),
            context="Online observation.state",
        )
        if previous_state is None:
            memory_prev = self._build_zero_prism_memory_state(
                batch_size=visual_t.shape[0],
                device=visual_t.device,
                dtype=visual_t.dtype,
            )
        else:
            memory_prev = self._normalize_prism_memory_state(
                previous_state,
                batch_size=visual_t.shape[0],
                device=visual_t.device,
                dtype=visual_t.dtype,
                context="Cached PRISM memory state",
            )

        signature_t = None
        if self.use_path_signature:
            if PATH_SIGNATURE_KEY not in batch:
                raise KeyError(
                    "Online PRISM memory update requires the current path signature in the batch."
                )
            signature_t = self._project_signature_tensor(
                self._select_current_step_feature(batch[PATH_SIGNATURE_KEY], context="Current path signature"),
                context="Current path signature",
            )
        delta_signature_t = None
        if self.use_delta_signature:
            if DELTA_SIGNATURE_KEY not in batch:
                raise KeyError(
                    "Online PRISM memory update requires the current delta signature in the batch."
                )
            delta_signature_t = self._project_delta_signature_tensor(
                self._select_current_step_feature(batch[DELTA_SIGNATURE_KEY], context="Current delta signature"),
                context="Current delta signature",
            )
        next_state, _ = self._update_prism_memory_step(
            memory_prev=memory_prev,
            visual_t=visual_t,
            state_t=state_t,
            signature_t=signature_t,
            delta_signature_t=delta_signature_t,
            valid_t=torch.ones((visual_t.shape[0],), dtype=torch.bool, device=visual_t.device),
        )
        return next_state

    def generate_actions(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        *,
        prism_memory_state: Tensor | None = None,
    ) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_combined_global_conditioning(
            batch,
            prism_memory_state=prism_memory_state,
        )

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_combined_global_conditioning(batch)

        # Forward diffusion.
        trajectory = batch[ACTION]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://huggingface.co/papers/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: PrismDiffusionConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.resize = None

        crop_shape = config.crop_shape
        if crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy shape mirrors the runtime preprocessing order: resize -> crop.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        if config.crop_shape is not None:
            dummy_shape_h_w = config.crop_shape
        elif config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: resize if configured, then crop if configured.

        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: PrismDiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://huggingface.co/papers/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out
