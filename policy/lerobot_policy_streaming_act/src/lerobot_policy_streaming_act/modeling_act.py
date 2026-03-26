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
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .configuration_act import ACTConfig, FIRST_FRAME_ANCHOR_KEY, StreamingACTConfig
from .prefix_sequence import (
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class StreamingACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = StreamingACTConfig
    name = "streaming_act"

    def __init__(
        self,
        config: StreamingACTConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = StreamingACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = StreamingACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._visual_prefix_memory_state = None
        self._visual_prefix_memory_update_count = 0
        self._visual_prefix_memory_last_state_norm = 0.0

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.visual_observation_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.visual_observation_features]

        if self.config.use_visual_prefix_memory:
            memory_token, memory_state = self.model.compute_online_visual_prefix_memory_token(
                batch,
                previous_state=self._visual_prefix_memory_state,
            )
            self._visual_prefix_memory_state = memory_state.detach()
            self._visual_prefix_memory_update_count += 1
            self._visual_prefix_memory_last_state_norm = float(
                self._visual_prefix_memory_state.norm(dim=-1).mean().detach().cpu().item()
            )
            actions = self.model(
                batch,
                visual_prefix_memory_token=memory_token,
                skip_prefix_sequence_validation=True,
            )[0]
        else:
            actions = self.model(batch)[0]
        return actions

    def get_visual_prefix_memory_debug_stats(self) -> dict[str, float | int | bool]:
        state = self._visual_prefix_memory_state
        return {
            "enabled": bool(self.config.use_visual_prefix_memory),
            "initialized": state is not None,
            "num_slots": int(self.config.num_memory_slots),
            "update_count": int(self._visual_prefix_memory_update_count),
            "state_norm": float(self._visual_prefix_memory_last_state_norm),
        }

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.visual_observation_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.visual_observation_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTPolicy(StreamingACTPolicy):
    config_class = ACTConfig
    name = "act"


class StreamingACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class StreamingACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for StreamingACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config
        self.use_path_signature = config.use_path_signature
        self.use_first_frame_anchor = config.use_first_frame_anchor
        self.use_visual_prefix_memory = config.use_visual_prefix_memory

        if self.use_path_signature:
            assert config.signature_dim > 0, (
                "`signature_dim` must be > 0 when `use_path_signature=True` so that "
                "`self.signature_proj` can be initialized."
            )
            self.signature_proj = nn.Sequential(
                nn.Linear(config.signature_dim, config.signature_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.signature_dropout),
                nn.Linear(config.signature_hidden_dim, config.dim_model),
            )
        else:
            self.signature_proj = None

        if self.config.use_vae:
            self.vae_encoder = StreamingACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = StreamingACTEncoder(config)
        self.decoder = StreamingACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        if self.use_first_frame_anchor:
            self.anchor_token_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.anchor_token_proj = nn.Linear(config.dim_model, config.dim_model)
        if self.use_visual_prefix_memory:
            self.visual_prefix_memory_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.visual_prefix_memory_update = nn.GRUCell(
                input_size=config.dim_model * 2,
                hidden_size=config.dim_model,
            )
            self.visual_prefix_memory_extra_updates = nn.ModuleList(
                nn.GRUCell(
                    input_size=config.dim_model * 2,
                    hidden_size=config.dim_model,
                )
                for _ in range(config.num_memory_slots - 1)
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = StreamingACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _validate_prefix_sequence_inputs(self, batch: dict[str, Tensor], batch_size: int) -> None:
        if not self.config.use_prefix_sequence_training:
            return

        prefix_mask = batch.get(PREFIX_MASK_KEY)
        assert prefix_mask is not None, (
            f"`{PREFIX_MASK_KEY}` is required when `use_prefix_sequence_training=True`."
        )
        assert prefix_mask.ndim == 2, (
            f"`{PREFIX_MASK_KEY}` must have shape (batch_size, T_prefix). "
            f"Got ndim={prefix_mask.ndim}, shape={tuple(prefix_mask.shape)}."
        )
        assert prefix_mask.shape[0] == batch_size, (
            f"Batch mismatch for `{PREFIX_MASK_KEY}`: expected {batch_size}, "
            f"got {prefix_mask.shape[0]}."
        )
        assert prefix_mask.shape[1] == self.config.prefix_train_max_steps, (
            f"`{PREFIX_MASK_KEY}` second dim must equal "
            f"`prefix_train_max_steps={self.config.prefix_train_max_steps}`. "
            f"Got {prefix_mask.shape[1]}."
        )
        prefix_mask = prefix_mask.to(dtype=torch.bool)
        valid_lengths = prefix_mask.sum(dim=1)
        assert torch.all(valid_lengths > 0), (
            f"Every prefix row must contain at least one valid step in `{PREFIX_MASK_KEY}`."
        )
        monotonic_mask = prefix_mask[:, 1:] <= prefix_mask[:, :-1]
        assert torch.all(monotonic_mask), (
            f"`{PREFIX_MASK_KEY}` must use left-aligned valid steps with right padding only."
        )
        last_valid_positions = valid_lengths - 1
        gathered_last_valid = prefix_mask.gather(1, last_valid_positions.unsqueeze(1)).squeeze(1)
        assert torch.all(gathered_last_valid), (
            f"`{PREFIX_MASK_KEY}` is missing the last valid prefix element for at least one batch row."
        )

        prefix_state = batch.get(PREFIX_STATE_KEY)
        assert prefix_state is not None, (
            f"`{PREFIX_STATE_KEY}` is required when `use_prefix_sequence_training=True`."
        )
        assert prefix_state.ndim == 3, (
            f"`{PREFIX_STATE_KEY}` must have shape (batch_size, T_prefix, state_dim). "
            f"Got ndim={prefix_state.ndim}, shape={tuple(prefix_state.shape)}."
        )
        assert prefix_state.shape[0] == batch_size, (
            f"Batch mismatch for `{PREFIX_STATE_KEY}`: expected {batch_size}, "
            f"got {prefix_state.shape[0]}."
        )
        assert prefix_state.shape[1] == prefix_mask.shape[1], (
            f"`{PREFIX_STATE_KEY}` time dim must match `{PREFIX_MASK_KEY}`. "
            f"Got state_time={prefix_state.shape[1]}, mask_time={prefix_mask.shape[1]}."
        )
        assert prefix_state.shape[2] == self.config.robot_state_feature.shape[0], (
            f"`{PREFIX_STATE_KEY}` state dim must equal "
            f"`observation.state` dim {self.config.robot_state_feature.shape[0]}. "
            f"Got {prefix_state.shape[2]}."
        )

        if self.use_path_signature:
            prefix_signature = batch.get(PREFIX_PATH_SIGNATURE_KEY)
            assert prefix_signature is not None, (
                f"`{PREFIX_PATH_SIGNATURE_KEY}` is required when "
                "`use_prefix_sequence_training=True`."
            )
            assert prefix_signature.ndim == 3, (
                f"`{PREFIX_PATH_SIGNATURE_KEY}` must have shape "
                f"(batch_size, T_prefix, signature_dim). "
                f"Got ndim={prefix_signature.ndim}, shape={tuple(prefix_signature.shape)}."
            )
            assert prefix_signature.shape[0] == batch_size, (
                f"Batch mismatch for `{PREFIX_PATH_SIGNATURE_KEY}`: expected {batch_size}, "
                f"got {prefix_signature.shape[0]}."
            )
            assert prefix_signature.shape[1] == prefix_mask.shape[1], (
                f"`{PREFIX_PATH_SIGNATURE_KEY}` time dim must match `{PREFIX_MASK_KEY}`. "
                f"Got sig_time={prefix_signature.shape[1]}, mask_time={prefix_mask.shape[1]}."
            )
            assert prefix_signature.shape[2] == self.config.signature_dim, (
                f"`{PREFIX_PATH_SIGNATURE_KEY}` signature dim must equal "
                f"`signature_dim={self.config.signature_dim}`. Got {prefix_signature.shape[2]}."
            )

        for prefix_image_key, prefix_image_feature in self.config.prefix_image_features.items():
            assert prefix_image_key in batch, (
                f"`{prefix_image_key}` is required when `use_prefix_sequence_training=True`."
            )
            prefix_images = batch[prefix_image_key]
            assert prefix_images.ndim == 5, (
                f"`{prefix_image_key}` must have shape (batch_size, T_prefix, C, H, W). "
                f"Got ndim={prefix_images.ndim}, shape={tuple(prefix_images.shape)}."
            )
            assert prefix_images.shape[0] == batch_size, (
                f"Batch mismatch for `{prefix_image_key}`: expected {batch_size}, "
                f"got {prefix_images.shape[0]}."
            )
            assert prefix_images.shape[1] == prefix_mask.shape[1], (
                f"`{prefix_image_key}` time dim must match `{PREFIX_MASK_KEY}`. "
                f"Got image_time={prefix_images.shape[1]}, mask_time={prefix_mask.shape[1]}."
            )
            expected_image_shape = tuple(prefix_image_feature.shape[1:])
            assert tuple(prefix_images.shape[2:]) == expected_image_shape, (
                f"`{prefix_image_key}` trailing dims must equal {expected_image_shape}. "
                f"Got {tuple(prefix_images.shape[2:])}."
            )

    def _build_zero_visual_prefix_memory_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.zeros(
            (batch_size, self.config.num_memory_slots, self.config.dim_model),
            device=device,
            dtype=dtype,
        )

    def _normalize_visual_prefix_memory_state(
        self,
        hidden: Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        context: str,
    ) -> Tensor:
        expected_shape = (batch_size, self.config.num_memory_slots, self.config.dim_model)
        if hidden.ndim == 2:
            if self.config.num_memory_slots != 1:
                raise ValueError(
                    f"{context} expects hidden state shape {expected_shape}. "
                    f"Got legacy single-slot shape {tuple(hidden.shape)}."
                )
            if hidden.shape != (batch_size, self.config.dim_model):
                raise ValueError(
                    f"{context} shape mismatch. Expected {(batch_size, self.config.dim_model)}, "
                    f"got {tuple(hidden.shape)}."
                )
            hidden = hidden.unsqueeze(1)
        elif hidden.ndim != 3:
            raise ValueError(
                f"{context} must have shape {expected_shape}. Got {tuple(hidden.shape)}."
            )
        if tuple(hidden.shape) != expected_shape:
            raise ValueError(
                f"{context} shape mismatch. Expected {expected_shape}, got {tuple(hidden.shape)}."
            )
        return hidden.to(device=device, dtype=dtype)

    def _iter_visual_prefix_memory_updates(self) -> list[nn.GRUCell]:
        updates = [self.visual_prefix_memory_update, *self.visual_prefix_memory_extra_updates]
        assert len(updates) == self.config.num_memory_slots, (
            "Number of visual prefix memory updaters must match `num_memory_slots`. "
            f"Got updaters={len(updates)} vs slots={self.config.num_memory_slots}."
        )
        return updates

    def _pool_visual_features_for_memory(self, features: Tensor) -> Tensor:
        pooled = self.visual_prefix_memory_pool(features).flatten(1)
        assert pooled.ndim == 2 and pooled.shape[1] == self.config.dim_model, (
            "Visual prefix memory pooled features must have shape "
            f"(batch_like, {self.config.dim_model}). Got {tuple(pooled.shape)}."
        )
        return pooled

    def _encode_images_for_visual_prefix_memory(self, images: Tensor) -> Tensor:
        if images.ndim == 4:
            flat_images = images
            batch_size = images.shape[0]
            time_steps = None
        elif images.ndim == 5:
            batch_size, time_steps = images.shape[:2]
            flat_images = images.reshape(batch_size * time_steps, *images.shape[2:])
        else:
            raise ValueError(
                "Visual prefix memory images must have shape (B, C, H, W) or "
                f"(B, T, C, H, W). Got {tuple(images.shape)}."
            )

        features = self.backbone(flat_images)["feature_map"]
        features = self.encoder_img_feat_input_proj(features)
        pooled = self._pool_visual_features_for_memory(features)
        if time_steps is None:
            return pooled
        return pooled.reshape(batch_size, time_steps, self.config.dim_model)

    def _reduce_camera_embeddings_for_visual_prefix_memory(
        self,
        camera_embeddings: list[Tensor],
    ) -> Tensor:
        if not camera_embeddings:
            raise ValueError("Visual prefix memory requires at least one camera embedding.")
        if len(camera_embeddings) == 1:
            return camera_embeddings[0]
        return torch.stack(camera_embeddings, dim=0).mean(dim=0)

    def _project_prefix_states_for_visual_prefix_memory(self, prefix_state: Tensor) -> Tensor:
        batch_size, time_steps, state_dim = prefix_state.shape
        flat_prefix_state = prefix_state.reshape(batch_size * time_steps, state_dim)
        projected = self.encoder_robot_state_input_proj(flat_prefix_state)
        return projected.reshape(batch_size, time_steps, self.config.dim_model)

    def _scan_visual_prefix_memory(
        self,
        *,
        visual_embeddings: Tensor,
        state_embeddings: Tensor,
        prefix_mask: Tensor,
        initial_state: Tensor | None = None,
    ) -> Tensor:
        batch_size, time_steps, hidden_dim = visual_embeddings.shape
        assert state_embeddings.shape == (batch_size, time_steps, hidden_dim), (
            "Visual prefix memory state embeddings must match visual embeddings. "
            f"Got visual={tuple(visual_embeddings.shape)} vs "
            f"state={tuple(state_embeddings.shape)}."
        )
        assert prefix_mask.shape == (batch_size, time_steps), (
            "Visual prefix memory mask must match the prefix sequence shape. "
            f"Got mask={tuple(prefix_mask.shape)} vs sequence={(batch_size, time_steps)}."
        )

        if initial_state is None:
            hidden = self._build_zero_visual_prefix_memory_state(
                batch_size=batch_size,
                device=visual_embeddings.device,
                dtype=visual_embeddings.dtype,
            )
        else:
            hidden = self._normalize_visual_prefix_memory_state(
                initial_state,
                batch_size=batch_size,
                device=visual_embeddings.device,
                dtype=visual_embeddings.dtype,
                context="Visual prefix memory initial state",
            )

        prefix_mask = prefix_mask.to(dtype=torch.bool, device=visual_embeddings.device)
        slot_updates = self._iter_visual_prefix_memory_updates()
        for step_idx in range(time_steps):
            step_input = torch.cat(
                [visual_embeddings[:, step_idx], state_embeddings[:, step_idx]],
                dim=-1,
            )
            updated_hidden = torch.stack(
                [
                    slot_update(step_input, hidden[:, slot_idx, :])
                    for slot_idx, slot_update in enumerate(slot_updates)
                ],
                dim=1,
            )
            mask_t = prefix_mask[:, step_idx].view(batch_size, 1, 1)
            hidden = torch.where(mask_t, updated_hidden, hidden)
        return hidden

    def _compute_visual_prefix_memory_token_from_prefix_sequence(
        self,
        batch: dict[str, Tensor],
    ) -> Tensor:
        assert PREFIX_STATE_KEY in batch, (
            f"`{PREFIX_STATE_KEY}` is required to reconstruct visual prefix memory during training."
        )
        assert PREFIX_MASK_KEY in batch, (
            f"`{PREFIX_MASK_KEY}` is required to reconstruct visual prefix memory during training."
        )
        camera_embeddings = [
            self._encode_images_for_visual_prefix_memory(batch[prefix_image_key])
            for prefix_image_key in self.config.prefix_image_features
        ]
        visual_embeddings = self._reduce_camera_embeddings_for_visual_prefix_memory(camera_embeddings)
        state_embeddings = self._project_prefix_states_for_visual_prefix_memory(batch[PREFIX_STATE_KEY])
        memory_state = self._scan_visual_prefix_memory(
            visual_embeddings=visual_embeddings,
            state_embeddings=state_embeddings,
            prefix_mask=batch[PREFIX_MASK_KEY],
        )
        return memory_state

    def compute_online_visual_prefix_memory_token(
        self,
        batch: dict[str, Tensor],
        *,
        previous_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not self.use_visual_prefix_memory:
            raise RuntimeError(
                "`compute_online_visual_prefix_memory_token` requires "
                "`use_visual_prefix_memory=True`."
            )
        assert OBS_IMAGES in batch, (
            "Online visual prefix memory update requires current-step images in `batch[OBS_IMAGES]`."
        )
        assert OBS_STATE in batch, (
            "Online visual prefix memory update requires `observation.state` in the batch."
        )

        camera_embeddings = [
            self._encode_images_for_visual_prefix_memory(img) for img in batch[OBS_IMAGES]
        ]
        visual_embedding = self._reduce_camera_embeddings_for_visual_prefix_memory(camera_embeddings)
        state_embedding = self.encoder_robot_state_input_proj(batch[OBS_STATE])
        if previous_state is None:
            hidden = self._build_zero_visual_prefix_memory_state(
                batch_size=visual_embedding.shape[0],
                device=visual_embedding.device,
                dtype=visual_embedding.dtype,
            )
        else:
            hidden = self._normalize_visual_prefix_memory_state(
                previous_state,
                batch_size=visual_embedding.shape[0],
                device=visual_embedding.device,
                dtype=visual_embedding.dtype,
                context="Cached visual prefix memory state",
            )

        step_input = torch.cat([visual_embedding, state_embedding], dim=-1)
        next_state = torch.stack(
            [
                slot_update(step_input, hidden[:, slot_idx, :])
                for slot_idx, slot_update in enumerate(self._iter_visual_prefix_memory_updates())
            ],
            dim=1,
        )
        return next_state, next_state

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        visual_prefix_memory_token: Tensor | None = None,
        skip_prefix_sequence_validation: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of current-step images.
            [FIRST_FRAME_ANCHOR_KEY] (optional): (B, C, H, W) first-frame anchor image.
            [PREFIX_STATE_KEY] (optional): (B, T_prefix, state_dim) prefix state sequence.
            [PREFIX_PATH_SIGNATURE_KEY] (optional): (B, T_prefix, signature_dim) prefix signature sequence.
            [PREFIX_MASK_KEY] (optional): (B, T_prefix) prefix valid mask.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        if not skip_prefix_sequence_validation:
            self._validate_prefix_sequence_inputs(batch, batch_size)

        if self.use_path_signature:
            path_signature_key = "observation.path_signature"
            assert path_signature_key in batch, (
                f"`{path_signature_key}` is required when `use_path_signature=True`."
            )
            path_signature = batch[path_signature_key]
            assert path_signature.ndim == 2, (
                f"`{path_signature_key}` must have shape (batch_size, signature_dim). "
                f"Got ndim={path_signature.ndim}, shape={tuple(path_signature.shape)}."
            )
            assert path_signature.shape[0] == batch_size, (
                f"Batch mismatch for `{path_signature_key}`: expected {batch_size}, "
                f"got {path_signature.shape[0]}."
            )
            assert path_signature.shape[1] == self.config.signature_dim, (
                f"`{path_signature_key}` second dim must be `signature_dim={self.config.signature_dim}`. "
                f"Got {path_signature.shape[1]}."
            )

            path_signature = path_signature.to(dtype=self.signature_proj[0].weight.dtype)
            signature_embed = self.signature_proj(path_signature)
            assert signature_embed.shape == (batch_size, self.config.dim_model), (
                f"`signature_embed` must have shape ({batch_size}, {self.config.dim_model}). "
                f"Got {tuple(signature_embed.shape)}."
            )
            signature_embed = signature_embed.unsqueeze(1)  # (B, 1, D)

        if self.use_first_frame_anchor:
            assert FIRST_FRAME_ANCHOR_KEY in batch, (
                f"`{FIRST_FRAME_ANCHOR_KEY}` is required when `use_first_frame_anchor=True`."
            )
            anchor_image = batch[FIRST_FRAME_ANCHOR_KEY]
            assert anchor_image.ndim == 4, (
                f"`{FIRST_FRAME_ANCHOR_KEY}` must have shape (batch_size, C, H, W). "
                f"Got ndim={anchor_image.ndim}, shape={tuple(anchor_image.shape)}."
            )
            assert anchor_image.shape[0] == batch_size, (
                f"Batch mismatch for `{FIRST_FRAME_ANCHOR_KEY}`: expected {batch_size}, "
                f"got {anchor_image.shape[0]}."
            )
            anchor_features = self.backbone(anchor_image)["feature_map"]
            anchor_features = self.encoder_img_feat_input_proj(anchor_features)
            anchor_embed = self.anchor_token_pool(anchor_features).flatten(1)
            anchor_embed = self.anchor_token_proj(anchor_embed)
            assert anchor_embed.shape == (batch_size, self.config.dim_model), (
                f"`anchor_embed` must have shape ({batch_size}, {self.config.dim_model}). "
                f"Got {tuple(anchor_embed.shape)}."
            )
            anchor_embed = anchor_embed.unsqueeze(1)  # (B, 1, D)

        visual_prefix_memory_embed = None
        if self.use_visual_prefix_memory:
            if visual_prefix_memory_token is None:
                visual_prefix_memory_embed = self._compute_visual_prefix_memory_token_from_prefix_sequence(
                    batch
                )
            else:
                if visual_prefix_memory_token.ndim != 3:
                    raise ValueError(
                        "Visual prefix memory token override must have shape "
                        f"(batch_size, num_memory_slots, dim_model). Got {tuple(visual_prefix_memory_token.shape)}."
                    )
                expected_shape = (
                    batch_size,
                    self.config.num_memory_slots,
                    self.config.dim_model,
                )
                if tuple(visual_prefix_memory_token.shape) != expected_shape:
                    raise ValueError(
                        "Visual prefix memory token override shape mismatch. "
                        f"Expected {expected_shape}, got {tuple(visual_prefix_memory_token.shape)}."
                    )
                visual_prefix_memory_embed = visual_prefix_memory_token.to(
                    device=batch[OBS_STATE].device,
                    dtype=self.encoder_latent_input_proj.weight.dtype,
                )

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.visual_observation_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        extra_memory_tokens = []
        extra_memory_pos_embed = []
        if self.use_first_frame_anchor:
            anchor_token = anchor_embed.transpose(0, 1).to(
                device=encoder_out.device, dtype=encoder_out.dtype
            )  # (1, B, D)
            anchor_pos_embed = torch.zeros(
                (1, 1, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )
            extra_memory_tokens.append(anchor_token)
            extra_memory_pos_embed.append(anchor_pos_embed)
        if self.use_path_signature:
            signature_token = signature_embed.transpose(0, 1).to(
                device=encoder_out.device, dtype=encoder_out.dtype
            )  # (1, B, D)
            signature_pos_embed = torch.zeros(
                (1, 1, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )
            extra_memory_tokens.append(signature_token)
            extra_memory_pos_embed.append(signature_pos_embed)
        if self.use_visual_prefix_memory:
            assert visual_prefix_memory_embed is not None
            visual_prefix_memory_token = visual_prefix_memory_embed.transpose(0, 1).to(
                device=encoder_out.device,
                dtype=encoder_out.dtype,
            )
            visual_prefix_memory_pos_embed = torch.zeros(
                (self.config.num_memory_slots, 1, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )
            extra_memory_tokens.append(visual_prefix_memory_token)
            extra_memory_pos_embed.append(visual_prefix_memory_pos_embed)

        if extra_memory_tokens:
            # Extra non-image context tokens are injected into encoder memory before decoder cross-attention.
            encoder_out = torch.cat([*extra_memory_tokens, encoder_out], dim=0)
            encoder_in_pos_embed = torch.cat([*extra_memory_pos_embed, encoder_in_pos_embed], dim=0)
            assert encoder_out.shape[0] == encoder_in_pos_embed.shape[0], (
                "Encoder token length and positional embedding length must match after "
                "extra memory token injection."
            )

        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class StreamingACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([StreamingACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class StreamingACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class StreamingACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([StreamingACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class StreamingACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class StreamingACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
