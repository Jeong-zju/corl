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

try:
    from policy_imports import install_lerobot_policies_namespace_shim
except ModuleNotFoundError:
    def install_lerobot_policies_namespace_shim() -> None:
        return None

install_lerobot_policies_namespace_shim()

from .configuration_streaming_act import FIRST_FRAME_ANCHOR_KEY, StreamingACTConfig
from .prefix_sequence import (
    DELTA_SIGNATURE_KEY,
    PATH_SIGNATURE_KEY,
    PREFIX_DELTA_SIGNATURE_KEY,
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


_VAE_LOG_VAR_MIN = -10.0
_VAE_LOG_VAR_MAX = 10.0


def _stable_vae_log_variance(log_sigma_x2: Tensor) -> Tensor:
    return log_sigma_x2.float().clamp(
        min=_VAE_LOG_VAR_MIN,
        max=_VAE_LOG_VAR_MAX,
    )


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
        self._visual_prefix_anchor_embedding = None
        self._visual_prefix_memory_update_count = 0
        self._visual_prefix_memory_last_state_norm = 0.0
        self.model.reset_deploy_debug()

    def _prepare_observation_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.config.visual_observation_features:
            return batch
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        batch[OBS_IMAGES] = [batch[key] for key in self.config.visual_observation_features]
        return batch

    def _update_online_visual_prefix_memory(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.config.use_visual_prefix_memory:
            return None
        (
            memory_token,
            memory_state,
            anchor_embedding,
        ) = self.model.compute_online_visual_prefix_memory_token(
            batch,
            previous_state=self._visual_prefix_memory_state,
            first_frame_anchor_embedding=self._visual_prefix_anchor_embedding,
        )
        self._visual_prefix_memory_state = memory_state.detach()
        self._visual_prefix_anchor_embedding = (
            None if anchor_embedding is None else anchor_embedding.detach()
        )
        self._visual_prefix_memory_update_count += 1
        self._visual_prefix_memory_last_state_norm = float(
            self._visual_prefix_memory_state.norm(dim=-1).mean().detach().cpu().item()
        )
        return memory_token

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed
        batch = self._prepare_observation_batch(batch)
        memory_token = self._update_online_visual_prefix_memory(batch)

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(
                batch,
                visual_prefix_memory_token=memory_token,
                visual_prefix_anchor_embedding=self._visual_prefix_anchor_embedding,
            )
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(
                batch,
                visual_prefix_memory_token=memory_token,
                visual_prefix_anchor_embedding=self._visual_prefix_anchor_embedding,
            )[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        *,
        visual_prefix_memory_token: Tensor | None = None,
        visual_prefix_anchor_embedding: Tensor | None = None,
    ) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()
        batch = self._prepare_observation_batch(batch)

        if self.config.use_visual_prefix_memory:
            if visual_prefix_memory_token is None:
                visual_prefix_memory_token = self._update_online_visual_prefix_memory(batch)
                visual_prefix_anchor_embedding = self._visual_prefix_anchor_embedding
            actions = self.model(
                batch,
                visual_prefix_memory_token=visual_prefix_memory_token,
                visual_prefix_anchor_embedding=visual_prefix_anchor_embedding,
                skip_prefix_sequence_validation=True,
            )[0]
        else:
            actions = self.model(batch)[0]
        return actions

    def get_visual_prefix_memory_debug_stats(self) -> dict[str, float | int | bool]:
        state = self._visual_prefix_memory_state
        anchor = self._visual_prefix_anchor_embedding
        return {
            "enabled": bool(self.config.use_visual_prefix_memory),
            "initialized": state is not None,
            "num_slots": int(self.config.active_visual_prefix_memory_num_slots),
            "signature_indexed_slot_memory": bool(
                getattr(self.config, "use_signature_indexed_slot_memory", False)
            ),
            "signature_conditioned": bool(
                getattr(self.config, "use_signature_conditioned_visual_prefix_memory", False)
            ),
            "uses_delta_signature": bool(getattr(self.config, "use_delta_signature", False)),
            "uses_first_frame_anchor_routing": bool(
                getattr(self.config, "use_first_frame_anchor_in_slot_routing", False)
            ),
            "anchor_initialized": anchor is not None,
            "anchor_norm": (
                0.0
                if anchor is None
                else float(anchor.norm(dim=-1).mean().detach().cpu().item())
            ),
            "update_count": int(self._visual_prefix_memory_update_count),
            "state_norm": float(self._visual_prefix_memory_last_state_norm),
        }

    def get_deploy_debug_snapshot(self) -> dict:
        snapshot = self.model.get_deploy_debug_snapshot()
        snapshot["visual_memory_stats"] = self.get_visual_prefix_memory_debug_stats()
        return snapshot

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
            mu_hat_f32 = mu_hat.float()
            log_sigma_x2_hat_f32 = _stable_vae_log_variance(log_sigma_x2_hat)
            mean_kld = (
                (
                    -0.5
                    * (
                        1
                        + log_sigma_x2_hat_f32
                        - mu_hat_f32.pow(2)
                        - log_sigma_x2_hat_f32.exp()
                    )
                )
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        aux_losses = self.model.get_visual_prefix_memory_aux_losses()
        if "slot_memory_balance_loss" in aux_losses:
            balance_raw = aux_losses["slot_memory_balance_loss"]
            balance_weighted = balance_raw * self.config.slot_memory_balance_loss_coef
            loss = loss + balance_weighted
            loss_dict["slot_memory_balance_loss"] = balance_raw.item()
            loss_dict["slot_memory_balance_loss_weighted"] = balance_weighted.item()
        if "slot_memory_entropy_loss" in aux_losses:
            entropy_raw = aux_losses["slot_memory_entropy_loss"]
            entropy_weighted = entropy_raw * self.config.slot_memory_entropy_loss_coef
            loss = loss + entropy_weighted
            loss_dict["slot_memory_entropy_loss"] = entropy_raw.item()
            loss_dict["slot_memory_entropy_loss_weighted"] = entropy_weighted.item()
        if "slot_memory_consistency_loss" in aux_losses:
            consistency_raw = aux_losses["slot_memory_consistency_loss"]
            consistency_weighted = (
                consistency_raw * self.config.slot_memory_consistency_loss_coef
            )
            loss = loss + consistency_weighted
            loss_dict["slot_memory_consistency_loss"] = consistency_raw.item()
            loss_dict["slot_memory_consistency_loss_weighted"] = consistency_weighted.item()
        loss_dict.update(self.model.get_visual_prefix_memory_log_stats())

        return loss, loss_dict


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

    def __init__(self, config: StreamingACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config
        self.use_path_signature = config.use_path_signature
        self.use_delta_signature = config.use_delta_signature
        self.use_first_frame_anchor = config.use_first_frame_anchor
        self.use_first_frame_anchor_in_slot_routing = (
            config.use_first_frame_anchor_in_slot_routing
        )
        self.use_visual_prefix_memory = config.use_visual_prefix_memory
        self.use_signature_conditioned_visual_prefix_memory = (
            config.use_signature_conditioned_visual_prefix_memory
        )
        self.use_signature_indexed_slot_memory = config.use_signature_indexed_slot_memory
        self.use_memory_conditioned_encoder_film = config.use_memory_conditioned_encoder_film
        self.active_memory_num_slots = config.active_visual_prefix_memory_num_slots
        self._last_visual_prefix_memory_aux_losses: dict[str, Tensor] = {}
        self._last_visual_prefix_memory_log_stats: dict[str, float] = {}
        self.reset_deploy_debug()

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
        if self.use_delta_signature:
            assert config.signature_dim > 0, (
                "`signature_dim` must be > 0 when `use_delta_signature=True` so that "
                "`self.delta_signature_proj` can be initialized."
            )
            self.delta_signature_proj = nn.Sequential(
                nn.Linear(config.signature_dim, config.signature_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.signature_dropout),
                nn.Linear(config.signature_hidden_dim, config.dim_model),
            )
        else:
            self.delta_signature_proj = None

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
            if self.use_signature_indexed_slot_memory:
                slot_route_component_count = (
                    1
                    + int(config.slot_memory_use_delta_routing)
                    + int(config.use_first_frame_anchor_in_slot_routing)
                )
                slot_route_input_dim = config.dim_model * slot_route_component_count
                slot_write_input_dim = config.dim_model * (
                    2 + slot_route_component_count
                )
                slot_state_input_dim = (
                    config.dim_model * 2 + config.slot_memory_routing_hidden_dim
                )
                slot_read_query_input_dim = (
                    config.dim_model * 2 + config.slot_memory_routing_hidden_dim
                )
                self.slot_memory_route_proj = nn.Sequential(
                    nn.Linear(
                        slot_route_input_dim,
                        config.slot_memory_routing_hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        config.slot_memory_routing_hidden_dim,
                        config.slot_memory_routing_hidden_dim,
                    ),
                )
                self.slot_memory_route_query_proj = nn.Linear(
                    config.slot_memory_routing_hidden_dim,
                    config.slot_memory_routing_hidden_dim,
                    bias=False,
                )
                self.slot_memory_route_key_proj = nn.Linear(
                    config.dim_model,
                    config.slot_memory_routing_hidden_dim,
                    bias=False,
                )
                self.slot_memory_write_proj = nn.Sequential(
                    nn.Linear(slot_write_input_dim, config.dim_model),
                    nn.GELU(),
                    nn.Linear(config.dim_model, config.dim_model),
                )
                self.slot_memory_candidate_proj = nn.Sequential(
                    nn.Linear(slot_state_input_dim, config.dim_model),
                    nn.GELU(),
                    nn.Linear(config.dim_model, config.dim_model),
                )
                self.slot_memory_gate_proj = nn.Sequential(
                    nn.Linear(slot_state_input_dim, config.dim_model),
                    nn.GELU(),
                    nn.Linear(config.dim_model, 1),
                )
                self.slot_memory_read_query_proj = nn.Sequential(
                    nn.Linear(slot_read_query_input_dim, config.dim_model),
                    nn.GELU(),
                    nn.Linear(config.dim_model, config.dim_model),
                )
                self.slot_memory_read_key_proj = nn.Linear(
                    config.dim_model,
                    config.dim_model,
                    bias=False,
                )
                self.slot_memory_read_value_proj = nn.Linear(
                    config.dim_model,
                    config.dim_model,
                    bias=False,
                )
            else:
                visual_prefix_memory_input_dim = config.dim_model * (
                    2
                    + int(self.use_signature_conditioned_visual_prefix_memory)
                    + int(self.use_delta_signature)
                )
                self.visual_prefix_memory_update = nn.GRUCell(
                    input_size=visual_prefix_memory_input_dim,
                    hidden_size=config.dim_model,
                )
                self.visual_prefix_memory_extra_updates = nn.ModuleList(
                    nn.GRUCell(
                        input_size=visual_prefix_memory_input_dim,
                        hidden_size=config.dim_model,
                    )
                    for _ in range(config.num_memory_slots - 1)
                )
            if self.use_memory_conditioned_encoder_film:
                self.visual_prefix_memory_encoder_film = nn.Sequential(
                    nn.Linear(config.dim_model, config.dim_model),
                    nn.GELU(),
                    nn.Linear(config.dim_model, config.dim_model * 2),
                )
                # Start from the identity transform so existing training dynamics stay unchanged
                # until the model learns to use memory-conditioned modulation.
                nn.init.zeros_(self.visual_prefix_memory_encoder_film[-1].weight)
                nn.init.zeros_(self.visual_prefix_memory_encoder_film[-1].bias)
            else:
                self.visual_prefix_memory_encoder_film = None
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

    def get_visual_prefix_memory_aux_losses(self) -> dict[str, Tensor]:
        return dict(self._last_visual_prefix_memory_aux_losses)

    def get_visual_prefix_memory_log_stats(self) -> dict[str, float]:
        return dict(self._last_visual_prefix_memory_log_stats)

    def reset_deploy_debug(self) -> None:
        self._latest_deploy_debug: dict[str, object] = {}
        self._latest_decoder_token_layout: dict[str, object] = {}
        self._latest_visual_prefix_memory_step_stats: dict[str, Tensor | float | int | bool] = {}
        self._latest_observation_image_token_layout: dict[str, object] = {}

    @staticmethod
    def _debug_tensor(tensor: Tensor | None) -> Tensor | None:
        if tensor is None:
            return None
        return tensor.detach().float().cpu()

    @staticmethod
    def _debug_norm(tensor: Tensor | None) -> float:
        if tensor is None:
            return 0.0
        return float(tensor.detach().float().norm(dim=-1).mean().cpu().item())

    @staticmethod
    def _debug_slot_std(tensor: Tensor | None) -> float:
        if tensor is None or tensor.ndim != 3 or tensor.shape[1] <= 1:
            return 0.0
        slot_std = tensor.detach().float().std(dim=1, unbiased=False)
        return float(slot_std.norm(dim=-1).mean().cpu().item())

    @staticmethod
    def _categorical_entropy_from_probabilities(probabilities: Tensor) -> Tensor:
        probabilities_f32 = probabilities.float()
        log_probabilities = probabilities_f32.clamp_min(
            torch.finfo(torch.float32).tiny
        ).log()
        return -(probabilities_f32 * log_probabilities).sum(dim=-1)

    @staticmethod
    def _bounded_slot_memory_consistency_loss(
        readout_context: Tensor,
        write_base: Tensor,
    ) -> Tensor:
        readout_bounded = torch.tanh(readout_context.float())
        write_bounded = torch.tanh(write_base.float())
        return F.mse_loss(
            readout_bounded,
            write_bounded,
            reduction="none",
        ).mean(dim=-1)

    def _build_slot_memory_identity(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if not self.use_signature_indexed_slot_memory:
            return torch.zeros(
                (batch_size, self.active_memory_num_slots, self.config.dim_model),
                device=device,
                dtype=dtype,
            )
        scale = float(getattr(self.config, "slot_memory_identity_scale", 0.0))
        if scale <= 0.0:
            return torch.zeros(
                (batch_size, self.active_memory_num_slots, self.config.dim_model),
                device=device,
                dtype=dtype,
            )
        slot_idx = torch.arange(
            self.active_memory_num_slots,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        dim_idx = torch.arange(
            self.config.dim_model,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
        inv_freq = torch.exp(
            -math.log(10000.0)
            * (
                2.0
                * torch.floor(dim_idx / 2.0)
                / float(max(1, self.config.dim_model))
            )
        )
        angles = (slot_idx + 1.0) * inv_freq
        even_dims = (dim_idx.to(torch.long) % 2 == 0)
        identity = torch.where(even_dims, torch.sin(angles), torch.cos(angles))
        if self.active_memory_num_slots > 1:
            identity = identity - identity.mean(dim=0, keepdim=True)
        identity = F.normalize(identity, dim=-1) * math.sqrt(float(self.config.dim_model))
        identity = identity.to(dtype=dtype) * scale
        return identity.unsqueeze(0).expand(batch_size, -1, -1)

    def _memory_with_slot_identity(self, memory_state: Tensor) -> Tensor:
        if not self.use_signature_indexed_slot_memory:
            return memory_state
        # SISM slots otherwise remain permutation-symmetric when they start from zeros.
        # The fixed identity is an address signal, not recurrent content.
        identity = self._build_slot_memory_identity(
            batch_size=memory_state.shape[0],
            device=memory_state.device,
            dtype=memory_state.dtype,
        )
        return memory_state + identity

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
        if self.use_delta_signature:
            prefix_delta_signature = batch.get(PREFIX_DELTA_SIGNATURE_KEY)
            assert prefix_delta_signature is not None, (
                f"`{PREFIX_DELTA_SIGNATURE_KEY}` is required when "
                "`use_delta_signature=True` and `use_prefix_sequence_training=True`."
            )
            assert prefix_delta_signature.ndim == 3, (
                f"`{PREFIX_DELTA_SIGNATURE_KEY}` must have shape "
                f"(batch_size, T_prefix, signature_dim). "
                f"Got ndim={prefix_delta_signature.ndim}, "
                f"shape={tuple(prefix_delta_signature.shape)}."
            )
            assert prefix_delta_signature.shape[0] == batch_size, (
                f"Batch mismatch for `{PREFIX_DELTA_SIGNATURE_KEY}`: expected {batch_size}, "
                f"got {prefix_delta_signature.shape[0]}."
            )
            assert prefix_delta_signature.shape[1] == prefix_mask.shape[1], (
                f"`{PREFIX_DELTA_SIGNATURE_KEY}` time dim must match `{PREFIX_MASK_KEY}`. "
                f"Got delta_time={prefix_delta_signature.shape[1]}, "
                f"mask_time={prefix_mask.shape[1]}."
            )
            assert prefix_delta_signature.shape[2] == self.config.signature_dim, (
                f"`{PREFIX_DELTA_SIGNATURE_KEY}` signature dim must equal "
                f"`signature_dim={self.config.signature_dim}`. "
                f"Got {prefix_delta_signature.shape[2]}."
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
            (batch_size, self.active_memory_num_slots, self.config.dim_model),
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
        expected_shape = (batch_size, self.active_memory_num_slots, self.config.dim_model)
        if hidden.ndim == 2:
            if self.active_memory_num_slots != 1:
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

    def _normalize_first_frame_anchor_embedding(
        self,
        anchor_embedding: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        context: str,
    ) -> Tensor:
        if anchor_embedding is None:
            raise ValueError(
                f"{context} requires a first-frame visual anchor embedding when "
                "`use_first_frame_anchor_in_slot_routing=True`."
            )
        if anchor_embedding.ndim == 3 and anchor_embedding.shape[1] == 1:
            anchor_embedding = anchor_embedding.squeeze(1)
        expected_shape = (batch_size, self.config.dim_model)
        if anchor_embedding.ndim != 2 or tuple(anchor_embedding.shape) != expected_shape:
            raise ValueError(
                f"{context} first-frame visual anchor embedding must have shape "
                f"{expected_shape}. Got {tuple(anchor_embedding.shape)}."
            )
        return anchor_embedding.to(device=device, dtype=dtype)

    def _iter_visual_prefix_memory_updates(self) -> list[nn.GRUCell]:
        if self.use_signature_indexed_slot_memory:
            raise RuntimeError(
                "`_iter_visual_prefix_memory_updates` is only valid for the legacy "
                "GRU-style visual prefix memory path."
            )
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

    def _flatten_camera_image_batch(
        self,
        camera_images: list[Tensor],
        *,
        context: str,
    ) -> tuple[Tensor, int, int | None, int]:
        if not camera_images:
            raise ValueError(f"{context} requires at least one camera tensor.")

        ref_shape = tuple(camera_images[0].shape)
        ref_ndim = camera_images[0].ndim
        if ref_ndim not in {4, 5}:
            raise ValueError(
                f"{context} camera tensors must have shape (B, C, H, W) or (B, T, C, H, W). "
                f"Got ndim={ref_ndim}, shape={ref_shape}."
            )

        for camera_idx, images in enumerate(camera_images[1:], start=1):
            if images.ndim != ref_ndim:
                raise ValueError(
                    f"{context} camera tensor {camera_idx} ndim mismatch. "
                    f"Expected {ref_ndim}, got {images.ndim}."
                )
            if tuple(images.shape) != ref_shape:
                raise ValueError(
                    f"{context} camera tensor {camera_idx} shape mismatch. "
                    f"Expected {ref_shape}, got {tuple(images.shape)}."
                )

        num_cameras = len(camera_images)
        if ref_ndim == 4:
            batch_size = ref_shape[0]
            time_steps = None
            flat_images = torch.cat(camera_images, dim=0)
        else:
            batch_size, time_steps = ref_shape[:2]
            flat_images = torch.cat(
                [images.reshape(batch_size * time_steps, *images.shape[2:]) for images in camera_images],
                dim=0,
            )

        return flat_images, batch_size, time_steps, num_cameras

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

    def _encode_multi_camera_images_for_visual_prefix_memory(
        self,
        camera_images: list[Tensor],
    ) -> Tensor:
        flat_images, batch_size, time_steps, num_cameras = self._flatten_camera_image_batch(
            camera_images,
            context="Visual prefix memory",
        )
        features = self.backbone(flat_images)["feature_map"]
        features = self.encoder_img_feat_input_proj(features)
        pooled = self._pool_visual_features_for_memory(features)
        if time_steps is None:
            pooled = pooled.reshape(num_cameras, batch_size, self.config.dim_model)
        else:
            pooled = pooled.reshape(num_cameras, batch_size, time_steps, self.config.dim_model)
        if num_cameras == 1:
            return pooled[0]
        return pooled.mean(dim=0)

    @staticmethod
    def _current_camera_suffix(camera_key: str) -> str:
        if camera_key.startswith("observation.images."):
            return camera_key.removeprefix("observation.images.")
        if camera_key == "observation.image":
            return "main"
        return camera_key

    @staticmethod
    def _prefix_camera_suffix(prefix_image_key: str) -> str:
        return prefix_image_key.removeprefix("observation.prefix_images.")

    def _prefix_images_match_observation_camera_order(self) -> bool:
        current_suffixes = [
            self._current_camera_suffix(key)
            for key in self.config.visual_observation_features
        ]
        prefix_suffixes = [
            self._prefix_camera_suffix(key)
            for key in self.config.prefix_image_features
        ]
        return bool(current_suffixes) and current_suffixes == prefix_suffixes

    def _encode_multi_camera_prefix_images_for_visual_prefix_memory(
        self,
        camera_images: list[Tensor],
        *,
        prefix_mask: Tensor,
        reuse_current_observation_tokens: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        flat_images, batch_size, time_steps, num_cameras = self._flatten_camera_image_batch(
            camera_images,
            context="Visual prefix memory",
        )
        if time_steps is None:
            raise ValueError(
                "Prefix visual memory expects camera tensors with shape "
                "(B, T, C, H, W)."
            )

        features = self.backbone(flat_images)["feature_map"]
        current_image_pos_seq = None
        if reuse_current_observation_tokens:
            current_image_pos_seq = self.encoder_cam_feat_pos_embed(features).to(
                dtype=features.dtype
            )
        features = self.encoder_img_feat_input_proj(features)
        pooled = self._pool_visual_features_for_memory(features)
        visual_embeddings = pooled.reshape(
            num_cameras,
            batch_size,
            time_steps,
            self.config.dim_model,
        )
        if num_cameras == 1:
            visual_embeddings = visual_embeddings[0]
        else:
            visual_embeddings = visual_embeddings.mean(dim=0)

        current_observation_tokens = None
        if reuse_current_observation_tokens:
            _, dim_model, feat_h, feat_w = features.shape
            prefix_mask = prefix_mask.to(device=features.device, dtype=torch.bool)
            valid_lengths = prefix_mask.sum(dim=1).clamp_min(1)
            current_positions = valid_lengths - 1

            camera_features = features.reshape(
                num_cameras,
                batch_size,
                time_steps,
                dim_model,
                feat_h,
                feat_w,
            )
            current_gather_index = current_positions.view(
                1, batch_size, 1, 1, 1, 1
            ).expand(num_cameras, batch_size, 1, dim_model, feat_h, feat_w)
            current_features = camera_features.gather(
                dim=2,
                index=current_gather_index,
            ).squeeze(2)
            self._latest_observation_image_token_layout = {
                "camera_keys": list(self.config.visual_observation_features),
                "num_cameras": int(num_cameras),
                "feature_h": int(feat_h),
                "feature_w": int(feat_w),
            }

            assert current_image_pos_seq is not None
            if current_image_pos_seq.shape[0] == 1:
                current_image_pos_seq = current_image_pos_seq.expand(
                    num_cameras, -1, -1, -1
                )
                current_image_pos_seq = einops.rearrange(
                    current_image_pos_seq,
                    "cam c h w -> (cam h w) 1 c",
                )
            else:
                current_image_pos_seq = current_image_pos_seq.reshape(
                    num_cameras,
                    batch_size,
                    time_steps,
                    dim_model,
                    feat_h,
                    feat_w,
                )
                current_image_pos_seq = current_image_pos_seq.gather(
                    dim=2,
                    index=current_gather_index,
                ).squeeze(2)
                current_image_pos_seq = einops.rearrange(
                    current_image_pos_seq,
                    "cam b c h w -> (cam h w) b c",
                )

            current_image_token_seq = einops.rearrange(
                current_features,
                "cam b c h w -> (cam h w) b c",
            )
            current_observation_tokens = (
                current_image_token_seq,
                current_image_pos_seq,
            )

        return visual_embeddings, current_observation_tokens

    def _reduce_camera_embeddings_for_visual_prefix_memory(
        self,
        camera_embeddings: list[Tensor],
    ) -> Tensor:
        if not camera_embeddings:
            raise ValueError("Visual prefix memory requires at least one camera embedding.")
        if len(camera_embeddings) == 1:
            return camera_embeddings[0]
        return torch.stack(camera_embeddings, dim=0).mean(dim=0)

    def _encode_multi_camera_observation_tokens(
        self,
        camera_images: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        flat_images, batch_size, time_steps, num_cameras = self._flatten_camera_image_batch(
            camera_images,
            context="Observation image encoding",
        )
        if time_steps is not None:
            raise ValueError(
                "Observation image encoding expects current-step camera tensors with shape "
                f"(B, C, H, W). Got time_steps={time_steps}."
            )

        cam_features = self.backbone(flat_images)["feature_map"]
        cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
        cam_features = self.encoder_img_feat_input_proj(cam_features)

        _, dim_model, feat_h, feat_w = cam_features.shape
        self._latest_observation_image_token_layout = {
            "camera_keys": list(self.config.visual_observation_features),
            "num_cameras": int(num_cameras),
            "feature_h": int(feat_h),
            "feature_w": int(feat_w),
        }
        cam_features = cam_features.reshape(num_cameras, batch_size, dim_model, feat_h, feat_w)
        if cam_pos_embed.shape[0] == 1:
            cam_pos_embed = cam_pos_embed.expand(num_cameras, -1, -1, -1)
            cam_pos_embed = einops.rearrange(cam_pos_embed, "cam c h w -> (cam h w) 1 c")
        else:
            cam_pos_embed = cam_pos_embed.reshape(num_cameras, batch_size, dim_model, feat_h, feat_w)
            cam_pos_embed = einops.rearrange(cam_pos_embed, "cam b c h w -> (cam h w) b c")

        return (
            einops.rearrange(cam_features, "cam b c h w -> (cam h w) b c"),
            cam_pos_embed,
        )

    def _project_prefix_states_for_visual_prefix_memory(self, prefix_state: Tensor) -> Tensor:
        batch_size, time_steps, state_dim = prefix_state.shape
        flat_prefix_state = prefix_state.reshape(batch_size * time_steps, state_dim)
        projected = self.encoder_robot_state_input_proj(flat_prefix_state)
        return projected.reshape(batch_size, time_steps, self.config.dim_model)

    def _project_signature_tensor(
        self,
        signature: Tensor,
        *,
        context: str,
    ) -> Tensor:
        if self.signature_proj is None:
            raise RuntimeError(f"{context} requires `self.signature_proj` to be initialized.")
        if signature.ndim not in {2, 3}:
            raise ValueError(
                f"{context} must have shape (B, D_sig) or (B, T, D_sig). Got {tuple(signature.shape)}."
            )
        if signature.shape[-1] != self.config.signature_dim:
            raise ValueError(
                f"{context} trailing dim must equal signature_dim={self.config.signature_dim}. "
                f"Got {signature.shape[-1]}."
            )
        dtype = self.signature_proj[0].weight.dtype
        if signature.ndim == 2:
            return self.signature_proj(signature.to(dtype=dtype))
        batch_size, time_steps, signature_dim = signature.shape
        flat_signature = signature.reshape(batch_size * time_steps, signature_dim)
        projected = self.signature_proj(flat_signature.to(dtype=dtype))
        return projected.reshape(batch_size, time_steps, self.config.dim_model)

    def _project_delta_signature_tensor(
        self,
        delta_signature: Tensor,
        *,
        context: str,
    ) -> Tensor:
        if self.delta_signature_proj is None:
            raise RuntimeError(
                f"{context} requires `self.delta_signature_proj` to be initialized."
            )
        if delta_signature.ndim not in {2, 3}:
            raise ValueError(
                f"{context} must have shape (B, D_sig) or (B, T, D_sig). Got {tuple(delta_signature.shape)}."
            )
        if delta_signature.shape[-1] != self.config.signature_dim:
            raise ValueError(
                f"{context} trailing dim must equal signature_dim={self.config.signature_dim}. "
                f"Got {delta_signature.shape[-1]}."
            )
        dtype = self.delta_signature_proj[0].weight.dtype
        if delta_signature.ndim == 2:
            return self.delta_signature_proj(delta_signature.to(dtype=dtype))
        batch_size, time_steps, signature_dim = delta_signature.shape
        flat_delta_signature = delta_signature.reshape(batch_size * time_steps, signature_dim)
        projected = self.delta_signature_proj(flat_delta_signature.to(dtype=dtype))
        return projected.reshape(batch_size, time_steps, self.config.dim_model)

    def _build_visual_prefix_memory_step_input(
        self,
        *,
        visual_t: Tensor,
        state_t: Tensor,
        signature_t: Tensor | None,
        delta_signature_t: Tensor | None,
    ) -> Tensor:
        inputs = [visual_t, state_t]
        if self.use_signature_conditioned_visual_prefix_memory:
            if signature_t is None:
                raise ValueError(
                    "Signature-conditioned visual prefix memory update requires `signature_t`."
                )
            inputs.append(signature_t)
            if self.use_delta_signature:
                if delta_signature_t is None:
                    raise ValueError(
                        "Delta-signature-conditioned visual prefix memory update requires "
                        "`delta_signature_t` when `use_delta_signature=True`."
                )
                inputs.append(delta_signature_t)
        return torch.cat(inputs, dim=-1)

    def _build_slot_memory_route_features(
        self,
        *,
        signature_t: Tensor | None,
        delta_signature_t: Tensor | None,
        first_frame_anchor_t: Tensor | None,
        context: str,
    ) -> Tensor:
        if not self.use_signature_indexed_slot_memory:
            raise RuntimeError(
                "`_build_slot_memory_route_features` requires "
                "`use_signature_indexed_slot_memory=True`."
            )
        if signature_t is None:
            raise ValueError(f"{context} requires `signature_t` for slot routing.")
        route_features = [signature_t]
        if self.config.slot_memory_use_delta_routing:
            if delta_signature_t is None:
                raise ValueError(
                    f"{context} requires `delta_signature_t` when "
                    "`slot_memory_use_delta_routing=True`."
                )
            route_features.append(delta_signature_t)
        if self.config.use_first_frame_anchor_in_slot_routing:
            if first_frame_anchor_t is None:
                raise ValueError(
                    f"{context} requires `first_frame_anchor_t` when "
                    "`use_first_frame_anchor_in_slot_routing=True`."
                )
            route_features.append(
                first_frame_anchor_t.to(
                    device=signature_t.device,
                    dtype=signature_t.dtype,
                )
            )
        route_features = torch.cat(route_features, dim=-1)
        expected_dim = self.config.dim_model * (
            1
            + int(self.config.slot_memory_use_delta_routing)
            + int(self.config.use_first_frame_anchor_in_slot_routing)
        )
        if route_features.ndim != 2 or route_features.shape[1] != expected_dim:
            raise ValueError(
                f"{context} route features must have shape (batch_size, {expected_dim}). "
                f"Got {tuple(route_features.shape)}."
            )
        return route_features

    def _compute_signature_indexed_slot_memory_route(
        self,
        *,
        memory_prev: Tensor,
        route_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        route_hidden = self.slot_memory_route_proj(route_features)
        routing_query = self.slot_memory_route_query_proj(route_hidden).unsqueeze(1)
        routing_keys = self.slot_memory_route_key_proj(
            self._memory_with_slot_identity(memory_prev)
        )
        routing_logits = torch.matmul(
            routing_query, routing_keys.transpose(1, 2)
        ).squeeze(1) / math.sqrt(self.config.slot_memory_routing_hidden_dim)
        routing_temperature = float(
            getattr(self.config, "slot_memory_routing_temperature", 1.0)
        )
        routing_logits = routing_logits / max(routing_temperature, 1e-6)
        routing_logits_f32 = routing_logits.float()
        if self.config.slot_memory_use_softmax_routing:
            routing_distribution = torch.softmax(routing_logits_f32, dim=-1)
            routing_weights = routing_distribution.to(dtype=routing_logits.dtype)
        else:
            routing_weights_f32 = torch.sigmoid(routing_logits_f32)
            routing_weights = routing_weights_f32.to(dtype=routing_logits.dtype)
            routing_distribution = routing_weights_f32 / routing_weights_f32.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(torch.finfo(torch.float32).tiny)
        return route_hidden, routing_logits, routing_weights, routing_distribution

    def _record_visual_prefix_memory_step_debug(
        self,
        *,
        memory_prev: Tensor,
        memory_next: Tensor,
        visual_t: Tensor,
        state_t: Tensor,
        signature_t: Tensor | None,
        delta_signature_t: Tensor | None,
        first_frame_anchor_t: Tensor | None,
        step_stats: dict[str, Tensor],
    ) -> None:
        memory_prev_norm = memory_prev.detach().float().norm(dim=-1)
        memory_next_norm = memory_next.detach().float().norm(dim=-1)
        memory_delta_norm = (memory_next - memory_prev).detach().float().norm(dim=-1)
        slot_identity = self._build_slot_memory_identity(
            batch_size=memory_prev.shape[0],
            device=memory_prev.device,
            dtype=memory_prev.dtype,
        )
        debug: dict[str, Tensor | float | int | bool] = {
            "enabled": bool(self.use_visual_prefix_memory),
            "signature_indexed_slot_memory": bool(self.use_signature_indexed_slot_memory),
            "num_slots": int(self.active_memory_num_slots),
            "slot_identity_scale": float(
                getattr(self.config, "slot_memory_identity_scale", 0.0)
            ),
            "slot_identity_norm": self._debug_norm(slot_identity),
            "memory_prev_slot_std": self._debug_slot_std(memory_prev),
            "memory_next_slot_std": self._debug_slot_std(memory_next),
            "visual_embedding_norm": self._debug_norm(visual_t),
            "state_embedding_norm": self._debug_norm(state_t),
            "signature_embedding_norm": self._debug_norm(signature_t),
            "delta_signature_embedding_norm": self._debug_norm(delta_signature_t),
            "first_frame_anchor_embedding_norm": self._debug_norm(first_frame_anchor_t),
            "memory_prev_norm": self._debug_tensor(memory_prev_norm),
            "memory_next_norm": self._debug_tensor(memory_next_norm),
            "memory_delta_norm": self._debug_tensor(memory_delta_norm),
        }
        for key in (
            "routing_logits",
            "routing_weights",
            "routing_distribution",
            "routing_weights_zero_signature",
            "routing_delta_from_zero_signature",
            "gate",
            "write_strength",
            "readout_logits",
            "readout_weights",
        ):
            if key in step_stats:
                debug[key] = self._debug_tensor(step_stats[key])
        self._latest_visual_prefix_memory_step_stats = debug

    def _read_signature_indexed_slot_memory_context(
        self,
        *,
        memory_state: Tensor,
        visual_t: Tensor,
        state_t: Tensor,
        signature_t: Tensor | None,
        delta_signature_t: Tensor | None,
        first_frame_anchor_t: Tensor | None,
        route_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if route_hidden is None:
            route_features = self._build_slot_memory_route_features(
                signature_t=signature_t,
                delta_signature_t=delta_signature_t,
                first_frame_anchor_t=first_frame_anchor_t,
                context="Signature-indexed slot memory readout",
            )
            route_hidden = self.slot_memory_route_proj(route_features)
        read_query_input = torch.cat([visual_t, state_t, route_hidden], dim=-1)
        read_query = self.slot_memory_read_query_proj(read_query_input).unsqueeze(1)
        read_keys = self.slot_memory_read_key_proj(
            self._memory_with_slot_identity(memory_state)
        )
        read_values = self.slot_memory_read_value_proj(memory_state)
        read_logits = torch.matmul(
            read_query, read_keys.transpose(1, 2)
        ).squeeze(1) / math.sqrt(self.config.dim_model)
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
        first_frame_anchor_t: Tensor | None,
        valid_t: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size, num_slots, hidden_dim = memory_prev.shape
        route_features = self._build_slot_memory_route_features(
            signature_t=signature_t,
            delta_signature_t=delta_signature_t,
            first_frame_anchor_t=first_frame_anchor_t,
            context="Signature-indexed slot memory update",
        )
        route_hidden, routing_logits, routing_weights, routing_distribution = (
            self._compute_signature_indexed_slot_memory_route(
                memory_prev=memory_prev,
                route_features=route_features,
            )
        )
        _, _, zero_signature_routing_weights, _ = self._compute_signature_indexed_slot_memory_route(
            memory_prev=memory_prev,
            route_features=torch.zeros_like(route_features),
        )
        write_input = torch.cat([visual_t, state_t, route_features], dim=-1)
        write_base = self.slot_memory_write_proj(write_input)
        memory_prev_with_identity = self._memory_with_slot_identity(memory_prev)
        slot_state_input = torch.cat(
            [
                memory_prev_with_identity,
                write_base.unsqueeze(1).expand(batch_size, num_slots, hidden_dim),
                route_hidden.unsqueeze(1).expand(
                    batch_size,
                    num_slots,
                    self.config.slot_memory_routing_hidden_dim,
                ),
            ],
            dim=-1,
        )
        candidate = torch.tanh(self.slot_memory_candidate_proj(slot_state_input))
        gate = torch.sigmoid(self.slot_memory_gate_proj(slot_state_input))
        write_strength = routing_weights.unsqueeze(-1) * gate
        updated_hidden = memory_prev + write_strength * (candidate - memory_prev)

        if valid_t is not None:
            valid_t = valid_t.to(dtype=torch.bool, device=memory_prev.device)
            if valid_t.shape != (batch_size,):
                raise ValueError(
                    "Signature-indexed slot memory valid mask must have shape "
                    f"({batch_size},). Got {tuple(valid_t.shape)}."
                )
            updated_hidden = torch.where(
                valid_t.view(batch_size, 1, 1), updated_hidden, memory_prev
            )

        readout_context, readout_logits, readout_weights = (
            self._read_signature_indexed_slot_memory_context(
                memory_state=updated_hidden,
                visual_t=visual_t,
                state_t=state_t,
                signature_t=signature_t,
                delta_signature_t=delta_signature_t,
                first_frame_anchor_t=first_frame_anchor_t,
                route_hidden=route_hidden,
            )
        )
        return updated_hidden, {
            "routing_logits": routing_logits,
            "routing_weights": routing_weights,
            "routing_distribution": routing_distribution,
            "routing_weights_zero_signature": zero_signature_routing_weights,
            "routing_delta_from_zero_signature": routing_weights - zero_signature_routing_weights,
            "write_base": write_base,
            "candidate": candidate,
            "gate": gate,
            "write_strength": write_strength.squeeze(-1),
            "readout_context": readout_context,
            "readout_logits": readout_logits,
            "readout_weights": readout_weights,
        }

    def _update_visual_prefix_memory_step(
        self,
        *,
        memory_prev: Tensor,
        visual_t: Tensor,
        state_t: Tensor,
        signature_t: Tensor | None,
        delta_signature_t: Tensor | None,
        first_frame_anchor_t: Tensor | None,
        valid_t: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.use_signature_indexed_slot_memory:
            return self._update_signature_indexed_slot_memory_step(
                memory_prev=memory_prev,
                visual_t=visual_t,
                state_t=state_t,
                signature_t=signature_t,
                delta_signature_t=delta_signature_t,
                first_frame_anchor_t=first_frame_anchor_t,
                valid_t=valid_t,
            )
        batch_size = memory_prev.shape[0]
        step_input = self._build_visual_prefix_memory_step_input(
            visual_t=visual_t,
            state_t=state_t,
            signature_t=signature_t,
            delta_signature_t=delta_signature_t,
        )
        updated_hidden = torch.stack(
            [
                slot_update(step_input, memory_prev[:, slot_idx, :])
                for slot_idx, slot_update in enumerate(self._iter_visual_prefix_memory_updates())
            ],
            dim=1,
        )
        if valid_t is None:
            return updated_hidden, {}
        valid_t = valid_t.to(dtype=torch.bool, device=memory_prev.device)
        if valid_t.shape != (batch_size,):
            raise ValueError(
                "Visual prefix memory valid mask must have shape "
                f"({batch_size},). Got {tuple(valid_t.shape)}."
            )
        return torch.where(valid_t.view(batch_size, 1, 1), updated_hidden, memory_prev), {}

    def _build_visual_prefix_memory_aux_losses(
        self,
        *,
        routing_distribution_sum: Tensor | None,
        routing_entropy_sum: Tensor | None,
        consistency_loss_sum: Tensor | None,
        valid_step_count: Tensor | None,
        device: torch.device,
    ) -> dict[str, Tensor]:
        aux_losses: dict[str, Tensor] = {}
        if valid_step_count is None:
            return aux_losses
        valid_step_count = valid_step_count.float().clamp_min(1.0)
        if (
            self.use_signature_indexed_slot_memory
            and self.config.slot_memory_balance_loss_coef > 0.0
        ):
            if routing_distribution_sum is None:
                raise RuntimeError(
                    "Missing routing distribution statistics for slot-memory "
                    "balance loss."
                )
            average_routing = routing_distribution_sum.float() / valid_step_count
            uniform = torch.full_like(
                average_routing,
                fill_value=1.0 / float(self.active_memory_num_slots),
            )
            aux_losses["slot_memory_balance_loss"] = F.mse_loss(
                average_routing,
                uniform,
                reduction="mean",
            )
        if (
            self.use_signature_indexed_slot_memory
            and self.config.slot_memory_entropy_loss_coef > 0.0
        ):
            if routing_entropy_sum is None:
                raise RuntimeError(
                    "Missing routing entropy statistics for slot-memory entropy loss."
                )
            max_entropy = math.log(float(max(1, self.active_memory_num_slots)))
            if max_entropy <= 0.0:
                aux_losses["slot_memory_entropy_loss"] = torch.zeros(
                    (),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                aux_losses["slot_memory_entropy_loss"] = (
                    routing_entropy_sum.float() / valid_step_count
                ) / max_entropy
        if (
            self.use_signature_indexed_slot_memory
            and self.config.slot_memory_consistency_loss_coef > 0.0
        ):
            if consistency_loss_sum is None:
                raise RuntimeError(
                    "Missing readout/write statistics for slot-memory consistency loss."
                )
            aux_losses["slot_memory_consistency_loss"] = (
                consistency_loss_sum.float() / valid_step_count
            )
        return aux_losses

    def _scan_visual_prefix_memory(
        self,
        *,
        visual_embeddings: Tensor,
        state_embeddings: Tensor,
        signature_embeddings: Tensor | None,
        delta_signature_embeddings: Tensor | None,
        first_frame_anchor_embedding: Tensor | None,
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
        if signature_embeddings is not None:
            assert signature_embeddings.shape == (batch_size, time_steps, hidden_dim), (
                "Visual prefix memory signature embeddings must match visual embeddings. "
                f"Got signature={tuple(signature_embeddings.shape)} vs "
                f"visual={tuple(visual_embeddings.shape)}."
            )
        if delta_signature_embeddings is not None:
            assert delta_signature_embeddings.shape == (batch_size, time_steps, hidden_dim), (
                "Visual prefix memory delta-signature embeddings must match visual embeddings. "
                f"Got delta_signature={tuple(delta_signature_embeddings.shape)} vs "
                f"visual={tuple(visual_embeddings.shape)}."
            )
        if self.config.use_first_frame_anchor_in_slot_routing:
            first_frame_anchor_embedding = self._normalize_first_frame_anchor_embedding(
                first_frame_anchor_embedding,
                batch_size=batch_size,
                device=visual_embeddings.device,
                dtype=visual_embeddings.dtype,
                context="Visual prefix memory scan",
            )
        elif first_frame_anchor_embedding is not None:
            first_frame_anchor_embedding = first_frame_anchor_embedding.to(
                device=visual_embeddings.device,
                dtype=visual_embeddings.dtype,
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
        need_balance_loss = (
            self.use_signature_indexed_slot_memory
            and self.config.slot_memory_balance_loss_coef > 0.0
        )
        need_entropy_loss = (
            self.use_signature_indexed_slot_memory
            and self.config.slot_memory_entropy_loss_coef > 0.0
        )
        need_consistency_loss = (
            self.use_signature_indexed_slot_memory
            and self.config.slot_memory_consistency_loss_coef > 0.0
        )
        routing_distribution_sum = None
        routing_entropy_sum = None
        consistency_loss_sum = None
        valid_step_count = None
        collect_slot_metrics = self.use_signature_indexed_slot_memory
        metric_valid_count = None
        metric_routing_weight_sum = None
        metric_routing_distribution_sum = None
        metric_readout_weight_sum = None
        metric_write_strength_sum = None
        metric_routing_entropy_sum = None
        metric_routing_max_sum = None
        metric_routing_std_sum = None
        metric_readout_entropy_sum = None
        metric_routing_delta_abs_sum = None
        if collect_slot_metrics:
            metric_device = hidden.device
            metric_slot_shape = (self.active_memory_num_slots,)
            metric_valid_count = torch.zeros((), device=metric_device, dtype=torch.float32)
            metric_routing_weight_sum = torch.zeros(
                metric_slot_shape, device=metric_device, dtype=torch.float32
            )
            metric_routing_distribution_sum = torch.zeros(
                metric_slot_shape, device=metric_device, dtype=torch.float32
            )
            metric_readout_weight_sum = torch.zeros(
                metric_slot_shape, device=metric_device, dtype=torch.float32
            )
            metric_write_strength_sum = torch.zeros(
                metric_slot_shape, device=metric_device, dtype=torch.float32
            )
            metric_routing_entropy_sum = torch.zeros(
                (), device=metric_device, dtype=torch.float32
            )
            metric_routing_max_sum = torch.zeros((), device=metric_device, dtype=torch.float32)
            metric_routing_std_sum = torch.zeros((), device=metric_device, dtype=torch.float32)
            metric_readout_entropy_sum = torch.zeros(
                (), device=metric_device, dtype=torch.float32
            )
            metric_routing_delta_abs_sum = torch.zeros(
                (), device=metric_device, dtype=torch.float32
            )
        for step_idx in range(time_steps):
            hidden, step_stats = self._update_visual_prefix_memory_step(
                memory_prev=hidden,
                visual_t=visual_embeddings[:, step_idx],
                state_t=state_embeddings[:, step_idx],
                signature_t=(
                    None if signature_embeddings is None else signature_embeddings[:, step_idx]
                ),
                delta_signature_t=(
                    None
                    if delta_signature_embeddings is None
                    else delta_signature_embeddings[:, step_idx]
                ),
                first_frame_anchor_t=first_frame_anchor_embedding,
                valid_t=prefix_mask[:, step_idx],
            )
            if need_balance_loss:
                step_distribution = step_stats["routing_distribution"].float()
                masked_distribution = (
                    step_distribution
                    * prefix_mask[:, step_idx].to(step_distribution.dtype).unsqueeze(-1)
                )
                step_distribution_sum = masked_distribution.sum(dim=0)
                if routing_distribution_sum is None:
                    routing_distribution_sum = step_distribution_sum
                else:
                    routing_distribution_sum = routing_distribution_sum + step_distribution_sum
            if need_entropy_loss:
                step_entropy = self._categorical_entropy_from_probabilities(
                    step_stats["routing_distribution"]
                )
                masked_step_entropy = (
                    step_entropy * prefix_mask[:, step_idx].to(step_entropy.dtype)
                ).sum()
                if routing_entropy_sum is None:
                    routing_entropy_sum = masked_step_entropy
                else:
                    routing_entropy_sum = routing_entropy_sum + masked_step_entropy
            if collect_slot_metrics:
                metric_mask = prefix_mask[:, step_idx].to(dtype=torch.float32)
                metric_mask_2d = metric_mask.unsqueeze(-1)
                metric_valid_count = metric_valid_count + metric_mask.sum()

                routing_weights = step_stats["routing_weights"].detach().float()
                routing_distribution = step_stats["routing_distribution"].detach().float()
                readout_weights = step_stats["readout_weights"].detach().float()
                write_strength = step_stats["write_strength"].detach().float()

                metric_routing_weight_sum = metric_routing_weight_sum + (
                    routing_weights * metric_mask_2d
                ).sum(dim=0)
                metric_routing_distribution_sum = metric_routing_distribution_sum + (
                    routing_distribution * metric_mask_2d
                ).sum(dim=0)
                metric_readout_weight_sum = metric_readout_weight_sum + (
                    readout_weights * metric_mask_2d
                ).sum(dim=0)
                metric_write_strength_sum = metric_write_strength_sum + (
                    write_strength * metric_mask_2d
                ).sum(dim=0)

                routing_entropy = self._categorical_entropy_from_probabilities(
                    routing_distribution
                )
                readout_entropy = self._categorical_entropy_from_probabilities(
                    readout_weights
                )
                metric_routing_entropy_sum = metric_routing_entropy_sum + (
                    routing_entropy * metric_mask
                ).sum()
                metric_routing_max_sum = metric_routing_max_sum + (
                    routing_weights.max(dim=-1).values * metric_mask
                ).sum()
                metric_routing_std_sum = metric_routing_std_sum + (
                    routing_weights.std(dim=-1, unbiased=False) * metric_mask
                ).sum()
                metric_readout_entropy_sum = metric_readout_entropy_sum + (
                    readout_entropy * metric_mask
                ).sum()
                if "routing_weights_zero_signature" in step_stats:
                    zero_routing_weights = (
                        step_stats["routing_weights_zero_signature"].detach().float()
                    )
                    metric_routing_delta_abs_sum = metric_routing_delta_abs_sum + (
                        (routing_weights - zero_routing_weights).abs().mean(dim=-1)
                        * metric_mask
                    ).sum()
            if need_consistency_loss:
                step_consistency = self._bounded_slot_memory_consistency_loss(
                    step_stats["readout_context"],
                    step_stats["write_base"],
                )
                masked_step_consistency = (
                    step_consistency * prefix_mask[:, step_idx].to(step_consistency.dtype)
                ).sum()
                if consistency_loss_sum is None:
                    consistency_loss_sum = masked_step_consistency
                else:
                    consistency_loss_sum = consistency_loss_sum + masked_step_consistency
            if need_balance_loss or need_entropy_loss or need_consistency_loss:
                step_valid_count = prefix_mask[:, step_idx].to(torch.float32).sum()
                if valid_step_count is None:
                    valid_step_count = step_valid_count
                else:
                    valid_step_count = valid_step_count + step_valid_count
        aux_losses = self._build_visual_prefix_memory_aux_losses(
            routing_distribution_sum=routing_distribution_sum,
            routing_entropy_sum=routing_entropy_sum,
            consistency_loss_sum=consistency_loss_sum,
            valid_step_count=valid_step_count,
            device=hidden.device,
        )
        if collect_slot_metrics:
            assert metric_valid_count is not None
            metric_denom = metric_valid_count.clamp_min(1.0)
            max_entropy = math.log(float(max(1, self.active_memory_num_slots)))
            routing_entropy = metric_routing_entropy_sum / metric_denom
            readout_entropy = metric_readout_entropy_sum / metric_denom
            log_stats: dict[str, float] = {
                "slot_memory/valid_prefix_steps": float(
                    metric_valid_count.detach().cpu().item()
                ),
                "slot_memory/routing_entropy": float(
                    routing_entropy.detach().cpu().item()
                ),
                "slot_memory/routing_max": float(
                    (metric_routing_max_sum / metric_denom).detach().cpu().item()
                ),
                "slot_memory/routing_std": float(
                    (metric_routing_std_sum / metric_denom).detach().cpu().item()
                ),
                "slot_memory/routing_delta_from_zero_abs_mean": float(
                    (metric_routing_delta_abs_sum / metric_denom).detach().cpu().item()
                ),
                "slot_memory/readout_entropy": float(
                    readout_entropy.detach().cpu().item()
                ),
                "slot_memory/write_strength_mean": float(
                    (metric_write_strength_sum.sum() / metric_denom)
                    .detach()
                    .cpu()
                    .item()
                ),
                "slot_memory/memory_final_slot_std": self._debug_slot_std(hidden),
            }
            if self.config.use_first_frame_anchor_in_slot_routing:
                log_stats["slot_memory/first_frame_anchor_norm"] = self._debug_norm(
                    first_frame_anchor_embedding
                )
            if max_entropy > 0.0:
                log_stats["slot_memory/routing_entropy_normalized"] = float(
                    (routing_entropy / max_entropy).detach().cpu().item()
                )
                log_stats["slot_memory/readout_entropy_normalized"] = float(
                    (readout_entropy / max_entropy).detach().cpu().item()
                )

            routing_weight_mean = metric_routing_weight_sum / metric_denom
            routing_distribution_mean = metric_routing_distribution_sum / metric_denom
            readout_weight_mean = metric_readout_weight_sum / metric_denom
            write_strength_mean = metric_write_strength_sum / metric_denom
            for slot_idx in range(int(self.active_memory_num_slots)):
                log_stats[f"slot_memory/routing_weight/slot_{slot_idx}"] = float(
                    routing_weight_mean[slot_idx].detach().cpu().item()
                )
                log_stats[f"slot_memory/routing_distribution/slot_{slot_idx}"] = float(
                    routing_distribution_mean[slot_idx].detach().cpu().item()
                )
                log_stats[f"slot_memory/readout_weight/slot_{slot_idx}"] = float(
                    readout_weight_mean[slot_idx].detach().cpu().item()
                )
                log_stats[f"slot_memory/write_strength/slot_{slot_idx}"] = float(
                    write_strength_mean[slot_idx].detach().cpu().item()
                )
            self._last_visual_prefix_memory_log_stats = log_stats
        return hidden, aux_losses

    def _compute_visual_prefix_memory_context(
        self,
        *,
        memory_state: Tensor,
        visual_embedding: Tensor,
        state_embedding: Tensor,
        signature_embedding: Tensor | None,
        delta_signature_embedding: Tensor | None,
        first_frame_anchor_embedding: Tensor | None,
    ) -> Tensor:
        if memory_state.ndim != 3:
            raise ValueError(
                "Visual prefix memory state must have shape "
                f"(batch_size, num_memory_slots, dim_model). Got {tuple(memory_state.shape)}."
            )
        if not self.use_signature_indexed_slot_memory:
            return memory_state.mean(dim=1)
        if not self.config.slot_memory_use_readout_pooling:
            return memory_state.mean(dim=1)
        readout_context, _, _ = self._read_signature_indexed_slot_memory_context(
            memory_state=memory_state,
            visual_t=visual_embedding,
            state_t=state_embedding,
            signature_t=signature_embedding,
            delta_signature_t=delta_signature_embedding,
            first_frame_anchor_t=first_frame_anchor_embedding,
        )
        return readout_context

    def _apply_visual_prefix_memory_encoder_film(
        self,
        encoder_tokens: Tensor,
        *,
        memory_context: Tensor,
        exclude_prefix_tokens: int = 0,
    ) -> Tensor:
        if self.visual_prefix_memory_encoder_film is None:
            raise RuntimeError(
                "`_apply_visual_prefix_memory_encoder_film` requires "
                "`self.visual_prefix_memory_encoder_film` to be initialized."
            )
        if encoder_tokens.ndim != 3:
            raise ValueError(
                "Encoder tokens must have shape (sequence, batch_size, dim_model). "
                f"Got {tuple(encoder_tokens.shape)}."
            )
        if memory_context.ndim != 2:
            raise ValueError(
                "Memory context must have shape (batch_size, dim_model). "
                f"Got {tuple(memory_context.shape)}."
            )
        seq_len, batch_size, dim_model = encoder_tokens.shape
        if memory_context.shape != (batch_size, dim_model):
            raise ValueError(
                "Memory context shape mismatch. "
                f"Expected {(batch_size, dim_model)}, got {tuple(memory_context.shape)}."
            )
        if exclude_prefix_tokens < 0 or exclude_prefix_tokens > seq_len:
            raise ValueError(
                "`exclude_prefix_tokens` must lie in [0, sequence_length]. "
                f"Got {exclude_prefix_tokens} for sequence_length={seq_len}."
            )
        if exclude_prefix_tokens == seq_len:
            return encoder_tokens

        film_params = self.visual_prefix_memory_encoder_film(
            memory_context.to(dtype=encoder_tokens.dtype)
        )
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        target_tokens = encoder_tokens[exclude_prefix_tokens:]
        conditioned = target_tokens * (1.0 + gamma.unsqueeze(0)) + beta.unsqueeze(0)
        if exclude_prefix_tokens == 0:
            return conditioned
        return torch.cat([encoder_tokens[:exclude_prefix_tokens], conditioned], dim=0)

    def _compute_visual_prefix_memory_token_from_prefix_sequence(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor], tuple[Tensor, Tensor] | None, Tensor | None]:
        assert PREFIX_STATE_KEY in batch, (
            f"`{PREFIX_STATE_KEY}` is required to reconstruct visual prefix memory during training."
        )
        assert PREFIX_MASK_KEY in batch, (
            f"`{PREFIX_MASK_KEY}` is required to reconstruct visual prefix memory during training."
        )
        prefix_camera_images = [
            batch[prefix_image_key] for prefix_image_key in self.config.prefix_image_features
        ]
        (
            visual_embeddings,
            current_observation_tokens,
        ) = self._encode_multi_camera_prefix_images_for_visual_prefix_memory(
            prefix_camera_images,
            prefix_mask=batch[PREFIX_MASK_KEY],
            reuse_current_observation_tokens=(
                bool(self.config.visual_observation_features)
                and self._prefix_images_match_observation_camera_order()
            ),
        )
        state_embeddings = self._project_prefix_states_for_visual_prefix_memory(batch[PREFIX_STATE_KEY])
        first_frame_anchor_embedding = None
        if self.config.use_first_frame_anchor_in_slot_routing:
            prefix_mask = batch[PREFIX_MASK_KEY].to(
                device=visual_embeddings.device,
                dtype=torch.bool,
            )
            if not torch.all(prefix_mask[:, 0]):
                raise ValueError(
                    "First-frame anchor slot routing expects the first prefix "
                    "position to be valid for every batch row."
                )
            first_frame_anchor_embedding = visual_embeddings[:, 0]
        signature_embeddings = None
        delta_signature_embeddings = None
        uses_signature_routed_memory = (
            self.use_signature_indexed_slot_memory
            or self.use_signature_conditioned_visual_prefix_memory
        )
        requires_delta_memory_signature = (
            self.use_signature_indexed_slot_memory
            and self.config.slot_memory_use_delta_routing
        ) or (
            self.use_signature_conditioned_visual_prefix_memory
            and self.use_delta_signature
        )
        if uses_signature_routed_memory:
            assert PREFIX_PATH_SIGNATURE_KEY in batch, (
                f"`{PREFIX_PATH_SIGNATURE_KEY}` is required to reconstruct "
                "signature-routed visual prefix memory during training."
            )
            signature_embeddings = self._project_signature_tensor(
                batch[PREFIX_PATH_SIGNATURE_KEY],
                context="Prefix path-signature sequence",
            )
            if requires_delta_memory_signature:
                assert PREFIX_DELTA_SIGNATURE_KEY in batch, (
                    f"`{PREFIX_DELTA_SIGNATURE_KEY}` is required to reconstruct "
                    "delta-signature-routed visual prefix memory during training."
                )
                delta_signature_embeddings = self._project_delta_signature_tensor(
                    batch[PREFIX_DELTA_SIGNATURE_KEY],
                    context="Prefix delta-signature sequence",
                )
        memory_state, aux_losses = self._scan_visual_prefix_memory(
            visual_embeddings=visual_embeddings,
            state_embeddings=state_embeddings,
            signature_embeddings=signature_embeddings,
            delta_signature_embeddings=delta_signature_embeddings,
            first_frame_anchor_embedding=first_frame_anchor_embedding,
            prefix_mask=batch[PREFIX_MASK_KEY],
        )
        return (
            memory_state,
            aux_losses,
            current_observation_tokens,
            first_frame_anchor_embedding,
        )

    def compute_online_visual_prefix_memory_token(
        self,
        batch: dict[str, Tensor],
        *,
        previous_state: Tensor | None = None,
        first_frame_anchor_embedding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
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

        visual_embedding = self._encode_multi_camera_images_for_visual_prefix_memory(
            batch[OBS_IMAGES]
        )
        if self.config.use_first_frame_anchor_in_slot_routing:
            if first_frame_anchor_embedding is None:
                first_frame_anchor_embedding = visual_embedding.detach()
            first_frame_anchor_embedding = self._normalize_first_frame_anchor_embedding(
                first_frame_anchor_embedding,
                batch_size=visual_embedding.shape[0],
                device=visual_embedding.device,
                dtype=visual_embedding.dtype,
                context="Online visual prefix memory update",
            )
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
        signature_embedding = None
        delta_signature_embedding = None
        uses_signature_routed_memory = (
            self.use_signature_indexed_slot_memory
            or self.use_signature_conditioned_visual_prefix_memory
        )
        requires_delta_memory_signature = (
            self.use_signature_indexed_slot_memory
            and self.config.slot_memory_use_delta_routing
        ) or (
            self.use_signature_conditioned_visual_prefix_memory
            and self.use_delta_signature
        )
        if uses_signature_routed_memory:
            assert PATH_SIGNATURE_KEY in batch, (
                "Online signature-routed visual prefix memory update requires "
                f"`{PATH_SIGNATURE_KEY}` in the batch."
            )
            signature_embedding = self._project_signature_tensor(
                batch[PATH_SIGNATURE_KEY],
                context="Current path signature",
            )
            if requires_delta_memory_signature:
                assert DELTA_SIGNATURE_KEY in batch, (
                    "Online delta-signature-routed visual prefix memory update requires "
                    f"`{DELTA_SIGNATURE_KEY}` in the batch."
                )
                delta_signature_embedding = self._project_delta_signature_tensor(
                    batch[DELTA_SIGNATURE_KEY],
                    context="Current delta signature",
                )
        next_state, step_stats = self._update_visual_prefix_memory_step(
            memory_prev=hidden,
            visual_t=visual_embedding,
            state_t=state_embedding,
            signature_t=signature_embedding,
            delta_signature_t=delta_signature_embedding,
            first_frame_anchor_t=first_frame_anchor_embedding,
            valid_t=torch.ones(
                (visual_embedding.shape[0],),
                dtype=torch.bool,
                device=visual_embedding.device,
            ),
        )
        self._record_visual_prefix_memory_step_debug(
            memory_prev=hidden,
            memory_next=next_state,
            visual_t=visual_embedding,
            state_t=state_embedding,
            signature_t=signature_embedding,
            delta_signature_t=delta_signature_embedding,
            first_frame_anchor_t=first_frame_anchor_embedding,
            step_stats=step_stats,
        )
        return next_state, next_state, first_frame_anchor_embedding

    def get_deploy_debug_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "decoder_token_layout": dict(self._latest_decoder_token_layout),
            "slot_memory": dict(self._latest_visual_prefix_memory_step_stats),
        }
        if self.decoder.layers:
            attn_weights = self.decoder.layers[-1].last_cross_attn_weights
            if attn_weights is not None:
                attn = attn_weights
                if attn.ndim == 4:
                    # (B, heads, decoder_queries, encoder_source_tokens)
                    attn = attn.mean(dim=1)
                if attn.ndim == 3 and attn.shape[0] > 0:
                    snapshot["decoder_cross_attention"] = self._debug_tensor(attn[0])
        return snapshot

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        visual_prefix_memory_token: Tensor | None = None,
        visual_prefix_anchor_embedding: Tensor | None = None,
        skip_prefix_sequence_validation: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of current-step images.
            [FIRST_FRAME_ANCHOR_KEY] (optional): (B, C, H, W) first-frame anchor image.
            [PATH_SIGNATURE_KEY] (optional): (B, signature_dim) current path signature.
            [DELTA_SIGNATURE_KEY] (optional): (B, signature_dim) current delta signature.
            [PREFIX_STATE_KEY] (optional): (B, T_prefix, state_dim) prefix state sequence.
            [PREFIX_PATH_SIGNATURE_KEY] (optional): (B, T_prefix, signature_dim) prefix signature sequence.
            [PREFIX_DELTA_SIGNATURE_KEY] (optional): (B, T_prefix, signature_dim) prefix delta-signature sequence.
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
        self._last_visual_prefix_memory_aux_losses = {}
        self._last_visual_prefix_memory_log_stats = {}
        if not skip_prefix_sequence_validation:
            self._validate_prefix_sequence_inputs(batch, batch_size)

        signature_embed = None
        signature_step_embed = None
        if self.use_path_signature:
            assert PATH_SIGNATURE_KEY in batch, (
                f"`{PATH_SIGNATURE_KEY}` is required when `use_path_signature=True`."
            )
            path_signature = batch[PATH_SIGNATURE_KEY]
            assert path_signature.ndim == 2, (
                f"`{PATH_SIGNATURE_KEY}` must have shape (batch_size, signature_dim). "
                f"Got ndim={path_signature.ndim}, shape={tuple(path_signature.shape)}."
            )
            assert path_signature.shape[0] == batch_size, (
                f"Batch mismatch for `{PATH_SIGNATURE_KEY}`: expected {batch_size}, "
                f"got {path_signature.shape[0]}."
            )
            assert path_signature.shape[1] == self.config.signature_dim, (
                f"`{PATH_SIGNATURE_KEY}` second dim must be `signature_dim={self.config.signature_dim}`. "
                f"Got {path_signature.shape[1]}."
            )
            signature_step_embed = self._project_signature_tensor(
                path_signature,
                context="Current path signature",
            )
            assert signature_step_embed.shape == (batch_size, self.config.dim_model), (
                f"`signature_embed` must have shape ({batch_size}, {self.config.dim_model}). "
                f"Got {tuple(signature_step_embed.shape)}."
            )
            signature_embed = signature_step_embed.unsqueeze(1)  # (B, 1, D)
        delta_signature_embed = None
        delta_signature_step_embed = None
        if self.use_delta_signature:
            assert DELTA_SIGNATURE_KEY in batch, (
                f"`{DELTA_SIGNATURE_KEY}` is required when `use_delta_signature=True`."
            )
            delta_signature = batch[DELTA_SIGNATURE_KEY]
            assert delta_signature.ndim == 2, (
                f"`{DELTA_SIGNATURE_KEY}` must have shape (batch_size, signature_dim). "
                f"Got ndim={delta_signature.ndim}, shape={tuple(delta_signature.shape)}."
            )
            assert delta_signature.shape[0] == batch_size, (
                f"Batch mismatch for `{DELTA_SIGNATURE_KEY}`: expected {batch_size}, "
                f"got {delta_signature.shape[0]}."
            )
            assert delta_signature.shape[1] == self.config.signature_dim, (
                f"`{DELTA_SIGNATURE_KEY}` second dim must be "
                f"`signature_dim={self.config.signature_dim}`. "
                f"Got {delta_signature.shape[1]}."
            )
            delta_signature_step_embed = self._project_delta_signature_tensor(
                delta_signature,
                context="Current delta signature",
            )
            assert delta_signature_step_embed.shape == (batch_size, self.config.dim_model), (
                f"`delta_signature_embed` must have shape ({batch_size}, {self.config.dim_model}). "
                f"Got {tuple(delta_signature_step_embed.shape)}."
            )
            delta_signature_embed = delta_signature_step_embed.unsqueeze(1)  # (B, 1, D)

        anchor_embed = None
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
        first_frame_anchor_step_embed = None
        cached_observation_image_tokens = None
        if self.use_visual_prefix_memory:
            if visual_prefix_memory_token is None:
                (
                    visual_prefix_memory_embed,
                    self._last_visual_prefix_memory_aux_losses,
                    cached_observation_image_tokens,
                    first_frame_anchor_step_embed,
                ) = self._compute_visual_prefix_memory_token_from_prefix_sequence(batch)
            else:
                if visual_prefix_memory_token.ndim != 3:
                    raise ValueError(
                        "Visual prefix memory token override must have shape "
                        f"(batch_size, num_memory_slots, dim_model). Got {tuple(visual_prefix_memory_token.shape)}."
                    )
                expected_shape = (
                    batch_size,
                    self.active_memory_num_slots,
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
                self._last_visual_prefix_memory_aux_losses = {}
                if self.config.use_first_frame_anchor_in_slot_routing:
                    first_frame_anchor_step_embed = (
                        self._normalize_first_frame_anchor_embedding(
                            visual_prefix_anchor_embedding,
                            batch_size=batch_size,
                            device=visual_prefix_memory_embed.device,
                            dtype=visual_prefix_memory_embed.dtype,
                            context="Visual prefix memory token override",
                        )
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
            latent_std = _stable_vae_log_variance(log_sigma_x2).mul(0.5).exp()
            latent_sample = mu + latent_std.to(dtype=mu.dtype) * torch.randn_like(mu)
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
        current_robot_state_embed = None
        # Robot state token.
        if self.config.robot_state_feature:
            current_robot_state_embed = self.encoder_robot_state_input_proj(batch[OBS_STATE])
            encoder_in_tokens.append(current_robot_state_embed)
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        encoder_1d_token_count = len(encoder_in_tokens)

        image_token_seq = None
        image_pos_seq = None
        if self.config.visual_observation_features:
            if cached_observation_image_tokens is None:
                image_token_seq, image_pos_seq = self._encode_multi_camera_observation_tokens(
                    batch[OBS_IMAGES]
                )
            else:
                image_token_seq, image_pos_seq = cached_observation_image_tokens

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        if image_token_seq is not None and image_pos_seq is not None:
            encoder_in_tokens = torch.cat([encoder_in_tokens, image_token_seq], dim=0)
            encoder_in_pos_embed = torch.cat([encoder_in_pos_embed, image_pos_seq], dim=0)

        if self.use_memory_conditioned_encoder_film:
            assert visual_prefix_memory_embed is not None, (
                "Memory-conditioned encoder FiLM requires `visual_prefix_memory_embed`."
            )
            assert current_robot_state_embed is not None, (
                "Memory-conditioned encoder FiLM requires a projected robot state token."
            )
            assert image_token_seq is not None, (
                "Memory-conditioned encoder FiLM requires current-step image tokens "
                "to derive the readout query."
            )
            memory_context = self._compute_visual_prefix_memory_context(
                memory_state=visual_prefix_memory_embed,
                visual_embedding=image_token_seq.mean(dim=0),
                state_embedding=current_robot_state_embed,
                signature_embedding=signature_step_embed,
                delta_signature_embedding=delta_signature_step_embed,
                first_frame_anchor_embedding=first_frame_anchor_step_embed,
            )
            # Keep the latent token untouched and modulate the current-step observation tokens.
            encoder_in_tokens = self._apply_visual_prefix_memory_encoder_film(
                encoder_in_tokens,
                memory_context=memory_context,
                exclude_prefix_tokens=1,
            )

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        extra_memory_tokens = []
        extra_memory_pos_embed = []
        extra_memory_token_labels: list[str] = []
        # Decoder cross-attention memory order:
        # [anchor?, signature?, delta_signature?, prefix_memory_slots?, encoder_tokens...].
        # These extra 1D tokens intentionally use zero positional embeddings; token type
        # is conveyed by their dedicated projection path and fixed insertion order.
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
            extra_memory_token_labels.append("anchor")
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
            extra_memory_token_labels.append("path_signature")
        if self.use_delta_signature:
            assert delta_signature_embed is not None
            delta_signature_token = delta_signature_embed.transpose(0, 1).to(
                device=encoder_out.device,
                dtype=encoder_out.dtype,
            )  # (1, B, D)
            delta_signature_pos_embed = torch.zeros(
                (1, 1, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )
            extra_memory_tokens.append(delta_signature_token)
            extra_memory_pos_embed.append(delta_signature_pos_embed)
            extra_memory_token_labels.append("delta_signature")
        if self.use_visual_prefix_memory:
            assert visual_prefix_memory_embed is not None
            visual_prefix_memory_for_decoder = self._memory_with_slot_identity(
                visual_prefix_memory_embed
            )
            visual_prefix_memory_token = visual_prefix_memory_for_decoder.transpose(0, 1).to(
                device=encoder_out.device,
                dtype=encoder_out.dtype,
            )
            visual_prefix_memory_pos_embed = torch.zeros(
                (self.active_memory_num_slots, 1, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )
            extra_memory_tokens.append(visual_prefix_memory_token)
            extra_memory_pos_embed.append(visual_prefix_memory_pos_embed)
            extra_memory_token_labels.extend(
                f"memory_slot_{slot_idx}"
                for slot_idx in range(int(self.active_memory_num_slots))
            )

        if extra_memory_tokens:
            # Extra non-image context tokens are injected into encoder memory before decoder cross-attention.
            encoder_out = torch.cat([*extra_memory_tokens, encoder_out], dim=0)
            encoder_in_pos_embed = torch.cat([*extra_memory_pos_embed, encoder_in_pos_embed], dim=0)
            assert encoder_out.shape[0] == encoder_in_pos_embed.shape[0], (
                "Encoder token length and positional embedding length must match after "
                "extra memory token injection."
            )

        image_layout = dict(self._latest_observation_image_token_layout)
        num_cameras = int(image_layout.get("num_cameras", 0) or 0)
        feature_h = int(image_layout.get("feature_h", 0) or 0)
        feature_w = int(image_layout.get("feature_w", 0) or 0)
        image_token_count = num_cameras * feature_h * feature_w
        self._latest_decoder_token_layout = {
            "extra_memory_token_labels": list(extra_memory_token_labels),
            "extra_memory_token_count": int(len(extra_memory_token_labels)),
            "encoder_1d_token_count": int(encoder_1d_token_count),
            "image_token_start": int(len(extra_memory_token_labels) + encoder_1d_token_count),
            "image_token_count": int(image_token_count),
            "camera_keys": list(image_layout.get("camera_keys", [])),
            "num_cameras": int(num_cameras),
            "feature_h": int(feature_h),
            "feature_w": int(feature_w),
            "source_token_count": int(encoder_out.shape[0]),
        }

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

    def __init__(self, config: StreamingACTConfig, is_vae_encoder: bool = False):
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
    def __init__(self, config: StreamingACTConfig):
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
    def __init__(self, config: StreamingACTConfig):
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
    def __init__(self, config: StreamingACTConfig):
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
        self.last_cross_attn_weights: Tensor | None = None

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
        need_cross_attn_weights = not self.training
        x, cross_attn_weights = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
            need_weights=need_cross_attn_weights,
            average_attn_weights=True,
        )
        self.last_cross_attn_weights = (
            None if cross_attn_weights is None else cross_attn_weights.detach()
        )
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
