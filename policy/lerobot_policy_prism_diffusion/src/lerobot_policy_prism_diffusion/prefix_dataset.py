from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lerobot_policy_streaming_act.prefix_sequence import (
    DELTA_SIGNATURE_KEY,
    PATH_SIGNATURE_KEY,
    PREFIX_DELTA_SIGNATURE_KEY,
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
    build_prefix_mask,
    clone_prefix_stats,
    is_supported_prefix_camera_key,
    pad_prefix_tensor,
    prefix_image_key_from_camera_key,
    select_prefix_positions,
)


def _ensure_prefix_image_sequence(camera_key: str, tensor: Tensor | Any) -> Tensor:
    tensor = torch.as_tensor(tensor)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(
            f"Prefix images for `{camera_key}` must have shape (T, C, H, W). "
            f"Got shape={tuple(tensor.shape)}."
        )
    return tensor


def _ensure_prefix_vector_sequence(feature_key: str, tensor: Tensor | Any) -> Tensor:
    tensor = torch.as_tensor(tensor)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Prefix feature `{feature_key}` must have shape (T, D). "
            f"Got shape={tuple(tensor.shape)}."
        )
    return tensor


def _to_python_int(value: Tensor | Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


class PrismDiffusionPrefixDataset(torch.utils.data.Dataset):
    """Attach PRISM prefix sequences without altering native diffusion windows."""

    def __init__(
        self,
        base_dataset,
        *,
        prefix_train_max_steps: int,
        prefix_frame_stride: int,
        prefix_pad_value: float,
        use_path_signature: bool,
        use_delta_signature: bool,
        prefix_image_cache_reader=None,
    ) -> None:
        super().__init__()
        if prefix_train_max_steps <= 0:
            raise ValueError(
                "`prefix_train_max_steps` must be positive when prefix-sequence mode "
                f"is enabled, got {prefix_train_max_steps}."
            )
        if prefix_frame_stride <= 0:
            raise ValueError(
                f"`prefix_frame_stride` must be positive, got {prefix_frame_stride}."
            )

        self.base_dataset = base_dataset
        self.prefix_train_max_steps = int(prefix_train_max_steps)
        self.prefix_frame_stride = int(prefix_frame_stride)
        self.prefix_pad_value = float(prefix_pad_value)
        self.use_path_signature = bool(use_path_signature)
        self.use_delta_signature = bool(use_delta_signature)
        self.prefix_image_cache_reader = prefix_image_cache_reader
        self.camera_keys = tuple(
            key
            for key in base_dataset.meta.camera_keys
            if is_supported_prefix_camera_key(key)
        )
        if not self.camera_keys:
            raise ValueError(
                "PRISM diffusion prefix mode requires at least one regular observation "
                f"camera key. Got dataset camera_keys={tuple(base_dataset.meta.camera_keys)}."
            )

        self.meta = base_dataset.meta
        self.meta.stats = clone_prefix_stats(
            base_stats=self.meta.stats,
            camera_keys=self.camera_keys,
            use_path_signature=self.use_path_signature,
            use_delta_signature=self.use_delta_signature,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str):
        return getattr(self.base_dataset, name)

    def _camera_uses_prefix_image_cache(self, camera_key: str) -> bool:
        return bool(
            self.prefix_image_cache_reader is not None
            and self.prefix_image_cache_reader.has_key(camera_key)
        )

    def _build_prefix_indices(self, *, abs_idx: int, ep_idx: int) -> list[int]:
        episode_meta = self.base_dataset.meta.episodes[ep_idx]
        ep_start = int(episode_meta["dataset_from_index"])
        sequence_length = abs_idx - ep_start + 1
        if sequence_length <= 0:
            raise ValueError(
                "Current index must lie inside the episode span. "
                f"Got abs_idx={abs_idx}, ep_start={ep_start}, ep_idx={ep_idx}."
            )

        positions = select_prefix_positions(
            sequence_length=sequence_length,
            prefix_train_max_steps=self.prefix_train_max_steps,
            prefix_frame_stride=self.prefix_frame_stride,
        )
        abs_indices = [ep_start + position for position in positions]
        if abs_indices[-1] != abs_idx:
            raise AssertionError(
                "PRISM diffusion prefix construction bug: the last valid prefix "
                "element must match the current decision step. "
                f"Got last={abs_indices[-1]}, current={abs_idx}."
            )
        return abs_indices

    def _query_prefix_non_video_features(
        self,
        *,
        abs_indices: list[int],
    ) -> dict[str, Tensor]:
        query_indices: dict[str, list[int]] = {"observation.state": abs_indices}
        if self.use_path_signature:
            query_indices[PATH_SIGNATURE_KEY] = abs_indices
        if self.use_delta_signature:
            query_indices[DELTA_SIGNATURE_KEY] = abs_indices

        for camera_key in self.camera_keys:
            if camera_key in self.base_dataset.meta.video_keys:
                continue
            if self._camera_uses_prefix_image_cache(camera_key):
                continue
            query_indices[camera_key] = abs_indices

        return self.base_dataset._query_hf_dataset(query_indices)

    def _query_prefix_videos(
        self,
        *,
        abs_indices: list[int],
        ep_idx: int,
    ) -> dict[str, Tensor]:
        video_query_indices = {
            camera_key: abs_indices
            for camera_key in self.camera_keys
            if camera_key in self.base_dataset.meta.video_keys
            and not self._camera_uses_prefix_image_cache(camera_key)
        }
        if not video_query_indices:
            return {}

        query_timestamps = self.base_dataset._get_query_timestamps(
            current_ts=0.0,
            query_indices=video_query_indices,
        )
        return self.base_dataset._query_videos(query_timestamps, ep_idx)

    def _apply_image_transforms(self, image_sequence: Tensor) -> Tensor:
        if self.base_dataset.image_transforms is None:
            return image_sequence
        return torch.stack(
            [self.base_dataset.image_transforms(frame) for frame in image_sequence],
            dim=0,
        )

    def _build_prefix_tensors(
        self,
        *,
        abs_indices: list[int],
        ep_idx: int,
    ) -> dict[str, Tensor]:
        hf_result = self._query_prefix_non_video_features(abs_indices=abs_indices)
        video_result = self._query_prefix_videos(abs_indices=abs_indices, ep_idx=ep_idx)

        prefix: dict[str, Tensor] = {}

        state_tensor = _ensure_prefix_vector_sequence(
            "observation.state",
            hf_result["observation.state"],
        )
        prefix[PREFIX_STATE_KEY] = pad_prefix_tensor(
            state_tensor,
            target_length=self.prefix_train_max_steps,
            pad_value=self.prefix_pad_value,
        )

        if self.use_path_signature:
            path_signature_tensor = _ensure_prefix_vector_sequence(
                PATH_SIGNATURE_KEY,
                hf_result[PATH_SIGNATURE_KEY],
            )
            prefix[PREFIX_PATH_SIGNATURE_KEY] = pad_prefix_tensor(
                path_signature_tensor,
                target_length=self.prefix_train_max_steps,
                pad_value=self.prefix_pad_value,
            )

        if self.use_delta_signature:
            delta_signature_tensor = _ensure_prefix_vector_sequence(
                DELTA_SIGNATURE_KEY,
                hf_result[DELTA_SIGNATURE_KEY],
            ).clone()
            # The first valid prefix element has no selected predecessor, so keep
            # its delta-signature convention deterministic.
            delta_signature_tensor[0] = 0
            prefix[PREFIX_DELTA_SIGNATURE_KEY] = pad_prefix_tensor(
                delta_signature_tensor,
                target_length=self.prefix_train_max_steps,
                pad_value=self.prefix_pad_value,
            )

        for camera_key in self.camera_keys:
            if self._camera_uses_prefix_image_cache(camera_key):
                image_tensor = self.prefix_image_cache_reader.get_many(
                    camera_key,
                    abs_indices,
                )
            elif camera_key in self.base_dataset.meta.video_keys:
                image_tensor = video_result[camera_key]
            else:
                image_tensor = hf_result[camera_key]

            image_sequence = _ensure_prefix_image_sequence(camera_key, image_tensor)
            image_sequence = self._apply_image_transforms(image_sequence)
            prefix[prefix_image_key_from_camera_key(camera_key)] = pad_prefix_tensor(
                image_sequence,
                target_length=self.prefix_train_max_steps,
                pad_value=0.0,
            )

        prefix[PREFIX_MASK_KEY] = build_prefix_mask(
            valid_length=len(abs_indices),
            target_length=self.prefix_train_max_steps,
        )
        return prefix

    def __getitem__(self, idx: int) -> dict[str, Tensor | Any]:
        item = dict(self.base_dataset[idx])
        if "episode_index" not in item or "index" not in item:
            raise KeyError(
                "Base diffusion dataset item must include `episode_index` and `index` "
                "so PRISM prefix sequences can be reconstructed."
            )

        ep_idx = _to_python_int(item["episode_index"])
        abs_idx = _to_python_int(item["index"])
        abs_indices = self._build_prefix_indices(abs_idx=abs_idx, ep_idx=ep_idx)
        item.update(self._build_prefix_tensors(abs_indices=abs_indices, ep_idx=ep_idx))
        return item
