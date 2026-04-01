from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature

PREFIX_STATE_KEY = "observation.prefix_state"
PREFIX_PATH_SIGNATURE_KEY = "observation.prefix_path_signature"
PREFIX_DELTA_SIGNATURE_KEY = "observation.prefix_delta_signature"
PREFIX_MASK_KEY = "observation.prefix_mask"
PREFIX_IMAGES_PREFIX = "observation.prefix_images."
PATH_SIGNATURE_KEY = "observation.path_signature"
DELTA_SIGNATURE_KEY = "observation.delta_signature"


def is_prefix_image_key(key: str) -> bool:
    return key.startswith(PREFIX_IMAGES_PREFIX)


def is_supported_prefix_camera_key(key: str) -> bool:
    return key.startswith("observation.images.") or key == "observation.image"


def prefix_image_key_from_camera_key(camera_key: str) -> str:
    if camera_key.startswith("observation.images."):
        return PREFIX_IMAGES_PREFIX + camera_key.removeprefix("observation.images.")
    if camera_key == "observation.image":
        return PREFIX_IMAGES_PREFIX + "main"
    raise ValueError(
        "Unsupported observation image key for prefix-sequence features: "
        f"{camera_key!r}."
    )


def select_prefix_positions(
    *,
    sequence_length: int,
    prefix_train_max_steps: int,
    prefix_frame_stride: int,
) -> list[int]:
    if sequence_length <= 0:
        raise ValueError(
            f"`sequence_length` must be positive, got {sequence_length}."
        )
    if prefix_train_max_steps <= 0:
        raise ValueError(
            "`prefix_train_max_steps` must be positive when prefix-sequence mode is "
            f"enabled, got {prefix_train_max_steps}."
        )
    if prefix_frame_stride <= 0:
        raise ValueError(
            f"`prefix_frame_stride` must be positive, got {prefix_frame_stride}."
        )

    positions = list(range(0, sequence_length, prefix_frame_stride))
    last_position = sequence_length - 1
    if positions[-1] != last_position:
        positions.append(last_position)
    if len(positions) > prefix_train_max_steps:
        # Keep the first prefix element so long-horizon training still exposes the
        # model to the episode's initial cue, and spend the remaining budget on the
        # most recent prefix tail ending at the current step.
        tail_budget = prefix_train_max_steps - 1
        if tail_budget <= 0:
            positions = [last_position]
        else:
            tail_positions = positions[-tail_budget:]
            if tail_positions[0] == positions[0]:
                positions = tail_positions
            else:
                positions = [positions[0], *tail_positions]
    return positions


def pad_prefix_tensor(
    tensor: Tensor,
    *,
    target_length: int,
    pad_value: float,
) -> Tensor:
    if tensor.ndim < 1:
        raise ValueError(
            "`tensor` must have at least one dimension for prefix padding. "
            f"Got shape={tuple(tensor.shape)}."
        )
    current_length = int(tensor.shape[0])
    if current_length <= 0:
        raise ValueError("Prefix tensor must contain at least one valid step before padding.")
    if current_length > target_length:
        raise ValueError(
            "Prefix tensor length exceeds the configured budget. "
            f"Got {current_length} > target_length={target_length}."
        )
    if current_length == target_length:
        return tensor

    pad_shape = (target_length - current_length, *tensor.shape[1:])
    pad_tensor = torch.full(
        pad_shape,
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad_tensor], dim=0)


def build_prefix_mask(
    *,
    valid_length: int,
    target_length: int,
    device: torch.device | None = None,
) -> Tensor:
    if valid_length <= 0:
        raise ValueError(f"`valid_length` must be positive, got {valid_length}.")
    if valid_length > target_length:
        raise ValueError(
            f"`valid_length` must be <= `target_length`, got {valid_length} > {target_length}."
        )
    mask = torch.zeros((target_length,), dtype=torch.bool, device=device)
    mask[:valid_length] = True
    return mask


def build_padded_prefix_from_history(
    history: Sequence[Tensor | Any],
    *,
    prefix_train_max_steps: int,
    prefix_frame_stride: int,
    pad_value: float,
) -> tuple[Tensor, Tensor]:
    if len(history) == 0:
        raise ValueError("Prefix history is empty.")

    positions = select_prefix_positions(
        sequence_length=len(history),
        prefix_train_max_steps=prefix_train_max_steps,
        prefix_frame_stride=prefix_frame_stride,
    )
    selected_items = [torch.as_tensor(history[position]) for position in positions]
    sequence_tensor = torch.stack(selected_items, dim=0)
    padded = pad_prefix_tensor(
        sequence_tensor,
        target_length=prefix_train_max_steps,
        pad_value=pad_value,
    )
    mask = build_prefix_mask(
        valid_length=len(positions),
        target_length=prefix_train_max_steps,
        device=padded.device,
    )
    return padded, mask


def clone_prefix_stats(
    *,
    base_stats: dict[str, dict[str, Any]],
    camera_keys: Sequence[str],
    use_path_signature: bool,
    use_delta_signature: bool,
) -> dict[str, dict[str, Any]]:
    stats = deepcopy(base_stats)
    if "observation.state" not in stats:
        raise KeyError(
            "`observation.state` stats are required to derive prefix-state stats."
        )
    stats[PREFIX_STATE_KEY] = deepcopy(stats["observation.state"])

    if use_path_signature:
        if PATH_SIGNATURE_KEY not in stats:
            raise KeyError(
                f"`{PATH_SIGNATURE_KEY}` stats are required when prefix-sequence mode "
                "is enabled with path signatures."
            )
        stats[PREFIX_PATH_SIGNATURE_KEY] = deepcopy(stats[PATH_SIGNATURE_KEY])
    if use_delta_signature:
        if DELTA_SIGNATURE_KEY not in stats:
            raise KeyError(
                f"`{DELTA_SIGNATURE_KEY}` stats are required when prefix-sequence mode "
                "is enabled with delta signatures."
            )
        stats[PREFIX_DELTA_SIGNATURE_KEY] = deepcopy(stats[DELTA_SIGNATURE_KEY])

    for camera_key in camera_keys:
        if camera_key not in stats:
            raise KeyError(
                f"Camera stats for `{camera_key}` are missing; cannot derive prefix image stats."
            )
        stats[prefix_image_key_from_camera_key(camera_key)] = deepcopy(stats[camera_key])
    return stats


def build_prefix_sequence_input_features(
    *,
    base_input_features: dict[str, PolicyFeature],
    prefix_train_max_steps: int,
    use_path_signature: bool,
    use_delta_signature: bool,
) -> dict[str, PolicyFeature]:
    if "observation.state" not in base_input_features:
        raise KeyError(
            "`observation.state` must exist in input_features before enabling prefix-sequence mode."
        )

    updated = dict(base_input_features)
    state_feature = base_input_features["observation.state"]
    updated[PREFIX_STATE_KEY] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(prefix_train_max_steps, *state_feature.shape),
    )
    updated[PREFIX_MASK_KEY] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(prefix_train_max_steps,),
    )

    if use_path_signature:
        signature_feature = base_input_features.get(PATH_SIGNATURE_KEY)
        if signature_feature is None:
            raise KeyError(
                f"`{PATH_SIGNATURE_KEY}` input feature is required when "
                "`use_prefix_sequence_training=True`."
            )
        updated[PREFIX_PATH_SIGNATURE_KEY] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(prefix_train_max_steps, *signature_feature.shape),
        )
    if use_delta_signature:
        delta_signature_feature = base_input_features.get(DELTA_SIGNATURE_KEY)
        if delta_signature_feature is None:
            raise KeyError(
                f"`{DELTA_SIGNATURE_KEY}` input feature is required when "
                "`use_prefix_sequence_training=True` and delta signatures are enabled."
            )
        updated[PREFIX_DELTA_SIGNATURE_KEY] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(prefix_train_max_steps, *delta_signature_feature.shape),
        )

    for key, feature in list(base_input_features.items()):
        if feature.type is not FeatureType.VISUAL:
            continue
        if is_prefix_image_key(key) or key == "observation.anchor_image":
            continue
        updated[prefix_image_key_from_camera_key(key)] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(prefix_train_max_steps, *feature.shape),
        )

    return updated


def _ensure_prefix_image_sequence(camera_key: str, tensor: Tensor) -> Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(
            f"Prefix images for `{camera_key}` must have shape (T, C, H, W). "
            f"Got shape={tuple(tensor.shape)}."
        )
    return tensor


def _ensure_prefix_vector_sequence(feature_key: str, tensor: Tensor) -> Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Prefix feature `{feature_key}` must have shape (T, D). "
            f"Got shape={tuple(tensor.shape)}."
        )
    return tensor


class PrefixSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        *,
        prefix_train_max_steps: int,
        prefix_frame_stride: int,
        prefix_pad_value: float,
        use_path_signature: bool,
        use_delta_signature: bool,
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
        self.camera_keys = tuple(
            key
            for key in base_dataset.meta.camera_keys
            if is_supported_prefix_camera_key(key)
        )
        if not self.camera_keys:
            raise ValueError(
                "Prefix-sequence mode requires at least one regular observation camera key. "
                f"Got dataset camera_keys={tuple(base_dataset.meta.camera_keys)}."
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

    def _load_current_item_without_videos(self, idx: int) -> dict[str, Tensor | Any]:
        self.base_dataset._ensure_hf_dataset_loaded()
        item = dict(self.base_dataset.hf_dataset[idx])

        if self.base_dataset.image_transforms is not None:
            for camera_key in self.camera_keys:
                if (
                    camera_key in item
                    and camera_key not in self.base_dataset.meta.video_keys
                ):
                    item[camera_key] = self.base_dataset.image_transforms(item[camera_key])

        task_idx = item["task_index"].item()
        item["task"] = self.base_dataset.meta.tasks.iloc[task_idx].name

        if (
            "subtask_index" in self.base_dataset.features
            and self.base_dataset.meta.subtasks is not None
        ):
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self.base_dataset.meta.subtasks.iloc[subtask_idx].name

        return item

    def _prefetch_video_sequence(
        self,
        *,
        abs_indices: list[int],
        current_ts: float,
        ep_idx: int,
    ) -> dict[str, Tensor]:
        video_query_indices = {
            camera_key: abs_indices
            for camera_key in self.camera_keys
            if camera_key in self.base_dataset.meta.video_keys
        }
        if not video_query_indices:
            return {}

        query_timestamps = self.base_dataset._get_query_timestamps(
            current_ts=current_ts,
            query_indices=video_query_indices,
        )
        return self.base_dataset._query_videos(query_timestamps, ep_idx)

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
                "Prefix sequence construction bug: the last valid prefix element must "
                f"match the current step. Got last={abs_indices[-1]}, current={abs_idx}."
            )
        return abs_indices

    def _query_prefix_tensors(
        self,
        *,
        current_item: dict[str, Tensor | Any],
        abs_indices: list[int],
        ep_idx: int,
        prefetched_video_result: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        history_abs_indices = abs_indices[:-1]
        query_indices: dict[str, list[int]] = {}
        if history_abs_indices:
            query_indices["observation.state"] = history_abs_indices
        if self.use_path_signature:
            if history_abs_indices:
                query_indices[PATH_SIGNATURE_KEY] = history_abs_indices
        if self.use_delta_signature:
            if history_abs_indices:
                query_indices[DELTA_SIGNATURE_KEY] = history_abs_indices
        if history_abs_indices:
            for camera_key in self.camera_keys:
                query_indices[camera_key] = history_abs_indices

        hf_result = (
            self.base_dataset._query_hf_dataset(query_indices)
            if query_indices
            else {}
        )
        prefix: dict[str, Tensor] = {}

        state_history = hf_result.get("observation.state")
        current_state = _ensure_prefix_vector_sequence(
            "observation.state",
            torch.as_tensor(current_item["observation.state"]),
        )
        if state_history is not None:
            state_tensor = torch.cat(
                [
                    _ensure_prefix_vector_sequence("observation.state", state_history),
                    current_state,
                ],
                dim=0,
            )
        else:
            state_tensor = current_state
        prefix[PREFIX_STATE_KEY] = pad_prefix_tensor(
            state_tensor,
            target_length=self.prefix_train_max_steps,
            pad_value=self.prefix_pad_value,
        )

        if self.use_path_signature:
            signature_history = hf_result.get(PATH_SIGNATURE_KEY)
            current_signature = _ensure_prefix_vector_sequence(
                PATH_SIGNATURE_KEY,
                torch.as_tensor(current_item[PATH_SIGNATURE_KEY]),
            )
            if signature_history is not None:
                signature_tensor = torch.cat(
                    [
                        _ensure_prefix_vector_sequence(PATH_SIGNATURE_KEY, signature_history),
                        current_signature,
                    ],
                    dim=0,
                )
            else:
                signature_tensor = current_signature
            prefix[PREFIX_PATH_SIGNATURE_KEY] = pad_prefix_tensor(
                signature_tensor,
                target_length=self.prefix_train_max_steps,
                pad_value=self.prefix_pad_value,
            )
        if self.use_delta_signature:
            delta_signature_history = hf_result.get(DELTA_SIGNATURE_KEY)
            current_delta_signature = _ensure_prefix_vector_sequence(
                DELTA_SIGNATURE_KEY,
                torch.as_tensor(current_item[DELTA_SIGNATURE_KEY]),
            )
            if delta_signature_history is not None:
                delta_signature_tensor = torch.cat(
                    [
                        _ensure_prefix_vector_sequence(
                            DELTA_SIGNATURE_KEY, delta_signature_history
                        ),
                        current_delta_signature,
                    ],
                    dim=0,
                )
            else:
                delta_signature_tensor = current_delta_signature
            prefix[PREFIX_DELTA_SIGNATURE_KEY] = pad_prefix_tensor(
                delta_signature_tensor,
                target_length=self.prefix_train_max_steps,
                pad_value=self.prefix_pad_value,
            )

        if prefetched_video_result is not None:
            video_result = prefetched_video_result
        elif self.base_dataset.meta.video_keys and history_abs_indices:
            query_timestamps = self.base_dataset._get_query_timestamps(
                current_ts=float(current_item["timestamp"].item()),
                query_indices=query_indices,
            )
            video_result = self.base_dataset._query_videos(query_timestamps, ep_idx)
        else:
            video_result = {}

        for camera_key in self.camera_keys:
            if camera_key in self.base_dataset.meta.video_keys:
                if prefetched_video_result is not None:
                    full_image_tensor = video_result.get(camera_key)
                    history_image_tensor = (
                        None
                        if full_image_tensor is None or int(full_image_tensor.shape[0]) <= 1
                        else full_image_tensor[:-1]
                    )
                else:
                    history_image_tensor = video_result.get(camera_key)
            else:
                history_image_tensor = hf_result.get(camera_key)

            history_sequence = None
            if history_image_tensor is not None:
                history_sequence = _ensure_prefix_image_sequence(
                    camera_key, history_image_tensor
                )
                if self.base_dataset.image_transforms is not None:
                    history_sequence = torch.stack(
                        [self.base_dataset.image_transforms(frame) for frame in history_sequence],
                        dim=0,
                    )

            current_image_tensor = _ensure_prefix_image_sequence(
                camera_key,
                torch.as_tensor(current_item[camera_key]),
            )
            image_tensor = (
                current_image_tensor
                if history_sequence is None
                else torch.cat([history_sequence, current_image_tensor], dim=0)
            )
            prefix[prefix_image_key_from_camera_key(camera_key)] = pad_prefix_tensor(
                image_tensor,
                target_length=self.prefix_train_max_steps,
                pad_value=0.0,
            )

        prefix[PREFIX_MASK_KEY] = build_prefix_mask(
            valid_length=len(abs_indices),
            target_length=self.prefix_train_max_steps,
        )
        return prefix

    def __getitem__(self, idx: int) -> dict[str, Tensor | Any]:
        can_prefetch_video_once = (
            len(self.base_dataset.meta.video_keys) > 0
            and getattr(self.base_dataset, "delta_indices", None) is None
        )
        item = (
            self._load_current_item_without_videos(idx)
            if can_prefetch_video_once
            else dict(self.base_dataset[idx])
        )
        if "episode_index" not in item or "index" not in item:
            raise KeyError(
                "Base dataset item must include `episode_index` and `index` so prefix "
                "sequences can be reconstructed."
            )
        ep_idx = int(item["episode_index"].item())
        abs_idx = int(item["index"].item())
        abs_indices = self._build_prefix_indices(abs_idx=abs_idx, ep_idx=ep_idx)
        prefetched_video_result = None
        if can_prefetch_video_once:
            prefetched_video_result = self._prefetch_video_sequence(
                abs_indices=abs_indices,
                current_ts=float(item["timestamp"].item()),
                ep_idx=ep_idx,
            )
            for camera_key in self.camera_keys:
                if camera_key not in self.base_dataset.meta.video_keys:
                    continue
                current_frame = prefetched_video_result[camera_key][-1]
                if self.base_dataset.image_transforms is not None:
                    current_frame = self.base_dataset.image_transforms(current_frame)
                item[camera_key] = current_frame
        item.update(
            self._query_prefix_tensors(
                current_item=item,
                abs_indices=abs_indices,
                ep_idx=ep_idx,
                prefetched_video_result=prefetched_video_result,
            )
        )
        return item
