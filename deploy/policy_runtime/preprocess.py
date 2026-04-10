from __future__ import annotations

from typing import Any

import numpy as np


def select_visual_observation_keys(cfg) -> list[str]:
    visual_features = getattr(cfg, "visual_observation_features", None)
    if visual_features is None:
        visual_features = getattr(cfg, "image_features", {})
    keys = list(visual_features)
    if not keys:
        raise RuntimeError("Policy has no visual observation input features.")
    return keys


def resolve_state_key(cfg) -> str:
    input_features = getattr(cfg, "input_features", None)
    if isinstance(input_features, dict):
        if "observation.state" in input_features:
            return "observation.state"
        for key in input_features:
            if str(key).endswith(".state"):
                return str(key)
    return "observation.state"


def resolve_env_state_key(cfg) -> str | None:
    input_features = getattr(cfg, "input_features", None)
    if isinstance(input_features, dict):
        if "observation.environment_state" in input_features:
            return "observation.environment_state"
        for key in input_features:
            if str(key).endswith(".environment_state"):
                return str(key)
    return None


def resolve_action_key(cfg) -> str:
    output_features = getattr(cfg, "output_features", None)
    if isinstance(output_features, dict):
        if "action" in output_features:
            return "action"
        if len(output_features) == 1:
            return str(next(iter(output_features)))
    return "action"


def _feature_shape(feature: Any) -> tuple[int, ...]:
    if feature is None:
        return ()
    if hasattr(feature, "shape"):
        return tuple(int(value) for value in getattr(feature, "shape"))
    if isinstance(feature, dict) and "shape" in feature:
        return tuple(int(value) for value in list(feature["shape"]))
    return ()


def _zeros_for_feature(cfg, key: str) -> np.ndarray:
    input_features = getattr(cfg, "input_features", None) or {}
    feature = input_features.get(key)
    shape = _feature_shape(feature)
    if not shape:
        raise KeyError(f"Could not resolve feature shape for `{key}`.")
    return np.zeros(shape, dtype=np.float32)


def build_raw_policy_observation(observation_packet: dict[str, Any], cfg):
    import torch

    state_key = resolve_state_key(cfg)
    env_state_key = resolve_env_state_key(cfg)
    visual_keys = select_visual_observation_keys(cfg)

    state = np.asarray(observation_packet["state"], dtype=np.float32).reshape(-1)
    obs: dict[str, object] = {
        state_key: torch.from_numpy(state.astype(np.float32, copy=False)),
    }

    if env_state_key is not None and observation_packet.get("env_state") is not None:
        env_state = np.asarray(
            observation_packet["env_state"],
            dtype=np.float32,
        ).reshape(-1)
        obs[env_state_key] = torch.from_numpy(env_state.astype(np.float32, copy=False))

    images = observation_packet.get("images", {})
    for visual_key in visual_keys:
        if visual_key not in images:
            raise KeyError(f"Missing required image input `{visual_key}`.")
        image = np.asarray(images[visual_key], dtype=np.uint8)
        if image.ndim != 3:
            raise RuntimeError(
                f"Expected image `{visual_key}` to have shape (H, W, C), got {image.shape}."
            )
        obs[visual_key] = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .contiguous()
            .float()
            / 255.0
        )

    input_features = getattr(cfg, "input_features", None) or {}
    signature_map = {
        "observation.path_signature": observation_packet.get("path_signature"),
        "observation.delta_signature": observation_packet.get("delta_signature"),
    }
    for feature_key, provided_value in signature_map.items():
        if feature_key not in input_features:
            continue
        if provided_value is None:
            value = _zeros_for_feature(cfg, feature_key)
        else:
            value = np.asarray(provided_value, dtype=np.float32).reshape(-1)
        obs[feature_key] = torch.from_numpy(value.astype(np.float32, copy=False))

    return obs


def finalize_preprocessed_observation(obs: dict[str, Any], cfg) -> dict[str, Any]:
    state_key = resolve_state_key(cfg)
    state_tensor = obs[state_key]
    state_device = state_tensor.device
    state_dtype = state_tensor.dtype

    for feature_key in ("observation.path_signature", "observation.delta_signature"):
        if feature_key not in obs:
            continue
        feature = obs[feature_key]
        if feature.ndim == 1:
            feature = feature.unsqueeze(0)
        elif feature.ndim != 2:
            raise RuntimeError(
                f"`{feature_key}` must be 1D or 2D after preprocessing, got {tuple(feature.shape)}."
            )
        obs[feature_key] = feature.to(device=state_device, dtype=state_dtype)
    return obs
