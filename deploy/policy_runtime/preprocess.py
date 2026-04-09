from __future__ import annotations

from typing import Any

import numpy as np

from deploy.bridge.protocol import ObservationPacket
from deploy.policy_runtime.loader import PolicyBundle


def build_raw_observation(packet: ObservationPacket) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Policy runtime preprocessing requires `torch`.") from exc

    raw: dict[str, Any] = {
        "observation.state": torch.from_numpy(
            np.asarray(packet.state, dtype=np.float32).reshape(-1)
        )
    }
    for key, image in packet.images.items():
        image_array = np.asarray(image)
        if image_array.ndim != 3 or image_array.shape[-1] != 3:
            raise ValueError(
                f"Observation image `{key}` must have shape (H, W, 3), got {image_array.shape}."
            )
        raw[key] = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            / 255.0
        )
    if packet.path_signature is not None:
        raw["observation.path_signature"] = torch.from_numpy(
            np.asarray(packet.path_signature, dtype=np.float32).reshape(-1)
        )
    if packet.delta_signature is not None:
        raw["observation.delta_signature"] = torch.from_numpy(
            np.asarray(packet.delta_signature, dtype=np.float32).reshape(-1)
        )
    return raw


def _ensure_batch_tensor_dims(obs: dict[str, Any], key: str, rank: int) -> None:
    tensor = obs[key]
    if tensor.ndim == rank - 1:
        obs[key] = tensor.unsqueeze(0)
    elif tensor.ndim != rank:
        raise RuntimeError(f"`{key}` must be rank {rank - 1} or {rank}, got shape={tuple(tensor.shape)}.")


def preprocess_packet(bundle: PolicyBundle, packet: ObservationPacket) -> dict[str, Any]:
    obs = build_raw_observation(packet)
    obs = bundle.preprocessor(obs)

    _ensure_batch_tensor_dims(obs, bundle.state_key, 2)
    for image_key in bundle.image_keys:
        if image_key not in obs:
            raise KeyError(f"Expected image key `{image_key}` after preprocessing.")
        _ensure_batch_tensor_dims(obs, image_key, 4)

    if bundle.use_path_signature:
        signature_key = "observation.path_signature"
        if signature_key not in obs:
            raise KeyError(f"Expected `{signature_key}` after preprocessing.")
        _ensure_batch_tensor_dims(obs, signature_key, 2)
        obs[signature_key] = obs[signature_key].to(
            device=obs[bundle.state_key].device,
            dtype=obs[bundle.state_key].dtype,
        )
    if bundle.use_delta_signature:
        delta_key = "observation.delta_signature"
        if delta_key not in obs:
            raise KeyError(f"Expected `{delta_key}` after preprocessing.")
        _ensure_batch_tensor_dims(obs, delta_key, 2)
        obs[delta_key] = obs[delta_key].to(
            device=obs[bundle.state_key].device,
            dtype=obs[bundle.state_key].dtype,
        )

    return obs

