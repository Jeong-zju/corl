from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from eval_policy import (
    DEFAULT_DELTA_SIGNATURE_KEY,
    DEFAULT_PATH_SIGNATURE_KEY,
    compute_online_signature_prefix,
    resolve_dataset_signature_inputs,
    tensor_to_numpy_vector,
)


def test_resolve_dataset_signature_inputs_falls_back_for_null_dataset_values() -> None:
    state = torch.tensor([1.0, 2.0], dtype=torch.float32)
    state_history: deque[np.ndarray] = deque()

    (
        path_signature_tensor,
        signature_vec,
        delta_signature_tensor,
        used_cached_path_signature,
        used_cached_delta_signature,
    ) = resolve_dataset_signature_inputs(
        item={
            DEFAULT_PATH_SIGNATURE_KEY: None,
            DEFAULT_DELTA_SIGNATURE_KEY: None,
        },
        state_value=state,
        state_history=state_history,
        sig_depth=2,
        signature_backend="simple",
        use_delta_signature=True,
        previous_signature_vec=None,
    )

    expected_signature = compute_online_signature_prefix(
        deque([tensor_to_numpy_vector(state)]),
        sig_depth=2,
        signature_backend="simple",
    )

    assert used_cached_path_signature is False
    assert used_cached_delta_signature is False
    assert delta_signature_tensor is not None
    assert len(state_history) == 1
    assert torch.allclose(
        path_signature_tensor,
        torch.from_numpy(expected_signature),
    )
    assert np.allclose(signature_vec, expected_signature)
    assert torch.allclose(
        delta_signature_tensor,
        torch.zeros_like(path_signature_tensor),
    )


def test_resolve_dataset_signature_inputs_prefers_cached_dataset_values() -> None:
    state = torch.tensor([3.0, 4.0], dtype=torch.float32)
    state_history: deque[np.ndarray] = deque()
    cached_path_signature = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    cached_delta_signature = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)

    (
        path_signature_tensor,
        signature_vec,
        delta_signature_tensor,
        used_cached_path_signature,
        used_cached_delta_signature,
    ) = resolve_dataset_signature_inputs(
        item={
            DEFAULT_PATH_SIGNATURE_KEY: cached_path_signature,
            DEFAULT_DELTA_SIGNATURE_KEY: cached_delta_signature,
        },
        state_value=state,
        state_history=state_history,
        sig_depth=2,
        signature_backend="simple",
        use_delta_signature=True,
        previous_signature_vec=None,
    )

    assert used_cached_path_signature is True
    assert used_cached_delta_signature is True
    assert delta_signature_tensor is not None
    assert len(state_history) == 1
    assert torch.allclose(path_signature_tensor, cached_path_signature)
    assert np.allclose(signature_vec, cached_path_signature.numpy())
    assert torch.allclose(delta_signature_tensor, cached_delta_signature)
