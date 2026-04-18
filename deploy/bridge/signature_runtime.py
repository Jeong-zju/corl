from __future__ import annotations

from collections import deque

import numpy as np

from common import ensure_runtime_paths
from config import PolicyConfig

ensure_runtime_paths()

from eval_helpers import (  # noqa: E402
    compute_delta_signature_step_np,
    compute_signatory_signature_np,
    compute_simple_signature_np,
    resolve_signature_backend,
)


class OnlineSignatureRuntime:
    def __init__(self, policy_config: PolicyConfig) -> None:
        self._use_path = bool(
            policy_config.type == "streaming_act" and policy_config.use_path_signature
        )
        self._use_delta = bool(
            policy_config.type == "streaming_act" and policy_config.use_delta_signature
        )
        self._signature_depth = int(policy_config.signature_depth)
        self._signature_dim = policy_config.signature_dim
        self._history: deque[np.ndarray] = deque()
        self._previous_signature: np.ndarray | None = None
        self._backend = (
            resolve_signature_backend(policy_config.signature_backend)
            if self.enabled
            else "disabled"
        )

    @property
    def enabled(self) -> bool:
        return self._use_path or self._use_delta

    @property
    def backend(self) -> str:
        return self._backend

    def reset(self) -> None:
        self._history.clear()
        self._previous_signature = None

    def update(self, state: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.enabled:
            return None, None

        state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
        self._history.append(state_vec.astype(np.float32, copy=True))
        if self._backend == "signatory":
            signature = compute_signatory_signature_np(
                np.stack(list(self._history), axis=0),
                self._signature_depth,
            )
        else:
            signature = compute_simple_signature_np(
                np.stack(list(self._history), axis=0),
                self._signature_depth,
            )

        if self._signature_dim is not None and signature.shape[0] != self._signature_dim:
            raise RuntimeError(
                "Signature dimension mismatch: "
                f"computed={signature.shape[0]}, expected={self._signature_dim}, "
                f"backend={self._backend}."
            )

        delta = None
        if self._use_delta:
            delta = compute_delta_signature_step_np(signature, self._previous_signature)
        self._previous_signature = signature.astype(np.float32, copy=True)

        if not self._use_path:
            signature = None
        return signature, delta
