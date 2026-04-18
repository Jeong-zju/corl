from __future__ import annotations

from collections import deque
from typing import Any

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
    def __init__(
        self,
        policy_config: PolicyConfig,
        *,
        loaded_policy_cfg: Any | None = None,
    ) -> None:
        self._use_path = bool(
            getattr(loaded_policy_cfg, "use_path_signature", policy_config.use_path_signature)
        )
        self._use_delta = bool(
            getattr(loaded_policy_cfg, "use_delta_signature", policy_config.use_delta_signature)
        )

        raw_history_length = getattr(loaded_policy_cfg, "history_length", None)
        if raw_history_length in {None, 0}:
            self._history_length = None
        else:
            self._history_length = int(raw_history_length)

        raw_signature_depth = getattr(
            loaded_policy_cfg,
            "signature_depth",
            policy_config.signature_depth,
        )
        self._signature_depth = int(raw_signature_depth)

        raw_signature_dim = getattr(
            loaded_policy_cfg,
            "signature_dim",
            policy_config.signature_dim,
        )
        self._signature_dim = (
            None if raw_signature_dim in {None, 0} else int(raw_signature_dim)
        )

        maxlen = self._history_length if self._history_length and self._history_length > 0 else None
        self._history: deque[np.ndarray] = deque(maxlen=maxlen)
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

    @property
    def history_length(self) -> int | None:
        return self._history_length

    def reset(self) -> None:
        self._history.clear()
        self._previous_signature = None

    def update(self, state: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.enabled:
            return None, None

        state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
        self._history.append(state_vec.astype(np.float32, copy=True))
        window = np.stack(list(self._history), axis=0)
        if self._history_length is not None and window.shape[0] < self._history_length:
            pad_len = self._history_length - window.shape[0]
            pad = np.repeat(window[:1], pad_len, axis=0)
            window = np.concatenate([pad, window], axis=0)

        if self._backend == "signatory":
            signature = compute_signatory_signature_np(
                window,
                self._signature_depth,
            )
        else:
            signature = compute_simple_signature_np(
                window,
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
