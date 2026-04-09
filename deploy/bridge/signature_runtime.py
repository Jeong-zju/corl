from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def compute_delta_signature_step_np(
    current_signature: np.ndarray,
    previous_signature: np.ndarray | None,
) -> np.ndarray:
    current = np.asarray(current_signature, dtype=np.float32)
    if previous_signature is None:
        return np.zeros_like(current, dtype=np.float32)
    previous = np.asarray(previous_signature, dtype=np.float32)
    if previous.shape != current.shape:
        raise ValueError(
            "Previous signature shape mismatch during online delta computation: "
            f"previous={previous.shape}, current={current.shape}."
        )
    return (current - previous).astype(np.float32, copy=False)


def compute_simple_signature_np(window: np.ndarray, sig_depth: int) -> np.ndarray:
    if window.ndim != 2:
        raise ValueError(f"Window must be 2D, got shape={window.shape}.")
    if sig_depth <= 0:
        raise ValueError(f"sig_depth must be > 0, got {sig_depth}.")

    deltas = np.diff(window, axis=0, prepend=window[:1]).astype(np.float32)
    moments = [
        np.sum(np.power(deltas, order, dtype=np.float32), axis=0)
        for order in range(1, sig_depth + 1)
    ]
    return np.concatenate(moments, axis=0).astype(np.float32)


def compute_signatory_signature_np(window: np.ndarray, sig_depth: int) -> np.ndarray:
    if window.ndim != 2:
        raise ValueError(f"Window must be 2D, got shape={window.shape}.")
    if window.shape[0] < 1:
        raise ValueError("Window must contain at least one point.")

    try:
        import torch
        import signatory
    except ImportError as exc:
        raise RuntimeError(
            "`signatory` backend requested but `torch` and `signatory` are not available."
        ) from exc

    effective_window = window
    if effective_window.shape[0] < 2:
        effective_window = np.repeat(effective_window[:1], 2, axis=0)

    path = torch.from_numpy(effective_window.astype(np.float32, copy=False)).unsqueeze(0)
    with torch.no_grad():
        signature = signatory.signature(path, depth=sig_depth)
    return signature.squeeze(0).cpu().numpy().astype(np.float32)


@dataclass(slots=True)
class SignatureStep:
    path_signature: np.ndarray
    delta_signature: np.ndarray
    step_index: int


class StreamingSignatureTracker:
    def __init__(
        self,
        *,
        enabled: bool,
        depth: int,
        backend: str = "simple",
        history_limit: int | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.depth = int(depth)
        self.backend = self._resolve_backend(backend)
        self.history_limit = None if history_limit is None else int(history_limit)
        self._states: list[np.ndarray] = []
        self._previous_signature: np.ndarray | None = None

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        normalized = str(backend).strip().lower()
        if normalized == "auto":
            try:
                import signatory  # noqa: F401
                import torch  # noqa: F401
            except ImportError:
                return "simple"
            return "signatory"
        if normalized not in {"simple", "signatory"}:
            raise ValueError(f"Unsupported signature backend: {backend!r}.")
        return normalized

    def reset(self) -> None:
        self._states.clear()
        self._previous_signature = None

    def update(self, state: np.ndarray) -> SignatureStep | None:
        if not self.enabled:
            return None

        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        self._states.append(state_array.copy())
        if self.history_limit is not None and self.history_limit > 0:
            excess = len(self._states) - self.history_limit
            if excess > 0:
                del self._states[:excess]

        window = np.stack(self._states, axis=0)
        if self.backend == "signatory":
            signature = compute_signatory_signature_np(window, self.depth)
        else:
            signature = compute_simple_signature_np(window, self.depth)
        delta = compute_delta_signature_step_np(signature, self._previous_signature)
        self._previous_signature = signature.copy()
        return SignatureStep(
            path_signature=signature,
            delta_signature=delta,
            step_index=len(self._states) - 1,
        )

