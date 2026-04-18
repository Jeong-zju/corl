from __future__ import annotations

from collections import deque
import json
from dataclasses import dataclass
from pathlib import Path
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
    resolve_code_root,
)

PATH_SIGNATURE_KEY = "observation.path_signature"
DELTA_SIGNATURE_KEY = "observation.delta_signature"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_training_split_artifact(policy_dir: Path) -> Path | None:
    resolved = policy_dir.resolve(strict=False)
    for candidate in (resolved, *resolved.parents):
        split_path = candidate / "dataset_split.json"
        if split_path.is_file():
            return split_path
    return None


def _resolve_dataset_root(split_payload: dict[str, Any]) -> Path | None:
    dataset_root_raw = split_payload.get("dataset_root")
    if not dataset_root_raw:
        return None

    dataset_root = Path(dataset_root_raw).expanduser().resolve(strict=False)
    if dataset_root.exists():
        return dataset_root

    dataset_repo_id = split_payload.get("dataset_repo_id")
    if dataset_repo_id:
        repo_dataset_root = resolve_code_root() / "data" / Path(str(dataset_repo_id))
        if repo_dataset_root.exists():
            return repo_dataset_root.resolve(strict=False)
    return dataset_root


@dataclass(frozen=True)
class SignatureDatasetSpec:
    dataset_root: Path
    backend: str | None
    window: str | None

    @property
    def summary(self) -> str:
        return (
            f"dataset_backend={self.backend or 'unknown'}, "
            f"dataset_window={self.window or 'unknown'}, "
            f"dataset_root={self.dataset_root}"
        )


def _maybe_load_signature_dataset_spec(
    *,
    loaded_policy_cfg: Any | None,
    policy_dir: Path | None,
) -> SignatureDatasetSpec | None:
    if loaded_policy_cfg is None or policy_dir is None:
        return None
    if not bool(getattr(loaded_policy_cfg, "use_path_signature", False)):
        return None

    split_artifact = _find_training_split_artifact(policy_dir)
    if split_artifact is None:
        return None

    split_payload = _load_json(split_artifact)
    dataset_root = _resolve_dataset_root(split_payload)
    if dataset_root is None:
        return None

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return None

    info_payload = _load_json(info_path)
    path_signature_info = info_payload.get("path_signature")
    if not isinstance(path_signature_info, dict):
        return None

    return SignatureDatasetSpec(
        dataset_root=dataset_root,
        backend=(
            None
            if path_signature_info.get("backend") in {None, "", "null"}
            else str(path_signature_info.get("backend"))
        ),
        window=(
            None
            if path_signature_info.get("window") in {None, "", "null"}
            else str(path_signature_info.get("window"))
        ),
    )


class OnlineSignatureRuntime:
    def __init__(
        self,
        policy_config: PolicyConfig,
        *,
        loaded_policy_cfg: Any | None = None,
        policy_dir: Path | None = None,
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

        self._previous_signature_raw: np.ndarray | None = None
        self._backend = (
            resolve_signature_backend(policy_config.signature_backend)
            if self.enabled
            else "disabled"
        )
        self._dataset_spec = _maybe_load_signature_dataset_spec(
            loaded_policy_cfg=loaded_policy_cfg,
            policy_dir=policy_dir,
        )
        if self._dataset_spec is not None:
            dataset_window = (self._dataset_spec.window or "").strip().lower()
            if dataset_window == "full_prefix":
                self._history_length = None

            dataset_backend = (self._dataset_spec.backend or "").strip().lower()
            if dataset_backend in {"simple", "signatory"} and dataset_backend != self._backend:
                raise RuntimeError(
                    "Online signature backend mismatch: "
                    f"deploy resolved backend={self._backend}, "
                    f"but dataset metadata requires backend={dataset_backend}. "
                    "Use the same signature backend as dataset generation."
                )

        maxlen = self._history_length if self._history_length and self._history_length > 0 else None
        self._history = deque(maxlen=maxlen)

    @property
    def enabled(self) -> bool:
        return self._use_path or self._use_delta

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def history_length(self) -> int | None:
        return self._history_length

    @property
    def normalization_summary(self) -> str:
        return "disabled"

    @property
    def dataset_summary(self) -> str:
        if self._dataset_spec is None:
            return "unknown"
        return self._dataset_spec.summary

    def reset(self) -> None:
        self._history.clear()
        self._previous_signature_raw = None

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
            raw_signature = compute_signatory_signature_np(
                window,
                self._signature_depth,
            )
        else:
            raw_signature = compute_simple_signature_np(
                window,
                self._signature_depth,
            )

        if self._signature_dim is not None and raw_signature.shape[0] != self._signature_dim:
            raise RuntimeError(
                "Signature dimension mismatch: "
                f"computed={raw_signature.shape[0]}, expected={self._signature_dim}, "
                f"backend={self._backend}."
            )

        raw_delta = None
        if self._use_delta:
            raw_delta = compute_delta_signature_step_np(
                raw_signature,
                self._previous_signature_raw,
            )
        self._previous_signature_raw = raw_signature.astype(np.float32, copy=True)

        signature = raw_signature
        delta = raw_delta
        if not self._use_path:
            signature = None
        return signature, delta
