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


def _normalize_mode_name(mode: Any) -> str:
    value = getattr(mode, "value", mode)
    return str(value).upper()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_numpy_stats(stats: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value, dtype=np.float32)
        for key, value in stats.items()
        if key in {"mean", "std", "min", "max", "q01", "q99", "q10", "q90"}
    }


def _normalize_values(
    values: np.ndarray,
    *,
    stats: dict[str, np.ndarray],
    mode_name: str,
    eps: float,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if mode_name == "IDENTITY":
        return array
    if mode_name == "MEAN_STD":
        return (array - stats["mean"]) / (stats["std"] + float(eps))
    if mode_name == "MIN_MAX":
        return ((array - stats["min"]) / (stats["max"] - stats["min"] + float(eps))) * 2.0 - 1.0
    if mode_name == "QUANTILES":
        return ((array - stats["q01"]) / (stats["q99"] - stats["q01"] + float(eps))) * 2.0 - 1.0
    if mode_name == "QUANTILE10":
        return ((array - stats["q10"]) / (stats["q90"] - stats["q10"] + float(eps))) * 2.0 - 1.0
    raise ValueError(f"Unsupported signature normalization mode: {mode_name}.")


def _find_training_split_artifact(policy_dir: Path) -> Path | None:
    resolved = policy_dir.resolve(strict=False)
    for candidate in (resolved, *resolved.parents):
        split_path = candidate / "dataset_split.json"
        if split_path.is_file():
            return split_path
    return None


@dataclass(frozen=True)
class SignatureNormalizationState:
    feature_stats: dict[str, dict[str, np.ndarray]]
    mode_name: str
    eps: float
    dataset_root: Path

    def normalize(self, feature_key: str, values: np.ndarray) -> np.ndarray:
        stats = self.feature_stats.get(feature_key)
        if stats is None:
            return np.asarray(values, dtype=np.float32)
        return _normalize_values(
            values,
            stats=stats,
            mode_name=self.mode_name,
            eps=self.eps,
        )

    @property
    def summary(self) -> str:
        enabled_keys = ",".join(sorted(self.feature_stats))
        return (
            f"enabled(keys={enabled_keys}, mode={self.mode_name}, "
            f"dataset_root={self.dataset_root})"
        )


def _maybe_load_signature_normalization_state(
    *,
    loaded_policy_cfg: Any | None,
    policy_dir: Path | None,
) -> SignatureNormalizationState | None:
    if loaded_policy_cfg is None or policy_dir is None:
        return None

    pre_normalized_keys = {
        str(key) for key in getattr(loaded_policy_cfg, "pre_normalized_observation_keys", ())
    }
    target_keys = {
        key
        for key in (PATH_SIGNATURE_KEY, DELTA_SIGNATURE_KEY)
        if key in pre_normalized_keys
    }
    if not target_keys:
        return None

    split_artifact = _find_training_split_artifact(policy_dir)
    if split_artifact is None:
        return None

    split_payload = _load_json(split_artifact)
    dataset_root_raw = split_payload.get("dataset_root")
    if not dataset_root_raw:
        return None

    dataset_root = Path(dataset_root_raw).expanduser().resolve(strict=False)
    if not dataset_root.exists():
        dataset_repo_id = split_payload.get("dataset_repo_id")
        if dataset_repo_id:
            repo_dataset_root = resolve_code_root() / "data" / Path(str(dataset_repo_id))
            if repo_dataset_root.exists():
                dataset_root = repo_dataset_root.resolve(strict=False)
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.is_file():
        return None

    stats_payload = _load_json(stats_path)
    feature_stats = {
        key: _load_numpy_stats(stats_payload[key])
        for key in target_keys
        if key in stats_payload and isinstance(stats_payload[key], dict)
    }
    if not feature_stats:
        return None

    normalization_mapping = getattr(loaded_policy_cfg, "normalization_mapping", {}) or {}
    mode_name = _normalize_mode_name(normalization_mapping.get("STATE", "MEAN_STD"))
    return SignatureNormalizationState(
        feature_stats=feature_stats,
        mode_name=mode_name,
        eps=1e-8,
        dataset_root=dataset_root,
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

        maxlen = self._history_length if self._history_length and self._history_length > 0 else None
        self._history: deque[np.ndarray] = deque(maxlen=maxlen)
        self._previous_signature_raw: np.ndarray | None = None
        self._backend = (
            resolve_signature_backend(policy_config.signature_backend)
            if self.enabled
            else "disabled"
        )
        self._normalization_state = _maybe_load_signature_normalization_state(
            loaded_policy_cfg=loaded_policy_cfg,
            policy_dir=policy_dir,
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

    @property
    def normalization_summary(self) -> str:
        if self._normalization_state is None:
            return "disabled"
        return self._normalization_state.summary

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
        if self._normalization_state is not None:
            if self._use_path:
                signature = self._normalization_state.normalize(
                    PATH_SIGNATURE_KEY,
                    raw_signature,
                )
            if self._use_delta and raw_delta is not None:
                delta = self._normalization_state.normalize(
                    DELTA_SIGNATURE_KEY,
                    raw_delta,
                )
        if not self._use_path:
            signature = None
        return signature, delta
