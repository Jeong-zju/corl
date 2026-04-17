from __future__ import annotations

import importlib
import json
import logging
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_delta_signature_sequence_np(signatures: np.ndarray) -> np.ndarray:
    signatures_array = np.asarray(signatures, dtype=np.float32)
    if signatures_array.ndim != 2:
        raise ValueError(
            "Expected signature trajectory with shape (T, signature_dim). "
            f"Got {signatures_array.shape}."
        )
    if signatures_array.shape[0] == 0:
        raise ValueError("Signature trajectory must contain at least one step.")

    delta = np.zeros_like(signatures_array, dtype=np.float32)
    if signatures_array.shape[0] > 1:
        delta[1:] = signatures_array[1:] - signatures_array[:-1]
    return delta


def compute_delta_signature_step_np(
    current_signature: np.ndarray,
    previous_signature: np.ndarray | None,
) -> np.ndarray:
    current = np.asarray(current_signature, dtype=np.float32)
    if current.ndim != 1:
        raise ValueError(
            "Current signature must be 1D when computing online delta signature. "
            f"Got shape={current.shape}."
        )
    if previous_signature is None:
        return np.zeros_like(current, dtype=np.float32)
    previous = np.asarray(previous_signature, dtype=np.float32)
    if previous.shape != current.shape:
        raise ValueError(
            "Previous signature shape mismatch when computing online delta signature. "
            f"previous={previous.shape}, current={current.shape}."
        )
    return current - previous


def resolve_single_visual_observation_feature(cfg) -> tuple[str, tuple[int, ...]]:
    visual_features = getattr(cfg, "visual_observation_features", None)
    if visual_features is None:
        visual_features = getattr(cfg, "image_features", {})
    if len(visual_features) == 0:
        raise RuntimeError("Policy has no regular observation image feature.")

    image_key = next(iter(visual_features))
    image_shape = tuple(visual_features[image_key].shape)
    if len(image_shape) != 3:
        raise RuntimeError(
            "Observation image feature must have shape (C, H, W). "
            f"Got key={image_key!r}, shape={image_shape}."
        )
    return image_key, image_shape


def _resolve_path_candidates(raw: Path) -> list[Path]:
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, PROJECT_ROOT / raw])

    ordered = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def resolve_policy_dir(policy_path: Path) -> Path:
    raw = policy_path.expanduser()
    ordered = _resolve_path_candidates(raw)

    for base in ordered:
        if (base / "model.safetensors").exists():
            return base
        nested = base / "pretrained_model"
        if (nested / "model.safetensors").exists():
            return nested
        last_nested = base / "checkpoints" / "last" / "pretrained_model"
        if (last_nested / "model.safetensors").exists():
            return last_nested

    probe_lines = "\n".join(f"- {path}" for path in ordered)
    raise FileNotFoundError(
        "Could not find policy weights. Checked these base paths:\n"
        f"{probe_lines}\n"
        "Expected one of:\n"
        "- <base>/model.safetensors\n"
        "- <base>/pretrained_model/model.safetensors\n"
        "- <base>/checkpoints/last/pretrained_model/model.safetensors"
    )


def _ensure_local_streaming_act_modules(
    repo_root: Path | None = None,
):
    repo_root = (
        repo_root.resolve(strict=False)
        if repo_root is not None
        else PROJECT_ROOT
    )
    streaming_act_src = (
        repo_root / "policy" / "lerobot_policy_streaming_act" / "src"
    )
    if not streaming_act_src.exists():
        raise FileNotFoundError(
            f"Streaming ACT package source not found: {streaming_act_src}"
        )

    streaming_act_src_str = str(streaming_act_src)
    if streaming_act_src_str not in sys.path:
        sys.path.insert(0, streaming_act_src_str)

    for module_name, module in list(sys.modules.items()):
        if module_name != "lerobot_policy_streaming_act" and not module_name.startswith(
            "lerobot_policy_streaming_act."
        ):
            continue

        module_file = getattr(module, "__file__", None)
        if module_file is None:
            sys.modules.pop(module_name, None)
            continue

        module_path = Path(module_file).resolve(strict=False)
        if not module_path.is_relative_to(streaming_act_src):
            sys.modules.pop(module_name, None)

    configuration_module = importlib.import_module(
        "lerobot_policy_streaming_act.configuration_streaming_act"
    )
    modeling_module = importlib.import_module(
        "lerobot_policy_streaming_act.modeling_streaming_act"
    )
    return (
        configuration_module.StreamingACTConfig,
        modeling_module.StreamingACTPolicy,
    )


def import_local_streaming_act_policy_class(repo_root: Path | None = None):
    _, policy_cls = _ensure_local_streaming_act_modules(repo_root=repo_root)
    return policy_cls


def _parse_config_with_class(
    *,
    config_cls,
    config_path: Path,
    drop_unknown_fields: bool,
):
    import draccus
    from dataclasses import fields

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config.pop("type", None)

    dropped_fields: tuple[str, ...] = ()
    if drop_unknown_fields:
        valid_fields = {field.name for field in fields(config_cls)}
        dropped_fields = tuple(sorted(key for key in raw_config if key not in valid_fields))
        if dropped_fields:
            raw_config = {
                key: value for key, value in raw_config.items() if key in valid_fields
            }

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(raw_config, handle)
            temp_path = Path(handle.name)

        with draccus.config_type("json"):
            cfg = draccus.parse(config_cls, str(temp_path), args=[])
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return cfg, dropped_fields


def load_streaming_act_config_from_pretrained_dir(
    policy_dir: Path,
    *,
    repo_root: Path | None = None,
):
    config_cls, _ = _ensure_local_streaming_act_modules(repo_root=repo_root)
    config_path = Path(policy_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Policy config not found: {config_path}")

    try:
        cfg, _ = _parse_config_with_class(
            config_cls=config_cls,
            config_path=config_path,
            drop_unknown_fields=False,
        )
        return cfg
    except Exception as exc:
        fallback_cfg, dropped_fields = _parse_config_with_class(
            config_cls=config_cls,
            config_path=config_path,
            drop_unknown_fields=True,
        )
        if not dropped_fields:
            raise exc

        LOGGER.warning(
            "Ignored unsupported Streaming ACT config fields while loading %s: %s",
            config_path,
            ", ".join(dropped_fields),
        )
        return fallback_cfg


def find_latest_run_dir(train_root: Path) -> Path | None:
    for resolved_root in _resolve_path_candidates(train_root.expanduser()):
        if not resolved_root.is_dir():
            continue
        candidates = [path for path in resolved_root.iterdir() if path.is_dir()]
        if not candidates:
            continue
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


def resolve_eval_policy_path(
    policy_path: Path | None,
    latest_run_dir: Path | None,
    train_output_root: Path | None,
) -> Path:
    if policy_path is None and latest_run_dir is None:
        if train_output_root is None:
            raise ValueError(
                "Either --policy-path, --latest-run-dir, or --train-output-root "
                "must be provided."
            )
        latest_run_dir = find_latest_run_dir(train_output_root)
        if latest_run_dir is None:
            raise FileNotFoundError(
                "Could not infer latest run from train_output_root="
                f"{train_output_root}"
            )

    if policy_path is None and latest_run_dir is not None:
        return resolve_policy_dir(latest_run_dir)

    if policy_path is None:
        raise FileNotFoundError("Policy path could not be resolved.")

    return resolve_policy_dir(policy_path)


def build_eval_observation(
    state_xy: tuple[float, float] | list[float],
    rgb_frame: np.ndarray,
    state_key: str,
    image_key: str,
    state_dim: int,
    state_vector: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> dict[str, object]:
    import torch

    state_vec = np.zeros((state_dim,), dtype=np.float32)
    if state_vector is None:
        copy_n = min(2, state_dim)
        state_vec[:copy_n] = np.asarray(state_xy[:copy_n], dtype=np.float32)
    else:
        provided = np.asarray(state_vector, dtype=np.float32).reshape(-1)
        copy_n = min(int(provided.shape[0]), state_dim)
        state_vec[:copy_n] = provided[:copy_n]
    return {
        state_key: torch.from_numpy(state_vec),
        image_key: torch.from_numpy(rgb_frame).permute(2, 0, 1).contiguous().float()
        / 255.0,
    }


def build_prefix_sequence_eval_inputs(
    *,
    obs: dict[str, object],
    cfg,
    state_key: str,
    image_key: str | None = None,
    image_keys: list[str] | tuple[str, ...] | None = None,
    signature_key: str | None,
    delta_signature_key: str | None,
    prefix_state_history: list,
    prefix_signature_history: list | None,
    prefix_delta_signature_history: list | None,
    prefix_image_history: list | None = None,
    prefix_image_histories: dict[str, list] | None = None,
) -> None:
    import torch

    from lerobot_policy_streaming_act.prefix_sequence import (
        PREFIX_MASK_KEY,
        PREFIX_DELTA_SIGNATURE_KEY,
        PREFIX_PATH_SIGNATURE_KEY,
        PREFIX_STATE_KEY,
        build_padded_prefix_from_history,
        prefix_image_key_from_camera_key,
    )

    if image_keys is None:
        if image_key is None:
            raise ValueError("Either `image_key` or `image_keys` must be provided.")
        resolved_image_keys = [image_key]
    else:
        resolved_image_keys = list(image_keys)
    if not resolved_image_keys:
        raise ValueError("At least one image key is required for prefix-sequence inputs.")

    if prefix_image_histories is None:
        if prefix_image_history is None:
            raise ValueError(
                "Either `prefix_image_history` or `prefix_image_histories` must be provided."
            )
        if len(resolved_image_keys) != 1:
            raise ValueError(
                "A single `prefix_image_history` list can only be used with one image key."
            )
        resolved_prefix_image_histories = {
            resolved_image_keys[0]: prefix_image_history,
        }
    else:
        resolved_prefix_image_histories = prefix_image_histories
        missing_histories = [
            key for key in resolved_image_keys if key not in resolved_prefix_image_histories
        ]
        if missing_histories:
            raise KeyError(
                "Missing prefix image histories for observation keys: "
                f"{missing_histories}."
            )

    prefix_state_history.append(obs[state_key].detach().clone())
    for current_image_key in resolved_image_keys:
        if current_image_key not in obs:
            raise KeyError(
                f"`{current_image_key}` must be present in the current observation before "
                "building prefix-sequence eval inputs."
            )
        resolved_prefix_image_histories[current_image_key].append(
            obs[current_image_key].detach().clone()
        )
    if signature_key is not None:
        if signature_key not in obs:
            raise KeyError(
                f"`{signature_key}` must be present in the current observation before "
                "building prefix-sequence eval inputs."
            )
        if prefix_signature_history is None:
            raise ValueError(
                "`prefix_signature_history` must be provided when `signature_key` is set."
            )
        prefix_signature_history.append(obs[signature_key].detach().clone())
    if delta_signature_key is not None:
        if delta_signature_key not in obs:
            raise KeyError(
                f"`{delta_signature_key}` must be present in the current observation before "
                "building prefix-sequence eval inputs."
            )
        if prefix_delta_signature_history is None:
            raise ValueError(
                "`prefix_delta_signature_history` must be provided when `delta_signature_key` is set."
            )
        prefix_delta_signature_history.append(obs[delta_signature_key].detach().clone())

    prefix_state, prefix_mask = build_padded_prefix_from_history(
        prefix_state_history,
        prefix_train_max_steps=int(cfg.prefix_train_max_steps),
        prefix_frame_stride=int(cfg.prefix_frame_stride),
        pad_value=float(cfg.prefix_pad_value),
    )
    if signature_key is not None:
        assert prefix_signature_history is not None
        prefix_signature, prefix_signature_mask = build_padded_prefix_from_history(
            prefix_signature_history,
            prefix_train_max_steps=int(cfg.prefix_train_max_steps),
            prefix_frame_stride=int(cfg.prefix_frame_stride),
            pad_value=float(cfg.prefix_pad_value),
        )
        if not torch.equal(prefix_mask, prefix_signature_mask):
            raise RuntimeError("Prefix state/signature masks diverged during online eval.")
    if delta_signature_key is not None:
        assert prefix_delta_signature_history is not None
        prefix_delta_signature, prefix_delta_signature_mask = build_padded_prefix_from_history(
            prefix_delta_signature_history,
            prefix_train_max_steps=int(cfg.prefix_train_max_steps),
            prefix_frame_stride=int(cfg.prefix_frame_stride),
            pad_value=float(cfg.prefix_pad_value),
        )
        if not torch.equal(prefix_mask, prefix_delta_signature_mask):
            raise RuntimeError(
                "Prefix state/delta-signature masks diverged during online eval."
            )

    for current_image_key in resolved_image_keys:
        prefix_images, prefix_image_mask = build_padded_prefix_from_history(
            resolved_prefix_image_histories[current_image_key],
            prefix_train_max_steps=int(cfg.prefix_train_max_steps),
            prefix_frame_stride=int(cfg.prefix_frame_stride),
            pad_value=0.0,
        )
        if not torch.equal(prefix_mask, prefix_image_mask):
            raise RuntimeError("Prefix state/image masks diverged during online eval.")
        obs[prefix_image_key_from_camera_key(current_image_key)] = prefix_images

    obs[PREFIX_STATE_KEY] = prefix_state
    obs[PREFIX_MASK_KEY] = prefix_mask
    if signature_key is not None:
        obs[PREFIX_PATH_SIGNATURE_KEY] = prefix_signature
    if delta_signature_key is not None:
        obs[PREFIX_DELTA_SIGNATURE_KEY] = prefix_delta_signature


def ensure_prefix_sequence_batch_dims(
    *,
    obs: dict[str, object],
    state_key: str,
    image_key: str | None = None,
    image_keys: list[str] | tuple[str, ...] | None = None,
    use_path_signature: bool,
    use_delta_signature: bool,
) -> None:
    from lerobot_policy_streaming_act.prefix_sequence import (
        PREFIX_MASK_KEY,
        PREFIX_DELTA_SIGNATURE_KEY,
        PREFIX_PATH_SIGNATURE_KEY,
        PREFIX_STATE_KEY,
        prefix_image_key_from_camera_key,
    )

    if image_keys is None:
        if image_key is None:
            raise ValueError("Either `image_key` or `image_keys` must be provided.")
        resolved_image_keys = [image_key]
    else:
        resolved_image_keys = list(image_keys)
    if not resolved_image_keys:
        raise ValueError("At least one image key is required for prefix-sequence inputs.")

    prefix_image_keys = [
        prefix_image_key_from_camera_key(current_image_key)
        for current_image_key in resolved_image_keys
    ]
    required_prefix_keys = [
        PREFIX_STATE_KEY,
        PREFIX_MASK_KEY,
        *prefix_image_keys,
    ]
    if use_path_signature:
        required_prefix_keys.append(PREFIX_PATH_SIGNATURE_KEY)
    if use_delta_signature:
        required_prefix_keys.append(PREFIX_DELTA_SIGNATURE_KEY)
    missing_prefix_keys = [key for key in required_prefix_keys if key not in obs]
    if missing_prefix_keys:
        raise KeyError(
            "Missing prefix-sequence keys after preprocessing: "
            f"{missing_prefix_keys}."
        )

    device = obs[state_key].device
    dtype = obs[state_key].dtype

    prefix_state = obs[PREFIX_STATE_KEY]
    if prefix_state.ndim == 2:
        prefix_state = prefix_state.unsqueeze(0)
    elif prefix_state.ndim != 3:
        raise RuntimeError(
            f"`{PREFIX_STATE_KEY}` must be 2D/3D after preprocessing, "
            f"got shape={tuple(prefix_state.shape)}"
        )
    obs[PREFIX_STATE_KEY] = prefix_state.to(device=device, dtype=dtype)

    if use_path_signature:
        prefix_signature = obs[PREFIX_PATH_SIGNATURE_KEY]
        if prefix_signature.ndim == 2:
            prefix_signature = prefix_signature.unsqueeze(0)
        elif prefix_signature.ndim != 3:
            raise RuntimeError(
                f"`{PREFIX_PATH_SIGNATURE_KEY}` must be 2D/3D after preprocessing, "
                f"got shape={tuple(prefix_signature.shape)}"
            )
        obs[PREFIX_PATH_SIGNATURE_KEY] = prefix_signature.to(device=device, dtype=dtype)
    if use_delta_signature:
        prefix_delta_signature = obs[PREFIX_DELTA_SIGNATURE_KEY]
        if prefix_delta_signature.ndim == 2:
            prefix_delta_signature = prefix_delta_signature.unsqueeze(0)
        elif prefix_delta_signature.ndim != 3:
            raise RuntimeError(
                f"`{PREFIX_DELTA_SIGNATURE_KEY}` must be 2D/3D after preprocessing, "
                f"got shape={tuple(prefix_delta_signature.shape)}"
            )
        obs[PREFIX_DELTA_SIGNATURE_KEY] = prefix_delta_signature.to(
            device=device,
            dtype=dtype,
        )

    prefix_mask = obs[PREFIX_MASK_KEY]
    if prefix_mask.ndim == 1:
        prefix_mask = prefix_mask.unsqueeze(0)
    elif prefix_mask.ndim != 2:
        raise RuntimeError(
            f"`{PREFIX_MASK_KEY}` must be 1D/2D after preprocessing, "
            f"got shape={tuple(prefix_mask.shape)}"
        )
    obs[PREFIX_MASK_KEY] = prefix_mask.to(device=device)

    for prefix_image_key in prefix_image_keys:
        prefix_images = obs[prefix_image_key]
        if prefix_images.ndim == 4:
            prefix_images = prefix_images.unsqueeze(0)
        elif prefix_images.ndim != 5:
            raise RuntimeError(
                f"`{prefix_image_key}` must be 4D/5D after preprocessing, "
                f"got shape={tuple(prefix_images.shape)}"
            )
        obs[prefix_image_key] = prefix_images.to(device=device, dtype=dtype)


def compute_signatory_signature_np(window: np.ndarray, sig_depth: int) -> np.ndarray:
    if window.ndim != 2:
        raise ValueError(f"Window must be 2D, got shape={window.shape}")
    if window.shape[0] == 0:
        raise ValueError("Window must contain at least one point.")

    try:
        import torch
        import signatory
    except ImportError as exc:
        raise ImportError(
            "`signatory` is required for the signatory backend. "
            "Install it first or use signature_backend='simple'."
        ) from exc

    # signatory.signature requires at least two points along the stream dimension.
    # Repeat the first point to represent a zero-length initial segment when only
    # one prefix state is available.
    if window.shape[0] < 2:
        window = np.repeat(window[:1], 2, axis=0)

    path = torch.from_numpy(window.astype(np.float32, copy=False)).unsqueeze(0)
    with torch.no_grad():
        signature = signatory.signature(path, depth=sig_depth)
    return signature.squeeze(0).cpu().numpy().astype(np.float32)


def compute_simple_signature_np(window: np.ndarray, sig_depth: int) -> np.ndarray:
    if window.ndim != 2:
        raise ValueError(f"Window must be 2D, got shape={window.shape}")
    if sig_depth <= 0:
        raise ValueError(f"sig_depth must be > 0, got {sig_depth}")

    deltas = np.diff(window, axis=0, prepend=window[:1]).astype(np.float32)
    moments = [
        np.sum(np.power(deltas, order, dtype=np.float32), axis=0)
        for order in range(1, sig_depth + 1)
    ]
    return np.concatenate(moments, axis=0).astype(np.float32)


def check_signatory_usable() -> tuple[bool, str]:
    probe = (
        "import torch\n"
        "import signatory\n"
        "x = torch.randn(1, 8, 2)\n"
        "y = signatory.signature(x, depth=2)\n"
        "print(tuple(y.shape))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ok = proc.returncode == 0
    detail = proc.stderr.strip() if proc.stderr.strip() else proc.stdout.strip()
    if proc.returncode < 0:
        signal_number = -proc.returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"signal {signal_number}"
        crash_detail = (
            f"probe crashed with {signal_name} (returncode={proc.returncode})"
        )
        detail = f"{crash_detail}: {detail}" if detail else crash_detail
    elif not detail:
        detail = f"probe exited with returncode={proc.returncode}"
    return ok, detail


def resolve_signature_backend(requested_backend: str) -> str:
    if requested_backend == "simple":
        return "simple"

    ok, detail = check_signatory_usable()
    if requested_backend == "signatory":
        if not ok:
            raise RuntimeError(
                "signatory backend requested but precheck failed. "
                "Fix the signatory/torch installation or rerun with "
                "`--signature-backend simple`. "
                f"Detail: {detail or 'unknown error'}"
            )
        return "signatory"

    if ok:
        return "signatory"
    print(
        "[WARN] signatory precheck failed; `--signature-backend auto` will "
        "continue with the simple backend. Pass `--signature-backend simple` "
        "to skip this probe. "
        f"Detail: {detail or 'unknown error'}"
    )
    return "simple"


def write_summary(output_dir: Path, summary: dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path
