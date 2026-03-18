from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np


def resolve_policy_dir(policy_path: Path) -> Path:
    raw = policy_path.expanduser()
    repo_root = Path(__file__).resolve().parents[2]

    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, repo_root / raw, repo_root / "main" / raw])

    for candidate in list(candidates):
        candidate_str = str(candidate)
        repo_root_str = str(repo_root)
        if candidate_str.startswith(f"{repo_root_str}/outputs/"):
            suffix = candidate_str[len(f"{repo_root_str}/outputs/") :]
            candidates.append(repo_root / "main" / "outputs" / suffix)
        if candidate_str.startswith(f"{repo_root_str}/main/outputs/"):
            suffix = candidate_str[len(f"{repo_root_str}/main/outputs/") :]
            candidates.append(repo_root / "outputs" / suffix)

    ordered = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)

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
