from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deploy.utils import repo_root_from_main


def _resolve_path_candidates(raw: Path, repo_root: Path) -> list[Path]:
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, repo_root / raw, repo_root / "main" / raw])

    ordered = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def resolve_policy_dir(policy_path: Path, repo_root: Path) -> Path:
    ordered = _resolve_path_candidates(policy_path, repo_root)
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


def ensure_streaming_act_importable(main_root: Path) -> None:
    streaming_act_src = main_root / "policy" / "lerobot_policy_streaming_act" / "src"
    if not streaming_act_src.exists():
        raise FileNotFoundError(f"Streaming ACT source not found: {streaming_act_src}")
    streaming_act_src_str = str(streaming_act_src)
    if streaming_act_src_str not in sys.path:
        sys.path.insert(0, streaming_act_src_str)


@dataclass(slots=True)
class PolicyBundle:
    policy_type: str
    policy_dir: Path
    device: str
    config: Any
    policy: Any
    preprocessor: Any
    postprocessor: Any
    state_key: str
    image_keys: tuple[str, ...]
    use_path_signature: bool
    use_delta_signature: bool
    supports_reset: bool

    def reset(self) -> None:
        if self.supports_reset:
            self.policy.reset()


def load_policy_bundle(
    *,
    main_root: Path,
    policy_path: str | Path,
    policy_type: str,
    device: str,
    n_action_steps: int | None = None,
    load_device: str | None = None,
) -> PolicyBundle:
    repo_root = repo_root_from_main(main_root)
    normalized_policy_type = str(policy_type).strip().lower()
    requested_device = str(device)
    effective_load_device = (
        str(load_device)
        if load_device is not None
        else ("cpu" if requested_device.split(":", 1)[0] == "cuda" else requested_device)
    )

    if normalized_policy_type == "streaming_act":
        ensure_streaming_act_importable(main_root)

    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing LeRobot runtime dependencies. Install `lerobot`, `torch`, and the "
            "policy package environment before launching the policy server."
        ) from exc

    if normalized_policy_type == "streaming_act":
        from lerobot_policy_streaming_act.modeling_streaming_act import StreamingACTPolicy

        policy_cls = StreamingACTPolicy
    elif normalized_policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        policy_cls = ACTPolicy
    else:
        raise ValueError(
            f"Unsupported policy type: {policy_type!r}. Expected `act` or `streaming_act`."
        )

    resolved_policy_dir = resolve_policy_dir(Path(policy_path), repo_root)
    local_files_only = resolved_policy_dir.is_dir()
    print(f"[load] Using policy directory: {resolved_policy_dir}", flush=True)

    config_load_start_s = time.perf_counter()
    print("[load] Loading policy config...", flush=True)
    cfg = PreTrainedConfig.from_pretrained(
        resolved_policy_dir,
        local_files_only=local_files_only,
    )
    print(
        f"[timing] Policy config loaded in {(time.perf_counter() - config_load_start_s):.2f}s",
        flush=True,
    )
    cfg.device = effective_load_device
    if n_action_steps is not None:
        cfg.n_action_steps = int(n_action_steps)

    policy_load_start_s = time.perf_counter()
    print(
        "[load] Loading policy weights "
        f"(load_device={effective_load_device}, runtime_device={requested_device})...",
        flush=True,
    )
    policy = policy_cls.from_pretrained(
        resolved_policy_dir,
        config=cfg,
        local_files_only=local_files_only,
    )
    print(
        f"[timing] Policy weights loaded in {(time.perf_counter() - policy_load_start_s):.2f}s",
        flush=True,
    )
    if effective_load_device != requested_device:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Moving the loaded policy to the runtime device requires `torch`."
            ) from exc
        move_start_s = time.perf_counter()
        print(
            f"[load] Moving policy from {effective_load_device} to {requested_device}...",
            flush=True,
        )
        policy.to(requested_device)
        if requested_device.split(":", 1)[0] == "cuda":
            torch.cuda.synchronize()
        print(
            f"[timing] Policy moved to {requested_device} in {(time.perf_counter() - move_start_s):.2f}s",
            flush=True,
        )
    cfg.device = requested_device
    if hasattr(policy, "config"):
        policy.config.device = requested_device
    policy.eval()
    effective_cfg = getattr(policy, "config", cfg)
    chunk_size = getattr(effective_cfg, "chunk_size", None)
    effective_n_action_steps = getattr(effective_cfg, "n_action_steps", None)
    print(
        "[load] Policy config summary: "
        f"policy_type={normalized_policy_type}, "
        f"device={requested_device}, "
        f"chunk_size={chunk_size}, "
        f"n_action_steps={effective_n_action_steps}",
        flush=True,
    )
    if effective_n_action_steps is not None and int(effective_n_action_steps) > 1:
        print(
            "[warn] Policy runtime is using n_action_steps>1. "
            "For responsive online deployment, prefer `--n-action-steps 1`.",
            flush=True,
        )

    preprocessor_overrides = {
        "device_processor": {"device": requested_device},
        "rename_observations_processor": {"rename_map": {}},
    }
    processor_load_start_s = time.perf_counter()
    print("[load] Initializing pre/post processors...", flush=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=resolved_policy_dir,
        preprocessor_overrides=preprocessor_overrides,
    )
    print(
        "[timing] Pre/post processors initialized in "
        f"{(time.perf_counter() - processor_load_start_s):.2f}s",
        flush=True,
    )

    visual_features = getattr(cfg, "visual_observation_features", None)
    if visual_features is None:
        visual_features = getattr(cfg, "image_features", {})
    image_keys = tuple(str(key) for key in visual_features.keys())
    if "observation.state" not in getattr(cfg, "input_features", {}):
        raise RuntimeError("Checkpoint does not expose `observation.state` as an input feature.")

    return PolicyBundle(
        policy_type=normalized_policy_type,
        policy_dir=resolved_policy_dir,
        device=requested_device,
        config=cfg,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        state_key="observation.state",
        image_keys=image_keys,
        use_path_signature=bool(getattr(cfg, "use_path_signature", False)),
        use_delta_signature=bool(getattr(cfg, "use_delta_signature", False)),
        supports_reset=hasattr(policy, "reset"),
    )
