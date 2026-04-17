from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import sys

import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from common import ensure_runtime_paths
from config import DeployConfig
from policy_runtime.preprocess import (
    build_raw_policy_observation,
    finalize_preprocessed_observation,
    resolve_action_key,
    resolve_state_key,
    select_visual_observation_keys,
)

ensure_runtime_paths()

from eval_helpers import (  # noqa: E402
    import_local_streaming_act_policy_class,
    load_streaming_act_config_from_pretrained_dir,
    resolve_policy_dir,
)
class PolicyRuntime:
    def __init__(self, config: DeployConfig) -> None:
        self.deploy_config = config
        self.policy_dir: Path | None = None
        self.policy = None
        self.cfg = None
        self.preprocessor = None
        self.postprocessor = None
        self.state_key = config.policy.state_key
        self.action_key = config.policy.action_key
        self.visual_keys = list(config.policy.image_keys.values())

    def load(self) -> None:
        try:
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_pre_post_processors
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing LeRobot deployment dependencies. "
                "Please install the `main/environment.yml` environment first."
            ) from exc

        policy_path = self.deploy_config.policy.path
        if policy_path is None:
            raise ValueError("`policy.path` is required in the deploy YAML.")

        policy_type = self.deploy_config.policy.type
        if policy_type == "streaming_act":
            policy_cls = import_local_streaming_act_policy_class()
        elif policy_type == "prism_diffusion":
            raise ValueError(
                "Unsupported policy type for deploy runtime: 'prism_diffusion'. "
                "The current deploy runtime supports only 'act' and 'streaming_act'. "
                "Use scripts/eval_policy.py for PRISM Diffusion checkpoints "
                "until deploy support is added."
            )
        elif policy_type == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy

            policy_cls = ACTPolicy
        else:
            raise ValueError(f"Unsupported policy type: {policy_type!r}")

        self.policy_dir = resolve_policy_dir(policy_path)
        local_files_only = self.policy_dir.is_dir()
        if policy_type == "streaming_act":
            cfg = load_streaming_act_config_from_pretrained_dir(self.policy_dir)
        else:
            cfg = PreTrainedConfig.from_pretrained(
                self.policy_dir,
                local_files_only=local_files_only,
            )

        load_device = (
            self.deploy_config.policy.load_device or self.deploy_config.policy.device
        )
        cfg.device = load_device
        policy = policy_cls.from_pretrained(
            self.policy_dir,
            config=cfg,
            local_files_only=local_files_only,
        )

        cfg = policy.config
        cfg.device = self.deploy_config.policy.device
        if self.deploy_config.policy.n_action_steps is not None:
            cfg.n_action_steps = int(self.deploy_config.policy.n_action_steps)

        if hasattr(policy, "to"):
            policy.to(self.deploy_config.policy.device)
        policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()

        preprocessor_overrides = {
            "device_processor": {"device": self.deploy_config.policy.device},
            "rename_observations_processor": {"rename_map": {}},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=self.policy_dir,
            preprocessor_overrides=preprocessor_overrides,
        )

        self.policy = policy
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.state_key = resolve_state_key(cfg)
        self.action_key = resolve_action_key(cfg)
        self.visual_keys = select_visual_observation_keys(cfg)

    def reset(self) -> None:
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def infer(self, observation_packet: dict[str, Any]) -> dict[str, Any]:
        if self.policy is None or self.cfg is None:
            raise RuntimeError("Policy has not been loaded.")

        import torch

        if observation_packet.get("reset"):
            self.reset()

        raw_obs = build_raw_policy_observation(observation_packet, self.cfg)
        obs = self.preprocessor(raw_obs)
        obs = finalize_preprocessed_observation(obs, self.cfg)

        start_s = time.perf_counter()
        with torch.no_grad():
            predicted_action = self.policy.select_action(obs)
        predicted_action = self.postprocessor(predicted_action)
        runtime_ms = (time.perf_counter() - start_s) * 1000.0

        action = predicted_action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return {
            "action": action,
            "runtime_ms": float(runtime_ms),
        }
