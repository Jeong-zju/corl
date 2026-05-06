from __future__ import annotations

from collections import deque
import time
from pathlib import Path
from typing import Any
import sys

import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from common import ensure_runtime_paths
from config import DeployConfig, PolicyConfig
from gripper_hysteresis import (
    GripperHysteresis,
    complete_gripper_hysteresis_config_from_dataset_stats,
)
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
from dataset_utils import find_dataset_split_file, load_dataset_split  # noqa: E402


def apply_deploy_policy_overrides(
    cfg: Any,
    deploy_policy: PolicyConfig,
) -> tuple[float, bool]:
    temporal_ensemble_coeff = float(deploy_policy.temporal_ensemble_coeff)
    temporal_ensemble_enabled = temporal_ensemble_coeff != 0.0

    if hasattr(cfg, "temporal_ensemble_coeff"):
        cfg.temporal_ensemble_coeff = (
            temporal_ensemble_coeff if temporal_ensemble_enabled else None
        )
    elif temporal_ensemble_enabled:
        raise ValueError(
            "Deploy `policy.temporal_ensemble_coeff` is only supported for policies "
            "whose config exposes `temporal_ensemble_coeff`."
        )

    if temporal_ensemble_enabled:
        # Temporal ensembling must query the policy at every control step.
        cfg.n_action_steps = 1
    elif deploy_policy.n_action_steps is not None:
        cfg.n_action_steps = int(deploy_policy.n_action_steps)

    return temporal_ensemble_coeff, temporal_ensemble_enabled


def _resolve_training_dataset_root(policy_dir: Path) -> Path | None:
    split_path = find_dataset_split_file(policy_dir)
    if split_path is None:
        return None
    split_spec = load_dataset_split(split_path)
    return Path(split_spec.dataset_root).expanduser().resolve()


class PolicyRuntime:
    def __init__(self, config: DeployConfig) -> None:
        self.deploy_config = config
        self.policy_dir: Path | None = None
        self.policy = None
        self.cfg = None
        self.preprocessor = None
        self.postprocessor = None
        self.gripper_hysteresis: GripperHysteresis | None = None
        self.state_key = config.policy.state_key
        self.action_key = config.policy.action_key
        self.visual_keys = list(config.policy.image_keys.values())
        self.temporal_ensemble_coeff = float(config.policy.temporal_ensemble_coeff)
        self.temporal_ensemble_enabled = self.temporal_ensemble_coeff != 0.0
        self._supports_action_chunk_prediction = False
        self._open_loop_action_queue: deque[tuple[np.ndarray, int]] = deque()

    def load(self) -> None:
        try:
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_pre_post_processors
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing LeRobot deployment dependencies. "
                "Please install the repository `environment.yml` environment first."
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
        (
            self.temporal_ensemble_coeff,
            self.temporal_ensemble_enabled,
        ) = apply_deploy_policy_overrides(cfg, self.deploy_config.policy)

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
        (
            self.temporal_ensemble_coeff,
            self.temporal_ensemble_enabled,
        ) = apply_deploy_policy_overrides(cfg, self.deploy_config.policy)

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
        self._supports_action_chunk_prediction = hasattr(policy, "predict_action_chunk")
        self._open_loop_action_queue.clear()

        if self.deploy_config.gripper_hysteresis.enabled:
            gripper_config = self.deploy_config.gripper_hysteresis
            if (
                not gripper_config.closed_values
                or not gripper_config.open_values
                or not gripper_config.close_thresholds
                or not gripper_config.open_thresholds
            ):
                dataset_root = _resolve_training_dataset_root(self.policy_dir)
                if dataset_root is None:
                    raise ValueError(
                        "`gripper_hysteresis` is enabled, but deploy could not find "
                        "dataset_split.json near the policy path. Provide explicit "
                        "closed/open values and thresholds in the deploy YAML."
                    )
                gripper_config = complete_gripper_hysteresis_config_from_dataset_stats(
                    gripper_config,
                    dataset_root=dataset_root,
                    action_key=self.action_key,
                )
            self.gripper_hysteresis = GripperHysteresis(gripper_config)

    def reset(self) -> None:
        self._open_loop_action_queue.clear()
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()
        if self.gripper_hysteresis is not None:
            self.gripper_hysteresis.reset()

    @property
    def execution_summary(self) -> str:
        if self.temporal_ensemble_enabled:
            return f"temporal_ensemble(coeff={self.temporal_ensemble_coeff:g})"
        n_action_steps = (
            1
            if self.cfg is None
            else int(getattr(self.cfg, "n_action_steps", 1))
        )
        return f"open_loop(n_action_steps={n_action_steps})"

    @staticmethod
    def _debug_to_numpy(value: Any) -> Any:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return {
                str(key): PolicyRuntime._debug_to_numpy(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [PolicyRuntime._debug_to_numpy(item) for item in value]
        return value

    def _collect_debug_snapshot(self) -> dict[str, Any]:
        getter = getattr(self.policy, "get_deploy_debug_snapshot", None)
        if not callable(getter):
            return {}
        return {"debug": self._debug_to_numpy(getter())}

    def _apply_action_filters(
        self,
        action: np.ndarray,
        observation_packet: dict[str, Any],
    ) -> np.ndarray:
        if self.gripper_hysteresis is not None:
            action = self.gripper_hysteresis.apply(
                action,
                current_state=observation_packet.get("state"),
            )
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def _preprocess_observation(self, observation_packet: dict[str, Any]) -> dict[str, Any]:
        raw_obs = build_raw_policy_observation(observation_packet, self.cfg)
        obs = self.preprocessor(raw_obs)
        return finalize_preprocessed_observation(obs, self.cfg)

    def _run_select_action(
        self,
        obs: dict[str, Any],
    ) -> tuple[np.ndarray, float]:
        import torch

        start_s = time.perf_counter()
        with torch.no_grad():
            predicted_action = self.policy.select_action(obs)
        predicted_action = self.postprocessor(predicted_action)
        runtime_ms = (time.perf_counter() - start_s) * 1000.0

        action = predicted_action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return action, float(runtime_ms)

    def _run_open_loop_chunk(
        self,
        obs: dict[str, Any],
        *,
        obs_seq: int,
    ) -> tuple[np.ndarray, float]:
        import torch

        start_s = time.perf_counter()
        with torch.no_grad():
            predicted_chunk = self.policy.predict_action_chunk(obs)
        predicted_chunk = self.postprocessor(predicted_chunk)
        runtime_ms = (time.perf_counter() - start_s) * 1000.0

        chunk = predicted_chunk.detach().cpu().numpy().astype(np.float32)
        if chunk.ndim == 2:
            chunk = chunk[None, ...]
        if chunk.ndim != 3 or chunk.shape[0] != 1:
            raise RuntimeError(
                "Expected open-loop chunk prediction to have shape "
                f"(1, chunk_size, action_dim), got {tuple(chunk.shape)}."
            )

        n_action_steps = max(1, int(getattr(self.cfg, "n_action_steps", 1)))
        scheduled_actions = [
            np.asarray(step, dtype=np.float32).reshape(-1)
            for step in chunk[0, :n_action_steps]
        ]
        if not scheduled_actions:
            raise RuntimeError("Policy returned an empty action chunk during open-loop deploy.")

        self._open_loop_action_queue.extend(
            (action, obs_seq) for action in scheduled_actions[1:]
        )
        return scheduled_actions[0], float(runtime_ms)

    def _build_inference_result(
        self,
        *,
        action: np.ndarray,
        observation_packet: dict[str, Any],
        runtime_ms: float | None,
        obs_seq: int,
        message: str,
        include_debug: bool,
    ) -> dict[str, Any]:
        result = {
            "action": self._apply_action_filters(action, observation_packet),
            "runtime_ms": None if runtime_ms is None else float(runtime_ms),
            "obs_seq": int(obs_seq),
            "message": message,
        }
        if include_debug:
            result.update(self._collect_debug_snapshot())
        return result

    def infer(
        self,
        observation_packet: dict[str, Any],
        *,
        collect_debug: bool = False,
    ) -> dict[str, Any]:
        if self.policy is None or self.cfg is None:
            raise RuntimeError("Policy has not been loaded.")

        if observation_packet.get("reset"):
            self.reset()

        obs_seq = int(observation_packet.get("seq", 0))
        if not self.temporal_ensemble_enabled and self._open_loop_action_queue:
            action, source_obs_seq = self._open_loop_action_queue.popleft()
            return self._build_inference_result(
                action=action,
                observation_packet=observation_packet,
                runtime_ms=None,
                obs_seq=source_obs_seq,
                message="open_loop_cached",
                include_debug=False,
            )

        obs = self._preprocess_observation(observation_packet)
        if not self.temporal_ensemble_enabled and self._supports_action_chunk_prediction:
            action, runtime_ms = self._run_open_loop_chunk(obs, obs_seq=obs_seq)
            return self._build_inference_result(
                action=action,
                observation_packet=observation_packet,
                runtime_ms=runtime_ms,
                obs_seq=obs_seq,
                message="open_loop_predict",
                include_debug=collect_debug,
            )

        action, runtime_ms = self._run_select_action(obs)
        return self._build_inference_result(
            action=action,
            observation_packet=observation_packet,
            runtime_ms=runtime_ms,
            obs_seq=obs_seq,
            message=(
                "temporal_ensemble"
                if self.temporal_ensemble_enabled
                else "policy_eval"
            ),
            include_debug=collect_debug,
        )
