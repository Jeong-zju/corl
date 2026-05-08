from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
from collections import deque
from types import ModuleType
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
    load_pretrained_config_from_pretrained_dir,
    resolve_policy_dir,
    quiet_transformers_loading,
)
from policy_imports import ensure_lerobot_policy_imports  # noqa: E402
from dataset_utils import find_dataset_split_file, load_dataset_split  # noqa: E402


RTC_POLICY_TYPES = {"pi0", "pi05", "smolvla"}
VLA_POLICY_TYPES = RTC_POLICY_TYPES
TEMPORAL_ENSEMBLE_POLICY_TYPES = {"act", "streaming_act"}
DEPLOY_RTC_POLICY_TYPES = {"pi05", "smolvla"}

_LEROBOT_POLICY_SUBPACKAGES_BY_TYPE = {
    "act": "act",
    "pi0": "pi0",
    "pi05": "pi05",
    "diffusion": "diffusion",
    "smolvla": "smolvla",
}

_LEROBOT_POLICY_SUBPACKAGE_DEPENDENCIES = {
    "pi0": ("rtc",),
    "pi05": ("rtc",),
    "smolvla": ("rtc",),
}

_LEROBOT_CONFIG_MODULES_BY_TYPE = {
    "act": "lerobot.policies.act.configuration_act",
    "pi0": "lerobot.policies.pi0.configuration_pi0",
    "pi05": "lerobot.policies.pi05.configuration_pi05",
    "smolvla": "lerobot.policies.smolvla.configuration_smolvla",
}

_LEROBOT_PROCESSOR_MODULES_BY_TYPE = {
    "act": "lerobot.policies.act.processor_act",
    "pi0": "lerobot.policies.pi0.processor_pi0",
    "pi05": "lerobot.policies.pi05.processor_pi05",
    "smolvla": "lerobot.policies.smolvla.processor_smolvla",
}


def _make_namespace_package(name: str, package_path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(package_path / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_path)]
    module.__spec__ = spec
    return module


def _install_lerobot_policies_namespace_shim() -> Path | None:
    """Avoid LeRobot 0.5.0's eager `lerobot.policies` imports during deploy.

    LeRobot's policy package imports every policy config at package import time.
    On some Python 3.12 / older LeRobot combinations, a dataclass bug
    raises during import before deploy can reach the policy it actually needs.
    Treating `lerobot.policies` as a namespace package lets us import only the
    concrete policy submodules we need.
    """
    lerobot_spec = importlib.util.find_spec("lerobot")
    search_locations = getattr(lerobot_spec, "submodule_search_locations", None)
    if not search_locations:
        return None

    policies_path: Path | None = None
    for root in search_locations:
        candidate = Path(root) / "policies"
        if candidate.is_dir():
            policies_path = candidate
            break
    if policies_path is None:
        return None

    existing = sys.modules.get("lerobot.policies")
    if existing is not None and getattr(existing, "__path__", None):
        existing_paths = [Path(path) for path in getattr(existing, "__path__", ())]
        existing_origin = getattr(getattr(existing, "__spec__", None), "origin", None)
        if existing_origin is None and existing_paths and existing_paths[0] == policies_path:
            return existing_paths[0]

    module = _make_namespace_package("lerobot.policies", policies_path)
    sys.modules["lerobot.policies"] = module
    parent = sys.modules.get("lerobot")
    if parent is not None:
        setattr(parent, "policies", module)
    return policies_path


def _install_lerobot_policy_subpackage_shim_by_name(package_name: str) -> None:
    policies_path = _install_lerobot_policies_namespace_shim()
    if policies_path is None:
        return

    full_name = f"lerobot.policies.{package_name}"
    existing = sys.modules.get(full_name)
    if existing is not None and getattr(existing, "__path__", None):
        return

    package_path = policies_path / package_name
    if not package_path.is_dir():
        return

    module = _make_namespace_package(full_name, package_path)
    sys.modules[full_name] = module
    parent = sys.modules.get("lerobot.policies")
    if parent is not None:
        setattr(parent, package_name, module)


def _install_lerobot_policy_subpackage_shim(policy_type: str) -> None:
    package_names = []
    package_name = _LEROBOT_POLICY_SUBPACKAGES_BY_TYPE.get(policy_type)
    if package_name is not None:
        package_names.append(package_name)
    package_names.extend(_LEROBOT_POLICY_SUBPACKAGE_DEPENDENCIES.get(policy_type, ()))

    for name in package_names:
        _install_lerobot_policy_subpackage_shim_by_name(name)


def _import_lerobot_policy_submodule(policy_type: str, module_name: str):
    _install_lerobot_policy_subpackage_shim(policy_type)
    return importlib.import_module(module_name)


def _load_lerobot_pretrained_config_class(policy_type: str):
    _install_lerobot_policies_namespace_shim()
    if policy_type == "streaming_act":
        from lerobot_policy_streaming_act.configuration_streaming_act import (
            StreamingACTConfig,
        )

        return StreamingACTConfig
    if policy_type in {"pi0", "pi05"}:
        ensure_lerobot_policy_imports(policy_type)
        config_module_name = _LEROBOT_CONFIG_MODULES_BY_TYPE[policy_type]
        module = importlib.import_module(config_module_name)
        if policy_type == "pi0":
            return module.PI0Config
        return module.PI05Config
    config_module_name = _LEROBOT_CONFIG_MODULES_BY_TYPE.get(policy_type)
    if config_module_name is not None:
        module = _import_lerobot_policy_submodule(policy_type, config_module_name)
        if policy_type == "act":
            return module.ACTConfig
        if policy_type == "smolvla":
            return module.SmolVLAConfig
    raise ValueError(f"Unsupported policy type for deploy config loading: {policy_type!r}")


def _register_deploy_processor_steps(policy_type: str) -> None:
    if policy_type == "streaming_act":
        importlib.import_module("lerobot_policy_streaming_act.processor_streaming_act")
        return

    if policy_type in {"pi0", "pi05"}:
        ensure_lerobot_policy_imports(policy_type)

    processor_module_name = _LEROBOT_PROCESSOR_MODULES_BY_TYPE.get(policy_type)
    if processor_module_name is not None:
        _import_lerobot_policy_submodule(policy_type, processor_module_name)


def _make_deploy_pre_post_processors(
    *,
    policy_type: str,
    policy_cfg: Any,
    pretrained_path: Path,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
):
    _register_deploy_processor_steps(policy_type)

    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        ACTION,
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )


    return (
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            overrides=preprocessor_overrides or {},
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            overrides=postprocessor_overrides or {},
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def apply_deploy_policy_overrides(
    policy_type: str,
    cfg: Any,
    deploy_policy: PolicyConfig,
) -> tuple[float, bool]:
    temporal_ensemble_coeff = float(deploy_policy.temporal_ensemble_coeff)
    temporal_ensemble_enabled = temporal_ensemble_coeff != 0.0

    if policy_type not in TEMPORAL_ENSEMBLE_POLICY_TYPES and temporal_ensemble_enabled:
        raise ValueError(
            f"Deploy `policy.temporal_ensemble_coeff` is not supported for policy "
            f"type {policy_type!r}."
        )

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


def _coerce_rtc_attention_schedule(value: str):
    from lerobot.configs.types import RTCAttentionSchedule

    normalized = str(value).strip().lower()
    for member in RTCAttentionSchedule:
        if member.name.lower() == normalized or str(member.value).lower() == normalized:
            return member
    raise ValueError(f"Unsupported RTC prefix attention schedule: {value!r}")


def apply_deploy_rtc_overrides(policy_type: str, cfg: Any, deploy_policy: PolicyConfig) -> bool:
    rtc = deploy_policy.rtc
    if not rtc.enabled:
        return False
    if policy_type not in DEPLOY_RTC_POLICY_TYPES:
        raise ValueError(
            f"Deploy RTC is only supported for {sorted(DEPLOY_RTC_POLICY_TYPES)}, "
            f"got {policy_type!r}."
        )

    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    cfg.rtc_config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=_coerce_rtc_attention_schedule(
            rtc.prefix_attention_schedule
        ),
        max_guidance_weight=float(rtc.max_guidance_weight),
        execution_horizon=int(rtc.execution_horizon),
        debug=bool(rtc.debug),
        debug_maxlen=int(rtc.debug_maxlen),
    )
    cfg.n_action_steps = int(rtc.execution_horizon)
    return True


def _resolve_training_dataset_root(policy_dir: Path) -> Path | None:
    split_path = find_dataset_split_file(policy_dir)
    if split_path is None:
        return None
    split_spec = load_dataset_split(split_path)
    return Path(split_spec.dataset_root).expanduser().resolve()



def _missing_dependency_error(
    *,
    policy_name: str,
    extra_name: str,
    exc: ModuleNotFoundError,
) -> RuntimeError:
    missing_module = getattr(exc, "name", None)
    if missing_module and not str(missing_module).startswith("lerobot"):
        return RuntimeError(
            f"Missing Python dependency for {policy_name} deploy: "
            f"`{missing_module}`. Install this project's deployment requirements "
            "with `python -m pip install -r requirements.txt`, or install the "
            f"LeRobot extra directly with `python -m pip install \"lerobot[{extra_name}]==0.5.0\"`."
        )
    return RuntimeError(
        f"{policy_name} deploy support is missing from this LeRobot installation. "
        f"Install LeRobot with the {policy_name} extra, for example "
        f'`python -m pip install "lerobot[{extra_name}]==0.5.0"`.'
    )


def resolve_deploy_policy_class(policy_type: str, deploy_policy: PolicyConfig):
    _install_lerobot_policies_namespace_shim()
    if policy_type == "streaming_act":
        return import_local_streaming_act_policy_class()
    if policy_type in {"pi0", "pi05"}:
        try:
            ensure_lerobot_policy_imports(policy_type)
            module_name = (
                "lerobot.policies.pi0.modeling_pi0"
                if policy_type == "pi0"
                else "lerobot.policies.pi05.modeling_pi05"
            )
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            policy_name = "Pi0" if policy_type == "pi0" else "Pi0.5"
            raise _missing_dependency_error(
                policy_name=policy_name,
                extra_name="pi" if policy_type == "pi0" else "pi",
                exc=exc,
            ) from exc
        return module.PI0Policy if policy_type == "pi0" else module.PI05Policy
    if policy_type == "smolvla":
        try:
            module = _import_lerobot_policy_submodule(
                policy_type,
                "lerobot.policies.smolvla.modeling_smolvla",
            )
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error(
                policy_name="SmolVLA",
                extra_name="smolvla",
                exc=exc,
            ) from exc
        return module.SmolVLAPolicy
    if policy_type == "act":
        try:
            module = _import_lerobot_policy_submodule(
                policy_type,
                "lerobot.policies.act.modeling_act",
            )
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error(
                policy_name="ACT",
                extra_name="act",
                exc=exc,
            ) from exc
        return module.ACTPolicy
    raise ValueError(f"Unsupported policy type: {policy_type!r}")


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
        self.action_smoothing_enabled = False
        self._smoothed_action: np.ndarray | None = None
        self.rtc_enabled = False
        self._rtc_action_queue: deque[np.ndarray] = deque()
        self._rtc_prev_chunk_left_over = None
        self._rtc_last_inference_delay_steps = 0

    def load(self) -> None:
        policy_path = self.deploy_config.policy.path
        if policy_path is None:
            raise ValueError("`policy.path` is required in the deploy YAML.")

        policy_type = self.deploy_config.policy.type
        if policy_type == "prism_diffusion":
            raise ValueError(
                "Unsupported policy type for deploy runtime: 'prism_diffusion'. "
                "The current deploy runtime supports 'act', 'streaming_act', "
                "'pi0', 'pi05', 'smolvla'. "
                "Use scripts/eval_policy.py for PRISM Diffusion checkpoints "
                "until deploy support is added."
            )
        if policy_type in VLA_POLICY_TYPES and not self.deploy_config.policy.task:
            raise ValueError(
                f"`policy.task` is required when deploying {policy_type!r} policies."
            )

        try:
            PreTrainedConfig = _load_lerobot_pretrained_config_class(policy_type)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing LeRobot deployment dependencies. "
                "Please install the repository `environment.yml` environment first."
            ) from exc

        policy_cls = resolve_deploy_policy_class(policy_type, self.deploy_config.policy)

        self.policy_dir = resolve_policy_dir(policy_path)
        local_files_only = self.policy_dir.is_dir()
        if policy_type == "streaming_act":
            cfg = load_streaming_act_config_from_pretrained_dir(self.policy_dir)
        else:
            policy_label = {
                "act": "ACT",
                "pi0": "Pi0",
                "pi05": "Pi0.5",
                "smolvla": "SmolVLA",
            }.get(policy_type, "policy")
            cfg = load_pretrained_config_from_pretrained_dir(
                PreTrainedConfig,
                self.policy_dir,
                policy_label=policy_label,
            )
        (
            self.temporal_ensemble_coeff,
            self.temporal_ensemble_enabled,
        ) = apply_deploy_policy_overrides(
            policy_type,
            cfg,
            self.deploy_config.policy,
        )
        self.rtc_enabled = apply_deploy_rtc_overrides(
            policy_type,
            cfg,
            self.deploy_config.policy,
        )

        load_device = (
            self.deploy_config.policy.load_device or self.deploy_config.policy.device
        )
        cfg.device = load_device
        with quiet_transformers_loading():
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
        ) = apply_deploy_policy_overrides(
            policy_type,
            cfg,
            self.deploy_config.policy,
        )
        self.rtc_enabled = apply_deploy_rtc_overrides(
            policy_type,
            cfg,
            self.deploy_config.policy,
        )
        self.action_smoothing_enabled = False

        if hasattr(policy, "to"):
            policy.to(self.deploy_config.policy.device)
        policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()
        self._smoothed_action = None
        self._reset_rtc_state()

        preprocessor_overrides = {
            "device_processor": {"device": self.deploy_config.policy.device},
            "rename_observations_processor": {"rename_map": {}},
        }
        preprocessor, postprocessor = _make_deploy_pre_post_processors(
            policy_type=policy_type,
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
        self._smoothed_action = None
        self._reset_rtc_state()
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()
        if self.gripper_hysteresis is not None:
            self.gripper_hysteresis.reset()

    @property
    def execution_summary(self) -> str:
        if self.rtc_enabled:
            rtc = self.deploy_config.policy.rtc
            return (
                "rtc("
                f"execution_horizon={rtc.execution_horizon}, "
                f"schedule={rtc.prefix_attention_schedule}, "
                f"max_guidance_weight={rtc.max_guidance_weight:g}"
                ")"
            )
        if self.temporal_ensemble_enabled:
            return f"temporal_ensemble(coeff={self.temporal_ensemble_coeff:g})"
        if self.action_smoothing_enabled:
            return f"action_smoothing(coeff={self.temporal_ensemble_coeff:g})"
        n_action_steps = (
            1
            if self.cfg is None
            else int(getattr(self.cfg, "n_action_steps", 1))
        )
        return f"open_loop(n_action_steps={n_action_steps})"

    def _reset_rtc_state(self) -> None:
        self._rtc_action_queue.clear()
        self._rtc_prev_chunk_left_over = None
        self._rtc_last_inference_delay_steps = 0

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

    def _apply_action_smoothing(self, action: np.ndarray) -> np.ndarray:
        vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if not self.action_smoothing_enabled:
            return vector

        # Use an exponential moving average as the deploy-side fallback for
        # policies that do not expose native temporal ensembling.
        if self._smoothed_action is None or self._smoothed_action.shape != vector.shape:
            self._smoothed_action = vector.copy()
            return self._smoothed_action.copy()

        decay = float(np.exp(-self.temporal_ensemble_coeff))
        self._smoothed_action = (
            decay * self._smoothed_action + (1.0 - decay) * vector
        ).astype(np.float32, copy=False)
        return self._smoothed_action.copy()

    def _apply_action_filters(
        self,
        action: np.ndarray,
        observation_packet: dict[str, Any],
    ) -> np.ndarray:
        action = self._apply_action_smoothing(action)
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

    def _resolve_rtc_inference_delay_steps(self) -> int:
        configured = self.deploy_config.policy.rtc.inference_delay_steps
        if configured is not None:
            return int(configured)
        return int(max(0, self._rtc_last_inference_delay_steps))

    def _record_rtc_runtime(self, runtime_ms: float) -> None:
        period_ms = 1000.0 / float(self.deploy_config.runtime.control_hz)
        if period_ms <= 0.0:
            self._rtc_last_inference_delay_steps = 0
            return
        self._rtc_last_inference_delay_steps = int(max(0, np.ceil(runtime_ms / period_ms)))

    def _run_rtc_predict_action(
        self,
        obs: dict[str, Any],
    ) -> tuple[np.ndarray, float]:
        if self._rtc_action_queue:
            return self._rtc_action_queue.popleft(), 0.0

        execution_horizon = int(self.deploy_config.policy.rtc.execution_horizon)
        inference_delay = self._resolve_rtc_inference_delay_steps()

        start_s = time.perf_counter()
        normalized_chunk = self.policy.predict_action_chunk(
            obs,
            prev_chunk_left_over=self._rtc_prev_chunk_left_over,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
        runtime_ms = (time.perf_counter() - start_s) * 1000.0
        self._record_rtc_runtime(runtime_ms)

        if normalized_chunk.ndim != 3:
            raise RuntimeError(
                "RTC predict_action_chunk must return shape (B, T, action_dim), "
                f"got {tuple(normalized_chunk.shape)}."
            )

        # Keep the still-normalized leftover for the next RTC planning call.
        self._rtc_prev_chunk_left_over = normalized_chunk[
            :,
            min(execution_horizon, normalized_chunk.shape[1]) :,
            :,
        ].detach()

        predicted_chunk = self.postprocessor(normalized_chunk)
        chunk_np = (
            predicted_chunk.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        if chunk_np.ndim == 3:
            chunk_np = chunk_np[0]
        elif chunk_np.ndim != 2:
            raise RuntimeError(
                "RTC postprocessed action chunk must have shape (T, action_dim) "
                f"or (B, T, action_dim), got {chunk_np.shape}."
            )

        num_actions = min(execution_horizon, chunk_np.shape[0])
        for action in chunk_np[:num_actions]:
            self._rtc_action_queue.append(np.asarray(action, dtype=np.float32).reshape(-1))

        if not self._rtc_action_queue:
            raise RuntimeError("RTC action chunk was empty.")
        return self._rtc_action_queue.popleft(), float(runtime_ms)

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
        obs = self._preprocess_observation(observation_packet)
        if self.rtc_enabled:
            action, runtime_ms = self._run_rtc_predict_action(obs)
        else:
            action, runtime_ms = self._run_select_action(obs)
        return self._build_inference_result(
            action=action,
            observation_packet=observation_packet,
            runtime_ms=runtime_ms,
            obs_seq=obs_seq,
            message=(
                "rtc"
                if self.rtc_enabled
                else
                "temporal_ensemble"
                if self.temporal_ensemble_enabled
                else "policy_eval"
            ),
            include_debug=collect_debug,
        )
