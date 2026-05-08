from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


_LEROBOT_POLICY_SUBPACKAGES_BY_TYPE = {
    "act": "act",
    "diffusion": "diffusion",
    "pi05": "pi05",
    "smolvla": "smolvla",
}

_LEROBOT_POLICY_SUBPACKAGE_DEPENDENCIES = {
    "pi05": ("rtc",),
    "smolvla": ("rtc",),
}

_LEROBOT_FACTORY_CONFIG_SUBPACKAGES = (
    # LeRobot's factory imports config modules for every built-in policy. Keep
    # those parent packages as namespace shims so package __init__.py files do
    # not pull optional modeling dependencies for inactive policies.
    "groot",
    "pi05",
    "rtc",
)


def _make_namespace_package(name: str, package_path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(package_path / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_path)]
    module.__spec__ = spec
    return module


def resolve_lerobot_policies_path() -> Path | None:
    """Locate the installed `lerobot.policies` package directory."""
    lerobot_spec = importlib.util.find_spec("lerobot")
    search_locations = getattr(lerobot_spec, "submodule_search_locations", None)
    if not search_locations:
        return None

    for root in search_locations:
        policies_path = Path(root) / "policies"
        if policies_path.is_dir():
            return policies_path
    return None


def install_lerobot_policies_namespace_shim() -> Path | None:
    """Replace `lerobot.policies` with a namespace package when possible.

    LeRobot's default `lerobot.policies` package eagerly imports every policy
    submodule during package initialisation. Treating it as a namespace package
    lets us import only the concrete submodules needed by the active policy.
    """
    policies_path = resolve_lerobot_policies_path()
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


def install_lerobot_policy_subpackage_shim_by_name(package_name: str) -> None:
    policies_path = install_lerobot_policies_namespace_shim()
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


def _remap_pi05_vision_tower_state_dict_keys(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Bridge PI05 checkpoints saved with HF's nested SigLIP vision module key."""
    fixed_state_dict: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key.replace("vision_tower.vision_model.", "vision_tower.")
        if new_key != key and new_key in fixed_state_dict:
            logging.warning(
                "Skipping PI05 checkpoint key %s because remapped key %s already exists.",
                key,
                new_key,
            )
            continue
        fixed_state_dict[new_key] = value
    return fixed_state_dict


class _SuppressPI05VisionEmbeddingWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(
            "Vision embedding key might need handling:"
        )


def install_pi05_checkpoint_compatibility_patch() -> None:
    """Patch LeRobot 0.5.x PI05 checkpoint loading for current pi05_base keys."""
    try:
        modeling_pi05 = importlib.import_module("lerobot.policies.pi05.modeling_pi05")
    except ModuleNotFoundError as exc:
        if exc.name != "lerobot.policies.pi05.modeling_pi05":
            raise
        return

    policy_cls = getattr(modeling_pi05, "PI05Policy", None)
    if policy_cls is None or getattr(
        policy_cls, "_corl_pi05_checkpoint_compatibility_patch", False
    ):
        return

    original_fix_keys = getattr(policy_cls, "_fix_pytorch_state_dict_keys", None)
    if original_fix_keys is None:
        return

    def _fix_pytorch_state_dict_keys_with_pi05_compat(self, state_dict, model_config):
        state_dict = _remap_pi05_vision_tower_state_dict_keys(dict(state_dict))
        root_logger = logging.getLogger()
        warning_filter = _SuppressPI05VisionEmbeddingWarning()
        root_logger.addFilter(warning_filter)
        try:
            fixed_state_dict = original_fix_keys(self, state_dict, model_config)
        finally:
            root_logger.removeFilter(warning_filter)
        return _remap_pi05_vision_tower_state_dict_keys(dict(fixed_state_dict))

    policy_cls._fix_pytorch_state_dict_keys = (
        _fix_pytorch_state_dict_keys_with_pi05_compat
    )
    policy_cls._corl_pi05_checkpoint_compatibility_patch = True


def ensure_lerobot_policy_imports(policy_type: str) -> None:
    """Install namespace shims for the active policy's LeRobot imports."""
    install_lerobot_policies_namespace_shim()
    for package_name in _LEROBOT_FACTORY_CONFIG_SUBPACKAGES:
        install_lerobot_policy_subpackage_shim_by_name(package_name)
    package_name = _LEROBOT_POLICY_SUBPACKAGES_BY_TYPE.get(policy_type)
    if package_name is not None:
        install_lerobot_policy_subpackage_shim_by_name(package_name)
    for dependency_name in _LEROBOT_POLICY_SUBPACKAGE_DEPENDENCIES.get(
        policy_type, ()
    ):
        install_lerobot_policy_subpackage_shim_by_name(dependency_name)

    # SmolVLA's pretrained processor pipeline references a custom registry step
    # defined in its processor module. The namespace shim bypasses package
    # __init__ imports, so load the module explicitly to register that step.
    if policy_type == "smolvla":
        importlib.import_module("lerobot.policies.smolvla.processor_smolvla")
    elif policy_type == "pi05":
        # PI05's pretrained processor pipeline and pi05_base checkpoint both
        # need side effects that would normally happen through package imports.
        importlib.import_module("lerobot.policies.pi05.processor_pi05")
        install_pi05_checkpoint_compatibility_patch()


def import_lerobot_policy_submodule(policy_type: str, module_name: str):
    ensure_lerobot_policy_imports(policy_type)
    return importlib.import_module(module_name)


def import_lerobot_policy_config_class(policy_type: str):
    """Import the concrete LeRobot policy config class for one policy type."""
    ensure_lerobot_policy_imports(policy_type)
    if policy_type == "act":
        from lerobot.policies.act.configuration_act import ACTConfig

        return ACTConfig
    if policy_type == "diffusion":
        from lerobot.policies.diffusion.configuration_diffusion import (
            DiffusionConfig,
        )

        return DiffusionConfig
    if policy_type == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        return PI05Config
    if policy_type == "smolvla":
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        return SmolVLAConfig
    raise ValueError(f"Unsupported LeRobot policy type: {policy_type!r}")


def make_standard_pre_post_processors(
    config: Any,
    *,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
):
    """Build the default LeRobot pre/post processor pair for a policy config."""
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        NormalizerProcessorStep,
        PolicyAction,
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    preprocessor_overrides = dict(preprocessor_overrides or {})
    postprocessor_overrides = dict(postprocessor_overrides or {})
    rename_map = dict(
        preprocessor_overrides.get("rename_observations_processor", {}).get(
            "rename_map",
            {},
        )
        or {}
    )
    input_device = preprocessor_overrides.get("device_processor", {}).get(
        "device",
        config.device,
    )
    output_device = postprocessor_overrides.get("device_processor", {}).get(
        "device",
        "cpu",
    )

    pre_normalized_keys = set(getattr(config, "pre_normalized_observation_keys", ()))
    normalize_observation_keys = {
        key for key in config.input_features if key not in pre_normalized_keys
    }

    input_steps = [
        RenameObservationsProcessorStep(rename_map=rename_map),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=input_device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=input_device,
            normalize_observation_keys=normalize_observation_keys,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device=output_device),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


__all__ = [
    "ensure_lerobot_policy_imports",
    "import_lerobot_policy_submodule",
    "import_lerobot_policy_config_class",
    "install_lerobot_policies_namespace_shim",
    "install_lerobot_policy_subpackage_shim_by_name",
    "install_pi05_checkpoint_compatibility_patch",
    "make_standard_pre_post_processors",
    "resolve_lerobot_policies_path",
]
