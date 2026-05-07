from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


_LEROBOT_POLICY_SUBPACKAGES_BY_TYPE = {
    "act": "act",
    "diffusion": "diffusion",
    "smolvla": "smolvla",
}

_LEROBOT_POLICY_SUBPACKAGE_DEPENDENCIES = {
    "smolvla": ("rtc",),
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


def ensure_lerobot_policy_imports(policy_type: str) -> None:
    """Install namespace shims for the active policy's LeRobot imports."""
    install_lerobot_policies_namespace_shim()
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
    "make_standard_pre_post_processors",
    "resolve_lerobot_policies_path",
]
