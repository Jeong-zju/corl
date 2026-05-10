from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from policy_imports import ensure_lerobot_policy_imports


def _clear_lerobot_modules() -> None:
    for module_name in tuple(sys.modules):
        if module_name == "lerobot" or module_name.startswith("lerobot."):
            sys.modules.pop(module_name, None)


def test_factory_config_namespace_shim_does_not_execute_inactive_package_init(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_root = tmp_path / "lerobot" / "policies"
    (package_root / "groot").mkdir(parents=True)
    (package_root / "pi05").mkdir(parents=True)
    (package_root / "rtc").mkdir(parents=True)
    (package_root / "xvla").mkdir(parents=True)
    (tmp_path / "lerobot" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "groot" / "__init__.py").write_text(
        "raise RuntimeError('groot package init imported')\n",
        encoding="utf-8",
    )
    (package_root / "groot" / "configuration_groot.py").write_text(
        "VALUE = 'groot_config'\n",
        encoding="utf-8",
    )
    (package_root / "pi05" / "__init__.py").write_text(
        "raise RuntimeError('pi05 package init imported')\n",
        encoding="utf-8",
    )
    (package_root / "pi05" / "configuration_pi05.py").write_text(
        "VALUE = 'pi05_config'\n",
        encoding="utf-8",
    )
    (package_root / "rtc" / "__init__.py").write_text(
        "raise RuntimeError('rtc package init imported')\n",
        encoding="utf-8",
    )
    (package_root / "rtc" / "configuration_rtc.py").write_text(
        "VALUE = 'rtc_config'\n",
        encoding="utf-8",
    )
    (package_root / "xvla" / "__init__.py").write_text(
        "raise RuntimeError('xvla package init imported')\n",
        encoding="utf-8",
    )
    (package_root / "xvla" / "configuration_xvla.py").write_text(
        "VALUE = 'xvla_config'\n",
        encoding="utf-8",
    )

    _clear_lerobot_modules()
    monkeypatch.syspath_prepend(str(tmp_path))

    try:
        ensure_lerobot_policy_imports("act")

        groot_config = importlib.import_module(
            "lerobot.policies.groot.configuration_groot"
        )
        pi05_config = importlib.import_module(
            "lerobot.policies.pi05.configuration_pi05"
        )
        rtc_config = importlib.import_module("lerobot.policies.rtc.configuration_rtc")
        xvla_config = importlib.import_module(
            "lerobot.policies.xvla.configuration_xvla"
        )

        assert groot_config.VALUE == "groot_config"
        assert pi05_config.VALUE == "pi05_config"
        assert rtc_config.VALUE == "rtc_config"
        assert xvla_config.VALUE == "xvla_config"
        assert "lerobot.policies.xvla.processor_xvla" not in sys.modules
        assert "lerobot.policies.xvla.modeling_xvla" not in sys.modules
    finally:
        _clear_lerobot_modules()


def test_pi05_imports_register_processor_and_patch_checkpoint_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "lerobot" / "policies"
    (package_root / "groot").mkdir(parents=True)
    (package_root / "pi05").mkdir(parents=True)
    (package_root / "rtc").mkdir(parents=True)
    (tmp_path / "lerobot" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "lerobot" / "processor.py").write_text(
        """
class ProcessorStepRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(step_cls):
            cls._registry[name] = step_cls
            return step_cls
        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry[name]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "groot" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "pi05" / "__init__.py").write_text(
        "raise RuntimeError('pi05 package init imported')\n",
        encoding="utf-8",
    )
    (package_root / "pi05" / "processor_pi05.py").write_text(
        """
from lerobot.processor import ProcessorStepRegistry

@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
class Pi05PrepareStateTokenizerProcessorStep:
    pass
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (package_root / "pi05" / "modeling_pi05.py").write_text(
        """
import logging

class PI05Policy:
    def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
        for key in state_dict:
            if "patch_embedding" in key:
                logging.warning("Vision embedding key might need handling: %s", key)
        return dict(state_dict)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (package_root / "rtc" / "__init__.py").write_text("", encoding="utf-8")

    _clear_lerobot_modules()
    monkeypatch.syspath_prepend(str(tmp_path))

    try:
        ensure_lerobot_policy_imports("pi05")

        from lerobot.processor import ProcessorStepRegistry
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        assert (
            ProcessorStepRegistry.get(
                "pi05_prepare_state_tokenizer_processor_step"
            ).__name__
            == "Pi05PrepareStateTokenizerProcessorStep"
        )
        assert PI05Policy._corl_pi05_checkpoint_compatibility_patch is True

        old_key = (
            "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
            "embeddings.patch_embedding.weight"
        )
        fixed_state_dict = PI05Policy()._fix_pytorch_state_dict_keys(
            {old_key: object()},
            model_config=None,
        )

        assert old_key not in fixed_state_dict
        assert (
            "paligemma_with_expert.paligemma.model.vision_tower."
            "embeddings.patch_embedding.weight"
            in fixed_state_dict
        )
    finally:
        _clear_lerobot_modules()


def test_xvla_imports_register_processor_without_model_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "lerobot" / "policies"
    (package_root / "groot").mkdir(parents=True)
    (package_root / "pi05").mkdir(parents=True)
    (package_root / "rtc").mkdir(parents=True)
    (package_root / "xvla").mkdir(parents=True)
    (tmp_path / "lerobot" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "lerobot" / "processor.py").write_text(
        """
class ProcessorStepRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(step_cls):
            cls._registry[name] = step_cls
            return step_cls
        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry[name]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "groot" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "pi05" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "rtc" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "xvla" / "__init__.py").write_text(
        "raise RuntimeError('xvla package init imported')\n",
        encoding="utf-8",
    )
    (package_root / "xvla" / "processor_xvla.py").write_text(
        """
from lerobot.processor import ProcessorStepRegistry

@ProcessorStepRegistry.register(name="xvla_image_to_float")
class XVLAImageToFloatProcessorStep:
    pass
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (package_root / "xvla" / "modeling_xvla.py").write_text(
        "raise RuntimeError('xvla model imported')\n",
        encoding="utf-8",
    )

    _clear_lerobot_modules()
    monkeypatch.syspath_prepend(str(tmp_path))

    try:
        ensure_lerobot_policy_imports("xvla")

        from lerobot.processor import ProcessorStepRegistry

        assert (
            ProcessorStepRegistry.get("xvla_image_to_float").__name__
            == "XVLAImageToFloatProcessorStep"
        )
        assert "lerobot.policies.xvla.modeling_xvla" not in sys.modules
    finally:
        _clear_lerobot_modules()
