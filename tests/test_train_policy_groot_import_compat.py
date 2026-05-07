from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from train_policy import _import_groot_n1_with_kw_only_dataclass_compatibility
from train_policy import install_groot_action_input_batch_feature_compatibility_patch
from train_policy import _remap_groot_legacy_vision_model_state_dict_keys


def test_import_groot_n1_with_kw_only_dataclass_compatibility_recovers_from_bug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "lerobot"
    groot_root = package_root / "policies" / "groot"
    groot_root.mkdir(parents=True)

    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "policies" / "__init__.py").write_text("", encoding="utf-8")
    (groot_root / "__init__.py").write_text("", encoding="utf-8")
    (groot_root / "groot_n1.py").write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class GR00TN15Config:\n"
        '    compute_dtype: str = "float32"\n'
        "    backbone_cfg: dict\n",
        encoding="utf-8",
    )

    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "lerobot" or name.startswith("lerobot.")
    }
    for name in original_modules:
        sys.modules.pop(name, None)

    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = _import_groot_n1_with_kw_only_dataclass_compatibility()
        cfg = module.GR00TN15Config(backbone_cfg={"hidden_size": 1})
        assert cfg.compute_dtype == "float32"
        assert cfg.backbone_cfg == {"hidden_size": 1}
    finally:
        for name in list(sys.modules):
            if name == "lerobot" or name.startswith("lerobot."):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)


def test_install_groot_action_input_batch_feature_compatibility_patch_wraps_dicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "lerobot"
    groot_root = package_root / "policies" / "groot"
    groot_root.mkdir(parents=True)

    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "policies" / "__init__.py").write_text("", encoding="utf-8")
    (groot_root / "__init__.py").write_text("", encoding="utf-8")
    (groot_root / "groot_n1.py").write_text(
        "class GR00TN15:\n"
        "    def prepare_input(self, inputs):\n"
        "        return {'backbone_inputs': inputs}, {'embodiment_id': 3, 'token': 'ok'}\n",
        encoding="utf-8",
    )

    class FakeBatchFeature(dict):
        def __init__(self, data=None, **kwargs):
            super().__init__(data or {}, **kwargs)

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    fake_transformers = ModuleType("transformers")
    fake_transformers.BatchFeature = FakeBatchFeature

    original_lerobot_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "lerobot" or name.startswith("lerobot.")
    }
    original_transformers_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "transformers" or name.startswith("transformers.")
    }
    for name in original_lerobot_modules:
        sys.modules.pop(name, None)
    for name in original_transformers_modules:
        sys.modules.pop(name, None)

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    try:
        module = _import_groot_n1_with_kw_only_dataclass_compatibility()
        install_groot_action_input_batch_feature_compatibility_patch()
        model = module.GR00TN15()
        backbone_inputs, action_inputs = model.prepare_input({"image": "pixels"})

        assert backbone_inputs == {"backbone_inputs": {"image": "pixels"}}
        assert type(action_inputs) is FakeBatchFeature
        assert action_inputs.embodiment_id == 3
        assert action_inputs["token"] == "ok"
    finally:
        for name in list(sys.modules):
            if name == "lerobot" or name.startswith("lerobot."):
                sys.modules.pop(name, None)
            if name == "transformers" or name.startswith("transformers."):
                sys.modules.pop(name, None)
        sys.modules.update(original_lerobot_modules)
        sys.modules.update(original_transformers_modules)


def test_remap_groot_legacy_vision_model_state_dict_keys_prefers_canonical_keys(
) -> None:
    canonical_value = object()
    legacy_value = object()
    other_value = object()
    state_dict = {
        "_groot_model.backbone.eagle_model.vision_model.embeddings.patch_embedding.weight": canonical_value,
        "_groot_model.backbone.eagle_model.vision_model.vision_model.embeddings.patch_embedding.weight": legacy_value,
        "something_else": other_value,
    }

    remapped_state_dict, changed = _remap_groot_legacy_vision_model_state_dict_keys(
        state_dict
    )

    assert changed is True
    assert (
        remapped_state_dict[
            "_groot_model.backbone.eagle_model.vision_model.embeddings.patch_embedding.weight"
        ]
        is canonical_value
    )
    assert (
        "_groot_model.backbone.eagle_model.vision_model.vision_model.embeddings.patch_embedding.weight"
        not in remapped_state_dict
    )
    assert remapped_state_dict["something_else"] is other_value


def test_main_installs_groot_compatibility_before_importing_lerobot_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import train_policy

    events: list[str] = []
    patch_called = False

    def fake_install_groot_transformers_loading_compatibility_patch() -> None:
        nonlocal patch_called
        patch_called = True
        events.append("patch")

    monkeypatch.setattr(
        train_policy,
        "install_groot_transformers_loading_compatibility_patch",
        fake_install_groot_transformers_loading_compatibility_patch,
    )

    def register_module(name: str, *, package: bool = False, **attrs) -> ModuleType:
        module = ModuleType(name)
        if package:
            module.__path__ = []  # type: ignore[attr-defined]
        for attr_name, attr_value in attrs.items():
            setattr(module, attr_name, attr_value)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child_name, module)
        return module

    register_module("lerobot", package=True)
    register_module("lerobot.configs", package=True)
    register_module(
        "lerobot.configs.default",
        DatasetConfig=type("DatasetConfig", (), {}),
        WandBConfig=type("WandBConfig", (), {}),
    )
    register_module(
        "lerobot.configs.train",
        TrainPipelineConfig=type("TrainPipelineConfig", (), {}),
    )
    register_module("lerobot.scripts", package=True)
    register_module("lerobot.scripts.lerobot_train", train=object())
    register_module("lerobot.policies", package=True)
    register_module("lerobot.policies.act", package=True)
    register_module(
        "lerobot.policies.act.configuration_act",
        ACTConfig=type("ACTConfig", (), {}),
    )
    register_module("lerobot.policies.diffusion", package=True)
    register_module(
        "lerobot.policies.diffusion.configuration_diffusion",
        DiffusionConfig=type("DiffusionConfig", (), {}),
    )
    register_module("lerobot.policies.pi05", package=True)
    register_module(
        "lerobot.policies.pi05.configuration_pi05",
        PI05Config=type("PI05Config", (), {}),
    )

    class StopAfterImports(RuntimeError):
        pass

    monkeypatch.setattr(
        train_policy,
        "resolve_training_dataset_root",
        lambda *args, **kwargs: (_ for _ in ()).throw(StopAfterImports()),
    )

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.split(".", 1)[0] == "lerobot" and not patch_called:
            raise AssertionError(f"lerobot imported before Groot patch: {name}")
        if level == 0 and name.split(".", 1)[0] == "lerobot":
            events.append(f"import:{name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(StopAfterImports):
        train_policy.main(
            [
                "--dataset",
                "zeno-ai/BookOriginRelocation",
                "--policy",
                "pi05",
            ]
        )

    assert patch_called is True
    assert events[0] == "patch"
    assert any(event.startswith("import:lerobot") for event in events[1:])
