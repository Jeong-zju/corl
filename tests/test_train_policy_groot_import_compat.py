from __future__ import annotations

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
