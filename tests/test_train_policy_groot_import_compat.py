from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from train_policy import _import_groot_n1_with_kw_only_dataclass_compatibility


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
