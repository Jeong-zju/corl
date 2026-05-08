from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from policy_imports import ensure_lerobot_policy_imports


def test_factory_config_namespace_shim_does_not_execute_inactive_package_init(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_root = tmp_path / "lerobot" / "policies"
    (package_root / "groot").mkdir(parents=True)
    (package_root / "pi05").mkdir(parents=True)
    (package_root / "rtc").mkdir(parents=True)
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

    for module_name in tuple(sys.modules):
        if module_name == "lerobot" or module_name.startswith("lerobot."):
            sys.modules.pop(module_name, None)
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

        assert groot_config.VALUE == "groot_config"
        assert pi05_config.VALUE == "pi05_config"
        assert rtc_config.VALUE == "rtc_config"
    finally:
        for module_name in tuple(sys.modules):
            if module_name == "lerobot" or module_name.startswith("lerobot."):
                sys.modules.pop(module_name, None)
