from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULTS_ROOT = Path(__file__).resolve().parents[1] / "bash" / "defaults"


def defaults_file_path(env_name: str, policy_name: str) -> Path:
    return DEFAULTS_ROOT / env_name / f"{policy_name}.yaml"


def load_policy_mode_defaults(
    mode: str,
    env_name: str,
    policy_name: str,
) -> dict[str, Any]:
    path = defaults_file_path(env_name=env_name, policy_name=policy_name)
    if not path.exists():
        raise FileNotFoundError(
            "Policy defaults file not found:\n"
            f"- mode={mode}\n"
            f"- env={env_name}\n"
            f"- policy={policy_name}\n"
            f"- path={path}"
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at top level in defaults file: {path}")

    mode_defaults = data.get(mode, {})
    if mode_defaults is None:
        mode_defaults = {}
    if not isinstance(mode_defaults, dict):
        raise TypeError(
            f"Expected mapping for '{mode}' section in defaults file: {path}"
        )
    return mode_defaults
