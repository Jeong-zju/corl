from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


DEPLOY_ROOT = Path(__file__).resolve().parent
MAIN_ROOT = DEPLOY_ROOT.parent
REPO_ROOT = MAIN_ROOT.parent
SCRIPTS_ROOT = MAIN_ROOT / "scripts"


def ensure_runtime_paths() -> None:
    for path in (DEPLOY_ROOT, SCRIPTS_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(data).__name__}.")
    return data


def resolve_path(
    raw_path: str | None,
    *,
    config_path: Path | None = None,
    must_exist: bool = False,
) -> Path | None:
    if raw_path is None:
        return None

    text = str(raw_path).strip()
    if not text:
        return None

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        base_dir = config_path.parent if config_path is not None else Path.cwd()
        resolved = (base_dir / candidate).resolve(strict=False)

    if resolved.exists():
        return resolved
    if must_exist:
        raise FileNotFoundError(
            "Could not resolve existing path for "
            f"{raw_path!r}. Resolved to: {resolved}"
        )
    return resolved
