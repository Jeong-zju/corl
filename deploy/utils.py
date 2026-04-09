from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def bootstrap_main_pythonpath(current_file: str | Path) -> Path:
    """Ensure `/.../main` is importable as the package root."""
    main_root = Path(current_file).resolve().parents[2]
    main_root_str = str(main_root)
    if main_root_str not in sys.path:
        sys.path.insert(0, main_root_str)
    return main_root


def repo_root_from_main(main_root: str | Path) -> Path:
    return Path(main_root).resolve().parent


def load_mapping_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    suffix = config_path.suffix.lower()
    raw = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        loaded = json.loads(raw)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config support requires `PyYAML`. Install it or use a JSON config."
            ) from exc
        loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {config_path} must load to a mapping.")
    return loaded


def nested_mapping_get(
    mapping: dict[str, Any],
    key: str,
    *,
    default: Any = None,
) -> Any:
    value = mapping.get(key, default)
    if value is None:
        return default
    if not isinstance(value, dict):
        raise ValueError(f"Expected `{key}` to be a mapping, got {type(value).__name__}.")
    return value

