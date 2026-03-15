from __future__ import annotations

from importlib import import_module


_ENV_MODULES = {
    "h_shape": "env.h_shape_env",
    "braidedhub": "env.braidedhub_env",
}


def get_env_choices() -> tuple[str, ...]:
    return tuple(_ENV_MODULES)


def get_env_module(env_name: str):
    try:
        module_name = _ENV_MODULES[env_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_ENV_MODULES))
        raise ValueError(
            f"Unsupported env={env_name!r}. Expected one of: {supported}"
        ) from exc
    return import_module(module_name)
