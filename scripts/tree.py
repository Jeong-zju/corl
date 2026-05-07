"""Minimal `tree` compatibility shim for GR00T deploy and training.

LeRobot's GR00T policy imports the optional third-party `tree` package and
uses only `map_structure`. Our project environment does not depend on the
external package, so we provide a small local fallback that covers the nested
containers used by the deploy/runtime code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable


def _map_structure_leaf(func: Callable[..., Any], *structures: Any) -> Any:
    return func(*structures)


def map_structure(func: Callable[..., Any], *structures: Any) -> Any:
    """Apply `func` to matching leaves across one or more nested structures."""

    if not structures:
        raise TypeError("map_structure requires at least one structure.")

    first = structures[0]

    if isinstance(first, Mapping):
        keys = list(first.keys())
        for other in structures[1:]:
            if not isinstance(other, Mapping):
                raise TypeError("All structures must have the same nested layout.")
            if list(other.keys()) != keys:
                raise ValueError("All mapping structures must share the same keys.")
        return {
            key: map_structure(func, *(structure[key] for structure in structures))
            for key in keys
        }

    if isinstance(first, list):
        length = len(first)
        for other in structures[1:]:
            if not isinstance(other, list) or len(other) != length:
                raise ValueError("All list structures must have the same length.")
        return [
            map_structure(func, *(structure[index] for structure in structures))
            for index in range(length)
        ]

    if isinstance(first, tuple):
        length = len(first)
        for other in structures[1:]:
            if not isinstance(other, tuple) or len(other) != length:
                raise ValueError("All tuple structures must have the same length.")
        mapped = [
            map_structure(func, *(structure[index] for structure in structures))
            for index in range(length)
        ]
        if hasattr(first, "_fields"):
            try:
                return type(first)(*mapped)
            except Exception:
                return tuple(mapped)
        return tuple(mapped)

    return _map_structure_leaf(func, *structures)


__all__ = ["map_structure"]
