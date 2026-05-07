from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))


def test_tree_map_structure_fallback_handles_nested_containers() -> None:
    sys.modules.pop("tree", None)
    tree = importlib.import_module("tree")

    assert Path(tree.__file__).resolve() == REPO_ROOT / "main" / "scripts" / "tree.py"

    result = tree.map_structure(
        lambda x: x + 1,
        {"a": [1, 2], "b": (3, {"c": 4})},
    )

    assert result == {"a": [2, 3], "b": (4, {"c": 5})}


def test_tree_map_structure_fallback_supports_multiple_structures() -> None:
    sys.modules.pop("tree", None)
    tree = importlib.import_module("tree")

    result = tree.map_structure(
        lambda left, right: left + right,
        {"a": [1, 2], "b": (3,)},
        {"a": [10, 20], "b": (30,)},
    )

    assert result == {"a": [11, 22], "b": (33,)}
