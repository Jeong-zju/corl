from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "data" / "upload_dataset_to_hf.py"
SPEC = importlib.util.spec_from_file_location("upload_dataset_to_hf", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _make_fake_dataset_root(dataset_root: Path) -> Path:
    (dataset_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "zeno",
                "fps": 30,
                "total_episodes": 5,
                "total_frames": 1234,
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
    return dataset_root


def test_resolve_local_upload_path_supports_main_data_relative_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    main_root = repo_root / "main"
    data_root = main_root / "data"
    dataset_root = _make_fake_dataset_root(data_root / "zeno-ai" / "demo_dataset")

    resolved = MODULE.resolve_local_upload_path(
        "zeno-ai/demo_dataset",
        cwd=repo_root,
        main_root=main_root,
        data_root=data_root,
    )

    assert resolved == dataset_root.resolve()


def test_resolve_local_upload_path_supports_data_prefix(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    main_root = repo_root / "main"
    data_root = main_root / "data"
    dataset_root = _make_fake_dataset_root(data_root / "zeno-ai" / "demo_dataset")

    resolved = MODULE.resolve_local_upload_path(
        "data/zeno-ai/demo_dataset",
        cwd=repo_root,
        main_root=main_root,
        data_root=data_root,
    )

    assert resolved == dataset_root.resolve()


def test_resolve_local_upload_path_promotes_meta_info_to_dataset_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    main_root = repo_root / "main"
    data_root = main_root / "data"
    dataset_root = _make_fake_dataset_root(data_root / "zeno-ai" / "demo_dataset")

    resolved = MODULE.resolve_local_upload_path(
        str(dataset_root / "meta" / "info.json"),
        cwd=repo_root,
        main_root=main_root,
        data_root=data_root,
    )

    assert resolved == dataset_root.resolve()


def test_resolve_repo_id_infers_namespace_name_for_two_level_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "main" / "data"
    dataset_root = _make_fake_dataset_root(data_root / "zeno-ai" / "demo_dataset")

    repo_id = MODULE.resolve_repo_id(
        dataset_root,
        repo_id=None,
        repo_namespace=None,
        data_root=data_root,
    )

    assert repo_id == "zeno-ai/demo_dataset"


def test_resolve_repo_id_maps_robocasa_datasets_to_zeno_ai_by_default(tmp_path: Path) -> None:
    data_root = tmp_path / "main" / "data"
    dataset_root = _make_fake_dataset_root(data_root / "robocasa" / "composite" / "task_a")

    repo_id = MODULE.resolve_repo_id(
        dataset_root,
        repo_id=None,
        repo_namespace=None,
        data_root=data_root,
    )

    assert repo_id == "zeno-ai/robocasa-composite-task_a"


def test_resolve_repo_id_supports_explicit_repo_namespace_for_deeper_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "main" / "data"
    dataset_root = _make_fake_dataset_root(data_root / "robocasa" / "composite" / "task_a")

    repo_id = MODULE.resolve_repo_id(
        dataset_root,
        repo_id=None,
        repo_namespace="robocasa",
        data_root=data_root,
    )

    assert repo_id == "robocasa/composite-task_a"


def test_resolve_repo_id_rejects_repo_id_and_namespace_together(tmp_path: Path) -> None:
    data_root = tmp_path / "main" / "data"
    dataset_root = _make_fake_dataset_root(data_root / "robocasa" / "composite" / "task_a")

    with pytest.raises(ValueError, match="either `--repo-id` or `--repo-namespace`"):
        MODULE.resolve_repo_id(
            dataset_root,
            repo_id="zeno-ai/manual-repo",
            repo_namespace="zeno-ai",
            data_root=data_root,
        )


def test_validate_upload_options_rejects_large_folder_incompatible_flags() -> None:
    with pytest.raises(ValueError, match="does not support `--path-in-repo`"):
        MODULE.validate_upload_options(
            large_folder=True,
            path_in_repo="subdir",
            commit_message=None,
            commit_description=None,
            num_workers=None,
        )

    with pytest.raises(ValueError, match="positive integer"):
        MODULE.validate_upload_options(
            large_folder=False,
            path_in_repo=None,
            commit_message=None,
            commit_description=None,
            num_workers=0,
        )
