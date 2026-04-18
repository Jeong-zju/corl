from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from policy_defaults import (
    load_policy_mode_defaults_for_cli,
    load_policy_mode_defaults_for_dataset,
    resolve_cli_dataset_defaults_path,
    resolve_dataset_defaults_path,
)
from train_policy import parse_args, resolve_training_dataset_root


def _make_fake_lerobot_dataset_root(dataset_root: Path) -> Path:
    (dataset_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0"}),
        encoding="utf-8",
    )
    (dataset_root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
    (dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_text(
        "",
        encoding="utf-8",
    )
    return dataset_root


def test_resolve_dataset_defaults_path_prefers_exact_robocasa_task_defaults() -> None:
    path = resolve_dataset_defaults_path(
        dataset_selector="robocasa/composite/ArrangeBreadBasket",
        policy_name="act",
    )

    assert path is not None
    assert path.as_posix().endswith(
        "main/bash/defaults/robocasa/composite/ArrangeBreadBasket/act.yaml"
    )

    defaults, defaults_path = load_policy_mode_defaults_for_dataset(
        mode="train",
        dataset_selector="robocasa/composite/ArrangeBreadBasket",
        policy_name="act",
    )

    assert defaults_path == path
    assert defaults["dataset_root"] == "data/robocasa/composite/ArrangeBreadBasket"
    assert defaults["dataset_repo_id"] == "robocasa/composite/ArrangeBreadBasket"


@pytest.mark.parametrize(
    ("policy_name", "expected_defaults_path", "expected_output_suffix"),
    (
        (
            "diffusion",
            "main/bash/defaults/robocasa/atomic/CloseFridge/diffusion.yaml",
            "robocasa/atomic/CloseFridge/diffusion",
        ),
        (
            "prism_diffusion",
            "main/bash/defaults/robocasa/atomic/CloseFridge/prism_diffusion.yaml",
            "robocasa/atomic/CloseFridge/prism-diffusion",
        ),
    ),
)
def test_resolve_dataset_defaults_path_supports_close_fridge_diffusion_variants(
    policy_name: str,
    expected_defaults_path: str,
    expected_output_suffix: str,
) -> None:
    defaults, defaults_path = load_policy_mode_defaults_for_dataset(
        mode="train",
        dataset_selector="robocasa/atomic/CloseFridge",
        policy_name=policy_name,
    )

    assert defaults_path is not None
    assert defaults_path.as_posix().endswith(expected_defaults_path)
    assert defaults["dataset_root"] == "data/robocasa/atomic/CloseFridge"
    assert defaults["dataset_repo_id"] == "robocasa/atomic/CloseFridge"
    assert defaults["output_root"].endswith(expected_output_suffix)


def test_close_fridge_diffusion_eval_defaults_enable_robocasa_horizon_inference() -> None:
    defaults, defaults_path = load_policy_mode_defaults_for_dataset(
        mode="eval",
        dataset_selector="robocasa/atomic/CloseFridge",
        policy_name="diffusion",
    )

    assert defaults_path is not None
    assert defaults_path.as_posix().endswith(
        "main/bash/defaults/robocasa/atomic/CloseFridge/diffusion.yaml"
    )
    assert defaults["task"] == "CloseFridge"
    assert defaults["max_steps"] is None
    assert defaults["robocasa_conda_env"] == "robocasa"
    assert defaults["robocasa_split"] == "target"


def test_resolve_dataset_defaults_path_keeps_broad_robocasa_collection_defaults() -> None:
    defaults, defaults_path = load_policy_mode_defaults_for_dataset(
        mode="train",
        dataset_selector="robocasa/composite",
        policy_name="streaming_act",
    )

    assert defaults_path is not None
    assert defaults_path.as_posix().endswith(
        "main/bash/defaults/robocasa/composite/streaming_act.yaml"
    )
    assert defaults["dataset_root"] == "data/robocasa/composite"
    assert defaults["dataset_repo_id"] == "robocasa/composite"
    assert defaults["dataset_tasks"] == ["ArrangeBreadBasket"]


def test_resolve_dataset_defaults_path_keeps_broad_robocasa_atomic_defaults() -> None:
    defaults, defaults_path = load_policy_mode_defaults_for_dataset(
        mode="train",
        dataset_selector="robocasa/atomic",
        policy_name="streaming_act",
    )

    assert defaults_path is not None
    assert defaults_path.as_posix().endswith(
        "main/bash/defaults/robocasa/atomic/streaming_act.yaml"
    )
    assert defaults["dataset_root"] == "data/robocasa/atomic"
    assert defaults["dataset_repo_id"] == "robocasa/atomic"
    assert defaults["dataset_tasks"] == ["CloseFridge"]
    assert defaults["output_root"].endswith("robocasa/atomic/streaming-act-prism")


def test_resolve_cli_dataset_defaults_path_prefers_task_defaults_for_broad_robocasa_dataset() -> None:
    path = resolve_cli_dataset_defaults_path(
        dataset_selector="robocasa",
        task_selector="CloseFridge",
        policy_name="streaming_act",
    )

    assert path is not None
    assert path.as_posix().endswith(
        "main/bash/defaults/robocasa/atomic/CloseFridge/streaming_act.yaml"
    )

    defaults, defaults_path = load_policy_mode_defaults_for_cli(
        mode="train",
        dataset_selector="robocasa",
        task_selector="CloseFridge",
        policy_name="streaming_act",
    )

    assert defaults_path == path
    assert defaults["dataset_root"] == "data/robocasa/atomic/CloseFridge"
    assert defaults["dataset_repo_id"] == "robocasa/atomic/CloseFridge"
    assert defaults["signature_cache_mode"] == "ram"


def test_train_parse_args_uses_task_specific_streaming_act_defaults_with_broad_robocasa_dataset() -> None:
    args = parse_args(
        [
            "--dataset",
            "robocasa",
            "--task",
            "CloseFridge",
            "--policy",
            "streaming_act",
        ]
    )

    assert args.task == "CloseFridge"
    assert args._policy_defaults_dataset_root == "data/robocasa/atomic/CloseFridge"
    assert args._policy_defaults_dataset_repo_id == "robocasa/atomic/CloseFridge"
    assert args.output_root.as_posix() == (
        "outputs/train/robocasa/atomic/CloseFridge/streaming-act-prism"
    )
    assert args.signature_cache_mode == "ram"


def test_resolve_training_dataset_root_uses_exact_named_child_from_dataset_tasks(
    tmp_path: Path,
) -> None:
    local_data_root = tmp_path / "data"
    collection_root = local_data_root / "robocasa" / "composite"
    _make_fake_lerobot_dataset_root(collection_root / "ArrangeBreadBasket")
    _make_fake_lerobot_dataset_root(collection_root / "ArrangeBreadBasket_raw")
    _make_fake_lerobot_dataset_root(collection_root / "ArrangeBreadBasket_merge")

    resolved = resolve_training_dataset_root(
        dataset="robocasa/composite",
        defaults_dataset_root="data/robocasa/composite",
        local_data_root=local_data_root,
        exact_task_names=("ArrangeBreadBasket",),
    )

    assert resolved == (collection_root / "ArrangeBreadBasket").resolve()


def test_resolve_training_dataset_root_rejects_multiple_collection_tasks(
    tmp_path: Path,
) -> None:
    local_data_root = tmp_path / "data"
    collection_root = local_data_root / "robocasa" / "composite"
    _make_fake_lerobot_dataset_root(collection_root / "ArrangeBreadBasket")
    _make_fake_lerobot_dataset_root(collection_root / "PickPlaceCounterToSink")

    with pytest.raises(NotImplementedError, match="dataset_tasks"):
        resolve_training_dataset_root(
            dataset="robocasa/composite",
            defaults_dataset_root="data/robocasa/composite",
            local_data_root=local_data_root,
            exact_task_names=("ArrangeBreadBasket", "PickPlaceCounterToSink"),
        )
