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
from train_policy import (
    DELTA_SIGNATURE_FEATURE_KEY,
    GROOT_DEFAULT_ATTN_IMPLEMENTATION,
    PATH_SIGNATURE_FEATURE_KEY,
    _ensure_groot_transformers_loading_attrs,
    drop_dataset_features_from_metadata,
    parse_args,
    resolve_training_dataset_root,
)


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


def test_drop_dataset_features_from_metadata_can_exclude_signature_inputs() -> None:
    info = {
        "features": {
            "observation.images.front": {"dtype": "image", "shape": [3, 224, 224]},
            "observation.state": {"dtype": "float32", "shape": [7]},
            PATH_SIGNATURE_FEATURE_KEY: {"dtype": "float32", "shape": [64]},
            DELTA_SIGNATURE_FEATURE_KEY: {"dtype": "float32", "shape": [64]},
            "action": {"dtype": "float32", "shape": [7]},
        },
        "path_signature": {"key": PATH_SIGNATURE_FEATURE_KEY},
        "delta_signature": {"key": DELTA_SIGNATURE_FEATURE_KEY},
    }
    stats = {
        "observation.state": {"mean": [0.0]},
        PATH_SIGNATURE_FEATURE_KEY: {"mean": [0.0]},
        DELTA_SIGNATURE_FEATURE_KEY: {"mean": [0.0]},
    }

    updated_info, updated_stats, removed_keys = drop_dataset_features_from_metadata(
        info,
        stats,
        feature_keys=(
            PATH_SIGNATURE_FEATURE_KEY,
            DELTA_SIGNATURE_FEATURE_KEY,
        ),
    )

    assert removed_keys == (DELTA_SIGNATURE_FEATURE_KEY, PATH_SIGNATURE_FEATURE_KEY)
    assert PATH_SIGNATURE_FEATURE_KEY not in updated_info["features"]
    assert DELTA_SIGNATURE_FEATURE_KEY not in updated_info["features"]
    assert PATH_SIGNATURE_FEATURE_KEY not in updated_stats
    assert DELTA_SIGNATURE_FEATURE_KEY not in updated_stats
    assert "path_signature" not in updated_info
    assert "delta_signature" not in updated_info


def test_ensure_groot_transformers_loading_attrs_adds_missing_post_init_attrs() -> None:
    class DummyGroot:
        def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
            assert all_submodels is False
            return {}

    model = DummyGroot()

    _ensure_groot_transformers_loading_attrs(model)

    assert model.all_tied_weights_keys == {}
    assert model._tp_plan == {}
    assert model._ep_plan == {}
    assert model._pp_plan == {}
    assert model._keep_in_fp32_modules == set()
    assert model._keep_in_fp32_modules_strict == set()
    assert model._no_split_modules == set()


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
        (
            "smolvla",
            "main/bash/defaults/robocasa/atomic/CloseFridge/smolvla.yaml",
            "robocasa/atomic/CloseFridge/smolvla",
        ),
        (
            "groot",
            "main/bash/defaults/robocasa/atomic/CloseFridge/groot.yaml",
            "robocasa/atomic/CloseFridge/groot",
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


def test_train_parse_args_uses_smolvla_defaults() -> None:
    args = parse_args(
        [
            "--dataset",
            "robocasa/atomic/CloseFridge",
            "--policy",
            "smolvla",
        ]
    )

    assert args._policy_defaults_dataset_root == "data/robocasa/atomic/CloseFridge"
    assert args._policy_defaults_dataset_repo_id == "robocasa/atomic/CloseFridge"
    assert args.output_root.as_posix() == (
        "outputs/train/robocasa/atomic/CloseFridge/smolvla"
    )
    assert args.policy_path == "lerobot/smolvla_base"
    assert args.n_obs_steps == 1
    assert args.chunk_size == 50
    assert args.n_action_steps == 50
    assert args.smolvla_freeze_vision_encoder is True


def test_train_parse_args_uses_groot_defaults_and_ignores_signature() -> None:
    args = parse_args(
        [
            "--dataset",
            "robocasa/atomic/CloseFridge",
            "--policy",
            "groot",
        ]
    )

    assert args._policy_defaults_dataset_root == "data/robocasa/atomic/CloseFridge"
    assert args._policy_defaults_dataset_repo_id == "robocasa/atomic/CloseFridge"
    assert args.output_root.as_posix() == (
        "outputs/train/robocasa/atomic/CloseFridge/groot"
    )
    assert args.policy_path is None
    assert args.groot_base_model_path == "nvidia/GR00T-N1.5-3B"
    assert args.groot_tokenizer_assets_repo == "lerobot/eagle2hg-processor-groot-n1p5"
    assert args.groot_attn_implementation == GROOT_DEFAULT_ATTN_IMPLEMENTATION
    assert args.n_obs_steps == 1
    assert args.chunk_size == 50
    assert args.n_action_steps == 50
    assert args.groot_ignore_signature_features is True


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


@pytest.mark.parametrize(
    ("policy_name", "expected_output_suffix"),
    (
        ("act", "robocasa/composite/OrganizeVegetables/act"),
        ("diffusion", "robocasa/composite/OrganizeVegetables/diffusion"),
        ("streaming_act", "robocasa/composite/OrganizeVegetables/streaming-act-prism"),
        ("smolvla", "robocasa/composite/OrganizeVegetables/smolvla"),
        ("groot", "robocasa/composite/OrganizeVegetables/groot"),
    ),
)
def test_resolve_dataset_defaults_path_supports_organize_vegetables_policy_defaults(
    policy_name: str,
    expected_output_suffix: str,
) -> None:
    defaults, defaults_path = load_policy_mode_defaults_for_dataset(
        mode="train",
        dataset_selector="robocasa/composite/OrganizeVegetables",
        policy_name=policy_name,
    )

    assert defaults_path is not None
    assert defaults_path.as_posix().endswith(
        f"main/bash/defaults/robocasa/composite/OrganizeVegetables/{policy_name}.yaml"
    )
    assert defaults["dataset_root"] == "data/robocasa/composite/OrganizeVegetables"
    assert defaults["dataset_repo_id"] == "robocasa/composite/OrganizeVegetables"
    assert defaults["output_root"].endswith(expected_output_suffix)


@pytest.mark.parametrize(
    "task_name",
    (
        "OrganizeVegetables",
        "PackFoodByTemp",
        "StoreLeftoversByType",
        "BeverageSorting",
        "PackIdenticalLunches",
        "CreateChildFriendlyFridge",
        "LoadCondimentsInFridge",
    ),
)
def test_resolve_dataset_defaults_path_supports_task_specific_composite_memory_defaults(
    task_name: str,
) -> None:
    defaults, defaults_path = load_policy_mode_defaults_for_dataset(
        mode="train",
        dataset_selector=f"robocasa/composite/{task_name}",
        policy_name="streaming_act",
    )

    assert defaults_path is not None
    assert defaults_path.as_posix().endswith(
        f"main/bash/defaults/robocasa/composite/{task_name}/streaming_act.yaml"
    )
    assert defaults["dataset_root"] == f"data/robocasa/composite/{task_name}"
    assert defaults["dataset_repo_id"] == f"robocasa/composite/{task_name}"
    assert defaults["use_visual_prefix_memory"] is True
    assert defaults["use_signature_indexed_slot_memory"] is True
    assert defaults["use_memory_conditioned_encoder_film"] is True
    assert defaults["signature_cache_mode"] == "ram"


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
