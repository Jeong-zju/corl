from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from eval_policy import configure_streaming_act_eval_runtime, parse_args


def test_parse_args_accepts_robocasa_split() -> None:
    args = parse_args(
        [
            "--env",
            "robocasa",
            "--policy",
            "act",
            "--robocasa-split",
            "all",
            "--task",
            "CloseFridge",
        ]
    )

    assert args.robocasa_split == "all"
    assert args.eval_robocasa_split == "all"


def test_parse_args_uses_task_specific_robocasa_defaults_without_dataset() -> None:
    args = parse_args(
        [
            "--env",
            "robocasa",
            "--policy",
            "act",
            "--task",
            "CloseFridge",
        ]
    )

    assert args.task == "CloseFridge"
    assert args.train_output_root.as_posix() == "outputs/train/robocasa/atomic/CloseFridge/act"
    assert args._policy_defaults_dataset_root == "data/robocasa/atomic/CloseFridge"


def test_parse_args_uses_task_specific_robocasa_defaults_with_broad_dataset() -> None:
    args = parse_args(
        [
            "--dataset",
            "robocasa",
            "--policy",
            "act",
            "--task",
            "CloseFridge",
        ]
    )

    assert args.task == "CloseFridge"
    assert args.train_output_root.as_posix() == "outputs/train/robocasa/atomic/CloseFridge/act"
    assert args._policy_defaults_dataset_root == "data/robocasa/atomic/CloseFridge"


def test_parse_args_infers_robocasa_task_from_leaf_dataset() -> None:
    args = parse_args(
        [
            "--env",
            "robocasa",
            "--policy",
            "act",
            "--dataset",
            "robocasa/composite/ArrangeBreadBasket",
        ]
    )

    assert args.task == "ArrangeBreadBasket"
    assert args.eval_task_spec == "ArrangeBreadBasket"
    assert args.eval_task_names == ("ArrangeBreadBasket",)


def test_parse_args_infers_robocasa_task_from_collection_defaults() -> None:
    args = parse_args(
        [
            "--env",
            "robocasa",
            "--policy",
            "streaming_act",
            "--dataset",
            "robocasa/composite",
        ]
    )

    assert args.task == "ArrangeBreadBasket"
    assert args.eval_task_spec == "ArrangeBreadBasket"
    assert args.eval_task_names == ("ArrangeBreadBasket",)


def test_parse_args_canonicalizes_task_csv() -> None:
    args = parse_args(
        [
            "--env",
            "robocasa",
            "--policy",
            "act",
            "--task",
            " ArrangeBreadBasket , ArrangeBreadBasket, PickPlaceCounterToSink ",
        ]
    )

    assert args.task == "ArrangeBreadBasket,PickPlaceCounterToSink"
    assert args.eval_task_spec == "ArrangeBreadBasket,PickPlaceCounterToSink"
    assert args.eval_task_names == (
        "ArrangeBreadBasket",
        "PickPlaceCounterToSink",
    )


def test_parse_args_robocasa_builtin_max_steps_falls_back_to_dataset_inference() -> None:
    args = parse_args(
        [
            "--env",
            "robocasa",
            "--policy",
            "diffusion",
        ]
    )

    assert args.max_steps is None
    assert args.eval_max_steps is None


def test_parse_args_robocasa_explicit_max_steps_is_preserved() -> None:
    args = parse_args(
        [
            "--env",
            "robocasa",
            "--policy",
            "diffusion",
            "--max-steps",
            "250",
        ]
    )

    assert args.max_steps == 250
    assert args.eval_max_steps == 250


def test_parse_args_streaming_act_loads_temporal_ensemble_default_from_defaults() -> None:
    args = parse_args(
        [
            "--env",
            "robocasa",
            "--policy",
            "streaming_act",
            "--task",
            "CloseFridge",
        ]
    )

    assert args.temporal_ensemble_coeff == 0.0
    assert args.eval_temporal_ensemble_coeff == 0.0


def test_parse_args_streaming_act_accepts_temporal_ensemble_override() -> None:
    args = parse_args(
        [
            "--env",
            "robocasa",
            "--policy",
            "streaming_act",
            "--task",
            "CloseFridge",
            "--temporal-ensemble-coeff",
            "0.025",
        ]
    )

    assert args.temporal_ensemble_coeff == 0.025
    assert args.eval_temporal_ensemble_coeff == 0.025


def test_configure_streaming_act_eval_runtime_disables_temporal_ensemble_when_coeff_is_zero() -> None:
    cfg = SimpleNamespace(
        n_action_steps=100,
        chunk_size=100,
        temporal_ensemble_coeff=None,
    )
    policy = SimpleNamespace()

    reconfigured = configure_streaming_act_eval_runtime(
        policy=policy,
        cfg=cfg,
        requested_n_action_steps=100,
        requested_temporal_ensemble_coeff=0.0,
    )

    assert reconfigured is True
    assert cfg.n_action_steps == 100
    assert cfg.temporal_ensemble_coeff is None
    assert not hasattr(policy, "temporal_ensembler")


def test_configure_streaming_act_eval_runtime_enables_temporal_ensemble_and_forces_single_step() -> None:
    cfg = SimpleNamespace(
        n_action_steps=100,
        chunk_size=100,
        temporal_ensemble_coeff=None,
    )
    policy = SimpleNamespace()
    captured_factory_args: list[tuple[float, int]] = []

    def _fake_temporal_ensembler(coeff: float, chunk_size: int):
        captured_factory_args.append((coeff, chunk_size))
        return {"coeff": coeff, "chunk_size": chunk_size}

    reconfigured = configure_streaming_act_eval_runtime(
        policy=policy,
        cfg=cfg,
        requested_n_action_steps=50,
        requested_temporal_ensemble_coeff=0.05,
        temporal_ensembler_factory=_fake_temporal_ensembler,
    )

    assert reconfigured is True
    assert cfg.n_action_steps == 1
    assert cfg.temporal_ensemble_coeff == 0.05
    assert policy.temporal_ensembler == {"coeff": 0.05, "chunk_size": 100}
    assert captured_factory_args == [(0.05, 100)]


def test_configure_streaming_act_eval_runtime_rejects_negative_temporal_ensemble_coeff() -> None:
    cfg = SimpleNamespace(
        n_action_steps=100,
        chunk_size=100,
        temporal_ensemble_coeff=None,
    )
    policy = SimpleNamespace()

    with pytest.raises(ValueError, match="temporal-ensemble-coeff"):
        configure_streaming_act_eval_runtime(
            policy=policy,
            cfg=cfg,
            requested_n_action_steps=100,
            requested_temporal_ensemble_coeff=-0.01,
        )
