from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from eval_policy import parse_args


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
