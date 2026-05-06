from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from policy_defaults import load_policy_mode_defaults_for_dataset
from train_policy import (
    fresh_distributed_output_marker_matches,
    parse_args,
    register_lerobot_fresh_distributed_output_reservation,
    reserve_fresh_distributed_output_dir,
    resolve_resume_run_state,
    resolve_train_run_stamp,
)


def _make_resumable_checkpoint(checkpoint_dir: Path, *, step: int) -> Path:
    pretrained_model_dir = checkpoint_dir / "pretrained_model"
    training_state_dir = checkpoint_dir / "training_state"
    pretrained_model_dir.mkdir(parents=True, exist_ok=True)
    training_state_dir.mkdir(parents=True, exist_ok=True)
    (pretrained_model_dir / "train_config.json").write_text(
        json.dumps({"job_name": "resume-smoke"}),
        encoding="utf-8",
    )
    (training_state_dir / "training_step.json").write_text(
        json.dumps({"step": int(step)}),
        encoding="utf-8",
    )
    return checkpoint_dir


def test_resolve_resume_run_state_prefers_latest_run_and_last_checkpoint(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs" / "train" / "demo"

    older_run = output_root / "20260416_120000"
    older_ckpt = _make_resumable_checkpoint(
        older_run / "checkpoints" / "000100",
        step=100,
    )

    latest_run = output_root / "20260417_120000"
    latest_ckpt = _make_resumable_checkpoint(
        latest_run / "checkpoints" / "000200",
        step=200,
    )
    (latest_run / "checkpoints" / "last").symlink_to(Path("000200"))

    os.utime(older_run, (1_000, 1_000))
    os.utime(latest_run, (2_000, 2_000))

    resolved = resolve_resume_run_state(output_root)

    assert resolved.run_dir == latest_run.resolve()
    assert resolved.checkpoint_dir == latest_ckpt.resolve()
    assert resolved.pretrained_model_dir == (latest_ckpt / "pretrained_model").resolve()
    assert resolved.train_config_path == (
        latest_ckpt / "pretrained_model" / "train_config.json"
    ).resolve()


def test_resolve_resume_run_state_skips_newer_runs_without_resumable_checkpoint(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs" / "train" / "demo"

    resumable_run = output_root / "20260416_120000"
    resumable_ckpt = _make_resumable_checkpoint(
        resumable_run / "checkpoints" / "000300",
        step=300,
    )

    incomplete_run = output_root / "20260417_120000"
    (incomplete_run / "checkpoints" / "000400" / "pretrained_model").mkdir(
        parents=True,
        exist_ok=True,
    )

    os.utime(resumable_run, (1_000, 1_000))
    os.utime(incomplete_run, (2_000, 2_000))

    resolved = resolve_resume_run_state(output_root)

    assert resolved.run_dir == resumable_run.resolve()
    assert resolved.checkpoint_dir == resumable_ckpt.resolve()


def test_parse_args_supports_resume_flag_and_defaults_expose_resume() -> None:
    defaults, defaults_path = load_policy_mode_defaults_for_dataset(
        mode="train",
        dataset_selector="zeno-ai/day3_5_Exp1_processed",
        policy_name="act",
    )

    assert defaults_path is not None
    assert "resume" in defaults
    assert defaults["resume"] is False
    assert defaults["distributed"] == {
        "enabled": False,
        "launcher": "accelerate",
        "num_processes": 1,
        "gpu_ids": "all",
        "num_machines": 1,
        "machine_rank": 0,
        "main_process_ip": None,
        "main_process_port": None,
    }

    args = parse_args(
        [
            "--dataset",
            "zeno-ai/day3_5_Exp1_processed",
            "--policy",
            "act",
            "--resume",
        ]
    )

    assert args.resume is True


def test_resolve_train_run_stamp_prefers_shared_env(monkeypatch) -> None:
    monkeypatch.setenv("CORL_TRAIN_RUN_STAMP", "20260417_123456")
    assert resolve_train_run_stamp() == "20260417_123456"


def test_resolve_train_run_stamp_falls_back_to_now(monkeypatch) -> None:
    monkeypatch.delenv("CORL_TRAIN_RUN_STAMP", raising=False)
    assert (
        resolve_train_run_stamp(now=dt.datetime(2026, 4, 17, 12, 34, 56))
        == "20260417_123456"
    )


def test_reserve_fresh_distributed_output_dir_main_creates_marker(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "train" / "demo" / "20260417_123456"

    reservation = reserve_fresh_distributed_output_dir(
        output_dir=output_dir,
        launch_id="launch-1",
        run_stamp="20260417_123456",
        world_size=4,
        is_main_process=True,
    )

    assert reservation is not None
    assert reservation.output_dir == output_dir.resolve()
    assert reservation.marker_path.is_file()
    assert fresh_distributed_output_marker_matches(
        output_dir,
        launch_id="launch-1",
    )


def test_reserve_fresh_distributed_output_dir_rejects_preexisting_run(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "train" / "demo" / "20260417_123456"
    output_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists before"):
        reserve_fresh_distributed_output_dir(
            output_dir=output_dir,
            launch_id="launch-1",
            run_stamp="20260417_123456",
            world_size=4,
            is_main_process=True,
        )


def test_reserve_fresh_distributed_output_dir_rejects_stale_marker(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "train" / "demo" / "20260417_123456"
    reserve_fresh_distributed_output_dir(
        output_dir=output_dir,
        launch_id="launch-1",
        run_stamp="20260417_123456",
        world_size=4,
        is_main_process=True,
    )

    with pytest.raises(FileExistsError, match="already exists before"):
        reserve_fresh_distributed_output_dir(
            output_dir=output_dir,
            launch_id="launch-1",
            run_stamp="20260417_123456",
            world_size=4,
            is_main_process=True,
        )


def test_lerobot_validate_patch_allows_reserved_fresh_distributed_dir(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "train" / "demo" / "20260417_123456"
    reservation = reserve_fresh_distributed_output_dir(
        output_dir=output_dir,
        launch_id="launch-1",
        run_stamp="20260417_123456",
        world_size=4,
        is_main_process=True,
    )

    class DummyTrainPipelineConfig:
        def __init__(self, output_dir: Path) -> None:
            self.output_dir = output_dir
            self.resume = False
            self.validate_resume_values: list[bool] = []
            self.validate_output_dir_types: list[type] = []

        def validate(self) -> str:
            self.validate_resume_values.append(bool(self.resume))
            self.validate_output_dir_types.append(type(self.output_dir))
            if self.resume:
                raise ValueError("A config_path is expected when resuming a run.")
            if (
                isinstance(self.output_dir, Path)
                and self.output_dir.is_dir()
                and not self.resume
            ):
                raise FileExistsError(f"Output directory {self.output_dir} exists")
            return "validated"

    register_lerobot_fresh_distributed_output_reservation(
        DummyTrainPipelineConfig,
        reservation,
    )
    cfg = DummyTrainPipelineConfig(output_dir)

    assert cfg.validate() == "validated"
    assert cfg.output_dir == output_dir
    assert cfg.resume is False
    assert cfg.validate_resume_values == [False]
    assert cfg.validate_output_dir_types == [str]
