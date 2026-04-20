from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from eval_helpers import resolve_policy_dir


def test_resolve_policy_dir_uses_latest_numbered_checkpoint_without_last(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "outputs" / "train" / "demo" / "20260420_120000"
    older = run_dir / "checkpoints" / "025000" / "pretrained_model"
    latest = run_dir / "checkpoints" / "050000" / "pretrained_model"
    older.mkdir(parents=True)
    latest.mkdir(parents=True)
    (older / "model.safetensors").write_text("", encoding="utf-8")
    (latest / "model.safetensors").write_text("", encoding="utf-8")

    assert resolve_policy_dir(run_dir) == latest.resolve()
