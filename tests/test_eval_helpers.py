from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

import eval_helpers
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


def test_load_pretrained_config_from_pretrained_dir_falls_back_on_unknown_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / "config.json").write_text("{}", encoding="utf-8")

    calls: list[bool] = []

    class DummyConfig:
        foo = 7

    def fake_parse_config_with_class(
        *,
        config_cls,
        config_path,
        drop_unknown_fields,
    ):
        calls.append(drop_unknown_fields)
        if not drop_unknown_fields:
            raise RuntimeError("unknown field: type")
        return DummyConfig(), ("type", "unused_field")

    monkeypatch.setattr(
        eval_helpers,
        "_parse_config_with_class",
        fake_parse_config_with_class,
    )

    cfg = eval_helpers.load_pretrained_config_from_pretrained_dir(
        DummyConfig,
        policy_dir,
        policy_label="ACT",
    )

    assert isinstance(cfg, DummyConfig)
    assert cfg.foo == 7
    assert calls == [False, True]
