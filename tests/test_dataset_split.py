from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from dataset_utils import build_dataset_split


def _make_fake_dataset_root(dataset_root: Path, *, total_episodes: int) -> Path:
    (dataset_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": int(total_episodes),
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
    (dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_text(
        "",
        encoding="utf-8",
    )
    return dataset_root


def test_build_dataset_split_uses_all_episodes_for_training_when_test_ratio_is_zero(
    tmp_path: Path,
) -> None:
    dataset_root = _make_fake_dataset_root(
        tmp_path / "demo_dataset",
        total_episodes=5,
    )

    split_spec = build_dataset_split(
        dataset_arg="demo_dataset",
        dataset_root=dataset_root,
        dataset_repo_id="demo/dataset",
        test_ratio=0.0,
        split_seed=42,
        split_shuffle=True,
    )

    assert split_spec.train_episode_indices == [0, 1, 2, 3, 4]
    assert split_spec.test_episode_indices == []
    assert split_spec.train_count == 5
    assert split_spec.test_count == 0
    assert split_spec.test_ratio == 0.0


def test_build_dataset_split_allows_single_episode_when_test_ratio_is_zero(
    tmp_path: Path,
) -> None:
    dataset_root = _make_fake_dataset_root(
        tmp_path / "single_episode_dataset",
        total_episodes=1,
    )

    split_spec = build_dataset_split(
        dataset_arg="single_episode_dataset",
        dataset_root=dataset_root,
        dataset_repo_id="demo/single_episode",
        test_ratio=0.0,
        split_seed=42,
        split_shuffle=False,
    )

    assert split_spec.train_episode_indices == [0]
    assert split_spec.test_episode_indices == []


def test_build_dataset_split_rejects_single_episode_when_test_ratio_is_positive(
    tmp_path: Path,
) -> None:
    dataset_root = _make_fake_dataset_root(
        tmp_path / "single_episode_dataset",
        total_episodes=1,
    )

    with pytest.raises(ValueError, match="at least 2 episodes"):
        build_dataset_split(
            dataset_arg="single_episode_dataset",
            dataset_root=dataset_root,
            dataset_repo_id="demo/single_episode",
            test_ratio=0.2,
            split_seed=42,
            split_shuffle=False,
        )
