from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from train_policy import _remap_pi05_legacy_vision_tower_state_dict_keys
from train_policy import should_log_dataset_feature_filtering


def test_remap_pi05_legacy_vision_tower_state_dict_keys_prefers_canonical_keys() -> None:
    canonical_value = object()
    legacy_value = object()
    other_value = object()
    state_dict = {
        "model.paligemma_with_expert.paligemma.model.vision_tower.embeddings.patch_embedding.weight": canonical_value,
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight": legacy_value,
        "something_else": other_value,
    }

    remapped_state_dict, changed = _remap_pi05_legacy_vision_tower_state_dict_keys(
        state_dict
    )

    assert changed is True
    assert (
        remapped_state_dict[
            "model.paligemma_with_expert.paligemma.model.vision_tower.embeddings.patch_embedding.weight"
        ]
        is canonical_value
    )
    assert (
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
        not in remapped_state_dict
    )
    assert remapped_state_dict["something_else"] is other_value


def test_should_log_dataset_feature_filtering_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CORL_VERBOSE_DATASET_FEATURE_FILTERS", raising=False)

    assert should_log_dataset_feature_filtering() is False


def test_should_log_dataset_feature_filtering_accepts_truthy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORL_VERBOSE_DATASET_FEATURE_FILTERS", "1")

    assert should_log_dataset_feature_filtering() is True
