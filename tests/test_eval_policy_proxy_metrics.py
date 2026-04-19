from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from eval_policy import (
    OfflineProxyThresholds,
    build_action_group_feature_map,
    build_action_groups,
    build_episode_pass_proxy,
    compute_criticality_scores,
    select_critical_window_indices,
    select_final_window_indices,
    summarize_normalized_error_matrix,
)


def test_build_action_groups_handles_zeno_style_base_and_bimanual_names() -> None:
    feature_names = [
        "base_vx",
        "base_vy",
        "base_omega",
        "left_joint_0",
        "left_joint_1",
        "right_joint_0",
        "right_joint_1",
    ]

    groups = build_action_groups(feature_names)
    feature_map = build_action_group_feature_map(feature_names, groups)

    assert list(groups) == ["base", "left_arm", "right_arm"]
    assert groups["base"].tolist() == [0, 1, 2]
    assert groups["left_arm"].tolist() == [3, 4]
    assert groups["right_arm"].tolist() == [5, 6]
    assert feature_map["base"] == ["base_vx", "base_vy", "base_omega"]
    assert feature_map["left_arm"] == ["left_joint_0", "left_joint_1"]
    assert feature_map["right_arm"] == ["right_joint_0", "right_joint_1"]


def test_select_critical_window_indices_prefers_high_delta_steps_and_can_fallback() -> None:
    actions = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    states = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = compute_criticality_scores(
        action_matrix=actions,
        state_matrix=states,
        action_scale=np.ones((2,), dtype=np.float32),
        state_scale=np.ones((2,), dtype=np.float32),
        source="combined_delta",
    )
    indices, mode = select_critical_window_indices(
        criticality_scores=scores,
        fraction=0.25,
    )

    assert mode == "topk_criticality"
    assert indices.tolist() == [2]

    fallback_indices, fallback_mode = select_critical_window_indices(
        criticality_scores=np.zeros((4,), dtype=np.float32),
        fraction=0.25,
    )

    assert fallback_mode == "fallback_final_window"
    assert fallback_indices.tolist() == select_final_window_indices(4, 0.25).tolist()


def test_build_episode_pass_proxy_captures_final_window_critical_window_and_spike_failures() -> None:
    feature_names = [
        "base_vx",
        "base_vy",
        "left_joint_0",
        "left_joint_1",
    ]
    groups = build_action_groups(feature_names)
    normalized_error_matrix = np.asarray(
        [
            [0.2, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0],
            [1.4, 1.4, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    normalized_error = summarize_normalized_error_matrix(
        normalized_error_matrix,
        action_groups=groups,
    )
    final_window_error = summarize_normalized_error_matrix(
        normalized_error_matrix[select_final_window_indices(4, 0.25)],
        action_groups=groups,
    )
    critical_window_error = summarize_normalized_error_matrix(
        normalized_error_matrix[[2]],
        action_groups=groups,
    )
    pass_proxy = build_episode_pass_proxy(
        normalized_error=normalized_error,
        final_window_error=final_window_error,
        critical_window_error=critical_window_error,
        thresholds=OfflineProxyThresholds(
            final_window_max_mean_l2=0.9,
            critical_window_max_mean_l2=1.0,
            max_step_l2=1.3,
        ),
    )

    assert normalized_error["per_group"]["left_arm"]["mean_l2_error"] == 0.5
    assert final_window_error["mean_l2_error"] == pytest.approx(0.98994946)
    assert critical_window_error["per_group"]["left_arm"]["mean_l2_error"] == 2.0
    assert pass_proxy["passed"] is False
    assert pass_proxy["failure_reasons"] == [
        "final_window",
        "critical_window",
        "step_spike",
    ]
