from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main"))
sys.path.insert(0, str(REPO_ROOT / "main" / "deploy"))

from deploy.config import PolicyConfig
from deploy.policy_runtime.loader import apply_deploy_policy_overrides


def _make_policy_config(**overrides) -> PolicyConfig:
    base = dict(
        type="streaming_act",
        path=Path("."),
        device="cpu",
        load_device=None,
        n_action_steps=50,
        temporal_ensemble_coeff=0.0,
        state_dim=17,
        action_dim=17,
        arm_dof=7,
        base_action_dim=3,
        state_key="observation.state",
        action_key="action",
        image_keys={
            "left": "observation.images.left",
            "right": "observation.images.right",
            "top": "observation.images.top",
        },
        use_path_signature=False,
        use_delta_signature=False,
        signature_depth=1,
        signature_dim=None,
        signature_backend="simple",
    )
    base.update(overrides)
    return PolicyConfig(**base)


def test_apply_deploy_policy_overrides_keeps_open_loop_when_coeff_is_zero() -> None:
    cfg = SimpleNamespace(
        n_action_steps=10,
        temporal_ensemble_coeff=None,
    )

    coeff, enabled = apply_deploy_policy_overrides(cfg, _make_policy_config())

    assert coeff == 0.0
    assert enabled is False
    assert cfg.temporal_ensemble_coeff is None
    assert cfg.n_action_steps == 50


def test_apply_deploy_policy_overrides_forces_single_step_when_coeff_is_nonzero() -> None:
    cfg = SimpleNamespace(
        n_action_steps=25,
        temporal_ensemble_coeff=None,
    )

    coeff, enabled = apply_deploy_policy_overrides(
        cfg,
        _make_policy_config(
            n_action_steps=50,
            temporal_ensemble_coeff=0.01,
        ),
    )

    assert coeff == 0.01
    assert enabled is True
    assert cfg.temporal_ensemble_coeff == 0.01
    assert cfg.n_action_steps == 1
