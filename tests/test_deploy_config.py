from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main"))
sys.path.insert(0, str(REPO_ROOT / "main" / "deploy"))

from deploy.config import load_deploy_config


def test_load_deploy_config_parses_vla_task_and_groot_attention(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "deploy.yaml"
    config_path.write_text(
        """
policy:
  type: groot
  path: ./outputs/groot/checkpoints/last
  task: Return the book to its original location.
  groot_attn_implementation: sdpa
  image_keys:
    left: observation.images.left
    right: observation.images.right
    top: observation.images.top
command:
  deadzone_linear_x: 0.02
  deadzone_linear_y: 0.03
  deadzone_angular_z: 0.04
  mutual_exclusion:
    enabled: true
    rules:
      - source_index: 0
        threshold: 0.4
        target_indices: [4, 10]
      - source_index: 2
        threshold: 0.2
        target_indices: [1]
        mask_value: -1.0
""",
        encoding="utf-8",
    )

    cfg = load_deploy_config(config_path)

    assert cfg.policy.type == "groot"
    assert cfg.policy.task == "Return the book to its original location."
    assert cfg.policy.groot_attn_implementation == "sdpa"
    assert cfg.command.deadzone_linear_x == 0.02
    assert cfg.command.deadzone_linear_y == 0.03
    assert cfg.command.deadzone_angular_z == 0.04
    assert cfg.command.mutual_exclusion.enabled is True
    assert len(cfg.command.mutual_exclusion.rules) == 2
    assert cfg.command.mutual_exclusion.rules[0].source_index == 0
    assert cfg.command.mutual_exclusion.rules[0].target_indices == (4, 10)
    assert cfg.command.mutual_exclusion.rules[1].mask_value == -1.0
