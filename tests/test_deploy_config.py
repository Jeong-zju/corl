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
""",
        encoding="utf-8",
    )

    cfg = load_deploy_config(config_path)

    assert cfg.policy.type == "groot"
    assert cfg.policy.task == "Return the book to its original location."
    assert cfg.policy.groot_attn_implementation == "sdpa"
