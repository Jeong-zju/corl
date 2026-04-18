from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))

from env.robocasa_env import RoboCasaRolloutResult, evaluate_policy


class _DummyFeature:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _DummyEnv:
    def __init__(self, task: str, action_dim: int) -> None:
        self.task = task
        self.action_dim = action_dim
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _make_args(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        num_rollouts=1,
        max_steps=5,
        max_episodes_rendered=0,
        fps=20,
        seed=7,
        task="CloseFridge",
        output_dir=output_dir,
        eval_num_rollouts=1,
        eval_max_steps=5,
        eval_max_episodes_rendered=0,
        eval_fps=20,
        eval_seed=7,
        eval_task_names=("CloseFridge",),
        eval_task_spec="CloseFridge",
        eval_robocasa_conda_env="",
        eval_robocasa_split="target",
    )


def _make_cfg() -> SimpleNamespace:
    image_feature = _DummyFeature((3, 32, 32))
    state_feature = _DummyFeature((16,))
    action_feature = _DummyFeature((12,))
    return SimpleNamespace(
        input_features={
            "observation.state": state_feature,
            "observation.images.main": image_feature,
        },
        image_features={
            "observation.images.main": image_feature,
        },
        output_features={
            "action": action_feature,
        },
        use_path_signature=False,
        use_delta_signature=False,
        use_prefix_sequence_training=False,
        use_visual_prefix_memory=False,
        use_signature_indexed_slot_memory=False,
    )


def test_robocasa_env_eval_supports_diffusion_checkpoints(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "eval"
    args = _make_args(output_dir)
    cfg = _make_cfg()
    env = _DummyEnv(task="CloseFridge", action_dim=12)
    captured_details_paths: list[Path] = []
    captured_video_paths: list[Path | None] = []

    monkeypatch.setattr(
        "env.robocasa_env.list_available_robocasa_tasks",
        lambda conda_env=None: ["CloseFridge"],
    )
    monkeypatch.setattr(
        "env.robocasa_env._resolve_task_dataset_root",
        lambda task_name, args, policy_dir: None,
    )
    monkeypatch.setattr(
        "env.robocasa_env._resolve_task_max_steps",
        lambda requested_max_steps, task_name, task_dataset_root: (
            int(requested_max_steps),
            {
                "source": "cli",
                "requested_max_steps": int(requested_max_steps),
                "dataset_root": None,
                "dataset_longest_episode_length": None,
                "conservative_margin": None,
            },
        ),
    )
    monkeypatch.setattr(
        "env.robocasa_env.create_robocasa_benchmark_env",
        lambda **kwargs: env,
    )
    monkeypatch.setattr(
        "env.robocasa_env._maybe_create_tqdm",
        lambda **kwargs: None,
    )

    def _fake_run_single_rollout(
        *,
        env,
        policy_adapter,
        max_steps,
        seed,
        video_path,
        details_path,
        fps,
        video_image_key,
        video_image_keys,
        step_progress=None,
    ) -> RoboCasaRolloutResult:
        captured_details_paths.append(details_path)
        captured_video_paths.append(video_path)
        return RoboCasaRolloutResult(
            task=str(env.task),
            seed=seed,
            max_steps=int(max_steps),
            num_steps=3,
            total_reward=1.5,
            success=True,
            success_details={"task": True},
            terminated=False,
            truncated=False,
            done_reason="success",
            initial_info={},
            final_info={},
            video_path=None,
            details_path=str(details_path),
        )

    monkeypatch.setattr(
        "env.robocasa_env._run_single_rollout",
        _fake_run_single_rollout,
    )

    evaluate_policy(
        policy_type="diffusion",
        args=args,
        policy=object(),
        cfg=cfg,
        preprocessor=lambda obs: obs,
        postprocessor=lambda action: action,
        policy_dir=tmp_path / "policy",
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["policy_type"] == "diffusion"
    assert summary["tasks"] == ["CloseFridge"]
    assert summary["success_count"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["per_task"][0]["output_dir"] == str(output_dir)
    assert summary["per_task"][0]["summary_path"] == str(output_dir / "summary.json")
    assert captured_details_paths == [output_dir / "rollout_000.json"]
    assert captured_video_paths == [None]
    assert env.closed is True


def test_robocasa_env_eval_still_rejects_prism_diffusion(
    tmp_path: Path,
) -> None:
    args = _make_args(tmp_path / "eval")
    cfg = _make_cfg()

    try:
        evaluate_policy(
            policy_type="prism_diffusion",
            args=args,
            policy=object(),
            cfg=cfg,
            preprocessor=lambda obs: obs,
            postprocessor=lambda action: action,
            policy_dir=tmp_path / "policy",
        )
    except NotImplementedError as exc:
        assert "prism_diffusion" in str(exc)
    else:
        raise AssertionError("Expected RoboCasa env eval to reject prism_diffusion.")
