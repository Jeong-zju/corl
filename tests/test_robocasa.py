from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from ..benchmarks import create_robocasa_benchmark_env, RandomRoboCasaPolicy
except ImportError:
    tests_dir = Path(__file__).resolve().parent
    main_dir = tests_dir.parent
    main_dir_str = str(main_dir)
    if main_dir_str not in sys.path:
        sys.path.insert(0, main_dir_str)
    from benchmarks import create_robocasa_benchmark_env, RandomRoboCasaPolicy


def main() -> None:
    output_root = (
        Path(__file__).resolve().parents[1] / "outputs" / "test_robocasa_playback"
    )
    env = create_robocasa_benchmark_env(
        conda_env="robocasa",
        task="PickPlaceCounterToSink",
        enable_render=True,
    )
    try:
        _, info = env.reset(seed=0, record_observation=False)
        action = env.sample_random_action()
        _, reward, terminated, truncated, step_info = env.step(
            action,
            record_observation=False,
        )
        result = env.rollout(
            policy=RandomRoboCasaPolicy(),
            max_steps=10,
            seed=0,
            record_observations=False,
            video_path=output_root / "single_rollout.mp4",
            details_path=output_root / "single_rollout.json",
            video_fps=20,
        )
        evaluation = env.evaluate_policy(
            policy=RandomRoboCasaPolicy(),
            num_rollouts=2,
            max_steps=10,
            seed=0,
            record_observations=False,
            output_dir=output_root / "eval",
            save_videos=True,
            video_fps=20,
        )
    finally:
        env.close()

    print("reset_info:")
    print(json.dumps(info, indent=2, default=str))
    print("\nstep_result:")
    print(
        json.dumps(
            {
                "reward": reward,
                "success": step_info["success"],
                "terminated": terminated,
                "truncated": truncated,
            },
            indent=2,
            default=str,
        )
    )
    print("\nrollout_summary:")
    print(json.dumps(result.to_summary_dict(), indent=2, default=str))
    print("\neval_summary:")
    print(json.dumps(evaluation.to_summary_dict(), indent=2, default=str))
    print("\nsaved_artifacts:")
    print(
        json.dumps(
            {
                "single_rollout_video": str(output_root / "single_rollout.mp4"),
                "single_rollout_details": str(output_root / "single_rollout.json"),
                "eval_summary": str(output_root / "eval" / "summary.json"),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
