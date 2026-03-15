import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from .imitation_dataset_utils import (
        make_lerobot_base_image,
        render_lerobot_frame,
        start_ffmpeg_raw_writer,
    )
    from .shared_channel_double_loop_map import (
        BraidedHub2DEnv,
        TASK_ID_TO_GOAL_NAME,
        build_default_map_config,
        build_task_spec,
    )
except ImportError:
    from imitation_dataset_utils import (
        make_lerobot_base_image,
        render_lerobot_frame,
        start_ffmpeg_raw_writer,
    )
    from shared_channel_double_loop_map import (
        BraidedHub2DEnv,
        TASK_ID_TO_GOAL_NAME,
        build_default_map_config,
        build_task_spec,
    )


DEFAULT_OUTPUT_DIR = Path("outputs/eval/braidedhub_fourstart_act")
DEFAULT_NUM_ROLLOUTS = 20
DEFAULT_MAX_STEPS = 240
DEFAULT_FPS = 20
DEFAULT_SEED = 42
DEFAULT_MAX_ACTION_STEP = 2.5
BRANCH1_PHASE_BY_BIT = {0: "branch1_upper_region", 1: "branch1_lower_region"}
BRANCH2_PHASE_BY_BIT = {0: "branch2_upper_region", 1: "branch2_lower_region"}
BRANCH1_PHASES = frozenset(BRANCH1_PHASE_BY_BIT.values())
BRANCH2_PHASES = frozenset(BRANCH2_PHASE_BY_BIT.values())


def resolve_policy_dir(policy_path: Path) -> Path:
    raw = policy_path.expanduser()
    repo_root = Path(__file__).resolve().parents[2]

    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, repo_root / raw, repo_root / "main" / raw])

    for p in list(candidates):
        p_str = str(p)
        root_str = str(repo_root)
        if p_str.startswith(f"{root_str}/outputs/"):
            suffix = p_str[len(f"{root_str}/outputs/") :]
            candidates.append(repo_root / "main" / "outputs" / suffix)
        if p_str.startswith(f"{root_str}/main/outputs/"):
            suffix = p_str[len(f"{root_str}/main/outputs/") :]
            candidates.append(repo_root / "outputs" / suffix)

    ordered = []
    seen = set()
    for p in candidates:
        rp = p.resolve(strict=False)
        if rp not in seen:
            seen.add(rp)
            ordered.append(rp)

    for base in ordered:
        if (base / "model.safetensors").exists():
            return base
        nested = base / "pretrained_model"
        if (nested / "model.safetensors").exists():
            return nested
        last_nested = base / "checkpoints" / "last" / "pretrained_model"
        if (last_nested / "model.safetensors").exists():
            return last_nested

    probe_lines = "\n".join(f"- {p}" for p in ordered)
    raise FileNotFoundError(
        "Could not find policy weights. Checked these base paths:\n"
        f"{probe_lines}\n"
        "Expected one of:\n"
        "- <base>/model.safetensors\n"
        "- <base>/pretrained_model/model.safetensors\n"
        "- <base>/checkpoints/last/pretrained_model/model.safetensors"
    )


def build_eval_observation(
    state_xy: tuple[float, float],
    rgb_frame: np.ndarray,
    state_key: str,
    image_key: str,
    state_dim: int,
) -> dict[str, object]:
    import torch

    state_vec = np.zeros((state_dim,), dtype=np.float32)
    copy_n = min(2, state_dim)
    state_vec[:copy_n] = np.asarray(state_xy[:copy_n], dtype=np.float32)
    return {
        state_key: torch.from_numpy(state_vec),
        image_key: torch.from_numpy(rgb_frame).permute(2, 0, 1).contiguous().float()
        / 255.0,
    }


def build_balanced_task_schedule(num_rollouts: int, seed: int) -> list[int]:
    task_ids = sorted(TASK_ID_TO_GOAL_NAME)
    repeated = [task_ids[idx % len(task_ids)] for idx in range(num_rollouts)]
    rng = np.random.default_rng(seed)
    rng.shuffle(repeated)
    return [int(task_id) for task_id in repeated]


def detect_branch_mismatch(
    task_spec,
    phase_name: str,
) -> dict[str, str] | None:
    expected_branch1 = BRANCH1_PHASE_BY_BIT[int(task_spec.task_bits[0])]
    expected_branch2 = BRANCH2_PHASE_BY_BIT[int(task_spec.task_bits[1])]

    if phase_name in BRANCH1_PHASES and phase_name != expected_branch1:
        return {
            "stage": "H1",
            "expected_phase": expected_branch1,
            "observed_phase": phase_name,
        }
    if phase_name in BRANCH2_PHASES and phase_name != expected_branch2:
        return {
            "stage": "H2",
            "expected_phase": expected_branch2,
            "observed_phase": phase_name,
        }
    return None


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an ACT policy in the braidedhub implicit-cue map "
            "and save one rollout video per episode."
        )
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        required=True,
        help="Checkpoint dir, pretrained_model dir, or training run dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where rollout videos and summary are saved.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=DEFAULT_NUM_ROLLOUTS,
        help="Number of evaluation rollouts.",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-action-step",
        type=float,
        default=DEFAULT_MAX_ACTION_STEP,
        help="Clamp action magnitude to avoid implausibly large jumps.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu/mps")
    return parser.parse_args()


def ensure_lerobot_importable(repo_root: Path) -> None:
    lerobot_src = repo_root / "ACT-wholebody-torque/lerobot/src"
    if not lerobot_src.exists():
        raise FileNotFoundError(f"LeRobot source not found: {lerobot_src}")
    sys.path.insert(0, str(lerobot_src))


def main() -> None:
    args = build_args()
    import torch

    repo_root = Path(__file__).resolve().parents[2]
    ensure_lerobot_importable(repo_root)

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy_dir = resolve_policy_dir(args.policy_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = ACTPolicy.from_pretrained(policy_dir)
    cfg = policy.config
    cfg.device = args.device
    policy.eval()
    policy.to(args.device)

    preprocessor_overrides = {
        "device_processor": {"device": args.device},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_dir,
        preprocessor_overrides=preprocessor_overrides,
    )

    if len(cfg.image_features) == 0:
        raise RuntimeError(
            "ACT policy has no image input feature; this eval script assumes visual input."
        )
    image_key = list(cfg.image_features.keys())[0]
    image_shape = tuple(cfg.image_features[image_key].shape)
    image_hw = (int(image_shape[1]), int(image_shape[2]))

    if cfg.robot_state_feature is None:
        raise RuntimeError("ACT policy has no observation.state feature.")
    state_key = "observation.state"
    state_dim = int(cfg.robot_state_feature.shape[0])

    map_config = build_default_map_config()
    env = BraidedHub2DEnv(map_config=map_config, rng_seed=args.seed)
    base_img = make_lerobot_base_image(map_config, image_size=max(image_hw))
    if base_img.shape[0] != image_hw[0] or base_img.shape[1] != image_hw[1]:
        raise RuntimeError(
            "Rendered image size mismatch between dataset image features and eval map image. "
            f"Policy expects HxW={image_hw}, base_img={base_img.shape[:2]}."
        )

    results = []
    success_count = 0
    branch_failure_count = 0
    task_success_counts = {int(task_id): 0 for task_id in sorted(TASK_ID_TO_GOAL_NAME)}
    task_rollout_counts = {int(task_id): 0 for task_id in sorted(TASK_ID_TO_GOAL_NAME)}
    task_branch_failure_counts = {
        int(task_id): 0 for task_id in sorted(TASK_ID_TO_GOAL_NAME)
    }
    task_schedule = build_balanced_task_schedule(args.num_rollouts, args.seed)

    for ep_idx, task_id in enumerate(task_schedule):
        task_spec = build_task_spec(task_id)
        task_rollout_counts[task_id] += 1
        state_xy = tuple(float(v) for v in env.reset(task_id=task_id))
        if env.start_region_name != task_spec.start_region_name:
            raise RuntimeError(
                "Environment reset returned a start region that does not match the task. "
                f"task={task_spec.task_code}, expected_start={task_spec.start_region_name}, "
                f"got={env.start_region_name}"
            )
        policy.reset()

        done = False
        episode_reward = 0.0
        video_path = output_dir / f"rollout_{ep_idx:03d}_task_{task_spec.task_code}.mp4"
        writer = start_ffmpeg_raw_writer(video_path, image_hw[1], image_hw[0], args.fps)
        if writer.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin for rollout video writing.")

        trajectory = [state_xy]
        last_info = {
            **env.last_info,
            "phase_name": env.get_phase_name(state_xy),
            "branch_mismatch": False,
            "branch_mismatch_stage": None,
            "expected_branch_phase": None,
            "observed_branch_phase": None,
            "failure_reason": None,
        }

        for _step_idx in range(args.max_steps):
            frame = render_lerobot_frame(base_img, config=map_config, robot_xy=state_xy)
            writer.stdin.write(frame.astype(np.uint8).tobytes())

            obs = build_eval_observation(
                state_xy=state_xy,
                rgb_frame=frame,
                state_key=state_key,
                image_key=image_key,
                state_dim=state_dim,
            )
            obs = preprocessor(obs)

            with torch.no_grad():
                action = policy.select_action(obs)
            action = postprocessor(action)
            action_np = action.squeeze(0).detach().cpu().numpy()

            dx = float(action_np[0]) if action_np.shape[0] >= 1 else 0.0
            dy = float(action_np[1]) if action_np.shape[0] >= 2 else 0.0
            norm = math.sqrt(dx * dx + dy * dy)
            if norm > args.max_action_step and norm > 1e-8:
                scale = args.max_action_step / norm
                dx *= scale
                dy *= scale

            next_state, reward, done, info = env.step((dx, dy))
            state_xy = (float(next_state[0]), float(next_state[1]))
            trajectory.append(state_xy)
            episode_reward += float(reward)
            phase_name = env.get_phase_name(state_xy)
            mismatch = detect_branch_mismatch(task_spec=task_spec, phase_name=phase_name)
            last_info = {
                **info,
                "phase_name": phase_name,
                "branch_mismatch": False,
                "branch_mismatch_stage": None,
                "expected_branch_phase": None,
                "observed_branch_phase": None,
                "failure_reason": None,
            }
            if mismatch is not None:
                last_info = {
                    **last_info,
                    "branch_mismatch": True,
                    "branch_mismatch_stage": mismatch["stage"],
                    "expected_branch_phase": mismatch["expected_phase"],
                    "observed_branch_phase": mismatch["observed_phase"],
                    "failure_reason": "wrong_branch",
                    "success": False,
                }
                final_frame = render_lerobot_frame(
                    base_img,
                    config=map_config,
                    robot_xy=state_xy,
                )
                writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                break

            if done:
                final_frame = render_lerobot_frame(
                    base_img,
                    config=map_config,
                    robot_xy=state_xy,
                )
                writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                break

        writer.stdin.close()
        code = writer.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed on rollout {ep_idx} with exit code {code}")

        success = bool(last_info.get("success", False)) and not bool(
            last_info.get("branch_mismatch", False)
        )
        if success:
            success_count += 1
            task_success_counts[task_id] += 1
        if last_info.get("branch_mismatch", False):
            branch_failure_count += 1
            task_branch_failure_counts[task_id] += 1

        result = {
            "episode_index": ep_idx,
            "task_id": task_id,
            "task_code": task_spec.task_code,
            "start_region_name": str(last_info.get("start_region_name")),
            "target_goal_name": str(last_info.get("target_goal_name")),
            "video_path": str(video_path),
            "final_position": [float(state_xy[0]), float(state_xy[1])],
            "final_phase_name": str(last_info.get("phase_name")),
            "reached_goal": last_info.get("reached_goal"),
            "branch_mismatch": bool(last_info.get("branch_mismatch", False)),
            "branch_mismatch_stage": last_info.get("branch_mismatch_stage"),
            "expected_branch_phase": last_info.get("expected_branch_phase"),
            "observed_branch_phase": last_info.get("observed_branch_phase"),
            "failure_reason": last_info.get("failure_reason"),
            "success": success,
            "steps": int(len(trajectory) - 1),
            "sum_reward": float(episode_reward),
            "collision_rejections": int(
                sum(
                    1
                    for idx in range(1, len(env.trajectory))
                    if env.trajectory[idx] == env.trajectory[idx - 1]
                )
            ),
        }
        results.append(result)
        print(
            f"[{ep_idx + 1:03d}/{args.num_rollouts:03d}] "
            f"task={task_spec.task_code}->{task_spec.target_goal_name} "
            f"success={success} steps={result['steps']} "
            f"reached={result['reached_goal']} "
            f"branch_mismatch={result['branch_mismatch']} "
            f"video={video_path.name}"
        )

    per_task = {
        str(task_id): {
            "goal_name": TASK_ID_TO_GOAL_NAME[task_id],
            "rollouts": int(task_rollout_counts[task_id]),
            "success_count": int(task_success_counts[task_id]),
            "wrong_branch_failures": int(task_branch_failure_counts[task_id]),
            "success_rate": float(
                task_success_counts[task_id] / max(1, task_rollout_counts[task_id])
            ),
        }
        for task_id in sorted(TASK_ID_TO_GOAL_NAME)
    }
    summary = {
        "num_rollouts": args.num_rollouts,
        "success_count": success_count,
        "success_rate": float(success_count / max(1, args.num_rollouts)),
        "wrong_branch_failures": branch_failure_count,
        "seed": args.seed,
        "fps": args.fps,
        "max_steps": args.max_steps,
        "max_action_step": args.max_action_step,
        "policy_dir": str(policy_dir),
        "per_task": per_task,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved {args.num_rollouts} rollout videos to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(
        f"Success rate: {summary['success_rate']:.3f} ({success_count}/{args.num_rollouts})"
    )
    print(f"Wrong-branch failures: {branch_failure_count}/{args.num_rollouts}")
    for task_id in sorted(TASK_ID_TO_GOAL_NAME):
        task_summary = per_task[str(task_id)]
        print(
            f"  task {task_id} ({task_summary['goal_name']}): "
            f"{task_summary['success_count']}/{task_summary['rollouts']} "
            f"= {task_summary['success_rate']:.3f}, "
            f"wrong_branch={task_summary['wrong_branch_failures']}"
        )


if __name__ == "__main__":
    main()
