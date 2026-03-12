import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def create_h_shape_grid(
    grid_size=(200, 200),
    total_width=6.0,
    total_height=8.0,
    bar_thickness=1.5,
    mid_bar_height=1.0,
    wall_thickness=0.3,
):
    rows, cols = grid_size
    hw = total_width / 2
    hh = total_height / 2
    bt = bar_thickness
    mh = mid_bar_height / 2
    wt = wall_thickness

    xmin, xmax = -hw - 1.0, hw + 1.0
    ymin, ymax = -hh - 1.0, hh + 1.0

    xs = np.linspace(xmin, xmax, cols)
    ys = np.linspace(ymin, ymax, rows)
    X, Y = np.meshgrid(xs, ys)

    left_inner = (X > -hw + wt) & (X < -hw + bt - wt) & (Y > -hh + wt) & (Y < hh - wt)
    right_inner = (X > hw - bt + wt) & (X < hw - wt) & (Y > -hh + wt) & (Y < hh - wt)
    mid_inner = (
        (X > -hw + bt - wt) & (X < hw - bt + wt) & (Y > -mh + wt) & (Y < mh - wt)
    )

    free = left_inner | right_inner | mid_inner
    grid = np.ones((rows, cols), dtype=np.float64)
    grid[free] = 0.0
    extent = (xmin, xmax, ymin, ymax)
    return grid, extent


def world_to_grid(point, extent, shape):
    xmin, xmax, ymin, ymax = extent
    rows, cols = shape
    x, y = point
    col = int(round((x - xmin) / (xmax - xmin) * (cols - 1)))
    row = int(round((y - ymin) / (ymax - ymin) * (rows - 1)))
    row = int(np.clip(row, 0, rows - 1))
    col = int(np.clip(col, 0, cols - 1))
    return row, col


def grid_to_world(row, col, extent, shape):
    xmin, xmax, ymin, ymax = extent
    rows, cols = shape
    x = np.interp(col, [0, cols - 1], [xmin, xmax])
    y = np.interp(row, [0, rows - 1], [ymin, ymax])
    return (float(x), float(y))


def is_collision_free(point, grid, extent):
    row, col = world_to_grid(point, extent, grid.shape)
    return bool(grid[row, col] == 0.0)


def segment_collision_free(p0, p1, grid, extent, num_checks=24):
    for t in np.linspace(0.0, 1.0, num_checks):
        p = (
            float(p0[0] * (1.0 - t) + p1[0] * t),
            float(p0[1] * (1.0 - t) + p1[1] * t),
        )
        if not is_collision_free(p, grid, extent):
            return False
    return True


def find_fixed_h_corners(grid, extent):
    rows, cols = grid.shape
    free = np.argwhere(grid == 0)
    if free.shape[0] == 0:
        raise RuntimeError("No free cells found in the H-map.")

    ul = ur = ll = lr = None
    ul_score = ur_score = ll_score = lr_score = None

    for row, col in free:
        x, y = grid_to_world(int(row), int(col), extent, grid.shape)
        if x < 0.0 and y > 0.0:
            score = (int(col), -int(row))
            if ul_score is None or score < ul_score:
                ul = (int(row), int(col))
                ul_score = score
        if x > 0.0 and y > 0.0:
            score = (-int(col), -int(row))
            if ur_score is None or score < ur_score:
                ur = (int(row), int(col))
                ur_score = score
        if x < 0.0 and y < 0.0:
            score = (int(col), int(row))
            if ll_score is None or score < ll_score:
                ll = (int(row), int(col))
                ll_score = score
        if x > 0.0 and y < 0.0:
            score = (-int(col), int(row))
            if lr_score is None or score < lr_score:
                lr = (int(row), int(col))
                lr_score = score

    if any(v is None for v in (ul, ur, ll, lr)):
        raise RuntimeError("Could not locate all four deterministic H corners.")

    return {
        "upper_left": grid_to_world(*ul, extent, grid.shape),
        "upper_right": grid_to_world(*ur, extent, grid.shape),
        "lower_left": grid_to_world(*ll, extent, grid.shape),
        "lower_right": grid_to_world(*lr, extent, grid.shape),
    }


def world_to_pixel(point, extent, size):
    xmin, xmax, ymin, ymax = extent
    h, w = size
    x, y = point
    px = int(round((x - xmin) / (xmax - xmin) * (w - 1)))
    py = int(round((y - ymin) / (ymax - ymin) * (h - 1)))
    px = int(np.clip(px, 0, w - 1))
    py = int(np.clip(py, 0, h - 1))
    return px, py


def draw_disk(img, center, radius, color):
    cx, cy = center
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius
    img[mask] = np.array(color, dtype=np.uint8)


def make_base_image(grid, image_hw):
    rows, cols = grid.shape
    h, w = image_hw
    row_idx = np.linspace(0, rows - 1, h).astype(int)
    col_idx = np.linspace(0, cols - 1, w).astype(int)
    sampled = grid[np.ix_(row_idx, col_idx)]
    gray = np.where(sampled == 0, np.uint8(240), np.uint8(55)).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def render_frame(base_img, extent, agent_pos):
    frame = base_img.copy()
    h, w = frame.shape[:2]
    agent_px = world_to_pixel(agent_pos, extent, (h, w))
    draw_disk(frame, agent_px, max(2, int(min(h, w) * 0.025)), (240, 70, 70))
    return frame


def start_ffmpeg_writer(path, width, height, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def resolve_policy_dir(policy_path: Path) -> Path:
    # Common confusion in this repo: training script run from `main/` writes to `main/outputs`,
    # but users may pass `/.../corl/outputs/...`. Try both layouts automatically.
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

    # Deduplicate while preserving order
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

    probe_lines = "\n".join(f"- {p}" for p in ordered)
    raise FileNotFoundError(
        "Could not find policy weights. Checked these base paths:\n"
        f"{probe_lines}\n"
        "Expected either '<base>/model.safetensors' or '<base>/pretrained_model/model.safetensors'."
    )


def build_eval_observation(
    state_xy,
    rgb_frame,
    state_key,
    image_key,
    state_dim,
):
    state_vec = np.zeros((state_dim,), dtype=np.float32)
    copy_n = min(2, state_dim)
    state_vec[:copy_n] = np.asarray(state_xy[:copy_n], dtype=np.float32)

    obs = {
        state_key: torch.from_numpy(state_vec),
        image_key: torch.from_numpy(rgb_frame).permute(2, 0, 1).contiguous().float()
        / 255.0,
    }
    return obs


def build_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ACT policy in H-shape env and save one video per rollout. "
            "Start/goal selection matches dataset generation exactly."
        )
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        required=True,
        help="Checkpoint dir or pretrained_model dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/h_shape_act"),
        help="Directory where rollout videos and summary are saved.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=20,
        help="Number of evaluations (start/goal alternate upper/lower corners by rollout index).",
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-threshold", type=float, default=0.20)
    parser.add_argument("--max-action-step", type=float, default=0.30)
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu/mps")
    return parser.parse_args()


def main():
    args = build_args()

    repo_root = Path(__file__).resolve().parents[2]
    lerobot_src = repo_root / "ACT-wholebody-torque/lerobot/src"
    if not lerobot_src.exists():
        raise FileNotFoundError(f"LeRobot source not found: {lerobot_src}")
    sys.path.insert(0, str(lerobot_src))

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy_dir = resolve_policy_dir(args.policy_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load through policy class so config parsing is done via PreTrainedConfig choice registry.
    policy = ACTPolicy.from_pretrained(policy_dir)
    cfg = policy.config
    # cfg.temporal_ensemble_coeff = 1.0
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
    image_shape = tuple(cfg.image_features[image_key].shape)  # (C, H, W)
    image_hw = (int(image_shape[1]), int(image_shape[2]))

    if cfg.robot_state_feature is None:
        raise RuntimeError("ACT policy has no observation.state feature.")
    state_key = "observation.state"
    state_dim = int(cfg.robot_state_feature.shape[0])

    grid, extent = create_h_shape_grid()
    base_img = make_base_image(grid, image_hw)
    corners = find_fixed_h_corners(grid, extent)

    results = []
    success_count = 0

    for ep_idx in range(args.num_rollouts):
        # Keep start/goal selection identical to data generation script:
        # even episode -> upper-left to upper-right, odd episode -> lower-left to lower-right.
        use_upper = ep_idx % 2 == 0
        if use_upper:
            start = corners["upper_left"]
            goal = corners["upper_right"]
        else:
            start = corners["lower_left"]
            goal = corners["lower_right"]
        agent = (float(start[0]), float(start[1]))
        done = False
        success = False
        episode_reward = 0.0

        video_path = output_dir / f"rollout_{ep_idx:03d}.mp4"
        writer = start_ffmpeg_writer(video_path, image_hw[1], image_hw[0], args.fps)

        trajectory = [agent]
        for step in range(args.max_steps):
            frame = render_frame(base_img, extent, agent)
            writer.stdin.write(frame.astype(np.uint8).tobytes())

            obs_values = [agent[0], agent[1]]
            obs = build_eval_observation(
                state_xy=obs_values,
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

            proposed = (agent[0] + dx, agent[1] + dy)
            if segment_collision_free(agent, proposed, grid, extent):
                agent = proposed
            else:
                proposed_half = (agent[0] + dx * 0.5, agent[1] + dy * 0.5)
                if segment_collision_free(agent, proposed_half, grid, extent):
                    agent = proposed_half

            trajectory.append(agent)
            dist = float(np.linalg.norm(np.array(agent) - np.array(goal)))
            reward = -dist
            if dist < args.success_threshold:
                done = True
                success = True
                reward = 1.0

            episode_reward += reward
            if done:
                final_frame = render_frame(base_img, extent, agent)
                writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                break

        writer.stdin.close()
        code = writer.wait()
        if code != 0:
            raise RuntimeError(
                f"ffmpeg failed on rollout {ep_idx} with exit code {code}"
            )

        if success:
            success_count += 1

        final_distance = float(np.linalg.norm(np.array(agent) - np.array(goal)))
        result = {
            "episode_index": ep_idx,
            "video_path": str(video_path),
            "start": [float(start[0]), float(start[1])],
            "goal": [float(goal[0]), float(goal[1])],
            "final_position": [float(agent[0]), float(agent[1])],
            "final_distance": final_distance,
            "success": bool(success),
            "steps": int(len(trajectory) - 1),
            "sum_reward": float(episode_reward),
        }
        results.append(result)
        print(
            f"[{ep_idx + 1:03d}/{args.num_rollouts:03d}] "
            f"success={success} steps={result['steps']} final_dist={final_distance:.3f} "
            f"video={video_path.name}"
        )

    summary = {
        "num_rollouts": args.num_rollouts,
        "success_count": success_count,
        "success_rate": float(success_count / max(1, args.num_rollouts)),
        "seed": args.seed,
        "fps": args.fps,
        "max_steps": args.max_steps,
        "success_threshold": args.success_threshold,
        "policy_dir": str(policy_dir),
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nSaved {args.num_rollouts} rollout videos to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(
        f"Success rate: {summary['success_rate']:.3f} ({success_count}/{args.num_rollouts})"
    )


if __name__ == "__main__":
    main()
