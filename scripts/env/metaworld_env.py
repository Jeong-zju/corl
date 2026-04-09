from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import MethodType

from eval_helpers import (
    compute_delta_signature_step_np,
    compute_signatory_signature_np,
    compute_simple_signature_np,
    ensure_prefix_sequence_batch_dims,
    resolve_signature_backend,
    write_summary,
)
from eval_metaworld_policy import (
    configure_headless_opengl_defaults,
    patch_lerobot_metaworld_env,
)


DEFAULT_METAWORLD_TASKS = "assembly-v3,dial-turn-v3,handle-press-side-v3"
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_PATH_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_DELTA_SIGNATURE_KEY = "observation.delta_signature"


def _select_visual_observation_keys(cfg) -> list[str]:
    visual_features = getattr(cfg, "visual_observation_features", None)
    if visual_features is None:
        visual_features = getattr(cfg, "image_features", {})
    return list(visual_features)


def _compute_online_signature_prefix(
    state_history: deque,
    sig_depth: int,
    signature_backend: str,
):
    import numpy as np

    if len(state_history) == 0:
        raise ValueError("state_history is empty; cannot compute path signature.")
    window = np.stack(list(state_history), axis=0).astype(np.float32, copy=False)
    if signature_backend == "signatory":
        return compute_signatory_signature_np(window, sig_depth)
    return compute_simple_signature_np(window, sig_depth)


def _install_streaming_act_eval_hooks(
    *,
    policy,
    cfg,
    preprocessor,
    args,
):
    import torch
    from lerobot_policy_streaming_act.prefix_sequence import (
        PREFIX_DELTA_SIGNATURE_KEY,
        PREFIX_MASK_KEY,
        PREFIX_PATH_SIGNATURE_KEY,
        PREFIX_STATE_KEY,
        build_padded_prefix_from_history,
        prefix_image_key_from_camera_key,
    )

    use_path_signature = bool(getattr(cfg, "use_path_signature", False))
    use_prefix_sequence_training = bool(
        getattr(cfg, "use_prefix_sequence_training", False)
    )
    use_visual_prefix_memory = bool(getattr(cfg, "use_visual_prefix_memory", False))
    use_signature_indexed_slot_memory = bool(
        getattr(cfg, "use_signature_indexed_slot_memory", False)
    )
    use_delta_signature = bool(getattr(cfg, "use_delta_signature", False))
    build_explicit_prefix_eval_inputs = (
        use_prefix_sequence_training and not use_visual_prefix_memory
    )

    if not use_path_signature and not build_explicit_prefix_eval_inputs:
        return preprocessor, lambda: None

    if DEFAULT_STATE_KEY not in getattr(cfg, "input_features", {}):
        raise RuntimeError(
            "Meta-World eval requires `observation.state` when using streaming_act "
            "online signature/prefix inputs."
        )

    signature_backend = None
    state_history = None
    previous_signature_vec = None
    if use_path_signature:
        signature_backend = resolve_signature_backend(
            getattr(args, "signature_backend", "auto")
        )
        if int(getattr(cfg, "history_length", 0)) <= 0:
            raise ValueError(
                "Meta-World streaming_act eval requires `history_length > 0` when "
                "`use_path_signature=True`."
            )
        if int(getattr(cfg, "signature_depth", 0)) <= 0:
            raise ValueError(
                "Meta-World streaming_act eval requires `signature_depth > 0` when "
                "`use_path_signature=True`."
            )
        if int(getattr(cfg, "signature_dim", 0)) <= 0:
            raise ValueError(
                "Meta-World streaming_act eval requires `signature_dim > 0` when "
                "`use_path_signature=True`."
            )
        print(
            "[info] online path-signature eval enabled: "
            f"backend={signature_backend}, history<= {cfg.history_length}, "
            f"depth={cfg.signature_depth}, dim={cfg.signature_dim}"
        )
        state_history = deque(maxlen=int(cfg.history_length))
        if use_delta_signature:
            print(
                "[info] online delta-signature eval enabled: "
                f"key={DEFAULT_DELTA_SIGNATURE_KEY}, rule=g_t-g_(t-1), first_step=zeros"
            )

    visual_keys = _select_visual_observation_keys(cfg)
    prefix_state_history = [] if build_explicit_prefix_eval_inputs else None
    prefix_signature_history = (
        [] if build_explicit_prefix_eval_inputs and use_path_signature else None
    )
    prefix_delta_signature_history = (
        [] if build_explicit_prefix_eval_inputs and use_delta_signature else None
    )
    prefix_image_histories = (
        {key: [] for key in visual_keys}
        if build_explicit_prefix_eval_inputs
        else None
    )
    if build_explicit_prefix_eval_inputs:
        print(
            "[info] explicit prefix-sequence eval enabled: "
            f"max_steps={cfg.prefix_train_max_steps}, stride={cfg.prefix_frame_stride}, "
            f"cameras={visual_keys}"
        )
    elif use_visual_prefix_memory:
        print(
            "[info] visual prefix memory eval enabled: "
            + (
                "the policy updates signature-indexed slot memory from the true current observation at each step"
                if use_signature_indexed_slot_memory
                else "the policy updates recurrent memory from the true current observation at each step"
            )
        )

    original_reset = policy.reset

    def reset_histories() -> None:
        nonlocal previous_signature_vec
        if state_history is not None:
            state_history.clear()
        previous_signature_vec = None
        if prefix_state_history is not None:
            prefix_state_history.clear()
        if prefix_signature_history is not None:
            prefix_signature_history.clear()
        if prefix_delta_signature_history is not None:
            prefix_delta_signature_history.clear()
        if prefix_image_histories is not None:
            for history in prefix_image_histories.values():
                history.clear()

    def patched_reset(self, *reset_args, **reset_kwargs):
        reset_histories()
        return original_reset(*reset_args, **reset_kwargs)

    policy.reset = MethodType(patched_reset, policy)

    def wrapped_preprocessor(observation):
        nonlocal previous_signature_vec

        observation = dict(observation)
        if DEFAULT_STATE_KEY not in observation:
            raise KeyError(
                f"`{DEFAULT_STATE_KEY}` missing from Meta-World eval observation."
            )

        state = observation[DEFAULT_STATE_KEY]
        if not torch.is_tensor(state):
            state = torch.as_tensor(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        elif state.ndim != 2:
            raise RuntimeError(
                f"`{DEFAULT_STATE_KEY}` must be 1D/2D before preprocessing, "
                f"got shape={tuple(state.shape)}"
            )
        if state.shape[0] != 1:
            raise RuntimeError(
                "Meta-World streaming_act eval currently expects `n_envs=1` so it can "
                f"maintain one online signature history per rollout. Got batch={state.shape[0]}."
            )
        observation[DEFAULT_STATE_KEY] = state

        for visual_key in visual_keys:
            if visual_key not in observation:
                continue
            image = observation[visual_key]
            if not torch.is_tensor(image):
                image = torch.as_tensor(image)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            elif image.ndim != 4:
                raise RuntimeError(
                    f"`{visual_key}` must be 3D/4D before preprocessing, "
                    f"got shape={tuple(image.shape)}"
                )
            if image.shape[0] != 1:
                raise RuntimeError(
                    "Meta-World streaming_act eval currently expects `n_envs=1` for "
                    f"`{visual_key}`. Got batch={image.shape[0]}."
                )
            observation[visual_key] = image

        if use_path_signature:
            assert state_history is not None
            state_vec = (
                state[0].detach().cpu().numpy().astype("float32", copy=False).reshape(-1)
            )
            state_history.append(state_vec.copy())
            signature_vec = _compute_online_signature_prefix(
                state_history=state_history,
                sig_depth=int(cfg.signature_depth),
                signature_backend=str(signature_backend),
            )
            if signature_vec.shape[0] != int(cfg.signature_dim):
                raise RuntimeError(
                    "Online signature dimension mismatch during Meta-World eval: "
                    f"got {signature_vec.shape[0]}, expected {cfg.signature_dim}."
                )
            observation[DEFAULT_PATH_SIGNATURE_KEY] = torch.from_numpy(
                signature_vec.astype("float32", copy=False)
            ).unsqueeze(0)
            if use_delta_signature:
                delta_signature_vec = compute_delta_signature_step_np(
                    signature_vec,
                    previous_signature_vec,
                )
                observation[DEFAULT_DELTA_SIGNATURE_KEY] = torch.from_numpy(
                    delta_signature_vec.astype("float32", copy=False)
                ).unsqueeze(0)
                previous_signature_vec = signature_vec.astype("float32", copy=True)

        if build_explicit_prefix_eval_inputs:
            assert prefix_state_history is not None
            assert prefix_image_histories is not None

            prefix_state_history.append(state[0].detach().clone())
            prefix_state, prefix_mask = build_padded_prefix_from_history(
                prefix_state_history,
                prefix_train_max_steps=int(cfg.prefix_train_max_steps),
                prefix_frame_stride=int(cfg.prefix_frame_stride),
                pad_value=float(cfg.prefix_pad_value),
            )
            observation[PREFIX_STATE_KEY] = prefix_state.unsqueeze(0)
            observation[PREFIX_MASK_KEY] = prefix_mask.unsqueeze(0)

            if use_path_signature:
                assert prefix_signature_history is not None
                if DEFAULT_PATH_SIGNATURE_KEY not in observation:
                    raise KeyError(
                        f"`{DEFAULT_PATH_SIGNATURE_KEY}` missing before prefix construction."
                    )
                prefix_signature_history.append(
                    observation[DEFAULT_PATH_SIGNATURE_KEY][0].detach().clone()
                )
                prefix_signature, prefix_signature_mask = build_padded_prefix_from_history(
                    prefix_signature_history,
                    prefix_train_max_steps=int(cfg.prefix_train_max_steps),
                    prefix_frame_stride=int(cfg.prefix_frame_stride),
                    pad_value=float(cfg.prefix_pad_value),
                )
                if not torch.equal(prefix_mask, prefix_signature_mask):
                    raise RuntimeError(
                        "Prefix state/signature masks diverged during Meta-World eval."
                    )
                observation[PREFIX_PATH_SIGNATURE_KEY] = prefix_signature.unsqueeze(0)

            if use_delta_signature:
                assert prefix_delta_signature_history is not None
                if DEFAULT_DELTA_SIGNATURE_KEY not in observation:
                    raise KeyError(
                        f"`{DEFAULT_DELTA_SIGNATURE_KEY}` missing before prefix construction."
                    )
                prefix_delta_signature_history.append(
                    observation[DEFAULT_DELTA_SIGNATURE_KEY][0].detach().clone()
                )
                (
                    prefix_delta_signature,
                    prefix_delta_signature_mask,
                ) = build_padded_prefix_from_history(
                    prefix_delta_signature_history,
                    prefix_train_max_steps=int(cfg.prefix_train_max_steps),
                    prefix_frame_stride=int(cfg.prefix_frame_stride),
                    pad_value=float(cfg.prefix_pad_value),
                )
                if not torch.equal(prefix_mask, prefix_delta_signature_mask):
                    raise RuntimeError(
                        "Prefix state/delta-signature masks diverged during Meta-World eval."
                    )
                observation[PREFIX_DELTA_SIGNATURE_KEY] = (
                    prefix_delta_signature.unsqueeze(0)
                )

            for visual_key in visual_keys:
                if visual_key not in observation:
                    raise KeyError(
                        "Meta-World eval observation is missing required visual key "
                        f"`{visual_key}` for explicit prefix-sequence inputs."
                    )
                prefix_image_histories[visual_key].append(
                    observation[visual_key][0].detach().clone()
                )
                prefix_images, prefix_image_mask = build_padded_prefix_from_history(
                    prefix_image_histories[visual_key],
                    prefix_train_max_steps=int(cfg.prefix_train_max_steps),
                    prefix_frame_stride=int(cfg.prefix_frame_stride),
                    pad_value=0.0,
                )
                if not torch.equal(prefix_mask, prefix_image_mask):
                    raise RuntimeError(
                        "Prefix state/image masks diverged during Meta-World eval."
                    )
                observation[prefix_image_key_from_camera_key(visual_key)] = (
                    prefix_images.unsqueeze(0)
                )

        processed = preprocessor(observation)
        if use_path_signature:
            if DEFAULT_PATH_SIGNATURE_KEY not in processed:
                raise KeyError(
                    f"`{DEFAULT_PATH_SIGNATURE_KEY}` missing after preprocessor."
                )
            path_signature = processed[DEFAULT_PATH_SIGNATURE_KEY]
            if path_signature.ndim == 1:
                path_signature = path_signature.unsqueeze(0)
            elif path_signature.ndim != 2:
                raise RuntimeError(
                    f"`{DEFAULT_PATH_SIGNATURE_KEY}` must be 1D/2D after preprocessing, "
                    f"got shape={tuple(path_signature.shape)}"
                )
            processed[DEFAULT_PATH_SIGNATURE_KEY] = path_signature.to(
                device=processed[DEFAULT_STATE_KEY].device,
                dtype=processed[DEFAULT_STATE_KEY].dtype,
            )

        if use_delta_signature:
            if DEFAULT_DELTA_SIGNATURE_KEY not in processed:
                raise KeyError(
                    f"`{DEFAULT_DELTA_SIGNATURE_KEY}` missing after preprocessor."
                )
            delta_signature = processed[DEFAULT_DELTA_SIGNATURE_KEY]
            if delta_signature.ndim == 1:
                delta_signature = delta_signature.unsqueeze(0)
            elif delta_signature.ndim != 2:
                raise RuntimeError(
                    f"`{DEFAULT_DELTA_SIGNATURE_KEY}` must be 1D/2D after preprocessing, "
                    f"got shape={tuple(delta_signature.shape)}"
                )
            processed[DEFAULT_DELTA_SIGNATURE_KEY] = delta_signature.to(
                device=processed[DEFAULT_STATE_KEY].device,
                dtype=processed[DEFAULT_STATE_KEY].dtype,
            )

        if build_explicit_prefix_eval_inputs:
            ensure_prefix_sequence_batch_dims(
                obs=processed,
                state_key=DEFAULT_STATE_KEY,
                image_keys=visual_keys,
                use_path_signature=use_path_signature,
                use_delta_signature=use_delta_signature,
            )

        return processed

    def restore_hooks() -> None:
        policy.reset = original_reset

    return wrapped_preprocessor, restore_hooks


def _flatten_video_paths(eval_info: dict[str, object]) -> list[str]:
    overall = eval_info.get("overall")
    if isinstance(overall, dict):
        video_paths = overall.get("video_paths")
        if isinstance(video_paths, list):
            return [str(path) for path in video_paths]
    return []


def _override_episode_length(envs: dict[str, dict[int, object]], max_steps: int) -> None:
    for task_map in envs.values():
        for vec_env in task_map.values():
            child_envs = getattr(vec_env, "envs", None)
            if not isinstance(child_envs, list):
                continue
            for env in child_envs:
                if hasattr(env, "_max_episode_steps"):
                    env._max_episode_steps = int(max_steps)


def _build_summary(
    *,
    args,
    policy_dir: Path,
    eval_info: dict[str, object],
    task_spec: str,
) -> dict[str, object]:
    overall = eval_info.get("overall", {})
    per_group = eval_info.get("per_group", {})
    per_task = eval_info.get("per_task", [])
    video_paths = _flatten_video_paths(eval_info)

    return {
        "policy_type": args.policy,
        "policy_dir": str(policy_dir),
        "env": "metaworld",
        "task": task_spec,
        "episodes_per_task": int(args.num_rollouts),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "num_tasks": int(len(per_task) if isinstance(per_task, list) else 0),
        "num_episodes": int(overall.get("n_episodes", 0))
        if isinstance(overall, dict)
        else 0,
        "success_rate": (
            float(overall.get("pc_success", 0.0)) / 100.0
            if isinstance(overall, dict)
            else 0.0
        ),
        "metrics": {
            "avg_sum_reward": (
                float(overall.get("avg_sum_reward", 0.0))
                if isinstance(overall, dict)
                else 0.0
            ),
            "avg_max_reward": (
                float(overall.get("avg_max_reward", 0.0))
                if isinstance(overall, dict)
                else 0.0
            ),
            "success_rate_percent": (
                float(overall.get("pc_success", 0.0))
                if isinstance(overall, dict)
                else 0.0
            ),
            "eval_seconds": (
                float(overall.get("eval_s", 0.0))
                if isinstance(overall, dict)
                else 0.0
            ),
            "eval_seconds_per_episode": (
                float(overall.get("eval_ep_s", 0.0))
                if isinstance(overall, dict)
                else 0.0
            ),
        },
        "video_dir": str(args.output_dir.resolve() / "videos"),
        "video_paths": video_paths,
        "per_group": per_group,
        "per_task": per_task,
        "eval_info_path": str(args.output_dir.resolve() / "eval_info.json"),
    }


def evaluate_policy(
    *,
    policy_type: str,
    args,
    policy,
    cfg,
    preprocessor,
    postprocessor,
    policy_dir: Path,
) -> None:
    import json
    import torch
    from lerobot.envs.configs import MetaworldEnv as MetaworldEnvConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs
    from lerobot.scripts.lerobot_eval import eval_policy_all

    if args.num_rollouts <= 0:
        raise ValueError("`--num-rollouts` must be positive for Meta-World eval.")
    if args.max_steps <= 0:
        raise ValueError("`--max-steps` must be positive for Meta-World eval.")
    if args.max_episodes_rendered is not None and args.max_episodes_rendered < 0:
        raise ValueError("`--max-episodes-rendered` must be >= 0 when provided.")

    configure_headless_opengl_defaults()
    patch_lerobot_metaworld_env()

    task_spec = str(args.task or DEFAULT_METAWORLD_TASKS).strip()
    max_episodes_rendered = (
        int(args.max_episodes_rendered)
        if args.max_episodes_rendered is not None
        else int(args.num_rollouts)
    )

    env_cfg = MetaworldEnvConfig(
        task=task_spec,
        fps=int(args.fps),
        episode_length=int(args.max_steps),
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_parallel_tasks=1,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"

    print(
        "[load] Building Meta-World envs: "
        f"task={task_spec}, episodes_per_task={args.num_rollouts}, "
        f"max_steps={args.max_steps}, render_videos_per_task={max_episodes_rendered}"
    )
    envs = make_env(
        env_cfg,
        n_envs=1,
        use_async_envs=False,
        trust_remote_code=False,
    )
    _override_episode_length(envs, int(args.max_steps))
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=cfg,
    )
    wrapped_preprocessor = preprocessor
    restore_eval_hooks = lambda: None
    if policy_type == "streaming_act":
        wrapped_preprocessor, restore_eval_hooks = _install_streaming_act_eval_hooks(
            policy=policy,
            cfg=cfg,
            preprocessor=preprocessor,
            args=args,
        )

    try:
        device_type = str(getattr(policy.config, "device", args.device)).split(":", 1)[0]
        use_amp = bool(getattr(cfg, "use_amp", False)) and device_type == "cuda"
        with torch.no_grad(), (
            torch.autocast(device_type=device_type) if use_amp else nullcontext()
        ):
            eval_info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=wrapped_preprocessor,
                postprocessor=postprocessor,
                n_episodes=int(args.num_rollouts),
                max_episodes_rendered=max_episodes_rendered,
                videos_dir=videos_dir,
                start_seed=int(args.seed),
                max_parallel_tasks=int(getattr(env_cfg, "max_parallel_tasks", 1)),
            )
    finally:
        restore_eval_hooks()
        close_envs(envs)

    eval_info_path = output_dir / "eval_info.json"
    eval_info_path.write_text(
        json.dumps(eval_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = _build_summary(
        args=args,
        policy_dir=policy_dir,
        eval_info=eval_info,
        task_spec=task_spec,
    )
    summary_path = write_summary(output_dir, summary)

    print("\nOverall Aggregated Metrics:")
    print(eval_info["overall"])
    print(f"\nSummary: {summary_path}")
    print(f"Eval info: {eval_info_path}")
    print(f"Videos: {videos_dir}")
    print(
        "Success rate: "
        f"{summary['success_rate']:.3f} "
        f"({summary['metrics']['success_rate_percent']:.1f}%)"
    )
