#!/usr/bin/env python3

import os
import sys

import numpy as np

import lerobot.scripts.lerobot_eval as base_eval
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


DEFAULT_MAX_EPISODES_RENDERED = 10
MAX_EPISODES_RENDERED = DEFAULT_MAX_EPISODES_RENDERED


def configure_headless_opengl_defaults() -> None:
    # Prefer EGL by default for Meta-World eval unless the user already chose a backend.
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
    if os.environ.get("MUJOCO_GL") == "egl" and not os.environ.get("PYOPENGL_PLATFORM"):
        os.environ["PYOPENGL_PLATFORM"] = "egl"


def patch_lerobot_metaworld_env() -> None:
    from gymnasium import spaces
    from lerobot.envs.metaworld import MetaworldEnv

    if getattr(MetaworldEnv, "_corl_environment_state_patch", False):
        return

    original_init = MetaworldEnv.__init__
    original_format_raw_obs = MetaworldEnv._format_raw_obs

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        raw_obs_space = getattr(self._env, "observation_space", None)
        raw_obs_shape = tuple(getattr(raw_obs_space, "shape", ()) or ())
        if not raw_obs_shape:
            return

        obs_spaces = getattr(self.observation_space, "spaces", None)
        if not isinstance(obs_spaces, dict) or "environment_state" in obs_spaces:
            return

        patched_spaces = dict(obs_spaces)
        patched_spaces["environment_state"] = spaces.Box(
            low=np.full(raw_obs_shape, -np.inf, dtype=np.float32),
            high=np.full(raw_obs_shape, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(patched_spaces)

    def patched_format_raw_obs(self, raw_obs):
        observation = original_format_raw_obs(self, raw_obs)
        if "environment_state" not in observation:
            observation = dict(observation)
            observation["environment_state"] = np.asarray(
                raw_obs,
                dtype=np.float32,
            ).reshape(-1).copy()
        return observation

    MetaworldEnv.__init__ = patched_init
    MetaworldEnv._format_raw_obs = patched_format_raw_obs
    MetaworldEnv._corl_environment_state_patch = True


def extract_local_cli_args(argv: list[str]) -> tuple[int, list[str]]:
    passthrough_args = [argv[0]]
    max_episodes_rendered = DEFAULT_MAX_EPISODES_RENDERED

    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg.startswith("--max-episodes-rendered="):
            value_text = arg.split("=", 1)[1]
        elif arg == "--max-episodes-rendered":
            if index + 1 >= len(argv):
                raise SystemExit("Missing value for `--max-episodes-rendered`.")
            index += 1
            value_text = argv[index]
        else:
            passthrough_args.append(arg)
            index += 1
            continue

        try:
            max_episodes_rendered = int(value_text)
        except ValueError as exc:
            raise SystemExit(
                "`--max-episodes-rendered` must be an integer."
            ) from exc
        if max_episodes_rendered < 0:
            raise SystemExit(
                "`--max-episodes-rendered` must be >= 0."
            )
        index += 1

    return max_episodes_rendered, passthrough_args


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig) -> None:
    base_eval.logging.info(base_eval.pformat(base_eval.asdict(cfg)))

    device = base_eval.get_safe_torch_device(cfg.policy.device, log=True)

    base_eval.torch.backends.cudnn.benchmark = True
    base_eval.torch.backends.cuda.matmul.allow_tf32 = True
    base_eval.set_seed(cfg.seed)

    base_eval.logging.info(
        base_eval.colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}"
    )
    base_eval.logging.info(
        f"Rendering up to {MAX_EPISODES_RENDERED} episode videos per task."
    )

    base_eval.logging.info("Making environment.")
    envs = base_eval.make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    base_eval.logging.info("Making policy.")
    policy = base_eval.make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = base_eval.make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    env_preprocessor, env_postprocessor = base_eval.make_env_pre_post_processors(
        env_cfg=cfg.env,
        policy_cfg=cfg.policy,
    )

    with base_eval.torch.no_grad(), (
        base_eval.torch.autocast(device_type=device.type)
        if cfg.policy.use_amp
        else base_eval.nullcontext()
    ):
        info = base_eval.eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=MAX_EPISODES_RENDERED,
            videos_dir=base_eval.Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)

    base_eval.close_envs(envs)

    with open(base_eval.Path(cfg.output_dir) / "eval_info.json", "w") as f:
        base_eval.json.dump(info, f, indent=2)

    base_eval.logging.info("End of eval")


def main() -> None:
    global MAX_EPISODES_RENDERED

    MAX_EPISODES_RENDERED, sys.argv = extract_local_cli_args(sys.argv)
    configure_headless_opengl_defaults()
    init_logging()
    register_third_party_plugins()
    patch_lerobot_metaworld_env()

    eval_main()


if __name__ == "__main__":
    main()
