import argparse
from pathlib import Path

from env import get_env_choices, get_env_module


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env", choices=get_env_choices(), default="panda_route")
    known_args, _ = bootstrap.parse_known_args(argv)
    env_module = get_env_module(known_args.env)
    if not hasattr(env_module, "get_replay_defaults"):
        raise RuntimeError(
            f"Environment {known_args.env!r} does not provide replay defaults."
        )
    defaults = env_module.get_replay_defaults()

    parser = argparse.ArgumentParser(
        description="Launch an interactive dataset replay viewer.",
    )
    parser.add_argument("--env", choices=get_env_choices(), default=known_args.env)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=("dataset_path" not in defaults or defaults["dataset_path"] is None),
        default=defaults.get("dataset_path"),
        help="Path to a panda_route raw or processed .npz dataset.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Index inside the filtered replay episode list.",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help="Select by source episode_id instead of filtered list index.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Filter replay episodes by task id before selection.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only keep successful episodes before selection.",
    )
    parser.add_argument(
        "--list-episodes",
        action="store_true",
        help="Print the filtered replay episode list and exit.",
    )
    parser.add_argument(
        "--max-list-episodes",
        type=int,
        default=defaults["max_list_episodes"],
        help="Maximum number of replay episodes to print in the listing.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=defaults["fps"],
        help="Replay speed in frames per second.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=defaults["image_size"],
        help="Offscreen render size used by the environment renderer.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the current episode when it reaches the end.",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Open the viewer with replay paused on the first frame.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="If no episode is specified, start from the first filtered episode without prompting.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(argv)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    env_module = get_env_module(args.env)
    if not hasattr(env_module, "replay_dataset_with_viewer"):
        raise RuntimeError(
            f"Environment {args.env!r} does not implement interactive replay."
        )
    env_module.replay_dataset_with_viewer(args)


if __name__ == "__main__":
    main()
