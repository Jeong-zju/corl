#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_STATE_FILENAME = ".hf_checkpoint_upload_state.json"

PRETRAINED_MODEL_DIRNAME = "pretrained_model"
TRAINING_STATE_DIRNAME = "training_state"
TRAIN_CONFIG_FILENAME = "train_config.json"
TRAINING_STEP_FILENAME = "training_step.json"
MODEL_FILENAME = "model.safetensors"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload local training checkpoints to a Hugging Face model repository. "
            "Supports one-shot upload and watch mode."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Training run directory containing a `checkpoints/` subdirectory.",
    )
    source_group.add_argument(
        "--train-output-root",
        type=str,
        default=None,
        help=(
            "Directory containing timestamped training runs. The newest run is "
            "selected automatically."
        ),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target Hugging Face model repo id, e.g. `user/my-policy`.",
    )
    visibility_group = parser.add_mutually_exclusive_group()
    visibility_group.add_argument(
        "--private",
        action="store_true",
        help="Create the remote model repository as private if it does not exist.",
    )
    visibility_group.add_argument(
        "--public",
        action="store_true",
        help="Create the remote model repository as public if it does not exist.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Target branch or revision. Defaults to `main`.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token. Falls back to cached login or `HF_TOKEN`.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "pretrained_model"),
        default="full",
        help=(
            "`full` uploads the whole checkpoint directory. `pretrained_model` only "
            "uploads the exported model folder."
        ),
    )
    parser.add_argument(
        "--remote-prefix",
        type=str,
        default=None,
        help=(
            "Optional prefix inside the remote repo. By default, the script uses "
            "the local run path relative to the project root."
        ),
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help=(
            "Optional JSON file used to remember uploaded checkpoints. Defaults to "
            "`<run_dir>/.hf_checkpoint_upload_state.json`."
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll for newly created checkpoints and upload them.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Polling interval in seconds for `--watch`. Defaults to 30.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the resolved upload plan without contacting Hugging Face.",
    )
    return parser


def require_huggingface_hub():
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`huggingface_hub` is required to upload checkpoints. Activate the "
            "intended environment and install it before running this script."
        ) from exc
    return HfApi


def resolve_visibility(*, private: bool, public: bool) -> bool | None:
    if private:
        return True
    if public:
        return False
    return None


def resolve_path(raw_path: str) -> Path:
    raw = Path(raw_path).expanduser()
    candidates = (
        [raw]
        if raw.is_absolute()
        else [Path.cwd() / raw, PROJECT_ROOT / raw, REPO_ROOT / raw]
    )
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.exists():
            return resolved
    return candidates[0].resolve(strict=False)


def list_run_dirs(train_output_root: Path) -> list[Path]:
    if not train_output_root.is_dir():
        raise FileNotFoundError(
            f"Training output root does not exist or is not a directory: {train_output_root}"
        )
    run_dirs = [path.resolve(strict=False) for path in train_output_root.iterdir() if path.is_dir()]
    run_dirs.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    return run_dirs


def resolve_latest_run_dir(train_output_root: Path) -> Path:
    run_dirs = list_run_dirs(train_output_root)
    if not run_dirs:
        raise FileNotFoundError(
            f"No training run directories were found under: {train_output_root}"
        )
    return run_dirs[0]


def resolve_current_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        run_dir = resolve_path(args.run_dir)
    else:
        run_dir = resolve_latest_run_dir(resolve_path(args.train_output_root))

    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not contain `checkpoints/`: {run_dir}"
        )
    return run_dir


def resolve_state_file(run_dir: Path, raw_state_file: str | None) -> Path:
    if raw_state_file is None:
        return run_dir / DEFAULT_STATE_FILENAME
    return resolve_path(raw_state_file)


def load_training_step(checkpoint_dir: Path) -> int | None:
    step_path = checkpoint_dir / TRAINING_STATE_DIRNAME / TRAINING_STEP_FILENAME
    if not step_path.is_file():
        return int(checkpoint_dir.name) if checkpoint_dir.name.isdigit() else None

    try:
        payload = json.loads(step_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return int(checkpoint_dir.name) if checkpoint_dir.name.isdigit() else None

    step = payload.get("step")
    return int(step) if isinstance(step, int) else None


def checkpoint_is_ready(checkpoint_dir: Path, *, mode: str) -> bool:
    pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIRNAME
    if not (
        pretrained_model_dir.is_dir()
        and (pretrained_model_dir / TRAIN_CONFIG_FILENAME).is_file()
        and (pretrained_model_dir / MODEL_FILENAME).is_file()
    ):
        return False

    if mode == "pretrained_model":
        return True

    training_state_dir = checkpoint_dir / TRAINING_STATE_DIRNAME
    return (
        training_state_dir.is_dir()
        and (training_state_dir / TRAINING_STEP_FILENAME).is_file()
    )


def list_ready_checkpoint_dirs(run_dir: Path, *, mode: str) -> list[Path]:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_dirs = [
        path.resolve(strict=False)
        for path in checkpoints_dir.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and checkpoint_is_ready(path, mode=mode)
    ]
    checkpoint_dirs.sort(key=lambda path: int(path.name))
    return checkpoint_dirs


def relative_run_path(run_dir: Path) -> PurePosixPath:
    try:
        relative = run_dir.resolve(strict=False).relative_to(
            PROJECT_ROOT.resolve(strict=False)
        )
        return PurePosixPath(relative.as_posix())
    except ValueError:
        return PurePosixPath(run_dir.name)


def format_display_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve(strict=False)).as_posix()
    except ValueError:
        return str(resolved)


def build_remote_path(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    mode: str,
    remote_prefix: str | None,
) -> PurePosixPath:
    run_prefix = PurePosixPath(remote_prefix) if remote_prefix else relative_run_path(run_dir)
    checkpoint_root = run_prefix / "checkpoints" / checkpoint_dir.name
    if mode == "pretrained_model":
        return checkpoint_root / PRETRAINED_MODEL_DIRNAME
    return checkpoint_root


def load_state(state_file: Path) -> dict[str, Any]:
    if not state_file.is_file():
        return {"uploaded_checkpoints": []}

    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"uploaded_checkpoints": []}

    if not isinstance(payload, dict):
        return {"uploaded_checkpoints": []}
    uploaded = payload.get("uploaded_checkpoints")
    if not isinstance(uploaded, list):
        payload["uploaded_checkpoints"] = []
    return payload


def save_state(
    *,
    state_file: Path,
    run_dir: Path,
    repo_id: str,
    mode: str,
    uploaded_checkpoints: set[str],
) -> None:
    payload = {
        "run_dir": str(run_dir),
        "repo_id": repo_id,
        "mode": mode,
        "uploaded_checkpoints": sorted(uploaded_checkpoints, key=int),
    }
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_commit_message(checkpoint_dir: Path) -> str:
    step = load_training_step(checkpoint_dir)
    if step is None:
        return f"Upload checkpoint {checkpoint_dir.name}"
    return f"Upload checkpoint step {step}"


def build_commit_description(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    mode: str,
) -> str:
    step = load_training_step(checkpoint_dir)
    lines = [
        f"Run dir: {run_dir}",
        f"Checkpoint dir: {checkpoint_dir}",
        f"Mode: {mode}",
    ]
    if step is not None:
        lines.append(f"Training step: {step}")
    return "\n".join(lines)


def print_plan(
    *,
    run_dir: Path,
    repo_id: str,
    revision: str,
    mode: str,
    remote_prefix: str | None,
    state_file: Path,
    watch: bool,
    poll_interval: float,
    checkpoints: list[Path],
) -> None:
    entries = []
    for checkpoint_dir in checkpoints:
        entries.append(
            {
                "checkpoint_dir": format_display_path(checkpoint_dir),
                "step": load_training_step(checkpoint_dir),
                "remote_path": str(
                    build_remote_path(
                        run_dir=run_dir,
                        checkpoint_dir=checkpoint_dir,
                        mode=mode,
                        remote_prefix=remote_prefix,
                    )
                ),
            }
        )

    plan = {
        "run_dir": format_display_path(run_dir),
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
        "mode": mode,
        "remote_prefix": remote_prefix,
        "state_file": format_display_path(state_file),
        "watch": watch,
        "poll_interval": poll_interval,
        "ready_checkpoints": entries,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def upload_checkpoint(
    *,
    api: Any,
    repo_id: str,
    revision: str,
    token: str | None,
    run_dir: Path,
    checkpoint_dir: Path,
    mode: str,
    remote_prefix: str | None,
) -> None:
    folder_path = (
        checkpoint_dir / PRETRAINED_MODEL_DIRNAME
        if mode == "pretrained_model"
        else checkpoint_dir
    )
    path_in_repo = build_remote_path(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        mode=mode,
        remote_prefix=remote_prefix,
    )
    step = load_training_step(checkpoint_dir)
    print(
        f"[upload] checkpoint={checkpoint_dir.name} "
        f"step={step if step is not None else 'unknown'} "
        f"src={format_display_path(folder_path)} dst={path_in_repo}"
    )
    commit_info = api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        path_in_repo=str(path_in_repo),
        commit_message=build_commit_message(checkpoint_dir),
        commit_description=build_commit_description(
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            mode=mode,
        ),
        token=token,
    )
    commit_url = getattr(commit_info, "commit_url", None)
    if commit_url:
        print(f"[upload] finished: {commit_url}")
    else:
        print(f"[upload] finished: https://huggingface.co/{repo_id}/tree/{revision}")


def upload_pending_checkpoints(
    *,
    api: Any,
    args: argparse.Namespace,
    run_dir: Path,
    state_file: Path,
) -> int:
    state = load_state(state_file)
    uploaded_checkpoints = {
        str(item)
        for item in state.get("uploaded_checkpoints", [])
        if isinstance(item, (str, int))
    }

    uploaded_this_round = 0
    for checkpoint_dir in list_ready_checkpoint_dirs(run_dir, mode=args.mode):
        if checkpoint_dir.name in uploaded_checkpoints:
            continue
        upload_checkpoint(
            api=api,
            repo_id=args.repo_id,
            revision=args.revision,
            token=args.token,
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            mode=args.mode,
            remote_prefix=args.remote_prefix,
        )
        uploaded_checkpoints.add(checkpoint_dir.name)
        uploaded_this_round += 1
        save_state(
            state_file=state_file,
            run_dir=run_dir,
            repo_id=args.repo_id,
            mode=args.mode,
            uploaded_checkpoints=uploaded_checkpoints,
        )
    return uploaded_this_round


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    visibility = resolve_visibility(private=args.private, public=args.public)

    try:
        initial_run_dir = resolve_current_run_dir(args)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    initial_state_file = resolve_state_file(initial_run_dir, args.state_file)
    initial_checkpoints = list_ready_checkpoint_dirs(initial_run_dir, mode=args.mode)

    print_plan(
        run_dir=initial_run_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        mode=args.mode,
        remote_prefix=args.remote_prefix,
        state_file=initial_state_file,
        watch=bool(args.watch),
        poll_interval=float(args.poll_interval),
        checkpoints=initial_checkpoints,
    )

    if args.dry_run:
        return 0

    HfApi = require_huggingface_hub()
    api = HfApi()
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=visibility,
        exist_ok=True,
        token=args.token,
    )
    print(f"Model repo ready: {repo_url}")

    while True:
        run_dir = resolve_current_run_dir(args)
        state_file = resolve_state_file(run_dir, args.state_file)
        uploaded_count = upload_pending_checkpoints(
            api=api,
            args=args,
            run_dir=run_dir,
            state_file=state_file,
        )

        if not args.watch:
            if uploaded_count == 0:
                print("[upload] no new ready checkpoints found.")
            return 0

        if uploaded_count == 0:
            print(
                f"[watch] no new checkpoints under "
                f"{format_display_path(run_dir / 'checkpoints')}; "
                f"sleeping for {max(args.poll_interval, 1.0):.1f}s"
            )
        time.sleep(max(args.poll_interval, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
