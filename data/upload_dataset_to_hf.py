#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_ROOT = SCRIPT_DIR.parent
DATA_ROOT = MAIN_ROOT / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a local dataset directory to a Hugging Face dataset repository. "
            "Relative dataset paths are resolved from `data/`."
        )
    )
    parser.add_argument(
        "dataset",
        type=str,
        nargs="?",
        help=(
            "Local dataset directory, a path under `data/`, or a path using "
            "`data/...` prefixes."
        ),
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_option",
        type=str,
        default=None,
        help="Alias of the positional dataset argument.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "Target Hugging Face dataset repo id. If omitted, the script tries to "
            "infer it from a `data/<namespace>/<name>` path."
        ),
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repository as private if it does not already exist.",
    )
    visibility.add_argument(
        "--public",
        action="store_true",
        help="Create the dataset repository as public if it does not already exist.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Target branch, tag, or revision to upload to. Defaults to `main`.",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default=None,
        help="Optional destination subdirectory inside the remote dataset repository.",
    )
    parser.add_argument(
        "--include",
        dest="include_patterns",
        nargs="+",
        action="append",
        default=None,
        help="Only upload files matching these glob patterns. Can be repeated.",
    )
    parser.add_argument(
        "--exclude",
        dest="exclude_patterns",
        nargs="+",
        action="append",
        default=None,
        help="Skip files matching these glob patterns. Can be repeated.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face user access token. If omitted, the script uses "
            "the locally cached login or `HF_TOKEN` environment variable."
        ),
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message for regular folder uploads.",
    )
    parser.add_argument(
        "--commit-description",
        type=str,
        default=None,
        help="Commit description for regular folder uploads.",
    )
    parser.add_argument(
        "--large-folder",
        action="store_true",
        help=(
            "Use `upload_large_folder()` for resumable multi-commit uploads. "
            "Recommended for very large datasets."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker count for `--large-folder` uploads.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and print the upload plan without creating a repo or uploading.",
    )
    return parser


def resolve_dataset_arg(
    parser: argparse.ArgumentParser,
    *,
    dataset: str | None,
    dataset_option: str | None,
) -> str:
    if dataset is not None and dataset_option is not None:
        parser.error("Specify the dataset either positionally or via --dataset, not both.")
    resolved = dataset_option if dataset_option is not None else dataset
    if resolved is None:
        parser.error("the following arguments are required: dataset")
    return resolved


def is_lerobot_dataset_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "meta/info.json").exists()
        and (path / "meta/stats.json").exists()
        and (path / "data").is_dir()
    )


def find_lerobot_dataset_root(candidate: Path) -> Path | None:
    expanded = candidate.expanduser()

    direct_candidates = [expanded]
    if expanded.is_file() and expanded.name == "info.json" and expanded.parent.name == "meta":
        direct_candidates.append(expanded.parent.parent)
    if expanded.is_file():
        direct_candidates.append(expanded.parent)

    for direct_candidate in direct_candidates:
        for parent in [direct_candidate, *direct_candidate.parents]:
            if is_lerobot_dataset_root(parent):
                return parent.resolve()
    return None


def build_candidate_paths(
    dataset: str,
    *,
    cwd: Path | None = None,
    main_root: Path = MAIN_ROOT,
    data_root: Path = DATA_ROOT,
) -> list[Path]:
    working_directory = cwd or Path.cwd()
    raw = Path(dataset).expanduser()
    if raw.is_absolute():
        return [raw]

    normalized = raw.as_posix()
    candidates: list[Path] = []
    if raw.exists() or normalized.startswith("."):
        candidates.append(working_directory / raw)
    if normalized.startswith("data/"):
        candidates.append(main_root / raw)
    candidates.extend(
        [
            main_root / raw,
            data_root / raw,
        ]
    )
    return candidates


def _resolve_existing_candidate(candidate: Path) -> Path | None:
    if not candidate.exists():
        return None

    dataset_root = find_lerobot_dataset_root(candidate)
    if dataset_root is not None:
        return dataset_root
    return candidate.resolve()


def resolve_local_upload_path(
    dataset: str,
    *,
    cwd: Path | None = None,
    main_root: Path = MAIN_ROOT,
    data_root: Path = DATA_ROOT,
) -> Path:
    ordered_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in build_candidate_paths(
        dataset,
        cwd=cwd,
        main_root=main_root,
        data_root=data_root,
    ):
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        resolved = _resolve_existing_candidate(candidate)
        if resolved is not None:
            return resolved

    dataset_text = str(dataset).strip()
    if "/" not in dataset_text and "\\" not in dataset_text:
        matches: list[Path] = []
        matched_seen: set[Path] = set()
        for candidate in sorted(data_root.glob(f"**/{dataset_text}")):
            resolved = _resolve_existing_candidate(candidate)
            if resolved is None or resolved in matched_seen:
                continue
            matched_seen.add(resolved)
            matches.append(resolved)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            candidate_text = "\n".join(f"- {path}" for path in matches)
            raise FileNotFoundError(
                "The dataset argument matched multiple local paths under `data/`. "
                "Please pass a more specific path or `--repo-id` explicitly.\n"
                f"{candidate_text}"
            )

    checked = "\n".join(f"- {path.resolve(strict=False)}" for path in ordered_candidates)
    raise FileNotFoundError(
        "Could not resolve a local dataset path from the provided argument. "
        "Checked these locations:\n"
        f"{checked}"
    )


def normalize_patterns(groups: list[list[str]] | None) -> list[str] | None:
    if not groups:
        return None

    normalized: list[str] = []
    for group in groups:
        for pattern in group:
            stripped = pattern.strip()
            if stripped:
                normalized.append(stripped)
    return normalized or None


def normalize_path_in_repo(path_in_repo: str | None) -> str | None:
    if path_in_repo is None:
        return None
    normalized = path_in_repo.strip().replace("\\", "/")
    if normalized in {"", ".", "./"}:
        return None
    if normalized.startswith("/"):
        raise ValueError("`--path-in-repo` must be a relative path inside the dataset repo.")
    return normalized.rstrip("/")


def resolve_repo_visibility(*, private: bool, public: bool) -> bool | None:
    if private:
        return True
    if public:
        return False
    return None


def infer_default_repo_id(
    local_path: Path,
    *,
    data_root: Path = DATA_ROOT,
) -> str | None:
    try:
        relative = local_path.resolve().relative_to(data_root.resolve())
    except ValueError:
        return None

    parts = relative.parts
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return None


def resolve_repo_id(
    local_path: Path,
    *,
    repo_id: str | None,
    data_root: Path = DATA_ROOT,
) -> str:
    if repo_id is not None:
        normalized = repo_id.strip()
        if not normalized:
            raise ValueError("`--repo-id` must not be empty.")
        return normalized

    inferred = infer_default_repo_id(local_path, data_root=data_root)
    if inferred is not None:
        return inferred

    raise ValueError(
        "Could not infer a Hugging Face repo id from the local path. "
        "Pass `--repo-id` explicitly. Automatic inference only works for paths "
        "under `data/<name>` or `data/<namespace>/<name>`."
    )


def validate_upload_options(
    *,
    large_folder: bool,
    path_in_repo: str | None,
    commit_message: str | None,
    commit_description: str | None,
    num_workers: int | None,
) -> None:
    if num_workers is not None and num_workers <= 0:
        raise ValueError("`--num-workers` must be a positive integer.")
    if not large_folder:
        return
    if path_in_repo is not None:
        raise ValueError("`--large-folder` does not support `--path-in-repo`.")
    if commit_message is not None:
        raise ValueError("`--large-folder` does not support `--commit-message`.")
    if commit_description is not None:
        raise ValueError("`--large-folder` does not support `--commit-description`.")


def load_local_dataset_summary(local_path: Path) -> dict[str, Any]:
    info_path = local_path / "meta" / "info.json"
    if not info_path.exists():
        return {}

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    summary: dict[str, Any] = {}
    for key in ("codebase_version", "robot_type", "fps", "total_episodes", "total_frames"):
        value = info.get(key)
        if value is not None:
            summary[key] = value
    return summary


def build_commit_message(local_path: Path, repo_id: str) -> str:
    return f"Upload dataset `{local_path.name}` to `{repo_id}`"


def print_upload_plan(
    *,
    local_path: Path,
    repo_id: str,
    revision: str,
    path_in_repo: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    large_folder: bool,
    private: bool | None,
) -> None:
    plan = {
        "local_path": str(local_path),
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "path_in_repo": path_in_repo,
        "mode": "upload_large_folder" if large_folder else "upload_folder",
        "allow_patterns": allow_patterns,
        "ignore_patterns": ignore_patterns,
        "private_on_create": private,
        "local_summary": load_local_dataset_summary(local_path),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def require_huggingface_hub():
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`huggingface_hub` is required to upload datasets. Activate the intended "
            "environment and install it before running this script."
        ) from exc
    return HfApi


def run_upload(
    *,
    local_path: Path,
    repo_id: str,
    revision: str,
    path_in_repo: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    token: str | None,
    commit_message: str | None,
    commit_description: str | None,
    large_folder: bool,
    num_workers: int | None,
    private: bool | None,
) -> None:
    HfApi = require_huggingface_hub()
    api = HfApi()

    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )
    print(f"Dataset repo ready: {repo_url}")

    if large_folder:
        upload_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "folder_path": str(local_path),
            "revision": revision,
            "private": private,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if num_workers is not None:
            upload_kwargs["num_workers"] = num_workers
        api.upload_large_folder(**upload_kwargs)
        print(f"Large-folder upload finished: https://huggingface.co/datasets/{repo_id}")
        return

    commit_info = api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        commit_message=commit_message,
        commit_description=commit_description,
        token=token,
    )
    commit_url = getattr(commit_info, "commit_url", None)
    if commit_url:
        print(f"Upload finished: {commit_url}")
    else:
        print(f"Upload finished for https://huggingface.co/datasets/{repo_id}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_arg = resolve_dataset_arg(
        parser,
        dataset=args.dataset,
        dataset_option=args.dataset_option,
    )

    try:
        local_path = resolve_local_upload_path(dataset_arg)
        path_in_repo = normalize_path_in_repo(args.path_in_repo)
        allow_patterns = normalize_patterns(args.include_patterns)
        ignore_patterns = normalize_patterns(args.exclude_patterns)
        private = resolve_repo_visibility(private=args.private, public=args.public)
        repo_id = resolve_repo_id(local_path, repo_id=args.repo_id)
        validate_upload_options(
            large_folder=args.large_folder,
            path_in_repo=path_in_repo,
            commit_message=args.commit_message,
            commit_description=args.commit_description,
            num_workers=args.num_workers,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))

    if not local_path.is_dir():
        parser.error(f"Expected a local directory to upload, got: {local_path}")

    commit_message = args.commit_message
    if commit_message is None and not args.large_folder:
        commit_message = build_commit_message(local_path, repo_id)

    print_upload_plan(
        local_path=local_path,
        repo_id=repo_id,
        revision=args.revision,
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        large_folder=args.large_folder,
        private=private,
    )

    if args.dry_run:
        return 0

    run_upload(
        local_path=local_path,
        repo_id=repo_id,
        revision=args.revision,
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        token=args.token,
        commit_message=commit_message,
        commit_description=args.commit_description,
        large_folder=args.large_folder,
        num_workers=args.num_workers,
        private=private,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
