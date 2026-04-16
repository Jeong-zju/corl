from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_LOCAL_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_LEROBOT_V30_COMPAT_CACHE_ROOT = (
    Path(__file__).resolve().parents[1] / ".cache" / "lerobot_v30"
)
DATASET_SPLIT_FILENAME = "dataset_split.json"
LEGACY_EPISODES_JSONL_PATH = Path("meta/episodes.jsonl")
LEGACY_EPISODES_STATS_JSONL_PATH = Path("meta/episodes_stats.jsonl")
LEGACY_TASKS_JSONL_PATH = Path("meta/tasks.jsonl")
LEROBOT_V30_COMPAT_METADATA_FILENAME = ".lerobot_v30_compat.json"


@dataclass(slots=True)
class DatasetSplitSpec:
    dataset_arg: str
    dataset_root: str
    dataset_repo_id: str
    total_episodes: int
    train_episode_indices: list[int]
    test_episode_indices: list[int]
    test_ratio: float
    split_seed: int
    split_shuffle: bool

    @property
    def train_count(self) -> int:
        return len(self.train_episode_indices)

    @property
    def test_count(self) -> int:
        return len(self.test_episode_indices)


def is_lerobot_dataset_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "meta/info.json").exists()
        and (path / "meta/stats.json").exists()
        and (path / "meta/episodes/chunk-000/file-000.parquet").exists()
        and (path / "data").is_dir()
    )


def is_legacy_lerobot_dataset_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    info_path = path / "meta/info.json"
    data_path = path / "data"
    if not info_path.exists() or not data_path.is_dir():
        return False
    if not (path / LEGACY_EPISODES_JSONL_PATH).exists():
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if str(info.get("codebase_version", "")).strip() != "v2.1":
        return False
    return (
        (path / LEGACY_EPISODES_JSONL_PATH).exists()
        and (path / LEGACY_EPISODES_STATS_JSONL_PATH).exists()
        and (path / LEGACY_TASKS_JSONL_PATH).exists()
    )


def is_supported_lerobot_dataset_root(path: Path) -> bool:
    return is_lerobot_dataset_root(path) or is_legacy_lerobot_dataset_root(path)


def find_lerobot_dataset_root(candidate: Path) -> Path | None:
    raw = candidate.expanduser()
    if is_supported_lerobot_dataset_root(raw):
        return raw.resolve()

    if not raw.exists() or not raw.is_dir():
        return None

    for info_path in sorted(raw.rglob("meta/info.json")):
        root = info_path.parent.parent
        if is_supported_lerobot_dataset_root(root):
            return root.resolve()
    return None


def validate_dataset_root(dataset_root: Path) -> None:
    if is_lerobot_dataset_root(dataset_root):
        return

    required = [
        dataset_root / "meta/info.json",
        dataset_root / "meta/stats.json",
        dataset_root / "meta/episodes/chunk-000/file-000.parquet",
        dataset_root / "data",
    ]
    missing = [path for path in required if not path.exists()]
    missing_s = "\n".join(f"- {path}" for path in missing)
    raise FileNotFoundError(
        "Dataset path is missing required LeRobot v3.0 files:\n"
        f"{missing_s}\n"
        f"dataset_root={dataset_root}"
    )


def _build_lerobot_v30_compat_signature(dataset_root: Path) -> dict[str, object]:
    signature: dict[str, object] = {
        "source_dataset_root": str(dataset_root.resolve()),
    }
    files: dict[str, dict[str, int]] = {}
    for rel_path in (
        Path("meta/info.json"),
        Path("meta/stats.json"),
        LEGACY_EPISODES_JSONL_PATH,
        LEGACY_EPISODES_STATS_JSONL_PATH,
        LEGACY_TASKS_JSONL_PATH,
    ):
        abs_path = dataset_root / rel_path
        if not abs_path.exists():
            continue
        stat = abs_path.stat()
        files[rel_path.as_posix()] = {
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    signature["files"] = files
    return signature


def _load_lerobot_v30_compat_metadata(dataset_root: Path) -> dict[str, object] | None:
    metadata_path = dataset_root / LEROBOT_V30_COMPAT_METADATA_FILENAME
    if not metadata_path.exists():
        return None
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _resolve_lerobot_v30_compat_cache_root(
    dataset_root: Path,
    *,
    dataset_repo_id: str | None,
    local_data_root: Path,
) -> Path:
    relative_parts: tuple[str, ...] = ()
    if dataset_repo_id:
        relative_parts = tuple(
            part
            for part in str(dataset_repo_id).replace("\\", "/").split("/")
            if part not in {"", ".", ".."}
        )
    if not relative_parts:
        try:
            relative_parts = dataset_root.resolve().relative_to(
                local_data_root.resolve()
            ).parts
        except ValueError:
            relative_parts = (dataset_root.name,)
    return DEFAULT_LEROBOT_V30_COMPAT_CACHE_ROOT.joinpath(*relative_parts)


def ensure_lerobot_dataset_v30_compat(
    dataset_root: Path,
    *,
    dataset_repo_id: str | None = None,
    local_data_root: Path = DEFAULT_LOCAL_DATA_ROOT,
) -> Path:
    resolved_root = dataset_root.resolve()
    if is_lerobot_dataset_root(resolved_root):
        return resolved_root
    if not is_legacy_lerobot_dataset_root(resolved_root):
        return resolved_root

    compat_root = _resolve_lerobot_v30_compat_cache_root(
        resolved_root,
        dataset_repo_id=dataset_repo_id,
        local_data_root=local_data_root.resolve(),
    )
    expected_metadata = _build_lerobot_v30_compat_signature(resolved_root)
    cached_metadata = _load_lerobot_v30_compat_metadata(compat_root)
    if (
        compat_root.exists()
        and is_lerobot_dataset_root(compat_root)
        and cached_metadata == expected_metadata
    ):
        return compat_root.resolve()

    compat_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = compat_root.parent / f".{compat_root.name}.tmp"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    if compat_root.exists():
        shutil.rmtree(compat_root)

    compat_hf_cache_root = DEFAULT_LEROBOT_V30_COMPAT_CACHE_ROOT / ".hf_datasets"
    compat_hf_home = DEFAULT_LEROBOT_V30_COMPAT_CACHE_ROOT / ".hf_home"
    compat_xdg_cache_root = DEFAULT_LEROBOT_V30_COMPAT_CACHE_ROOT / ".xdg_cache"
    compat_hf_cache_root.mkdir(parents=True, exist_ok=True)
    compat_hf_home.mkdir(parents=True, exist_ok=True)
    compat_xdg_cache_root.mkdir(parents=True, exist_ok=True)
    previous_hf_home = os.environ.get("HF_HOME")
    previous_hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    previous_xdg_cache_home = os.environ.get("XDG_CACHE_HOME")

    try:
        os.environ["HF_HOME"] = str(compat_hf_home)
        os.environ["HF_DATASETS_CACHE"] = str(compat_hf_cache_root)
        os.environ["XDG_CACHE_HOME"] = str(compat_xdg_cache_root)
        try:
            from lerobot.datasets.utils import (
                DEFAULT_DATA_FILE_SIZE_IN_MB,
                DEFAULT_VIDEO_FILE_SIZE_IN_MB,
            )
            from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
                convert_data,
                convert_episodes_metadata,
                convert_info,
                convert_tasks,
                convert_videos,
                validate_local_dataset_version,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing LeRobot dataset conversion dependencies required to use "
                "legacy v2.1 datasets for training."
            ) from exc
        validate_local_dataset_version(resolved_root)
        convert_info(
            resolved_root,
            temp_root,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        )
        convert_tasks(resolved_root, temp_root)
        episodes_metadata = convert_data(
            resolved_root,
            temp_root,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
        )
        episodes_videos_metadata = convert_videos(
            resolved_root,
            temp_root,
            DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        )
        convert_episodes_metadata(
            resolved_root,
            temp_root,
            episodes_metadata,
            episodes_videos_metadata,
        )
        (temp_root / LEROBOT_V30_COMPAT_METADATA_FILENAME).write_text(
            json.dumps(expected_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        shutil.move(str(temp_root), str(compat_root))
    finally:
        if previous_hf_home is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = previous_hf_home
        if previous_hf_datasets_cache is None:
            os.environ.pop("HF_DATASETS_CACHE", None)
        else:
            os.environ["HF_DATASETS_CACHE"] = previous_hf_datasets_cache
        if previous_xdg_cache_home is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = previous_xdg_cache_home
        if temp_root.exists():
            shutil.rmtree(temp_root)

    return compat_root.resolve()


def resolve_dataset_root(
    dataset: str | Path,
    *,
    local_data_root: Path = DEFAULT_LOCAL_DATA_ROOT,
) -> Path:
    dataset_text = str(dataset).strip()
    normalized_dataset_text = dataset_text.replace("\\", "/")
    raw_path = Path(dataset_text).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute() or raw_path.exists() or dataset_text.startswith("."):
        candidates.append(raw_path)

    if not raw_path.is_absolute():
        if normalized_dataset_text.startswith("main/data/"):
            candidates.append(
                local_data_root / normalized_dataset_text[len("main/data/") :]
            )
        if normalized_dataset_text.startswith("data/"):
            candidates.append(local_data_root / normalized_dataset_text[len("data/") :])
        candidates.append(local_data_root / dataset_text)
        candidates.append(local_data_root / dataset_text.replace("/", "_"))

    seen: set[Path] = set()
    for candidate in candidates:
        found = find_lerobot_dataset_root(candidate)
        if found is not None:
            return found
        seen.add(candidate.resolve(strict=False))

    if "/" not in dataset_text and "\\" not in dataset_text:
        for candidate in sorted(local_data_root.glob(f"**/{dataset_text}")):
            resolved = candidate.resolve(strict=False)
            if resolved in seen:
                continue
            found = find_lerobot_dataset_root(candidate)
            if found is not None:
                return found
            seen.add(resolved)

    candidate_text = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not resolve a local LeRobot dataset root for "
        f"`{dataset_text}`. Checked:\n{candidate_text}\n"
        f"local_data_root={local_data_root}"
    )


def infer_dataset_repo_id(
    dataset_root: Path,
    *,
    local_data_root: Path = DEFAULT_LOCAL_DATA_ROOT,
) -> str:
    resolved_root = dataset_root.resolve()
    try:
        return resolved_root.relative_to(local_data_root.resolve()).as_posix()
    except ValueError:
        return resolved_root.name


def load_dataset_info(dataset_root: Path) -> dict[str, object]:
    return json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))


def get_total_episodes(dataset_root: Path) -> int:
    info = load_dataset_info(dataset_root)
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes <= 0:
        raise ValueError(
            f"Dataset must contain at least 1 episode. "
            f"Got total_episodes={total_episodes} for {dataset_root}."
        )
    return total_episodes


def build_dataset_split(
    *,
    dataset_arg: str,
    dataset_root: Path,
    dataset_repo_id: str,
    test_ratio: float,
    split_seed: int,
    split_shuffle: bool,
) -> DatasetSplitSpec:
    resolved_test_ratio = float(test_ratio)
    if not (0.0 <= resolved_test_ratio < 1.0):
        raise ValueError(f"`test_ratio` must lie in [0, 1), got {test_ratio}.")

    total_episodes = get_total_episodes(dataset_root)
    episode_indices = np.arange(total_episodes, dtype=np.int64)
    if split_shuffle:
        rng = np.random.default_rng(split_seed)
        rng.shuffle(episode_indices)

    if resolved_test_ratio == 0.0:
        test_count = 0
    else:
        if total_episodes <= 1:
            raise ValueError(
                "Dataset must contain at least 2 episodes when `test_ratio > 0` so "
                f"both train and test splits are non-empty. Got total_episodes={total_episodes} "
                f"for {dataset_root}."
            )
        test_count = int(round(total_episodes * resolved_test_ratio))
        test_count = min(max(1, test_count), total_episodes - 1)
    train_count = total_episodes - test_count

    train_episode_indices = sorted(int(ep) for ep in episode_indices[:train_count].tolist())
    test_episode_indices = sorted(int(ep) for ep in episode_indices[train_count:].tolist())

    return DatasetSplitSpec(
        dataset_arg=str(dataset_arg),
        dataset_root=str(dataset_root.resolve()),
        dataset_repo_id=str(dataset_repo_id),
        total_episodes=int(total_episodes),
        train_episode_indices=train_episode_indices,
        test_episode_indices=test_episode_indices,
        test_ratio=resolved_test_ratio,
        split_seed=int(split_seed),
        split_shuffle=bool(split_shuffle),
    )


def save_dataset_split(output_dir: Path, split_spec: DatasetSplitSpec) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / DATASET_SPLIT_FILENAME
    split_path.write_text(
        json.dumps(asdict(split_spec), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return split_path


def load_dataset_split(split_path: Path) -> DatasetSplitSpec:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    return DatasetSplitSpec(
        dataset_arg=str(data["dataset_arg"]),
        dataset_root=str(data["dataset_root"]),
        dataset_repo_id=str(data["dataset_repo_id"]),
        total_episodes=int(data["total_episodes"]),
        train_episode_indices=[int(ep) for ep in data["train_episode_indices"]],
        test_episode_indices=[int(ep) for ep in data["test_episode_indices"]],
        test_ratio=float(data["test_ratio"]),
        split_seed=int(data["split_seed"]),
        split_shuffle=bool(data["split_shuffle"]),
    )


def find_dataset_split_file(start_path: Path) -> Path | None:
    current = start_path.resolve()
    if current.is_file():
        current = current.parent

    for parent in [current, *current.parents]:
        candidate = parent / DATASET_SPLIT_FILENAME
        if candidate.exists():
            return candidate
    return None


def parse_episode_range(range_spec: str) -> tuple[int, int]:
    start_text, end_text = str(range_spec).split(":", 1)
    return int(start_text), int(end_text)


def resolve_episode_indices_from_dataset_info(
    dataset_root: Path,
    split_name: str,
) -> list[int] | None:
    info = load_dataset_info(dataset_root)
    splits = info.get("splits", {})
    if not isinstance(splits, dict):
        return None

    split_spec = splits.get(split_name)
    if split_spec is None and split_name == "test":
        split_spec = splits.get("val")
    if split_spec is None:
        return None

    start, end = parse_episode_range(str(split_spec))
    if start < 0 or end < start:
        raise ValueError(
            f"Invalid split range {split_name}={split_spec!r} in {dataset_root / 'meta/info.json'}."
        )
    return list(range(start, end))
