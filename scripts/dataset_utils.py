from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_LOCAL_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DATASET_SPLIT_FILENAME = "dataset_split.json"


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


def find_lerobot_dataset_root(candidate: Path) -> Path | None:
    raw = candidate.expanduser()
    if is_lerobot_dataset_root(raw):
        return raw.resolve()

    if not raw.exists() or not raw.is_dir():
        return None

    for info_path in sorted(raw.rglob("meta/info.json")):
        root = info_path.parent.parent
        if is_lerobot_dataset_root(root):
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


def resolve_dataset_root(
    dataset: str | Path,
    *,
    local_data_root: Path = DEFAULT_LOCAL_DATA_ROOT,
) -> Path:
    dataset_text = str(dataset).strip()
    raw_path = Path(dataset_text).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute() or raw_path.exists() or dataset_text.startswith("."):
        candidates.append(raw_path)

    if not raw_path.is_absolute():
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
    if total_episodes <= 1:
        raise ValueError(
            f"Dataset must contain at least 2 episodes to create a train/test split. "
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
    if not (0.0 < float(test_ratio) < 1.0):
        raise ValueError(f"`test_ratio` must lie in (0, 1), got {test_ratio}.")

    total_episodes = get_total_episodes(dataset_root)
    episode_indices = np.arange(total_episodes, dtype=np.int64)
    if split_shuffle:
        rng = np.random.default_rng(split_seed)
        rng.shuffle(episode_indices)

    test_count = int(round(total_episodes * float(test_ratio)))
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
        test_ratio=float(test_ratio),
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
