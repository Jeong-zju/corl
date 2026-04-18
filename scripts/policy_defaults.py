from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_ROOT = PROJECT_ROOT / "bash" / "defaults"
DATA_ROOT = PROJECT_ROOT / "data"
ROBOCASA_TASK_COLLECTION_NAMES = frozenset({"atomic", "composite"})


def defaults_file_path(env_name: str, policy_name: str) -> Path:
    return DEFAULTS_ROOT / env_name / f"{policy_name}.yaml"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at top level in defaults file: {path}")
    return data


def _normalize_dataset_selector_candidates(dataset_selector: str) -> list[str]:
    raw = str(dataset_selector).strip().replace("\\", "/")
    if not raw:
        return []

    candidates: list[str] = []

    def add(value: str | Path | None) -> None:
        if value is None:
            return
        text = str(value).strip().replace("\\", "/")
        if not text:
            return
        if text.startswith("./"):
            text = text[2:]
        while "//" in text:
            text = text.replace("//", "/")
        if text and text not in candidates:
            candidates.append(text)

    raw_path = Path(raw).expanduser()
    add(raw)
    add(raw.replace("/", "_"))
    add(raw_path.name)
    if len(raw_path.parts) >= 2:
        add("_".join(raw_path.parts[-2:]))

    if raw.startswith("data/"):
        stripped = raw[len("data/") :]
        add(stripped)
        add(stripped.replace("/", "_"))
        add(Path(stripped).name)

    for root in (DATA_ROOT, PROJECT_ROOT):
        try:
            relative = raw_path.resolve(strict=False).relative_to(root.resolve())
        except Exception:
            continue
        add(relative.as_posix())
        add(relative.as_posix().replace("/", "_"))
        add(relative.name)
        if len(relative.parts) >= 2:
            add("_".join(relative.parts[-2:]))

    return candidates


def _normalize_defaults_match_text(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    if text.startswith("./"):
        text = text[2:]
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/")


def _normalize_cli_selector(value: str | Path | None) -> str | None:
    normalized = _normalize_defaults_match_text(value)
    return normalized or None


def _normalize_cli_task_items(
    value: str | list[str] | tuple[str, ...] | set[str] | None,
) -> tuple[str, ...]:
    if value is None:
        return ()

    raw_items = (
        value if isinstance(value, (list, tuple, set)) else str(value).split(",")
    )
    items: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = str(raw_item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        items.append(item)
    return tuple(items)


def _infer_robocasa_task_from_selector(
    selector: str | Path | None,
) -> str | None:
    normalized = _normalize_cli_selector(selector)
    if normalized is None:
        return None

    parts = [
        part
        for part in normalized.split("/")
        if part not in {"", ".", ".."}
    ]
    if parts[:2] == ["main", "data"]:
        parts = parts[2:]
    elif parts[:1] == ["data"]:
        parts = parts[1:]
    if len(parts) < 3 or parts[0] != "robocasa":
        return None
    if parts[1] not in ROBOCASA_TASK_COLLECTION_NAMES:
        return None

    task_name = str(parts[2]).strip()
    if not task_name or task_name in ROBOCASA_TASK_COLLECTION_NAMES:
        return None
    return task_name


def _defaults_match_specificity(value: str | Path | None) -> tuple[int, int]:
    normalized = _normalize_defaults_match_text(value)
    if not normalized:
        return (0, 0)
    parts = [
        part
        for part in normalized.split("/")
        if part not in {"", ".", ".."}
    ]
    return (len(parts), len(normalized))


def _score_defaults_candidate_match(
    selector_candidate: str,
    yaml_candidate: str,
) -> tuple[int, int, int] | None:
    normalized_selector = _normalize_defaults_match_text(selector_candidate)
    normalized_yaml = _normalize_defaults_match_text(yaml_candidate)
    if not normalized_selector or not normalized_yaml:
        return None

    yaml_depth, yaml_length = _defaults_match_specificity(normalized_yaml)
    if normalized_selector == normalized_yaml:
        return (3, yaml_depth, yaml_length)
    if "/" in normalized_yaml and normalized_selector.startswith(f"{normalized_yaml}/"):
        return (2, yaml_depth, yaml_length)
    return None


def _iter_defaults_yaml_match_candidates(
    value: str | Path | list[str] | tuple[str, ...] | None,
) -> list[str]:
    if value is None:
        return []

    raw_values = value if isinstance(value, (list, tuple)) else [value]
    candidates: list[str] = []
    for raw_value in raw_values:
        candidates.extend(
            _normalize_dataset_selector_candidates(str(raw_value))
        )
    return candidates


def resolve_dataset_defaults_path(
    dataset_selector: str,
    policy_name: str,
) -> Path | None:
    selector_candidates = _normalize_dataset_selector_candidates(dataset_selector)

    direct_matches: list[tuple[tuple[int, int], Path]] = []
    for candidate in selector_candidates:
        direct = DEFAULTS_ROOT / candidate / f"{policy_name}.yaml"
        if direct.exists():
            direct_matches.append((_defaults_match_specificity(candidate), direct))

    if direct_matches:
        return max(direct_matches, key=lambda item: item[0])[1]

    selector_candidate_set = set(selector_candidates)
    if not selector_candidate_set:
        return None

    best_match_path: Path | None = None
    best_match_score: tuple[int, int, int, int] | None = None
    for yaml_path in sorted(DEFAULTS_ROOT.rglob(f"{policy_name}.yaml")):
        try:
            data = _load_yaml_mapping(yaml_path)
        except Exception:
            continue
        train_cfg = data.get("train", {})
        if not isinstance(train_cfg, dict):
            continue

        eval_cfg = data.get("eval", {})
        if not isinstance(eval_cfg, dict):
            eval_cfg = {}

        yaml_candidates: list[tuple[str, int]] = []
        dataset_root = train_cfg.get("dataset_root")
        dataset_repo_id = train_cfg.get("dataset_repo_id")
        if dataset_root:
            yaml_candidates.extend(
                (candidate, 2)
                for candidate in _iter_defaults_yaml_match_candidates(dataset_root)
            )
        if dataset_repo_id:
            yaml_candidates.extend(
                (candidate, 2)
                for candidate in _iter_defaults_yaml_match_candidates(dataset_repo_id)
            )
        for task_value in (
            train_cfg.get("dataset_tasks"),
            train_cfg.get("task"),
            train_cfg.get("cil"),
            eval_cfg.get("task"),
            eval_cfg.get("cil"),
        ):
            yaml_candidates.extend(
                (candidate, 1)
                for candidate in _iter_defaults_yaml_match_candidates(task_value)
            )

        yaml_match_score: tuple[int, int, int, int] | None = None
        for selector_candidate in selector_candidate_set:
            for yaml_candidate, source_priority in set(yaml_candidates):
                base_score = _score_defaults_candidate_match(
                    selector_candidate=selector_candidate,
                    yaml_candidate=yaml_candidate,
                )
                if base_score is None:
                    continue
                score = (
                    base_score[0],
                    source_priority,
                    base_score[1],
                    base_score[2],
                )
                if yaml_match_score is None or score > yaml_match_score:
                    yaml_match_score = score

        if yaml_match_score is None:
            continue
        if best_match_score is None or yaml_match_score > best_match_score:
            best_match_score = yaml_match_score
            best_match_path = yaml_path
    return best_match_path


def resolve_cli_dataset_defaults_path(
    *,
    dataset_selector: str | Path | None,
    task_selector: str | list[str] | tuple[str, ...] | set[str] | None,
    policy_name: str,
) -> Path | None:
    normalized_dataset = _normalize_cli_selector(dataset_selector)
    task_items = _normalize_cli_task_items(task_selector)
    single_task = task_items[0] if len(task_items) == 1 else None

    if normalized_dataset is not None:
        if (
            single_task is not None
            and (
                normalized_dataset == "robocasa"
                or normalized_dataset.startswith("robocasa/")
            )
            and _infer_robocasa_task_from_selector(normalized_dataset) is None
        ):
            task_path = resolve_dataset_defaults_path(
                dataset_selector=single_task,
                policy_name=policy_name,
            )
            if task_path is not None:
                return task_path
        return resolve_dataset_defaults_path(
            dataset_selector=normalized_dataset,
            policy_name=policy_name,
        )

    if single_task is not None:
        return resolve_dataset_defaults_path(
            dataset_selector=single_task,
            policy_name=policy_name,
        )
    return None


def load_policy_mode_defaults_for_dataset(
    mode: str,
    dataset_selector: str,
    policy_name: str,
) -> tuple[dict[str, Any], Path | None]:
    path = resolve_dataset_defaults_path(dataset_selector, policy_name)
    if path is None:
        return {}, None
    data = _load_yaml_mapping(path)
    mode_defaults = data.get(mode, {})
    if mode_defaults is None:
        mode_defaults = {}
    if not isinstance(mode_defaults, dict):
        raise TypeError(
            f"Expected mapping for '{mode}' section in defaults file: {path}"
        )
    return mode_defaults, path


def load_policy_mode_defaults_for_cli(
    mode: str,
    *,
    dataset_selector: str | Path | None,
    task_selector: str | list[str] | tuple[str, ...] | set[str] | None,
    policy_name: str,
) -> tuple[dict[str, Any], Path | None]:
    path = resolve_cli_dataset_defaults_path(
        dataset_selector=dataset_selector,
        task_selector=task_selector,
        policy_name=policy_name,
    )
    if path is None:
        return {}, None
    data = _load_yaml_mapping(path)
    mode_defaults = data.get(mode, {})
    if mode_defaults is None:
        mode_defaults = {}
    if not isinstance(mode_defaults, dict):
        raise TypeError(
            f"Expected mapping for '{mode}' section in defaults file: {path}"
        )
    return mode_defaults, path


def load_policy_mode_defaults(
    mode: str,
    env_name: str,
    policy_name: str,
) -> dict[str, Any]:
    path = defaults_file_path(env_name=env_name, policy_name=policy_name)
    if not path.exists():
        raise FileNotFoundError(
            "Policy defaults file not found:\n"
            f"- mode={mode}\n"
            f"- env={env_name}\n"
            f"- policy={policy_name}\n"
            f"- path={path}"
        )

    data = _load_yaml_mapping(path)
    mode_defaults = data.get(mode, {})
    if mode_defaults is None:
        mode_defaults = {}
    if not isinstance(mode_defaults, dict):
        raise TypeError(
            f"Expected mapping for '{mode}' section in defaults file: {path}"
        )
    return mode_defaults
