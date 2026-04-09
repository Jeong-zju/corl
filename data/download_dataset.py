#!/usr/bin/env python3
"""从 Hugging Face 下载仓库快照，支持断点续传、自动重试和自动并发。

默认会下载到脚本所在目录，也就是 ``main/data/{dataset_id}``。例如：

    python data/download_dataset.py Yinpei/robomme_data_lerobot \
      --cache-dir ~/.cache/huggingface

会把数据下载到：

    data/Yinpei/robomme_data_lerobot
"""

import argparse
import os
import time
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None


AUTO_MAX_WORKERS = "auto"
MIN_AUTO_MAX_WORKERS = 8
MAX_AUTO_MAX_WORKERS = 64
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def resolve_local_dir(output_path: Path, dataset_name: str) -> Path:
    dataset_parts = [part for part in dataset_name.split("/") if part]
    if not dataset_parts or any(part in {".", ".."} for part in dataset_parts):
        raise ValueError(
            f"Invalid dataset name `{dataset_name}`. Expected a Hugging Face repo id such as `owner/name`."
        )
    return output_path.joinpath(*dataset_parts)


def get_available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def resolve_max_workers(max_workers: int | str | None) -> int:
    if isinstance(max_workers, int):
        if max_workers <= 0:
            raise ValueError(f"`max_workers` must be positive. Got {max_workers}.")
        return max_workers

    text = AUTO_MAX_WORKERS if max_workers is None else str(max_workers).strip().lower()
    if text != AUTO_MAX_WORKERS:
        resolved = int(text)
        if resolved <= 0:
            raise ValueError(f"`max_workers` must be positive. Got {max_workers}.")
        return resolved

    cpu_count = get_available_cpu_count()
    # snapshot_download is mostly network-bound, so scale above CPU count,
    # but clamp against a reasonable upper bound and file descriptor limits.
    cpu_based_limit = max(MIN_AUTO_MAX_WORKERS, cpu_count * 4)
    fd_based_limit = MAX_AUTO_MAX_WORKERS
    if resource is not None:
        try:
            soft_nofile, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        except (OSError, ValueError):
            soft_nofile = -1
        if soft_nofile and soft_nofile > 0:
            fd_based_limit = max(MIN_AUTO_MAX_WORKERS, soft_nofile // 8)

    return max(
        1,
        min(
            MAX_AUTO_MAX_WORKERS,
            cpu_based_limit,
            fd_based_limit,
        ),
    )


def download_dataset(
    dataset_name: str,
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
    repo_type: str = "dataset",
    cache_dir: str = "./.cache",
    max_retries: int = 10,
    retry_delay: int = 30,
    max_workers: int | str = AUTO_MAX_WORKERS,
):
    """
    从Hugging Face下载数据集，支持断点续传和自动重试

    Args:
        dataset_name: 数据集名称 (例如: "squad", "glue")
        output_dir: 输出根目录，默认是脚本所在的 data 目录
        repo_type: 仓库类型 ("dataset", "model", "space")
        cache_dir: 缓存目录
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        max_workers: 下载并发数，支持正整数或 "auto"
    """
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    cache_path = Path(cache_dir).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)

    local_dir = resolve_local_dir(output_path, dataset_name)
    resolved_max_workers = resolve_max_workers(max_workers)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "缺少 `huggingface_hub`，请先安装后再下载数据集，例如 `pip install huggingface_hub`。"
        ) from exc

    print(f"正在下载数据集: {dataset_name}")
    print(f"输出目录: {local_dir}")
    print(f"最大重试次数: {max_retries}")
    print(f"下载并发数: {resolved_max_workers}")

    for attempt in range(max_retries):
        try:
            print(f"\n尝试 {attempt + 1}/{max_retries}，缓存到 {cache_path}")

            snapshot_download(
                repo_id=dataset_name,
                repo_type=repo_type,
                local_dir=str(local_dir),
                cache_dir=str(cache_path),
                resume_download=True,
                max_workers=resolved_max_workers,
            )

            print(f"\n✓ 数据集下载完成: {local_dir}")
            return

        except Exception as e:
            print(f"\n✗ 下载失败: {e}")

            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"\n已达到最大重试次数 ({max_retries})，下载失败")
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "从 Hugging Face 下载仓库快照，支持断点续传、自动重试和自动并发。"
        ),
        epilog=(
            "示例:\n"
            "  python data/download_dataset.py Yinpei/robomme_data_lerobot \\\n"
            "    --cache-dir ~/.cache/huggingface\n\n"
            "默认输出目录:\n"
            "  data/Yinpei/robomme_data_lerobot"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset_name", type=str, help="数据集名称")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出根目录；默认是脚本所在的 data 目录，实际下载目录为 <output_dir>/<dataset_id>",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        help='仓库类型，可选 "dataset"、"model"、"space"',
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/.cache",
        help="缓存目录；下载失败后会基于缓存自动续传",
    )
    parser.add_argument("--max-retries", type=int, default=1000, help="最大重试次数")
    parser.add_argument("--retry-delay", type=int, default=30, help="重试延迟（秒）")
    parser.add_argument(
        "--max-workers",
        type=str,
        default=AUTO_MAX_WORKERS,
        help='下载并发数，传正整数或 "auto"（默认，会结合 CPU 和文件描述符上限估算）',
    )

    args = parser.parse_args()

    download_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        repo_type=args.repo_type,
        cache_dir=args.cache_dir,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        max_workers=args.max_workers,
    )
