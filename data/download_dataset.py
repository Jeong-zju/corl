#!/usr/bin/env python3
"""
从Hugging Face下载数据集的脚本，支持断点续传和自动重试
"""

import argparse
import time
from pathlib import Path
from huggingface_hub import snapshot_download


def download_dataset(
    dataset_name: str,
    output_dir: str = "./",
    repo_type: str = "dataset",
    cache_dir: str = "./.cache",
    max_retries: int = 10,
    retry_delay: int = 30,
):
    """
    从Hugging Face下载数据集，支持断点续传和自动重试

    Args:
        dataset_name: 数据集名称 (例如: "squad", "glue")
        output_dir: 输出目录
        repo_type: 仓库类型 ("dataset", "model", "space")
        cache_dir: 缓存目录
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    local_dir = output_path / dataset_name.replace("/", "_")

    print(f"正在下载数据集: {dataset_name}")
    print(f"输出目录: {local_dir}")
    print(f"最大重试次数: {max_retries}")

    for attempt in range(max_retries):
        try:
            print(f"\n尝试 {attempt + 1}/{max_retries}")

            snapshot_download(
                repo_id=dataset_name,
                repo_type=repo_type,
                local_dir=str(local_dir),
                cache_dir=str(cache_path),
                resume_download=True,
                max_workers=4,
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
        description="从Hugging Face下载数据集，支持断点续传"
    )
    parser.add_argument("dataset_name", type=str, help="数据集名称")
    parser.add_argument("--output-dir", type=str, default="./", help="输出目录")
    parser.add_argument("--repo-type", type=str, default="dataset", help="仓库类型")
    parser.add_argument("--cache-dir", type=str, default="./.cache", help="缓存目录")
    parser.add_argument("--max-retries", type=int, default=1000, help="最大重试次数")
    parser.add_argument("--retry-delay", type=int, default=30, help="重试延迟（秒）")

    args = parser.parse_args()

    download_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        repo_type=args.repo_type,
        cache_dir=args.cache_dir,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
