#!/usr/bin/env python3
"""
测试数据集加载脚本
功能：
1. 加载数据集 '/home/jeong/zeno/corl/data/zeno-ai/day3_5_Exp1'
2. 计算每条轨迹的signature (depth=4)
3. 使用t-SNE和PCA对signature进行降维可视化，观察聚类情况
"""

import json
import pandas as pd
import numpy as np
import torch
import signatory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def load_all_episodes(dataset_path):
    """加载所有episodes的元数据"""
    episodes_dir = dataset_path / "meta" / "episodes"
    all_episodes = []

    for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            all_episodes.append(df)

    return pd.concat(all_episodes, ignore_index=True)


def load_trajectory_data(dataset_path, from_idx, to_idx):
    """加载指定范围的轨迹数据"""
    data_dir = dataset_path / "data"
    all_data = []

    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            all_data.append(df)

    full_data = pd.concat(all_data, ignore_index=True)
    return full_data.iloc[from_idx:to_idx]


def compute_signature(trajectory, depth=4):
    """计算轨迹的signature

    Args:
        trajectory: numpy array of shape (T, D) where T is time steps, D is dimensions
        depth: signature depth

    Returns:
        signature tensor
    """
    # 转换为torch tensor并添加batch维度
    path = torch.tensor(trajectory, dtype=torch.float32).unsqueeze(0)  # (1, T, D)

    # 计算signature
    signature = signatory.signature(path, depth)

    return signature.squeeze(0)  # 移除batch维度


def main():
    # 数据集路径
    dataset_path = Path("/home/jeong/zeno/corl/data/zeno-ai/day3_5_Exp1")

    print("=" * 60)
    print("加载数据集...")
    print(f"数据集路径: {dataset_path}")
    print("=" * 60)

    # 读取元数据
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)

    print(f"\n数据集基本信息:")
    print(f"  总episodes数: {info['total_episodes']}")
    print(f"  总frames数: {info['total_frames']}")
    print(f"  FPS: {info['fps']}")
    print(f"  机器人类型: {info['robot_type']}")

    # 加载所有episodes
    print("\n加载所有episodes元数据...")
    episodes_df = load_all_episodes(dataset_path)
    print(f"总episodes数: {len(episodes_df)}")

    # 加载所有数据
    print("\n加载所有轨迹数据...")
    data_dir = dataset_path / "data"
    all_data = []
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            all_data.append(df)
    full_data = pd.concat(all_data, ignore_index=True)
    print(f"总数据行数: {len(full_data)}")

    # 计算每条轨迹的signature
    print("\n" + "=" * 60)
    print("计算轨迹signatures (depth=4)...")
    print("=" * 60)

    depth = 4
    signatures = []

    for idx, episode in episodes_df.iterrows():
        from_idx = int(episode["dataset_from_index"])
        to_idx = int(episode["dataset_to_index"])

        # 获取轨迹数据
        trajectory_data = full_data.iloc[from_idx:to_idx]

        # 提取observation.state作为轨迹
        obs_states = np.array(trajectory_data["observation.state"].tolist())  # (T, 14)

        # 计算signature
        sig = compute_signature(obs_states, depth=depth)
        signatures.append(sig.numpy())

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(episodes_df)} 条轨迹")

    signatures = np.array(signatures)  # (N, sig_dim)
    print(f"\nSignatures shape: {signatures.shape}")
    print(f"Signature维度: {signatures.shape[1]}")

    # 可视化
    print("\n" + "=" * 60)
    print("可视化signatures...")
    print("=" * 60)

    # 1. t-SNE 2D可视化
    print("计算t-SNE (2D)...")
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
    signatures_tsne_2d = tsne_2d.fit_transform(signatures)

    fig1 = plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        signatures_tsne_2d[:, 0],
        signatures_tsne_2d[:, 1],
        c=range(len(signatures)),
        cmap="viridis",
        alpha=0.6,
        s=30,
    )
    plt.colorbar(scatter, label="Trajectory Index")
    plt.title("t-SNE 2D Visualization of Trajectory Signatures", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path_2d = Path(__file__).parent / "trajectory_signatures_tsne_2d.png"
    plt.savefig(output_path_2d, dpi=150, bbox_inches="tight")
    print(f"t-SNE 2D可视化已保存到: {output_path_2d}")

    # 2. t-SNE 3D可视化
    print("计算t-SNE (3D)...")
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
    signatures_tsne_3d = tsne_3d.fit_transform(signatures)

    fig2 = plt.figure(figsize=(16, 12))

    # 3D视图
    ax1 = fig2.add_subplot(221, projection="3d")
    scatter1 = ax1.scatter(
        signatures_tsne_3d[:, 0],
        signatures_tsne_3d[:, 1],
        signatures_tsne_3d[:, 2],
        c=range(len(signatures)),
        cmap="viridis",
        alpha=0.6,
        s=20,
    )
    ax1.set_title("t-SNE 3D Visualization", fontsize=14)
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")
    ax1.set_zlabel("t-SNE Component 3")
    fig2.colorbar(scatter1, ax=ax1, label="Trajectory Index", shrink=0.5)

    # 2D投影: Component 1 vs 2
    ax2 = fig2.add_subplot(222)
    scatter2 = ax2.scatter(
        signatures_tsne_3d[:, 0],
        signatures_tsne_3d[:, 1],
        c=range(len(signatures)),
        cmap="viridis",
        alpha=0.6,
        s=30,
    )
    ax2.set_title("t-SNE 3D: Component 1 vs 2", fontsize=14)
    ax2.set_xlabel("t-SNE Component 1")
    ax2.set_ylabel("t-SNE Component 2")
    ax2.grid(True, alpha=0.3)
    fig2.colorbar(scatter2, ax=ax2, label="Trajectory Index")

    # 2D投影: Component 1 vs 3
    ax3 = fig2.add_subplot(223)
    scatter3 = ax3.scatter(
        signatures_tsne_3d[:, 0],
        signatures_tsne_3d[:, 2],
        c=range(len(signatures)),
        cmap="viridis",
        alpha=0.6,
        s=30,
    )
    ax3.set_title("t-SNE 3D: Component 1 vs 3", fontsize=14)
    ax3.set_xlabel("t-SNE Component 1")
    ax3.set_ylabel("t-SNE Component 3")
    ax3.grid(True, alpha=0.3)
    fig2.colorbar(scatter3, ax=ax3, label="Trajectory Index")

    # 2D投影: Component 2 vs 3
    ax4 = fig2.add_subplot(224)
    scatter4 = ax4.scatter(
        signatures_tsne_3d[:, 1],
        signatures_tsne_3d[:, 2],
        c=range(len(signatures)),
        cmap="viridis",
        alpha=0.6,
        s=30,
    )
    ax4.set_title("t-SNE 3D: Component 2 vs 3", fontsize=14)
    ax4.set_xlabel("t-SNE Component 2")
    ax4.set_ylabel("t-SNE Component 3")
    ax4.grid(True, alpha=0.3)
    fig2.colorbar(scatter4, ax=ax4, label="Trajectory Index")

    plt.tight_layout()

    output_path_3d = Path(__file__).parent / "trajectory_signatures_tsne_3d.png"
    plt.savefig(output_path_3d, dpi=150, bbox_inches="tight")
    print(f"t-SNE 3D可视化已保存到: {output_path_3d}")

    plt.show()

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
