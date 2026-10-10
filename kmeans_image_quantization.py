# -*- coding: utf-8 -*-
"""
基于 K-Means 的图像矢量量化
====================================
课程：Python 机器学习
实验：基于 K-Means 的图像矢量量化

功能：
  1. 读取彩色图片，将每个像素转换为 [R, G, B] 三维样本（全部像素参与训练，不抽样）
  2. 手写 K-Means 聚类算法（初始化中心 -> 分配样本 -> 更新中心 -> 判断收敛）
  3. 肘击法（Elbow Method）确定最优聚类数 K
  4. 用 K=2、最优 K、K=30 对图像进行矢量量化重建，对比压缩效果

运行方式：
  python3 kmeans_image_quantization.py
"""

import time

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# 中文字体设置（无中文字体环境时回退到英文标签）
for font in ["SimHei", "Arial Unicode MS", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [font]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_PATH = "data/flower.jpg"
RANDOM_SEEDS = [42, 123, 2024]  # 肘击法中每个 K 尝试的随机种子
K_MAX = 16                      # 肘击法扫描的 K 上限
K_COMPARE = [2, 30]             # 重建对比中除最优 K 外的固定取值


# ============================================================
# 第 1 部分：数据准备
# ============================================================
def load_image(path):
    """读取图片，缩放到 0~1，只保留 RGB 三通道，返回 (image, X, H, W)"""
    im = Image.open(path)
    image = np.array(im).astype(float) / 255.0   # 0~255 -> 0~1
    image = image[:, :, :3]                      # 只保留 R、G、B 三通道
    H, W, C = image.shape
    X = image.reshape(-1, 3)                     # (像素总数, 3)：每个像素一条样本
    print(f"图片形状: {image.shape}  (高={H}, 宽={W}, 通道={C})")
    print(f"样本矩阵形状: {X.shape}  (像素总数={H * W})")
    return image, X, H, W


# ============================================================
# 第 2 部分：手写 K-Means 算法
# ============================================================
def initCenter(dataSet, k, random_state=42):
    """第一步：初始化簇中心 —— 从数据集中随机选取 k 个样本，固定随机种子保证可复现"""
    rng = np.random.RandomState(random_state)
    n_samples = dataSet.shape[0]
    indices = rng.choice(n_samples, size=k, replace=False)
    return dataSet[indices].copy()


def myKmeans(dataSet, k, random_state=42, max_iter=200):
    """
    K-Means 主函数：分配 -> 更新 -> 判断收敛

    参数：
        dataSet: (n_samples, n_features) 样本矩阵
        k: 聚类数
        random_state: 随机种子
        max_iter: 最大迭代次数（防止不收敛时无限循环）

    返回：
        classCenter: (k, n_features) K 个簇中心（K 个代表颜色）
        clusterPoints: (n_samples, 2) 第 1 列是簇编号，第 2 列是样本到所属中心的距离平方
    """
    n_samples = dataSet.shape[0]

    # 第一步：初始化簇中心
    centers = initCenter(dataSet, k, random_state)

    # clusterPoints：第 1 列簇编号（-1 表示未分配），第 2 列距离平方
    clusterPoints = np.zeros((n_samples, 2))
    clusterPoints[:, 0] = -1

    for iteration in range(max_iter):
        # 第二步：分配样本 —— 向量化一次算出所有样本到所有中心的距离平方
        # dataSet[:, None, :] 形状 (n, 1, f)，centers[None, :, :] 形状 (1, k, f)
        distances = np.sum(
            (dataSet[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2
        )  # (n_samples, k)

        min_dist_indices = np.argmin(distances, axis=1)  # 每个样本最近中心的编号
        min_dist_values = np.min(distances, axis=1)      # 对应距离平方

        # 第四步：判断收敛 —— 所有样本的簇分配与上一轮完全相同
        if np.all(min_dist_indices == clusterPoints[:, 0]):
            print(f"  K={k}, seed={random_state}: 第 {iteration + 1} 次迭代收敛")
            break

        clusterPoints[:, 0] = min_dist_indices
        clusterPoints[:, 1] = min_dist_values

        # 第三步：更新中心 —— 每个簇的新中心 = 簇内样本均值；空簇保留原中心
        new_centers = centers.copy()
        for i in range(k):
            mask = clusterPoints[:, 0] == i
            if np.sum(mask) > 0:
                new_centers[i] = dataSet[mask].mean(axis=0)
        centers = new_centers

    return centers, clusterPoints


# ============================================================
# 第 3 部分：肘击法确定最优 K
# ============================================================
def elbow_method(X, k_range=range(1, K_MAX + 1), seeds=RANDOM_SEEDS):
    """
    肘击法：对每个 K 换多个随机种子跑 K-Means，取最小 SSE，返回 (K列表, SSE列表)
    SSE = clusterPoints[:, 1].sum()，即全部样本到所属簇中心距离平方之和
    """
    sse_list = []
    for k in k_range:
        best_sse = np.inf
        for seed in seeds:
            _, clusterPoints = myKmeans(X, k, random_state=seed)
            sse = clusterPoints[:, 1].sum()
            if sse < best_sse:
                best_sse = sse
        sse_list.append(best_sse)
        print(f"K={k:2d}, 最佳SSE={best_sse:.4f}")
    return list(k_range), sse_list


def find_elbow(k_list, sse_list):
    """
    自动定位肘部：用"下降量骤减"（二阶差分）寻找 SSE 下降明显变缓的位置。
    对相邻下降量 diff[i] = SSE[i] - SSE[i+1]，二阶差分 diff[i] - diff[i+1]
    越大，说明在第 i+2 个 K 处下降突然变缓（即肘部）。
    排除 K=1->2 的全局大跳变（从 0 个簇到 1 个簇的基准跳变，不是真正肘部）。
    返回 (最优K, 肘部索引)
    """
    sse_arr = np.array(sse_list, dtype=float)
    diff = sse_arr[:-1] - sse_arr[1:]          # 相邻下降量
    second_diff = diff[:-1] - diff[1:]         # 下降量的变化（>0 表示下降骤减）
    # 跳过 index 0（K=1->2 的全局跳变），找下降骤减最明显的位置
    idx = 1 + int(np.argmax(second_diff[1:]))
    return int(k_list[idx + 1]), idx + 1


def plot_elbow(k_list, sse_list, elbow_k, save_path="results/elbow_curve.png"):
    """绘制 SSE-K 肘部曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, sse_list, "bo-", linewidth=2, markersize=8, label="SSE")
    plt.axvline(elbow_k, color="red", linestyle="--", linewidth=1.5,
                label=f"Optimal K = {elbow_k}")
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("SSE (Sum of Squared Errors)", fontsize=12)
    plt.title("Elbow Method - SSE vs K", fontsize=14)
    plt.xticks(k_list)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"肘击曲线图已保存: {save_path}")

    # 打印各 K 的 SSE 下降量
    sse_arr = np.array(sse_list)
    print("\n各 K 的 SSE 下降量:")
    for i in range(len(sse_arr) - 1):
        print(f"  K={k_list[i]} -> K={k_list[i+1]}: 下降 {sse_arr[i] - sse_arr[i+1]:.4f}")


# ============================================================
# 第 4 部分：矢量量化重建与展示
# ============================================================
def quantize(image, X, H, W, k, random_state=42):
    """用 K 个代表颜色重建图像：每个像素替换为所属簇的中心颜色"""
    centers, clusterPoints = myKmeans(X, k, random_state=random_state)
    sse = clusterPoints[:, 1].sum()
    labels = clusterPoints[:, 0].astype(int)
    reconstructed = centers[labels].reshape(H, W, 3)
    return reconstructed, centers, sse


def plot_comparison(image, reconstructed_list, k_list,
                    save_path="results/reconstruction_comparison.png"):
    """并排显示原图与各 K 的重建图"""
    n = 1 + len(reconstructed_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    for i, (img, k) in enumerate(zip(reconstructed_list, k_list)):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"K={k} ({k} colors)", fontsize=14)
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"重建对比图已保存: {save_path}")


def compression_analysis(k_list):
    """计算不同 K 下的每像素位数与压缩比（相对 24 位真彩色）"""
    print("\n压缩率分析:")
    print(f"{'K':>4} | {'每像素位数':>8} | {'压缩比(相对24位)':>14}")
    print("-" * 36)
    for k in k_list:
        bits = np.log2(k)
        ratio = 24 / bits
        print(f"{k:>4} | {bits:>6.1f} bit | {ratio:>12.1f}:1")
    return


def main():
    print("=" * 60)
    print("第 1 部分：数据准备")
    print("=" * 60)
    image, X, H, W = load_image(DATA_PATH)

    # 显示原图
    plt.figure(figsize=(8, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original Image (彩色)")
    plt.tight_layout()
    plt.savefig("results/original_image.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 60)
    print("第 2 部分：手写 K-Means 算法（myKmeans 已定义）")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("第 3 部分：肘击法确定最优 K")
    print("=" * 60)
    k_list, sse_list = elbow_method(X)
    optimal_k, elbow_idx = find_elbow(k_list, sse_list)
    print(f"\n肘部位置：K = {optimal_k}（下降量骤减最明显处，SSE 下降明显变缓）")
    plot_elbow(k_list, sse_list, optimal_k)

    print("\n" + "=" * 60)
    print("第 4 部分：矢量量化重建与展示")
    print("=" * 60)
    k_values = [2, optimal_k, 30]
    reconstructed_list, sse_list_k = [], []
    for k in k_values:
        t0 = time.time()
        reconstructed, centers, sse = quantize(image, X, H, W, k)
        elapsed = time.time() - t0
        print(f"K={k}: 训练完成, 耗时={elapsed:.2f}s, SSE={sse:.4f}")
        reconstructed_list.append(reconstructed)
        sse_list_k.append(sse)

    plot_comparison(image, reconstructed_list, k_values)

    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    print(f"最优 K = {optimal_k}")
    compression_analysis(k_values)
    print("\n实验完成！")


if __name__ == "__main__":
    main()
