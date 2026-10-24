# -*- coding: utf-8 -*-
"""
实验11-1：EM（高斯混合模型 GMM）聚类
内容：
  1. 利用 make_blobs 生成三维数据；
  2. 在 BIC 曲线基础上结合肘部法确定最优聚类数 k；
  3. 用高斯混合模型对数据进行聚类；
  4. 可视化：三维数据散点 + PCA 降维到二维散点。
运行环境：Python 3.13 + numpy / scikit-learn / matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# ---------- 中文显示 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录：脚本所在目录
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------- 1. 生成三维数据 ----------
def make_3d_data():
    X, y_true = make_blobs(
        n_samples=600,
        n_features=3,          # 三维数据
        centers=4,              # 真实 4 个簇
        cluster_std=1.0,
        random_state=42,
    )
    print(f'[数据] 生成三维样本 X.shape = {X.shape}, 真实簇数 = 4')
    return X, y_true


# ---------- 2. BIC 曲线 + 肘部法选最优 k ----------
def select_k_by_bic_elbow(X, k_max=10):
    ks = list(range(1, k_max + 1))
    bics = []
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              random_state=42)
        gmm.fit(X)
        bics.append(gmm.bic(X))
    bics = np.asarray(bics)

    # 肘部法：BIC 相邻差值（下降量）首次降到最大下降量的某一比例以下时，
    # 视为"下降显著变缓"的肘部位置。
    diffs = bics[:-1] - bics[1:]          # diffs[i] = BIC(i) - BIC(i+1)
    max_drop = diffs.max()
    elbow_idx = None
    for i, d in enumerate(diffs):
        if d < 0.2 * max_drop:             # 下降量已不足最大下降的 20%
            elbow_idx = i + 1              # 对应 k = ks[i+1]
            break
    if elbow_idx is None:
        elbow_idx = int(np.argmin(bics))
    best_k = ks[elbow_idx]
    bic_min_idx = int(np.argmin(bics))

    print('[BIC] 各 k 对应 BIC 值：')
    for k, b in zip(ks, bics):
        mark = ''
        if k == best_k:
            mark += '  <- 肘部法最优'
        if k == ks[bic_min_idx]:
            mark += '  <- BIC 最小'
        print(f'   k={k:2d}  BIC={b:12.2f}{mark}')

    # 画 BIC 曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, bics, 'o-', color='steelblue', lw=2, label='BIC')
    ax.axvline(best_k, color='red', ls='--', lw=1.5,
               label=f'肘部法最优 k = {best_k}')
    ax.scatter([best_k], [bics[elbow_idx]], color='red', s=100, zorder=5)
    ax.set_xlabel('聚类数 k')
    ax.set_ylabel('BIC 值')
    ax.set_title('BIC 曲线与肘部法确定最优聚类数')
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'em1_bic_elbow.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[图] 已保存：{out}')

    return best_k


# ---------- 3. GMM 聚类 ----------
def fit_gmm(X, k):
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                          random_state=42)
    y_pred = gmm.fit_predict(X)
    print(f'[GMM] 使用 k={k} 拟合完成')
    print('   各簇权重 =', np.round(gmm.weights_, 3))
    print('   各簇均值 =\n', np.round(gmm.means_, 3))
    return gmm, y_pred


# ---------- 4. 三维数据可视化 ----------
def plot_3d(X, y_true, y_pred, k):
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_true,
                cmap='viridis', s=20, alpha=0.75)
    ax1.set_title('三维原始数据（真实标签）')
    ax1.set_xlabel('X1'); ax1.set_ylabel('X2'); ax1.set_zlabel('X3')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred,
                cmap='viridis', s=20, alpha=0.75)
    ax2.set_title(f'GMM 聚类结果（k = {k}）')
    ax2.set_xlabel('X1'); ax2.set_ylabel('X2'); ax2.set_zlabel('X3')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'em2_3d_cluster.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[图] 已保存：{out}')


# ---------- 5. PCA 降维到二维可视化 ----------
def plot_pca2d(X, y_true, y_pred, k):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f'[PCA] 前两主成分累计方差解释率 = '
          f'{pca.explained_variance_ratio_.sum():.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true,
                    cmap='viridis', s=20, alpha=0.75)
    axes[0].set_title('PCA 二维投影（真实标签）')
    axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')

    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred,
                    cmap='viridis', s=20, alpha=0.75)
    axes[1].set_title(f'PCA 二维投影（GMM 预测标签，k = {k}）')
    axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'em3_pca2d.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[图] 已保存：{out}')


def main():
    X, y_true = make_3d_data()
    best_k = select_k_by_bic_elbow(X, k_max=10)
    gmm, y_pred = fit_gmm(X, best_k)
    plot_3d(X, y_true, y_pred, best_k)
    plot_pca2d(X, y_true, y_pred, best_k)
    print('[完成] EM/GMM 实验全部结束。')


if __name__ == '__main__':
    main()
