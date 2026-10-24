# -*- coding: utf-8 -*-
"""
机器学习实验10 - 第1题：make_blobs 数据 K均值聚类
(1) K=2~10 的 Calinski-Harabasz 指数变化曲线
(2) K=4 时聚类效果图
"""
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == '__main__':
    # 注意：新版 sklearn 中 make_blobs 已从
    # sklearn.datasets.samples_generator 迁移到 sklearn.datasets
    X, y = make_blobs(n_samples=500, n_features=2,
                      centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.3, 0.1, 0.2],
                      random_state=42)

    # (1) K=2~10 的 Calinski-Harabasz 指数变化曲线
    ch_scores = []
    ks = list(range(2, 11))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        ch = calinski_harabasz_score(X, labels)
        ch_scores.append(ch)
        print(f'K={k:2d}  Calinski-Harabasz = {ch:.3f}')

    best_k = ks[int(np.argmax(ch_scores))]
    print(f'CH 指数最大值对应 K = {best_k}')

    plt.figure(figsize=(7, 5))
    plt.plot(ks, ch_scores, 'o-', color='steelblue', linewidth=2)
    plt.xlabel('K（簇数）')
    plt.ylabel('Calinski-Harabasz 指数')
    plt.title('K 值与 Calinski-Harabasz 指数变化曲线')
    plt.xticks(ks)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, 'q1_ch_curve.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    print('已保存：', p1)

    # (2) K=4 时聚类效果图
    km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels4 = km4.fit_predict(X)
    centers = km4.cluster_centers_

    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels4, cmap='viridis', s=30, alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X',
                s=200, edgecolors='black', label='簇中心')
    plt.title('K=4 时 K均值聚类效果图')
    plt.xlabel('特征 x1')
    plt.ylabel('特征 x2')
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, 'q1_k4_result.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print('已保存：', p2)
