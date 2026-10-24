# -*- coding: utf-8 -*-
"""
机器学习实验10 - 第2题：学生成绩 K均值聚类
比较数据归一化与未归一化情况下模型的测试预测精度（柱状图对比）
"""
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def cluster_accuracy(y_true, y_pred):
    """由于 KMeans 簇编号是任意的，先用匈牙利匹配把预测簇号映射到真实簇号，
    再计算一致率（即伪标签意义下的“预测精度”）。"""
    labels = np.unique(y_true)
    cost = np.zeros((len(labels), len(labels)), dtype=int)
    for i, p in enumerate(labels):
        for j, t in enumerate(labels):
            cost[i, j] = -np.sum((y_pred == p) & (y_true == t))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {labels[r]: labels[c] for r, c in zip(row_ind, col_ind)}
    y_mapped = np.array([mapping[p] for p in y_pred])
    return np.mean(y_mapped == y_true)


if __name__ == '__main__':
    # 表1 成绩样本：高数、英语、线性代数、近代史
    scores = np.array([
        [88, 64, 96, 85],
        [92, 99, 95, 94],
        [91, 87, 99, 95],
        [78, 99, 97, 81],
        [88, 78, 98, 84],
        [100, 95, 100, 92],
        [67, 89, 98, 90],
        [87, 89, 74, 96],
        [87, 98, 56, 78],
        [76, 82, 58, 93],
        [78, 67, 89, 54],
        [90, 98, 65, 76],
        [65, 66, 59, 78],
        [85, 89, 59, 87],
        [63, 58, 48, 65],
        [70, 78, 54, 91],
    ], dtype=float)

    # 由于成绩无真实类别标签，先用全量数据聚类得到“基准标签”，
    # 再划分训练/测试：在训练集上拟合 KMeans，预测测试集，
    # 与测试集基准标签做匈牙利匹配后得到一致率，作为“预测精度”。
    k = 3  # 优 / 中 / 一般 三档
    base_km = KMeans(n_clusters=k, random_state=42, n_init=10)
    y_base = base_km.fit_predict(scores)

    idx = np.arange(len(scores))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=42)
    x_train_raw, x_test_raw = scores[train_idx], scores[test_idx]
    y_test_base = y_base[test_idx]

    # --- 未归一化 ---
    km_raw = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x_train_raw)
    pred_raw = km_raw.predict(x_test_raw)
    acc_raw = cluster_accuracy(y_test_base, pred_raw)

    # --- 归一化（StandardScaler）---
    scaler = StandardScaler().fit(x_train_raw)
    x_train_s = scaler.transform(x_train_raw)
    x_test_s = scaler.transform(x_test_raw)
    km_std = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x_train_s)
    pred_std = km_std.predict(x_test_s)
    acc_std = cluster_accuracy(y_test_base, pred_std)

    print(f'未归一化时测试集预测精度：{acc_raw:.3f}')
    print(f'归一化后测试集预测精度：{acc_std:.3f}')

    plt.figure(figsize=(6, 5))
    bars = plt.bar(['未归一化', '归一化'], [acc_raw, acc_std],
                   color=['#9ecae1', '#3182bd'], width=0.5)
    for b, v in zip(bars, [acc_raw, acc_std]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                 ha='center', va='bottom')
    plt.ylim(0, 1.1)
    plt.ylabel('测试集预测精度')
    plt.title('归一化与未归一化情况下 K均值聚类精度对比')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'q2_normalize_compare.png')
    plt.savefig(p, dpi=150)
    plt.close()
    print('已保存：', p)
