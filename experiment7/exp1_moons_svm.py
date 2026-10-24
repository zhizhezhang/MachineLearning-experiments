# -*- coding: utf-8 -*-
"""
实验7-1：make_moons 数据集上 SMSVM 核函数与参数对预测精度的影响
  (1) 不同核函数(linear / poly / rbf)对测试精度的影响
  (2) rbf 核下不同 gamma(0.1, 0.5, 1, 5, 10) 对测试精度的影响
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------- 中文显示设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print('=' * 60)
    print('实验7-1：make_moons 数据集上 SVM 核函数与参数的影响')
    print('=' * 60)

    # 1. 生成 make_moons 数据并划分（100个样本，noise=0.5，测试集40%）
    x, y = make_moons(100, noise=0.5, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.4, random_state=42)
    print(f'训练集大小: {len(x_train)}，测试集大小: {len(x_test)}')

    # 数据标准化（SVM 对尺度敏感）
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    # ---------- (1) 不同核函数对预测精度的影响 ----------
    print('\n(1) 不同核函数的测试精度：')
    kernels = ['linear', 'poly', 'rbf']
    kernel_names = ['线性核 linear', '多项式核 poly', '高斯核 rbf']
    acc_kernels = []
    for k in kernels:
        clf = SVC(kernel=k)
        clf.fit(x_train_s, y_train)
        acc = clf.score(x_test_s, y_test)
        acc_kernels.append(acc)
        print(f'  核函数={k:7s}  测试精度 = {acc:.4f}')

    # 柱状图
    plt.figure(figsize=(7, 5))
    bars = plt.bar(kernel_names, acc_kernels,
                   color=['#4C72B0', '#55A868', '#C44E52'], width=0.5)
    plt.ylim(0, 1.0)
    plt.ylabel('测试集预测精度')
    plt.title('不同核函数对 SVM 预测精度的影响')
    for b, a in zip(bars, acc_kernels):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.01,
                 f'{a:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    p1 = os.path.join(BASE_DIR, 'result1_kernel.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f'  图已保存: {p1}')

    # ---------- (2) rbf 核不同 gamma 对预测精度的影响 ----------
    print('\n(2) rbf 核下不同 gamma 的测试精度：')
    gammas = [0.1, 0.5, 1, 5, 10]
    acc_gammas = []
    for g in gammas:
        clf = SVC(kernel='rbf', gamma=g)
        clf.fit(x_train_s, y_train)
        acc = clf.score(x_test_s, y_test)
        acc_gammas.append(acc)
        print(f'  gamma={g:<5}  测试精度 = {acc:.4f}')

    # 折线图（用 gamma 取值的序号作横坐标）
    plt.figure(figsize=(7, 5))
    x_pos = np.arange(len(gammas))
    plt.plot(x_pos, acc_gammas, marker='o', linewidth=2,
             color='#C44E52', label='测试精度')
    plt.xticks(x_pos, [str(g) for g in gammas])
    plt.ylim(0.6, 0.95)
    plt.xlabel('gamma 取值')
    plt.ylabel('测试集预测精度')
    plt.title('rbf 核不同 gamma 对 SVM 预测精度的影响')
    plt.grid(True, ls='--', alpha=0.5)
    for i, a in enumerate(acc_gammas):
        plt.text(i, a + 0.012, f'{a:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    p2 = os.path.join(BASE_DIR, 'result2_gamma.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f'  图已保存: {p2}')


if __name__ == '__main__':
    main()
