# -*- coding: utf-8 -*-
"""
机器学习实验8 —— 第1题：Iris 数据集与决策树深度影响
  (1) 决策树深度(2/6/10/15/20)对测试精度的影响，并画柱状图
  (2) 画决策树在训练数据上的分类效果图（决策边界）
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------- 中文显示设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # 1. 加载 iris，取第 0、2 列（花萼长度、花瓣长度）
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    Y = iris.target
    class_names = iris.target_names  # ['setosa' 'versicolor' 'virginica']

    # 2. 划分训练/测试集（测试集 40%，固定随机种子保证可复现）
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.4, random_state=42)
    print(f'训练集大小: {len(x_train)}，测试集大小: {len(x_test)}')

    # ---------- (1) 不同深度对测试精度的影响 ----------
    print('\n(1) 不同决策树深度的测试精度：')
    depths = [2, 6, 10, 15, 20]
    acc_test = []
    acc_train = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(x_train, y_train)
        a_tr = clf.score(x_train, y_train)
        a_te = clf.score(x_test, y_test)
        acc_train.append(a_tr)
        acc_test.append(a_te)
        print(f'  max_depth={d:<3d}  训练精度={a_tr:.4f}  测试精度={a_te:.4f}')

    # 柱状图
    plt.figure(figsize=(8, 5))
    x_pos = np.arange(len(depths))
    width = 0.35
    bars1 = plt.bar(x_pos - width / 2, acc_train, width,
                    label='训练集精度', color='#4C72B0')
    bars2 = plt.bar(x_pos + width / 2, acc_test, width,
                    label='测试集精度', color='#C44E52')
    plt.xticks(x_pos, [f'd={d}' for d in depths])
    plt.ylim(0.5, 1.05)
    plt.xlabel('决策树最大深度 max_depth')
    plt.ylabel('预测精度')
    plt.title('决策树深度对预测精度的影响')
    plt.legend()
    for b in bars1:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for b in bars2:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(BASE_DIR, 'result1_depth_accuracy.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f'  柱状图已保存: {p1}')

    # ---------- (2) 决策树在训练数据上的分类效果图（决策边界） ----------
    # 选深度=3 的树作为代表（边界较平滑，便于观察）
    clf_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf_viz.fit(x_train, y_train)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf_viz.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    # 训练样本散点
    colors = ['#4C72B0', '#55A868', '#C44E52']
    for c, i, name in zip(colors, [0, 1, 2], class_names):
        idx = y_train == i
        plt.scatter(x_train[idx, 0], x_train[idx, 1],
                    c=c, edgecolor='k', label=f'训练样本: {name}')
    # 测试样本用空心圆圈标出
    for c, i, name in zip(colors, [0, 1, 2], class_names):
        idx = y_test == i
        plt.scatter(x_test[idx, 0], x_test[idx, 1],
                    facecolors='none', edgecolors=c, marker='o', s=60,
                    label=f'测试样本: {name}')
    plt.xlabel('花萼长度 (cm)')
    plt.ylabel('花瓣长度 (cm)')
    plt.title('决策树(max_depth=3)在 Iris 训练数据上的分类效果图')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    p2 = os.path.join(BASE_DIR, 'result2_iris_boundary.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f'  分类效果图已保存: {p2}')

    # 额外：导出一棵树的结构图（便于实验报告粘贴）
    plt.figure(figsize=(14, 6))
    plot_tree(clf_viz,
              feature_names=['花萼长度', '花瓣长度'],
              class_names=list(class_names),
              filled=True, rounded=True, fontsize=9)
    plt.title('决策树结构 (max_depth=3)')
    p_tree = os.path.join(BASE_DIR, 'result2_tree_structure.png')
    plt.savefig(p_tree, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  决策树结构图已保存: {p_tree}')


if __name__ == '__main__':
    main()
