# -*- coding: utf-8 -*-
"""
第4题：make_classification 数据 + AdaBoost vs 随机森林 (n_estimators=5)
"""
import os
import warnings
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    # (1) AdaBoost 弱分类器数=5
    ada = AdaBoostClassifier(n_estimators=5, random_state=42)
    ada.fit(x_train, y_train)
    A1 = accuracy_score(y_test, ada.predict(x_test))

    # (2) 随机森林 弱分类器数=5
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(x_train, y_train)
    A2 = accuracy_score(y_test, rf.predict(x_test))

    print(f'A1 (AdaBoost, n_estimators=5) = {A1:.4f}')
    print(f'A2 (随机森林, n_estimators=5) = {A2:.4f}')

    # (3) 折线图：随弱分类器个数变化
    n_list = list(range(1, 21))
    ada_curve, rf_curve = [], []
    for n in n_list:
        a = AdaBoostClassifier(n_estimators=n, random_state=42)
        a.fit(x_train, y_train)
        ada_curve.append(accuracy_score(y_test, a.predict(x_test)))

        r = RandomForestClassifier(n_estimators=n, random_state=42)
        r.fit(x_train, y_train)
        rf_curve.append(accuracy_score(y_test, r.predict(x_test)))

    plt.figure(figsize=(9, 5))
    plt.plot(n_list, ada_curve, marker='o', label='AdaBoost')
    plt.plot(n_list, rf_curve, marker='s', label='随机森林')
    plt.scatter([5], [A1], s=120, facecolors='none', edgecolors='blue',
                linewidths=2, zorder=5)
    plt.scatter([5], [A2], s=120, facecolors='none', edgecolors='orange',
                linewidths=2, zorder=5)
    plt.annotate(f'A1={A1:.3f}', (5, A1), textcoords='offset points',
                 xytext=(8, 10), color='blue')
    plt.annotate(f'A2={A2:.3f}', (5, A2), textcoords='offset points',
                 xytext=(8, -15), color='darkorange')
    plt.xlabel('弱分类器个数 n_estimators')
    plt.ylabel('测试集准确率')
    plt.title('AdaBoost vs 随机森林 - 精度随弱分类器个数变化')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'q4_adaboost_vs_rf.png')
    plt.savefig(p, dpi=150)
    plt.close()
    print(f'图表已保存: {p}')


if __name__ == '__main__':
    main()
