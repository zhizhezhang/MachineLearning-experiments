# -*- coding: utf-8 -*-
"""
机器学习实验10 - 第4题：婚姻样本 朴素贝叶斯
x1 长相(0不帅/1帅), x2 身高(0矮/1高), x3 经济(0没钱/1有钱), y(0不嫁/1嫁)
"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB

if __name__ == '__main__':
    X = np.array([
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
    ])
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0])

    model = BernoulliNB()
    model.fit(X, y)
    print(f'训练集准确率：{model.score(X, y):.3f}')

    # 演示：几个典型男生的预测
    tests = np.array([
        [0, 0, 0],   # 不帅 + 矮 + 没钱
        [1, 1, 1],   # 帅 + 高 + 有钱
        [1, 0, 1],   # 帅 + 矮 + 有钱
    ])
    pred = model.predict(tests)
    proba = model.predict_proba(tests)
    for t, p, pr in zip(tests, pred, proba):
        print(f'特征{list(t)} -> 预测:{"嫁" if p == 1 else "不嫁"}  '
              f'(不嫁概率={pr[0]:.3f}, 嫁概率={pr[1]:.3f})')
