# -*- coding: utf-8 -*-
"""
机器学习实验10 - 第5题：天气/服装/情绪 朴素贝叶斯
预测“天气=好、服装=随意”的人的心情好坏
Weather: 不好=0/好=1；Clothes: 随意=0/精心=1；Mood: 难过=0/开心=1
"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB

if __name__ == '__main__':
    data = [
        ('好',   '随意', '开心'),
        ('好',   '精心', '开心'),
        ('好',   '随意', '难过'),
        ('不好', '精心', '难过'),
        ('不好', '随意', '开心'),
        ('好',   '随意', '开心'),
        ('好',   '精心', '开心'),
        ('不好', '精心', '难过'),
        ('不好', '随意', '难过'),
        ('好',   '随意', '开心'),
        ('不好', '精心', '开心'),
        ('好',   '随意', '难过'),
    ]
    w_map = {'不好': 0, '好': 1}
    c_map = {'随意': 0, '精心': 1}
    m_map = {'难过': 0, '开心': 1}

    X = np.array([[w_map[w], c_map[c]] for w, c, _ in data])
    y = np.array([m_map[m] for _, _, m in data])

    model = BernoulliNB()
    model.fit(X, y)
    print(f'训练集准确率：{model.score(X, y):.3f}')

    # 预测：天气=好(1)、服装=随意(0)
    x_new = np.array([[1, 0]])
    pred = model.predict(x_new)[0]
    proba = model.predict_proba(x_new)[0]
    print('预测样本：天气=好, 服装=随意')
    print(f'  -> 心情预测：{"开心" if pred == 1 else "难过"}')
    print(f'  -> 难过概率={proba[0]:.3f}, 开心概率={proba[1]:.3f}')
