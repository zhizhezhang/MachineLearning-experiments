import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    X, y = make_blobs(n_samples=500, n_features=2,
                      centers=[[1, 1], [2, 2]], cluster_std=[0.6, 0.6],
                      random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    clfs = [
        ('Logistic回归', LogisticRegression(max_iter=1000)),
        ('支持向量机', SVC(probability=True, random_state=42)),
        ('决策树', DecisionTreeClassifier(random_state=42)),
        ('神经网络', MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000,
                                   random_state=42)),
    ]
    names = [n for n, _ in clfs]

    # (1) 硬投票 & 软投票
    hard_voting = VotingClassifier(estimators=clfs, voting='hard')
    soft_voting = VotingClassifier(estimators=clfs, voting='soft')

    scores_hard = []
    for name, clf in clfs:
        clf.fit(x_train, y_train)
        scores_hard.append(accuracy_score(y_test, clf.predict(x_test)))
    hard_voting.fit(x_train, y_train)
    scores_hard.append(accuracy_score(y_test, hard_voting.predict(x_test)))
    names_hard = names + ['硬投票集成']

    soft_voting.fit(x_train, y_train)
    scores_soft = []
    for name, clf in clfs:
        scores_soft.append(accuracy_score(y_test, clf.predict(x_test)))
    scores_soft.append(accuracy_score(y_test, soft_voting.predict(x_test)))
    names_soft = names + ['软投票集成']

    print('(1) 硬投票/软投票 测试集精度：')
    for n, s in zip(names_hard, scores_hard):
        print(f'  {n}: {s:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(names_hard, scores_hard, color=['#4C72B0', '#55A868',
               '#C44E52', '#8172B2', '#DD8452'])
    axes[0].set_title('硬投票 VotingClassifier(voting=hard)')
    axes[0].set_ylabel('测试集准确率')
    axes[0].set_ylim(0.8, 1.0)
    for i, v in enumerate(scores_hard):
        axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center')

    axes[1].bar(names_soft, scores_soft, color=['#4C72B0', '#55A868',
                '#C44E52', '#8172B2', '#DD8452'])
    axes[1].set_title('软投票 VotingClassifier(voting=soft)')
    axes[1].set_ylabel('测试集准确率')
    axes[1].set_ylim(0.8, 1.0)
    for i, v in enumerate(scores_soft):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center')
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, 'q1_hard_soft_voting.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f'  图表已保存: {p1}')

    # (2) 平均法 & 加权平均法
    weights = [0.1, 0.4, 0.2, 0.3]
    probs, single_scores = [], []
    for name, clf in clfs:
        p = clf.predict_proba(x_test)
        probs.append(p)
        single_scores.append(accuracy_score(y_test, np.argmax(p, axis=1)))
    probs = np.array(probs)

    prob_mean = probs.mean(axis=0)
    score_mean = accuracy_score(y_test, np.argmax(prob_mean, axis=1))

    w = np.array(weights).reshape(-1, 1, 1)
    prob_weighted = (probs * w).sum(axis=0)
    score_weighted = accuracy_score(y_test, np.argmax(prob_weighted, axis=1))

    names_avg = names + ['平均法集成', '加权平均法集成']
    scores_avg = single_scores + [score_mean, score_weighted]

    print('\n(2) 平均法/加权平均法 测试集精度：')
    for n, s in zip(names_avg, scores_avg):
        print(f'  {n}: {s:.4f}')

    plt.figure(figsize=(10, 5))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3']
    plt.bar(names_avg, scores_avg, color=colors)
    plt.title('平均法 与 加权平均法(权重[0.1,0.4,0.2,0.3])')
    plt.ylabel('测试集准确率')
    plt.ylim(0.8, 1.0)
    for i, v in enumerate(scores_avg):
        plt.text(i, v + 0.005, f'{v:.3f}', ha='center')
    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, 'q1_avg_weighted_avg.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f'  图表已保存: {p2}')


if __name__ == '__main__':
    main()
