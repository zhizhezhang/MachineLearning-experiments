# -*- coding: utf-8 -*-
"""
第2题：红酒质量二分类预测
quality<=5  -> 0 (质量差)
quality>5   -> 1 (质量好)
"""
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(os.path.join(
        BASE_DIR, '机器学习实验9-集成学习-第2题红酒数据.csv'))
    print(f'数据规模: {df.shape}')

    df['label'] = (df['quality'] > 5).astype(int)
    y = df['label'].values
    X = df.drop(columns=['quality', 'label']).values

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clfs = [
        ('Logistic回归', LogisticRegression(max_iter=1000)),
        ('支持向量机', SVC(probability=True, random_state=42)),
        ('决策树', DecisionTreeClassifier(random_state=42)),
        ('神经网络', MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=2000,
                                   random_state=42)),
    ]

    names, scores = [], []
    for name, clf in clfs:
        clf.fit(x_train, y_train)
        acc = accuracy_score(y_test, clf.predict(x_test))
        names.append(name)
        scores.append(acc)
        print(f'  {name}: {acc:.4f}')

    voting = VotingClassifier(estimators=clfs, voting='soft')
    voting.fit(x_train, y_train)
    names.append('软投票集成')
    scores.append(accuracy_score(y_test, voting.predict(x_test)))
    print(f'  软投票集成: {scores[-1]:.4f}')

    ada = AdaBoostClassifier(n_estimators=50, random_state=42)
    ada.fit(x_train, y_train)
    names.append('AdaBoost')
    scores.append(accuracy_score(y_test, ada.predict(x_test)))
    print(f'  AdaBoost: {scores[-1]:.4f}')

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(x_train, y_train)
    names.append('随机森林')
    scores.append(accuracy_score(y_test, rf.predict(x_test)))
    print(f'  随机森林: {scores[-1]:.4f}')

    plt.figure(figsize=(11, 5))
    plt.bar(names, scores, color=['#4C72B0', '#55A868', '#C44E52',
            '#8172B2', '#DD8452', '#937860', '#DA8BC3'])
    plt.title('红酒质量二分类 - 各模型测试集准确率')
    plt.ylabel('准确率')
    plt.ylim(0.5, 1.0)
    for i, v in enumerate(scores):
        plt.text(i, v + 0.005, f'{v:.3f}', ha='center')
    plt.xticks(rotation=15)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'q2_redwine.png')
    plt.savefig(p, dpi=150)
    plt.close()
    print(f'  图表已保存: {p}')


if __name__ == '__main__':
    main()
