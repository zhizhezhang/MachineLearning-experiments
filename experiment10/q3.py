# -*- coding: utf-8 -*-
"""
机器学习实验10 - 第3题：make_hastie_10_2 上的特征选择 / PCA + SVM
A1 原始特征 / A2 RFE / A3 L1惩罚项 / A4 树模型 / A5 PCA，柱状图对比
"""
import os
import warnings
import matplotlib.pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == '__main__':
    X, y = make_hastie_10_2(n_samples=1000, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    def svm_score(xtr, xte):
        clf = SVC(kernel='linear', random_state=42)
        clf.fit(xtr, y_train)
        return clf.score(xte, y_test)

    # A1：原始全部特征
    A1 = svm_score(x_train, x_test)

    # A2：RFE 递归特征删除，保留 5 个特征
    estimator = SVC(kernel='linear', random_state=42)
    rfe = RFE(estimator, n_features_to_select=5, step=1)
    rfe.fit(x_train, y_train)
    A2 = svm_score(rfe.transform(x_train), rfe.transform(x_test))

    # A3：基于 L1 惩罚项的特征选择（SelectFromModel + LinearSVC(l1)）
    lsvc = LinearSVC(penalty='l1', dual=False, random_state=42)
    lsvc.fit(x_train, y_train)
    sfm_l1 = SelectFromModel(lsvc, prefit=True)
    A3 = svm_score(sfm_l1.transform(x_train), sfm_l1.transform(x_test))

    # A4：基于树模型（随机森林）的特征选择
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    sfm_rf = SelectFromModel(rf, threshold='mean')
    sfm_rf.fit(x_train, y_train)
    A4 = svm_score(sfm_rf.transform(x_train), sfm_rf.transform(x_test))

    # A5：PCA 降维到 5 个主成分
    pca = PCA(n_components=5, random_state=42)
    x_train_p = pca.fit_transform(x_train)
    x_test_p = pca.transform(x_test)
    A5 = svm_score(x_train_p, x_test_p)

    names = ['A1\n原始特征', 'A2\nRFE', 'A3\nL1惩罚项', 'A4\n树模型', 'A5\nPCA']
    scores = [A1, A2, A3, A4, A5]
    for n, s in zip(names, scores):
        print(f'{n.replace(chr(10), "")} 测试精度 = {s:.4f}')

    plt.figure(figsize=(8, 5))
    colors = ['#bdbdbd', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
    bars = plt.bar(names, scores, color=colors)
    for b, v in zip(bars, scores):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.3f}',
                 ha='center', va='bottom')
    plt.ylim(0, 1.0)
    plt.ylabel('测试集分类精度')
    plt.title('不同特征选择 / PCA 方法下 SVM 测试精度对比')
    plt.legend(bars, names, title='方法', loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'q3_feature_selection_compare.png')
    plt.savefig(p, dpi=150)
    plt.close()
    print('已保存：', p)
