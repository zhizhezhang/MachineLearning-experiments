# -*- coding: utf-8 -*-
"""
实验7-3：基于 SVM 的印第安人糖尿病预测
使用 207 条糖尿病诊断数据，训练 rbf 核 SVM 进行二分类预测，
输出训练/测试精度、混淆矩阵与分类报告，并画出混淆矩阵图。
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

# ---------- 中文显示设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print('=' * 60)
    print('实验7-3：基于 SVM 的糖尿病预测')
    print('=' * 60)

    # 1. 读取数据
    data_path = os.path.join(BASE_DIR, 'diabetes.csv')
    df = pd.read_csv(data_path)
    print(f'数据集大小: {df.shape}')
    print(f'特征列: {list(df.columns[:-1])}')
    print(f'各类样本数量:\n{df["Outcome"].value_counts()}\n')

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 2. 划分训练集/测试集（80% / 20%，分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. 数据标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. 训练 rbf 核 SVM
    clf = SVC(kernel='rbf', gamma='scale', C=1.0)
    clf.fit(X_train_s, y_train)

    # 5. 评估
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f'训练集精度: {clf.score(X_train_s, y_train):.4f}')
    print(f'测试集精度: {acc:.4f}')
    print(f'混淆矩阵:\n{cm}')
    print(f'分类报告:\n{classification_report(y_test, y_pred, digits=4)}')

    # 6. 画混淆矩阵图
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('糖尿病预测 SVM 混淆矩阵')
    plt.colorbar()
    classes = ['非糖尿病(0)', '糖尿病(1)']
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black',
                     fontsize=14)
    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'result3_diabetes_cm.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'混淆矩阵图已保存: {out}')


if __name__ == '__main__':
    main()
