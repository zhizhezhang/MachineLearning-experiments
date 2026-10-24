import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = [
    [59, 2, 43.4, 2, 1, 0],
    [36, 1, 57.2, 1, 1, 0],
    [61, 2, 190.0, 2, 1, 0],
    [58, 3, 128.0, 4, 3, 1],
    [55, 3, 80.0, 3, 4, 1],
    [61, 1, 94.4, 2, 1, 0],
    [38, 1, 76.0, 1, 1, 0],
    [42, 1, 240.0, 3, 2, 1],
    [50, 1, 74.0, 1, 1, 0],
    [58, 3, 68.6, 2, 2, 0],
    [68, 3, 132.8, 4, 2, 0],
    [25, 2, 94.6, 4, 3, 1],
    [52, 1, 56.0, 1, 1, 0],
    [31, 1, 47.8, 2, 1, 0],
    [36, 3, 31.6, 3, 1, 1],
    [42, 1, 66.2, 2, 1, 0]
]

columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
df = pd.DataFrame(data, columns=columns)

X = df[['x1', 'x2', 'x3', 'x4', 'x5']]
y = df['y']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print('回归系数：', model.coef_)
print('截距：', model.intercept_)

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

print('预测类别：', y_pred)
print('预测为1的概率：', y_prob)
print('真实类别：', y.values)
print('准确率：', accuracy_score(y, y_pred))
print('混淆矩阵：')
print(confusion_matrix(y, y_pred))
print('分类报告：')
print(classification_report(y, y_pred))