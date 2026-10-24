import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 中文图表设置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


X_iris = np.array([
    [4.8, 3.0, 1.4, 0.1],
    [4.9, 3.0, 1.4, 0.2],
    [4.3, 3.0, 1.1, 0.1],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [4.6, 3.4, 1.4, 0.3],
    [5.7, 2.8, 4.5, 1.3],
    [6.4, 3.2, 4.5, 1.5],
    [5.2, 2.7, 3.9, 1.4],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5],
    [5.7, 2.8, 4.5, 1.3],
    [6.3, 3.3, 6.0, 2.5],
    [5.8, 2.7, 5.1, 1.9],
    [7.1, 3.0, 5.9, 2.1],
    [6.3, 2.9, 5.6, 1.8],
    [6.5, 3.0, 5.8, 2.2],
    [7.6, 3.0, 6.6, 2.1]
])

y_iris = np.array(
    ["Iris-setosa"] * 6
    + ["Iris-versicolor"] * 6
    + ["Iris-virginica"] * 6
)

ids = np.arange(1, 19)

X_train_i, X_test_i, y_train_i, y_test_i, _, test_ids = (
    train_test_split(
        X_iris, y_iris, ids,
        test_size=6,
        random_state=42,
        stratify=y_iris
    )
)

iris_model = KNeighborsClassifier(
    n_neighbors=3,
    metric="euclidean"
)
iris_model.fit(X_train_i, y_train_i)
y_pred_i = iris_model.predict(X_test_i)

result_iris = pd.DataFrame({
    "样本ID": test_ids,
    "真实类别": y_test_i,
    "预测类别": y_pred_i,
    "是否正确": y_test_i == y_pred_i
})
print(result_iris.to_string(index=False))
print(f"\n测试准确率：{accuracy_score(y_test_i, y_pred_i):.2%}")

class_names = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
]
cm = confusion_matrix(
    y_test_i, y_pred_i, labels=class_names
)
print("\n混淆矩阵（行是真实类别，列是预测类别）：")
print(pd.DataFrame(
    cm, index=class_names, columns=class_names
))