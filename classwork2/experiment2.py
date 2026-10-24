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


# 每一行依次为：打斗镜头数、拥抱镜头数
X_movie = np.array([
    [101, 30],
    [80, 10],
    [76, 27],
    [10, 78],
    [25, 80],
    [45, 120],
    [18, 40],
    [89, 30],
    [54, 50]
])

y_movie = np.array([
    "动作片", "动作片", "动作片",
    "爱情片", "爱情片", "爱情片",
    "爱情片", "动作片", "动作片"
])

movie_model = KNeighborsClassifier(
    n_neighbors=3,
    metric="euclidean"
)
movie_model.fit(X_movie, y_movie)

fight = int(input("请输入打斗镜头数："))
hug = int(input("请输入拥抱镜头数："))

if fight < 0 or hug < 0:
    raise ValueError("镜头数量不能为负数")

new_movie = np.array([[fight, hug]])
prediction = movie_model.predict(new_movie)[0]

# 查看最近的3个训练样本，解释分类依据
distances, indices = movie_model.kneighbors(new_movie)

neighbors = pd.DataFrame({
    "样本序号": indices[0] + 1,
    "距离": distances[0],
    "电影类型": y_movie[indices[0]]
})

print("\n最近的3个样本：")
print(neighbors.to_string(index=False))
print("\n预测电影类型：", prediction)
print("小明是否会看：", "会看" if prediction == "动作片" else "不会看")