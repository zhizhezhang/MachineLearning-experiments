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


##########  生成 100 个样本，划分为 60 个训练样本和 40 个测试样本
X, y = make_moons(n_samples=100, noise=0.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

print("训练样本数：", len(X_train))
print("测试样本数：", len(X_test))

# K 从 1 到 50 变化时的测试准确率：
ks = list(range(1, 51))
accuracies = []

for k in ks:
    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

result_k = pd.DataFrame({
    "K": ks,
    "测试准确率": accuracies
})
print(result_k.to_string(index=False))

best_accuracy = max(accuracies)
best_ks = [
    k for k, acc in zip(ks, accuracies)
    if acc == best_accuracy
]
print("\n测试准确率最高的K：", best_ks)
print(f"最高测试准确率：{best_accuracy:.2%}")

plt.figure(figsize=(9, 5))
plt.plot(ks, accuracies, marker="o", markersize=3)
plt.xlabel("K值")
plt.ylabel("测试准确率")
plt.title("K值对KNN分类准确率的影响")
plt.xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("K值与准确率.png", dpi=200)
plt.show()

######### 比较 K=5、10、20 时，两种距离的表现
k_values = [5, 10, 20]
euclidean_scores = []
manhattan_scores = []

for k in k_values:
    for metric, scores in [
        ("euclidean", euclidean_scores),
        ("manhattan", manhattan_scores)
    ]:
        model = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric
        )
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))

comparison = pd.DataFrame({
    "K": k_values,
    "欧氏距离准确率": euclidean_scores,
    "曼哈顿距离准确率": manhattan_scores
})
print(comparison.to_string(index=False))

positions = np.arange(len(k_values))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(
    positions - width / 2, euclidean_scores,
    width, label="欧氏距离"
)
bars2 = ax.bar(
    positions + width / 2, manhattan_scores,
    width, label="曼哈顿距离"
)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.1%}",
            ha="center"
        )

ax.set_xticks(positions)
ax.set_xticklabels(k_values)
ax.set_xlabel("K值")
ax.set_ylabel("测试准确率")
ax.set_title("不同距离类型的KNN分类效果")
ax.set_ylim(0, 1.15)
ax.legend()
plt.tight_layout()
plt.savefig("距离类型对比.png", dpi=200)
plt.show()