import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 计算不同噪声下的测试精度
# ============================================================
noise_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
acc_list = []

for noise in noise_list:
    x, y = make_moons(n_samples=1000, noise=noise, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.4, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    acc_list.append(acc)

    print(f"噪声 noise = {noise:.1f}，测试精度 = {acc:.4f}")

# ============================================================
# 2. 画出预测精度随噪声的变化曲线
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(noise_list, acc_list, marker='o', linewidth=2)
plt.xlabel('噪声 noise')
plt.ylabel('测试精度 accuracy')
plt.title('Logistic回归预测精度随噪声的变化')
plt.xticks(noise_list)
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ============================================================
# 3. 画出样本数为1000、噪声为0.3时的样本与分类界线
# ============================================================
x, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n噪声 noise = 0.3 时，测试精度 = {acc:.4f}")

x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict_proba(grid_points)[:, 1]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(xx, yy, Z, levels=20, alpha=0.3, cmap='RdBu_r')

plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=2)

plt.scatter(
    x_train[:, 0], x_train[:, 1],
    c=y_train, cmap='RdBu_r',
    edgecolors='k', marker='o', s=40,
    label='训练数据'
)

plt.scatter(
    x_test[:, 0], x_test[:, 1],
    c=y_test, cmap='RdBu_r',
    edgecolors='k', marker='^', s=50,
    label='测试数据'
)

plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title(f'Logistic回归分类边界（noise=0.3，测试精度={acc:.3f}）')
plt.legend()
plt.show()