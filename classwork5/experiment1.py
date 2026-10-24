import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=[[1, 1], [2, 2]],
    cluster_std=[0.6, 0.6],
    random_state=42
)

#  划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =====================================================
# 实验（1）：单隐层，神经元个数 1-10，观察测试精度变化
# =====================================================
neurons = list(range(1, 11))
acc_neurons = []

for n in neurons:
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(n,),   # 1 个隐层，n 个神经元
            activation='relu',         # 未指定时，这里选用常用 relu
            solver='adam',
            max_iter=2000,
            random_state=42
        )
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    acc_neurons.append(acc)
    print(f"隐层神经元个数 = {n:2d}, 测试精度 = {acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(neurons, acc_neurons, marker='o', linewidth=2)
plt.xlabel('隐层神经元个数')
plt.ylabel('测试集预测精度')
plt.title('单隐层神经元个数对测试精度的影响')
plt.xticks(neurons)
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('exp1_neurons_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# 实验（2）：单隐层，3 个神经元，比较不同激活函数
# =====================================================
activations = ['identity', 'logistic', 'tanh', 'relu']
acc_activations = []

for act in activations:
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(3,),   # 1 个隐层，3 个神经元
            activation=act,
            solver='adam',
            max_iter=2000,
            random_state=42
        )
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    acc_activations.append(acc)
    print(f"激活函数 = {act:8s}, 测试精度 = {acc:.4f}")

plt.figure(figsize=(8, 5))
bars = plt.bar(activations, acc_activations, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
plt.xlabel('激活函数')
plt.ylabel('测试集预测精度')
plt.title('单隐层 3 神经元时不同激活函数的测试精度')
plt.ylim(0, 1.05)
plt.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, acc_activations):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.01,
        f'{acc:.3f}',
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.savefig('exp2_activation_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()