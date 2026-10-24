import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================
# 加载数据集
# =====================================================
data = load_breast_cancer()
X = data.data          # 形状：(569, 30)
y = data.target        # 0: malignant, 1: benign

print("样本数：", X.shape[0])
print("特征数：", X.shape[1])
print("类别名称：", data.target_names)
print("标签分布：", np.bincount(y))

# =====================================================
#划分训练集和测试集
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y          # 保持训练集和测试集中类别比例一致
)

# =====================================================
#特征征标准化
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
#构建神经网络模型
# =====================================================
# 输入层：30 个神经元（对应 30 个特征）
# 隐层：2 层，分别 30 和 15 个神经元
# 输出层：1 个神经元，用于二分类
mlp = MLPClassifier(
    hidden_layer_sizes=(30, 15),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42,
    early_stopping=True,      # 防止过拟合
    validation_fraction=0.1,
    n_iter_no_change=20
)

# =====================================================
#训练模型
# =====================================================
mlp.fit(X_train_scaled, y_train)

# =====================================================
#预测与评估
# =====================================================
y_pred = mlp.predict(X_test_scaled)
y_prob = mlp.predict_proba(X_test_scaled)[:, 1]   # 取良性概率用于 ROC

acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率：{acc:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=data.target_names))

cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵：")
print(cm)

# =====================================================
#绘制混淆矩阵
# =====================================================
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('乳腺癌分类混淆矩阵')
plt.colorbar()
tick_marks = np.arange(len(data.target_names))
plt.xticks(tick_marks, data.target_names, rotation=45)
plt.yticks(tick_marks, data.target_names)


for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.show()

# =====================================================
#绘制 ROC 曲线
# =====================================================
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC 曲线 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('假正率 FPR')
plt.ylabel('真正率 TPR')
plt.title('乳腺癌分类 ROC 曲线')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
#绘制训练损失曲线
# =====================================================
plt.figure(figsize=(6, 4))
plt.plot(mlp.loss_curve_, label='训练损失')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.title('神经网络训练损失曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()