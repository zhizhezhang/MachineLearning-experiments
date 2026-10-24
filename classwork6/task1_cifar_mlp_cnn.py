# -*- coding: utf-8 -*-
"""
实验6 任务1：利用 CIFAR-10 数据集构建图像识别模型
  1) 灰度图拉伸为向量 -> 2 层全连接神经网络（隐层激活 ReLU）
  2) 灰度图 -> 2 层卷积神经网络（卷积层 + ReLU + 最大池化层）
  3) 同一坐标系下绘制两网络的训练误差（loss）变化曲线
  4) 绘制两网络在测试集上精度的柱状图

运行环境：Python 3.x + PyTorch + torchvision + matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ---------- 超参数 ----------
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
HIDDEN_DIM = 256          # MLP 隐层维度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", DEVICE)

# ---------- 数据：CIFAR-10，转灰度图 ----------
# 灰度化后像素值归一化到 [0,1]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # RGB -> 单通道灰度图
    transforms.ToTensor(),                         # 转 Tensor，值域 [0,1]
])

data_root = "./data"
train_set = torchvision.datasets.CIFAR10(
    root=data_root, train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(
    root=data_root, train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# CIFAR-10 类别
classes = train_set.classes


# ========== 模型1：2 层全连接神经网络（MLP） ==========
# "层次为 2" = 输入层 -> 隐层（ReLU）-> 输出层
class MLP(nn.Module):
    def __init__(self, in_dim=32 * 32, hidden_dim=HIDDEN_DIM, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)   # 第 1 层：线性
        self.relu = nn.ReLU()                       # 激活
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # 第 2 层：线性输出

    def forward(self, x):
        # x: (B, 1, 32, 32) -> 拉伸为 (B, 1*32*32)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ========== 模型2：2 层卷积神经网络（CNN） ==========
# 每个卷积块 = 卷积层 + ReLU + 最大池化层；共 2 个卷积块，最后接全连接分类
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 第 1 个卷积块：1 -> 32 通道，kernel=3, padding=1；池化后 32x32 -> 16x16
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第 2 个卷积块：32 -> 64 通道；池化后 16x16 -> 8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 分类头：64 * 8 * 8 -> 10
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))   # (B,32,16,16)
        x = self.pool2(self.relu2(self.conv2(x)))   # (B,64,8,8)
        x = x.view(x.size(0), -1)                   # 展平
        x = self.fc(x)
        return x


# ---------- 通用训练 / 评估函数 ----------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_model(model, name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_history = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        loss_history.append(loss)
        if epoch % 1 == 0:
            print(f"[{name}] Epoch {epoch:>2}/{EPOCHS}  train_loss={loss:.4f}")

    acc = evaluate(model, test_loader)
    print(f"[{name}] 测试集准确率 = {acc * 100:.2f}%\n")
    return loss_history, acc


# ---------- 训练两个模型 ----------
print("=" * 50)
print("开始训练 MLP ...")
mlp_loss, mlp_acc = train_model(MLP(), "MLP")

print("=" * 50)
print("开始训练 CNN ...")
cnn_loss, cnn_acc = train_model(CNN(), "CNN")

# ---------- 图1：同一坐标系下训练误差曲线 ----------
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), mlp_loss, marker="o", label="MLP (2-layer FC)")
plt.plot(range(1, EPOCHS + 1), cnn_loss, marker="s", label="CNN (2-layer Conv)")
plt.xlabel("Epoch")
plt.ylabel("Training Loss (CrossEntropy)")
plt.title("Training Loss Curve on CIFAR-10 (Gray)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("task1_loss_curve.png", dpi=150)
plt.show()

# ---------- 图2：测试集精度柱状图 ----------
plt.figure(figsize=(6, 5))
bars = plt.bar(["MLP", "CNN"], [mlp_acc * 100, cnn_acc * 100],
               color=["#4C72B0", "#DD8452"], width=0.5)
for bar, v in zip(bars, [mlp_acc * 100, cnn_acc * 100]):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
             f"{v:.2f}%", ha="center", fontsize=11)
plt.ylim(0, 100)
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy on CIFAR-10 (Gray)")
plt.tight_layout()
plt.savefig("task1_accuracy_bar.png", dpi=150)
plt.show()

print("已保存：task1_loss_curve.png, task1_accuracy_bar.png")
