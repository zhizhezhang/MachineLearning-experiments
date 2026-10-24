# -*- coding: utf-8 -*-
"""
实验6 任务2：AirPassengers 国际航空乘客月度数据（1949-1960）单层 RNN 时序预测
  1) 加载并预处理数据集，做归一化
  2) 构造时序样本：用前 seq_len 个月数据预测下一个月乘客数量
  3) 搭建单层 RNN 模型完成训练
  4) 在测试集上预测，绘制真实值 vs 预测值对比曲线

理论问题写在文件末尾注释里。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ---------- 超参数 ----------
SEQ_LEN = 12          # 用前 12 个月（1 年）预测下一个月
HIDDEN_SIZE = 32
EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-3
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", DEVICE)

# ---------- 1. 加载数据 ----------
# 优先读同目录下的本地 CSV（国内访问 github raw 经常被重置）；
# 本地没有再尝试联网下载。
import os
LOCAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "airline-passengers.csv")
if os.path.exists(LOCAL_CSV):
    df = pd.read_csv(LOCAL_CSV)
else:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url)

data = df["Passengers"].values.astype(np.float32).reshape(-1, 1)
dates = pd.to_datetime(df["Month"])
print("数据形状：", data.shape, " 时间范围：", dates.iloc[0].date(), "->", dates.iloc[-1].date())

# ---------- 2. 归一化（只用训练集拟合 scaler，避免数据泄露） ----------
train_size = int(len(data) * TRAIN_RATIO)
train_raw = data[:train_size]
test_raw = data[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_raw)
test_scaled = scaler.transform(test_raw)

# 拼接后构造滑窗样本（测试集的前 SEQ_LEN 个点要借训练集末尾的窗口）
all_scaled = np.vstack([train_scaled, test_scaled])


# ---------- 3. 构造滑窗样本 ----------
def make_dataset(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i: i + seq_len])       # (seq_len, 1)
        y.append(series[i + seq_len])           # (1,)
    return np.array(X), np.array(y)


# 训练样本：来自 all_scaled 中对应训练区间的位置
train_end = train_size
X_train, y_train = make_dataset(all_scaled[:train_end], SEQ_LEN)
# 测试样本：从训练集末尾的 SEQ_LEN 个点开始滑，覆盖所有测试点
X_test, y_test = make_dataset(all_scaled[train_end - SEQ_LEN:], SEQ_LEN)

print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_test :", X_test.shape,  " y_test :", y_test.shape)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()     # (N, seq_len, 1)
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(TimeSeriesDataset(X_train, y_train),
                         batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test),
                         batch_size=BATCH_SIZE, shuffle=False)


# ---------- 4. 单层 RNN 模型 ----------
class SingleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, output_size=1):
        super().__init__()
        # num_layers=1 -> 单层 RNN；batch_first=True 让输入形状为 (B, seq_len, F)
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True,
                          nonlinearity="tanh")
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, seq_len, 1)
        out, h_n = self.rnn(x)        # out: (B, seq_len, H); h_n: (1, B, H)
        last = out[:, -1, :]          # 取最后一个时间步的输出
        return self.fc(last)          # (B, 1)


model = SingleRNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------- 5. 训练 ----------
loss_history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg = total_loss / len(train_loader.dataset)
    loss_history.append(avg)
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3}/{EPOCHS}  train_mse={avg:.5f}")

# ---------- 6. 测试集预测 + 反归一化 ----------
model.eval()
preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(DEVICE)
        preds.append(model(xb).cpu().numpy())
preds = np.concatenate(preds, axis=0)              # (len(test), 1)

# 反归一化回到原始乘客数量量级
preds_inv = scaler.inverse_transform(preds)
y_test_inv = scaler.inverse_transform(y_test)

# 训练集拟合曲线（用于画在同一张图上看整体效果）
with torch.no_grad():
    Xb = torch.from_numpy(X_train).float().to(DEVICE)
    train_pred = model(Xb).cpu().numpy()
train_pred_inv = scaler.inverse_transform(train_pred)
y_train_inv = scaler.inverse_transform(y_train)

# ---------- 7. 绘图 ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

# 7.1 训练 loss
axes[0].plot(loss_history, color="#4C72B0")
axes[0].set_title("Training MSE Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE (scaled)")
axes[0].grid(alpha=0.3)

# 7.2 真实值 vs 预测值（全时间轴）
# 训练段预测位置：从 SEQ_LEN 开始
train_pred_x = dates[SEQ_LEN: train_size]
test_pred_x = dates[train_size: ]

axes[1].plot(dates, data, label="Ground Truth", color="black", linewidth=1.5)
axes[1].plot(train_pred_x, train_pred_inv, label="Train Pred", color="#4C72B0", alpha=0.7)
axes[1].plot(test_pred_x, preds_inv, label="Test Pred", color="#DD8452", linewidth=1.8)
axes[1].axvline(dates[train_size], color="gray", linestyle="--", alpha=0.6, label="train/test split")
axes[1].set_title("AirPassengers: True vs Predicted (Single-layer RNN)")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Passengers")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("task2_rnn_result.png", dpi=150)
plt.show()

# 简单评估指标
mae = np.mean(np.abs(preds_inv - y_test_inv))
print(f"测试集 MAE（乘客数）: {mae:.2f}")

# ============================================================
# 理论问题
# ------------------------------------------------------------
# (1) 普通 RNN 在长序列训练中的主要缺陷：
#     难以建模长距离依赖。当输入序列很长时，梯度在沿时间步反向传播
#     （BPTT）的过程中容易出现「梯度消失」（多数情况）或「梯度爆炸」
#     （少数情况），导致网络无法学习到早期时间步和后期输出之间的关联，
#     训练收敛慢、效果差。
#
# (2) 造成该缺陷的数学原因：
#     RNN 在时间步 t 的隐状态为 h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)。
#     反向传播时，损失对某一时刻参数的梯度会连乘多项 W_hh^T 与 tanh
#     导数（tanh' ≤ 1）。当序列很长时：
#       - 若 |W_hh| 的谱半径 < 1，连乘后梯度按指数衰减 -> 梯度消失；
#       - 若 |W_hh| 的谱半径 > 1，连乘后梯度按指数放大 -> 梯度爆炸。
#     本质上是「随时间步反复使用同一组权重」造成的连乘效应，没有像 LSTM
#     那样的门控机制来选择性保留/遗忘信息，因此无法稳定地传递长程梯度。
# ============================================================
