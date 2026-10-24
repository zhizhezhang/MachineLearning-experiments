# -*- coding: utf-8 -*-
"""
实验6 任务3：IMDB 影评数据集 LSTM 文本情感二分类
  1) 加载 IMDB 数据集，做词汇截断、填充到固定长度
  2) 搭建 LSTM 网络完成文本情感二分类
  3) 输出训练 / 测试准确率，观察过拟合情况

数据使用 Keras 预处理好的 imdb.npz（公开 S3 链接），无需 TensorFlow。
理论问题写在文件末尾注释里。
"""

import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------- 超参数 ----------
MAX_WORDS = 20000     # 词表大小：只保留最常见的前 20000 个词
MAX_LEN = 200         # 每条评论截断 / 填充到固定长度 200
EMBED_DIM = 128
HIDDEN_SIZE = 64
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", DEVICE)

# ---------- 1. 下载并加载 IMDB（Keras 预处理格式） ----------
NPZ_URL = "https://s3.amazonaws.com/text-datasets/imdb.npz"
LOCAL_NPZ = "./imdb.npz"

if not os.path.exists(LOCAL_NPZ):
    print("下载 imdb.npz ...")
    urllib.request.urlretrieve(NPZ_URL, LOCAL_NPZ)

# Keras imdb.npz 内部键：x_train, y_train, x_test, y_test
with np.load(LOCAL_NPZ, allow_pickle=True) as d:
    x_train, y_train = d["x_train"], d["y_train"]
    x_test, y_test = d["x_test"], d["y_test"]

print("原始：", "x_train", x_train.shape, "x_test", x_test.shape)
print("标签分布：训练正例 %.2f / 负例 %.2f" % (
    y_train.mean(), 1 - y_train.mean()))


# ---------- 2. 词汇截断 + 填充到固定长度 ----------
def preprocess(sequences, max_len, max_words):
    """
    a) 词汇截断：把 >= max_words 的词索引替换为 oov 标记（简化为 max_words-1）
    b) 长度统一：不足 max_len 前面补 0，超过 max_len 取后 max_len 个词
    """
    out = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = [w if w < max_words else max_words - 1 for w in seq]
        if len(seq) > max_len:
            seq = seq[-max_len:]            # 截断：保留后 200 个词
        out[i, -len(seq):] = seq            # 填充：左边补 0
    return out


x_train_p = preprocess(x_train, MAX_LEN, MAX_WORDS)
x_test_p = preprocess(x_test, MAX_LEN, MAX_WORDS)

train_ds = TensorDataset(
    torch.from_numpy(x_train_p), torch.from_numpy(y_train.astype(np.float32)))
test_ds = TensorDataset(
    torch.from_numpy(x_test_p), torch.from_numpy(y_test.astype(np.float32)))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# ---------- 3. LSTM 模型 ----------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=MAX_WORDS, embed_dim=EMBED_DIM,
                 hidden_size=HIDDEN_SIZE, num_classes=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)   # 二分类，输出 1 logit

    def forward(self, x):
        # x: (B, L) 整数索引
        e = self.embed(x)                # (B, L, E)
        out, (h_n, c_n) = self.lstm(e)   # out: (B, L, H)
        last = h_n[-1]                    # (B, H) 取最后一层最后时间步
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)   # (B,) logit


model = LSTMClassifier().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ---------- 4. 训练 / 评估 ----------
def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logit = model(xb)
            loss = criterion(logit, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            pred = (torch.sigmoid(logit) >= 0.5).float()
            correct += (pred == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


history = {"train_loss": [], "test_loss": [],
           "train_acc": [], "test_acc": []}

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    te_loss, te_acc = run_epoch(test_loader, train=False)
    history["train_loss"].append(tr_loss)
    history["test_loss"].append(te_loss)
    history["train_acc"].append(tr_acc)
    history["test_acc"].append(te_acc)
    print(f"Epoch {epoch}/{EPOCHS}  "
          f"train_loss={tr_loss:.4f} acc={tr_acc * 100:.2f}%  |  "
          f"test_loss={te_loss:.4f} acc={te_acc * 100:.2f}%")

# ---------- 5. 绘图：loss 与准确率，观察过拟合 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history["train_loss"], marker="o", label="Train Loss")
axes[0].plot(history["test_loss"], marker="s", label="Test Loss")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history["train_acc"], marker="o", label="Train Acc")
axes[1].plot(history["test_acc"], marker="s", label="Test Acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("task3_lstm_curve.png", dpi=150)
plt.show()

print(f"\n最终：训练准确率 {history['train_acc'][-1] * 100:.2f}%，"
      f"测试准确率 {history['test_acc'][-1] * 100:.2f}%")
print("若 train_acc 明显高于 test_acc，且 test_loss 后期上升，即为过拟合。")


