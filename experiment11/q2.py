# -*- coding: utf-8 -*-
"""
实验11-2：自编码器（AutoEncoder）在 MNIST 上的实现
内容：
  1. 构建隐层数分别为 2 与 5 的两个自编码-解码网络（激活函数 ReLU）；
  2. 在同一坐标系下绘制两个网络的训练误差曲线；
  3. 从测试集取 5 张图，两行五列展示原图与 5 层网络的编码解码结果；
  4. 随机产生 5 个向量，送入 5 层网络的解码器生成图像。
运行环境：Python 3.13 + PyTorch 2.x (CPU) + torchvision + matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- 中文显示 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, 'data')

device = torch.device('cpu')
torch.manual_seed(42)
np.random.seed(42)


# ---------- 1. 数据：MNIST ----------
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True,
                              transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=0)
    print(f'[数据] 训练集 {len(train_set)} 张，测试集 {len(test_set)} 张')
    return train_loader, test_loader, test_set


# ---------- 2. 自编码器结构 ----------
class Autoencoder(nn.Module):
    """
    hidden_dims: encoder 中每个全连接隐藏层的宽度列表。
        e.g. [128, 32]   -> 784-128-32-128-784  (2 个隐层)
             [256,128,64,32,16] -> 784-...-16-...-784 (5 个隐层)
    隐层激活 ReLU；输出层 Sigmoid（像素值在 [0,1]）。
    """

    def __init__(self, hidden_dims):
        super().__init__()
        # encoder
        enc = []
        in_dim = 784
        for h in hidden_dims:
            enc.append(nn.Linear(in_dim, h))
            enc.append(nn.ReLU(inplace=True))
            in_dim = h
        self.encoder = nn.Sequential(*enc)

        # decoder（对称镜像）
        dec = []
        in_dim = hidden_dims[-1]
        for h in reversed(hidden_dims[:-1]):
            dec.append(nn.Linear(in_dim, h))
            dec.append(nn.ReLU(inplace=True))
            in_dim = h
        dec.append(nn.Linear(in_dim, 784))
        dec.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


# ---------- 3. 训练 ----------
def train(model, train_loader, epochs=10, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    for ep in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for x, _ in train_loader:
            x = x.view(-1, 784).to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        avg = total_loss / n
        losses.append(avg)
        print(f'   epoch {ep+1:2d}/{epochs}  MSE = {avg:.5f}')
    return losses


# ---------- 4. 可视化：训练损失曲线 ----------
def plot_loss_curve(loss2, loss5):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss2) + 1), loss2, 'o-',
             label='2 隐层自编码器')
    plt.plot(range(1, len(loss5) + 1), loss5, 's-',
             label='5 隐层自编码器')
    plt.xlabel('训练轮次 Epoch')
    plt.ylabel('MSE 重建误差')
    plt.title('两个自编码器训练误差变化曲线')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'ae1_loss_curve.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[图] 已保存：{out}')


# ---------- 5. 可视化：原图 vs 5 层网络重构 ----------
def plot_reconstruction(model5, test_set, n=5):
    model5.eval()
    idxs = [0, 5, 10, 15, 20]   # 取测试集中前 5 张不同位置的图
    with torch.no_grad():
        imgs = torch.stack([test_set[i][0] for i in idxs]).view(-1, 784)
        recon, _ = model5(imgs.to(device))
        recon = recon.view(-1, 28, 28).cpu().numpy()
        orig = imgs.view(-1, 28, 28).numpy()

    fig, axes = plt.subplots(2, n, figsize=(10, 4.5))
    for j in range(n):
        axes[0, j].imshow(orig[j], cmap='gray')
        axes[0, j].axis('off')
        axes[1, j].imshow(recon[j], cmap='gray')
        axes[1, j].axis('off')
    axes[0, 0].set_ylabel('原图', fontsize=12)
    axes[1, 0].set_ylabel('5层网络\n重建', fontsize=12)
    fig.suptitle('测试集原图（上）与 5 隐层自编码器重建结果（下）', fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'ae2_reconstruction.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[图] 已保存：{out}')


# ---------- 6. 可视化：随机 latent 向量解码生成图像 ----------
def plot_random_generation(model5, train_loader, n=5):
    """用训练集 latent 的均值/方差作为先验，采样 n 个向量送入 decoder。"""
    model5.eval()
    # 统计训练集 latent 的 μ、σ
    mu = torch.zeros(model5.encoder[-2].out_features)
    var = torch.zeros_like(mu)
    cnt = 0
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.view(-1, 784).to(device)
            z = model5.encoder(x)
            mu += z.sum(dim=0).cpu()
            var += (z.cpu() ** 2).sum(dim=0)
            cnt += x.size(0)
    mu /= cnt
    std = torch.sqrt(var / cnt - mu ** 2).clamp_min(1e-6)

    # 采样 n 个 latent
    torch.manual_seed(0)
    z_rand = mu + std * torch.randn(n, mu.shape[0])
    with torch.no_grad():
        gen = model5.decoder(z_rand.to(device)).view(-1, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(1, n, figsize=(10, 2.5))
    for j in range(n):
        axes[j].imshow(gen[j], cmap='gray')
        axes[j].axis('off')
    fig.suptitle('随机向量作为编码，经 5 层网络解码器生成的图像', fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'ae3_random_generation.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[图] 已保存：{out}')


def main():
    train_loader, test_loader, test_set = load_mnist()

    # 两个自编码器：隐层数 2 与 5
    model2 = Autoencoder(hidden_dims=[128, 32])
    model5 = Autoencoder(hidden_dims=[256, 128, 64, 32, 16])

    print('[训练] 2 隐层自编码器 ...')
    loss2 = train(model2, train_loader, epochs=10)
    print('[训练] 5 隐层自编码器 ...')
    loss5 = train(model5, train_loader, epochs=10)

    plot_loss_curve(loss2, loss5)
    plot_reconstruction(model5, test_set, n=5)
    plot_random_generation(model5, train_loader, n=5)
    print('[完成] 自编码器实验全部结束。')


if __name__ == '__main__':
    main()
