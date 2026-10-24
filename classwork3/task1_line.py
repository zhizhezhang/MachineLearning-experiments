"""第1题：手写最小二乘法与 sklearn 对比。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def main():
    # 固定种子，让每次运行产生相同的数据，方便核对。
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 50)
    y_true = 2 * x + 5
    # normal 的 scale 是标准差；题目给的是方差 2。
    noise = rng.normal(loc=0, scale=np.sqrt(2), size=50)
    y_noisy = y_true + noise

    # 手写一元线性回归最小二乘公式，拟合的是含噪声样本。
    k = np.sum((x - x.mean()) * (y_noisy - y_noisy.mean())) / np.sum((x - x.mean()) ** 2)
    b = y_noisy.mean() - k * x.mean()

    # sklearn 输入 X 必须是二维：(样本数, 特征数)。
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y_noisy)
    print("L0: y = 2x + 5")
    print(f"L1 (manual): y = {k:.6f}x + {b:.6f}")
    print(f"L2 (sklearn): y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}")
    difference = np.max(np.abs(k * x + b - model.predict(x.reshape(-1, 1))))
    print(f"两种方法预测值的最大绝对差: {difference:.3e}")

    grid = np.linspace(0, 10, 200)
    plt.figure(figsize=(9, 6))
    plt.scatter(x, y_true, label="X0: true points", facecolors="none", edgecolors="green")
    plt.scatter(x, y_noisy, label="Xn: noisy samples", color="gray", alpha=0.7)
    plt.plot(grid, 2 * grid + 5, "g-", label="L0: y=2x+5")
    plt.plot(grid, k * grid + b, color="royalblue", linewidth=3, label="L1: manual OLS")
    plt.plot(grid, model.predict(grid.reshape(-1, 1)), "r--", label="L2: sklearn")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Task 1: linear regression with Gaussian noise")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    plt.savefig(output / "task1_line.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
