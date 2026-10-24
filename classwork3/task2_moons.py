"""第2题：用两个坐标拟合 make_moons 的类别标签。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main():
    X, y = make_moons(n_samples=50, noise=0.3, random_state=42)
    # X 的两列是 X1、X2；y 是 0/1 标签。本题按要求做线性回归。
    model = LinearRegression()
    model.fit(X, y)
    fitted_y = model.predict(X)
    w1, w2 = model.coef_
    w0 = model.intercept_
    print(f"Y = {w1:.6f} * X1 + ({w2:.6f}) * X2 + ({w0:.6f})")
    print(f"训练样本 MSE: {mean_squared_error(y, fitted_y):.6f}")
    print("注意：fitted_y 是连续回归值，不是概率，也不一定在 [0, 1] 内。")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, color in [(0, "royalblue"), (1, "tomato")]:
        axes[0].scatter(X[y == label, 0], X[y == label, 1], color=color, label=f"Class {label}")
    axes[0].set(xlabel="X1", ylabel="X2", title="Two classes of make_moons")
    axes[0].legend()
    indices = np.arange(1, len(y) + 1)
    axes[1].scatter(indices, y, label="True y (0 or 1)", color="black")
    axes[1].scatter(indices, fitted_y, label="Fitted y", color="tomato", marker="x")
    axes[1].vlines(indices, y, fitted_y, color="gray", alpha=0.4, label="Residual")
    axes[1].set(xlabel="Sample index", ylabel="y", title="True y versus fitted y")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "task2_moons.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
