"""第4题：按文档表3训练，按表4测试。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    # 文档没有写单位；这里按这些数值通常对应的米和千克标图。
    X_train = np.array([0.86, 0.96, 1.12, 1.35, 1.55, 1.63, 1.71, 1.85]).reshape(-1, 1)
    y_train = np.array([12, 15, 20, 35, 48, 51, 59, 75])
    X_test = np.array([0.83, 1.08, 1.26, 1.51, 1.60, 1.67, 1.75]).reshape(-1, 1)
    y_test = np.array([11, 17, 27, 41, 50, 64, 66])

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(f"体重 = {model.coef_[0]:.6f} × 身高 + ({model.intercept_:.6f})")
    print("身高       真实体重       预测体重       误差(预测-真实)")
    for height, actual, pred in zip(X_test.ravel(), y_test, predicted):
        print(f"{height:.2f}       {actual:8.2f}       {pred:8.2f}       {pred - actual:8.2f}")
    print(f"测试集 MAE: {mean_absolute_error(y_test, predicted):.4f}")
    print(f"测试集 RMSE: {np.sqrt(mean_squared_error(y_test, predicted)):.4f}")
    print(f"测试集 R²: {r2_score(y_test, predicted):.6f}")

    grid = np.linspace(0.8, 1.9, 200).reshape(-1, 1)
    plt.figure(figsize=(9, 6))
    plt.scatter(X_train.ravel(), y_train, label="Training samples")
    plt.scatter(X_test.ravel(), y_test, color="green", marker="s", label="Test: actual")
    plt.scatter(X_test.ravel(), predicted, color="red", marker="x", s=70, label="Test: predicted")
    plt.plot(grid.ravel(), model.predict(grid), color="orange", label="Fitted line")
    plt.xlabel("Height (m, assumed)")
    plt.ylabel("Weight (kg, assumed)")
    plt.title("Task 4: weight prediction")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    plt.savefig(output / "task4_weight.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
