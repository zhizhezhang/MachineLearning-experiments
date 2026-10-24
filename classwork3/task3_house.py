"""第3题：按文档表1训练，按表2测试，不能混用。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    X_train = np.array([1710, 1362, 1786, 1719, 2198, 1562, 1689, 2090, 1774, 1077,
                        1040, 2324, 912, 1494, 1253, 854, 1004, 1296, 1114, 1339]).reshape(-1, 1)
    y_train = np.array([208500, 161500, 213500, 209000, 250310, 173000, 197000, 239000, 209900, 142600,
                        139500, 245000, 144000, 186119, 157000, 132000, 149000, 155190, 139000, 168200])
    X_test = np.array([1592, 1188, 1888, 1429, 940]).reshape(-1, 1)
    y_test = np.array([187120, 151099, 215315, 178927, 131432])

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(f"房价 = {model.coef_[0]:.6f} × 面积 + {model.intercept_:.6f}")
    print("面积(平方英尺)    真实房价(美元)    预测房价(美元)    误差(预测-真实)")
    for area, actual, pred in zip(X_test.ravel(), y_test, predicted):
        print(f"{area:12d}    {actual:14.2f}    {pred:14.2f}    {pred - actual:14.2f}")
    print(f"测试集 MAE: {mean_absolute_error(y_test, predicted):.2f} 美元")
    print(f"测试集 RMSE: {np.sqrt(mean_squared_error(y_test, predicted)):.2f} 美元")
    print(f"测试集 R²: {r2_score(y_test, predicted):.6f}")

    grid = np.linspace(800, 2400, 200).reshape(-1, 1)
    plt.figure(figsize=(9, 6))
    plt.scatter(X_train.ravel(), y_train, label="Training samples")
    plt.scatter(X_test.ravel(), y_test, color="green", marker="s", label="Test: actual")
    plt.scatter(X_test.ravel(), predicted, color="red", marker="x", s=70, label="Test: predicted")
    plt.plot(grid.ravel(), model.predict(grid), color="orange", label="Fitted line")
    plt.xlabel("Living area (square feet)")
    plt.ylabel("Price (USD)")
    plt.title("Task 3: house price prediction")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    plt.savefig(output / "task3_house.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
