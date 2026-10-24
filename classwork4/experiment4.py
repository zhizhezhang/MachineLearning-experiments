"""实验4：逻辑回归。直接运行本文件，结果保存至同目录的 results 文件夹。"""
from pathlib import Path
import argparse
import io
from contextlib import redirect_stdout

import numpy as np
import sklearn
import matplotlib
matplotlib.use("Agg")  # 保存图片，无需图形窗口
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
SEED = 42
OUT = Path(__file__).resolve().parent / "results"


def new_model():
    # 将标准化放在 Pipeline 内，每折只使用训练数据估计均值和标准差。
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=5000, random_state=SEED))


def experiment1():
    """1000个双月样本，60%训练、40%测试，比较不同噪声。"""
    print("实验1：make_moons 数据分类")
    noises = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
    scores = []
    for noise in noises:
        x, y = make_moons(n_samples=1000, noise=noise, random_state=SEED)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.4, random_state=SEED, stratify=y)
        model = new_model()
        model.fit(x_train, y_train)
        score = accuracy_score(y_test, model.predict(x_test))
        scores.append(score)
        print(f"噪声={noise:.1f}，测试集准确率={score:.2%}")

        if noise == 0.3:
            # 根据训练集拟合的模型绘制分类区域，黑线为 P(y=1)=0.5。
            xx, yy = np.meshgrid(
                np.linspace(x[:, 0].min()-0.5, x[:, 0].max()+0.5, 350),
                np.linspace(x[:, 1].min()-0.5, x[:, 1].max()+0.5, 350))
            p = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            p = p.reshape(xx.shape)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.contourf(xx, yy, p, levels=[0, 0.5, 1],
                        colors=["#dbeafe", "#ffedd5"], alpha=0.8)
            ax.contour(xx, yy, p, levels=[0.5], colors="black", linewidths=2)
            for label, color in [(0, "#2563eb"), (1, "#ea580c")]:
                tr, te = y_train == label, y_test == label
                ax.scatter(*x_train[tr].T, c=color, s=20, alpha=0.6,
                           label=f"训练集 类别{label}")
                ax.scatter(*x_test[te].T, c=color, s=30, marker="x",
                           label=f"测试集 类别{label}")
            ax.plot([], [], color="black", label="分类界线：P(y=1)=0.5")
            ax.set(xlabel="特征 x1", ylabel="特征 x2",
                   title=f"逻辑回归分类（样本数1000，噪声0.3，测试准确率{score:.2%}）")
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(OUT / "moons_boundary.png", dpi=180)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noises, scores, "o-", color="#2563eb", linewidth=2)
    for noise, score in zip(noises, scores):
        ax.annotate(f"{score:.2%}", (noise, score), xytext=(0, 10),
                    textcoords="offset points", ha="center")
    ax.set(xlabel="噪声标准差", ylabel="测试集准确率", ylim=(0.45, 1.0),
           title="逻辑回归预测准确率随噪声的变化（样本数1000）")
    ax.set_xticks(noises)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "noise_accuracy.png", dpi=180)
    plt.close(fig)
    np.savetxt(OUT / "noise_accuracy.csv", np.c_[noises, scores],
               delimiter=",", header="noise,test_accuracy", comments="", fmt="%.6f")
    print("说明：原始二维特征的逻辑回归分类界线是直线；双月数据本身为非线性结构。\n")


def evaluate_small_sample(x, y, names, title, filename):
    print(title)
    model = new_model()
    # 每次留下1个样本测试，用其余样本训练，共训练n次。
    loo_prob = cross_val_predict(model, x, y, cv=LeaveOneOut(),
                                 method="predict_proba")[:, 1]
    loo_pred = (loo_prob >= 0.5).astype(int)
    print(f"留一交叉验证准确率：{accuracy_score(y, loo_pred):.2%}")
    print("留一交叉验证混淆矩阵（行=真实类别，列=预测类别，顺序0、1）：")
    print(confusion_matrix(y, loo_pred, labels=[0, 1]))
    print(classification_report(y, loo_pred, labels=[0, 1], digits=4,
                                zero_division=0))
    # 评估结束后，使用全部数据拟合最终模型，以便预测未来输入。
    model.fit(x, y)
    fit_prob = model.predict_proba(x)[:, 1]
    fit_pred = model.predict(x)
    print(f"全样本训练准确率（仅反映拟合效果）：{accuracy_score(y, fit_pred):.2%}")
    print("序号  真实值  拟合预测  留一预测  留一P(y=1)")
    for i in range(len(y)):
        print(f"{i+1:>4}  {y[i]:>6}  {fit_pred[i]:>8}  {loo_pred[i]:>8}  {loo_prob[i]:>11.4f}")

    # 把标准化坐标中的参数换算为原始特征单位，方便写出回归方程。
    scaler, lr = model.steps[0][1], model.steps[1][1]
    coef = lr.coef_[0] / scaler.scale_
    intercept = lr.intercept_[0] - np.dot(coef, scaler.mean_)
    print("\n最终模型（默认L2正则化，C=1）：")
    print("z = " + f"{intercept:.6f}" + "".join(
        f" {value:+.6f}*{name}" for name, value in zip(names, coef)))
    print("P(y=1) = 1 / (1 + exp(-z))，概率 >= 0.5 时预测为1。")
    print("样本很少，交叉验证结果仅供本次实验比较，不能据此认定模型具有稳定泛化能力。\n")
    np.savetxt(OUT / f"{filename}.csv",
               np.c_[np.arange(1, len(y)+1), y, fit_pred, fit_prob, loo_pred, loo_prob],
               delimiter=",", comments="", fmt=["%d", "%d", "%d", "%.6f", "%d", "%.6f"],
               header="id,true,fit_prediction,fit_probability,loo_prediction,loo_probability")
    return model


def experiment2():
    # 每行依次为：年龄、VEGF、MVC、组织学分级、分期、是否转移。
    data = np.array([
        [59,2,43.4,2,1,0], [36,1,57.2,1,1,0],
        [61,2,190.0,2,1,0], [58,3,128.0,4,3,1],
        [55,3,80.0,3,4,1], [61,1,94.4,2,1,0],
        [38,1,76.0,1,1,0], [42,1,240.0,3,2,1],
        [50,1,74.0,1,1,0], [58,3,68.6,2,2,0],
        [68,3,132.8,4,2,0], [25,2,94.6,4,3,1],
        [52,1,56.0,1,1,0], [31,1,47.8,2,1,0],
        [36,3,31.6,3,1,1], [42,1,66.2,2,1,0]
    ], dtype=float)
    return evaluate_small_sample(data[:, :5], data[:, 5].astype(int),
        ["x1", "x2", "x3", "x4", "x5"],
        "实验2：肾细胞癌是否转移（0=无转移，1=有转移）", "kidney_predictions")


def experiment3():
    # 每行依次为PassengerId、Pclass、Sex、Age、SibSp、Parch、Fare、Embarked、Survived。
    data = np.array([
        [1,1,1,38,1,0,71.2,1,1], [2,1,1,35,1,0,53.1,3,1],
        [3,1,0,54,0,0,51.9,3,0], [4,3,1,16,0,0,79.2,1,1],
        [5,3,1,4,1,1,16.7,3,1], [6,1,1,58,0,0,26.6,3,1],
        [7,2,0,34,0,0,13,3,1], [8,1,0,28,0,0,35.5,3,1],
        [9,1,0,19,3,2,263,3,0], [10,1,1,49,1,0,76.7,1,1],
        [11,1,0,65,0,1,62,1,0], [12,1,0,45,1,0,83.4,3,0],
        [13,2,1,29,0,0,10.5,3,1], [14,3,0,25,0,0,7.65,3,0],
        [15,1,1,23,3,2,263,3,1], [16,1,0,46,1,0,61.2,3,0],
        [17,3,0,40,1,1,15.5,2,0], [18,1,0,23,0,1,63.4,1,1],
        [19,1,0,21,0,1,77.3,3,0], [20,1,0,47,0,0,52,3,0]
    ], dtype=float)
    # ID只是标识，不参与训练；按题目将Embarked视为有序数值。
    # Pclass及其他等级变量在本实验中也按有序数值使用。
    return evaluate_small_sample(data[:, 1:8], data[:, 8].astype(int),
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
        "实验3：泰坦尼克号生存预测（0=未获救，1=获救）", "titanic_predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验4：逻辑回归")
    parser.add_argument("--part", choices=["all", "1", "2", "3"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"NumPy {np.__version__}；Scikit-learn {sklearn.__version__}")
        print(f"random_state={SEED}\n")
        for number, run in [("1", experiment1), ("2", experiment2), ("3", experiment3)]:
            if args.part in ("all", number):
                run()
    result = buffer.getvalue()
    print(result)
    result_path = OUT / ("运行结果.txt" if args.part == "all" else f"第{args.part}题运行结果.txt")
    result_path.write_text(result, encoding="utf-8")
    print(f"结果已保存到：{OUT}")
