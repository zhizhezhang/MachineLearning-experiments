import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.filters import threshold_otsu
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================
# 1. 加载 MNIST 数据集
# =====================================================
print("正在加载 MNIST 数据集")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32) / 255.0     # 归一化到 [0,1]
y = mnist.target.astype(np.int64)

print("原始数据形状：", X.shape, y.shape)

# =====================================================
# 2. 划分训练集与测试集
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化（神经网络对尺度敏感）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# 3. 构建神经网络
#    输入层：784
#    隐层：2 层，分别 128、64 个神经元
#    输出层：10（由 MLPClassifier 自动生成，不包含在 hidden_layer_sizes 中）
# =====================================================
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    batch_size=128,
    learning_rate_init=1e-3,
    max_iter=50,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42,
    verbose=True
)

# =====================================================
# 4. 训练模型
# =====================================================
print("\n开始训练神经网络……")
mlp.fit(X_train, y_train)

# =====================================================
# 5. 在测试集上评估
# =====================================================
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nMNIST 测试集准确率：{acc:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred, digits=4))

# 绘制训练损失曲线
plt.figure(figsize=(6, 4))
plt.plot(mlp.loss_curve_, label='训练损失')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.title('神经网络训练损失曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# 6. 自写数字图片预处理函数
#    输入：图片路径（手机拍照或扫描均可）
#    输出：与 MNIST 风格一致的 28×28 灰度图像素向量
# =====================================================
def preprocess_handwritten_image(img_path, show=False):
    """
    将自写数字照片处理为 MNIST 风格：
    黑底白字、28×28、数字居中、占 20×20。
    """
    # 1. 读取灰度图
    img = Image.open(img_path).convert('L')
    img = np.array(img).astype(np.float32)

    # 2. 自动判断极性：让数字变成亮、背景变成暗
    #    先估计背景亮度：取四条边
    border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    bg_is_bright = border.mean() > 127   # 白底黑字 -> True

    # 3. Otsu 二值化
    try:
        thresh = threshold_otsu(img.astype(np.uint8))
    except Exception:
        thresh = 127

    if bg_is_bright:
        # 白底黑字：数字是暗的，所以取 < thresh
        binary = img < thresh
    else:
        # 黑底白字：数字是亮的，取 > thresh
        binary = img > thresh

    # 4. 兜底：如果前景像素太少或太多，翻转一次
    ratio = binary.mean()
    if ratio < 0.01 or ratio > 0.5:
        binary = ~binary

    # 5. 只保留最大连通区域，去掉噪点
    from scipy.ndimage import label, binary_fill_holes
    labeled, num = label(binary)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0                      # 忽略背景
        largest = sizes.argmax()
        binary = (labeled == largest)
    binary = binary_fill_holes(binary)   # 填补数字内部空洞

    # 6. 裁剪包围盒
    coords = np.argwhere(binary)
    if coords.size == 0:
        raise ValueError("未在图片中检测到数字，请检查图片背景与字迹对比度。")
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    digit_mask = binary[y0:y1, x0:x1]    # True = 数字

    # 7. 等比例缩放到 20×20
    h, w = digit_mask.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * 20 / h)))
    else:
        new_w = 20
        new_h = max(1, int(round(h * 20 / w)))

    digit_img = Image.fromarray(
        (digit_mask * 255).astype(np.uint8)   # True -> 255 白色
    ).resize((new_w, new_h), Image.LANCZOS)

    # 8. 居中贴到 28×28 黑色画布（黑底白字）
    canvas = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(digit_img, (paste_x, paste_y))

    arr = np.array(canvas).astype(np.float32) / 255.0

    if show:
        plt.figure(figsize=(3, 3))
        plt.imshow(arr, cmap='gray')
        plt.title('预处理后的 28×28 图像')
        plt.axis('off')
        plt.show()

    return arr.reshape(1, -1)


# =====================================================
# 7. 测试自写数字
# =====================================================
# 请将下面两个路径替换为你自己的图片路径
img_paths = ['digit2.jpg', 'digit3.jpg']

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for i, path in enumerate(img_paths):
    try:
        # 预处理
        x_input = preprocess_handwritten_image(path)
        # 标准化（使用与训练相同的 scaler）
        x_input_scaled = scaler.transform(x_input)

        # 预测
        pred = mlp.predict(x_input_scaled)[0]
        proba = mlp.predict_proba(x_input_scaled)[0]
        confidence = proba[pred]

        # 可视化
        axes[i].imshow(x_input.reshape(28, 28), cmap='gray')
        axes[i].set_title(f'预测：{pred}  (置信度 {confidence:.2%})')
        axes[i].axis('off')

        print(f"图片：{path}  ->  预测数字：{pred}，置信度：{confidence:.4f}")
        print("各类别概率：", np.round(proba, 4))

    except Exception as e:
        axes[i].set_title(f'图片 {path} 处理失败')
        axes[i].axis('off')
        print(f"处理 {path} 出错：{e}")

plt.tight_layout()
plt.show()