"""Task 2 实践 1：手写二维卷积，理解卷积核、padding 与输出尺寸。

核心卷积过程只使用 Python/NumPy，不调用 cv2.filter2D、scipy.signal.convolve2d、
torch.nn.Conv2d 等现成实现。
运行：python convolution.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)


def make_sample(path):
    """生成一张带明显边缘/形状的测试图（RGB），方便观察模糊、锐化与边缘检测。"""
    H = W = 256
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        img[y, :, 0] = int(30 + 80 * y / H)
        img[y, :, 1] = int(20 + 60 * y / H)
        img[y, :, 2] = int(10 + 100 * y / H)
    img[40:120, 40:180] = [255, 255, 255]          # 白色矩形
    yy, xx = np.ogrid[:H, :W]
    cx, cy, r = 170, 170, 45
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = [0, 0, 0]  # 黑色圆
    Image.fromarray(img).save(path)
    return img


def conv2d(image, kernel, padding=0, stride=1):
    """手写二维卷积。支持灰度 (H, W) 与 RGB (H, W, C)。

    padding 为在四周填充的像素圈数（用 0 填充），stride 为步长。
    """
    image = np.asarray(image)
    is_rgb = image.ndim == 3
    if is_rgb:
        H, W, C = image.shape
        out = np.stack([conv2d(image[:, :, c], kernel, padding, stride)
                        for c in range(C)], axis=-1)
        return out

    H, W = image.shape
    kh, kw = kernel.shape
    if padding:
        image = np.pad(image, ((padding, padding), (padding, padding)))
    Hp, Wp = image.shape
    out_h = (Hp - kh) // stride + 1
    out_w = (Wp - kw) // stride + 1
    result = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i * stride:i * stride + kh, j * stride:j * stride + kw]
            result[i, j] = np.sum(patch * kernel)
    return result


def to_uint8(arr):
    """把卷积结果裁剪到 [0, 255] 并转成 uint8，避免保存时溢出。"""
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_compare(images, titles, filename, cmap=None):
    n = len(images)
    fig, ax = plt.subplots(1, n, figsize=(3 * n, 3.4))
    for a, im, t in zip(ax, images, titles):
        a.imshow(im, cmap=cmap)
        a.set_title(t, fontsize=9)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, filename), dpi=130)
    plt.close()


def main():
    sample_path = os.path.join(HERE, "sample.png")
    rgb = make_sample(sample_path)

    # 读取图片 -> NumPy 数组，观察其表示
    arr = np.asarray(Image.open(sample_path))
    print("图像 NumPy 数组：")
    print("  shape =", arr.shape, "（H, W, C，即高度、宽度、通道）")
    print("  dtype =", arr.dtype)
    print("  min =", arr.min(), "max =", arr.max())
    print("  像素 (100, 100) 的 RGB =", arr[100, 100].tolist())
    print("  HWC（NumPy/Pillow 常用）与 CHW（PyTorch 常用）的区别：")
    print("    维度的排列顺序不同：HWC=(高,宽,通道)，CHW=(通道,高,宽)。")
    print("    PyTorch 张量多为 CHW，送入模型前常需 transpose((2,0,1))。\n")

    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] +
            0.114 * arr[:, :, 2]).astype(np.float64)

    # 四种卷积核
    mean3 = np.ones((3, 3)) / 9.0
    gauss5 = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]], dtype=np.float64) / 256.0
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    blurred_mean = conv2d(gray, mean3, padding=1)
    blurred_gauss = conv2d(gray, gauss5, padding=2)
    sharpened = conv2d(gray, sharpen, padding=1)
    gx = conv2d(gray, sobel_x, padding=1)
    gy = conv2d(gray, sobel_y, padding=1)
    edge = np.sqrt(gx ** 2 + gy ** 2)

    save_compare([to_uint8(gray), to_uint8(blurred_mean), to_uint8(blurred_gauss),
                  to_uint8(sharpened), to_uint8(edge)],
                 ["original", "mean blur 3x3", "gaussian blur 5x5",
                  "sharpen", "sobel edge"],
                 "conv_results.png", cmap="gray")
    print("已保存卷积结果对比图：figs/conv_results.png\n")

    # valid vs same padding
    h, w = gray.shape
    valid = conv2d(gray, mean3, padding=0)
    same = conv2d(gray, mean3, padding=1)
    print(f"输入尺寸 {h}x{w}，3x3 卷积核、stride=1：")
    print(f"  valid（padding=0）输出尺寸：{valid.shape}")
    print(f"  same（padding=1）输出尺寸：{same.shape}")
    print("  关系：out = (in - k + 2p) / stride + 1；")
    print("  padding=0 时 out = in - k + 1，padding=(k-1)/2 时 out = in。\n")

    # 保存一张彩色图（RGB 每个通道分别卷积）
    rgb_sharp = to_uint8(conv2d(arr.astype(np.float64), sharpen, padding=1))
    save_compare([arr, rgb_sharp], ["original RGB", "sharpen RGB"],
                 "rgb_sharpen.png")
    print("完成。")


if __name__ == "__main__":
    main()
