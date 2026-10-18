"""Task 2 推理脚本：输入一张本地图片，输出猫/狗、预测概率，并保存预处理后的图片。

用法：python predict.py <图片路径>
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

from cnn_train import CNN

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["cat", "dog"]


def preprocess(path, size=64):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0), img


def main():
    if len(sys.argv) < 2:
        print("用法：python predict.py <图片路径>")
        return
    path = sys.argv[1]
    model = CNN()
    model.load_state_dict(torch.load(os.path.join(HERE, "catdog.pt"),
                                     weights_only=True))
    model.eval()

    x, processed = preprocess(path)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0]
    pred = int(prob.argmax())
    out_path = os.path.join(HERE, "processed_input.png")
    processed.save(out_path)

    print(f"预测类别：{CLASS_NAMES[pred]}（{'猫' if pred == 0 else '狗'}）")
    print(f"概率：cat={prob[0]:.4f}, dog={prob[1]:.4f}")
    print(f"预处理后图片已保存：{out_path}")


if __name__ == "__main__":
    main()
