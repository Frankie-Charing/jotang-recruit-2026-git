"""Task 5 拓展题：让 CLIP 配对图片和文字（仅推理）。

依赖：pip install ftfy regex tqdm
      pip install git+https://github.com/openai/CLIP.git
首次运行会下载公开预训练权重（默认 RN50）。

数据：准备 4 张 224x224 左右的图片，分别命名 img0.jpg..img3.jpg，放在本目录下，
并在下方 IMAGES 与 CAPTIONS 中填写对应的英文描述。
运行：python clip_demo.py
"""

import os

import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

IMAGES = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
CAPTIONS = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a red car",
    "a photo of a beach",
]
# 故意加入一句“内容相近但关键信息错误”的描述，观察 CLIP 是否被误导
HARD_NEGATIVE = "a photo of a black cat"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)

    images = [preprocess(Image.open(os.path.join(HERE, p))).unsqueeze(0).to(device)
              for p in IMAGES]
    image_input = torch.cat(images, dim=0)
    texts = clip.tokenize(CAPTIONS + [HARD_NEGATIVE]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(texts)

    # L2 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print("image_features 形状：", tuple(image_features.shape))
    print("text_features 形状：", tuple(text_features.shape))

    # 余弦相似度 = 归一化后的点积
    sim = (image_features @ text_features.T).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(sim, cmap="viridis")
    ax.set_xticks(range(len(CAPTIONS) + 1))
    ax.set_xticklabels(CAPTIONS + ["(hard neg)"], rotation=45, ha="right")
    ax.set_yticks(range(len(IMAGES)))
    ax.set_yticklabels([f"img{i}" for i in range(len(IMAGES))])
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    color="white" if sim[i, j] < sim.max() else "black", fontsize=8)
    plt.colorbar(im)
    plt.title("image-text cosine similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "clip_similarity.png"), dpi=130)
    plt.close()

    for i in range(len(IMAGES)):
        best = int(np.argmax(sim[i]))
        print(f"{IMAGES[i]} 最接近：{CAPTIONS[best] if best < len(CAPTIONS) else HARD_NEGATIVE}"
              f"（相似度 {sim[i, best]:.3f}）")
    print("相似度热力图已保存到", FIG)


if __name__ == "__main__":
    main()
