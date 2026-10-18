"""Task 2 实践 2：猫狗分类 CNN 训练脚本。

真实数据：请把 Kaggle Dogs vs. Cats 数据放到 data/train/ 下（cat.0.jpg, dog.0.jpg ...），
然后运行：python cnn_train.py --data data/train

没有真实数据时可用 --demo 用程序生成的合成二分类图像快速验证整条流水线：
python cnn_train.py --demo
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageEnhance
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def make_demo_data(root, n_per_class=150):
    """生成合成二分类图像：类 0 = 带实心圆，类 1 = 带实心方块。仅用于流水线自检。"""
    os.makedirs(root, exist_ok=True)
    for c, (label, shape) in enumerate([("circle", "circle"), ("square", "square")]):
        for i in range(n_per_class):
            bg = tuple(np.random.randint(0, 120, size=3))
            img = Image.new("RGB", (64, 64), bg)
            px = np.random.randint(0, 64, size=2)
            size = np.random.randint(10, 28)
            overlay = Image.new("RGB", (64, 64), (0, 0, 0))
            from PIL import ImageDraw
            d = ImageDraw.Draw(overlay)
            fg = tuple(np.random.randint(150, 256, size=3))
            if shape == "circle":
                d.ellipse([px[0], px[1], px[0] + size, px[1] + size], fill=fg)
            else:
                d.rectangle([px[0], px[1], px[0] + size, px[1] + size], fill=fg)
            img = Image.blend(img, overlay, 1.0)
            img.save(os.path.join(root, f"{label}.{i}.jpg"))


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, size=64, augment=False):
        self.paths = paths
        self.labels = labels
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:  # 随机裁剪后放大回原尺寸
                w, h = img.size
                cw, ch = int(w * random.uniform(0.75, 1.0)), int(h * random.uniform(0.75, 1.0))
                x0 = random.randint(0, w - cw)
                y0 = random.randint(0, h - ch)
                img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h))
            if random.random() < 0.5:  # 颜色扰动
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        img = img.resize((self.size, self.size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(arr), self.labels[idx]


class CNN(nn.Module):
    def __init__(self, in_c=3, n_class=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_c, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_class),
        )
        self.first_conv = self.features[0]
        self.last_conv = self.features[-4]

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss_sum += criterion(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return loss_sum / total, correct / total


def train(model, train_loader, val_loader, epochs=15, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        tl, ta = evaluate(model, train_loader)
        vl, va = evaluate(model, val_loader)
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_acc"].append(ta)
        hist["val_acc"].append(va)
        print(f"  epoch {ep + 1:2d}: train_acc={ta:.4f} val_acc={va:.4f} "
              f"train_loss={tl:.4f} val_loss={vl:.4f}")
        if va > best_acc:
            best_acc, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(HERE, "catdog.pt"))
    return hist, best_acc


def plot_curves(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train")
    ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_structure(model, path):
    layers = ["input 3x64x64"]
    c, h, w = 3, 64, 64
    for m in model.features:
        if isinstance(m, nn.Conv2d):
            c = m.out_channels
            layers.append(f"conv {c}x{h}x{w}")
        elif isinstance(m, nn.MaxPool2d):
            h, w = h // 2, w // 2
            layers.append(f"pool {c}x{h}x{w}")
    layers.append("flatten")
    layers.append("fc 128")
    layers.append("fc 2")
    fig, ax = plt.subplots(figsize=(10, 3))
    for i, t in enumerate(layers):
        ax.add_patch(plt.Rectangle((i * 1.1, 0), 1, 0.6, fill=True,
                                   fc="steelblue" if "conv" in t else
                                      ("lightgreen" if "pool" in t else
                                       ("orange" if "fc" in t else "gray")),
                                   ec="black"))
        ax.text(i * 1.1 + 0.5, 0.3, t, ha="center", va="center", fontsize=7)
    ax.set_xlim(-0.2, len(layers) * 1.1)
    ax.set_ylim(-0.3, 0.9)
    ax.axis("off")
    ax.set_title("CNN structure")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def visualize_feature_maps(model, loader, path):
    model.eval()
    xb, yb = next(iter(loader))
    with torch.no_grad():
        f1 = model.first_conv(xb)           # 第一层卷积输出 (B, 16, H, W)
        mid = model.features[0:8](xb)       # 经过前两层 block 后的特征 (B, 32, H/4, W/4)
        f_last = model.last_conv(mid)       # 最后一层卷积输出 (B, 64, H/4, W/4)
    fig, ax = plt.subplots(2, 8, figsize=(14, 4))
    for i in range(8):
        ax[0, i].imshow(f1[0, i].numpy(), cmap="gray")
        ax[0, i].axis("off")
        ax[1, i].imshow(f_last[0, i].numpy(), cmap="gray")
        ax[1, i].axis("off")
    ax[0, 0].set_ylabel("first conv", fontsize=9)
    ax[1, 0].set_ylabel("last conv", fontsize=9)
    fig.suptitle("feature maps")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    if args.demo:
        demo_root = os.path.join(HERE, "demo_data")
        make_demo_data(demo_root)
        args.data = demo_root
        print("demo 模式：使用合成二分类图像验证流水线。")

    if not args.data or not os.path.isdir(args.data):
        print("请提供数据目录：python cnn_train.py --data data/train")
        return

    # 收集图片路径与标签（cat=0, dog=1）
    paths, labels = [], []
    for fn in sorted(os.listdir(args.data)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if fn.lower().startswith("cat") or fn.lower().startswith("circle"):
            labels.append(0)
        elif fn.lower().startswith("dog") or fn.lower().startswith("square"):
            labels.append(1)
        else:
            continue
        paths.append(os.path.join(args.data, fn))
    if len(paths) == 0:
        print("未找到图片。")
        return
    n_cat = sum(1 for l in labels if l == 0)
    n_dog = sum(1 for l in labels if l == 1)
    print(f"共 {len(paths)} 张图片：cat={n_cat}, dog={n_dog}")

    # 按类别分层划分训练/验证/测试
    idx = np.arange(len(paths))
    np.random.shuffle(idx)
    idx = idx.tolist()
    tr_idx, va_idx, te_idx = [], [], []
    for cls in (0, 1):
        cls_idx = [i for i in idx if labels[i] == cls]
        n_va = max(1, int(len(cls_idx) * 0.15))
        n_te = max(1, int(len(cls_idx) * 0.15))
        te_idx += cls_idx[:n_te]
        va_idx += cls_idx[n_te:n_te + n_va]
        tr_idx += cls_idx[n_te + n_va:]
    print(f"划分：train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    tr_ds = ImgDataset([paths[i] for i in tr_idx], [labels[i] for i in tr_idx], augment=True)
    va_ds = ImgDataset([paths[i] for i in va_idx], [labels[i] for i in va_idx], augment=False)
    te_ds = ImgDataset([paths[i] for i in te_idx], [labels[i] for i in te_idx], augment=False)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=32, shuffle=True)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=32)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=32)

    model = CNN()
    print(f"模型参数量：{sum(p.numel() for p in model.parameters())}")

    # 记录各层张量形状
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    print("各层张量形状：")
    for name, m in model.features.named_children():
        x = m(x)
        print(f"  {name}: {tuple(x.shape)}")

    hist, best_acc = train(model, tr_loader, va_loader, epochs=args.epochs)
    plot_curves(hist, os.path.join(FIG, "cnn_curves.png"))
    plot_structure(model, os.path.join(FIG, "cnn_structure.png"))
    visualize_feature_maps(model, te_loader, os.path.join(FIG, "feature_maps.png"))

    # 测试集 + 混淆矩阵
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds.append(model(xb).argmax(1))
            ys.append(yb)
    pred = torch.cat(preds).numpy()
    ytrue = torch.cat(ys).numpy()
    te_loss, te_acc = evaluate(model, te_loader)
    print(f"测试集：acc={te_acc:.4f} loss={te_loss:.4f}")
    print("混淆矩阵：\n", confusion_matrix(ytrue, pred))
    print(f"验证集最佳 acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
