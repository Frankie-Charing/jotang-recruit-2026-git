"""Task 1：用 PyTorch 在 make_moons 上训练一个 MLP。

包含：数据可视化、训练/验证/测试集划分、模型保存/加载、
loss/accuracy 曲线、二维决策边界、对照实验、混淆矩阵与错误样本分析。
运行：python train.py
输出：模型保存到 model.pt，图表保存到 figs/，对照实验表保存到 experiments.csv。
"""

import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)


def make_data(n_samples=1000, noise=0.2):
    """生成 make_moons 二分类数据，固定随机种子以便复现。"""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=SEED)
    return X.astype(np.float32), y.astype(np.int64)


def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(name)


class MLP(nn.Module):
    """简单多层感知机：in_dim -> [hidden]*n_hidden -> out_dim。"""

    def __init__(self, in_dim=2, hidden=32, out_dim=2, n_hidden=1, activation="relu"):
        super().__init__()
        dims = [in_dim] + [hidden] * n_hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(get_activation(activation))
        self.net = nn.Sequential(*layers)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.net(x)


def get_optimizer(name, params, lr):
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "sgd":
        return optim.SGD(params, lr=lr)
    raise ValueError(name)


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


def train_model(model, train_loader, val_loader, opt_name, lr, epochs=400, verbose=False):
    criterion = nn.CrossEntropyLoss()
    opt = get_optimizer(opt_name, model.parameters(), lr)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        tl, ta = evaluate(model, train_loader)
        vl, va = evaluate(model, val_loader)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
    if verbose:
        print(f"  final: train_acc={ta:.4f} val_acc={va:.4f} "
              f"val_loss={vl:.4f} params={model.n_params}")
    return history


def decision_boundary(model, X, y, title, path, xx_range=(-1.5, 2.5), yy_range=(-1.0, 1.5)):
    model.eval()
    xs = np.linspace(*xx_range, 300)
    ys = np.linspace(*yy_range, 300)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    with torch.no_grad():
        zz = model(torch.from_numpy(grid)).argmax(1).numpy().reshape(xx.shape)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, alpha=0.4, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", s=12, label="class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", s=12, label="class 1")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_curves(history, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    X, y = make_data(1000, noise=0.2)

    # 1. 可视化原始数据，并按训练/验证/测试集划分
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
    print(f"数据划分：train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    plt.figure(figsize=(5, 4))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", s=10, label="class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", s=10, label="class 1")
    plt.title("make_moons raw data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "raw_data.png"), dpi=120)
    plt.close()

    X_tr_t = torch.from_numpy(X_tr)
    X_va_t = torch.from_numpy(X_va)
    X_te_t = torch.from_numpy(X_te)
    y_tr_t = torch.from_numpy(y_tr)
    y_va_t = torch.from_numpy(y_va)
    y_te_t = torch.from_numpy(y_te)
    train_loader = [(X_tr_t, y_tr_t)]
    val_loader = [(X_va_t, y_va_t)]
    test_loader = [(X_te_t, y_te_t)]

    # 2. 训练一个基础 MLP
    base_cfg = dict(hidden=32, n_hidden=1, activation="relu",
                    opt="adam", lr=0.01, epochs=400)
    model = MLP(in_dim=2, hidden=base_cfg["hidden"], out_dim=2,
                n_hidden=base_cfg["n_hidden"], activation=base_cfg["activation"])
    print("训练基础模型：", base_cfg)
    t0 = time.time()
    hist = train_model(model, train_loader, val_loader,
                       base_cfg["opt"], base_cfg["lr"], base_cfg["epochs"])
    print(f"  耗时 {time.time()-t0:.1f}s")
    plot_curves(hist, "base MLP curves", os.path.join(FIG, "base_curves.png"))
    decision_boundary(model, X_te, y_te, "base decision boundary",
                      os.path.join(FIG, "base_boundary.png"))

    # 3. 保存 / 加载模型，并做测试
    torch.save(model.state_dict(), os.path.join(HERE, "model.pt"))
    loaded = MLP(in_dim=2, hidden=base_cfg["hidden"], out_dim=2,
                 n_hidden=base_cfg["n_hidden"], activation=base_cfg["activation"])
    loaded.load_state_dict(torch.load(os.path.join(HERE, "model.pt")))
    test_loss, test_acc = evaluate(loaded, test_loader)
    print(f"测试集（加载后模型）：loss={test_loss:.4f} acc={test_acc:.4f}")

    # 4. 混淆矩阵
    loaded.eval()
    with torch.no_grad():
        logits = loaded(X_te_t)
    pred = logits.argmax(1).numpy()
    cm = confusion_matrix(y_te, pred)
    print("混淆矩阵：\n", cm)

    # 5. 对照实验（一次只改一个变量）
    experiments = [
        ("base", dict(hidden=32, n_hidden=1, activation="relu", opt="adam", lr=0.01)),
        ("hidden=8", dict(hidden=8, n_hidden=1, activation="relu", opt="adam", lr=0.01)),
        ("hidden=64", dict(hidden=64, n_hidden=1, activation="relu", opt="adam", lr=0.01)),
        ("n_hidden=2", dict(hidden=32, n_hidden=2, activation="relu", opt="adam", lr=0.01)),
        ("activation=tanh", dict(hidden=32, n_hidden=1, activation="tanh", opt="adam", lr=0.01)),
        ("lr=0.1", dict(hidden=32, n_hidden=1, activation="relu", opt="adam", lr=0.1)),
        ("optimizer=sgd", dict(hidden=32, n_hidden=1, activation="relu", opt="sgd", lr=0.1)),
    ]
    rows = []
    for name, cfg in experiments:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        m = MLP(in_dim=2, hidden=cfg["hidden"], out_dim=2,
                n_hidden=cfg["n_hidden"], activation=cfg["activation"])
        t0 = time.time()
        h = train_model(m, train_loader, val_loader, cfg["opt"], cfg["lr"],
                        base_cfg["epochs"])
        dt = time.time() - t0
        vl, va = h["val_loss"][-1], h["val_acc"][-1]
        print(f"{name:16s} val_acc={va:.4f} val_loss={vl:.4f} "
              f"params={m.n_params} time={dt:.1f}s")
        rows.append([name, cfg["hidden"], cfg["n_hidden"], cfg["activation"],
                     cfg["opt"], cfg["lr"], round(va, 4), round(vl, 4),
                     m.n_params, round(dt, 1)])
        decision_boundary(m, X_te, y_te, f"boundary: {name}",
                          os.path.join(FIG, f"boundary_{name.replace('=', '_')}.png"))
    with open(os.path.join(HERE, "experiments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "hidden", "n_hidden", "activation", "optimizer",
                    "lr", "val_acc", "val_loss", "params", "time_s"])
        w.writerows(rows)

    # 6. 错误样本分析
    wrong = np.where(pred != y_te)[0]
    print(f"测试集错误样本数：{len(wrong)} / {len(y_te)}")
    plt.figure(figsize=(6, 5))
    xs = np.linspace(-1.5, 2.5, 300)
    ys = np.linspace(-1.0, 1.5, 300)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    with torch.no_grad():
        zz = loaded(torch.from_numpy(grid)).argmax(1).numpy().reshape(xx.shape)
    plt.contourf(xx, yy, zz, alpha=0.4, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
    plt.scatter(X_te[y_te == 0, 0], X_te[y_te == 0, 1], c="blue", s=10, label="class 0")
    plt.scatter(X_te[y_te == 1, 0], X_te[y_te == 1, 1], c="red", s=10, label="class 1")
    if len(wrong):
        plt.scatter(X_te[wrong, 0], X_te[wrong, 1], facecolors="none",
                    edgecolors="black", s=80, label="misclassified")
    plt.title("misclassified samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "misclassified.png"), dpi=120)
    plt.close()

    print("全部完成。图表与模型已保存。")


if __name__ == "__main__":
    main()
