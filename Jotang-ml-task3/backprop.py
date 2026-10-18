"""Task 3：手写反向传播，并与 PyTorch autograd 对拍。

实践 1：简单计算链 f(x,y) = (x*y + x^2)^2
实践 2：两层网络 X -> Linear -> ReLU -> Linear -> Sigmoid 的梯度
运行：python backprop.py
"""

import numpy as np
import torch


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(z, 0.0)


def practice1():
    print("=" * 60)
    print("实践 1：简单计算链 f(x,y) = (x*y + x^2)^2")
    x, y = 2.0, 3.0

    # 分解计算图：u = x*y, v = x^2, w = u+v, f = w^2
    u = x * y
    v = x ** 2
    w = u + v
    f = w ** 2
    print(f"前向：u={u}, v={v}, w={w}, f={f}")

    # NumPy 手算梯度（从后向前，链式法则）
    df_dw = 2 * w
    dw_du, dw_dv = 1.0, 1.0
    du_dx, du_dy = y, x
    dv_dx = 2 * x
    df_dx = df_dw * dw_du * du_dx + df_dw * dw_dv * dv_dx
    df_dy = df_dw * dw_du * du_dy
    print(f"NumPy 梯度：df/dx={df_dx}, df/dy={df_dy}")

    # PyTorch autograd
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    ft = (xt * yt + xt ** 2) ** 2
    ft.backward()
    print(f"autograd 梯度：df/dx={xt.grad.item()}, df/dy={yt.grad.item()}")
    print(f"最大绝对误差：dx={abs(df_dx - xt.grad.item()):.3e}, "
          f"dy={abs(df_dy - yt.grad.item()):.3e}")
    print()


def practice2(batch=1):
    print("=" * 60)
    print(f"实践 2：两层网络梯度（batch={batch}）")
    in_dim, hidden = 2, 3
    rng = np.random.default_rng(0)
    X = rng.standard_normal((batch, in_dim)).astype(np.float64)
    y = rng.integers(0, 2, size=(batch, 1)).astype(np.float64)
    W1 = rng.standard_normal((in_dim, hidden)).astype(np.float64)
    b1 = rng.standard_normal((hidden,)).astype(np.float64)
    W2 = rng.standard_normal((hidden, 1)).astype(np.float64)
    b2 = rng.standard_normal((1,)).astype(np.float64)

    # ---- NumPy 前向 ----
    z1 = X @ W1 + b1            # (B, H)
    a1 = relu(z1)               # (B, H)
    z2 = a1 @ W2 + b2           # (B, 1)
    p = sigmoid(z2)             # (B, 1)
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    # ---- NumPy 反向 ----
    dL_dz2 = (p - y) / batch            # (B, 1)，BCE+Sigmoid 合并后
    dL_dW2 = a1.T @ dL_dz2              # (H, 1)
    dL_db2 = dL_dz2.sum(axis=0)         # (1,)
    dL_da1 = dL_dz2 @ W2.T              # (B, H)
    dL_dz1 = dL_da1 * (z1 > 0)          # ReLU 导数
    dL_dW1 = X.T @ dL_dz1               # (in, H)
    dL_db1 = dL_dz1.sum(axis=0)         # (H,)

    np_grads = {"W1": dL_dW1, "b1": dL_db1, "W2": dL_dW2, "b2": dL_db2}
    print("NumPy 梯度形状：",
          {k: v.shape for k, v in np_grads.items()})

    # ---- PyTorch 前向 + autograd ----
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    W1t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
    b1t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    W2t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
    b2t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

    z1t = Xt @ W1t + b1t
    a1t = torch.relu(z1t)
    z2t = a1t @ W2t + b2t
    loss_t = torch.nn.functional.binary_cross_entropy(
        torch.sigmoid(z2t), yt)
    loss_t.backward()

    torch_grads = {"W1": W1t.grad.numpy(), "b1": b1t.grad.numpy(),
                   "W2": W2t.grad.numpy(), "b2": b2t.grad.numpy()}

    print(f"损失：NumPy={loss:.6f}, PyTorch={loss_t.item():.6f}")
    ok = True
    for k in ["W1", "b1", "W2", "b2"]:
        err = np.max(np.abs(np_grads[k] - torch_grads[k]))
        print(f"  grad {k}: NumPy 形状={np_grads[k].shape}, "
              f"torch 形状={torch_grads[k].shape}, 最大绝对误差={err:.3e}")
        if err > 1e-8:
            ok = False
    print("结论：", "一致" if ok else "存在差异")
    print()


if __name__ == "__main__":
    practice1()
    practice2(batch=1)
    practice2(batch=4)
