"""Task 4 拓展题：手写最小 Scaled Dot-Product Attention 与 Multi-Head Attention。

主体计算不使用 nn.MultiheadAttention，仅用基础的矩阵运算。
运行：python attention.py
输出：figs/attention_heatmap.png, figs/multihead_heatmaps.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)


def softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V):
    """单头注意力。Q,K,V 形状 (L, d_k)。"""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / (d_k ** 0.5)  # (L, L)
    weights = softmax(scores, dim=-1)                 # 每一行是概率分布
    out = weights @ V
    return out, scores, weights


def multi_head_attention(Q, K, V, n_heads):
    """多头注意力。Q,K,V 形状 (L, d_model)，d_model 需被 n_heads 整除。"""
    d_model = Q.shape[-1]
    d_head = d_model // n_heads
    outs = []
    scores_all = []
    for h in range(n_heads):
        q = Q[:, h * d_head:(h + 1) * d_head]
        k = K[:, h * d_head:(h + 1) * d_head]
        v = V[:, h * d_head:(h + 1) * d_head]
        o, s, _ = scaled_dot_product_attention(q, k, v)
        outs.append(o)
        scores_all.append(s)
    return torch.cat(outs, dim=-1), scores_all


def main():
    torch.manual_seed(0)
    L, d_k = 6, 4
    Q = torch.randn(L, d_k)
    K = torch.randn(L, d_k)
    V = torch.randn(L, d_k)

    out, scores, weights = scaled_dot_product_attention(Q, K, V)
    print("单头注意力：")
    print("  attention scores 形状:", tuple(scores.shape))
    print("  attention weights 形状:", tuple(weights.shape))
    print("  每行权重和为 1：", torch.allclose(weights.sum(1), torch.ones(L)))

    plt.figure(figsize=(5, 4))
    im = plt.imshow(weights.numpy(), cmap="viridis")
    plt.colorbar(im)
    plt.title("single-head attention weights")
    plt.xlabel("key position")
    plt.ylabel("query position")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "attention_heatmap.png"), dpi=120)
    plt.close()

    # 多头
    d_model, n_heads = 8, 2
    Q2 = torch.randn(L, d_model)
    K2 = torch.randn(L, d_model)
    V2 = torch.randn(L, d_model)
    mh_out, mh_scores = multi_head_attention(Q2, K2, V2, n_heads)
    print("\n多头注意力（d_model=8, heads=2）：")
    print("  输出形状:", tuple(mh_out.shape))

    fig, ax = plt.subplots(1, n_heads, figsize=(10, 4))
    for h in range(n_heads):
        im = ax[h].imshow(mh_scores[h].numpy(), cmap="viridis")
        ax[h].set_title(f"head {h}")
        fig.colorbar(im, ax=ax[h])
    fig.suptitle("multi-head attention weights")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "multihead_heatmaps.png"), dpi=120)
    plt.close()
    print("图表已保存到", FIG)


if __name__ == "__main__":
    main()
