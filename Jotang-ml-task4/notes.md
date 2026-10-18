# Task 4（拓展题）：All you need is attention?! —— Transformer 阅读笔记

## 一、我用了哪些资料、怎么读的

- 原论文：《Attention Is All You Need》（Vaswani et al., NeurIPS 2017）。
- 辅助资料：The Illustrated Transformer（Jay Alammar）、若干中文讲解博客、官方 PyTorch `nn.Transformer` 文档。
- 阅读顺序：先看标题、摘要、引言和 Figure 1 建立整体印象 → 精读第 3 节 Model Architecture → 再看实验与结论。

**读不懂的地方与后来怎么理解的：**

- 一开始分不清 Q/K/V 到底是谁。后来结合检索类比：Q 是“我在找什么”，K 是“每个位置能提供什么”，V 是“真正要取的内容”。
- Positional Encoding 的公式为什么用 sin/cos？后来理解它是给每个位置一个确定、可外推的“位置指纹”，且不增加可学习参数。

**目前仍存在的疑问：** 为什么原论文用加法而不是拼接的方式把位置编码与词嵌入结合，以及固定位置编码相比可学习位置编码在不同任务上的优劣，还没有完全想清楚。

## 二、结构图（参考 Figure 1 重画）

```
                     Encoder × N                          Decoder × N
             ┌──────────────────────┐            ┌──────────────────────┐
   Inputs ──▶│ Embedding            │    Outputs─▶│ Embedding            │
   (shifted) │  + Positional Enc.   │            │  + Positional Enc.   │
             │   │                  │            │   │                  │
             │   ▼                  │            │   ▼                  │
             │ Multi-Head Attention │            │ Masked M-H Attention │
             │   │  + Add & Norm    │            │   │  + Add & Norm    │
             │   ▼                  │            │   ▼                  │
             │ Feed-Forward         │            │ Cross M-H Attention  │
             │   │  + Add & Norm    │            │   │  + Add & Norm    │
             │   ▼                  │            │   ▼                  │
             └──────────────────────┘            │ Feed-Forward         │
                       │                         │   │  + Add & Norm    │
                       ▼                         │   ▼                  │
                    (输出给 Decoder)              └──────────────────────┘
                                                          │
                                                     Linear + Softmax
```

- **Encoder**：多头自注意力 + 前馈网络，各带残差连接与 LayerNorm，重复 N 次，负责“理解输入”。
- **Decoder**：多一层 **Masked 自注意力**（看已生成的词）+ **Cross-Attention**（看 Encoder 输出）+ 前馈网络，负责“生成输出”。

## 三、核心机制

- **Scaled Dot-Product Attention**：`Attention(Q,K,V) = softmax(QKᵀ/√d_k)V`。除以 `√d_k` 是为了防止点积数值过大、softmax 进入梯度很小的饱和区。
- **Multi-Head Attention**：把 Q/K/V 投影到 h 个子空间分别做注意力，再拼接。让模型在不同表示子空间关注不同信息。
- **Position-wise FFN**：对每个位置独立做两层线性变换（中间 ReLU），相当于在注意力之后对每个位置做一次特征变换。
- **Positional Encoding**：用 sin/cos 函数给每个位置一个唯一向量，弥补“没有循环结构、不知道顺序”的缺陷。

## 四、回答问题

**1. 这篇论文最初处理的是什么任务？为什么不用 RNN/CNN？**

机器翻译（英德、英法）。不用 RNN 是因为 RNN 必须按顺序一步步算，难以并行，且长距离依赖容易丢失；不用 CNN 是因为卷积的感受野有限，要堆很多层才能建立长距离联系。Transformer 用注意力一步就能让任意两个位置直接交互，既好并行，又能建模长依赖。

**2. Self-Attention 中的 "Self" 是什么意思？和普通 Attention 的区别？**

“Self”指 Q、K、V **都来自同一个序列**，即序列内部每个词互相“看”其它词。普通 Attention 的 Q 和 K/V 往往来自**两个不同序列**（例如解码时 Q 来自 decoder、K/V 来自 encoder，这就是 Transformer Decoder 里的 cross-attention）。

**3. Q、K、V 分别来自哪里？为什么要经过不同的线性变换？**

Q、K、V 都是由同一输入（自注意力时）分别乘三个不同的权重矩阵 `W_Q, W_K, W_V` 得到。三个矩阵不同，是为了让模型学会“查询”“键”“值”三种不同角色，从而有更大的表达能力，而不是把同一个向量机械地重复使用。

**4. Attention 结果为什么要经过 softmax？每一行权重表示什么？**

softmax 把“查询与每个键的相似度分数”变成**和为 1 的概率分布**，作为加权系数。权重矩阵的**每一行**表示：某个查询位置对**所有键位置**的注意力分配（该行之和为 1），数值越大表示越“关注”对应位置。

**5. Multi-Head Attention 为什么要分成多个头？只是重复吗？**

不是简单重复。不同头在不同子空间里学习不同的关注模式（例如一个头关注语法、另一个头关注指代关系）。最后拼接起来，模型就能从多个角度理解序列，表达力更强。

**6. Transformer 没有循环结构，怎样知道词语先后顺序？**

靠 **Positional Encoding**：给每个位置叠加一个由 sin/cos 生成的、与位置相关的向量，这样即使自注意力本身对位置“一视同仁”，输入里也携带了位置信息。

**7. Decoder 中为什么需要 Masked Attention？**

因为 Decoder 是自回归地逐词生成，生成第 t 个词时**只能看到它前面的词**。Mask 把未来位置的分数置为 `-∞`，softmax 后权重为 0，从而防止“作弊”偷看后面的答案，保证训练与推理一致。

**8. 残差连接、LayerNorm、FFN 分别在哪？作用？**

- **残差连接**：每个子层（注意力、FFN）的输入直接加到输出上（`x + Sublayer(x)`），缓解深层网络的梯度消失，让训练更稳。
- **Layer Normalization**：对每个样本的每个位置做归一化，稳定训练、加速收敛。
- **FFN**：在每个注意力子层之后，对每个位置独立做两层线性变换，提供额外的非线性表达能力。

## 五、拓展（代码）

`attention.py` 手写了：

1. 单头 Scaled Dot-Product Attention：输出 attention 分数与权重矩阵，并画热力图（`figs/attention_heatmap.png`）；
2. 多头注意力（`figs/multihead_heatmaps.png`），主体计算未调用 `nn.MultiheadAttention`。

## 六、其它拓展问题的简要笔记

- **Transformer 用于其它模态**：ViT 把图像切成 patch，每个 patch 当作一个“词”，通过线性映射变成 embedding 后送入 Encoder；Attention 在这里让不同 patch 之间互相交流信息。与原论文的主要区别是：输入从词向量变成图像 patch 向量，且只用了 Encoder（分类任务）。
- **Encoder-only 代表**：BERT——双向看上下文，训练目标是掩码语言模型/下一句预测，适合理解类任务。
- **Decoder-only 代表**：GPT——只看上文，训练目标是自回归预测下一个词，适合生成类任务。
- **为什么不同任务选不同结构**：理解任务需要双向上下文，用 Encoder；生成任务需要逐个产出、不能看未来，用 Decoder；翻译等“输入到输出”任务则用 Encoder-Decoder。
