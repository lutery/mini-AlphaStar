## `gather` 这行代码的讲解

### 一句话结论

这是一个**按索引查表**操作。`scatter_index` 存的是"每个空间格子上有什么实体（编号）"，`gather` 拿这些编号去 `reduced_entity_embeddings` 里查出对应实体的嵌入向量，从而把"实体编号地图"变成"实体特征地图"。

---

### `torch.gather` 的本质

`torch.gather` 是在指定维度上，按照索引矩阵从输入张量中**挑选行（或列）**。它和 Python 的列表索引查表是同一个思路，只是批量、并行了。

用最简单的例子来建立直觉：

```python
# 有一个嵌入表：3 个实体，每个 4 维嵌入
embeddings = torch.tensor([
    [1,  2,  3,  4],    # 实体0: "空" (填充用的零向量)
    [5,  6,  7,  8],    # 实体1: 一个机枪兵
    [9, 10, 11, 12]     # 实体2: 一个水晶塔
])
# shape: [3, 4]

# 有一个索引表：2 个空间位置，每个位置想知道是哪个实体
index = torch.tensor([
    [1],    # 位置A: 是实体1（机枪兵）
    [2]     # 位置B: 是实体2（水晶塔）
])
# shape: [2, 1]

# gather 查表
result = embeddings.gather(0, index.expand(-1, 4))
# result[0] = embeddings[1] = [5,  6,  7,  8]   # 机枪兵的嵌入
# result[1] = embeddings[2] = [9, 10, 11, 12]   # 水晶塔的嵌入
```

`gather` 做的事就是：**对于 result 中每个位置 `[i, j]`，去 embedding 表的 `[index[i, j], j]` 位置取值**。

---

### 回到当前代码：逐步拆解

代码在第 99-116 行，先看完整上下文：

```python
# 1. 把 256 维嵌入降维到 32 维
reduced_entity_embeddings = F.relu(self.conv1(entity_embeddings.transpose(1, 2))).transpose(1, 2)
# shape: [B, entity_size, 32]

# 2. 用零向量（zero_bias）替换掉第 0 个位置的原实体
#    因为 scatter_map 中 0 表示"没有实体"
#    查表时查到 0，就会拿出这个全零向量，对结果无影响
batch_size = reduced_entity_embeddings.shape[0]
entity_size = reduced_entity_embeddings.shape[1]

device = next(self.parameters()).device
zero_bias = torch.zeros(batch_size, 1, 32, device=device)           # [B, 1, 32]
reduced_entity_embeddings = torch.cat([zero_bias,                    # 索引0 → 零向量
                                        reduced_entity_embeddings[:, 1:, :]], dim=1)
# shape: [B, entity_size+1, 32]
#        ↑ 多了一个零向量的索引0位置

# 3. 准备 scatter_index（索引矩阵）
#    scatter_index 来自 scatter_map，原始 shape = [B, 4, H, W]
#    4 是因为每个格子最多存 4 个实体的索引
scatter_index = scatter_index.reshape(batch_size, -1)                # [B, 4*H*W]
scatter_index = scatter_index.unsqueeze(-1).repeat(1, 1, 32)         # [B, 4*H*W, 32]
#                                                          ↑ 同一索引值在 32 个维度上全部复制

# 4. ★ 核心：按索引查表 ★
scatter_mid = reduced_entity_embeddings.gather(1, scatter_index.long())
# 参数:        dim=1（沿实体维查表）
#              index=scatter_index（空间位置→实体编号的映射）
```

---

### `gather` 这一步具体做了什么

画出张量形状对照：

```
                    reduced_entity_embeddings
                    ┌─────────────────────────┐
                    索引0: [0,  0,  0, ...,  0]  ← 零向量（空格子）
                    索引1: [a₁, a₂, a₃, ..., a₃₂]  ← 实体1的嵌入
                    索引2: [b₁, b₂, b₃, ..., b₃₂]  ← 实体2的嵌入
                    ...
                    索引N: [z₁, z₂, z₃, ..., z₃₂]  ← 实体N的嵌入
                    └─────────────────────────┘
                    shape: [B, entity_size+1, 32]
                               ↑ gather 沿这一维查


                    scatter_index
                    ┌──────────────────────────────────┐
                    [ 1, 1, 1, ..., 1,    # 空间位置(0,0,ch0): 实体1在这儿
                      0, 0, 0, ..., 0,    # 空间位置(0,1,ch0): 空格子
                      3, 3, 3, ..., 3,    # 空间位置(0,0,ch1): 实体3也在这儿
                      ... ]
                    └──────────────────────────────────┘
                    shape: [B, 4*H*W, 32]
                               ↑ 每个空间位置一行，32 个数字都一样（都是同一个实体编号）


                    gather 执行后 → scatter_mid
                    ┌──────────────────────────────────┐
                    [ a₁, a₂, a₃, ..., a₃₂,   # 实体1的嵌入
                      0,  0,  0,  ..., 0,     # 零向量（空格子）
                      d₁, d₂, d₃, ..., d₃₂,   # 实体3的嵌入
                      ... ]
                    └──────────────────────────────────┘
                    shape: [B, 4*H*W, 32]
```

然后继续处理：

```python
# 5. 把一维的空间位置列表 reshape 回二维地图
scatter_mid = scatter_mid.reshape(batch_size, 4, H, W, 32)
# shape: [B, 4, H, W, 32]

# 6. 同一个格子上可能有多达 4 个实体，把它们沿 scatter_volume 维求和合并
scatter_result = torch.sum(scatter_mid, dim=1)    # [B, H, W, 32]

# 7. 调整通道维到标准位置
scatter_result = scatter_result.permute(0, 3, 1, 2)   # [B, 32, H, W]
```

---

### 为什么要叫 scatter 却用 gather？

这和计算机图形学里的"scatter vs gather"概念有关：

- **Scatter（散射）**：`get_map_data()` 阶段，把实体的编号"撒"到空间地图上 → 产生的 `scatter_map` 是一张"编号地图"
- **Gather（收集/查表）**：当前阶段，拿着编号去嵌入表里"取回"实体的真实特征 → 用 `gather` 实现

整个过程是：

```
实体列表                    scatter_map（编号地图）            gather 后的特征地图
┌──────────┐   scatter    ┌─────────────────┐   gather     ┌─────────────────┐
│ 实体1:⋯  │ ──────────→  │  1  │ 0  │  3  │ ──────────→  │ emb₁│  0  │ emb₃│
│ 实体2:⋯  │  (记录编号)   │  0  │ 2  │  0  │  (查表替换)   │  0  │ emb₂│  0  │
│ 实体3:⋯  │              └─────────────────┘              └─────────────────┘
└──────────┘
```

`torch.gather` 在这个流程里承担的是后半部分：**把编号地图转换为特征地图**。

---

### 容易混淆的点

1. **`gather` 和 `scatter` 名称反直觉**：PyTorch 中 `gather` 是 scatter 的反操作。你 scatter 出去以后，要用 gather 拿回来。这个方法的命名 `self.scatter()` 指的是整体"scatter connection"的概念，但内部实现用了 gather。

2. **为什么要在 32 维上重复索引值**：`gather` 要求 index 的 shape 和输入除了 gather 维度外要兼容。每个实体有 32 维嵌入，所以要查 32 次。最简单的方法就是把同一个实体编号复制 32 份，gather 时每维各取各的。

3. **zero_bias 的作用**：因为 scatter_map 中 0 表示"没有实体"，gather 查到 0 就会取出 zero_bias（全 0），后续 sum 时这个位置就是 0，不贡献任何信息。如果不加 zero_bias，gather(0) 会取出原来实体列表的第 0 个实体，那是某个真实实体的嵌入，语义就错了。