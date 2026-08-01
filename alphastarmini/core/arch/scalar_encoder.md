现在我已经阅读了完整代码。下面是这 16 个标量特征逐一讲解。

---

## 标量特征总览

这 16 个标量是 StarCraft II 游戏状态中**数值/类别型信息**的向量表示，不包含空间位置信息（空间信息由 `spatial_encoder` 处理）、也不包含每个实体的详细信息（由 `entity_encoder` 处理）。它们被送入 `ScalarEncoder`，各自经过线性层 + ReLU 后拼接成一个整体嵌入 `embedded_scalar`，最终送给 LSTM（Core）做决策。

---

### 1. `agent_statistics` — 玩家自身经济/军事统计

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, 10)` |
| **来源** | 从 PySC2 的 `obs["player"]` 取 `[1:]`（去掉 player_id） |
| **含义** | 10 维向量，对应 SC2 的核心经济军事指标：矿物、瓦斯、已用人口/总人口、农民数、战斗单位数、总单位数等 |
| **编码方式** | 先取 `log(x + 1)`（压缩大数值），再过 `Linear(10, 64) + ReLU` |
| **角色** | 告诉模型"我现在有多少资源、多少兵"——是做任何决策最基础的信息 |

---

### 2. `home_race` — 自己的种族

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, 5)` |
| **含义** | one-hot 编码，5 种可能：Terran / Zerg / Protoss / Random / 预留位 |
| **编码方式** | `Linear(5, 32) + ReLU` |
| **特殊处理** | 这个嵌入**同时进入 `scalar_context`**，用于后续 gating（门控）机制 |

---

### 3. `away_race` — 对手的种族

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, 5)` |
| **含义** | 同 `home_race` 的编码方式，表示对手种族 |
| **编码方式** | `Linear(5, 32) + ReLU` |
| **特殊处理** | 也进入 `scalar_context`；训练时**10% 概率隐藏对手种族**（模拟天梯上对手选 Random 的情况）；如果对手是 Random，一旦看到对手单位就会把真实种族补充进观测 |

---

### 4. `upgrades` — 当前已研究的科技升级

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, Upgrades_Size)` — 约几十到上百维 |
| **含义** | 布尔向量，每位对应一种升级（如"折跃门研究完成""攻击+1""闪烁"等），1 = 已拥有 |
| **编码方式** | `Linear(Upgrades_Size, 128) + ReLU` |
| **角色** | 模型需要知道"我的追猎能不能闪烁""我的狂热者是不是加速的"来决定战术 |

---

### 5. `enemy_upgrades` — 敌人已研究的科技升级

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, Upgrades_Size)` |
| **含义** | 与 upgrades 对称，但表示**敌人**的科技。代码中注释了 "TODO: how to know enemy's upgrades?"——目前基本是全零向量，因为 SC2 默认不暴露敌人科技 |
| **编码方式** | `Linear(Upgrades_Size, 128) + ReLU` |

---

### 6. `time` — 游戏时间

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, 64)` |
| **含义** | 游戏进行到第几个 game_loop（SC2 一秒钟约 22.4 个 game_loop） |
| **编码方式** | **不经过线性层**，而是用两种编码拼接直接产出 64 维：<br>• 前 32 维：game_loop 的**二进制编码**（比特位展开）<br>• 后 32 维：game_loop/22.4 转为秒数后做**positional encoding** |
| **为什么重要** | 模型需要知道"现在是第一分钟还是第二十分钟"，这决定开局、中期、后期的策略完全不同 |

---

### 7. `available_actions` — 当前可执行的动作掩码

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, Actions_Size)` — 约 500+ 维 |
| **含义** | 布尔向量，每位对应一种动作类型（如"造狂热者""闪烁""造星门"），1 = 当前**理论上**可执行 |
| **编码方式** | `Linear(Actions_Size, 64) + ReLU` |
| **特殊处理** | 也进入 `scalar_context` |
| **实现细节** | 代码使用 `use_human_knowledge_for_available_actions=True`，即基于人类知识判断动作可用性（如：有追猎且有闪烁科技 → 闪烁动作标记为可用，即使正在冷却中），而非依赖游戏 API 的实时状态 |

---

### 8. `unit_counts_bow` — 场上各单位类型的数量统计（词袋）

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, All_Units_Size)` — 约 200+ 维，对应所有可能的单位/建筑类型 |
| **含义** | 每个维度的值 = 场上该类型单位的**当前数量**（如：有 3 个狂热者、1 个星门） |
| **编码方式** | 先 `sqrt(x)` 压缩大值，再过 `Linear(All_Units_Size, 64) + ReLU` |
| **与 `units_buildings` 的区别** | `unit_counts_bow` 是**当前在场上的**（可能被消灭了就从计数中消失），`units_buildings` 是**曾建造过的**（累计的） |

---

### 9. `mmr` — 匹配分（技能等级）

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, 7)` |
| **含义** | MMR（Match Making Rating）的 one-hot 编码。MMR 除以 1000 后取整，cap 在 6。所以 0 代表 0-999 分，6 代表 6000+ 分 |
| **编码方式** | `Linear(7, 64) + ReLU` |
| **SL 阶段** | 编码的是**被模仿的人类玩家**的 MMR（让模型学会不同水平的风格） |
| **RL 阶段** | 固定为 6200（顶级水平），让模型始终以最高水平为目标 |

---

### 10. `units_buildings` — 累计建造过的单位/建筑（累积统计）

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, All_Units_Size)` |
| **含义** | 布尔向量，每位 = 1 表示**曾建造过**该类型的单位或建筑（即使已被摧毁） |
| **编码方式** | `Linear(All_Units_Size, 32) + ReLU` |
| **特殊处理** | 进入 `scalar_context` |
| **角色** | 告诉模型"我曾经造过星门 → 我可能有空军路线"，这是对全局战略走向的刻画 |

---

### 11. `effects` — 当前活跃的状态效果

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, Effects_Size)` — 约 200+ 维 |
| **含义** | 布尔向量，每位 = 1 表示该效果当前在场上存在。效果包括：被攻击警告、隐形状态、闪烁冷却、护盾回复、兴奋剂效果等 |
| **编码方式** | `Linear(Effects_Size, 32) + ReLU` |
| **特殊处理** | 进入 `scalar_context` |

---

### 12. `upgrade` — 累计研究过的科技（累积统计）

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, Upgrades_Size)` |
| **含义** | 与 `upgrades`（第 4 个）内容相同，都是研究过的科技，但被归入"**累积统计**"组（与 `units_buildings`、`effects` 一起），走不同的处理通道 |
| **编码方式** | `Linear(Upgrades_Size, 32) + ReLU` |
| **特殊处理** | 进入 `scalar_context` |
| **为什么要两份** | 代码注释问道"what is the difference with upgrades_fc?"——区别在于语义角色：`upgrades` 是"当前快照"，`upgrade` 被归入累积统计，用于 baseline/value 网络预测长期回报。这是 AlphaStar 论文中 `cumulative_statistics` 的设计：将 units/buildings、effects、upgrades 拆成三个子向量分别编码后再拼起来 |

---

### 13. `beginning_build_order` — 前 20 个建造顺序

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, 20, All_Units_Size)` |
| **含义** | 游戏的**前 20 个建造动作**，每个位置 one-hot 表示造了什么（第 1 个造了农民，第 2 个造了水晶塔……） |
| **编码方式** | 先 `Linear(All_Units_Size + 20, 16)`（拼接了位置序号），然后过一个小 **Transformer**（3 层、2 头、d_k=d_v=8），最后展平为 `20 * 16 = 320` 维 |
| **特殊处理** | 进入 `scalar_context` |
| **为什么重要** | 建造顺序（build order）几乎是开局策略的指纹。Transformer 可以捕捉"第 3 个建筑和第 7 个建筑之间的关系" |

---

### 14. `last_delay` — 上次动作的延迟

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, 128)` |
| **含义** | one-hot 编码，表示**上次发出动作到现在经过了多少个 game_step** |
| **编码方式** | `Linear(128, 64) + ReLU` |
| **为什么重要** | 由于网络延迟和 APM 限制，玩家不能每帧都发指令。这个特征告诉模型"我多久没动作了"，影响动作时机判断 |

---

### 15. `last_action_type` — 上次执行的动作类型

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, Actions_Size)` — 约 500+ 维 |
| **含义** | one-hot 编码，表示**上一次执行的动作是什么类型**（移动、攻击、建造、闪烁……） |
| **编码方式** | `Linear(Actions_Size, 128) + ReLU` |
| **为什么重要** | 上下文连续性——模型不能孤立决策，需要知道"刚才在干什么"，比如刚造了星门，下一步很可能是造虚空或凤凰 |

---

### 16. `last_repeat_queued` — 上次动作的队列/重复标志

| 项目 | 说明 |
|---|---|
| **形状** | `(batch, 2)` |
| **含义** | 两个布尔值的 one-hot 表示：<br>• **queued**：上次动作是否以 shift+click 队列方式发出<br>• **repeat**：上次动作是否是重复指令（如按住快捷键连发） |
| **编码方式** | `Linear(2, 256) + ReLU` |
| **为什么 2 维却用 256 维 FC？** | 虽然输入只有 2 维，但输出维度大是为了让这个"小的离散信息"在拼接时对整体嵌入有足够的影响力 |

---

## 整体数据流示意图

```
16 个原始标量 (各不同维度)
    │
    ├─→ agent_statistics ──→ log(x+1) → Linear(10, 64) + ReLU ──────────────┐
    ├─→ home_race ──→ Linear(5, 32) + ReLU ────→ [embedded] + [scalar_ctx] ──┤
    ├─→ away_race ──→ Linear(5, 32) + ReLU ────→ [embedded] + [scalar_ctx] ──┤
    ├─→ upgrades ──→ Linear(UpSize, 128) + ReLU ─────────────────────────────┤
    ├─→ enemy_upgrades ──→ Linear(UpSize, 128) + ReLU ───────────────────────┤
    ├─→ time ──→ (binary_32 + posenc_32) ────────────────────────────────────┤
    ├─→ available_actions ──→ Linear(ActSize, 64) + ReLU ─→ [emb] + [ctx] ───┤
    ├─→ unit_counts_bow ──→ sqrt(x) → Linear(AllUnits, 64) + ReLU ───────────┤
    ├─→ mmr ──→ Linear(7, 64) + ReLU ────────────────────────────────────────┤
    ├─→ units_buildings ──→ Linear(AllUnits, 32) + ReLU ─→ [emb] + [ctx] ────┤
    ├─→ effects ──→ Linear(EffSize, 32) + ReLU ─────────→ [emb] + [ctx] ────┤
    ├─→ upgrade ──→ Linear(UpSize, 32) + ReLU ──────────→ [emb] + [ctx] ────┤
    ├─→ beginning_build_order ──→ Linear + Transformer ──→ [emb] + [ctx] ────┤
    ├─→ last_delay ──→ Linear(128, 64) + ReLU ────────────────────────────────┤
    ├─→ last_action_type ──→ Linear(ActSize, 128) + ReLU ─────────────────────┤
    └─→ last_repeat_queued ──→ Linear(2, 256) + ReLU ─────────────────────────┤
                                                                    │
                                              torch.cat → embedded_scalar (拼接)
                                                              │
                                                    Linear + ReLU → 最终标量嵌入
```

## 两类输出的区别

`ScalarEncoder.forward()` 输出两个向量：

| 输出 | 来源 | 用途 |
|---|---|---|
| `embedded_scalar` | **全部** 16 个标量的拼接 | 送入 LSTM (Core) 做序列决策 |
| `scalar_context` | 仅 **7 个**标量（home_race、away_race、available_actions、units_buildings、effects、upgrade、beginning_build_order）的拼接 | 送入各 action head 做 **gating（门控）**，让特定动作的选择依赖于这些上下文 |

`scalar_context` 的 7 个特征是精心挑选的——种族决定可用兵种、可用动作掩码限制合法动作空间、累积统计反映全局战略走向、建造顺序编码开局意图——这些都是对"现在该选什么动作"有直接约束作用的上下文信息。


---

这段代码在做一件事：**为前 20 个建造步骤创建一个 padding 掩码（mask）**。不是每局游戏都恰好有 20 个建造动作——游戏刚开始时可能只造了 3 个东西，剩下 17 个位置是空的（全零填充）。这个 mask 告诉后续的 Transformer："哪些位置有真实数据，哪些是 padding，处理时请忽略 padding。"

---

## 逐行拆解

先记住关键形状和变量：

| 变量 | 说明 |
|---|---|
| `beginning_build_order` | `(batch, 20, All_Units_Size)`，每个位置 one-hot 表示造了什么 |
| `SCHP.count_beginning_build_order` | 常量 `= 20`，即"最多记录前 20 个建造" |

---

### 第 1 行

```python
bo_sum = beginning_build_order.sum(dim=-1, keepdim=False)
```

**输入形状**: `(batch, 20, All_Units_Size)`  
**操作**: 沿最后一个维度（All_Units_Size）求和  
**输出形状**: `(batch, 20)`

直观理解：每个位置是一个 one-hot 向量（只有一个 1，其余为 0）。求和后：
- 有真实建造动作的位置 → 和为 `1`
- 全零的 padding 位置 → 和为 `0`

结果 `bo_sum[i, j]` = "第 i 个样本的第 j 个建造位置是否有真实数据"，值为 0 或 1。

---

### 第 2 行

```python
bo_sum = bo_sum.sum(dim=-1, keepdim=False)
```

**输入形状**: `(batch, 20)`  
**操作**: 沿最后一个维度（20 个位置）求和  
**输出形状**: `(batch,)`

将每个位置的 0/1 加起来，得到**该样本一共有多少个真实的建造动作**。

举例：
```
样本 0 造了 5 个东西 → bo_sum[0] = 5
样本 1 造了 12 个东西 → bo_sum[1] = 12
样本 2 刚开局还没造 → bo_sum[2] = 0
```

---

### 第 3 行

```python
bo_sum = bo_sum.unsqueeze(1)
```

**输入形状**: `(batch,)`  
**操作**: 在维度 1 插入一个新维度  
**输出形状**: `(batch, 1)`

举例：`[5, 12, 0]` → `[[5], [12], [0]]`

---

### 第 4 行

```python
bo_sum = bo_sum.repeat(1, SCHP.count_beginning_build_order)
```

**输入形状**: `(batch, 1)`  
**操作**: 沿维度 1 重复 20 次  
**输出形状**: `(batch, 20)`

举例：样本 0 的 `[5]` 重复 20 次 → `[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]`

---

### 第 5 行

```python
mask = torch.arange(SCHP.count_beginning_build_order)
```

**输出形状**: `(20,)`  
**值**: `[0, 1, 2, 3, ..., 19]`

这就是 20 个位置的下标（position index）。

---

### 第 6 行

```python
mask = mask.unsqueeze(0).repeat(batch_size, 1).to(bo_sum.device)
```

**输入形状**: `(20,)` → unsqueeze → `(1, 20)` → repeat → `(batch_size, 20)`  
**含义**: 把 `[0, 1, 2, ..., 19]` 复制 batch_size 份，每一行都是相同的位置序号。

```
mask[0] = [0, 1, 2, ..., 19]
mask[1] = [0, 1, 2, ..., 19]
mask[2] = [0, 1, 2, ..., 19]
```

---

### 第 7 行 — 核心判断

```python
mask = mask < bo_sum
```

这是一个**逐元素比较**。对于每个 `(batch, position)` 位置：

| batch 样本 | 建造总数 (bo_sum) | 位置 0 | 位置 1 | 位置 2 | 位置 3 | 位置 4 | 位置 5+ |
|---|---|---|---|---|---|---|---|
| 样本 0 | 5 | 0<5=**True** | 1<5=**True** | 2<5=**True** | 3<5=**True** | 4<5=**True** | 5<5=**False** |
| 样本 1 | 12 | True | True | ...(前 12 个 True) | ... | ... | 12<12=**False** |
| 样本 2 | 0 | 0<0=**False** | False | ... | ... | ... | ... |

**一句话**：位置下标小于建造总数 → `True`（有真实数据）；否则 → `False`（padding）。

输出形状: `(batch, 20)`

---

### 第 8 行

```python
mask = mask.unsqueeze(2).repeat(1, 1, SCHP.count_beginning_build_order)
```

**输入形状**: `(batch, 20)`  
**操作**: unsqueeze(2) → `(batch, 20, 1)` → repeat 20 次 → `(batch, 20, 20)`

**为什么 shape 要变成 `(batch, 20, 20)`？** 因为这个 mask 最终要传入 Transformer 的**自注意力机制**。自注意力计算 `Q × K^T` 得到的是一个 `(batch, 20, 20)` 的注意力分数矩阵（20 个 query × 20 个 key），mask 必须形状匹配，才能逐位屏蔽。

示意图（样本 0，共 5 个有效建造）：

```
           key位置→
          0  1  2  3  4  5  6 ... 19
query  0 [T  T  T  T  T  F  F ... F]
位置   1 [T  T  T  T  T  F  F ... F]
↓      2 [T  T  T  T  T  F  F ... F]
       3 [T  T  T  T  T  F  F ... F]
       4 [T  T  T  T  T  F  F ... F]
       5 [F  F  F  F  F  F  F ... F]
      ...
      19 [F  F  F  F  F  F  F ... F]
```

每一行都是相同的——这意味着**位置是否可见其他位置，只取决于 query 位置本身是否真实**。真实位置可以看到所有真实位置，padding 位置被完全屏蔽。

---

## 整体流程可视化

```
beginning_build_order (batch, 20, All_Units_Size)
    │  每个位置是一个 one-hot，或全零（padding）
    │
    ▼  sum(dim=-1): one-hot → 0 或 1
bo_sum (batch, 20)
    │  每格 = "这个位置有没有造东西"
    │
    ▼  sum(dim=-1): 统计总数
bo_sum (batch,)  
    │  如 [5, 12, 0] — 每个样本的有效建造数
    │
    ▼  unsqueeze(1) + repeat: 扩展回 20 列
bo_sum (batch, 20)
    │  如 [[5,5,...,5], [12,12,...,12], [0,0,...,0]]
    │
    ▼  mask = arange(20) < bo_sum: 位置下标 vs 有效数
mask (batch, 20)
    │  前 N 列为 True，后面为 False
    │
    ▼  unsqueeze(2).repeat: 扩展为注意力矩阵
mask (batch, 20, 20)
    │
    ▼  传入 Transformer 的 self-attention
    告诉 attention: "只关心真实建造位置之间的相互关系"
```

---

## 为什么需要这个 mask

这段代码后面紧跟着：

```python
x = torch.cat([beginning_build_order, seq], dim=2)     # 拼接位置序号
x = self.before_beginning_build_order(x)                # Linear → (batch, 20, 16)
x = self.beginning_build_order_transformer(x, mask=mask) # Transformer with mask
```

Transformer 处理的是一个**定长序列（20）**，但大多数情况下只有前几个位置有数据。如果不加 mask：
- 全零的 padding 位置也会参与注意力计算
- padding 会"稀释"真实数据的注意力权重
- 模型可能学到"关注空位置"的错误模式

加了 mask 后，Transformer 在处理自注意力时会屏蔽 padding 位置——真实建造只和真实建造互动，空位置被忽略。这和 NLP 中处理变长句子时的 padding mask 是完全一样的思路。