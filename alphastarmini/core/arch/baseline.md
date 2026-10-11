伪奖励的实现也确认了（`pseudo_reward.py`）。下面完整讲解这 5 个名字。

---

## 一句话核心结论

这 5 个名字是 AlphaStar 的 **5 条独立奖励通道**——**1 条真实奖励（winloss：胜负）+ 4 条伪奖励（build_order / built_units / upgrades / effects：与人类玩家统计量 z 的相似度）**。每条通道各配一个 Baseline（Critic）估值、各算各的策略梯度。名字 = 该通道"衡量什么"。

---

## 1. 总览表

| 通道 | 衡量什么 | 奖励怎么来 | 距离度量 | mAS 现状 | 损失权重 (vtrace, baseline) |
|---|---|---|---|---|---|
| **winloss** | 这局**赢没赢** | 游戏终局真实奖励（±1） | — | ✅ 使用 | **(10.0, 5.0)**，另享 UPGO |
| **build_order** | **建造顺序**像不像人类 | 与 z 的建造顺序的差异 | **Levenshtein**（编辑距离） | ✅ 使用 | (4.0, 1.0) |
| **built_units** | **部队构成**像不像人类 | 与 z 的单位数量词袋的差异 | **Hamming**（逐位差异） | ✅ 使用 | (6.0, 1.0) |
| **upgrades** | **科技升级**像不像人类 | 与 z 的升级集合的差异 | Hamming | ⚠️ 权重置 0，未启用 | (6.0, 1.0) |
| **effects** | **技能效果**像不像人类 | 与 z 的效果集合的差异 | Hamming | ⚠️ 权重置 0，未启用 | (6.0, 1.0) |

---

## 2. 背景动机：为什么除胜负外还要 4 条伪奖励？

两个原因（AlphaStar 的核心设计）：

1. **奖励太稀疏**：星际一局十几分钟，唯一的真实奖励是终局 ±1。纯靠它，前期几百步的操作根本得不到反馈。伪奖励在**每一步**都给“你打得像不像人类”的分数，把稀疏奖励变成稠密信号。
2. **策略条件化（z）**：训练前从人类 replay 统计里采样一份“目标风格” z（`z_build_order` 建造顺序、`z_unit_counts` 部队构成），策略网以 z 为条件生成动作，伪奖励则度量**实际执行与 z 的偏差**——“老师定了教案，每步都批改作业”。

---

## 3. 逐个详解（结合 pseudo_reward.py 代码）

### ① winloss —— 唯一的真实奖励

```python
if reward_name == 'winloss_baseline':
    rewards_tensor = torch.tensor(trajectories.reward, ...)   # 直接用游戏奖励
    return rewards_tensor
```

终局胜负（+1/-1）。它是最重要的通道（vtrace 权重 10、baseline 自身权重 5，还额外用于 UPGO 损失）。对应的 `winloss_baseline` 预测“当前局势下获胜的概率”。

### ② build_order —— 建造顺序伪奖励（Levenshtein 距离）

```python
def reward_by_build_order(bo, z_bo, gl=0):
    str1, str2 = list2str(bo), list2str(z_bo)        # 建造序列转成"字符串"（用 chr 编码每个兵种）
    dist = Levenshtein.distance(str1, str2)          # 编辑距离
    reward = dist * dist                              # 平方放大
    reward = min(reward, 50) / 50.0 * 0.8             # 截断 + 归一化到 [0, 0.8]
    reward = -reward                                  # 取负 → 奖励 ∈ [-0.8, 0]
    return reward
```

**Levenshtein（编辑距离）**：把一个序列变成另一个所需的最少"插入/删除/替换"次数。为什么用它？因为**建造顺序是序列，顺序有意义**——"兵营→工厂"和“工厂→兵营”是不同开局。编辑距离天生衡量序列相似度。

数值流程举例：你的建造序列和人类的差 3 步编辑 → 3²=9 → 9/50×0.8=0.144 → 奖励 −0.144。**差得越远，罚得越重（平方惩罚），但封顶 −0.8**（论文说“差两个传送门以上就到顶”）。

### ③ built_units —— 部队构成伪奖励（Hamming 距离 + 时间衰减）

```python
def reward_by_unit_counts(ucb, z_ucb, gl=0):
    dist = Levenshtein.hamming(str1, str2)    # Hamming 距离：等长向量逐位不同的个数
    reward = -dist
    return reward * time_decay_scale(gl)      # 乘时间衰减
```

**Hamming 距离**：两个等长向量**对应位置不相等的个数**。为什么不用编辑距离？因为单位数量是**词袋**（bag-of-words）——“你有 5 个狂徒 2 个追猎”没有顺序概念，只看每个槽位的数量差。

**时间衰减**（`time_decay_scale`，SC2 里 22.4 游戏循环 = 1 秒）：

| 游戏时间 | 衰减系数 | 含义 |
|---|---|---|
| < 8 分钟 | 1.0 | 前期严管：开局必须像人类 |
| 8~16 分钟 | 0.5 | 中期放松一半 |
| 16~24 分钟 | 0.25 | 后期进一步放松 |
| > 24 分钟 | 0.0 | 完全不管，自由发挥 |

设计直觉：**前期照葫芦画瓢（人类开局定式是优化的），后期允许 AI 走自己的路**——伪奖励是“扶手”不是“枷锁”，随时间逐渐拆除。

### ④⑤ upgrades / effects —— 论文有、mAS 没实现

论文设计（代码注释里保留了原文）：与 built_units 同构的 Hamming 距离——

- **upgrades**：你研发的科技升级集合 vs 人类 replay 里的（同样带 8/16/24 分钟时间衰减）；
- **effects**：你触发的技能效果集合 vs 人类的。

但 mAS 的实现里（第 173~186 行）：

```python
if reward_name == 'effects_baseline':
    weight_hamming = 0.0; weight_leven = 0.0     # 未把 effects 从 unit 统计里分离出来
if reward_name == 'upgrades_baseline':
    weight_hamming = 0.0; weight_leven = 0.0     # 同上，暂不启用
```

伪奖励恒为 0，但 **Baseline（Critic）网络和损失结构仍在**——相当于通道保留、信号未接。这是 mAS 的简化（作者注释：*"now we don't separate the effects from the unit, so we ignore this reward now"*）。

另外 mAS 加了全局缩放：`scale_leven = scale_hamming = 0.05`，防止伪奖励量级盖过真实胜负奖励。

---

## 4. Baseline 侧的对应差异（baseline.py 第 122~131 行）

每个 Critic 按自己的通道**挑选要看的累计统计**（估计“这条通道的未来回报”）：

| baseline_type | 额外看的累计统计 |
|---|---|
| winloss | units_buildings |
| build_order | units_buildings + **effects + upgrade**（全看——建造顺序影响一切） |
| built_units | units_buildings |
| upgrades | upgrade |
| effects | effects |

这也解释了之前算过的维度差：build_order 通道输入 1216 维，其余 1152 维。

---

## 5. 常见误区提醒

1. **这 5 个名字既是“奖励通道名”也是“baseline 名”**——一个名字两顶帽子：`build_order` 指伪奖励时是“Levenshtein 距离打分”，指 `build_order_baseline` 时是“预测该伪奖励累计值的价值网络”。语境要分清。
2. **伪奖励是负数（惩罚式）**——完美贴合人类 = 0，偏离 = 负。它是“批改作业扣分”，不是“做对加分”。
3. **build_order 和 built_units 的度量不能换**——序列用 Levenshtein、词袋用 Hamming，反着用会丢失/捏造顺序信息。
4. **时间衰减只作用于 Hamming 类**（`reward_by_build_order` 里衰减算了但没用，第 50~51 行注释注明）。
5. **mAS 里 upgrades/effects 通道是“空转”的**——Critic 在学、在返回值，但伪奖励恒 0，对策略的实际引导为零。看到这两项的 loss 数值不为零也别误会，那是 Critic 拟合 0 的回归损失。
6. **`22.4` 这个魔法数**是 SC2 的帧率换算（1 秒 = 22.4 游戏循环），时间衰减的“分钟”由它算出。

---

## 总结

```
                    ┌─ winloss ──── 真实奖励：终局胜负 ±1（主通道，权重最高 + UPGO）
                    │
人类统计 z（老师教案）├─ build_order ─ 建造序列 vs z：Levenshtein²，封顶 -0.8（前期严管）
                    ├─ built_units ─ 部队词袋 vs z：Hamming，×时间衰减（8/16/24 分钟逐级放松）
                    ├─ upgrades ─── 升级集合 vs z：Hamming（mAS 未启用，权重 0）
                    └─ effects ──── 效果集合 vs z：Hamming（mAS 未启用，权重 0）

每条通道：伪奖励 r ─► 自家 Baseline 预测 V(s) ─► 优势 = λ-回报 − V(s) ─► 各自的 V-trace 策略梯度
```

一句话：**winloss 是“赢没赢”，后四个是“打得像不像人类”——build_order 管开局套路（序列编辑距离）、built_units 管部队构成（词袋汉明距离）、upgrades 管科技、effects 管技能；它们把十几分钟才有一次的胜负信号，变成每一步都有的“模仿评分”，让 Critic 和策略在稀疏奖励的星际里也能持续学习。**