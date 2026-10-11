## `state.map_state` 详解

### 一句话结论

`state.map_state` 是 AlphaStar 架构中**三源状态**之一的**空间/地图状态**，它是一个 4D 张量，代表从 SC2 小地图（minimap）提取的多通道空间特征。

---

### 它是什么

在 `MsState`（Multi-source State，多源状态）类中定义（`state.py` 第 33-34 行）：

```python
# called spatital state in alphastar
# tensor, shape: [batch_size, channel_size, width, height]
self.map_state = map_state
```

AlphaStar 把游戏状态拆成了三个来源，分别用不同的编码器处理：

| 状态来源 | 编码器 | 数据类型 |
|---------|--------|---------|
| `entity_state` | EntityEncoder | 实体列表（单位、建筑等） |
| `statistical_state` | ScalarEncoder | 标量信息（资源、科技等） |
| **`map_state`** | **SpatialEncoder** | **空间/地图信息** |

---

### Shape

根据超参数配置，shape 有两种规模：

| 模式 | Shape | 说明 |
|------|-------|------|
| **AlphaStar 原始** | `[batch_size, 24, 128, 128]` | `minimap_size=128`, `map_channels=24` |
| **MiniStar（本项目）** | `[batch_size, 24, 64, 64]` | `minimap_size=64`, `map_channels=24` |

四个维度含义：
- **batch_size**：一批有多少个样本
- **24**：通道数，每个通道代表一种空间特征
- **64（或128）**：地图高度（小地图分辨率）
- **64（或128）**：地图宽度（小地图分辨率）

---

### 24 个通道的具体含义

通道由 `SpatialEncoder.get_map_data()` 函数（`spatial_encoder.py` 第 207-278 行）从 SC2 环境观测中提取。总共 24 个通道，由 9 种特征拼接而成：

| 通道索引 | 特征名 | 通道数 | 编码方式 | 含义 |
|---------|--------|--------|---------|------|
| 0-3 | **scatter_map** | 4 | 原始整数索引 | 每个格子上最多 4 个实体的索引号，用于 scatter 连接把实体嵌入"贴"到对应空间位置上 |
| 4-5 | **camera** | 2 | One-hot | 该位置是否在玩家当前主屏幕视野（camera）内 |
| 6 | **height_map** | 1 | 归一化浮点 | 地形高度，原始值 / 255 |
| 7-10 | **visibility** | 4 | One-hot | 战争迷雾可见度（不可见/已探索/可见/等等） |
| 11-12 | **creep** | 2 | One-hot | 虫族菌毯（creep）是否覆盖该位置 |
| 13-17 | **entity_owners** | 5 | One-hot | 该位置单位归属（自己/敌人/中立/盟友等） |
| 18-19 | **alerts** | 2 | One-hot | 攻击警报（"你的基地遭到攻击"之类） |
| 20-21 | **pathable** | 2 | One-hot | 该位置是否可通行 |
| 22-23 | **buildable** | 2 | One-hot | 该位置是否可建造 |

注意：前 4 个通道（scatter_map）在后续处理中会被替换——SpatialEncoder 会用这四个通道作为索引，从 `entity_embeddings` 中 gather 出对应实体的嵌入向量，然后拼回去（见 `spatial_encoder.py` 第 137-151 行的 scatter 逻辑）。

---

### 数据从哪里来

在 `SpatialEncoder.get_map_data()` 中，原始数据来自 SC2 环境返回的观测字典 `obs["feature_minimap"]`，包含上述 9 种小地图特征。函数把它们分别做 one-hot 编码或归一化后，沿通道维拼接，最终形成 `[batch_size, 24, H, W]` 的张量。

---

### 在整个模型中的作用

从 `arch_model.py` 的 `forward()` 第 168-171 行可以看到：

```python
if AHP.scatter_channels:
    map_skip, embedded_spatial = self.spatial_encoder(state.map_state, entity_embeddings)
else:
    map_skip, embedded_spatial = self.spatial_encoder(state.map_state)
```

SpatialEncoder 接收 `map_state`（以及可选的 `entity_embeddings` 做 scatter 连接），经过 1x1 卷积投影 + 3 次下采样（128→16 或 64→8）+ 4 个 ResBlock + FC 层，最终输出：
- **`embedded_spatial`**：`[batch_size, 256]`，1D 嵌入向量，送入 Core（LSTM）
- **`map_skip`**：跳连接（ResBlock 中间输出），给 LocationHead 做目标位置预测时用


===============

## 一句话核心结论

`multi_gpu_supvised_learning`（正确拼写应为 supervised，代码里少了个 "er"）是 `ArchModel.forward()` / `mimic_forward()` 里的一个**布尔开关**，作用是**改变返回值的打包方式**：为 True 时跳过 `ArgsActionLogits` / `ArgsAction` 这些 namedtuple 封装，直接返回一串**裸张量的扁平元组**。它不改任何计算逻辑，纯粹是为 **DDP 多卡监督学习**绕开 PyTorch 的一个兼容性坑而设的接口开关。

---

## 1. 它是什么：一个“返回格式切换器”

两个 forward 都有这个参数（默认 False）：

```python
def forward(self, ..., multi_gpu_supvised_learning=False, ...):          # 第 140 行
def mimic_forward(self, ..., multi_gpu_supvised_learning=False, ...):    # 第 340 行
```

### 默认路径（False）——正常单卡用法

`forward` 把结果装进 namedtuple 再返回（第 295~300、331~337 行）：

```python
action_logits = ArgsActionLogits(action_type=action_type_logits, delay=delay_logits, queue=queue_logits,
                                 units=units_logits, target_unit=target_unit_logits, 
                                 target_location=target_location_logits)   # ← 结构化 namedtuple
action = ArgsAction(action_type=action_type, delay=delay, ...)           # ← 采样动作也打包

return action_logits, action, hidden_state, select_units_num, entity_nums
```

### 多卡路径（True）——裸张量扁平返回（第 302~304 行）

```python
if multi_gpu_supvised_learning:
    return action_type, entity_nums, units, target_unit, target_location, action_type_logits, delay_logits, queue_logits, \
        units_logits, target_unit_logits, target_location_logits, select_units_num
```

`mimic_forward` 同理（第 491~493 行），只是多带 `hidden_state, unit_types_one`，并且**提前 return**，跳过后面的 baseline 计算分支。

**计算过程一模一样**——编码器、LSTM、六个 head 全都照常跑，变的只是“结果怎么装、返回给谁”。

---

## 2. 为什么需要它：DDP 和 namedtuple 的兼容坑

答案写在消费方 `sl_loss_multi_gpu.py` 的注释里（第 155~157 行，212~214 行重复出现）：

```python
# we can't make them all into as a list or into ArgsActionLogits
# if we do that, the pytorch DDP will cause a runtime error, just like the loss don't include all parameters
# This error is strange, so we choose to use a specific loss writing schema for multi-gpu calculation.
```

翻译过来：**如果把输出打包成 `ArgsActionLogits`（或 list），PyTorch DDP 会报"loss 没有包含所有参数”之类的运行时错误**。这个错误很诡异，所以作者干脆为多卡写了一套特殊的返回/损失模式——模型直接吐裸张量，损失函数逐个消费。

背后的机制直觉（帮你理解这个“诡异错误”）：

1. DDP 的每个进程持有一份模型副本，`loss.backward()` 后自动同步梯度；DDP 要求**反向传播能触达所有参数**；
2. `ArgsActionLogits` 这个 namedtuple 里**混装了各种东西**：logits 张量、离散采样值（无梯度）、掩码、列表——DDP 对输出做梯度钩子追踪和跨进程收集（gather/scatter）时，对嵌套 namedtuple 的处理非常脆弱；
3. 一旦某些 logits 被 namedtuple “包”得让 loss 侧没直接用上，DDP 就判定“有参数没参与 loss”，直接报错。

**解法就是绕开**：多卡时返回扁平的裸张量元组，loss 函数（`get_masked_classify_loss_for_multi_gpu`）把每个 logits 单独接走：

```python
# sl_loss_multi_gpu.py 第 158~169 行
action_pred, entity_nums, units, target_unit, target_location, action_type_logits, \
    delay_logits, queue_logits, units_logits, target_unit_logits, \
    target_location_logits, select_units_num = model.forward(state, ..., 
                                                             multi_gpu_supvised_learning=True)  # ← True

loss, loss_list = get_masked_classify_loss_for_multi_gpu(action_gt, action_pred, entity_nums, 
                                                         action_type_logits, delay_logits, queue_logits, ...)
```

---

## 3. 谁在用、谁没用（调用地图）

| 调用方 | 传值 | 场景 |
|---|---|---|
| `sl_loss_multi_gpu.py`（3 处：第 162、245、269 行） | **True** | 多卡 SL 训练的 loss 和精度计算 |
| `sl_multi_gpu_by_pickle.py` / `sl_multi_gpu_by_tensor.py` | （间接经上层） | **DDP 多卡训练脚本**（import `DistributedDataParallel`，用 `mp` 多进程 + `DistributedSampler`） |
| `agent.py`（第 261、290 行） | 显式 **False** | RL 路径（`action_logits_by_state` 等） |
| `sl_train_by_tensor.py` / `sl_train_by_pickle.py`（单卡 SL） | 默认 False | 正常单卡监督训练 |

一个有意思的细节：`get_sl_loss_for_tensor`（多卡版）里**模型每步跑两遍**——第一遍 `mimic_forward`（teacher forcing，算 loss），第二遍 `forward`（自由采样，只为了算 accuracy 统计），两遍都带 `multi_gpu_supvised_learning=True`。

另一个历史痕迹：`sl_train_by_pickle.py` 第 97 行有一行**被注释掉的** `nn.DataParallel` 代码——说明作者早期还试过 DataParallel 方案，后来多卡脚本统一走了 DDP。

---

## 4. 项目对多卡训练的官方态度（重要背景）

按仓库文档（AGENTS.md / README）：

- **多 GPU SL 训练不被推荐**：`sl_multi_gpu_by_*.py` 内存开销大、训练不稳定；
- README **建议只用单 GPU**（多 GPU 有已知的 PyTorch 不稳定性问题）。

所以这个开关对应的整条链路（`multi_gpu_supvised_learning=True` → `sl_loss_multi_gpu` → `sl_multi_gpu_by_*`）是**实验性代码**，主线训练（单卡 `sl_train_by_tensor`）根本不经过它。

---

## 5. 常见误区提醒

1. **它不影响任何计算**。别以为它控制梯度同步或显存分布——那些是 DDP 干的事。它只决定 forward 结束时“结果怎么打包返回”。
2. **拼写就是错的**（supvised → supervised），grep 时别按正确拼写搜，会搜不到。
3. **True 路径会跳过 baseline 分支**（第 302 行提前 return）。如果哪天要多卡训练又想要 baseline 输出，现有代码拿不到——需要自己改。
4. **返回值顺序是"裸约定"**：13 个（mimic 是 14 个）返回值全靠位置对应，没有名字。调用方少接一个变量就整体错位——这是这种扁平接口的代价（也是为什么单卡路径用 namedtuple 更安全）。
5. **DDP 的错误不是确定性复现的**“诡异 bug”（作者原话），不同 PyTorch 版本表现可能不同；这也是项目最终退回“单卡为主”的原因之一。

---

## 总结

```
ArchModel.forward / mimic_forward
        │
        ├─ multi_gpu_supvised_learning=False（默认）
        │     └─► 返回 ArgsActionLogits / ArgsAction namedtuple（结构化，单卡友好）
        │          用于：单卡 SL、RL、评估
        │
        └─ multi_gpu_supvised_learning=True
              └─► 返回裸张量扁平元组（13/14 个，按位置约定）
                   用于：DDP 多卡 SL（sl_multi_gpu_by_* → sl_loss_multi_gpu）
                   原因：namedtuple 会让 DDP 报"loss 不含所有参数"的诡异错误
```

一句话：**`multi_gpu_supvised_learning` 是多卡监督学习的"返回格式逃生舱"——因为 DDP 处理不了嵌套 namedtuple 的输出，模型在多卡模式下改吐裸张量、由专门的 loss 函数逐个接住；但这条多卡链路本身是项目里不推荐使用的实验性代码。**


==============


构造函数找到了，证据链完整。下面系统讲解。

---

## 一句话核心结论

`baseline_state` 和 `baseline_opponent_state` 是**专门喂给 5 个 Baseline（Critic）的“标量观测包”**——各含 **8 项特征**：从主模型的 16 项标量状态里**挑出 7 项与“局势估值”相关的**（资源、升级、部队构成、建造顺序等），再补一项 **Critic 专属的 `cumulative_score`（游戏累计积分，人类玩家都看不到）**。前者来自**我方视角**的观测，后者来自**对方视角**；打内置电脑时对方观测拿不到，传 `None`（内部用全零代替）。

---

## 1. 它们从哪来：一个“挑挑拣拣 + 加菜”的函数

采集入口在 `agent.py` 第 144~191 行：

```python
def get_baseline_state_from_multi_source_state(obs, msstate):
    scalar_list = msstate.statistical_state          # 主模型 state 里的 16 项标量

    # 从 16 项里按索引挑 7 项
    baseline_index = [0, 3, 7, 9, 10, 11, 12]
    baseline_state = [e for i, e in enumerate(scalar_list) if i in baseline_index]

    # 补上 Critic 专属的累计得分
    cumulative_score = torch.tensor(np.array(obs['score_cumulative']), ...).reshape(1, -1)
    baseline_state.append(cumulative_score)

    return baseline_state                             # 8 项列表
```

RL 交互时（`rl_vs_inner_bot_mp.py` 第 317 行）：

```python
state = agent_nn.preprocess_state_all(home_obs.observation, build_order=player_bo, last_list=last_list)
baseline_state = agent_nn.get_baseline_state_from_multi_source_state(home_obs.observation, state)
...
traj_step = Trajectory(state=state, baseline_state=baseline_state,
                       baseline_state_op=None,   # 打内置电脑：没有对方观测
                       ...)
```

自战对局（`actor_plus_z.py` 第 273~280 行）则两边都采：

```python
state    = preprocess_state_all(home_obs.observation, build_order=player_bo)   # 我方
state_op = preprocess_state_all(away_obs.observation)                          # 对方视角的完整观测
baseline_state    = get_baseline_state_from_multi_source_state(state)
baseline_state_op = get_baseline_state_from_multi_source_state(state_op)
```

---

## 2. 8 项内容明细

顺序与 `Baseline.pre_forward` 的解包（baseline.py 第 83~84 行）一一对应：

| # | 特征 | 来源索引 | 维度（MiniStar） | 含义 | `pre_forward` 里的处理 |
|---|---|---|---|---|---|
| 0 | `agent_statistics` | 16 项中的 0 | 10 | 矿/气/人口/军队价值等运营统计 | `log(x+1)` → Linear(10→48) + ReLU |
| 1 | `upgrades` | 3 | 320 | 已研发的科技升级（布尔） | Linear(320→32) + ReLU |
| 2 | `unit_counts_bow` | 7 | 兵种总数 | 各兵种当前数量（词袋） | Linear(→48) + ReLU |
| 3 | `units_buildings` | 9 | 兵种总数 | 累计造过哪些单位/建筑（布尔） | Linear(→16) + ReLU（按 baseline 类型取舍） |
| 4 | `effects` | 10 | 效果数 | 累计触发过哪些效果（布尔） | Linear(→16) + ReLU（按类型取舍） |
| 5 | `upgrade` | 11 | 320 | 累计升级统计（布尔） | Linear(→16) + ReLU（按类型取舍） |
| 6 | `beginning_build_order` | 12 | [20, 兵种总数] | 开局前 20 个建造顺序 | 拼序列 one-hot → Linear → 小 Transformer(d=16) → 压平 320 |
| 7 | `cumulative_score` | **新增** | 13 | **游戏累计积分**（见下） | `log(x+1)` → Linear(13→48) + ReLU |

各路嵌入拼接后（约 512 维，MiniStar），与 `lstm_output`（128）和对方同名特征（512）三路拼接 → 1152 → ResBlock 组 → 输出标量 → `atan` 压缩。

**`cumulative_score` 为什么特殊**（代码注释引自 AlphaStar 论文，第 180~183 行）：这些是**游戏内部追踪的积分指标，只给 baseline 用、推理时不可见、人类玩家游戏中也看不到**——包括总分、闲置生产/工作时间、单位与建筑总价值、摧毁总价值、采集矿/气总量与速率、支出总量等 13 项。相当于“考官专用的成绩单”：训练时 Critic 可以看它来判断局势好坏，但策略网不能看（避免依赖打分作弊）。

---

## 3. 挑了什么、扔了什么、为什么

对比主模型 `statistical_state` 的 16 项，Baseline **扔掉的 9 项**：

| 被扔掉的项 | 索引 | 为什么 Critic 不需要 |
|---|---|---|
| `home_race` / `away_race` | 1, 2 | 种族对“估值”影响小，且相对固定 |
| `enemy_upgrades` | 4 | 对方信息改由 `baseline_opponent_state` 整体提供（自战时） |
| `time` | 5 | 局势价值已隐含在部队/科技状态里 |
| `available_actions` | 6 | 这是**决策上下文**（哪些动作可执行），估值用不上 |
| `mmr` | 8 | 人类玩家天梯分，训练对局里无意义 |
| `last_delay` / `last_action_type` / `last_repeat_queued` | 13~15 | “上一步自己做了什么”是策略侧的记忆（已进 LSTM），估值不关心 |

**选取逻辑很清晰**：Critic 要回答“当前局势值多少分”，所以只看**局面性特征**（资源、兵力、科技、建造、得分）；不看**决策性特征**（可用动作、上次操作）。

---

## 4. `baseline_opponent_state`：对称采集 + 非对称兜底

| 场景 | 对方状态的来源 |
|---|---|
| **自战**（league/self-play，`actor_plus_z.py`） | 从**对方玩家的观测**（`away_obs`）完整采集一份同格式 8 项——你知道对方的资源、部队构成、建造顺序 |
| **打内置电脑**（`rl_vs_inner_bot_mp.py` 第 459 行） | `baseline_state_op=None` → `Baseline.forward` 里用 `torch.zeros_like` 顶替（第 186 行） |

这是一个“**非对称 Actor-Critic**”设计：**策略网看不到对方的运营数据**（它只能通过可见的实体和地图了解敌人），但 **Critic 在训练时可以看到**——估值更准，策略梯度的基线更好，而策略本身不会学会“偷看对面数据”这种部署时不存在的依赖。

---

## 5. 数据流全链路

```
游戏中每一步：
  home_obs.observation ─► preprocess_state_all ─► state（16 项标量 + 实体 + 地图）
                                    │
                                    └─► get_baseline_state_from_multi_source_state
                                          = 16 项里挑 7 项 [0,3,7,9,10,11,12] + score_cumulative
                                          → baseline_state（8 项，我方）
  away_obs.observation（仅自战）──► 同样流程 ──► baseline_opponent_state（8 项，对方）
        │
        ▼
  存入 Trajectory（随轨迹逐步记录）
        │
        ▼  学习时（alphastar_agent.rl_unroll 第 386~415 行）
  按时间堆叠 [batch, seq, …] → 逐帧切片成 [batch, …]
        │
        ▼
  arch_model.forward(..., baseline_state=..., baseline_opponent_state=...)
        │  （return_logits=True 且 return_baseline=True 才走）
        ▼
  5 × Baseline.forward(lstm_output, 我方 8 项, 对方 8 项)
```

（`arch_model.test()` 第 598~619 行手工构造的 8 项列表，顺序与此完全一致，可以当速查模板。）

---

## 6. 常见误区提醒

1. **它们不是“另一个 state”**——不含实体矩阵和地图，**只有 8 项标量**；实体/地图信息经 `lstm_output` 间接进入 Baseline。
2. **`cumulative_score` 只在 baseline 里出现**，主模型的 16 项标量里没有它——这是“Critic 专属信息、策略不可见”的关键设计，别把它混进 statistical_state。
3. **对方状态不是“敌方可见单位”**——那是实体/地图层面的事；这里的对方状态是**对方玩家的运营统计**（他的资源、他的建造顺序），只在自战且训练时可见。
4. **打电脑时传 None 不是 bug**：`Baseline.forward` 里有 `torch.zeros_like` 兜底，对方支路输出恒 0。
5. **挑选用的是索引列表 `[0,3,7,9,10,11,12]`**，与 `statistical_state` 的排列顺序强耦合——如果有人调整 16 项标量的顺序而忘了改这个索引，Baseline 会静悄悄地拿错特征（不报错、只掉点），是个隐蔽的坑。

---

## 总结

一句话：**`baseline_state` / `baseline_opponent_state` 是 Critic 专属的“局势成绩单”——从 16 项标量状态中挑出 7 项局面性特征（运营统计、升级、部队词袋、累计建筑/效果/升级、开局建造顺序），再附上策略不可见的 `cumulative_score` 游戏积分，我方一份、对方一份（自战才采集）；它们让 5 个价值网络能比策略看到更多、更准的局势信息，从而给出更稳的优势估计。**