`get_useful_masks`（第 238-259 行）是 `loss_function()` 预处理段的最后一块拼图。它解决的是 RL 张量化里一个躲不开的问题：**SC2 的实体数量和选兵数量每一步都在变，而张量必须是定长的**。

## 一句话结论

它生成两个布尔掩码：`selected_mask`（每步选兵序列的**有效槽位**）和 `entity_mask`（每步实体表的**有效实体**）——让后面的 log π、熵、KL 只在“真实存在”的位置上计算，padding 出来的位置全部归零。

## 为什么需要：变长 → 定长的代价

看两个输入的来源（第 460-461 行已转置成 `[T, B]`）：

- `select_units_num[t,b]`：第 b 条轨迹第 t 步**实际选了几个兵**。选兵是变长序列，但张量必须开到 `max_selected`（MiniStar 笔记本配置 = 48/16 = **3**，服务器 = 12）；
- `entity_num[t,b]`：那一步场上**真实实体个数**（可能几十个），但实体表张量固定开 `max_entities = 512`（`hyper_parameters.py:208` 注释明确说这个值不能缩减）。

于是每个张量里都有大量 padding 位。如果不管它们，log π 会在“根本不存在的实体”上算概率、熵会把 472 个死位也算进去——loss 全被 padding 污染。掩码就是干这个的。

## 逐段拆解：经典的“长度掩码”三步写法

```python
selected_mask = torch.arange(max_selected, device=device).float()   # ① [0,1,2,...,N-1]
selected_mask = selected_mask.repeat(T * B, 1)                      # ② 复制成 T*B 行
selected_mask = selected_mask < extend_select_units_num.reshape(-1).unsqueeze(-1)  # ③ 广播比较
```

这三行是一个值得背下来的 PyTorch 惯用法，③ 的广播是核心：

```
       左边 [T*B, N]              右边 [T*B, 1]           广播后逐行比较
[[0, 1, 2, 3],                 [[3],                  [[T, T, T, F],   ← 这步选了2个兵
 [0, 1, 2, 3],   <              [4],     →             [T, T, T, T],   ← 这步选了3个兵
 [0, 1, 2, 3],                  [1],                   [T, F, F, F],   ← 这步选了0个兵
 ...]                           ...]                   ...]
```

**数字含义**（以 MiniStar 笔记本配置为例，`max_selected = 3`）：

- 第 239 行 `max_selected = AHP.max_selected + 1`，第 240 行 `+ 1`——两个 **+1 是同一个意思**：选兵序列除了 k 个真兵，还有 **1 个“停止符”槽位**（`change_units_and_logits` 第 336-338 行里那个 `token = max_entities − 1` 就是它：模型选出这个特殊编号表示“选完了”）。所以选了 2 个兵 → 序列实际占 3 个槽 → 掩码是 `[T,T,T,F]`（2 真兵 + 1 停止符）。
- `entity_mask` 同理，只是没有 +1（实体表没有“停止符”的概念）：40 个真实实体 → 前 40 个 True、后 472 个 False。

最后 `reshape(T, B, N)` 还原成三维。**这里有个隐藏前提**：第 ② 步 `repeat` 生成的行序是行优先的（第 0 行对应第 0 个 (t,b)），而 `select_units_num` 恰好在第 460-461 行被 `transpose_sth` 转成了 `[T, B]`——`reshape(-1)` 展开的顺序正好是 t 优先。**如果上游忘了转置，这个掩码会整体错位且不报错**——这是三个 transpose 函数真正的“生死攸关”理由。

## 这两个掩码后来去了哪（三个消费者）

| 消费者 | 用途 | 效果 |
|---|---|---|
| `sum_vtrace_loss` / `sum_upgo_loss` → `get_logprob_and_rhos` → `RA.log_prob` | 算 log π(a) 时乘在结果上 | padding 实体/槽位不贡献 log_prob |
| `entropy_loss` → `get_kl_or_entropy` | 熵计算时乘在 log π 上 | 死位的“假熵”不计入 |
| `human_policy_kl_loss` → `get_kl_or_entropy` | KL 蒸馏同样掩码 | 同上 |

注意它和 `unit_type_entity_mask`（来自轨迹，第 267-268 行）的分工：`entity_mask` 管“**存在性**”（实体槽位是不是 padding），`unit_type_entity_mask` 管“**类型合法性**”（比如某动作只能指向战斗单位）——两者相乘才是完整的有效掩码，在 `log_prob` 第 469 行可以看到它们被乘在一起。

## 两个值得留意的小问题（读代码的“火眼金睛”练习）

1. **第 241、247 行的 `print(...) if 1 else None`**：条件恒真，这是调试残留——正式训练时每一步都会打印掩码张量，白白拖慢速度。全文件还有几处类似的 `if 1`（比如 285 行），你可以找找看，顺便体会“研究代码”和“工程代码”的差距。
2. **掩码是“乘在 log_softmax 之后”，不是真正的 masked softmax**：`rl_algo.py:479-505` 的 `cross_entropy_mask_class` 里，先 `log_softmax` 再乘 mask。严格的 masked softmax 应该在 softmax **之前**把无效位的 logits 压成 −∞，否则 padding 位仍会分走一点概率质量、稀释有效位的 log_prob。对本项目（one-hot 目标 + 有效动作永不出现在 padding 位）影响很小，但这是面试级细节：**mask 乘在归一化前 vs 后，语义完全不同**。

## 自测

1. 某步选了 0 个兵（比如该动作不需要选兵），`selected_mask` 那一行长什么样？（提示：+1 的停止符还在不在？）
2. 为什么 `entity_mask` 的 512 不能像 `max_selected` 那样按 `Select_Scale` 缩小？（提示：看 `hyper_parameters.py:208` 的注释和实体编码的维度。）
3. 如果把第 460-461 行的 `transpose_sth` 删掉，这个函数的输出会怎么错？（形状会对吗？值会怎么错位？）

---

至此 `loss_function()` 的预处理段（第 447-472 行）四个环节全部讲完：`rl_unroll` 重算 → 三个 transpose 换轴 → `get_useful_masks` 造掩码 → 轨迹重排 `[T, B]`。下一行（第 486 行）就进入 for 循环、正式开始算损失了。回复“**下一步**”，进入第 3 步：TD(λ) 价值学习（`td_lambda_loss` + `lambda_returns`）。

---

这两个函数是**纯 Python 数据结构变换**（一行 PyTorch 都没有），核心语法就三样：`namedtuple`、`*` 解包、`zip` 转置。按你的要求以语法为主，先看清“变的是什么”，再逐个语法点拆。

## 一句话结论

`stack_namedtuple` 和 `namedtuple_zip` 合起来干一件事：**把“B 条轨迹、每条是一个 namedtuple”的嵌套结构，转置成“一个 namedtuple、每个字段按 [时间][batch] 排列”**——即 `[batch][field]→[time]` 变成 `[field]→[time]→[batch]`，好和前面 `transpose_baselines` 得到的 `[T, B]` 对齐。

## 背景：变换前后数据长什么样

进 `loss_function` 时，`trajectories` 是**列表，长度 = batch**：

```python
trajectories[0].reward  → [r0, r1, ..., r_{T-1}]   # 轨迹0 的每步奖励
结构：[batch] → Trajectory(22个字段) → [time]
```

出去时（rl_loss.py:472 之后）变成：

```python
trajectories.reward[3]  → (r3_b0, r3_b1, ..., r3_b_{B-1})  # 第3步、所有batch的奖励
结构：Trajectory(22个字段) → [time] → [batch]
```

`np.array(trajectories.is_final)` 这时才能直接得到 `[T, B]` 矩阵（`td_lambda_loss` 第 162 行就是这么用的）。

## 语法点 1：namedtuple 到底是什么

`rl_utils.py:54`：

```python
Trajectory = collections.namedtuple('Trajectory', TRAJECTORY_FIELDS)
```

这行**动态创建了一个类**。它生成的实例同时是两种东西：

```python
p = Trajectory(state=..., reward=...)   # 假设只有两个字段

p.state          # ① 按名访问（tuple 没有的能力）
p[0]             # ② 按下标访问：state 就是第 0 个字段
len(p) == 2      # ③ 长度 = 字段数
for x in p: ...  # ④ 可迭代！按【声明顺序】依次吐出字段值
a, b = p         # ⑤ 可解包
```

第 ④ 条是后面一切的关键：**namedtuple 就是一个带名字的 tuple，遍历它 = 按字段声明顺序逐个取值**。`TRAJECTORY_FIELDS`（第 29-52 行）的顺序因此是全项目的“契约”——字段顺序变了，所有 `zip` 分组就全错。

另外注意：namedtuple **不可变**（没有 `setattr`，这也是为什么 `ArgsActionLogits` 要自己写成带 `__slots__` 的普通类——`transpose_target_logits` 需要写回）。

## 语法点 2：`*` 解包 + `zip` 转置（本课的主角）

`zip(*sequence)` 是 Python 最经典的转置惯用法，分两步理解：

```python
rows = [[1, 2, 3],
        [4, 5, 6]]

zip(*rows)        # ① * 把 rows 拆成两个独立参数：zip([1,2,3], [4,5,6])
                  # ② zip 每轮从每个参数各取一个元素配成 tuple
list(zip(*rows))
# [(1, 4),        #   第一轮：各行第0个
#  (2, 5),        #   第二轮：各行第1个
#  (3, 6)]        #   第三轮：各行第2个
#  行列互换！
```

`*` 在函数调用位置的语义永远是“**把列表拆成一个个独立的位置参数**”：

```python
def f(a, b, c): ...
f(*[1, 2, 3])   # 等价于 f(1, 2, 3)
```

所以 `zip(*trajectory)` 中，`trajectory` 是“B 个 Trajectory 组成的列表”，`*` 把它们拆成 zip 的 B 个参数；**每个 Trajectory 本身是 tuple（可迭代）**，于是 zip 每轮从每条轨迹取“同位置的字段值”配对——第 0 轮配出所有轨迹的 `state`，第 1 轮配出所有轨迹的 `baseline_state`……**22 轮正好按字段分组**。

## 语法点 3：`_make` 类方法

```python
Trajectory._make(d)
```

`_make` 是 namedtuple 自带的类方法：**从任何可迭代对象构造实例**，等价于 `Trajectory(*d)`，只是写法更声明式。这里 `d` 是 22 个“按字段分组的列表”，`_make` 把它们重新装回 namedtuple 的 22 个槽位。

## 逐行拆两个函数

**`stack_namedtuple`（第 85-98 行）——按“字段”聚合：**

```python
d = list(zip(*trajectory))      # zip 返回迭代器，list() 落地成 22 个 tuple
new = Trajectory._make(d)       # 重新装回 namedtuple
```

变换效果（B=2 条轨迹、各 T=2 步）：

```
之前: trajectory = [Traj0(state=[s0,s1], reward=[0,1]),
                    Traj1(state=[t0,t1], reward=[2,3])]
zip(*trajectory) 第0轮 → (Traj0.state, Traj1.state) = ([s0,s1], [t0,t1])
              第1轮 → ([0,1], [2,3])
之后: new.state  == ([s0,s1], [t0,t1])     # [field][batch][time]
      new.reward == ([0,1], [2,3])
```

**`namedtuple_zip`（第 101-113 行）——对每个字段再做一次“batch↔time”转置：**

```python
d = [list(zip(*z)) for z in trajectory]    # ← 关键：遍历的对象是 namedtuple 本身！
new = Trajectory._make(d)
```

- `for z in trajectory`：这里迭代的是 **namedtuple**（语法点 1 的第 ④ 条），`z` 依次取到 22 个字段的值——第 1 轮 `z = trajectory.state`，第 2 轮 `z = trajectory.baseline_state`……
- `zip(*z)`：`z` 是“长度 B 的列表，每个元素是长度 T 的列表”，又是一转置——这次交换的是 **batch 和 time**。
- 列表推导把这 22 次转置的结果收集回 `d`，再 `_make` 装回。

```
z = new.state == ([s0,s1], [t0,t1])
zip(*z) 第0轮 → (s0, t0)     # 时间步0、两条轨迹
        第1轮 → (s1, t1)     # 时间步1
最终: result.state[0] == (s0, t0)    # [field][time][batch] ✓
```

两次变换串联 = **field / batch / time 三个轴的重排**。顺带一个对称美：`zip(*x)` 对矩形数据几乎是自逆的——`zip(*zip(*x)) ≈ x`，所以这两个函数“反着用”就能变回去（`namedtuple_deepcopy` 里就有类似组合）。

## 可运行的最小 Demo（粘到解释器里走一遍）

```python
from collections import namedtuple

Traj = namedtuple('Traj', ['state', 'reward'])

batch = [Traj(state=['s0_b0', 's1_b0'], reward=[0, 1]),   # 轨迹0
         Traj(state=['s0_b1', 's1_b1'], reward=[2, 3])]   # 轨迹1

# --- stack_namedtuple 的核心 ---
d1 = list(zip(*batch))
stacked = Traj._make(d1)
print(stacked.state)        # (['s0_b0','s1_b0'], ['s0_b1','s1_b1'])

# --- namedtuple_zip 的核心 ---
d2 = [list(zip(*z)) for z in stacked]
zipped = Traj._make(d2)
print(zipped.state)         # [('s0_b0','s0_b1'), ('s1_b0','s1_b1')]
print(zipped.reward[1])     # (1, 3)  ← t=1 时刻所有 batch 的 reward
```

## 三个值得注意的坑（语法背后的语义）

1. **zip 按最短截断，且不报错**。如果同一 batch 里两条轨迹长度不同，`zip` 会悄悄丢掉多出来的尾部。本项目隐含假设同一 batch 轨迹等长（`rl_unroll` 里的 `view(batch_size, seq_length)` 也要求这样），否则会先在别处崩。Python 3.10+ 有 `zip(strict=True)` 可以显式检查——这个项目是 3.7，只能靠自己保证。
2. **`try/except` 吞异常是坏味道**。两个函数都是“打印异常 → `return None`"（第 92-96 行）。None 一旦返回，错误会在下游很远处以 `AttributeError: 'NoneType'` 的形式爆出来，排查时别被表面报错位置骗了——真正的错误在打印日志里。
3. **同族函数**：第 69 行 `namedtuple_one_list` 用了 `itertools.chain(*l)`（把多层列表摊平再 zip），是第三个变体；你在 `sum_vtrace_loss` 第 266 行还会见到 `Trajectory(*tuple(item[:-1] for item in trajectories))`——用**生成器**直接构造 namedtuple，和 `_make` 是同一件事的两种写法。到时候别陌生。

## 自测

1. `Trajectory._make(d)` 和 `Trajectory(*d)` 有区别吗？什么时候只能用其中一种？（提示：想想 `_make` 接受的是什么类型的参数。）
2. `namedtuple_zip` 里如果写成 `for z in trajectory.state` 会发生什么？（提示：那遍历的就不是 namedtuple 而是普通列表了，`_make` 会收到什么？）
3. 为什么 `TRAJECTORY_FIELDS` 的字段顺序在这个项目里是“契约”？哪两处代码依赖它？（至少说出 zip 分组和 `ACTION_FIELDS` 掩码索引两处中的一处原理即可。）

---

语法地基打完，`loss_function()` 预处理段（447-472 行）你现在应该能逐行读懂了。回复“**下一步**”，进入第 3 步：TD(λ) 价值学习。

-------

这两行都是给价值损失（`td_lambda_loss` 的输出）**乘权重**，但两个系数的来源和作用层面完全不同。我刚把两条调用链都核实了一遍：

## 一句话结论

`baseline_cost` 是**每路奖励各不相同、写死在超参数里的静态权重**（来自 AlphaStar 论文的 relative weightings）；`baseline_weight` 是**所有路统一、由训练器传进来的运行时总旋钮**（本项目中恒为 1）。两者连乘后得到这条奖励流的价值损失的最终权重。

## 两个系数的来源

**`baseline_cost`——静态、按路设置**

第 501 行从元组解包来：`vtrace_cost, baseline_cost, reward_name = costs_and_rewards`，源头是 `get_baseline_hyperparameters()`：

| 奖励路 | (vtrace_cost, **baseline_cost**) | 价值损失权重 |
|---|---|---|
| winloss | (10.0, **5.0**) | 5.0 |
| build_order | (4.0, **1.0**) | 1.0 |
| built_units | (6.0, **1.0**) | 1.0 |
| upgrades / effects | (6.0, **1.0**) | 1.0 |

它编码的是论文里的设计决策：**胜负的价值头要重点学**（5 倍权重），距离类奖励的价值头只做辅助（1 倍）。对照同一循环里 vtrace 侧的兄弟代码（第 519-522 行 `vtrace_cost * vtrace_weight`），还能读出每路“策略学习 : 价值学习”的配比——winloss 是 10:5（策略是价值的 2 倍），伪奖励路全是 4:1、6:1（策略远重于价值）。直觉：伪奖励路的价值头只是给策略梯度提供降噪用的优势，不必学得极准；winloss 是主目标，价值要学扎实。

**`baseline_weight`（即 `learner_baseline_weight`）——动态、全局统一**

调用链一路传下来：

```
rl_vs_inner_bot_mp.py:75   BASELINE_WEIGHT = 1          ← 启动器定义
        ↓ 701 行传给
learner.py:52/104          self.baseline_weight = 1     ← Learner 构造参数，存成属性
        ↓ 249 行传给
rl_loss.py:452             learner_baseline_weight=1    ← loss_function 的形参
        ↓ 510 行
rl_loss.py:513             loss_baseline = baseline_weight * loss_baseline
```

它和 `baseline_cost` 的三点区别：① **不分路**，5 路价值损失乘同一个数；② **不在 loss 里定义**，而是训练系统外部可调的旋钮——你想“整体放大/缩小价值学习强度”（比如价值头收敛太慢时临时加大）时动它，不用改代码里的分路配比；③ 在本项目的两条启动路径里**都恒为 1**（`BASELINE_WEIGHT = 1`），即当前实际不起作用——但接线留好了，这就是“设计期权重”和“运行期权重”分层的老练做法。

## 合起来的效果

```
loss_baseline(最终) = baseline_cost(按路: 5.0 或 1.0) × baseline_weight(全局: 1) × td_lambda_loss
```

两行拆开写而不是合并成一个系数，正是为了保持“分路配比”和“全局缩放”这两个语义独立的自由度各自可调。

## 顺带纠正一个注释

你在第 510 行写的“看来是评价权重”不太准确——`baseline_weight` 是**价值损失（baseline 头）的总权重**，“评价”在 RL 语境里通常指 critic/价值，这样写以后回看容易歧义，建议改成“价值损失的全局权重（运行时可调，默认 1）”。

## 自测

1. 假设你观察到 winloss 的价值曲线收敛明显慢于伪奖励路，想**只**加大 winloss 价值学习的力度——动 `baseline_cost` 还是 `baseline_weight`？想五路一起加大呢？
2. `only_update_baseline=True` 时（第 518 行），价值损失还受不受这两个系数影响？（提示：`vtrace_weight` 只乘在 vtrace 那边。）
3. 为什么伪奖励路的价值头权重全给 1.0，唯独 winloss 给 5.0？联系第 2 步讲的“伪奖励量级被 0.05 缩放、0.8 封顶”想想。

回复“**下一步**”进入第 4 步：V-trace。

---

没关系，是我上一讲塞得太满了——一上来就 6 个字段、5 个阶段、张量变形，任谁都会晕。**先把上一讲整个忘掉**，我们从一个只有 3 个动作的玩具游戏重新来。

## 这个函数其实在讲一个故事

想象你有两本“想法记录”：

- **旧账本**（`behavior_logits`）：actor 打游戏当时，策略给每个动作打的分。那是“过去的你”；
- **新想法**（`target_logits`）：learner 用最新参数重新算的分。这是“现在的你”。

`get_logprob_and_rhos` 做的事就是**翻旧账**：把当时做的动作翻出来，分别问两个“你”同一个问题——

> “给这个动作打多少分（概率）？”

然后把两个答案一比，就知道**你的想法变了多少**。这个“变化幅度”就是 ρ（重要性权重）。

## 用数字算一遍（整段代码的灵魂就这几行）

设简化游戏只有 3 个动作：上、中、下。

```
旧账本 μ：  上 0.2   中 0.5   下 0.3      ← 打游戏当时的想法
当时实际选了：中
新想法 π：  上 0.1   中 0.7   下 0.2      ← 训练几步之后的想法
```

第一步，问两个策略“中”的概率各是多少（取对数）：

```
log π(中) = log 0.7 ≈ −0.36      ← 新策略：我觉得中挺好
log μ(中) = log 0.5 ≈ −0.69      ← 旧策略：当时我觉得中一般
```

第二步，算比值（对数相减 = 概率相除）：

```
ρ = exp(−0.36 − (−0.69)) = exp(0.33) ≈ 1.4
```

ρ = 1.4 的意思：**“现在你比当时更看好这个动作”**。

第三步，砍一刀：

```
ρ = min(1.4, 1.0) = 1.0     ← 只砍大的，小的保留
```

反过来算一个：若新想法变成 中 0.4，则 ρ = 0.4/0.5 = **0.8** → 保留 0.8。ρ = 0.8 的意思：“你现在不如当时那么看好了，这条旧经验打 8 折使用”。

**ρ 的完整直觉就这一句话：旧经验还能不能信、打几折。** 现在比当时更看好（ρ>1）→ 砍到 1，不放大；现在不如当时看好（ρ<1）→ 按比例打折。

## 对照代码：60 行浓缩成本质 4 行

```python
target_log_prob   = log_prob(target_logits,   actions)   # 问新策略：这动作概率多少？
behavior_log_prob = log_prob(behavior_logits, actions)   # 问旧账本：当时概率多少？
clipped_rhos      = clamp(exp(target − behavior), max=1) # 相除→汇率→砍到1
masks             = ...                                  # 顺手记下：这步这个字段有没有效
```

返回的三样东西：`target_log_prob`（带梯度，用于学习）、`clipped_rhos`（折扣率）、`masks`（有效位）。完。

## 那剩下 50 行在干嘛？

一句话：**包装**。

6 个动作字段形状五花八门——`action_type` 是几百个动作的打分表，`target_location` 是整张地图每个格子的打分表，`units` 是“选了几个兵、每个兵在 512 个实体里选谁”的一串打分表。`log_prob` 这个“问答函数”只认一种统一格式，所以前面 50 行都在做“把不同尺寸的包裹装进统一规格的箱子”：

- 什么 `view`、`reshape`、`filter_by_for_lists` —— 全是搬运和摆形，不碰任何数值逻辑；
- `target_location` 那个 for 循环 —— 只是把 (x, y) 坐标换算成地图摊平后的编号（就像把“第 3 行第 5 列”换成“第 3×宽度+5 个格子”），因为打分表是摊平存的一维表；
- `units` 的特殊处理 —— 因为选兵是“选了一串”，要把一串的概率**加起来**（联合概率：相乘 → 取 log 后变相加）。

这些细节**现在一个都不用记**。你只要守住那 4 行本质，以后每次读这个函数，先在脑子里放“旧账本 vs 新想法 vs 翻旧账”这幅图，剩下的当装箱清单扫一眼就行。

## 检查一下（就两道，都很短）

1. 旧账本给“中”打 0.5，新想法给“中”打 0.05，ρ 是多少？砍不砍？
2. ρ 恒等于 1 的前提是什么？（提示：回到第 1 步——什么情况下“现在的你”和“账本里的你”想法完全一样？）

答出来，这个函数你就懂了；哪一段还是糊的，直接指给我（比如“装箱那段”或“ρ 那三个数”），我只重讲那一段。