三个概念都查清楚了。下面按"先背景、再两个字典、最后逐步拆函数"的顺序讲。

---

## 一句话核心结论

`ACTIONS_STAT` 和 `GENERAL_ACTION_INFO_MASK` 是两张**"动作 → 可用兵种"的静态先验查表**；`get_batch_unit_type_mask` 则是把这两张表结合**当前这局游戏里实际存在的单位**，算出一个 **"哪些实体槽位对当前动作合法（可以被选中）"的掩码**，用来在 `selected_units_head` 里做 masked softmax，防止网络选中非法单位。

---

## 0. 先理解它解决什么问题（背景）

AlphaStar 的动作是分层的，大致是：

```
动作类型(action_type) → 延迟(delay) → 是否排队(queue) → 选中哪些单位(selected_units) → 目标单位(target_unit) / 目标位置
```

难点在于：**动作和单位之间存在游戏规则限定的"配对关系"**。比如：

- `Train_Zealot_quick`（训练狂热者）只能点在自己的 **Gateway（传送门）** 上；
- `Build_Pylon_pt`（造水晶塔）只能由 **Probe（探机）** 执行；
- `Effect_GravitonBeam` 只能由 **Phoenix（凤凰）** 释放，且只能抓"非重型"单位。

如果让神经网络对战场上几百个实体随意打分，它会大量选中非法单位（游戏里这条指令直接无效），训练效率极低。所以工程上把"哪个动作能作用在哪种兵种上"**预先查好、写死成表**，再结合当前观测把非法槽位直接屏蔽掉。这就是这两个字典存在的意义。

---

## 1. `ACTIONS_STAT` 是什么

定义在 `alphastarmini/third/action_dict.py`（第 579 行起），注释说明它借鉴自 **DI-Star** 项目。

**结构**：一个 dict，键是 mAS 内部的动作 ID（0~563 的稀疏编号），值是一个统计字典：

```python
ACTIONS_STAT = {
    49: {'action_name': 'Train_Zealot_quick',
         'selected_type': [62],              # SC2 原始 unit_type ID：62 = Gateway
         'target_type': [],                  # 不需要目标单位
         'selected_type_name': ['Gateway'],  # 人类可读的名字
         'target_type_name': []},
    36: {'action_name': 'Build_Assimilator_unit',
         'selected_type': [84],              # 84 = Probe
         'target_type': [344, 342, 343, ...],# 气泉等
         ...},
    ...
}
```

每个字段的意思：

| 字段 | 含义 |
|---|---|
| `action_name` | 动作名，如 `Train_Zealot_quick` |
| `selected_type` | **该动作可以"选中/指挥"的兵种 ID 列表**（SC2 原始 ID，不是 mAS 内部索引） |
| `target_type` | 该动作可以"作用/攻击"的目标兵种 ID 列表 |
| `selected_type_name` / `target_type_name` | 对应的兵种名字（纯可读性用途） |

**要点**：
- 这些 ID 是 SC2 的**原始 unit_type ID**（比如 62=Gateway、84=Probe），和 `obs["raw_units"][:, FeatureUnit.unit_type]` 里取出来的值同一种编码。
- 它回答的问题是：**"动作 X 能作用在哪些兵种上？"**——纯粹是静态游戏知识，跟当前战局无关。

---

## 2. `GENERAL_ACTION_INFO_MASK` 是什么

它是从另一个更大的字典 **`ACTION_INFO_MASK`** 过滤出来的子集（`action_dict.py` 第 933~946 行）。

**先讲 `ACTION_INFO_MASK`**：它给每个动作 ID 记录了完整描述，比如：

```python
2: {'name': 'Attack_pt', 'func_type': 'raw_cmd_pt', 'ability_id': 3674,
    'general_id': 0, 'queued': True,
    'selected_units': True, 'target_units': False, 'target_location': True,
    'avail_unit_type': ['TERRAN_MARINE', ...], 'avail_unit_type_id': [48, ...]},
```

关键字段：
- `selected_units`：这个动作**是否需要选中单位**（True/False）
- `target_units` / `target_location`：是否需要目标单位 / 目标位置
- `avail_unit_type_id`：**能执行这个动作的兵种 ID 列表**（和 `ACTIONS_STAT` 的 `selected_type` 语义几乎一样）
- `general_id`：见下面

**为什么要设一个 "general" 概念？** 因为 AlphaStar 把很多"同族"动作合并。例如 `Attack_Battlecruiser_pt`（大和舰专属攻击）的 `general_id=3674`，而 `Attack_pt`（通用攻击）的 `general_id=0`——后者是这一族的"代表动作"。游戏里动作族（general action）约 564 个，但具体展开的动作（含各种单位变体）有上千个。

**`GENERAL_ACTION_INFO_MASK` 的定义**：从 `ACTION_INFO_MASK` 中只挑出 `general_id` 为 `None` 或 `0` 的条目（即"代表动作"本身），构成子集：

```python
GENERAL_ACTION_INFO_MASK = {}
for k, v in ACTION_INFO_MASK.items():
    general_id = v['general_id']
    if general_id is None or general_id == 0:
        GENERAL_ACTION_INFO_MASK[k] = v   # 只保留 general 动作
```

（同一段循环还顺便建了 `ACT_TO_GENERAL_ACT`，用于把任意具体动作映射回它的 general 动作。）

**要点**：`get_batch_unit_type_mask` 里只用它的两个字段：`selected_units`（判断"要不要做实体掩码"）和 `avail_unit_type_id`（合法兵种集合）。

---

## 3. `get_batch_unit_type_mask` 逐步拆解

```python
def get_batch_unit_type_mask(action_types, obs_list):
    # action_types: (batch,) 每个元素是 0~563 的动作 ID
    # obs_list: batch 个观测，每个 obs["raw_units"] 形状 [num_units, 特征列数]
    unit_type_mask_list = []
    for idx, action in enumerate(action_types):
        action = action.item()   # 张量 → python 标量

        # —— 步骤 1：查 GENERAL_ACTION_INFO_MASK，看这个动作需要选中单位吗 ——
        info_1 = {"selected_units": False, "avail_unit_type_id": []}   # 默认：不需要
        if action in AD.GENERAL_ACTION_INFO_MASK:
            info_1 = AD.GENERAL_ACTION_INFO_MASK[action]

        # —— 步骤 2：查 ACTIONS_STAT，拿这个动作的 selected_type ——
        info_2 = {"selected_type": []}                                  # 默认：空
        if action in AD.ACTIONS_STAT:
            info_2 = AD.ACTIONS_STAT[action]

        unit_type_mask = np.zeros([1, AHP.max_entities])   # 一行 = 战场实体槽位
        if info_1["selected_units"]:
            # —— 步骤 3：两张表的合法兵种取并集 ——
            set_all = set(info_1["avail_unit_type_id"]) | set(info_2["selected_type"])

            # —— 步骤 4：遍历这局实际存在的单位，把合法单位的槽位置 1 ——
            raw_units_types = obs_list[idx]["raw_units"][:, FeatureUnit.unit_type]
            for i, t in enumerate(raw_units_types):
                if t in set_all and i < AHP.max_entities:
                    unit_type_mask[0, i] = 1

        unit_type_mask_list.append(unit_type_mask)

    unit_type_masks = np.concatenate(unit_type_mask_list, axis=0)  # [batch, max_entities]
    return unit_type_masks
```

**每一步在干什么：**

1. **查 `GENERAL_ACTION_INFO_MASK`** → 判断这个动作**是否需要选中单位**。`no_op`、`move_camera` 这类不需要选单位的动作，`selected_units=False`，直接跳过，mask 保持全 0（后面 `selected_units_head` 对全 0 的行有专门的兜底处理）。
2. **查 `ACTIONS_STAT`** → 拿到 `selected_type`。两个查询都带默认值兜底，因为两张大表都不是全动作覆盖的。
3. **取并集**：`avail_unit_type_id ∪ selected_type`。两套表来源不同（AlphaStar 原版 vs DI-Star），兵种列表可能有出入，**取并集是为了不漏掉任何合法兵种**（宁可多放行，不可误杀）。
4. **按实体遍历**：`raw_units` 是当前帧的原始单位矩阵，每一行是一个单位，第 `FeatureUnit.unit_type` 列存它的兵种 ID。逐个检查：如果该单位的兵种 ID ∈ `set_all`，并且它的槽位下标 `< max_entities`（实体上限），就在 mask 对应位置写 1。

**输出形状**：`[batch_size, AHP.max_entities]`。注意掩码是**按"实体槽位"**而不是按"兵种"打标的——因为 `selected_units_head` 是逐个实体打分（类似 pointer network 从实体嵌入里挑），所以必须告诉它"第 i 个实体合法与否"。

**生活类比**：就像点餐机。动作类型决定了"你要买什么菜"，两张表是"每种菜只能用哪些食材"，`raw_units` 是"今天厨房里实际有哪些食材"，最后算出来的 mask 就是菜单上**今天真正能点的菜**——不是所有菜都能点（厨房没有），更不是任何东西都能当食材（规则不允许）。

---

## 4. 在模型里怎么被使用

`arch_model.py` 第 223~245 行：

```python
if obs_list is not None:
    unit_type_entity_mask = L.get_batch_unit_type_mask(action_type.squeeze(dim=1), obs_list)
    unit_type_entity_mask = torch.tensor(unit_type_entity_mask, dtype=torch.bool, device=action_type.device)

units_logits, units, ... = self.selected_units_head(..., unit_type_entity_mask=unit_type_entity_mask)
```

- `action_type` 是上一级 head 采样出的动作（形状 `[batch, 1]`），`squeeze` 成 `[batch]`；
- 掩码转成 bool tensor 传给 `selected_units_head`，head 内部对它做 **masked softmax**：只有 mask=1 的实体才有概率被选中，非法实体概率强制为 0；
- 训练和 RL 都用它（`rl_utils.py` 第 153 行也在调用）。

---

## 5. 常见误区提醒

1. **`selected_type` 和 `avail_unit_type_id` 里的数字不是 mAS 内部索引**。它们是 SC2 原始 unit_type ID（如 62=Gateway、84=Probe）。想转成 mAS 的 one-hot 索引需要走 `unit_tpye_to_unit_type_index()`（utils.py 第 39 行，构建了 `all_dict` 映射）。
2. **mask 掩的是"槽位"不是"兵种类别"**。形状是 `[batch, max_entities]`，第 i 位对应 raw_units 第 i 行那个单位，而不是某个兵种 ID 的类别维度。
3. **全 0 行是合法的**。`no_op` 这类动作不需要选单位，它的 mask 行全 0，`selected_units_head` 对全 0 有专门处理（通常退化为选槽位 0 / 不加约束），不要当成 bug。
4. **这是"静态规则 + 动态观测"的结合**：兵种→动作的合法性是写死的，但"这个兵种今天在不在场上"来自 `obs_list`，所以同样的动作在不同帧、不同对局，mask 不同。

---

## 总结关系图

```
ACTION_INFO_MASK（全部动作的详细描述）
   └─ 过滤 general_id∈{None, 0} ──► GENERAL_ACTION_INFO_MASK（代表动作族）
                                          │ 提供 selected_units / avail_unit_type_id
ACTIONS_STAT（DI-Star 的动作×兵种先验表） ─┤ 提供 selected_type
                                          │
                                          ▼
                              get_batch_unit_type_mask(action_types, obs_list)
                                          │  ∪ 合法兵种集，对照 raw_units 实存单位
                                          ▼
                          unit_type_entity_mask [batch, max_entities]
                                          │
                                          ▼
                    selected_units_head 里做 masked softmax（屏蔽非法实体）
```

一句话：**两张静态表回答"动作 X 允许作用于哪些兵种"，`get_batch_unit_type_mask` 把答案和当前战场的实体一对照，产出"这批实体里哪些可以被选中"的动态掩码**，是 AlphaStar 实体选择机制里"可用性掩码"的工程实现。