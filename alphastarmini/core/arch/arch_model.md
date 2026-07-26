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