#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Baseline Encoder."
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.core.arch.spatial_encoder import ResBlock1D

from alphastarmini.lib.alphastar_transformer import Transformer

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS

__author__ = "Ruo-Ze Liu"

debug = False


class Baseline(nn.Module):
    '''    
    Inputs: prev_state, scalar_features, opponent_observations, cumulative_score, action_type, lstm_output
    Outputs:
        winloss_baseline - A baseline value used for the `action_type` argument
    '''

    def __init__(self, baseline_type='winloss',
                 n_statistics=SFS.agent_statistics,
                 n_cumulatscore=SFS.cumulative_score,
                 baseline_input=AHP.winloss_baseline_input_size,
                 n_upgrades=SFS.upgrades, 
                 n_units_buildings=SFS.unit_counts_bow, 
                 n_effects=SFS.effects, n_upgrade=SFS.upgrade,
                 n_resblocks=AHP.n_resblocks, 
                 original_32=AHP.original_32,
                 original_64=AHP.original_64,
                 original_128=AHP.original_128,
                 original_256=AHP.original_256):
        super().__init__()
        self.baseline_type = baseline_type
        if baseline_type == 'build_order':
            baseline_input = AHP.build_order_baseline_input_size
        elif baseline_type == 'built_units':
            baseline_input = AHP.built_units_baseline_input_size
        elif baseline_type == 'upgrades':
            baseline_input = AHP.upgrades_baseline_input_size
        elif baseline_type == 'effects':
            baseline_input = AHP.effects_baseline_input_size

        self.statistics_fc = nn.Linear(n_statistics, original_64)  # with relu
        self.cumulatscore_fc = nn.Linear(n_cumulatscore, original_64)  # with relu
        self.upgrades_fc = nn.Linear(n_upgrades, original_128)  # with relu
        self.unit_counts_bow_fc = nn.Linear(n_units_buildings, original_64)  # A bag-of-words unit count from `entity_list`, with relu
        self.units_buildings_fc = nn.Linear(n_units_buildings, original_32)  # with relu, also goto scalar_context
        self.effects_fc = nn.Linear(n_effects, original_32)  # with relu, also goto scalar_context
        self.upgrade_fc = nn.Linear(n_upgrade, original_32)  # with relu, also goto scalar_context. What is the difference with upgrades_fc?

        self.build_order_model_size = 16
        self.before_beginning_build_order = nn.Linear(n_units_buildings + SCHP.count_beginning_build_order, 
                                                      self.build_order_model_size)  # without relu  
        self.beginning_build_order_transformer = Transformer(d_model=self.build_order_model_size, 
                                                             d_inner=self.build_order_model_size * 2,
                                                             n_layers=3, n_head=2, 
                                                             d_k=8, d_v=8, 
                                                             dropout=0.)
        self.relu = nn.ReLU()

        self.embed_fc = nn.Linear(baseline_input, original_256)  # with relu
        self.resblock_stack = nn.ModuleList([
            ResBlock1D(inplanes=original_256, planes=original_256, seq_len=1)
            for _ in range(n_resblocks)])

        self.out_fc = nn.Linear(original_256, 1) 

    def pre_forward(self, various_observations):
        '''
        various_observations,：是专门喂给 5 个 Baseline（Critic）的“标量观测包”——各含 8 项特征：从主模型的 16 项标量状态里挑出 7 项与“局势估值”相关的（资源、升级、部队构成、建造顺序等），再补一项 Critic 专属的 cumulative_score（游戏累计积分，人类玩家都看不到）。前者来自我方视角的观测，后者来自对方视角；打内置电脑时对方观测拿不到，传 None（内部用全零代替）
        '''

        '''
        agent_statistics: 矿/气/人口/军队价值等运营统计, shape (batch, 10)
        upgrades: 已研发的科技升级（布尔）, shape (batch, 320)
        unit_counts_bow: 各兵种当前数量（词袋）（batch，1）
        units_buildings: 累计造过哪些单位/建筑（布尔） （batch， 1）
        effects: 累计触发过哪些效果（布尔） （batch, 1)
        upgrade: 累计升级统计（布尔） (batch, 320)
        beginning_build_order: 开局前 20 个建造顺序 (batch, 20, 建筑的one-hot编码）
        cumulative_score: 游戏累计积分 这些是游戏内部追踪的积分指标，只给 baseline 用、推理时不可见、人类玩家游戏中也看不到——包括总分、闲置生产/工作时间、单位与建筑总价值、摧毁总价值、采集矿/气总量与速率、支出总量等 13 项。相当于“考官专用的成绩单”：训练时 Critic 可以看它来判断局势好坏，但策略网不能看（避免依赖打分作弊）。相当于全图信息
        '''
        [agent_statistics, upgrades, unit_counts_bow,
         units_buildings, effects, upgrade, beginning_build_order, cumulative_score] = various_observations

        # These features are all concatenated together to yield `action_type_input`, 
        # passed through a linear of size 256, then passed through 16 ResBlocks with 256 hidden units 
        # and layer normalization, passed through a ReLU, then passed through 
        # a linear with 1 hidden unit.

        device = next(self.parameters()).device
        embedded_scalar_list = [] # 将每一个游戏状态的嵌入向量保存到该列表中

        # agent_statistics: Embedded by taking log(agent_statistics + 1) and passing through a linear of size 64 and a ReLU
        # +1 是保证数值不为0 torch.log 是将数值压缩
        the_log_statistics = torch.log(agent_statistics + 1) 
        x = F.relu(self.statistics_fc(the_log_statistics)) # x shape is （batch, original_64)
        del agent_statistics, the_log_statistics
        embedded_scalar_list.append(x)

        # 同上
        score_log_statistics = torch.log(cumulative_score + 1)
        x = F.relu(self.cumulatscore_fc(score_log_statistics))
        del cumulative_score, score_log_statistics
        embedded_scalar_list.append(x)

        # 同上，唯一不同的是由于数量级本身就足够小，所以不需要进行log压缩数量级
        # upgrades: The boolean vector of whether an upgrade is present is embedded through a linear of size 128 and a ReLU
        x = F.relu(self.upgrades_fc(upgrades))
        del upgrades
        embedded_scalar_list.append(x)

        # unit_counts_bow: A bag-of-words unit count from `entity_list`. 
        # The unit count vector is embedded by square rooting, passing through a linear layer, and passing through a ReLU
        # 同上
        x = F.relu(self.unit_counts_bow_fc(unit_counts_bow))
        del unit_counts_bow
        embedded_scalar_list.append(x)

        # cumulative_statistics: The cumulative statistics (including units, buildings, effects, and upgrades) are preprocessed 
        # into a boolean vector of whether or not statistic is present in a human game. 
        # That vector is split into 3 sub-vectors of units/buildings, effects, and upgrades, 
        # and each subvector is passed through a linear of size 32 and a ReLU, and concatenated together.
        # The embedding is also added to `scalar_context`

        cumulative_statistics = []  # it is different in different baseline
        # 不同的类型使用不同计算方式
        '''
        通道	衡量什么	奖励怎么来	距离度量	mAS 现状	损失权重 (vtrace, baseline)
        winloss	这局赢没赢	游戏终局真实奖励（±1）	—	✅ 使用	(10.0, 5.0)，另享 UPGO
        build_order	建造顺序像不像人类	与 z 的建造顺序的差异	Levenshtein（编辑距离）	✅ 使用	(4.0, 1.0)
        built_units	部队构成像不像人类	与 z 的单位数量词袋的差异	Hamming（逐位差异）	✅ 使用	(6.0, 1.0)
        upgrades	科技升级像不像人类	与 z 的升级集合的差异	Hamming	⚠️ 权重置 0，未启用	(6.0, 1.0)
        effects	技能效果像不像人类	与 z 的效果集合的差异	Hamming	⚠️ 权重置 0，未启用	(6.0, 1.0)

        其余具体看md文档
        '''
        # 这里也是同上，只是根据不同的评判基准
        if self.baseline_type == "winloss" or self.baseline_type == "build_order" or self.baseline_type == "built_units":
            x = F.relu(self.units_buildings_fc(units_buildings))
            cumulative_statistics.append(x)
        if self.baseline_type == "effects" or self.baseline_type == "build_order":
            x = F.relu(self.effects_fc(effects))
            cumulative_statistics.append(x)
        if self.baseline_type == "upgrades" or self.baseline_type == "build_order":    
            x = F.relu(self.upgrade_fc(upgrade))
            cumulative_statistics.append(x)

        embedded_scalar_list.extend(cumulative_statistics)
        del units_buildings, effects, upgrade, cumulative_statistics

        # beginning_build_order: The first 20 constructed entities are converted to a 2D tensor of size 
        # [20, num_entity_types], concatenated with indices and the binary encodings 
        # (as in the Entity Encoder) of where entities were constructed (if applicable). 
        # The concatenation is passed through a transformer similar to the one in the entity encoder, 
        # but with keys, queries, and values of 8 and with a MLP hidden size of 32. 
        # The embedding is also added to `scalar_context`.
        print("beginning_build_order:", beginning_build_order) if debug else None
        print("beginning_build_order.shape:", beginning_build_order.shape) if debug else None

        batch_size = beginning_build_order.shape[0]

        # 这三行是在给“开局建造顺序”这个序列做位置编码（positional encoding）——给 20 个建造步骤的每一步打上“你是第几步”的标签，让后面的 Transformer 能感知顺序。
        seq = torch.arange(SCHP.count_beginning_build_order)
        # SCHP.count_beginning_build_order = 20
        # → tensor([0, 1, 2, ..., 19])，形状 [20]
        # 含义：20 个建造步骤的"序号"
        seq = L.tensor_one_hot(seq, SCHP.count_beginning_build_order)
        # 每个序号转成 one-hot：[20] → [20, 20]
        # 第 0 行 = [1,0,0,...,0]，第 1 行 = [0,1,0,...,0] ...
        # 数学上就是 torch.eye(20)（单位矩阵）
        seq = seq.unsqueeze(0).repeat(batch_size, 1, 1).to(beginning_build_order.device)
        # [20,20] → [1,20,20] → [batch,20,20]
        # 给 batch 里每个样本复制同一份位置编码（位置与样本无关，天然共享）
        # .to(...) 搬到与建造顺序特征相同的设备（GPU）

        bo_sum = beginning_build_order.sum(dim=-1, keepdim=False) # 将最后一个维度one-hot编码合并，[batch, 20, E] → [batch, 20]：每行 one-hot 求和
        bo_sum = bo_sum.sum(dim=-1, keepdim=False) # 已建造的行：one-hot 有一个 1 → 行和 = 1 ，计算合并后的和，从而得到建造了几个建筑，没有建造建筑的行为全零
        bo_sum = bo_sum.unsqueeze(1) # 插一维，为展开做准备
        bo_sum = bo_sum.repeat(1, SCHP.count_beginning_build_order) 
        # [batch, 1] → [batch, 20]：同一个数复制 20 份
        # 样本A: [3, 3, 3, ..., 3]    样本B: [5, 5, 5, ..., 5]  

        # 掩码，因为可能还在建造这么多建筑，所以需要知道哪些位置是还没建造建筑的地方
        mask = torch.arange(SCHP.count_beginning_build_order)
        mask = mask.unsqueeze(0).repeat(batch_size, 1).to(bo_sum.device)
        mask = mask < bo_sum
        mask = mask.unsqueeze(2).repeat(1, 1, SCHP.count_beginning_build_order)

        # add the seq info, referenced by the processing way of DI-star
        x = torch.cat([beginning_build_order, seq], dim=2) # 将建造建筑和建造顺序拼接起来
        x = self.before_beginning_build_order(x) # 混合提取特征

        # like in entity encoder, we add a sequence mask
        x = self.beginning_build_order_transformer(x, mask=mask) # 提取特征
        x = x.reshape(x.shape[0], SCHP.count_beginning_build_order * self.build_order_model_size)

        embedded_scalar_list.append(x) # 将建造顺序和建造建筑的特征也添加进去
        embedded_scalar = torch.cat(embedded_scalar_list, dim=1) # 然后将特征cat起来，cat要求拼接维度以外的维度相同就可以拼接，这里就相当于
        # 把不通特征都拼接到一个list列表中

        del x, mask, bo_sum, seq, embedded_scalar_list

        # 返回一个混合了对局势评价的特征提取后的特征矩阵
        # shape 应该是【batch，512】
        return embedded_scalar

    def forward(self, lstm_output, various_observations, opponent_observations=None):
        '''
        lstm_output: (batch_size * sequence_length, hidden_dim)  是 Core（主干 LSTM）每个时间步的输出——整个模型的"局势记忆"：它把当前帧的三路信息（标量统计、实体、地图）与过去所有帧的记忆（hidden_state）融合成一个向量，作为后续所有动作 head 的总出发点
        various_observations,：是专门喂给 5 个 Baseline（Critic）的“标量观测包”——各含 8 项特征：从主模型的 16 项标量状态里挑出 7 项与“局势估值”相关的（资源、升级、部队构成、建造顺序等），再补一项 Critic 专属的 cumulative_score（游戏累计积分，人类玩家都看不到）。前者来自我方视角的观测，后者来自对方视角；打内置电脑时对方观测拿不到，传 None（内部用全零代替）
        opponent_observations：是专门喂给 5 个 Baseline（Critic）的“标量观测包”——各含 8 项特征：从主模型的 16 项标量状态里挑出 7 项与“局势估值”相关的（资源、升级、部队构成、建造顺序等），再补一项 Critic 专属的 cumulative_score（游戏累计积分，人类玩家都看不到）。前者来自我方视角的观测，后者来自对方视角；打内置电脑时对方观测拿不到，传 None（内部用全零代替）
        '''
        # check and improve it
        # 返回一个混合了对局势评价的特征提取后的特征矩阵
        #【batch，512】
        player_scalar_out = self.pre_forward(various_observations)
        del various_observations

        # AlphaStar: The baseline extracts those same observations from `opponent_observations`.
        if opponent_observations is not None:
            # 对手的局势信息（是对手自己的视角数据），私有数据，方便评分
            opponenet_scalar_out = self.pre_forward(opponent_observations)
            del opponent_observations
        else:
            # 如果没有传递则对手的数据为空
            opponenet_scalar_out = torch.zeros_like(player_scalar_out)

        # AlphaStar: These features are all concatenated together to yield `action_type_input`
        # 拼接历史局势的整条记忆，玩家视角的局势特征提取矩阵，对手视角的特征提取矩阵
        action_type_input = torch.cat([lstm_output, player_scalar_out, opponenet_scalar_out], dim=1)
        del lstm_output, player_scalar_out, opponenet_scalar_out
        print("action_type_input.shape:", action_type_input.shape) if debug else None

        # AlphaStar: passed through a linear of size 256
        # 对整个局势的特征矩阵进行特征缓和提取
        # 特征压缩
        x = self.embed_fc(action_type_input)
        del action_type_input
        print("x.shape:", x.shape) if debug else None

        # AlphaStar: then passed through 16 ResBlocks with 256 hidden units and layer normalization,
        # Note: tje layer normalization is already inside each resblock
        x = x.unsqueeze(-1)
        # 特征融合
        for resblock in self.resblock_stack:
            x = resblock(x)
        x = x.squeeze(-1)    
        print("x.shape:", x.shape) if debug else None

        # AlphaStar: passed through a ReLU
        x = F.relu(x)

        # AlphaStar: then passed through a linear with 1 hidden unit.
        baseline = self.out_fc(x)
        # 到这里完成了特征的提取
        print("baseline:", baseline) if debug else None

        # AlphaStar: This baseline value is transformed by ((2.0 / PI) * atan((PI / 2.0) * baseline)) and is used as the baseline value
        # 这是 Baseline（Critic）输出的“软压缩”变换——一个 S 形函数，把 Linear 层的无界输出压进 (−1, 1) 区间
        # 具体看md文档
        out = (2.0 / np.pi) * torch.atan((np.pi / 2.0) * baseline)
        print("out:", out) if debug else None

        del x, baseline

        # 返回每个样本的评价值，范围-1～1之间
        return out


def test():
    base_line = Baseline()

    batch_size = 2
    # dummy scalar list
    scalar_list = []

    agent_statistics = torch.ones(batch_size, SFS.agent_statistics)
    upgrades = torch.randn(batch_size, SFS.upgrades)
    unit_counts_bow = torch.randn(batch_size, SFS.unit_counts_bow)
    units_buildings = torch.randn(batch_size, SFS.units_buildings)
    effects = torch.randn(batch_size, SFS.effects)
    upgrade = torch.randn(batch_size, SFS.upgrade)
    beginning_build_order = torch.randn(batch_size, SCHP.count_beginning_build_order, 
                                        int(SFS.beginning_build_order / SCHP.count_beginning_build_order))

    cumulative_score = torch.ones(batch_size, SFS.cumulative_score)
    scalar_list.append(agent_statistics)
    scalar_list.append(upgrades)
    scalar_list.append(unit_counts_bow)
    scalar_list.append(units_buildings)
    scalar_list.append(effects)
    scalar_list.append(upgrade)
    scalar_list.append(beginning_build_order)
    scalar_list.append(cumulative_score)

    opponenet_scalar_out = scalar_list

    lstm_output = torch.ones(batch_size, AHP.lstm_hidden_dim)

    out = base_line.forward(lstm_output, scalar_list, opponenet_scalar_out)

    print("out:", out) if debug else None
    print("out.shape:", out.shape) if debug else None

    if debug:
        print("This is a test!")
