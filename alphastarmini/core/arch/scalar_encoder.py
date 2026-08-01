#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Scalar Encoder."

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib.alphastar_transformer import Transformer

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

from alphastarmini.lib import utils as L

from alphastarmini.third import alphastar_available_actions as AAA

__author__ = "Ruo-Ze Liu"

debug = False


class ScalarEncoder(nn.Module):
    '''    
    Inputs: scalar_features, entity_list
    Outputs:
        embedded_scalar - A 1D tensor of embedded scalar features
        scalar_context - A 1D tensor of certain scalar features we want to use as context for gating later
    '''

    # assume we have most for some minutes game
    use_positional_encoding_for_time = AHP.positional_encoding_time

    # the embedding_size should be 32 now
    # we use both positional encoding and binary econding for time
    time_embedding_size = int(SFS.time / 2)

    if use_positional_encoding_for_time:
        max_game_seconds = 30 * 60  # assume we have max 30 minutes games
        time_encoding_all = L.positional_encoding(max_position=max_game_seconds, embedding_size=time_embedding_size, add_batch_dim=False)

    use_human_knowledge_for_available_actions = True

    def __init__(self, n_statistics=10, n_upgrades=SFS.upgrades, 
                 n_action_num=SFS.available_actions, n_units_buildings=SFS.unit_counts_bow, 
                 n_effects=SFS.effects, n_upgrade=SFS.upgrade,
                 n_possible_actions=SFS.last_action_type, 
                 n_delay=SFS.last_delay,
                 n_possible_values=SFS.last_repeat_queued,
                 original_32=AHP.original_32,
                 original_64=AHP.original_64,
                 original_128=AHP.original_128,
                 original_256=AHP.original_256,
                 original_512=AHP.original_512):
        super().__init__()
        self.statistics_fc = nn.Linear(n_statistics, original_64)  # with relu
        self.home_race_fc = nn.Linear(5, original_32)  # with relu, also goto scalar_context
        self.away_race_fc = nn.Linear(5, original_32)  # with relu, also goto scalar_context
        self.upgrades_fc = nn.Linear(n_upgrades, original_128)  # with relu
        self.enemy_upgrades_fc = nn.Linear(n_upgrades, original_128)  # with relu
        self.time_fc = original_64  # a transformer positional encoder

        # additional features
        self.available_actions_fc = nn.Linear(n_action_num, original_64)  # with relu, also goto scalar_context
        self.unit_counts_bow_fc = nn.Linear(n_units_buildings, original_64)  # A bag-of-words unit count from `entity_list`, with relu
        self.mmr_fc = nn.Linear(7, original_64)  # mmr is from 0 to 6 (by divison by 1000), with relu

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
                                                             dropout=0.)  # make dropout=0 to make training and testing consistent
        # [20, num_entity_types], into transformer with q,k,v, also goto scalar_context
        self.last_delay_fc = nn.Linear(n_delay, original_64)  # with relu
        self.last_action_type_fc = nn.Linear(n_possible_actions, original_128)  # with relu
        self.last_repeat_queued_fc = nn.Linear(n_possible_values, original_256)  # with relu

        self.fc_1 = nn.Linear(AHP.scalar_encoder_fc1_input, original_512)  # with relu
        self.fc_2 = nn.Linear(AHP.scalar_encoder_fc2_input, original_512)  # with relu

        self.relu = nn.ReLU()

    @classmethod
    def preprocess_numpy(cls, obs, build_order=None, last_list=None):
        scalar_list = []

        player = obs["player"]
        print('player:', player) if debug else None

        # The first is player_id, so we don't need it.
        player_statistics = player[1:]
        print('player_statistics:', player_statistics) if debug else None

        agent_statistics = np.array(player_statistics, dtype=np.float32).reshape(1, -1)
        print('agent_statistics:', agent_statistics) if debug else None

        home_race = np.zeros((1, 5))
        if "home_race_requested" in obs:
            home_race_requested = obs["home_race_requested"].item()
        else:
            home_race_requested = 0
        home_race[0, home_race_requested] = 1

        away_race = np.zeros((1, 5))
        if "away_race_requested" in obs:
            away_race_requested = obs["away_race_requested"].item()
        else:
            away_race_requested = 0
        away_race[0, away_race_requested] = 1

        if "action_result" in obs:
            action_result = obs["action_result"]
            print('action_result:', action_result) if debug else None

        if "alerts" in obs:
            alerts = obs["alerts"]
            print('alerts:', alerts) if debug else None

        # implement the upgrades
        upgrades = np.zeros((1, SFS.upgrades))
        obs_upgrades = obs["upgrades"]
        print('obs_upgrades:', obs_upgrades) if debug else None

        for u in obs_upgrades:
            assert u >= 0 
            assert u < SFS.upgrades
            upgrades[0, u] = 1

        # question: how to know enemy's upgrades?
        # TODO: implment the enemy upgrades
        enemy_upgrades = np.zeros((1, SFS.upgrades))

        # time conver to gameloop
        time = np.zeros((1, SFS.time))
        game_loop = obs["game_loop"]
        print('game_loop:', game_loop) if debug else None

        if cls.use_positional_encoding_for_time:
            embedding_size = cls.time_embedding_size

            # transform game_loop to 32 binary vecoter
            time_encoding_1 = L.unpackbits_for_largenumber(game_loop, num_bits=embedding_size).astype(np.float32).reshape(1, -1)
            print('time_encoding_1:', time_encoding_1) if debug else None 

            # note, we use binary encoding here for half of time encoding
            time[0, :embedding_size] = time_encoding_1

            # a second is 22.4 game_loop
            seconds = int(game_loop / 22.4)
            seconds = min(cls.max_game_seconds - 1, seconds)

            # transform seconds to positional encdoing
            time_encoding_2 = (cls.time_encoding_all[seconds]).astype(np.float32).reshape(1, -1)
            print('time_encoding_2:', time_encoding_2) if debug else None 

            # note, we use postionoal encoding here for half of time encoding
            time[0, embedding_size:] = time_encoding_2
            del time_encoding_1, time_encoding_2
        else:
            # transform game_loop to 64 binary vecoter
            time_encoding = L.unpackbits_for_largenumber(game_loop, num_bits=SFS.time).astype(np.float32).reshape(1, -1)
            print('time_encoding:', time_encoding) if debug else None 

            # note, we use binary encoding here for all of time encoding
            time = time_encoding
            del time_encoding

        # note: if we use raw action, this key doesn't exist
        # the_available_actions = obs["available_actions"] 
        # print('the_available_actions:', the_available_actions) if debug else None
        available_actions = np.ones((1, SFS.available_actions))
        if cls.use_human_knowledge_for_available_actions:
            available_actions = AAA.get_available_actions_raw_data(obs)

        # implement the unit_counts_bow
        unit_counts_bow = L.calculate_unit_counts_bow_numpy(obs)
        print('unit_counts_bow:', unit_counts_bow) if debug else None
        print('torch.sum(unit_counts_bow):', np.sum(unit_counts_bow)) if debug else None

        # implement the build order
        beginning_build_order = np.zeros((1, SCHP.count_beginning_build_order, 
                                          int(SFS.beginning_build_order / SCHP.count_beginning_build_order)))
        print('beginning_build_order.shape:', beginning_build_order.shape) if debug else None

        if build_order is not None:
            # implement the beginning_build_order       
            # TODO: add the entities pos        
            for i, bo in enumerate(build_order):
                if i < SCHP.count_beginning_build_order:
                    assert bo < SFS.unit_counts_bow
                    beginning_build_order[0, i, bo] = 1
                else:
                    break

            print("beginning_build_order:", beginning_build_order) if debug else None
            print("sum(beginning_build_order):", np.sum(beginning_build_order).item()) if debug else None

        mmr = np.zeros((1, SFS.mmr))

        # implment it
        units_buildings = L.calculate_unit_buildings_numpy(obs)

        # implement the effects
        effects = np.zeros((1, SFS.effects))

        # we now use feature_effects to represent it
        feature_effects = obs["feature_effects"]
        print('feature_effects:', feature_effects) if debug else None
        for effect in feature_effects:
            e = effect.effect_id
            assert e >= 0 
            assert e < SFS.effects
            effects[0, e] = 1

        # the raw effects are reserved for use
        raw_effects = obs["raw_effects"]
        print('raw_effects:', raw_effects) if debug else None

        # implement the upgrade
        upgrade = np.zeros((1, SFS.upgrades))
        for u in obs_upgrades:
            assert u >= 0 
            assert u < SFS.upgrades
            upgrade[0, u] = 1

        last_delay = np.zeros((1, SFS.last_delay))
        last_action_type = np.zeros((1, SFS.last_action_type))
        last_repeat_queued = np.zeros((1, SFS.last_repeat_queued))

        if last_list is not None:
            [last_delay_value, last_action_type_value, last_repeat_queued_value] = last_list
            last_delay_value = min(SFS.last_delay - 1, last_delay_value)
            last_delay[0, last_delay_value] = 1

            assert last_action_type_value < SFS.last_action_type

            last_action_type[0, last_action_type_value] = 1
            last_repeat_queued[0, last_repeat_queued_value] = 1

            print('last_delay', last_delay) if debug else None
            print('last_action_type', last_action_type) if debug else None
            print('last_repeat_queued', last_repeat_queued) if debug else None

        # note: if we use raw action, this property is always empty
        last_actions = obs["last_actions"]
        print('last_actions:', last_actions) if debug else None

        scalar_list.append(agent_statistics)
        scalar_list.append(home_race)
        scalar_list.append(away_race)
        scalar_list.append(upgrades)
        scalar_list.append(enemy_upgrades)
        scalar_list.append(time)

        scalar_list.append(available_actions)
        scalar_list.append(unit_counts_bow)
        scalar_list.append(mmr)
        scalar_list.append(units_buildings)
        scalar_list.append(effects)
        scalar_list.append(upgrade)

        scalar_list.append(beginning_build_order)
        scalar_list.append(last_delay)
        scalar_list.append(last_action_type)
        scalar_list.append(last_repeat_queued)

        del agent_statistics, home_race, away_race, upgrades, enemy_upgrades, time
        del available_actions, unit_counts_bow, mmr, units_buildings, effects
        del beginning_build_order, last_delay, last_action_type, last_repeat_queued

        return scalar_list

    def forward(self, scalar_list):
        '''
        scalar_list: 标量列表，在本项目中主要是传入游戏的资源信息：金钱、能源、人口等-玩家自身经济/军事统计
        '''
        # 提取各种资源信息，具体看md文档
        [agent_statistics, home_race, away_race, upgrades, enemy_upgrades, time, available_actions, unit_counts_bow,
         mmr, units_buildings, effects, upgrade, beginning_build_order, last_delay, last_action_type,
         last_repeat_queued] = scalar_list

        # todo
        embedded_scalar_list = [] # 将每个标量资源提取映射到emb后存储到该列表中
        scalar_context_list = [] # 精选出来的信息，
        '''
        home_race	种族决定了可用兵种/建筑池。你是 Protoss，就不能造机枪兵。门控可以屏蔽掉其他种族专属的动作特征
        away_race	对手种族影响战术选择。对面是 Zerg 时防一波狗/毒爆相关动作应增强
        available_actions	最直接的约束——你连星门都没有，"造虚空"这个动作维度就应该被压到几乎不可能
        units_buildings	你造过什么建筑？有星门→空军可选。累积建造历史塑造了你的科技树路线
        effects	当前状态效果。闪烁在冷却？那么闪烁动作相关维度降权。被攻击中？撤回/防御动作升权
        upgrade	研究过什么科技（累积）。有"冲锋"科技→狂热者的冲锋相关战术可选
        beginning_build_order	开局路线。前 3 个建筑是"双气开"还是"速二基地"→完全不同的中期动作分布

        主要是用来对AI的可用的动作空间进行约束，避免选择一些无效的动作
        
        '''

        # agent_statistics: Embedded by taking log(agent_statistics + 1) and passing through a linear of size 64 and a ReLU
        # 玩家自身经济/军事统计，10 维向量，对应 SC2 的核心经济军事指标：矿物、瓦斯、已用人口/总人口、农民数、战斗单位数、总单位数等
        # 神经网络的线性层本质是矩阵乘法 Wx + b，它期望输入在一个稳定、有界的范围内（理想是均值 ≈ 0、标准差 ≈ 1，或至少别差几个数量级）。
        # 由于这些维度的值非常大，需要用torch log将值下降到可用使用的大小
        # 因为 log(0) = -∞（负无穷大），而游戏中很多指标的初始值就是 0（瓦斯、某些统计）。+ 1 确保10个维度里面避免有维度是0导致计算出错
        the_log_statistics = torch.log(agent_statistics + 1)
        x = F.relu(self.statistics_fc(the_log_statistics)) # 提取到玩家自身经济/军事统计的特征
        del agent_statistics, the_log_statistics
        embedded_scalar_list.append(x)

        # race: Both races are embedded into a one-hot with maximum 5, and embedded through a linear of size 32 and a ReLU.
        # 获取自己种族嵌入编码
        x = F.relu(self.home_race_fc(home_race.float()))
        del home_race
        embedded_scalar_list.append(x)
        # The embedding is also added to `scalar_context`.
        scalar_context_list.append(x)

        # race: Both races are embedded into a one-hot with maximum 5, and embedded through a linear of size 32 and a ReLU.
        # 获取对手种族的嵌入编码
        x = F.relu(self.away_race_fc(away_race.float()))
        del away_race
        # TODO: During training, the opponent's requested race is hidden in 10% of matches, to simulate playing against the Random race.
        embedded_scalar_list.append(x)
        # The embedding is also added to `scalar_context`.
        scalar_context_list.append(x)
        # TODO: If we don't know the opponent's race (either because they are random or it is hidden), 
        # we add their true race to the observation once we observe one of their units.

        # upgrades: The boolean vector of whether an upgrade is present is embedded through a linear of size 128 and a ReLU
        # 当前已研究的科技升级嵌入编码
        x = F.relu(self.upgrades_fc(upgrades))
        del upgrades
        embedded_scalar_list.append(x)

        # enemy_upgrades: Embedded the same as upgrades
        # 敌人已研究的科技升级嵌入编码
        x = F.relu(self.enemy_upgrades_fc(enemy_upgrades))
        del enemy_upgrades
        embedded_scalar_list.append(x)

        # time: A transformer positional encoder encoded the time into a 1D tensor of size 64
        # do it in the preprocess
        # 游戏时间，不经过线性层，而是用两种编码拼接直接产出 64 维
        #  前 32 维：game_loop 的二进制编码（比特位展开），后 32 维：game_loop/22.4 转为秒数后做positional encoding
        # 模型需要知道"现在是第一分钟还是第二十分钟"，这决定开局、中期、后期的策略完全不同
        x = time
        embedded_scalar_list.append(x)

        # available_actions: From `entity_list`, we compute which actions may be available and which can never be available. 
        # For example, the agent controls a Stalker and has researched the Blink upgrade, 
        # then the Blink action may be available (even though in practice it may be on cooldown). 
        # The boolean vector of action availability is passed through a linear of size 64 and a ReLU.
        # 获取当前当前可执行的动作掩码的嵌入，(batch, Actions_Size) — 约 500+ 维
        # 布尔向量，每位对应一种动作类型（如"造狂热者""闪烁""造星门"），1 = 当前理论上可执行
        # 代码使用 use_human_knowledge_for_available_actions=True，即基于人类知识判断动作可用性（如：有追猎且有闪烁科技 → 闪烁动作标记为可用，即使正在冷却中），而非依赖游戏 API 的实时状态
        x = F.relu(self.available_actions_fc(available_actions))
        del available_actions
        embedded_scalar_list.append(x)
        # The embedding is also added to `scalar_context`
        scalar_context_list.append(x)

        # unit_counts_bow: A bag-of-words unit count from `entity_list`. 
        # The unit count vector is embedded by square rooting, passing through a linear layer, and passing through a ReLU
        # note make sure unit_counts_bow all >= 0, otherwise torch.sqrt will produce nan !

        print('unit_counts_bow', unit_counts_bow) if debug else None
        assert (unit_counts_bow >= 0).all()

        #       # 场上各单位类型的数量统计，(batch, All_Units_Size) — 约 200+ 维，对应所有可能的单位/建筑类型
        # 	每个维度的值 = 场上该类型单位的当前数量（如：有 3 个狂热者、1 个星门）
        # 先 sqrt(x) 压缩大值，再过 Linear(All_Units_Size, 64) + ReLU
        unit_counts_bow = torch.sqrt(unit_counts_bow)
        x = F.relu(self.unit_counts_bow_fc(unit_counts_bow))
        del unit_counts_bow
        embedded_scalar_list.append(x)

        # mmr: During supervised learning, this is the MMR of the player we are trying to imitate. Elsewhere, this is fixed at 6200. 
        # MMR is mapped to a one-hot of min(mmr / 1000, 6) with maximum 6, then passed through a linear of size 64 and a ReLU
        # 匹配分（技能等级）MMR（Match Making Rating）的 one-hot 编码。MMR 除以 1000 后取整，cap 在 6。所以 0 代表 0-999 分，6 代表 6000+ 分
        # 编码的是被模仿的人类玩家的 MMR（让模型学会不同水平的风格）
        x = F.relu(self.mmr_fc(mmr))
        del mmr
        embedded_scalar_list.append(x)

        # cumulative_statistics: The cumulative statistics (including units, buildings, effects, and upgrades) are preprocessed 
        # into a boolean vector of whether or not statistic is present in a human game. 
        # That vector is split into 3 sub-vectors of units/buildings, effects, and upgrades, 
        # and each subvector is passed through a linear of size 32 and a ReLU, and concatenated together.
        # The embedding is also added to `scalar_context`
        #  累计建造过的单位/建筑（累积统计），(batch, All_Units_Size)，布尔向量，每位 = 1 表示曾建造过该类型的单位或建筑（即使已被摧毁）
        # 告诉模型"我曾经造过星门 → 我可能有空军路线"，这是对全局战略走向的刻画
        x = F.relu(self.units_buildings_fc(units_buildings))
        del units_buildings
        embedded_scalar_list.append(x)
        scalar_context_list.append(x)

        # 当前活跃的状态效果
        # (batch, Effects_Size) — 约 200+ 维
        # 布尔向量，每位 = 1 表示该效果当前在场上存在。效果包括：被攻击警告、隐形状态、闪烁冷却、护盾回复、兴奋剂效果等
        x = F.relu(self.effects_fc(effects))
        del effects
        embedded_scalar_list.append(x)
        scalar_context_list.append(x)

        # 累计研究过的科技（累积统计）
        # (batch, Upgrades_Size)
        # 与 upgrades（第 4 个）内容相同，都是研究过的科技，但被归入"累积统计"组（与 units_buildings、effects 一起），走不同的处理通道
        x = F.relu(self.upgrade_fc(upgrade))
        del upgrade
        embedded_scalar_list.append(x)
        scalar_context_list.append(x)

        # beginning_build_order: The first 20 constructed entities are converted to a 2D tensor of size 
        # [20, num_entity_types], concatenated with indices and the binary encodings 
        # (as in the Entity Encoder) of where entities were constructed (if applicable). 
        # The concatenation is passed through a transformer similar to the one in the entity encoder, 
        # but with keys, queries, and values of 8 and with a MLP hidden size of 32. 
        # The embedding is also added to `scalar_context`.
        print("beginning_build_order:", beginning_build_order) if debug else None
        print("beginning_build_order.shape:", beginning_build_order.shape) if debug else None

        # beginning_build_order — 前 20 个建造顺序
        # (batch, 20, All_Units_Size)
        # 游戏的前 20 个建造动作，每个位置 one-hot 表示造了什么（第 1 个造了农民，第 2 个造了水晶塔……）
        batch_size = beginning_build_order.shape[0]

        # 这里是创建一个位置序号，为每一个位置序号创建one-hot编码
        seq = torch.arange(SCHP.count_beginning_build_order)
        seq = L.tensor_one_hot(seq, SCHP.count_beginning_build_order) # 将序列转换为one-hot编码，shape is （SCHP.count_beginning_build_order，SCHP.count_beginning_build_order）
        # seq.unsqueeze(0)：（1，  SCHP.count_beginning_build_order，SCHP.count_beginning_build_order）
        # .repeat(batch_size, 1, 1)：（batch_size，  SCHP.count_beginning_build_order，SCHP.count_beginning_build_order）
        seq = seq.unsqueeze(0).repeat(batch_size, 1, 1).to(beginning_build_order.device) # 

        # 为前 20 个建造步骤创建一个 padding 掩码（mask）。不是每局游戏都恰好有 20 个建造动作——游戏刚开始时可能只造了 3 个东西，剩下 17 个位置是空的（全零填充）。这个 mask 告诉后续的 Transformer："哪些位置有真实数据，哪些是 padding，处理时请忽略 padding。
        bo_sum = beginning_build_order.sum(dim=-1, keepdim=False) # bo_sum: (batch, 20)，每个位置是一个 one-hot 向量（只有一个 1，其余为 0）。求和后有真实建筑的不为0，有建筑的为0
        bo_sum = bo_sum.sum(dim=-1, keepdim=False) # 沿最后一个维度（20 个位置）求和，shape is (batch,)，将每个位置的 0/1 加起来，得到该样本一共有多少个真实的建造动作。
        bo_sum = bo_sum.unsqueeze(1) # shape is (batch, 1),举例：[5, 12, 0] → [[5], [12], [0]]
        bo_sum = bo_sum.repeat(1, SCHP.count_beginning_build_order) # 操作: 沿维度 1 重复 count_beginning_build_order 次, shape is (batch, 20)

        mask = torch.arange(SCHP.count_beginning_build_order) # mask shape (SCHP.count_beginning_build_order,) 序列
        # mask.unsqueeze(0): (1, SCHP.count_beginning_build_order)
        # .repeat(batch_size, 1)：(batch_size, SCHP.count_beginning_build_order)
        mask = mask.unsqueeze(0).repeat(batch_size, 1).to(bo_sum.device)
        # 通过这种对比的方式，按照每个样本已经建筑建筑的数量，获取有效建筑的bool掩码
        # (batch_size, SCHP.count_beginning_build_order) 1的位置表示有效建筑，0的位置表示padding
        mask = mask < bo_sum
        # mask.unsqueeze(2)：(batch_size, SCHP.count_beginning_build_order， 1) 
        # .repeat(1, 1, SCHP.count_beginning_build_order)：(batch_size, SCHP.count_beginning_build_order， SCHP.count_beginning_build_order) 
        mask = mask.unsqueeze(2).repeat(1, 1, SCHP.count_beginning_build_order)

        # add the seq info, referenced by the processing way of DI-star
        # 后面将每一个顺序及按照的建筑类型，加上序号的one-hot编码进行拼接，可能是增加位置顺序信息
        x = torch.cat([beginning_build_order, seq], dim=2)
        # 提取前20个建造建筑的信息嵌入编码
        x = self.before_beginning_build_order(x)

        # like in entity encoder, we add a sequence mask
        # 这里实在计算开头前20个建筑中每个建筑的互相影响如何，估计这个必要重要
        # 不过感觉也可以计算出历史所有建筑的互相影响情况
        x = self.beginning_build_order_transformer(x, mask=mask)
        # x shape is (batch, SCHP.count_beginning_build_order * self.build_order_model_size)
        x = x.reshape(x.shape[0], SCHP.count_beginning_build_order * self.build_order_model_size)

        embedded_scalar_list.append(x)
        scalar_context_list.append(x)
        del mask, bo_sum, seq

        # last_delay: The delay between when we last acted and the current observation, in game steps. 
        # This may be different from what we requested due to network latency or APM limits. 
        # It is encoded into a one-hot with maximum 128 and passed through a linear of size 64 and a ReLU
        # last_delay shape is (batch, 128),one-hot 编码，表示上次发出动作到现在经过了多少个 game_step
        # 由于网络延迟和 APM 限制，玩家不能每帧都发指令。这个特征告诉模型"我多久没动作了"，影响动作时机判断
        x = F.relu(self.last_delay_fc(last_delay))
        del last_delay
        embedded_scalar_list.append(x)

        # last_action_type: The last action type is encoded into a one-hot with maximum equal 
        # to the number of possible actions, and passed through a linear of size 128 and a ReLU
        # last_action_type shape is (batch, Actions_Size) — 约 500+ 维
        # one-hot 编码，表示上一次执行的动作是什么类型（移动、攻击、建造、闪烁……）
        x = F.relu(self.last_action_type_fc(last_action_type))
        del last_action_type
        embedded_scalar_list.append(x)

        # last_repeat_queued: Some other action arguments (queued and repeat) are one-hots with 
        # maximum equal to the number of possible values for those arguments, 
        # and jointly passed through a linear of size 256 and ReLU
        # last_repeat_queued shape is 	(batch, 2)
        # 两个布尔值的 one-hot 表示：
        # • queued：上次动作是否以 shift+click 队列方式发出
        # • repeat：上次动作是否是重复指令（如按住快捷键连发）
        x = F.relu(self.last_repeat_queued_fc(last_repeat_queued))
        del last_repeat_queued
        embedded_scalar_list.append(x)

        # 将所有的游戏的资源信息嵌入信息组合起来，并提取特征
        embedded_scalar = torch.cat(embedded_scalar_list, dim=1)
        embedded_scalar_out = F.relu(self.fc_1(embedded_scalar))

        scalar_context = torch.cat(scalar_context_list, dim=1)
        scalar_context_out = F.relu(self.fc_2(scalar_context))

        del x, embedded_scalar_list, scalar_context_list, embedded_scalar, scalar_context

        # embedded_scalar_out 所有游戏资源信息提取后的信息
        # scalar_context_list 是从 16 个标量特征中精选出来的 7 个，专门用于在 ActionTypeHead 中做"门控"——它们不参与 LSTM 的序列记忆，而是像一个"条件开关"，根据游戏上下文动态调节动作选择。
        return embedded_scalar_out, scalar_context_out


def test(debug=False):

    scalar_encoder = ScalarEncoder()

    batch_size = 2
    # dummy scalar list
    scalar_list = []

    agent_statistics = torch.ones(batch_size, SFS.agent_statistics)
    home_race = torch.randn(batch_size, SFS.home_race)
    away_race = torch.randn(batch_size, SFS.away_race)
    upgrades = torch.randn(batch_size, SFS.upgrades)
    enemy_upgrades = torch.randn(batch_size, SFS.upgrades)
    time = torch.randn(batch_size, SFS.time)

    available_actions = torch.randn(batch_size, SFS.available_actions)
    unit_counts_bow = torch.ones(batch_size, SFS.unit_counts_bow)
    mmr = torch.randn(batch_size, SFS.mmr)
    units_buildings = torch.ones(batch_size, SFS.units_buildings)
    effects = torch.randn(batch_size, SFS.effects)
    upgrade = torch.randn(batch_size, SFS.upgrade)

    beginning_build_order = torch.randn(batch_size, SCHP.count_beginning_build_order, 
                                        int(SFS.beginning_build_order / SCHP.count_beginning_build_order))
    last_delay = torch.randn(batch_size, SFS.last_delay)
    last_action_type = torch.randn(batch_size, SFS.last_action_type)
    last_repeat_queued = torch.randn(batch_size, SFS.last_repeat_queued)

    scalar_list.append(agent_statistics)
    scalar_list.append(home_race)
    scalar_list.append(away_race)
    scalar_list.append(upgrades)
    scalar_list.append(enemy_upgrades)
    scalar_list.append(time)

    scalar_list.append(available_actions)
    scalar_list.append(unit_counts_bow)
    scalar_list.append(mmr)
    scalar_list.append(units_buildings)
    scalar_list.append(effects)
    scalar_list.append(upgrade)

    scalar_list.append(beginning_build_order)
    scalar_list.append(last_delay)
    scalar_list.append(last_action_type)
    scalar_list.append(last_repeat_queued)

    embedded_scalar, scalar_context = scalar_encoder.forward(scalar_list)

    print("embedded_scalar:", embedded_scalar) if debug else None
    print("embedded_scalar.shape:", embedded_scalar.shape) if debug else None

    print("scalar_context:", scalar_context) if debug else None
    print("scalar_context.shape:", scalar_context.shape) if debug else None

    if debug:
        print("This is a test!")
