#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Computing the pseudo-reward (based on distance between the agent and a human player)."

# code for computing the pseudo-reward

import time
import random

import numpy as np

import torch

import Levenshtein

from alphastarmini.core.rl import rl_utils as U
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
import alphastarmini.lib.edit_distance as ED


__author__ = "Ruo-Ze Liu"

debug = False


def list2str(l):
    # note: the maximus accept number for chr() is 1114111, else raise a ValueError
    return ''.join([chr(int(i)) for i in l])


def reward_by_build_order(bo, z_bo, gl=0):
    str1 = list2str(bo)
    str2 = list2str(z_bo)

    # 计算两个字符之间的距离，作为AI和人类之间建造建筑顺序的差异距离
    dist = Levenshtein.distance(str1, str2)
    print('dist:', dist) if debug else None

    # the cost is the squared distance between the 
    # true built entity and the agent's entity
    # 一个非负、无上界的数。距离越远，它平方后爆炸得越快（dist=10 → 100，dist=20 → 400）
    reward = dist * dist # 如果差距越大则reward越大

    # rescaled to be within [0, 0.8] with the maximum of 0.8 when it is more than two gateways away.
    # Question: how to do it ?
    # 这两行在做奖励的“封顶 + 归一化 + 变符号”：把一个可能无界的平方距离，压缩到 [-0.8, 0] 这个固定区间内的惩罚值。
    # 为什么要截断（第 ① 步）：如果智能体的建造顺序和人类差得离谱（比如乱玩一通），dist² 可能是几百上千。不封顶的话，这一个奖励信号会在 loss 里占巨大比重，梯度被它主导，训练就崩了。封顶的含义是：“差得离谱”和“差得很远”受到的惩罚一样——反正都不像人类，没必要区分 50 分和 500 分的烂。
    # 为什么上限是 0.8 而不是 1.0（第 ③ 步）：这对应 AlphaStar 论文的原文描述 "rescaled to be within [0, 0.8]"。目的是量级协调——输赢奖励（winloss）通常是 ±1，这个模仿惩罚的最大量级被刻意压到 0.8，保证“像不像人类”这个中间奖励不会盖过“赢没赢”这个最终目标。
    reward = min(reward, 50) / 50.0 * 0.8
    # 把“距离”（越大越糟）变成“奖励”（越大越好）
    # 距离为 0（和人类建造顺序完全一样）→ reward = 0（不惩罚）
    # 距离越大 → 负得越多（惩罚越重）
    reward = -reward
    print('reward:', reward) if debug else None

    scale = time_decay_scale(game_loop=gl)
    # note, build_order seems not to use time decay scale

    return reward


def reward_by_unit_counts(ucb, z_ucb, gl=0):
    str1 = list2str(ucb)
    str2 = list2str(z_ucb)
    # 也是通过建造单位的名字转换为字符串后通过汉明码计算两者的距离
    dist = Levenshtein.hamming(str1, str2)
    print('dist:', dist) if debug else None

    # 这是论文的设计差异 + 两种距离的性质差异共同决定的：建造顺序的错是“连锁性、战略性”的错误，值得用平方非线性放大；单位数量的差是“相互独立”的差异，线性计数就够公平了。
    # 单位数量的差异是独立的。 “狂热者少造了 2 个”和“追猎者多造了 3 个”之间没有连锁关系，就是两个各自独立的小偏差。Hamming 距离本身就是“逐位比较、差异累加”，它天然就是线性的公平计分：每多一种单位类型数量不匹配，就多扣 1 分。再平方反而会扭曲含义——“配比差 4 种”并不是比“配比差 2 种”严重 4 倍的战略失误，只是多了两类不匹配而已
    reward = - dist # 同样将奖励转至为负数，这样距离越小奖励越大，距离越大奖励越小
    print('reward:', reward) if debug else None

    # 前期要像人类，后期逐渐“放手”，24 分钟后完全不约束
    scale = time_decay_scale(game_loop=gl)
    print('scale:', scale) if debug else None

    # hamming distance use time decay scale
    return reward * scale # 这个的作用就是超过了一定的时间就不限制模型预测建造单位的类型


def time_decay_scale(game_loop):
    # game_loop: 这里应该是游戏经过的时间帧
    
    # 通过游戏的帧数计算游戏经过的时间
    time_seconds = int(game_loop / 22.4)
    time_minutes = int(time_seconds / 60)

    # 如果游戏经过的时间越久，返回来的系数就越短
    scale = 1.
    if time_minutes > 24:
        scale = 0.
    elif time_minutes > 16:
        scale = 0.25
    elif time_minutes > 8:
        scale = 0.5

    return scale


def compute_pseudoreward(trajectories, reward_name, device):
    """Computes the relevant pseudoreward from trajectories.

    See Methods and detailed_architecture.txt for details.

    Paper description:
      These pseudo-rewards measure the edit distance between sampled 
      and executed build orders, and the Hamming distance 
      between sampled and executed cumulative statistics.
    """

    # The build order reward is the negative Levenshtein distance between 
    # the true (human replay) build order and the agent's build order, except 
    # that in the case $lev_{a, b}(i - 1, j - 1) + 1_{\(a != b\)}$, 
    # instead of $1_{\(a != b\)}$ in the case where the types of the units 
    # are the same, the cost is the squared distance between the 
    # true built entity and the agent's entity, rescaled to be within [0, 0.8] 
    # with the maximum of 0.8 when it is more than two gateways away. A reward 
    # is given whenever the agent's build order changes (because 
    # it has built something new). Units that aren't built (like 
    # auto turrets or larva), worker units, and supply buildings are 
    # skipped in the build order. 

    ''' 
    my notes: Actually, there are so many ambiguous descriptions in this place, e.g., what 
    are the "two gateways away"? and why the units aren't built includes "auto turrets" (I think 
    they are built by players). Moreover, is it the supply buildings that mean the building 
    for the supplement, so, do they include "overlords" or "pylons"? The except case in the 
    calculation of Levenshtein distance is also unclear.

    '''

    # The built units reward is the Hamming distance between the entities built 
    # in some human replay and the entities the agent has built. After 8 minutes, 
    # the reward is multiplied by 0.5. After 16 minutes, the reward is multiplied 
    # by an additional 0.5. After 24 minutes, there are no more rewards.

    ''' 
    my notes: First, it seems here don't use the build order, actually, it should be seen that 
    it use the bag-of-words here. One question: is the reward be calculated in every step? Another 
    question: if the reward is calculated in each step, does it means the agent accept a negative (I 
    think the reward will be the negatived distance) reward every time until the agent builds the 
    same unit counts as the human player (which is a very strong assumption, meaning that we want 
    the agent do nearly the same as the human player). However, we can see that the restriction is 
    smaller as the time passed (e.g, after 8 minutes, the reward is halved). However, is the choice 
    of the "8" too experiential? I think here brings too much human knowledge here..

    '''

    print("reward name:", reward_name) if debug else None

    # add to use reward_name to judge to use the win loss reward
    if reward_name == 'winloss_baseline': # 仅返回真实输赢奖励
        print('trajectories.reward', trajectories.reward) if debug else None

        rewards_tensor = torch.tensor(trajectories.reward, dtype=torch.float32, device=device)
        return rewards_tensor

    # if we don't use replays to generate reward, just return the result reward
    # if trajectories.build_order[0][0] is None and trajectories.z_build_order[0][0] is None:
    #     rewards_tensor = torch.tensor(trajectories.reward, dtype=torch.float32, device=device)
    #     return rewards_tensor
    # 后续的奖励就是在计算游戏中途奖励，避免奖励稀疏的问题
    print("trajectories.build_order", trajectories.build_order) if debug else None
    print("trajectories.z_build_order", trajectories.z_build_order) if debug else None
    print("trajectories.unit_counts", trajectories.unit_counts) if debug else None
    print("trajectories.z_unit_counts", trajectories.z_unit_counts) if debug else None
    print("trajectories.game_loop", trajectories.game_loop) if debug else None

    # mAS: note we use scale to make the reward of leven and hamming not to much
    # mini版本的超参数
    scale_leven = 0.05
    scale_hamming = 0.05

    weight_leven = 1.0 * scale_leven
    weight_hamming = 1.0 * scale_hamming

    # 注意：reward_name 是每次传入单独的一个，无法同时满足多个
    # 所以以下的置零操作金存在于一条链路
    # in order to distinguish between build_order and built_units 
    if reward_name == 'build_order_baseline': # 建造顺序
        weight_hamming = 0.0 # todo 这个参数的作用
    if reward_name == 'built_units_baseline': # 建造单位
        weight_leven = 0.0    # todo 这个参数的作用

    # AlphaStar: The built units reward is the Hamming distance between the effects in some human replay 
    # and the effects the agent has created. After 8 minutes, the reward is multiplied by 0.5. 
    # After 16 minutes, the reward is multiplied by an additional 0.5. After 24 minutes, there are no more rewards.
    if reward_name == 'effects_baseline': # 这个是什么？
        # now we don't separate the effects from the unit, 
        # so we ignore this reward now
        weight_hamming = 0.0
        weight_leven = 0.0

    # AlphaStar: The built units reward is the Hamming distance between the upgrades in some human replay and the upgrades 
    # the agent has researched. After 8 minutes, the reward is multiplied by 0.5. After 16 minutes, the reward is 
    # multiplied by an additional 0.5. After 24 minutes, there are no more rewards.
    if reward_name == 'upgrades_baseline': # 这个是什么？
        # now we don't separate the upgrades from the unit, 
        # so we ignore this reward now
        weight_hamming = 0.0
        weight_leven = 0.0

    rewards_traj = [] # 这里存储的就是建筑和单位和人类是否相似的奖励
    # 根据采集的轨迹，计算伪奖励
    for t1, t2, t3, t4, t5 in zip(trajectories.build_order, trajectories.z_build_order,
                                  trajectories.unit_counts, trajectories.z_unit_counts,
                                  trajectories.game_loop):
        # Calculate the Levenshtein distance,
        leven_batch = [] # 存储建造顺序的奖励
        for b1, b2, b5 in zip(t1, t2, t5): # 计算建造顺序的奖励，对比人类开局的建造顺序
            r = reward_by_build_order(b1, b2, b5)
            leven_batch.append(r)
        print('leven_batch:', leven_batch) if debug else None

        # Calculate the Hamming distance,
        hamming_batch = [] # 存储建造单位的类型和数量是否和人类一致
        for b1, b2, b5 in zip(t3, t4, t5):
            r = reward_by_unit_counts(b1, b2, b5)
            hamming_batch.append(r)
        print('hamming_batch:', hamming_batch) if debug else None

        reward_batch = []
        for r1, r2 in zip(leven_batch, hamming_batch): # 将这两个奖励按照指定的权重合并起来
            reward_batch.append(weight_leven * r1 + weight_hamming * r2)
        print('reward_batch:', reward_batch) if debug else None

        rewards_traj.append(reward_batch)

    print('rewards_traj:', rewards_traj) if debug else None 

    rewards_numpy = np.array(rewards_traj)
    rewards_tensor = torch.tensor(rewards_numpy, dtype=torch.float32, device=device)

    # 返回创建建筑的顺序和类型以及建造单位的类型和数量是否和人类选手一致
    # 这种就是典型的为稀疏奖励构建中途奖励的方法
    return rewards_tensor


def test():
    levenshtein = ED.levenshtein_recur
    hammingDist = ED.hammingDist

    Start = 0
    Stop = SCHP.max_unit_type

    print("Stop is :", Stop) if debug else None

    limit = 5

    l_1 = []
    l_2 = []

    # random
    #[l_1.append(random.randrange(Start, Stop)) for i in range(limit)]
    #[l_2.append(random.randrange(Start, Stop)) for i in range(limit)]

    # specfic
    l_1 = [13, 23, 45, 1114111]
    l_2 = [13, 45, 1114110]

    s_1 = list2str(l_1)
    s_2 = list2str(l_2)

    print("edit distance between 'l_1', 'l_2'", Levenshtein.distance(s_1, s_2))

    # note: hamming distance need the two lists have equal length
    l_1 = [13, 23, 45, 1114111]
    l_2 = [13, 21, 45, 1114110]

    s_1 = list2str(l_1)
    s_2 = list2str(l_2)

    print("hamming distance between 'l_1', 'l_2'", Levenshtein.hamming(s_1, s_2))  

    return


if __name__ == '__main__':
    test()
