#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Library for RL losses."

# modified from AlphaStar pseudo-code
import traceback
import collections
import itertools
import gc

import numpy as np

import torch
import torch.nn as nn

from alphastarmini.core.rl.rl_utils import Trajectory
from alphastarmini.core.rl.action import ArgsActionLogits
from alphastarmini.core.rl.action import ArgsAction

from alphastarmini.core.rl import rl_algo as RA
from alphastarmini.core.rl import rl_utils as RU
from alphastarmini.core.rl import pseudo_reward as PR

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS


__author__ = "Ruo-Ze Liu"

debug = False

FIELDS_WEIGHT_1 = [1, 0, 1, 0.01, 1, 1]  # [0, 0, 0, 1, 0, 0]
FIELDS_WEIGHT_2 = [1, 0, 1, 1, 1, 1]

WINLOSS_BASELINE_COSTS = (10.0, 5.0, "winloss_baseline")

ACTION_FIELDS = [
    'action_type',  
    'delay',
    'queue',
    'units',
    'target_unit',
    'target_location',
]


def filter_by_for_lists(action_fields, target_list, device):
    return torch.cat([getattr(b, action_fields).to(device) for a in target_list for b in a], dim=0)


def filter_by_for_masks(action_fields, target_mask, device):
    return torch.tensor(target_mask, device=device)[:, :, ACTION_FIELDS.index(action_fields)]


def entropy_loss(target_logits, trajectories, selected_mask, entity_mask):
    """Computes the entropy loss for a set of logits."""

    device = target_logits.action_type.device
    mask = torch.tensor(trajectories.masks, device=device)
    unit_type_entity_mask = torch.tensor(np.array(trajectories.unit_type_entity_mask), dtype=torch.float32, device=device)
    [selected_mask, entity_mask, unit_type_entity_mask] = [x.view(-1, x.shape[-1]) for x in [selected_mask, entity_mask, unit_type_entity_mask]]

    loss = 0 
    for i, field in enumerate(ACTION_FIELDS):  
        x = get_kl_or_entropy(target_logits, field, RA.entropy, mask, selected_mask, entity_mask, unit_type_entity_mask)
        loss = loss + x * FIELDS_WEIGHT_2[i]
        del x

    del mask, unit_type_entity_mask, selected_mask, entity_mask

    return loss


def human_policy_kl_loss(target_logits, trajectories, selected_mask, entity_mask):
    """Computes the KL loss to the human policy."""

    device = target_logits.action_type.device
    mask = torch.tensor(trajectories.masks, device=device)
    unit_type_entity_mask = torch.tensor(np.array(trajectories.unit_type_entity_mask), dtype=torch.float32, device=device)
    [selected_mask, entity_mask, unit_type_entity_mask] = [x.view(-1, x.shape[-1]) for x in [selected_mask, entity_mask, unit_type_entity_mask]]

    loss = 0  
    loss_dict = {}
    for i, field in enumerate(ACTION_FIELDS):
        t_logits = filter_by_for_lists(field, trajectories.teacher_logits, device) 
        x = get_kl_or_entropy(target_logits, field, RA.kl, mask, selected_mask, entity_mask, unit_type_entity_mask, t_logits)
        x = x * FIELDS_WEIGHT_2[i]

        loss = loss + x
        loss_dict[field] = x.item()
        del x, t_logits

    del mask, unit_type_entity_mask, selected_mask, entity_mask

    return loss, loss_dict


def get_kl_or_entropy(target_logits, field, func, mask, selected_mask, entity_mask, unit_type_entity_mask, t_logits=None):
    logits = getattr(target_logits, field)
    logits = logits.view(AHP.sequence_length * AHP.batch_size, *tuple(logits.shape[2:]))
    all_logits = [logits]
    if t_logits is not None:
        all_logits.append(t_logits)

    if field == "units":
        #all_logits = [change_units_logits(logit, select_units_num, entity_nums) for logit in all_logits]
        max_selected = AHP.max_selected + 1
        all_logits = [logit.view(AHP.sequence_length * AHP.batch_size, max_selected, AHP.max_entities) for logit in all_logits]
    elif field == "target_unit":
        all_logits = [logit.view(AHP.sequence_length * AHP.batch_size * 1, AHP.max_entities) for logit in all_logits]     
    elif field == "target_location":
        all_logits = [logit.view(AHP.sequence_length * AHP.batch_size, SCHP.world_size * SCHP.world_size) for logit in all_logits]

    if field == "units":
        x = func(all_logits, selected_mask, entity_mask.unsqueeze(-2), unit_type_entity_mask.unsqueeze(-2))
        x = x.sum(dim=-2)
    else:
        if field == "target_unit":
            x = func(all_logits, entity_mask=entity_mask, unit_type_entity_mask=unit_type_entity_mask)
        else:
            x = func(all_logits)
    x = x.sum(dim=-1)
    if t_logits is None:
        x = x / torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32, device=logits.device))     # Normalize by actions available.

    mask = mask[:, :, ACTION_FIELDS.index(field)].view(-1, 1)
    x = torch.mean(x * mask)

    del logits, t_logits, all_logits, mask, selected_mask, entity_mask, unit_type_entity_mask

    return x


def human_policy_kl_loss_action(target_logits, trajectories):

    device = target_logits.action_type.device
    seconds = torch.tensor(trajectories.game_loop, device=device).view(AHP.sequence_length * AHP.batch_size) / 22.4
    flag = seconds < (4 * 60)  # the first 4 minutes
    i = 0  # index for action_type
    field = ACTION_FIELDS[i]  # action_type
    mask = torch.tensor(trajectories.masks, device=device)[:, :, i].view(-1, 1)

    logits = getattr(target_logits, field) 
    logits = logits.view(AHP.sequence_length * AHP.batch_size, *tuple(logits.shape[2:]))
    t_logits = filter_by_for_lists(field, trajectories.teacher_logits, device) 

    kl = RA.kl([logits, t_logits]).sum(dim=-1) * flag
    kl = torch.mean(kl * mask)

    del seconds, flag, mask, logits, t_logits

    return kl


def td_lambda_loss(baselines, rewards, trajectories, device): 

    # trajectories.is_final：获取的是当前轨迹是否到了游戏结束的时候，true表示结束
    # ~np.array：然后就是对结果取反，然后true就表示还没结束
    discounts = ~np.array(trajectories.is_final[:-1], dtype=np.bool)  # note, use '~' must ensure the type is bool
    discounts = torch.tensor(discounts, device=device)

    baselines = baselines # 预测的价值，bellma公式中target的Q值
    rewards = rewards[:-1]  # rewards should be T_0 -> T_{n-1}

    with torch.no_grad(): # 这里之所以加no_gard，是因为不能将梯度从这里流过去，否则模型为了能够预测对，会通过修改returns的值，从而导致预测的价值不对
        returns = RA.lambda_returns(baselines[1:], rewards, discounts, lambdas=0.8)

    print('returns', returns) if debug else None
    print('baselines[0]', baselines[0]) if debug else None

    result = returns - baselines[:-1] # 对比模型预测的价值和计算出来的回报之间的差值，从而拉近两者的距离进行训练
    print('result', result) if debug else None

    del discounts, baselines, rewards, returns

    td_lambda_loss = 0.5 * torch.mean(torch.square(result))
    print('td_lambda_loss', td_lambda_loss.item()) if debug else None

    del result

    return td_lambda_loss


def get_baseline_hyperparameters():
    # Accroding to detailed_architecture.txt, baselines contains the following 5 items:
    # Thus BASELINE_COSTS_AND_REWARDS also have 5 entry
    # winloss_baseline_costs = (1.0, 10.0, "winloss_baseline") 
    winloss_baseline_costs = WINLOSS_BASELINE_COSTS # 计算的损失仅管输赢，不加工

    # AlphaStar: The updates are computed similar to Winloss, except without UPGO, applied using Build Order baseline, and 
    # with relative weightings 4.0 for the policy and 1.0 for the baseline.
    build_order_baseline_costs = (4.0, 1.0, "build_order_baseline") # 只管建造顺序损失

    # AlphaStar: The updates are computed similar to Winloss, except without UPGO, applied using Built Units baseline, 
    # and with relative weightings 6.0 for the policy and 1.0 for the baseline.
    built_units_baseline_costs = (6.0, 1.0, "built_units_baseline") # 尽管建造单位损失

    # AlphaStar: The updates are computed similar to Winloss, except without UpGO, applied using Upgrades baseline, and 
    # with relative weightings 6.0 for the policy and 1.0 for the baseline.
    upgrades_baseline_costs = (6.0, 1.0, "upgrades_baseline") # 

    # AlphaStar: The updates are computed similar to Winloss, except without UPGO, applied using Effects baseline, and 
    # with relative weightings 6.0 for the policy and 1.0 for the baseline.
    effects_baseline_costs = (6.0, 1.0, "effects_baseline")

    BASELINE_COSTS_AND_REWARDS = [winloss_baseline_costs, build_order_baseline_costs, built_units_baseline_costs,
                                  upgrades_baseline_costs, effects_baseline_costs]

    return BASELINE_COSTS_AND_REWARDS


def transpose_target_logits(target_logits):
    # 把 rl_unroll 输出的 6 个动作字段 logits 从 [batch, seq, …]（B 在前） 换成 [seq, batch, …]（T 在前）——因为后面所有 RL 算法（λ-回报、V-trace、UPGO）都是沿着第 0 维=时间轴做递推扫描的。
    for i, field in enumerate(ACTION_FIELDS):  # 遍历模型预测的动作类型
        logits = getattr(target_logits, field) # 取出对应动作的字段，如：目标位置、选择的单位
        #logits = logits.view(AHP.batch_size, AHP.sequence_length, *tuple(logits.shape[1:]))
        logits = logits.transpose(0, 1).contiguous() # 交换SEQ和BATCH两个字段
        logits = logits.view(AHP.sequence_length, AHP.batch_size, *tuple(logits.shape[2:])) 
        setattr(target_logits, field, logits) # 重新设置回target_logits

    del logits
    return target_logits


def transpose_baselines(baseline_list):
    baselines = [baseline.view(AHP.batch_size, AHP.sequence_length).transpose(0, 1).contiguous() for baseline in baseline_list]
    return baselines


def transpose_sth(x):
    x = x.view(AHP.batch_size, AHP.sequence_length).transpose(0, 1).contiguous()
    return x


def get_useful_masks(select_units_num, entity_num, device):
    '''
    它解决的是 RL 张量化里一个躲不开的问题：SC2 的实体数量和选兵数量每一步都在变，而张量必须是定长的

    select_units_num[t,b]：第 b 条轨迹第 t 步实际选了几个兵。选兵是变长序列，但张量必须开到 max_selected（MiniStar 笔记本配置 = 48/16 = 3，服务器 = 12）；
    entity_num[t,b]：那一步场上真实实体个数（可能几十个），但实体表张量固定开 max_entities = 512（hyper_parameters.py:208 注释明确说这个值不能缩减）。
    '''
    max_selected = AHP.max_selected + 1
    extend_select_units_num = select_units_num + 1

    print('extend_select_units_num', extend_select_units_num) if 1 else None

    selected_mask = torch.arange(max_selected, device=device).float()
    selected_mask = selected_mask.repeat(AHP.sequence_length * AHP.batch_size, 1)
    selected_mask = selected_mask < extend_select_units_num.reshape(-1).unsqueeze(dim=-1)

    print('selected_mask', selected_mask) if 1 else None

    entity_mask = torch.arange(AHP.max_entities, device=device).float()
    entity_mask = entity_mask.repeat(AHP.sequence_length * AHP.batch_size, 1)
    entity_mask = entity_mask < entity_num.reshape(-1).unsqueeze(dim=-1)

    selected_mask = selected_mask.reshape(AHP.sequence_length, AHP.batch_size, max_selected)
    entity_mask = entity_mask.reshape(AHP.sequence_length, AHP.batch_size, AHP.max_entities)

    del select_units_num, entity_num

    return selected_mask, entity_mask


def sum_vtrace_loss(target_logits_all, trajectories, baselines, rewards, selected_mask, entity_mask, device):
    '''
    target_logits_all	目标策略 π 的 logits	6 个字段各不相同（见下）	是（唯一）
    trajectories	actor 采样时记录的一切	Trajectory 22 字段，[时间][batch]	否（纯数据）
    baselines	本路价值 V(s_t)	[T, B]	否（no_grad 使用）
    rewards	本路奖励流	[T, B]	否
    selected_mask	选兵序列有效槽位	[T, B, 4]（3 兵+1 停止符）	否
    entity_mask	实体表有效位置	[T, B, 512]	否
    device	计算设备	cuda:0 / cpu	—
    '''
    """Computes the split v-trace policy gradient loss."""
    print('sum_vtrace_pg_loss') if debug else None

    # 遍历 trajectories 中的每一个成员，然后取出每一个成员的最后一步，然后重新组装成一个新的 Trajectory
    trajectories = Trajectory(*tuple(item[:-1] for item in trajectories)) 
    discounts = torch.tensor(~np.array(trajectories.is_final, dtype=np.bool), dtype=torch.float32, device=device) # 又是得到 1 标识没结束；0 标识结束
    # unit_type_entity_mask：当前这步选的这个动作类型，按游戏规则允许选中场上哪些实体
    unit_type_entity_mask = torch.tensor(np.array(trajectories.unit_type_entity_mask), dtype=torch.float32, device=device)

    values = baselines[:-1]
    [rewards, selected_mask, entity_mask] = [a[:-1] for a in [rewards, selected_mask, entity_mask]]
    mask_provided = [selected_mask, entity_mask, unit_type_entity_mask]

    #fields_weight = [0, 0, 0, 0, 0, 0]
    #fields_weight = [1, 0, 1, 0, 1, 1]
    fields_weight = FIELDS_WEIGHT_1

    loss = 0.
    for i, field in enumerate(ACTION_FIELDS):
        target_log_prob, clipped_rhos, masks = get_logprob_and_rhos(target_logits_all, field, trajectories, mask_provided)
        with torch.no_grad():
            weighted_advantage = RA.vtrace_advantages(clipped_rhos, rewards, discounts, values, baselines[-1])[1].reshape(-1)

        loss_field = (-target_log_prob) * weighted_advantage * masks.reshape(-1)
        loss_field = loss_field.mean() * fields_weight[i]
        print('field', field, 'loss_val', loss_field.item()) if 1 else None

        loss = loss + loss_field 
        del loss_field, weighted_advantage, target_log_prob, clipped_rhos, masks

    del trajectories, discounts, unit_type_entity_mask, rewards
    del values, selected_mask, entity_mask, mask_provided, fields_weight

    return loss


def sum_upgo_loss(target_logits_all, trajectories, baselines, selected_mask, entity_mask, device):
    """Computes the split upgo policy gradient loss."""
    print('sum_upgo_loss') if debug else None

    trajectories = Trajectory(*tuple(item[:-1] for item in trajectories))
    discounts = torch.tensor(~np.array(trajectories.is_final, dtype=np.bool), dtype=torch.float32, device=device)
    unit_type_entity_mask = torch.tensor(np.array(trajectories.unit_type_entity_mask), dtype=torch.float32, device=device)
    reward = torch.tensor(np.array(trajectories.reward), dtype=torch.float32, device=device)

    values = baselines[:-1]
    returns = RA.upgo_returns(values, reward, discounts, baselines[-1])
    [selected_mask, entity_mask] = [a[:-1] for a in [selected_mask, entity_mask]]
    mask_provided = [selected_mask, entity_mask, unit_type_entity_mask]

    #fields_weight = [0, 0, 0, 0, 0, 0]
    #fields_weight = [1, 0, 1, 0, 1, 1]
    fields_weight = FIELDS_WEIGHT_1

    loss = 0.
    for i, field in enumerate(ACTION_FIELDS):
        target_log_prob, clipped_rhos, masks = get_logprob_and_rhos(target_logits_all, field, trajectories, mask_provided)
        with torch.no_grad():
            weighted_advantage = ((returns - values) * clipped_rhos).reshape(-1)

        loss_field = (-target_log_prob) * weighted_advantage * masks.reshape(-1)
        loss_field = loss_field.mean() * fields_weight[i]
        print('field', field, 'loss_val', loss_field.item()) if debug else None

        loss = loss + loss_field
        del loss_field, weighted_advantage, target_log_prob, clipped_rhos, masks

    del trajectories, discounts, unit_type_entity_mask, reward, fields_weight
    del values, returns, selected_mask, entity_mask, mask_provided

    return loss


def change_units_and_logits(behavior_logits, gt_units, select_units_num, entity_nums):
    [batch_size, select_size, units_size] = behavior_logits.shape
    padding = torch.zeros(batch_size, 1, units_size, dtype=behavior_logits.dtype, device=behavior_logits.device)
    token = torch.tensor(AHP.max_entities - 1, dtype=torch.long, device=padding.device)

    padding[:, 0] = L.tensor_one_hot(token, units_size).reshape(-1).float()
    behavior_logits = torch.cat([behavior_logits, padding], dim=1)
    select_units_num = select_units_num.reshape(-1).long()
    entity_nums = entity_nums.reshape(-1).long()

    # print('behavior_logits[0][1]', behavior_logits[0][1]) if 1 else None
    # print('behavior_logits.shape', behavior_logits.shape) if 1 else None

    behavior_logits[torch.arange(batch_size), -1] = (L.tensor_one_hot(entity_nums, units_size).float() - 1) * 1e9

    # print('behavior_logits[0][1]', behavior_logits[0][1]) if 1 else None
    # print('behavior_logits[0][-1]', behavior_logits[0][-1]) if 1 else None
    # print('behavior_logits.shape', behavior_logits.shape) if 1 else None

    gt_units = gt_units.long()

    padding = torch.zeros(batch_size, 1, 1, dtype=gt_units.dtype, device=gt_units.device)
    token = torch.tensor(AHP.max_entities - 1, dtype=padding.dtype, device=padding.device)
    padding[:, 0] = token

    gt_units = torch.cat([gt_units, padding], dim=1)

    # print('gt_units[0][1]', gt_units[0][1]) if 1 else None
    # print('gt_units.shape', gt_units.shape) if 1 else None

    gt_units[torch.arange(batch_size), -1] = entity_nums.unsqueeze(dim=1)

    # print('gt_units[0][1]', gt_units[0][1]) if 1 else None
    # print('gt_units.shape', gt_units.shape) if 1 else None

    # print('stop', stop)

    del padding, token, select_units_num, entity_nums

    gt_units = gt_units.long()

    return behavior_logits, gt_units


def get_logprob_and_rhos(target_logits_all, field, trajectories, mask_provided):
    target_logits = getattr(target_logits_all, field)[:-1]
    device = target_logits.device
    [selected_mask, entity_mask, unit_type_entity_mask] = mask_provided
    mask_used = [None, None, None] 

    [sequence_length, batch_size] = target_logits.shape[0:2]
    seqbatch_shape = sequence_length * batch_size
    target_logits = target_logits.view(seqbatch_shape, *tuple(target_logits.shape[2:]))

    behavior_logits = filter_by_for_lists(field, trajectories.behavior_logits, device)
    actions = filter_by_for_lists(field, trajectories.action, device)
    masks = filter_by_for_masks(field, trajectories.masks, device)
    max_selected = AHP.max_selected + 1

    all_logits = [target_logits, behavior_logits]
    show = False
    if field == "units":
        select_units_num = torch.tensor(trajectories.player_select_units_num, dtype=torch.float32, device=device)
        entity_num = torch.tensor(trajectories.entity_num, dtype=torch.float32, device=device)
        modified_behavior_logits, actions = change_units_and_logits(behavior_logits, actions, select_units_num, entity_num)
        all_logits = [target_logits, modified_behavior_logits]
        all_logits = [logit.reshape(sequence_length * batch_size, max_selected, AHP.max_entities) for logit in all_logits]
        mask_used = [selected_mask, entity_mask, unit_type_entity_mask]
        show = True
    elif field == "target_unit":
        all_logits = [logit.reshape(sequence_length * batch_size, 1, AHP.max_entities) for logit in all_logits]
        mask_used = [None, entity_mask, None]
    elif field == "target_location":
        all_logits = [logit.reshape(sequence_length * batch_size, SCHP.world_size * SCHP.world_size) for logit in all_logits]
        actions_tmp = torch.zeros(seqbatch_shape, 1, dtype=torch.int64, device=device)
        for i, pos in enumerate(actions):
            [x, y] = pos
            index = SCHP.world_size * y + x
            actions_tmp[i][0] = index
        actions = actions_tmp

    [target_logits, behavior_logits] = all_logits

    target_log_prob = RA.log_prob(target_logits, actions, mask_used, max_selected, show=show)
    behavior_log_prob = RA.log_prob(behavior_logits, actions, mask_used, max_selected, show=show)

    # print('target_log_prob', target_log_prob) if show else None
    # print('target_log_prob.shape', target_log_prob.shape) if show else None

    # print('behavior_log_prob', behavior_log_prob) if show else None
    # print('behavior_log_prob.shape', behavior_log_prob.shape) if show else None

    with torch.no_grad():
        clipped_rhos = RA.compute_cliped_importance_weights(target_log_prob, behavior_log_prob)

    if field == 'units':
        clipped_rhos = clipped_rhos.reshape(seqbatch_shape, -1).sum(dim=-1)
        # print('clipped_rhos', clipped_rhos) if show else None
        # print('clipped_rhos.shape', clipped_rhos.shape) if show else None

    clipped_rhos = clipped_rhos.reshape(sequence_length, batch_size)

    del behavior_log_prob, target_logits, behavior_logits, all_logits, target_logits_all
    del actions, selected_mask, entity_mask, unit_type_entity_mask, mask_used, max_selected

    return target_log_prob, clipped_rhos, masks


def loss_function(agent, trajectories, use_opponent_state=True, 
                  no_replay_learn=False, only_update_baseline=False,
                  learner_baseline_weight=1, show=False):
    """Computes the loss of trajectories given weights."""

    # target_logits: ArgsActionLogits
    target_logits, baselines, select_units_num, entity_num = agent.rl_unroll(trajectories, 
                                                                             use_opponent_state, 
                                                                             show=show)
    device = target_logits.action_type.device

    # transpose to [seq_size x batch_size x -1]
    target_logits = transpose_target_logits(target_logits) # 调整动作维度顺序
    # 传进本函数的是 5 个形状 [B, T] 的张量组成的 list——baselines[k][b][t] = 第 k 路奖励、第 b 条轨迹、第 t 步的价值估计。这 5 路正好对应第 2 步讲过的 5 种伪奖励（实际 3 路有效），每路配一个专属价值头。
    baselines = transpose_baselines(baselines) # 它把 5 个价值头（baseline）输出的 V(s_t) 从 [B, T] 转成 [T, B]——和 logits 换轴的动机完全相同：下游所有 TD 计算都要沿第 0 维（时间）做 [:-1] / [1:] 的错位切片。

    # transpose to [seq_size x batch_size x -1]
    # 以下也同上，调换维度的顺序
    select_units_num = transpose_sth(select_units_num) 
    entity_num = transpose_sth(entity_num)

    # get used masks
    selected_mask, entity_mask = get_useful_masks(select_units_num, entity_num, device)
    del select_units_num, entity_num

    # note, we change the structure of the trajectories
    # shape before: [dict_name x batch_size x seq_size]
    # 以下两个方法具体看md文档
    trajectories = RU.stack_namedtuple(trajectories) 

    # shape after: [dict_name x seq_size x batch_size]
    trajectories = RU.namedtuple_zip(trajectories) 

    # We use a number of actor-critic losses - one for the winloss baseline, which
    # outputs the probability of victory, and one for each pseudo-reward
    # associated with following the human strategy statistic z.
    BASELINE_COSTS_AND_REWARDS = get_baseline_hyperparameters()

    loss_all = 0.
    loss_dict = {}

    # Vtrace Loss:
    reward_index = 0
    loss_actor_critic = 0.

    for baseline, costs_and_rewards in zip(baselines, BASELINE_COSTS_AND_REWARDS):
        if no_replay_learn:
            if reward_index != 0:
                break

        # 提取超参数
        vtrace_cost, baseline_cost, reward_name = costs_and_rewards
        print("reward_name:", reward_name) if debug else None

        rewards = PR.compute_pseudoreward(trajectories, reward_name, device) # 获取轨迹奖励
        print("rewards:", rewards) if 0 else None
        print("rewards not 0:", rewards[rewards != 0]) if 0 else None

        # The action_type argument, delay, and all other arguments are separately updated 
        # using a separate ("split") VTrace Actor-Critic losses. 
        baseline_weight = learner_baseline_weight # 看来是评价权重
        loss_baseline = td_lambda_loss(baseline, rewards, trajectories, device) # 得到价值预测误差
        # 这两行都是给价值损失（td_lambda_loss 的输出）乘权重
        loss_baseline = baseline_cost * loss_baseline
        loss_baseline = baseline_weight * loss_baseline
        loss_dict.update({reward_name + "-loss_baseline:": loss_baseline.item()}) # 跟踪打印输出
        loss_actor_critic += loss_baseline # 为什么要乘以以上两个 baseline_cost 和  baseline_weight，看md文档

        # we add vtrace loss
        vtrace_weight = 0 if only_update_baseline else 1
        loss_vtrace = sum_vtrace_loss(target_logits, trajectories, baseline, rewards, selected_mask, entity_mask, device)
        loss_vtrace = vtrace_cost * loss_vtrace
        #loss_vtrace = vtrace_weight * loss_vtrace
        loss_vtrace = vtrace_weight * loss_vtrace
        loss_dict.update({reward_name + "-loss_vtrace:": loss_vtrace.item()})
        loss_actor_critic += loss_vtrace
        reward_index += 1
        del loss_baseline, loss_vtrace, rewards

    # Upgo Loss:
    # The weighting of these updates will be considered 1.0. action_type, delay, and other arguments are 
    # also similarly separately updated using UPGO, in the same way as the VTrace Actor-Critic loss, with relative weight 1.0. 
    # AlphaStar: loss_upgo = UPGO_WEIGHT * split_upgo_loss(target_logits, baselines.winloss_baseline, trajectories)
    UPGO_COST = 1.0
    winloss_baseline = baselines[0]
    upgo_weight = 0 if only_update_baseline else 1
    loss_upgo = sum_upgo_loss(target_logits, trajectories, winloss_baseline, selected_mask, entity_mask, device)
    loss_upgo = UPGO_COST * loss_upgo
    loss_upgo = upgo_weight * loss_upgo
    loss_dict.update({"loss_upgo:": loss_upgo.item()})
    del baselines, BASELINE_COSTS_AND_REWARDS

    # Distillation Loss:
    # There is an distillation loss with weight 2e-3 on all action arguments, to match the output logits of the fine-tuned supervised policy 
    # which has been given the same observation. If the trajectory was conditioned on `cumulative_statistics`, there is an additional 
    # distillation loss of weight 1e-1 on the action type logits for the first four minutes of the game.
    # Thus ALL_KL_COST = 2e-3 and ACTION_TYPE_KL_COST = 1e-1
    ALL_KL_COST = 2e-3
    ACTION_TYPE_KL_COST = 1e-1

    # for all arguments
    all_kl_loss, all_kl_loss_dict = human_policy_kl_loss(target_logits, trajectories, selected_mask, entity_mask)
    all_kl_loss = ALL_KL_COST * all_kl_loss
    loss_dict.update({"all_kl_loss:": all_kl_loss.item()})
    for key, value in all_kl_loss_dict.items():
        loss_dict.update({"all_kl_loss_" + key + ":": value})

    action_type_kl_loss = human_policy_kl_loss_action(target_logits, trajectories)
    action_type_kl_loss = ACTION_TYPE_KL_COST * action_type_kl_loss
    loss_dict.update({"action_type_kl_loss:": action_type_kl_loss.item()})

    loss_kl = all_kl_loss + action_type_kl_loss
    #loss_kl = 0 * loss_kl
    loss_dict.update({"loss_kl:": loss_kl.item()})
    del all_kl_loss, action_type_kl_loss

    # Entropy Loss:
    # There is an entropy loss with weight 1e-4 on all action arguments, masked by which arguments are possible for a given action type.
    # Thus ENT_WEIGHT = 1e-4
    ENT_COST = 1e-4

    # note: we want to maximize the entropy so we gradient descent the -entropy. Original AlphaStar pseudocode is wrong
    # AlphaStar: loss_ent = entropy_loss(trajectories.behavior_logits, trajectories.masks)
    loss_ent = -entropy_loss(target_logits, trajectories, selected_mask, entity_mask)
    loss_ent = ENT_COST * loss_ent
    loss_dict.update({"loss_ent:": loss_ent.item()})
    del trajectories, selected_mask, entity_mask, target_logits

    loss_all = loss_actor_critic + loss_upgo + loss_kl + loss_ent
    del loss_actor_critic, loss_upgo, loss_kl, loss_ent

    return loss_all, loss_dict


def test():

    pass
