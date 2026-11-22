#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Library for RL algorithms."

# modified from AlphaStar pseudo-code
import traceback
import collections
import itertools
import gc

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.core.rl.rl_utils import Trajectory
from alphastarmini.core.rl.action import ArgsActionLogits

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

__author__ = "Ruo-Ze Liu"

debug = False


def reverse_seq(sequence):
    """Reverse sequence along dim 0.
    Args:
      sequence: Tensor of shape [T, B, ...].
    Returns:
      Tensor of same shape as sequence with dim 0 reversed up to sequence_lengths.
    """
    return torch.flip(sequence, [0])


def lambda_returns(values_tp1, rewards, discounts, lambdas=0.8):
    """Computes lambda returns.
    """
    # assert v_tp1 = torch.concat([values[1:, :], torch.unsqueeze(bootstrap_value, 0)], axis=0)
    # `back_prop=False` prevents gradients flowing into values and
    # bootstrap_value, which is what you want when using the bootstrapped
    # lambda-returns in an update as targets for values.
    return multistep_forward_view(rewards, discounts, values_tp1, lambdas)


def multistep_forward_view(rewards, pcontinues, state_values, lambda_,):
    """Evaluates complex backups (forward view of eligibility traces).
      ```python
      result[t] = rewards[t] + pcontinues[t] * (lambda_[t] * result[t+1] + (1-lambda_[t]) * state_values[t])
      result[last] = rewards[last] + pcontinues[last] * state_values[last]
      ```
      This operation evaluates multistep returns where lambda_ parameter controls
      mixing between full returns and boostrapping.
      ```
    Args:
      rewards: Tensor of shape `[T, B]` containing rewards.
      pcontinues: Tensor of shape `[T, B]` containing discounts.
      state_values: Tensor of shape `[T, B]` containing state values.
      lambda_: Mixing parameter lambda.
    Returns:
        Tensor of shape `[T, B]` containing multistep returns.
    """
    # Regroup:
    #   result[t] = (rewards[t] + pcontinues[t]*(1-lambda_)*state_values[t]) +
    #               pcontinues[t]*lambda_*result[t + 1]
    # Define:
    #   sequence[t] = rewards[t] + pcontinues[t]*(1-lambda_)*state_values[t]
    #   discount[t] = pcontinues[t]*lambda_
    # Substitute:
    #   result[t] = sequence[t] + discount[t]*result[t + 1]
    # Boundary condition:
    #   result[last] = rewards[last] + pcontinues[last]*state_values[last]
    # Add and subtract the same quantity at BC:
    #   state_values[last] =
    #       lambda_*state_values[last] + (1-lambda_)*state_values[last]
    # This makes:
    #   result[last] =
    #       (rewards[last] + pcontinues[last]*(1-lambda_)*state_values[last]) +
    #       pcontinues[last]*lambda_*state_values[last]
    # Substitute in definitions for sequence and discount:
    #   result[last] = sequence[last] + discount[last]*state_values[last]
    # Define:
    #   initial_value=state_values[last]
    # We get the following recurrent relationship:
    #   result[last] = sequence[last] + discount[last] * initial_value
    #   result[k] = sequence[k] + discount[k] * result[k + 1]
    # This matches the form of scan_discounted_sum:
    #   result = scan_sum_with_discount(sequence, discount,
    #                                   initial_value = state_values[last])
    
    # 这里其实就是ppo的轨迹回报累计计算
    # 只是这里优先计算每一个点的bellman价值
    sequence = rewards + pcontinues * state_values * (1 - lambda_)
    discount = pcontinues * lambda_

    return scan_discounted_sum(sequence, discount, state_values[-1], reverse=True)


def scan_discounted_sum(sequence, decay, initial_value, reverse=False):
    """Evaluates a cumulative discounted sum along dimension 0.
      ```python
      if reverse = False:
        result[1] = sequence[1] + decay[1] * initial_value
        result[k] = sequence[k] + decay[k] * result[k - 1]
      if reverse = True:
        result[last] = sequence[last] + decay[last] * initial_value
        result[k] = sequence[k] + decay[k] * result[k + 1]
      ```
    Respective dimensions T, B and ... have to be the same for all input tensors.
    T: temporal dimension of the sequence; B: batch dimension of the sequence.
    Args:
      sequence: Tensor of shape `[T, B, ...]` containing values to be summed.
      decay: Tensor of shape `[T, B, ...]` containing decays/discounts.
      initial_value: Tensor of shape `[B, ...]` containing initial value.
      reverse: Whether to process the sum in a reverse order.
    Returns:
      Cumulative sum with discount. Same shape and type as `sequence`.
    todo 后面再看
    """

    elems = [sequence, decay]
    if reverse:
        elems = [reverse_seq(s) for s in elems]
        [sequence, decay] = elems

    elems = [s.unsqueeze(0) for s in elems]
    elems = torch.cat(elems, dim=0) 
    elems = torch.transpose(elems, 0, 1)

    def scan(foo, x, initial_value, debug=False):
        res = []
        a_ = initial_value.clone().detach()
        res.append(foo(a_, x[0]).unsqueeze(0))
        a_ = foo(a_, x[0])

        for i in range(1, len(x)):
            res.append(foo(a_, x[i]).unsqueeze(0))
            a_ = foo(a_, x[i])

        del a_

        return torch.cat(res)

    result = scan(lambda a, x: x[0] + x[1] * a, elems, initial_value=initial_value)

    del scan

    if reverse:
        result = reverse_seq(result)

    del elems

    return result


def vtrace_advantages(clipped_rhos, rewards, discounts, values, bootstrap_value, lamda=0.8):
    """Computes v-trace return advantages.

    see below function "vtrace_from_importance_weights"
    """
    return vtrace_from_importance_weights(rhos=clipped_rhos, discounts=discounts, rewards=rewards, 
                                          values=values, bootstrap_value=bootstrap_value,
                                          lamda=lamda)


VTraceReturns = collections.namedtuple('VTraceReturns', 'vs pg_advantages')


def vtrace_from_importance_weights(
        rhos, discounts, rewards, values, bootstrap_value,
        clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0, lamda=0.8):
    r"""
    https://github.com/deepmind/trfl/blob/40884d4bb39f99e4a642acdbe26113914ad0acec/trfl/vtrace_ops.py#L154
    V-trace from log importance weights.Calculates V-trace actor critic targets as described in
    "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al. In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and NUM_ACTIONS refers to the number of actions. 
    This code also supports the case where all tensors have the same number of additional dimensions, e.g.,
    `rewards` is `[T, B, C]`, `values` is `[T, B, C]`, `bootstrap_value` is `[B, C]`.
    Args:
      log_rhos: A float32 tensor of shape `[T, B, NUM_ACTIONS]` representing the
        log importance sampling weights, i.e.
        log(target_policy(a) / behaviour_policy(a)). V-trace performs operations
        on rhos in log-space for numerical stability.
      discounts: A float32 tensor of shape `[T, B]` with discounts encountered when following the behaviour policy.
      rewards: A float32 tensor of shape `[T, B]` containing rewards generated by following the behaviour policy.
      values: A float32 tensor of shape `[T, B]` with the value function estimates wrt. the target policy.
      bootstrap_value: A float32 of shape `[B]` with the value function estimate at time T.
      clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
        importance weights (rho) when calculating the baseline targets (vs).
        rho^bar in the paper. If None, no clipping is applied.
      clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
        on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). If
        None, no clipping is applied.
      name: The name scope that all V-trace operations will be created in.
    Returns:
      A VTraceReturns namedtuple (vs, pg_advantages) where:
        vs: A float32 tensor of shape `[T, B]`. Can be used as target to
          train a baseline (V(x_t) - vs_t)^2.
        pg_advantages: A float32 tensor of shape `[T, B]`. Can be used as the
          advantage in the calculation of policy gradients.

    rhos 重要性权重 ρ_t = π(a)/μ(a)，“旧经验打几折”	[T, B]
    discounts	γ_t：这步之后游戏还继续吗（继续=1，终局=0）	[T, B]	
    rewards	本路奖励流 r_t	[T, B]
    values	价值头对当前状态的估计 V(s_t)	[T, B]
    bootstrap_value	序列末端的价值 V(s_{T-1})	[B]
    clip_rho_threshold	算价值目标 vs 时 ρ 的截断上界 ρ̄	标量	
    clip_pg_rho_threshold	算策略梯度优势时 ρ 的另一个上界 ρ̄_pg	标量
    lamda	TD 误差往回传的衰减系数（和 λ-回报里的 λ 同角色）	标量

    1. rhos [T, B] —— 上一讲的主角：每个 (t, b) 一个“折扣率”。来源是 get_logprob_and_rhos 算出的 clipped_rhos（进函数前已经被砍到 ≤1）。注意函数 docstring 写的是 log_rhos——那是从 trfl 库抄来的过时注释，这份实现实际接收的是已 exp、已截断的 ρ，别被 docstring 骗了。

    2. discounts [T, B] —— 来自 ~is_final（第 3 步讲过：~ 必须配 dtype=np.bool）。本项目的取值只有 0/1：局内不打折、局边界一刀切断。

    3. rewards [T, B] —— 本路奖励（第 2 步：winloss 真实输赢 / Levenshtein / Hamming），在 sum_vtrace_loss 里已切 [:-1] 对齐到“转移”。

    4. values [T, B] —— 传进来的是 baselines[:-1]：价值头对本路状态的估计 V(s_t)。不含最后一步（最后一步另有用途，见下）。

    5. bootstrap_value [B] —— baselines[-1]：窗口末端状态的估价。没有时间维，只有 batch 维——它是链条尽头的“一个点”，负责把“窗口外的未来值多少”折算进来。这是五个张量里唯一形状不同的。

    6/7. 两个 clip 阈值（标量） —— trfl 原版里 ρ̄（管价值目标的收敛方向）和 ρ̄_pg（管策略梯度的方差）是两个独立的旋钮，IMPALA 论文里可以取不同值。但在本项目里都是空转：ρ 在上游已经被砍到 ≤1，函数内部第 213/216/270 行的 clamp(max=1) 对已经 ≤1 的数再砍一次，什么也不做。读代码时知道“这两个旋钮在这里没被使用”即可，学理论时再去 IMPALA 论文看它们的区别。

    8. lamda = 0.8 —— 单步 TD 误差 δ_t 往回传播时每步乘的衰减（γλc），和 λ-回报里那个 λ 是同一个思想：越远的影响越小。

    对齐图（五者怎么站位）
    """

    if clip_rho_threshold is not None:
        clip_rho_threshold = torch.tensor(clip_rho_threshold, dtype=torch.float32, device=values.device)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = torch.tensor(clip_pg_rho_threshold, dtype=torch.float32, device=values.device)

    # Make sure tensor ranks are consistent.
    # 限制新旧策略比率阈值的范围
    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    else:
        clipped_rhos = rhos
    cs = torch.clamp(rhos, max=1.)

    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat(
        [values[1:], bootstrap_value.unsqueeze(0)], axis=0)

    # rewards + discounts * values_t_plus_1 - values：这个应该是得到的优势，小于0表示新的动作策略不如旧策略，大于0表示新的动作策略优于新策略
    # clipped_rhos 是 新旧策略的比率，乘上 ρ 打折扣（旧经验该信几折）。δ 是一步的意外
    # 只看一步的 δ 太短视——价值网络初期不准，单步 TD 误差噪声大。所以 V-trace 要把未来所有步的意外也卷进来，拼成 vs。这就是 vs 和 td_lambda_loss 里 λ-回报的共同思想：多步信息比单步稳
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
    # torch.flip 这个操作是将序列倒序
    flip_discounts = torch.flip(discounts, dims=[0])
    flip_cs = torch.flip(cs, dims=[0])
    flip_deltas = torch.flip(deltas, dims=[0])

    # 将三个序对应位置拼接起来
    sequences = [item for item in zip(flip_discounts, flip_cs, flip_deltas)]

    del flip_discounts, flip_cs, flip_deltas, deltas, values_t_plus_1, cs

    # V-trace vs are calculated through a 
    # scan from the back to the beginning
    # of the given trajectory.
    def scanfunc(acc, sequence_item):
        discount_t, c_t, delta_t = sequence_item
        # return delta_t + lamda * discount_t * c_t * acc
        # └δ┘    └λ┘   └γ┘      └c┘ └上一轮的累积┘
        return delta_t + lamda * discount_t * c_t * acc # 这行在把“未来的意外”打折搬到今天

    initial_values = torch.zeros_like(bootstrap_value, device=bootstrap_value.device)

    # our implemented scan function for pytorch
    def scan(foo, x, initial_value):
        res = []
        a_ = initial_value.clone().detach()
        res.append(foo(a_, x[0]).unsqueeze(0))

        # should not miss this line
        a_ = foo(a_, x[0])

        for i in range(1, len(x)):
            res.append(foo(a_, x[i]).unsqueeze(0))
            a_ = foo(a_, x[i])

        del a_

        return torch.cat(res)

    # 从末端往回滚”的循环本身（倒序、累积器、翻回来）就是 scan 框架在干的事——上一问讲过机制。语义上，展开后长这样：
    # vs偏移_0 = δ_0 + (λγc)δ_1 + (λγc)²δ_2 + (λγc)³δ_3 + ...
    # 每个未来时刻的“意外”对今天的影响力按距离指数衰减
    vs_minus_v_xs = scan(foo=scanfunc, x=sequences, initial_value=initial_values)
    del scanfunc, sequences, initial_values

    # Reverse the results back to original order.
    vs_minus_v_xs = torch.flip(vs_minus_v_xs, dims=[0])

    # Add V(x_s) to get v_s.
    # 这行是 V-trace 的最后一道拼装。一句话结论：scan 滚出来的只是“修正量”（偏移），这行把每步的价值头估计 V(s_t) 加回去，凑成完整的 vs——代码自己的注释就写着 "Add V(x_s) to get v_s"。
    # 具体看md
    vs = torch.add(vs_minus_v_xs, values)

    # Advantage for policy gradient.
    vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], axis=0)

    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
    else:
        clipped_pg_rhos = rhos

    # 这里的优势是已经累计了轨迹优势的对比
    '''
    括号里的部分：r_t + γ·vs_{t+1} − V(s_t) —— 眼熟吗？就是 TD 误差。但注意中段用的是 vs_{t+1}（上一讲讲的多步修匀价值），不是单步 V(s′)。翻译成人话：

    “走完这一步，实际拿到的（即时奖励 + 修正后的未来估计），比我走之前预估的好多少？”

    结果 > 0：这一步比预期好 → 下游策略梯度会抬高 π(a_t)，鼓励以后多走这种棋；
    结果 < 0：这一步比预期差 → 压低 π(a_t)。
    '''
    pg_advantages = (clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))

    del clipped_pg_rhos, rewards, discounts, vs_t_plus_1, values, vs_minus_v_xs, bootstrap_value

    # Make sure no gradients backpropagated through the returned values.
    # 切断梯度，两个返回值只当常数用
    # vs 和 pg_advantages 里都含有 values（价值头的输出，带梯度）。如果不断开，下游 (−log π) × 优势 的总损失会把梯度一路漏进价值头——价值头会学会往对自己有利的方向篡改 V，让“优势”变小，从而白捡低 loss。这就是第 3 步讲过的“自己追自己尾巴”的变体，必须断。
    return VTraceReturns(vs=vs.detach(), pg_advantages=pg_advantages.detach())


def simple_vtrace_advantages(clipped_rhos, rewards, discounts, values, bootstrap):
    '''Inspried by DI-Star '''
    lamda = 0.8  # 0.8

    bootstrap = bootstrap.unsqueeze(0)

    all_values = torch.cat([values, bootstrap], dim=0)
    values_tp1 = all_values[1:]
    delta = clipped_rhos * (rewards + discounts * values_tp1 - values)

    value_vtrace = torch.empty_like(all_values)
    value_vtrace[-1] = bootstrap
    for i in range(all_values.shape[0] - 2, -1, -1):
        value_vtrace[i] = all_values[i] + delta[i] + discounts[i] * lamda * clipped_rhos[i] * \
            (value_vtrace[i + 1] - all_values[i + 1])

    values_vtrace_tp1 = value_vtrace[1:]
    advantages = clipped_rhos * (rewards + discounts * values_vtrace_tp1 - values)
    return (value_vtrace[:-1].detach(), advantages.detach())


def upgo_returns(values, rewards, discounts, bootstrap, debug=False):
    """Computes the UPGO return targets.

    Args:
      values: Estimated state values. Shape [T, B].
      rewards: Rewards received moving to the next state. Shape [T, B].
      discounts: If the step is NOT final. Shape [T, B].
      bootstrap: Bootstrap values. Shape [B].
    Returns:
      UPGO return targets. Shape [T, B].
    """
    # we change it to pytorch version
    # next_values = np.concatenate((values[1:], np.expand_dims(bootstrap, axis=0)), axis=0)
    # next_values. Shape [T, B].
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(dim=0)], dim=0)

    # print("next_values:", next_values) if debug else None
    # print("rewards:", rewards) if debug else None
    # print("discounts:", discounts) if debug else None
    # print("discounts * next_values:", discounts * next_values) if debug else None
    # print("rewards + discounts * next_values:", rewards + discounts * next_values) if debug else None
    # print("values:", values) if debug else None
    # print("(rewards + discounts * next_values) >= values:", (rewards + discounts * next_values) >= values) if debug else None

    # Upgo can be viewed as a lambda return! The trace continues (i.e. lambda =
    # 1.0) if r_t + V_tp1 > V_t. original G_t = r_t +  
    lambdas = (rewards + discounts * next_values) >= values
    print("lambdas:", lambdas) if debug else None

    # change the bool tensor to float tensor
    lambdas = lambdas.float()
    print("lambdas:", lambdas) if debug else None

    # Shift lambdas left one slot, such that V_t matches indices with lambda_tp1.
    # lambdas = np.concatenate((lambdas[1:], np.ones_like(lambdas[-1:])), axis=0)
    lambdas = torch.cat([lambdas[1:], torch.ones_like(lambdas[-1:], device=lambdas.device)], dim=0)
    print("lambdas:", lambdas) if debug else None

    return lambda_returns(next_values, rewards, discounts, lambdas)


def entropy(all_logits, selected_mask=None, entity_mask=None, unit_type_entity_mask=None, 
            outlier_remove=False):
    [policy_logits] = all_logits
    log_policy = F.log_softmax(policy_logits, dim=-1) 
    policy = torch.exp(log_policy)  # more stable than softmax(policy_logits)
    x = log_policy

    if selected_mask is not None:
        x = x * selected_mask.unsqueeze(-1)    
    if entity_mask is not None:
        x = x * entity_mask
    if unit_type_entity_mask is not None:
        x = x * unit_type_entity_mask

    x = -policy * x

    # if len(x.shape) == 3:
    #     print('policy[0, 1, 44]', policy[0, 1, 44])
    #     print('x[0, 1, 44]', x[0, 1, 44])
    #     print('policy[2, 1, 40]', policy[2, 1, 40])
    #     print('x[2, 1, 40]', x[2, 1, 40])

    x = remove_outlier(x, outlier_remove)

    # if len(x.shape) == 3:
    #     print(stop)

    #     print('entropy x', x)
    #     print('x.shape', x.shape)

    del all_logits, policy_logits, log_policy, policy
    del selected_mask, entity_mask, unit_type_entity_mask

    return x


def kl(all_logits, selected_mask=None, entity_mask=None, 
       unit_type_entity_mask=None, outlier_remove=False):
    [student_logits, teacher_logits] = all_logits
    s_logprobs = F.log_softmax(student_logits, dim=-1)
    t_logprobs = F.log_softmax(teacher_logits, dim=-1)
    teacher_probs = torch.exp(t_logprobs)  # more stable than softmax(teacher_logits)
    x = t_logprobs - s_logprobs

    # if len(x.shape) == 3:
    #     print('x[0, 1, 44]', x[0, 1, 44])
    #     print('t_logprobs[0, 1, 44]', t_logprobs[0, 1, 44])
    #     print('s_logprobs[0, 1, 44]', s_logprobs[0, 1, 44])
    #     print('student_logits[0, 1, 44]', student_logits[0, 1, 44])
    #     print('teacher_logits[0, 1, 44]', teacher_logits[0, 1, 44])

    #     print('x[2, 1, 40]', x[2, 1, 40])

    # print('kl x', x)
    # print('x.shape', x.shape)

    if selected_mask is not None:
        x = x * selected_mask.unsqueeze(-1) 
    if entity_mask is not None:
        x = x * entity_mask
    if unit_type_entity_mask is not None:
        x = x * unit_type_entity_mask

    x = remove_outlier(x, outlier_remove)
    x = teacher_probs * x

    del all_logits, student_logits, teacher_logits, s_logprobs, t_logprobs, teacher_probs
    del selected_mask, entity_mask, unit_type_entity_mask

    return x


def compute_cliped_importance_weights(target_log_prob, behavior_log_prob):
    return torch.clamp(torch.exp(target_log_prob - behavior_log_prob), max=1.)


def remove_outlier(x, remove=False):
    outlier_mask = (torch.abs(x) > 1e8)
    if remove:
        x = x * ~outlier_mask
    else:
        if outlier_mask.any() > 0:
            index = outlier_mask.nonzero(as_tuple=True)
            print("index:", index) if 1 else None
            print("x[index]:", x[index]) if 1 else None
            print(stop)

            del index

    del outlier_mask

    return x


def log_prob(logits, actions, mask_used, max_selected, show=False, outlier_remove=False):
    """Returns the log probability of taking an action given the logits."""
    [selected_mask, entity_mask, unit_type_entity_mask] = mask_used

    # print('logits.shape', logits.shape) if show else None
    # print('logits[0][0]', logits[0][0]) if show else None
    # print('logits[0][1]', logits[0][1]) if show else None

    # print('actions.shape', actions.shape) if show else None
    # print('actions[0][0]', actions[0][0]) if show else None
    # print('actions[0][1]', actions[0][1]) if show else None

    if unit_type_entity_mask is not None:
        unit_type_entity_mask = unit_type_entity_mask.reshape(-1, unit_type_entity_mask.shape[-1])
        # print('unit_type_entity_mask[0]', unit_type_entity_mask[0]) if show else None
        # print('unit_type_entity_mask.shape', unit_type_entity_mask.shape) if show else None

    if entity_mask is not None:
        entity_mask = entity_mask.reshape(-1, entity_mask.shape[-1])
        # print('entity_mask[0]', entity_mask[0]) if show else None
        # print('entity_mask.shape', entity_mask.shape) if show else None

    if len(logits.shape) == 3:
        select_size = logits.shape[1]
        logits = logits.view(-1, logits.shape[-1])

        if select_size > 1:
            unit_type_entity_mask = unit_type_entity_mask.unsqueeze(1).repeat(1, select_size, 1)
            entity_mask = entity_mask.unsqueeze(1).repeat(1, select_size, 1)
            unit_type_entity_mask = unit_type_entity_mask.view(-1, unit_type_entity_mask.shape[-1])
            entity_mask = entity_mask.view(-1, entity_mask.shape[-1])
            entity_mask = entity_mask * unit_type_entity_mask

            # print('entity_mask[0]', entity_mask[0]) if show else None
            # print('entity_mask.shape', entity_mask.shape) if show else None

    if len(actions.shape) == 3:
        actions = actions.view(-1, actions.shape[-1])

    del unit_type_entity_mask

    def cross_entropy_mask_class(pred, soft_targets, mask=None):
        x = F.log_softmax(pred, dim=-1)

        # print('x[12]', x[12]) if show else None
        # print('x[25]', x[25]) if show else None
        # print('x.shape', x.shape) if show else None

        x = - soft_targets * x

        # print('x[12]', x[12]) if show else None
        # print('x[25]', x[25]) if show else None
        # print('x.shape', x.shape) if show else None

        if mask is not None:
            x = x * mask

        # print('mask[0]', mask[0]) if show else None
        # print('mask[1]', mask[1]) if show else None
        # print('mask.shape', mask.shape) if show else None
        # print('x[0]', x[0]) if show else None
        # print('x[1]', x[1]) if show else None
        # print('x.shape', x.shape) if show else None

        # print('x[0]', x[0]) if show else None
        # print('x.shape', x.shape) if show else None        

        x = torch.sum(x, -1)
        # print('x[0]', x[0]) if show else None
        # print('x[1]', x[1]) if show else None
        # print('x.shape', x.shape) if show else None

        del soft_targets, pred

        return x

    # print('logits', logits[0]) if show else None
    # print('logits.shape', logits.shape) if show else None

    # print('actions', actions[0]) if show else None
    # print('actions.shape', actions.shape) if show else None

    # actions: shape [BATCH_SIZE, 1] to [BATCH_SIZE]
    actions = torch.squeeze(actions, dim=-1)
    actions = F.one_hot(actions, num_classes=logits.shape[-1])

    # print('actions[0]', actions[0]) if show else None
    # print('actions[1]', actions[1]) if show else None
    # print('actions.shape', actions.shape) if show else None

    # print('logits[0]', logits[0]) if show else None
    # print('logits[1]', logits[1]) if show else None
    # print('logits.shape', logits.shape) if show else None

    loss_result = cross_entropy_mask_class(logits, actions, entity_mask)

    del logits, actions, entity_mask

    if selected_mask is not None:
        # 展平掩码
        selected_mask = selected_mask.view(-1, selected_mask.shape[-1])

        selected_mask_index = torch.sum(selected_mask.float(), dim=-1)
        # print('selected_mask_index', selected_mask_index) if show else None

        select_index = selected_mask_index > 2

        if len(loss_result.shape) == 1:
            loss_result = loss_result.view(-1, max_selected)

            #print(stop) if show else None

        # print('loss_result[0]', loss_result[0]) if show else None
        # print('loss_result.shape', loss_result.shape) if show else None    
        loss_result = loss_result * selected_mask

        # print('selected_mask[0]', selected_mask[0]) if show else None
        # print('selected_mask.shape', selected_mask.shape) if show else None

        # loss_result_2 = loss_result[select_index]
        # if len(loss_result_2):
        #     print('loss_result_2', torch.sum(loss_result_2, dim=-1)) if show else None

        loss_result = remove_outlier(loss_result, outlier_remove)
        loss_result = torch.sum(loss_result, dim=-1)

        # print('loss_result', loss_result) if show else None

        del selected_mask

    loss_result = -loss_result

    # print('loss_result', loss_result[0]) if show else None
    # print('loss_result.shape', loss_result.shape) if show else None

    return loss_result


def test(debug=True):

    test_td_lamda_loss = True
    test_upgo_return = True
    test_vtrace_advantage = True

    if test_td_lamda_loss:
        batch_size = 2
        seq_len = 4

        device = 'cpu'

        is_final = [[0, 0], [0, 1], [1, 0], [0, 0]]
        baselines = [[-0.5, 0.90], [-0.7, 0.95], [-0.95, 0.5], [0.5, 0.6]]
        rewards = [[0, 0], [0, 1], [-1, 0], [0, 0]]

        baselines = torch.tensor(baselines, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        discounts = ~np.array(is_final[:-1], dtype=np.bool)  # don't forget dtype=np.bool!, or '~' ouput -1 instead of 1
        discounts = torch.tensor(discounts, dtype=torch.float, device=device)

        print("discounts:", discounts) if debug else None

        boostrapvales = baselines[1:]
        rewards_short = rewards[:-1]

        if 1:
            # The baseline is then updated using TDLambda, with relative weighting 10.0 and lambda 0.8.
            returns = lambda_returns(boostrapvales, rewards_short, discounts, lambdas=0.8)

            # returns = stop_gradient(returns)
            returns = returns.detach()
            print("returns:", returns) if debug else None

            result = returns - baselines[:-1]
            print("result:", result) if debug else None

            # change to pytorch version
            td_lambda_loss = 0.5 * torch.mean(torch.square(result))
            print('td_lambda_loss', td_lambda_loss)

        if 2:
            lambdas = 0.8

            returns = torch.empty_like(rewards_short)

            returns[-1, :] = rewards_short[-1, :] + discounts[-1, :] * boostrapvales[-1, :]
            for t in reversed(range(rewards_short.shape[0] - 1)):
                returns[t, :] = rewards_short[t, :] + discounts[t, :] * (lambdas * returns[t + 1, :] +
                                                                         (1 - lambdas) * boostrapvales[t, :])
            print("returns:", returns) if debug else None

            result = returns.detach() - baselines[:-1]
            print("result:", result) if debug else None

            td_lambda_loss_2 = 0.5 * torch.pow(result, 2).mean()
            print('td_lambda_loss_2', td_lambda_loss_2)

    if test_upgo_return:
        batch_size = 2
        seq_len = 4

        device = 'cpu'

        is_final = [[0, 0], [0, 1], [1, 0], [0, 0]]
        baselines = [[-0.5, 0.90], [-0.7, 0.95], [-0.95, 0.5], [0.5, 0.6]]
        rewards = [[0, 0], [0, 1], [-1, 0], [0, 0]]

        baselines = torch.tensor(baselines, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        discounts = ~np.array(is_final[:-1], dtype=np.bool)  # note the discount is not the similar indicies as in lamda returns
        discounts = torch.tensor(discounts, dtype=torch.float, device=device)

        rewards_short = rewards[:-1]

        if 1:
            values = baselines[:-1]
            bootstrap = baselines[-1]

            returns = upgo_returns(values, rewards_short, discounts, bootstrap)

            returns = returns.detach()
            print("upgo returns:", returns) if debug else None

            result = returns - values
            print("upgo result:", result) if debug else None

        if 2:
            pass

    if test_vtrace_advantage:
        batch_size = 2
        seq_len = 4

        device = 'cpu'

        is_final = [[0, 0], [0, 1], [1, 0], [0, 0]]
        baselines = [[-0.5, 0.90], [-0.7, 0.95], [-0.95, 0.5], [0.5, 0.6]]
        rewards = [[0, 0], [0, 1], [-1, 0], [0, 0]]

        baselines = torch.tensor(baselines, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        discounts = ~np.array(is_final[:-1], dtype=np.bool)  # note the discount is not the similar indicies as in lamda returns
        discounts = torch.tensor(discounts, dtype=torch.float, device=device)

        rewards_short = rewards[:-1]
        values = baselines[:-1]
        bootstrap = baselines[-1]

        if 1:
            clipped_rhos = [[0.95, 0.99], [0.976, 1], [0.90, 0.74]]
            clipped_rhos = torch.tensor(clipped_rhos, dtype=torch.float, device=device)

            weighted_advantage = vtrace_advantages(clipped_rhos, rewards_short, discounts, values, bootstrap)[1]

            print('weighted_advantage_1', weighted_advantage) if debug else None
            print('weighted_advantage_1.shape', weighted_advantage.shape) if debug else None

        if 2:
            clipped_rhos = [[0.95, 0.99], [0.976, 1], [0.90, 0.74]]
            clipped_rhos = torch.tensor(clipped_rhos, dtype=torch.float, device=device)

            weighted_advantage = simple_vtrace_advantages(clipped_rhos, rewards_short, discounts, values, bootstrap)[1]

            print('weighted_advantage_2', weighted_advantage) if debug else None
            print('weighted_advantage_2.shape', weighted_advantage.shape) if debug else None            
