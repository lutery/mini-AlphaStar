#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Action Type Head."

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysc2.lib.actions import RAW_FUNCTIONS
from alphastarmini.core.arch.spatial_encoder import ResBlock1D
from alphastarmini.lib.glu import GLU

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS

from alphastarmini.lib.sc2 import raw_actions_mapping_protoss as RAMP

__author__ = "Ruo-Ze Liu"

debug = False


class ActionTypeHead(nn.Module):
    '''
    Inputs: lstm_output, scalar_context
    Outputs:
        action_type_logits - The logits corresponding to the probabilities of taking each action
        action_type - The action_type sampled from the action_type_logits
        autoregressive_embedding - Embedding that combines information from `lstm_output` and all previous sampled arguments. 
        To see the order arguments are sampled in, refer to the network diagram
    '''

    def __init__(self, lstm_dim=AHP.lstm_hidden_dim, n_resblocks=AHP.n_resblocks, 
                 is_sl_training=True, temperature=AHP.temperature, original_256=AHP.original_256,
                 max_action_num=LS.action_type_encoding, context_size=AHP.context_size, 
                 autoregressive_embedding_size=AHP.autoregressive_embedding_size,
                 use_action_type_mask=AHP.use_action_type_mask):
        super().__init__()
        self.is_sl_training = is_sl_training
        self.temperature = temperature

        self.embed_fc = nn.Linear(lstm_dim, original_256)  # with relu
        self.resblock_stack = nn.ModuleList([
            ResBlock1D(inplanes=original_256, planes=original_256, seq_len=1)
            for _ in range(n_resblocks)])

        self.max_action_num = max_action_num # 所能执行的动作分布号码
        self.glu_1 = GLU(input_size=original_256, context_size=context_size,
                         output_size=max_action_num) # 主要结合游戏的全部信息和影响动作的游戏状态信息预测接下来的动作分布

        self.fc_1 = nn.Linear(max_action_num, original_256)
        self.glu_2 = GLU(input_size=original_256, context_size=context_size,
                         output_size=autoregressive_embedding_size)
        self.glu_3 = GLU(input_size=lstm_dim, context_size=context_size,
                         output_size=autoregressive_embedding_size)
        self.softmax = nn.Softmax(dim=-1)

        self.use_action_type_mask = use_action_type_mask # 超参数启用了动作掩码
        self.is_rl_training = False # 当前是否处于 RL 训练模式

    def set_rl_training(self, staus):
        self.is_rl_training = staus

    def forward(self, lstm_output, scalar_context, action_type_mask=None, action_type=None):
        '''
        lstm_output	Tensor	✅ 必传	LSTM 输出的序列状态理解，综合了游戏所有状态的特征信息 (batch_size * seq_len, lstm_hidden_dim)
        scalar_context	Tensor	✅ 必传	精选的 7 个标量上下文，用于门控（gating）
        action_type_mask	Tensor	❌ 可选	动作合法性掩码，RL 训练时屏蔽非法动作
        action_type	Tensor	❌ 可选	教师信号——监督学习时传入人类真实动作，RL/推理时为 None action_type 形状：(batch_size, 1)，每行是一个动作 ID（0 ~ 563 之间的整数）。

        '''

        batch_size = lstm_output.shape[0]

        # AlphaStar: The action type head embeds `lstm_output` into a 1D tensor of size 256
        # 投影到 256 维（original_256），经过 4 个 ResBlock
        x = self.embed_fc(lstm_output) # 压缩游戏状态的特征编码维度 # (b*s, 128) → (b*s, 256)

        # AlphaStar: passes it through 16 ResBlocks with layer normalization each of size 256, and applies a ReLU. 
        # QUESTION: There is no map, how to use resblocks?
        # ANSWER: USE resblock1D
        # input shape is [batch_size x seq_size x embedding_size]
        # note that embedding_size is equal to channel_size in conv1d
        # we change this to [batch_size x embedding_size x seq_size]
        x = x.unsqueeze(-1) # (b*s, 256, 1)  给 Conv1d 准备的
        for resblock in self.resblock_stack:
            x = resblock(x)
        x = F.relu(x)
        x = x.squeeze(-1) #  # (b*s, 256)

        # AlphaStar: The output is converted to a tensor with one logit for each possible 
        # action type through a `GLU` gated by `scalar_context`.
        # 被 scalar_context 门控后产生动作类型 logits
        action_type_logits = self.glu_1(x, scalar_context)

        # inspired by the DI-star project, in action_type_head
        # 这段代码的核心目的：在 RL 训练时，强制阻止模型选择"当前游戏状态下不可能执行"的动作，同时保证监督学习中教师信号不被误杀。
        if self.is_rl_training and self.use_action_type_mask and action_type_mask is not None:
            action_type_mask = action_type_mask.bool() # action_type_mask 传入时是一个 float 张量，形状 (batch_size * seq_len, Actions_Size)，值只有 0 和 1：，缓缓为bool张量
            if action_type is not None:       # 教师信号保护，将掩码中教师信号对应的位置设置为True
                for i, a in enumerate(action_type):
                    '''
                    # 示例：batch=3，人类教师在这 3 个样本中分别选了动作 42、128、7
                    action_type = [[42],
                                [128],
                                [7]]

                    # 循环做：把每个样本中教师选的动作，在 mask 里强制标为 True
                    # 样本 0：mask[0, 42] = True   ← 即使原本是 False，也拉回 True
                    # 样本 1：mask[1, 128] = True
                    # 样本 2：mask[2, 7] = True
                    '''
                    action_type_mask[i, a.item()] = True
            action_type_logits = action_type_logits + (~action_type_mask * (-1e9)) # 根据可用的动作，将不可用的动作的logits设置为一个极小值
            del action_type_mask

        print('action_type_logits:', action_type_logits) if debug else None
        print('action_type_logits.shape:', action_type_logits.shape) if debug else None

        # AlphaStar: `action_type` is sampled from these logits using a multinomial with temperature 0.8. 
        # Note that during supervised learning, `action_type` will be the ground truth human action 
        # type, and temperature is 1.0 (and similarly for all other arguments).
        # 如果是探索，则鼓励探索，会将预测的logits的每一个值接近，如果
        temperature = self.temperature if self.is_rl_training else 1
        action_type_logits = action_type_logits / temperature
        print('action_type_logits:', action_type_logits) if debug else None
        print('action_type_logits.shape:', action_type_logits.shape) if debug else None

        # note, torch.multinomial need samples to non-negative, finite and have a non-zero sum
        # which is different with tf.multinomial which can accept negative values like log(action_type_probs)
        # 如果调用方传入了教师信号（如 mimic_forward 传入 gt_action_type），整段 if 跳过——直接使用传入的动作，不做采样。SL 训练不需要模型自己猜，它只要学会"为什么人类在这个局面下选了这个动作"。
        if action_type is None:
            action_type_probs = self.softmax(action_type_logits) # softmax 把没被掩码杀掉的 logits 转为概率分布。被掩码打过 -1e9 的位置，exp(-1e9) 在浮点精度下为 0，概率严格为 0。
            action_type_probs = action_type_probs.reshape(batch_size, -1) # 这里的 batch_size 实际上是 batch_size * sequence_length
            # 这个 reshape 是防御性编程——PyTorch 的 multinomial 要求输入恰好是 2D (n, categories)。如果上游因某些原因变成了 3D（比如多了一个多余维度），reshape 能把它拍回正确的形状。正常情况下是 no-op，输入 (batch_size, 564)，输出还是 (batch_size, 564)。
            print('action_type_probs:', action_type_probs) if debug else None
            print('action_type_probs.shape:', action_type_probs.shape) if debug else None

            # multinomial(probs, 1) 的意思是：对每个样本，按概率分布随机抽取 1 个类别，返回类别序号。
            '''
            概率分布: [0.119, 0.072, 0.0, 0.0, 0.809, ...]
                                ↑      ↑
                                绝不可能抽到

            采样结果示例:
            样本 0: 第 4 个动作 (概率 0.809)     → 返回 4
            样本 1: 第 0 个动作 (概率 0.119)     → 返回 0
            样本 2: 第 4 个动作 (概率 0.809)     → 返回 4
            '''
            action_type = torch.multinomial(action_type_probs, 1)
            action_type = action_type.reshape(batch_size, -1)
            del action_type_probs

        # change action_type to one_hot version
        '''
        action_type = [[4],        → one_hot → [[0, 0, 0, 0, 1, 0, ..., 0],   ← 第 4 位是 1
               [0],                     [1, 0, 0, 0, 0, 0, ..., 0],   ← 第 0 位是 1
               [4]]                     [0, 0, 0, 0, 1, 0, ..., 0]]   ← 第 4 位是 1
              shape (3, 1)              shape (3, 1, 564)
        '''
        action_type_one_hot = L.tensor_one_hot(action_type, self.max_action_num) # # (batch, 1, max_action_num)
        '''
        (3, 1, 564) → squeeze(-2) → (3, 564)
        '''
        action_type_one_hot = action_type_one_hot.squeeze(-2) # (batch, max_action_num)

        # AlphaStar: `autoregressive_embedding` is then generated by first applying a ReLU 
        # and linear layer of size 256 to the one-hot version of `action_type`
        z = F.relu(self.fc_1(action_type_one_hot)) # 把 564 个离散动作类型，映射到 256 维的稠密嵌入空间。选"移动"和选"攻击"会得到不同的 256 维向量，向量之间的距离反映了动作之间的语义相似性

        # AlphaStar: and projecting it to a 1D tensor of size 1024 through a `GLU` gated by `scalar_context`.
        # 这是两条路径中的第一条。它回答的问题是："我选了这个动作类型，它蕴含了什么信息需要告诉后面的 heads？"
        z = self.glu_2(z, scalar_context) # 用 scalar_context 做门控后再投影到 1024 维——门控会根据当前局势（种族、可用动作、建筑历史等）筛选 z 中"值得传下去"的维度。比如你是 Protoss，"移动狂热者"相关的维度会被门控加强，"巡逻医疗运输机"相关的维度会被压弱。

        # AlphaStar: That projection is added to another projection of `lstm_output` into a 1D tensor of size 
        # 1024 gated by `scalar_context` to yield `autoregressive_embedding`.
        # 它回答的问题是："在所有合法/重要的动作类型中，LSTM 对当前局势的总体理解是什么？"
        t = self.glu_3(lstm_output, scalar_context) # glu_3 的输入是 128 维（lstm_hidden_dim），输出 1024 维。

        '''
        	z（动作路径）	t（LSTM 路径）
            输入来源	已选定的具体动作类型（one-hot）	LSTM 对局势的整体理解
            输入维度	564（= 动作类型数）	128（= lstm_hidden_dim）
            语义	"我选了攻击"——一个确定的事实	"局势复杂，敌人在逼近，我的兵在左侧"——一个全局的局势感
            条件依赖	依赖前一步的选择结果	不依赖动作选择，纯粹来自观测序列
        '''

        # the add operation may auto broadcasting, so we need an assert test
        '''
        z (batch, 1024)  +  t (batch, 1024)  =  autoregressive_embedding (batch, 1024)
        "我在攻击"           "局势是..."           "在目前局势下，我在攻击"

        加法不是简单拼接——两个向量的对应维度相加，意味着 z 和 t 在同一空间中对齐。模型通过训练学会了：

        z 里的第 i 个维度表示"动作类型相关的第 i 个属性"
        t 里的第 i 个维度表示"局势相关的第 i 个属性"
        加起来就是"动作 + 局势"的联合表示

        todo：为什么叫"自回归嵌入"（autoregressive embedding）？ 因为它在后续 heads 中会被逐步更新。看一下
        '''
        autoregressive_embedding = z + t

        del action_type_one_hot, lstm_output, scalar_context, x, z, t

        '''
        action_type_logits: 预测选择动作的logits分布 shape (batch, max_action_num)
        action_type: 选择的动作（随机采样或者外部传入的专家动作）shape (batch, 1)
        autoregressive_embedding: 动作 + 局势"的联合表示 shape （batch， autoregressive_embedding_size）
        '''
        return action_type_logits, action_type, autoregressive_embedding


def test():
    batch_size = 2
    lstm_output = torch.randn(batch_size * AHP.sequence_length, AHP.lstm_hidden_dim)
    scalar_context = torch.randn(batch_size * AHP.sequence_length, AHP.context_size)
    action_type_head = ActionTypeHead()

    print("lstm_output:", lstm_output) if debug else None
    print("lstm_output.shape:", lstm_output.shape) if debug else None

    print("scalar_context:", scalar_context) if debug else None
    print("scalar_context.shape:", scalar_context.shape) if debug else None

    action_type_logits, action_type, autoregressive_embedding = action_type_head.forward(lstm_output, scalar_context)

    print("action_type_logits:", action_type_logits) if debug else None
    print("action_type_logits.shape:", action_type_logits.shape) if debug else None
    print("action_type:", action_type) if debug else None
    print("action_type.shape:", action_type.shape) if debug else None
    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
