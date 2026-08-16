#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Delay Head."

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

__author__ = "Ruo-Ze Liu"

debug = False


def checkNaNandInf(val, name):
    if torch.isnan(val).any():
        print(name, 'Find nan:', val) if debug else None
    if torch.isinf(val).any():
        print(name, 'Find inf:', val) if debug else None


class DelayHead(nn.Module):
    '''
    Inputs: autoregressive_embedding
    Outputs:
        delay_logits - The logits corresponding to the probabilities of each delay
        delay - The sampled delay
        autoregressive_embedding - Embedding that combines information from `lstm_output` and all previous sampled arguments. 

    在已选定动作类型的基础上，决定"发出这条指令后，等多少游戏步再发下一条指令"——即决定智能体的操作节奏。它输出 128 个候选延迟的概率分布，采样出一个 delay，并把这个 delay 编码回 autoregressive_embedding，喂给后续的 head

    在 StarCraft II 里，游戏每步（game step）都会给 agent 一个观测，agent 也可以每步都发指令。但人类高手并不会每步都操作——发完一条指令后往往要等几帧、十几帧，观察战况再发下一条。

        AlphaStar 论文里用一个专门的参数来建模这个行为，就叫 delay：它的含义是"从上一条指令发出，到发出下一条指令之间，等待的游戏步数"。

        delay = 4：操作很快（暴击流微操）
        delay = 32：正常节奏（StarCraft_Hyper_Parameters.sc2_default_delay = 32，环境里的默认值）
        delay 最大 128：几乎不动
    '''

    def __init__(self, autoregressive_embedding_size=AHP.autoregressive_embedding_size, 
                 original_256=AHP.original_256, max_delay=SFS.last_delay):
        '''
        autoregressive_embedding_size	1024	输入/输出的自回归嵌入维度
        original_256	256	中间隐层维度
        max_delay	128	候选延迟的类别数（delay_logits 的维度）
        '''
        super().__init__()
        self.fc_1 = nn.Linear(autoregressive_embedding_size, original_256)  # with relu
        self.fc_2 = nn.Linear(original_256, original_256)  # with relu
        self.max_delay = max_delay # 最大的延迟，避免无限期延迟，使用one-hot编码，将延迟从实数变成有限范围的整数

        self.embed_fc = nn.Linear(original_256, max_delay)  # no relu

        self.fc_3 = nn.Linear(max_delay, original_256)  # with relu
        self.fc_4 = nn.Linear(original_256, original_256)  # with relu
        self.project = nn.Linear(original_256, autoregressive_embedding_size)

        self.softmax = nn.Softmax(dim=-1)

        self.is_rl_training = False

    def set_rl_training(self, staus):
        self.is_rl_training = staus

    def forward(self, autoregressive_embedding, delay=None):
        '''
        autoregressive_embedding: 传入之前根据游戏资源、地图信息等选择预测的动作+局势的嵌入表示，包含二者的信息 （batch， autoregressive_embedding_size）

        delay：它的含义是"从上一条指令发出，到发出下一条指令之间，等待的游戏步数"。delay = i - record_i   # 上一次记录动作的帧 与 本次动作帧 之差
        RL 推理（forward，第 230 行）——自己采样
        老师强迫（teacher forcing），喂专家标签
        '''

        # 从局势嵌入预测 delay 分布
        # AlphaStar: `autoregressive_embedding` is decoded using a 2-layer (each with size 256) 
        # linear network with ReLUs,
        x = F.relu(self.fc_1(autoregressive_embedding))
        x = F.relu(self.fc_2(x))

        # AlphaStar: before being embedded into `delay_logits` that has size 128 (one for each 
        # possible requested delay in game steps).
        # note: no temperature used here
        # 输入 autoregressive_embedding 此时已经包含了：LSTM 输出 + action_type 的编码。两层 256 维 ReLU 网络解码后，映射到 128 个 logits——每个 logit 对应"等待 k 个游戏步"的分数（k = 0~127）。代码注释里也引用了 AlphaStar 原文："embedded into delay_logits that has size 128 (one for each possible requested delay in game steps)"
        delay_logits = self.embed_fc(x) # shape is （batch， max_delay

        # AlphaStar: `delay` is sampled from `delay_logits` using a multinomial, though unlike all other arguments,
        # no temperature is applied to `delay_logits` before sampling.
        if delay is None: # 采样 delay（无温度）
            delay_probs = self.softmax(delay_logits)        
            delay = torch.multinomial(delay_probs, 1)  # [batch, 1]
            del delay_probs

        # 把 delay 编码回自回归嵌入（关键的自回归机制）
        # AlphaStar: Similar to `action_type`, `delay` is projected to a 1D tensor of size 1024 through 
        # a 2-layer (each with size 256) linear network with ReLUs, and added to `autoregressive_embedding`
        # similar to action_type here, change it to one_hot version
        delay_one_hot = L.tensor_one_hot(delay, self.max_delay) # [batch, 1, 128]
        delay_one_hot = delay_one_hot.squeeze(-2) # [batch, 128]
        z = F.relu(self.fc_3(delay_one_hot))
        z = F.relu(self.fc_4(z))
        t = self.project(z)  # [batch, 1024]

        # the operation may auto broadcasting, so we need a test
        autoregressive_embedding = autoregressive_embedding + t # [batch, 1024]

        del delay_one_hot, x, z, t

        '''
        delay_logits: 预测操作延迟的logits分布
        delay： 实际的操作延迟（采样或者外部传入的专家）
        autoregressive_embedding：在原先游戏资源、地图信息等选择预测的动作+局势的嵌入表示继续增加了操作延迟（下一次什么时候在预测动作操作）的信息
        '''
        return delay_logits, delay, autoregressive_embedding


def test():
    batch_size = 2
    autoregressive_embedding = torch.randn(batch_size, AHP.autoregressive_embedding_size)
    delay_head = DelayHead()

    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    delay_logits, delay, autoregressive_embedding = delay_head.forward(autoregressive_embedding)

    print("delay_logits:", delay_logits) if debug else None
    print("delay_logits.shape:", delay_logits.shape) if debug else None
    print("delay:", delay) if debug else None
    print("delay.shape:", delay.shape) if debug else None
    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
