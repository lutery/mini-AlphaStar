#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Queue Head."

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

__author__ = "Ruo-Ze Liu"

debug = False


class QueueHead(nn.Module):
    '''
    Inputs: autoregressive_embedding, action_type, embedded_entity
    它决定刚选出的这条指令是"立即执行"还是"排队执行"（SC2 的 queued 参数：0=立即，1=加入命令队列）。它和 DelayHead 几乎同构，但有 3 个关键区别：输出只有 2 类、采样前有温度 0.8、且只有动作本身支持排队时，queue 的编码才写回 autoregressive_embedding。
    你在星际里按 Shift 键连续下达指令，后一条指令不会立刻打断当前指令，而是排进命令队列等当前动作完成后自动执行——这就是 queued。它在原始动作接口里是 RawActions 的第一个参数，取值 0（立即执行）或 1（排队执行）。

    为什么模型要专门学它？

    建造/训练场景：一个基地可以同时排队训练多个农民、一条兵营可以排队出多个兵——"排队"让运营更顺滑；
    操作场景：如果所有指令都立即执行，前后两条指令会互相打断（比如移动中要攻击），排队可以把意图序列化。
    但注意：有些动作根本没有 queued 参数（no_op、raw_move_camera、部分释放类技能），对它们而言"是否排队"毫无意义。这个特性后面会引出 QueueHead 最精巧的 mask 机制。
    Outputs:
        queued_logits - The logits corresponding to the probabilities of queueing and not queueing
        queued - Whether or no to queue this action.
        autoregressive_embedding - Embedding that combines information from `lstm_output` and all previous sampled arguments. 
    '''

    def __init__(self, input_size=AHP.autoregressive_embedding_size, 
                 original_256=AHP.original_256,
                 max_queue=SFS.last_repeat_queued, is_sl_training=True, temperature=AHP.temperature):
        super().__init__()
        self.is_sl_training = is_sl_training
        self.temperature = temperature # self.temperature = AHP.temperature = 0.8

        self.fc_1 = nn.Linear(input_size, original_256)  # with relu
        self.fc_2 = nn.Linear(original_256, original_256)  # with relu
        self.max_queue = max_queue

        self.embed_fc = nn.Linear(original_256, max_queue)  # max_queue = SFS.last_repeat_queued = 2——二分类：类别 0 = 立即执行，类别 1 = 排队执行。

        self.fc_3 = nn.Linear(max_queue, original_256)  # with relu
        self.fc_4 = nn.Linear(original_256, original_256)  # with relu
        self.project = nn.Linear(original_256, AHP.autoregressive_embedding_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.is_rl_training = False

    def set_rl_training(self, staus):
        self.is_rl_training = staus

    def forward(self, autoregressive_embedding, action_type, embedded_entity=None, queue=None):
        '''
        autoregressive_embedding：在原先游戏资源、地图信息等选择预测的动作+局势的嵌入表示继续增加了操作延迟（下一次什么时候在预测动作操作）的信息 （batch， autoregressive_embedding_size）
        action_type: 选择的动作（随机采样或者外部传入的专家动作）shape (batch, 1)
        embedded_entity：[batch_size, fc1_output_size]，包含每个样本的总体嵌入表示
        queue： 外部传入的标签（SL teacher forcing 时用） [batch, 1]
        '''
        # AlphaStar: Queued Head is similar to the delay head except a temperature of 0.8 
        # AlphaStar: is applied to the logits before sampling,
        x = self.fc_1(autoregressive_embedding)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)

        # note: temperature is used here, compared to delay head
        # AlphaStar: the size of `queued_logits` is 2 (for queueing and not queueing),
        # 预测下一个操作指令是否排队执行
        queue_logits = self.embed_fc(x) # shape （batch，max_queue）

        # 这里很明显，如果是为了训练，则需要降低不同logits之间的差异
        # todo 温度 < 1 会拉大 logits 差距，让分布更尖锐（更贪心、更接近确定性）。RL 阶段模型已有一定水平，采样希望"稳一点"；SL 阶段要跟专家行为对齐、保留多样性，所以不压尖。
        temperature = self.temperature if self.is_rl_training else 1 # 
        queue_logits = queue_logits / temperature

        # 如果外部没有转入专家数据，则这里自行采样
        if queue is None:
            queue_probs = self.softmax(queue_logits) 
            queue = torch.multinomial(queue_probs, 1)
            del queue_probs

        # similar to action_type here, change it to one_hot version
        # 同样转换成one-hot编码，（batch, 1， max_queue/2）
        queue_one_hot = L.tensor_one_hot(queue, self.max_queue)

        # to make the dim of queue_one_hot as queue
        queue_one_hot = queue_one_hot.squeeze(-2) # shape （batch, max_queue/2）

        z = self.relu(self.fc_3(queue_one_hot))
        z = self.relu(self.fc_4(z))
        t = self.project(z) # shape （batch， autoregressive_embedding_size）

        # AlphaStar: and the projected `queued` is not added to `autoregressive_embedding` 
        # if queuing is not possible for the chosen `action_type`
        # note: projected `queued` is not added to `autoregressive_embedding` if queuing is not 
        # possible for the chosen `action_type`
        mask = L.action_can_be_queued_mask(action_type).float() # shape is (batch, 1) 返回一个表示动作是排队执行的动作还是立即执行的动作矩阵
        # 对比 DelayHead 的 autoregressive_embedding = autoregressive_embedding + t——差别就在多乘了一个 mask。
        # 只有需要排队的动作指令信息加入到autoregressive_embedding，不排队的不用加入（大概率手续这个值会进入一个仅针对排队动作的判断预测器预测
        # 如果混入了不需要排队的动作信息会加大干扰 todo 确认链路）
        autoregressive_embedding = autoregressive_embedding + mask * t
        del queue_one_hot, x, z, t, mask, action_type


        '''
        queue_logits: 根据autoregressive_embedding判断接下来的动作是排队还是立即执行 （batch，max_queue）
        queue: 针对queue_logits的采样或者外部传入的专家数据 （batch，1）
        autoregressive_embedding: 游戏资源、地图信息等选择预测的动作+局势的嵌入+操作延迟（下一次什么时候在预测动作操作）+ 针对执行动作指令action_type是否需要立即执行的掩码信息 （batch， autoregressive_embedding_size）
        '''
        return queue_logits, queue, autoregressive_embedding


def test():
    batch_size = 2
    autoregressive_embedding = torch.randn(batch_size, AHP.autoregressive_embedding_size)
    action_type = torch.randint(low=0, high=SFS.available_actions, size=(batch_size, 1))
    queue_head = QueueHead()

    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    queue_logits, queue, autoregressive_embedding = queue_head.forward(autoregressive_embedding, action_type)

    print("queue_logits:", queue_logits) if debug else None
    print("queue_logits.shape:", queue_logits.shape) if debug else None
    print("queue:", queue) if debug else None
    print("queue.shape:", queue.shape) if debug else None
    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
