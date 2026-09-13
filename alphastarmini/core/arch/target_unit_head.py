#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Target Unit Head."

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

__author__ = "Ruo-Ze Liu"

debug = False


class TargetUnitHead(nn.Module):
    '''
    Inputs: autoregressive_embedding, action_type, entity_embeddings
    Outputs:
        target_unit_logits - The logits corresponding to the probabilities of targeting a unit
        target_unit - The sampled target unit
    '''

    def __init__(self, embedding_size=AHP.entity_embedding_size, 
                 max_number_of_unit_types=SCHP.max_unit_type, 
                 is_sl_training=True, temperature=AHP.temperature,
                 original_256=AHP.original_256, original_32=AHP.original_32,
                 max_selected=1, autoregressive_embedding_size=AHP.autoregressive_embedding_size):
        super().__init__()
        self.is_sl_training = is_sl_training
        self.temperature = temperature

        self.max_number_of_unit_types = max_number_of_unit_types
        self.func_embed = nn.Linear(max_number_of_unit_types, original_256)  # with relu

        self.conv_1 = nn.Conv1d(in_channels=embedding_size, 
                                out_channels=original_32, kernel_size=1, stride=1,
                                padding=0, bias=True)
        self.fc_1 = nn.Linear(autoregressive_embedding_size, original_256)
        self.fc_2 = nn.Linear(original_256, original_32)

        self.small_lstm = nn.LSTM(original_32, original_32, 1, dropout=0.0, batch_first=True)

        # We mostly target one unit
        self.max_selected = 1

        self.softmax = nn.Softmax(dim=-1)

        self.is_rl_training = False

    def set_rl_training(self, staus):
        self.is_rl_training = staus

    def forward(self, autoregressive_embedding, action_type, entity_embeddings, entity_num, target_unit=None):
        '''
        Inputs:
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
            action_type: [batch_size x 1]
            entity_embeddings: [batch_size x entity_size x embedding_size]
            entity_num: [batch_size]
            autoregressive_embedding：游戏资源、地图信息等选择预测的动作+局势的嵌入+操作延迟（下一次什么时候在预测动作操作）+ 针对执行动作指令action_type是否需要立即执行的掩码信息 （batch， autoregressive_embedding_size），新加入了根据动作选择了要操作的实体单位的信息 
            action_type:选择的动作（随机采样或者外部传入的专家动作）shape (batch, 1)
            entity_embeddings: (b, lq/实体数量 512，dim)，包含每个实体的嵌入表示，这个注释一个固定的实体槽位，即实体的嵌入放到这个槽位内，具体有多少个实体由entity_nums决定
            entity_nums:每个样本的有效实体数量
        Output:
            target_unit_logits: [batch_size x max_selected x entity_size]
            target_unit: [batch_size x max_selected x 1]
        '''

        batch_size = entity_embeddings.shape[0]

        # entity_embeddings shape is [batch_size x entity_size x embedding_size]
        entity_size = entity_embeddings.shape[1]

        # `func_embed` is computed the same as in the Selected Units head, 
        # and used in the same way for the query (added to the output of the `autoregressive_embedding` 
        # passed through a linear of size 256). 获取当前动作能作用的兵种
        unit_types_one_hot = L.action_can_apply_to_targeted_mask(action_type)

        device = next(self.parameters()).device
        unit_types_one_hot = unit_types_one_hot.to(device)

        # unit_types_mask shape: [batch_size x self.max_number_of_unit_types]
        # 将当前能够作用的兵种信息进行压缩维度，得到 [batch_size x 256]
        the_func_embed = F.relu(self.func_embed(unit_types_one_hot))

        # the_func_embed shape: [batch_size x 256]
        print("the_func_embed:", the_func_embed) if debug else None
        print("the_func_embed.shape:", the_func_embed.shape) if debug else None

        # generate the length mask for all entities
        mask = torch.arange(entity_size, device=device).float() # shape （entity_size，）
        mask = mask.repeat(batch_size, 1) # shape is (batch_size, entity_size)

        # mask: [batch_size, entity_size]
        mask = mask < entity_num.unsqueeze(dim=1) # 将 mask 转换为bool掩码矩阵，这样就可以得到一个每个样本哪些是有效实体，哪些是无效实体的矩阵
        print("mask:", mask) if debug else None
        print("mask.shape:", mask.shape) if debug else None

        assert mask.dtype == torch.bool

        # Because we mostly target one unit, we don't need a mask.

        # The query is then passed through a ReLU and a linear of size 32, 
        # and the query is applied to the keys which are created the 
        # same way as in the Selected Units head to get `target_unit_logits`.
        # input : [batch_size x entity_size x embedding_size]
        # 同样压缩每个输入实体的维度
        key = self.conv_1(entity_embeddings.transpose(-1, -2)).transpose(-1, -2) #  [batch_size x entity_size x key_size], note key_size = 32

        # output : [batch_size x entity_size x key_size], note key_size = 32
        print("key:", key) if debug else None
        print("key.shape:", key.shape) if debug else None

        # AlphaStar: The query is then passed through a ReLU and a linear of size 32, 
        # and the query is applied to the keys which are created the same way as in 
        # the Selected Units head to get `target_unit_logits`.
        x = self.fc_1(autoregressive_embedding) # 提取全局信息的矩阵
        x = the_func_embed + x # 将当前动作能勾作用的兵种信息嵌入到 autoregressive_embedding 中 [batch_size x 256]
        query = self.fc_2(x).unsqueeze(1) # [batch_size， 1， 32]

        # below is matrix multiply
        # key_shape: [batch_size x entity_size x key_size], note key_size = 32
        # query_shape: [batch_size x seq_len x hidden_size], note hidden_size is also 32, seq_len = 1
        # 这里应该是类似qkv的查询，用实体去对比全局信息，确认当前局势下应该对哪些实体加强关注，哪些实体减少关注
        y = torch.bmm(key, query.transpose(-1, -2))

        # new y shape: [batch_size x entity_size]
        y = y.squeeze(-1)

        # fill the entity which should be selected a very large negetive value 
        target_unit_logits = y.masked_fill(~mask, -1e9) # 用掩码将非实体部分掩盖，target_unit_logits [batch_size x entity_size]

        temperature = self.temperature if self.is_rl_training else 1
        target_unit_logits = target_unit_logits / temperature
        print("target_unit_logits:", target_unit_logits) if debug else None
        print("target_unit_logits.shape:", target_unit_logits.shape) if debug else None

        # AlphaStar: If `action_type` does not involve targetting units, this head is ignored.
        target_unit_mask = L.action_involve_targeting_unit_mask(action_type).bool() # [batch, 1] ，返回是否需要选择一个目标的bool矩阵
        assert len(action_type.shape) == 2  
        assert target_unit_mask.dtype == torch.bool  
        no_target_unit_mask = ~target_unit_mask.squeeze(dim=1) # 取反，得到一个不需要选择目标的bool矩阵 [batch, 1] 

        if target_unit is None: # 如果没有专家数据指定要选择的目标，则直接根据qk的出来的注意力矩阵选择目标
            target_unit_probs = self.softmax(target_unit_logits) # [batch_size x entity_size]
            target_unit = torch.multinomial(target_unit_probs, 1) # [batch_size x 1]
            del target_unit_probs

            target_unit = target_unit.unsqueeze(dim=1) # [batch_size x 1 x 1]
            print("target_unit.shape:", target_unit.shape) if debug else None

            target_unit[no_target_unit_mask, 0] = entity_size - 1  # None index, the same as -1 将不需要选择目标的动作类型对应的样本置为一个占位符
            print("target_unit:", target_unit) if debug else None

        target_unit_logits = target_unit_logits.unsqueeze(dim=1) # [batch_size x 1 x entity_size]
        print("target_unit_logits.shape:", target_unit_logits.shape) if debug else None

        target_unit_logits[no_target_unit_mask] = 0.  # a magic number 同样将不需要选择目标的所有logits分布设置为0

        del x, y, mask, key, query, action_type
        del unit_types_one_hot, the_func_embed, no_target_unit_mask

        '''
        target_unit_logits: 每一个样本根据动作生成的选择实体目标的logits分布（针对 entity_embeddings 选择实体），但是如果动作类型不需要选择目标的全部设置为0 [batch_size x 1 x entity_size]
        target_unit：根据动作选择的目标实体索引（针对 entity_embeddings 选择实体），但是如果动作类型不需要选择目标的全部设置为entity_size - 1 [batch_size x 1 x 1]
        '''
        return target_unit_logits, target_unit


def test():
    action_type_sample = 352  # func: 352/Effect_WidowMineAttack_unit (1/queued [2]; 2/unit_tags [512]; 3/target_unit_tag [512])

    batch_size = 4
    autoregressive_embedding = torch.randn(batch_size, AHP.autoregressive_embedding_size)
    #action_type = torch.randint(low=0, high=SFS.available_actions, size=(batch_size, 1))
    action_type = torch.tensor([[0], [1], [168], [352]])

    entity_embeddings = torch.randn(batch_size, AHP.max_entities, AHP.entity_embedding_size)
    entity_nums = torch.tensor([1, 2, 3, 12])

    target_units_head = TargetUnitHead()

    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    target_unit_logits, target_unit = \
        target_units_head.forward(autoregressive_embedding, action_type, entity_embeddings, entity_nums)

    if target_unit_logits is not None:
        print("target_unit_logits:", target_unit_logits) if debug else None
        print("target_unit_logits.shape:", target_unit_logits.shape) if debug else None
    else:
        print("target_unit_logits is None!")

    if target_unit is not None:
        print("target_unit:", target_unit) if debug else None
        print("target_unit.shape:", target_unit.shape) if debug else None
    else:
        print("target_unit is None!")

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
