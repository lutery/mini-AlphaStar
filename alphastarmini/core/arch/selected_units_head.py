#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Selected Units Head."

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

import param as P

__author__ = "Ruo-Ze Liu"

debug = False


class SelectedUnitsHead(nn.Module):
    '''
    Inputs: autoregressive_embedding, action_type, entity_embeddings
    Outputs:
        units_logits - The logits corresponding to the probabilities of selecting each unit, 
            repeated for each of the possible 64 unit selections
        units - The units selected for this action.
        autoregressive_embedding - Embedding that combines information from `lstm_output` and all previous sampled arguments.

    是一个**"序列式实体选择器"（pointer-network 风格）：给定动作类型和场上所有实体的嵌入，它逐个、最多 12 次**挑选要选中的单位（每轮只选一个，选过的不能再选，可以中途选一个特殊的"结束符"EOF 提前终止），并在每轮之间用一个小 LSTM 记住"已经选了谁"，从而支持"一次框选多个单位"这种星际操作。

    SC2 的一条指令里，unit_tags 参数可能包含多个单位（比如框选 12 个农民去采气、选中 6 个 Gateway 一起出狂徒）。这意味着模型输出不是一个单位 ID，而是一个长度可变的单位序列：
    选中集合 = [单位₅, 单位₉, 单位₁₃, （结束）]     ← 序列长度由模型自己决定

    这就引出三个子问题，也是本文件的三个核心机制：

        选几个？ → 用 EOF（End-Of-File，序列生成里叫 EOS）特殊符号：模型随时可以选"结束"来终止；
        先后依赖？ → 选第 2 个时要"记得"已选了第 1 个（不能重复、且策略上比如"选中了基地就该继续选农民"）→ 小 LSTM 维护记忆；
        什么能选？ → 三重约束：长度掩码（场上实际存在的实体）、类型先验（func_embed 软引导 + 可选实体掩码硬屏蔽）、去重掩码。
    '''

    def __init__(self, embedding_size=AHP.entity_embedding_size, 
                 max_number_of_unit_types=SCHP.max_unit_type, is_sl_training=True, 
                 temperature=AHP.temperature, max_selected=AHP.max_selected,
                 original_256=AHP.original_256, original_32=AHP.original_32,
                 autoregressive_embedding_size=AHP.autoregressive_embedding_size,
                 use_unit_type_entity_mask=AHP.use_unit_type_entity_mask):
        super().__init__()
        self.is_sl_training = is_sl_training
        self.temperature = temperature

        self.max_number_of_unit_types = max_number_of_unit_types
        # 动作→合法兵种类型的先验注入，这里是注入当前选择要执行的动作所能操作对象类型的矩阵
        # 得到这个可操作对象矩阵的嵌入表示
        self.func_embed = nn.Linear(max_number_of_unit_types, original_256)  # with relu

        # PyTorch 的 Conv1d 规定输入必须是 [batch, channels, length]——通道维（channels）在倒数第二维，序列长度在最后一维。
        # kernel_size=1 意味着卷积核只看当前位置自己，不跨越相邻位置。于是对每个位置 i：Conv1d:  y[:, :, i] = W(16×64) · x[:, :, i] + b
        # # 等价写法（结果一模一样，只是写法不同）
        # key = nn.Linear(64, 16)(entity_embeddings)   # [batch, 512, 16]
        # 为什么必须"互不影响"？因为 512 个实体槽位是一个集合，不是序列——实体在槽位里的排列顺序是任意的（这一帧 Zealot 排第 3，下一帧可能排第 7）。如果卷积核 > 1，模型就会利用"相邻槽位"的信息，学到虚假的、依赖排序的规律。kernel=1 保证了置换不变性：无论实体怎么排序，每个实体得到的 key 只取决于它自己的嵌入。
        self.conv_1 = nn.Conv1d(in_channels=embedding_size, 
                                out_channels=original_32, kernel_size=1, stride=1,
                                padding=0, bias=True)

        self.fc_1 = nn.Linear(autoregressive_embedding_size, original_256)
        self.fc_2 = nn.Linear(original_256, original_32)

        '''
        想象人类玩家框选单位：先点一个 Zealot，再点一个 Stalker，再点一个哨兵……第 2 个选择依赖第 1 个：

        硬约束：同一个单位不能重复选（代码里用 mask 去重）；
        软策略：选了 3 个 Zealot 后，第 4 个可能想选个哨兵（开护盾）而不是第 4 个 Zealot——这是"组合"层面的决策。
        如果每一轮都从零开始、只看当前 embedding，模型就只能学"每个单位独立地看谁最合适"，学不到"组合搭配"。LSTM 的 hidden 状态就是为此服务的：它把"已经选了哪些单位"压缩成一个 16 维记忆，参与下一轮 query 的生成。

        （补充：autoregressive_embedding 里其实也写回了上一轮选中的实体信息，但那是"显式"的、经 project 变换的全局信号；LSTM 的 hidden 是"隐式"的、专门为这个选择序列服务的记忆，两者互补。）
        '''
        self.small_lstm = nn.LSTM(input_size=original_32, hidden_size=original_32, num_layers=1, 
                                  dropout=0.0, batch_first=True)

        self.max_selected = max_selected

        self.project = nn.Linear(original_32, autoregressive_embedding_size)
        self.softmax = nn.Softmax(dim=-1)

        # init a new tensor corrspond to end selection (also called EOF in NLP)
        # referenced by https://github.com/opendilab/DI-star in action_arg_head.py
        self.new_variable = nn.Parameter(torch.FloatTensor(original_32))
        nn.init.uniform_(self.new_variable, b=0.)
        #nn.init.uniform_(self.new_variable, b=1. / original_32)

        self.is_rl_training = False

        self.use_unit_type_entity_mask = use_unit_type_entity_mask

    def set_rl_training(self, staus):
        self.is_rl_training = staus

    def forward(self, autoregressive_embedding, action_type, entity_embeddings, entity_num, unit_type_entity_mask=None):
        '''
        autoregressive_embedding: 游戏资源、地图信息等选择预测的动作+局势的嵌入+操作延迟（下一次什么时候在预测动作操作）+ 针对执行动作指令action_type是否需要立即执行的掩码信息 （batch， autoregressive_embedding_size）
        action_type: 选择的动作（随机采样或者外部传入的专家动作）shape (batch, 1)
        entity_embeddings: (b, lq/实体数量 512，dim)，包含每个实体的嵌入表示，这里应该是整个画面上展示的所有实体
        entity_num：每个样本的有效实体数量
        unit_type_entity_mask：主动根据选择的动作，从观察列表中判断执行动作能够影响到选择到的实体掩码

        Inputs:
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
            action_type: [batch_size x 1]
            entity_embeddings: [batch_size x entity_size x embedding_size]
            entity_num: [batch_size]
        Output:
            units_logits: [batch_size x max_selected x entity_size]
            units: [batch_size x max_selected x 1]
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
        '''
        batch_size = entity_embeddings.shape[0]
        entity_size = entity_embeddings.shape[-2] # 等于 lq/实体数量 512
        device = next(self.parameters()).device
        key_size = self.new_variable.shape[0]
        original_ae = autoregressive_embedding

        # AlphaStar: If applicable, Selected Units Head first determines which entity types can accept `action_type`,
        # creates a one-hot of that type with maximum equal to the number of unit types,
        # and passes it through a linear of size 256 and a ReLU. This will be referred to in this head as `func_embed`.
        # QUESTION: one unit type or serveral unit types?
        # ANSWER: serveral unit types, each for one-hot
        # This is some places which introduce much human knowledge
        # （batch， ConstSize.All_Units_Size），其中0表示不可操作对象类型，1表示可以操作对象类型
        unit_types_one_hot = L.action_can_apply_to_selected_mask(action_type).to(device)

        # the_func_embed shape: [batch_size x 256]
        # 提取聚合到一个执行动作->可选择对象类型的特征嵌入
        # shape is (batch_size, original_256)
        the_func_embed = F.relu(self.func_embed(unit_types_one_hot))
        del unit_types_one_hot

        # AlphaStar: It also computes a mask of which units can be selected, initialised to allow selecting all entities 
        # that exist (including enemy units).
        # generate the length mask for all entities
        # mask shape is (entity_size，)
        mask = torch.arange(entity_size, device=device).float()
        # mask shape is (batch_size, entity_size，)
        mask = mask.repeat(batch_size, 1)

        # now the entity nums should be added 1 (including the EOF)
        # this is because we also want to compute the mean including key value of the EOF
        added_entity_num = entity_num + 1 # 这里是将EOF槽也算进去了，EOF代表选择对象结束的标识

        # mask: [batch_size, entity_size]
        # mask 代表是当前画面所有可操作的对象，而added_entity_num代表有效可以操作的实体对象
        # 所以这里是提前进行掩码，对每一个样本限制在有效对象内进行掩码
        mask = mask < added_entity_num.unsqueeze(dim=1)
        assert mask.dtype == torch.bool

        # AlphaStar: It then computes a key corresponding to each entity by feeding `entity_embeddings`
        # through a 1D convolution with 32 channels and kernel size 1,
        # and creates a new variable corresponding to ending unit selection.
        # input: [batch_size x entity_size x embedding_size]
        # output: [batch_size x entity_size x key_size], note key_size = 32
        # entity_embeddings：(b, lq/实体数量 512，dim)
        # entity_embeddings.transpose(-1, -2)：（batch_size, dim, lq/实体数量）
        # self.conv_1：把每个实体的 64 维嵌入，通过一个"逐实体独立"的线性变换，压缩成 16 维的 key 向量。 （batch_size, original_32, lq/实体数量）
        # .transpose(-1, -2)：（batch_size, lq/实体的数量，original_32）
        # 这里是进一步压碎每一个实体的嵌入表示
        # key这里是进一步压缩的每一个实体的向量表示
        key = self.conv_1(entity_embeddings.transpose(-1, -2)).transpose(-1, -2)

        # end index should be the same to the entity_num
        end_index = entity_num

        # replace the EOF with the new_variable 
        # use calculation to achieve it
        if False:
            key[torch.arange(batch_size), end_index] = self.new_variable
        else:
            # padding_end shape (batch_size, 1， original_32) 全零矩阵
            padding_end = torch.zeros(key.shape[0], 1, key.shape[2], dtype=key.dtype, device=key.device)
            # key shape （batch_size, lq/实体的数量，original_32），这里是将最后一个位置替换为填充0
            key = torch.cat([key[:, :-1, :], padding_end], dim=1)

            # flag 全1矩阵 shape （batch_size, lq/实体的数量，original_32）
            flag = torch.ones(key.shape, dtype=torch.bool, device=key.device)
            # 根据有效实体的数量，在对应位置设置False，表示到这里就结束了
            flag[torch.arange(batch_size), end_index] = False

            # [batch_size, entity_size, key_size]
            # 这段代码在做一个**"可微的定点替换"：把每个样本 entity_num 位置（EOF 结束符槽位）的 key 向量，替换成可学习的参数 new_variable。之所以不用一行 key[..., end_index] = new_variable 直接赋值，是因为索引赋值是 in-place 操作，会破坏反向传播**；所以作者改用"布尔 flag + 乘法"的纯张量运算，效果相同但梯度畅通。
            # torch.ones(key.shape, dtype=key.dtype, device=key.device)： shape （batch_size, lq/实体的数量，original_32）
            # self.new_variable （1， original_32）
            # end_embedding：（batch_size, lq/实体的数量，original_32），全是new_variable
            end_embedding = torch.ones(key.shape, dtype=key.dtype, device=key.device) * self.new_variable.reshape(1, -1)
            # 通过~flag，key_end_part shape 虽然是（batch_size, lq/实体的数量，original_32），但是里面的值仅留着只留 EOF 槽位的值
            key_end_part = end_embedding * ~flag

            # use calculation to replace new_variable
            key_main_part = key * flag # 这里使用相乘，EOF结束为止就变成了0，其余为止保持不变
            key = key_main_part + key_end_part # 这里将EOF位置的向量替换为new_variable可学习的张量
            #  （batch_size, lq/实体的数量，original_32）

            del padding_end, flag, end_embedding, key_main_part, key_end_part

        # calculate the average of keys (consider the entity_num)
        # mask [batch_size, entity_size]
        # mask.unsqueeze(dim=2)：[batch_size, entity_size = lq/实体数量 512， 1]
        # .repeat(1, 1, key.shape[-1])：[batch_size, entity_size， original_32]
        key_mask = mask.unsqueeze(dim=2).repeat(1, 1, key.shape[-1])
        # key （batch_size, lq/实体的数量，original_32）
        # key_mask [batch_size, entity_size = lq/实体数量 512， original_32]
        # key * key_mask：这两个相乘，进一步将有效对象和无效对象区分开 batch_size, lq/实体的数量，original_32）
        # torch.sum：将所有实体对象的嵌入合起来：（batch_size, 1，original_32）
        #  / entity_num.reshape(batch_size, 1)：将sum合起来的嵌入表示取平均值，shape （batch_size, 1，original_32）
        # 这里就有点像rag中每个样本总体有效对象的嵌入表示
        key_avg = torch.sum(key * key_mask, dim=1) / entity_num.reshape(batch_size, 1)
        del key_mask

        # creates a new variable corresponding to ending unit selection.
        # QUESTION: how to do that?
        # ANSWER: referred by the DI-star project, please see self.new_variable in init() method
        # todo 以下几个对象的作用
        units_logits = [] # 存储每一次选择对象预测时的logits分布
        units = [] # 存储每一次选择对象的实体索引
        hidden = None # LSTM的隐藏状态，
 
        # referneced by DI-star
        # represented which sample in the batch has end the selection
        # note is_end should be bool type to make sure it is a right whether mask 
        # todo 看起来是构建一个结束为止的张量，shape （batch_size,），初始全false
        # 确认选择对象
        is_end = torch.zeros(batch_size, device=device).bool()

        # in the first selection, we should not select the end_index
        # mask [batch_size, entity_size = lq/实体数量 512]
        # 这里是将EOF为止的mask设置为False todo为啥？
        mask[torch.arange(batch_size), end_index] = False

        # if we stop selection early, we should record in each sample we select how many items
        # torch.ones(batch_size, dtype=torch.long, device=device)： shape （batch_size，） 全1矩阵
        # * self.max_selected：构建一个能够选择最大实体数量的矩阵
        # 主要用来记录已经选择的实体数量
        select_units_num = torch.ones(batch_size, dtype=torch.long, device=device) * self.max_selected

        # AlphaStar: repeated for selecting up to 64 units
        # 提前结束的处理核心是三个机制：① 用 is_end 布尔标记记录"哪些样本已选中 EOF"；② 用 ~is_end 掩码阻止已结束样本把选中单位写回 autoregressive_embedding；③ 用 select_units_num[last_index] = i 只给结束样本记录实际选中数。而 units_logits / units 是无条件 append 的（没有专门阻止），靠 select_units_num 在后续阶段截断。
        for i in range(self.max_selected):
            if i == 1:
                # todo 这里为啥有设置会True
                mask[torch.arange(batch_size), end_index] = True  # in the second selection, we can select the EOF
                if self.is_rl_training and unit_type_entity_mask is not None:
                    unit_type_entity_mask[torch.arange(batch_size), end_index] = True

            # 进一步将`聚合到一个执行动作->可选择对象类型的特征嵌入`加入到autoregressive_embedding中
            x = self.fc_1(autoregressive_embedding) # （batch， original_256）
            x = self.fc_2(F.relu(x + the_func_embed)).unsqueeze(dim=1) # （batch， 1， original_32）

            # 这行代码是"选单位循环的记忆核心"：一个小 LSTM 每轮接收当前解码状态 x，输出一个 query（用于给 512 个实体打分），同时把内部状态 hidden 传到下一轮——这样第 2 次选单位时，模型"记得"第 1 次选了谁，多单位选择就变成了有先后依赖的序列决策。
            query, hidden = self.small_lstm(x, hidden)
            # key：（batch_size, lq/实体的数量，original_32）
            # query：（batch， 1， original_32）
            # query * key： （batch_size, lq/实体的数量，original_32）
            # torch.sum：（batch_size, lq/实体的数量）结合当前局势状态、历史状态得到权重系数，乘以每一个实体求和得到实体的分数
            # query 形状 [batch, 1, 16]，key 形状 [batch, 512, 16]，广播相乘后对最后一维求和——每个实体槽位得到一个点积分数。这就是 Pointer Network 的注意力打分：
            # query = "我现在想找什么样的单位"（由当前局势 + 已选历史决定）
            # key   = "每个单位是什么"（由实体嵌入 conv 而来）
            # y     = 两者匹配度
            y = torch.sum(query * key, dim=-1)

            # 将非实体的对象全部mask
            entity_logits = y.masked_fill(~mask, -1e9)
            if self.is_rl_training and self.use_unit_type_entity_mask and unit_type_entity_mask is not None:
                # 这里是外部主动传入可以操作的对象掩码
                entity_logits = entity_logits.masked_fill(~unit_type_entity_mask, -1e9)

            temperature = self.temperature if self.is_rl_training else 1
            entity_logits = entity_logits / temperature
            del x, y, query

            entity_probs = self.softmax(entity_logits) # 为每一个对象打分 （batch_size, lq/实体的数量）
            entity_id = torch.multinomial(entity_probs, 1) # 为每一个样本进行抽样，选择一个实体对象（batch_size, 1）

            # 这两行没有用 is_end 做任何屏蔽——已结束样本在后续轮次的采样结果照样被 append 进去。
            # 靠 select_units_num 在后续阶段截断
            units_logits.append(entity_logits.unsqueeze(-2)) # （batch_size, 1, lq/实体的数量）
            units.append(entity_id.unsqueeze(-2)) # （batch_size, 1，1）

            # 已经选择的对象不再进行选择，对应位置的mask设置为False
            # 而 is_end[last_index] = 1 之所以不会被后续预测影响，靠的是一条完整的因果链：EOF 槽位被 mask 屏蔽 → logits 被填成 -1e9 → softmax 后概率恰好为 0 → multinomial 永远采不到它 → 后续轮次 last_index 恒为 False → 不再触发任何写入。此外 is_end 本身是"只置 1、永不置 0"的单向棘轮，天然防覆盖。
            mask[torch.arange(batch_size), entity_id.squeeze(dim=1)] = False  # masked out so that it cannot be selected in future iterations.

            last_index = (entity_id.squeeze(dim=1) == end_index) # 这里代表预测到的结束位置
            is_end[last_index] = 1 # 对应样本是否结束选择的标识设置为1，只有检测到结束选择是才会设置为1，如果没有结束，由于last_index时false则不会设置为1

            # we record how many items we select in a sample
            # we select i + 1 items, but this include the EOF, so actually items should be i + 1 - 1
            select_units_num[last_index] = i # 更新对应样本已经选择的实体数量，这里也如上所说，只会更新选为end的样本

            # AlphaStar: The one-hot position of the selected entity is multiplied by the keys, 
            # reduced by the mean across the entities, passed through a linear layer of size 1024, 
            # and added to `autoregressive_embedding` for subsequent iterations. 
            entity_one_hot = L.tensor_one_hot(entity_id, entity_size).squeeze(-2)
            entity_one_hot_unsqueeze = entity_one_hot.unsqueeze(-2) 

            # 从 512 个实体的 key 矩阵里，精确"取出"本轮采样选中的那一个实体的 16 维 key 向量。实现方式是用一个 one-hot 向量做矩阵乘
            # key shape is （batch_size, lq/实体的数量，original_32）
            # entity_one_hot_unsqueeze shape is （batch_size, 1, lq/实体的数量)
            # torch.bmm(entity_one_hot_unsqueeze, key) shape is （batch_size, 1，original_32）
            # out shape is （batch_size，original_32）
            # 上一轮循环里，模型已经采样出 entity_id（[batch, 1]，本轮选中的实体下标）。现在要把"选中了谁"这个信息写回 autoregressive_embedding，让下一轮选择"知道"这一轮的结果。写回的第一步，就是拿到这个被选中实体的 key 向量。
            # 注意one-hot就是在某个位置为1，其余位置为0，所以可以用来查表提取数据
            # one-hot 向量乘矩阵，结果就是"第 j 行"——这就是线性代数里的"行选择"。用大白话说：one-hot 里唯一那个 1 的位置，决定了从 512 行里挑出哪一行。
            '''
            为什么用 one-hot 矩阵乘，而不是直接索引？
            你可能会问：key[torch.arange(batch), entity_id.squeeze(1)] 一行不就拿到了吗？确实可以，但这里用 one-hot 有两个理由：

            忠实于 AlphaStar 论文原文：论文写的就是 "The one-hot position of the selected entity is multiplied by the keys"，代码逐字复刻（代码注释第 220 行也引用了这句）。
            形式统一、便于扩展：one-hot 乘矩阵是"软选择"的通用形式。如果将来把 one-hot 换成概率分布（软注意力），bmm 写法不用改，直接换成分布向量即可；索引写法则完全行不通。

            注意一个细节：entity_id 是 multinomial 采样出来的离散值，tensor_one_hot 是用索引构造的（eye()[labels]），所以梯度不会穿过 entity_id——这正是期望的（离散选择不可导，policy gradient 会从别处处理）。这里 one-hot 纯粹是"选择器"，不是可微变量。
            '''
            out = torch.bmm(entity_one_hot_unsqueeze, key).squeeze(-2) # 拿到我选择到的实体压缩嵌入向量
            out = out - key_avg # 剪掉全场实体压缩嵌入的均值，减去它，写回的信息就变成"这个实体相对全场平均水平有多特别"——中心化后信号更干净，避免绝对尺度的漂移。
            t = self.project(out) # 还原为实体嵌入的维度
            # ~is_end：如果本轮选的是 EOF（结束符），不写回（EOF 没有实体信息，写回会污染后续）。在批量处理中也是将某些提前结束的样本数据避免持续加入到autoregressive_embedding污染
            # 否则就将选择的实体信息加入到全局的信息中
            autoregressive_embedding = autoregressive_embedding + t * ~is_end.unsqueeze(dim=1)

            if P.skip_autoregressive_embedding: # 超参数，如果不考虑全局的信息
                autoregressive_embedding = autoregressive_embedding - autoregressive_embedding
                autoregressive_embedding[:] = 0.

            del temperature, entity_logits, entity_probs, entity_id
            del last_index, entity_one_hot, entity_one_hot_unsqueeze, out, t

            # 这种"部分样本提前结束、整批继续跑"的设计，是为了保持 batch 内循环次数一致，方便张量并行——代价是已结束样本多做几轮无用的采样计算。
            if is_end.all():# 如果所有的样本都结束选择了就直接结束循环不再选择
                break
        # 通过以上循环选择，将需要选择的单位信息都加入到了 autoregressive_embedding中年

        # units_logits: [batch_size x select_units x entity_size]
        units_logits = torch.cat(units_logits, dim=1) # 将每次预测的选择logit分布组合起来

        # units: [batch_size x select_units x 1]
        units = torch.cat(units, dim=1) # 将每次预测选择的实体id组合起来

        # we use zero padding to make units_logits has the size of [batch_size x max_selected x entity_size]
        # TODO: change the padding
        padding_size = self.max_selected - units_logits.shape[1] # 这里是如果所有的样本都提前结束选择，则构建一个padding矩阵，将
        # units_logits和 units的第二个维度凑齐 max_selected的长度
        if padding_size > 0:
            pad_units_logits = torch.ones(units_logits.shape[0], padding_size, units_logits.shape[2],
                                          dtype=units_logits.dtype, device=units_logits.device) * (-1e9)
            units_logits = torch.cat([units_logits, pad_units_logits], dim=1) 

            pad_units = torch.zeros(units.shape[0], padding_size, units.shape[2],
                                    dtype=units.dtype, device=units.device)
            pad_units[:, :, 0] = entity_size - 1  # None index, the same as -1
            units = torch.cat([units, pad_units], dim=1)
            del pad_units, pad_units_logits

        # AlphaStar: If `action_type` does not involve selecting units, this head is ignored.

        # select_unit_mask: [batch_size x 1]
        # note select_unit_mask should be bool type to make sure it is a right whether mask 
        # 根据执行的动作选择该动作是否能够选择单位的掩码矩阵
        select_unit_mask = L.action_involve_selecting_units_mask(action_type).bool()

        # 取反获取无法支持选择单位的动作矩阵
        no_select_units_index = ~select_unit_mask.squeeze(dim=1)
        print("no_select_units_index:", no_select_units_index) if debug else None

        # 对于无法支持选择动作的样本，将其选择的相关信息置为0或者空
        select_units_num[no_select_units_index] = 0 # 选择0个单位
        #autoregressive_embedding[no_select_units_index] = original_ae[no_select_units_index]

        units_logits[no_select_units_index] = -1e9  # a magic number 实体分布预测全部只为极小值
        units[no_select_units_index, :, 0] = entity_size - 1  # None index, the same as -1 。entity_size - 1（= 512 − 1 = 511）是一个**“None / 空槽位”的占位标记**
        # 最后一个槽位。正常对局单位数远小于 512，这个槽位几乎总是 padding 槽；且它仍是非负、在合法范围内的整数，后续任何代码拿到它都不会崩
        # 更多看md文档

        print("select_units_num:", select_units_num) if debug else None
        print("autoregressive_embedding:", autoregressive_embedding) if debug else None

        del select_unit_mask, no_select_units_index, mask, is_end, key, key_avg

        '''
        units_logits shape is [batch_size x max_selected x entity_size] 其中有部分是padding，如果没有选择满最大选择实体单位的话，表示每次预测的实体logit分布
        units：[batch_size x select_units x 1] 其中有部分是padding，如果没有选择满最大选择实体单位的话，表示每次预测采样的实体id
        autoregressive_embedding：游戏资源、地图信息等选择预测的动作+局势的嵌入+操作延迟（下一次什么时候在预测动作操作）+ 针对执行动作指令action_type是否需要立即执行的掩码信息 （batch， autoregressive_embedding_size），新加入了根据动作选择了要操作的实体单位的信息 todo 但是有些动作无法选择单位，这样混进去会不会有问题？可能会根据units_logits、units、select_units_num来影响拉回吧
        select_units_num：【batch_size, 1]，存储每个样本选择了多少实体
        '''
        return units_logits, units, autoregressive_embedding, select_units_num

    def mimic_forward(self, autoregressive_embedding, action_type, entity_embeddings, entity_num, units, select_units_num,
                      show=False, unit_type_entity_mask=None):
        '''
        Inputs:
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
            action_type: [batch_size x 1]
            entity_embeddings: [batch_size x entity_size x embedding_size]
            entity_num: [batch_size]
        Output:
            units_logits: [batch_size x max_selected x entity_size]
            units: [batch_size x max_selected x 1]
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
        '''
        batch_size = entity_embeddings.shape[0]
        entity_size = entity_embeddings.shape[-2]
        device = next(self.parameters()).device
        key_size = self.new_variable.shape[0]
        original_ae = autoregressive_embedding

        # AlphaStar: If applicable, Selected Units Head first determines which entity types can accept `action_type`,
        # creates a one-hot of that type with maximum equal to the number of unit types,
        # and passes it through a linear of size 256 and a ReLU. This will be referred to in this head as `func_embed`.
        # QUESTION: one unit type or serveral unit types?
        # ANSWER: serveral unit types, each for one-hot
        # This is some places which introduce much human knowledge
        unit_types_one_hot = L.action_can_apply_to_selected_mask(action_type).to(device)

        # the_func_embed shape: [batch_size x 256]
        the_func_embed = F.relu(self.func_embed(unit_types_one_hot))  
        del unit_types_one_hot

        # AlphaStar: It also computes a mask of which units can be selected, initialised to allow selecting all entities 
        # that exist (including enemy units).
        # generate the length mask for all entities
        mask = torch.arange(entity_size, device=device).float()
        mask = mask.repeat(batch_size, 1)

        # now the entity nums should be added 1 (including the EOF)
        # this is because we also want to compute the mean including key value of the EOF
        added_entity_num = entity_num + 1

        # mask: [batch_size, entity_size]
        mask = mask < added_entity_num.unsqueeze(dim=1)
        print("mask:", mask) if debug else None
        print("mask.shape:", mask.shape) if debug else None

        assert mask.dtype == torch.bool

        # AlphaStar: It then computes a key corresponding to each entity by feeding `entity_embeddings`
        # through a 1D convolution with 32 channels and kernel size 1,
        # and creates a new variable corresponding to ending unit selection.

        # input: [batch_size x entity_size x embedding_size]
        # output: [batch_size x entity_size x key_size], note key_size = 32
        key = self.conv_1(entity_embeddings.transpose(-1, -2)).transpose(-1, -2)

        # end index should be the same to the entity_num
        end_index = entity_num

        # replace the EOF with the new_variable 
        # use calculation to achieve it
        if False:
            key[torch.arange(batch_size), end_index] = self.new_variable
        else:
            padding_end = torch.zeros(key.shape[0], 1, key.shape[2], dtype=key.dtype, device=key.device)
            key = torch.cat([key[:, :-1, :], padding_end], dim=1)

            flag = torch.ones(key.shape, dtype=torch.bool, device=key.device)
            flag[torch.arange(batch_size), end_index] = False

            # [batch_size, entity_size, key_size]
            end_embedding = torch.ones(key.shape, dtype=key.dtype, device=key.device) * self.new_variable.reshape(1, -1)
            key_end_part = end_embedding * ~flag

            # use calculation to replace new_variable
            key_main_part = key * flag
            key = key_main_part + key_end_part
            del padding_end, flag, end_embedding, key_main_part, key_end_part

        # calculate the average of keys (consider the entity_num)
        key_mask = mask.unsqueeze(dim=2).repeat(1, 1, key.shape[-1])
        key_avg = torch.sum(key * key_mask, dim=1) / entity_num.reshape(batch_size, 1)
        del key_mask

        # creates a new variable corresponding to ending unit selection.
        # QUESTION: how to do that?
        # ANSWER: referred by the DI-star project, please see self.new_variable in init() method
        units_logits_list = []
        hidden = None

        # consider the EOF
        select_units_num = select_units_num + 1

        # designed with reference to DI-star
        max_seq_len = select_units_num.max()

        # for select_units_num
        selected_mask = torch.arange(max_seq_len, device=device).float()
        selected_mask = selected_mask.repeat(batch_size, 1)

        # mask: [batch_size, max_seq_len]
        selected_mask = selected_mask < select_units_num.unsqueeze(dim=1)
        assert selected_mask.dtype == torch.bool

        # in the first selection, we should not select the end_index
        mask[torch.arange(batch_size), end_index] = False

        is_end = torch.zeros(batch_size, device=device).bool()

        # designed with reference to DI-star
        for i in range(max_seq_len):
            if i != 0:
                # in the second selection, we can select the EOF
                if i == 1:
                    mask[torch.arange(batch_size), end_index] = True
                    if self.is_rl_training and unit_type_entity_mask is not None:
                        unit_type_entity_mask[torch.arange(batch_size), end_index] = True

            # AlphaStar: the network passes `autoregressive_embedding` through a linear of size 256,
            x = self.fc_1(autoregressive_embedding)

            # AlphaStar: adds `func_embed`, and passes the combination through a ReLU and a linear of size 32.
            # x shape: [batch_size x seq_len x 32], note seq_len = 1
            x = self.fc_2(F.relu(x + the_func_embed)).unsqueeze(dim=1)

            # AlphaStar: The result is fed into a LSTM with size 32 and zero initial state to get a query.
            query, hidden = self.small_lstm(x, hidden)
            y = torch.sum(query * key, dim=-1)

            # original mask usage is wrong, we should not let 0 * logits, zero value logit is still large! 
            # we use a very big negetive value replaced by logits, like -1e9
            # y shape: [batch_size x entity_size]
            entity_logits = y.masked_fill(~mask, -1e9)
            if self.is_rl_training and self.use_unit_type_entity_mask and unit_type_entity_mask is not None:
                entity_logits = entity_logits.masked_fill(~unit_type_entity_mask, -1e9)

            temperature = 1  # self.temperature if self.is_rl_training else 1
            entity_logits = entity_logits / temperature
            del x, y, query, temperature

            # note, we add a dimension where is in the seq_one to help
            # we concat to the one : [batch_size x max_selected x ?]
            units_logits_list.append(entity_logits.unsqueeze(-2))
            del entity_logits

            # the last EOF should not be considered
            if i != max_seq_len - 1:

                entity_id = units[:, i]
                print('entity_id', entity_id[0]) if show else None
                print('entity_id.shape', entity_id[0].shape) if show else None

                last_index = (entity_id.squeeze(dim=1) == end_index)
                is_end[last_index] = 1

                # AlphaStar: That entity is masked out so that it cannot be selected in future iterations.
                mask[torch.arange(batch_size), entity_id.squeeze(dim=1)] = False
                print('mask', mask[0]) if show else None

                # AlphaStar: The one-hot position of the selected entity is multiplied by the keys, 
                # reduced by the mean across the entities, passed through a linear layer of size 1024, 
                # and added to `autoregressive_embedding` for subsequent iterations. 
                entity_one_hot = L.tensor_one_hot(entity_id, entity_size).squeeze(-2)
                entity_one_hot_unsqueeze = entity_one_hot.unsqueeze(-2) 

                # entity_one_hot_unsqueeze shape: [batch_size x seq_len x entity_size], note seq_len =1 
                # key_shape: [batch_size x entity_size x key_size], note key_size = 32
                out = torch.bmm(entity_one_hot_unsqueeze, key).squeeze(-2)

                # AlphaStar: reduced by the mean across the entities,
                out = out - key_avg

                # t shape: [batch_size, autoregressive_embedding_size]
                t = self.project(out)

                # TODO, whether should be select_mask[:, i + 1] or select_mask[:, i] ?
                autoregressive_embedding = autoregressive_embedding + t * selected_mask[:, i + 1].unsqueeze(dim=1)
                if P.skip_autoregressive_embedding:
                    autoregressive_embedding = autoregressive_embedding - autoregressive_embedding
                    autoregressive_embedding[:] = 0.

                del t, out, entity_one_hot_unsqueeze, entity_one_hot, last_index, entity_id

                print("autoregressive_embedding:", autoregressive_embedding) if debug else None

        # in SL, we make the selected can have 1 more, like 12 + 1
        max_selected = self.max_selected + 1
        units_logits_size = len(units_logits_list)

        if units_logits_size >= max_selected:
            # remove the last one
            units_logits = torch.cat(units_logits_list[:max_selected], dim=1)
        elif units_logits_size > 0 and units_logits_size < max_selected:
            units_logits = torch.cat(units_logits_list, dim=1)
            padding_size = max_selected - units_logits.shape[1]
            if padding_size > 0:
                pad_units_logits = torch.ones(units_logits.shape[0], padding_size, units_logits.shape[2],
                                              dtype=units_logits.dtype, device=units_logits.device) * (-1e9)
                units_logits = torch.cat([units_logits, pad_units_logits], dim=1)
        else:
            units_logits = torch.ones(batch_size, max_selected, entity_size,
                                      dtype=action_type.dtype, device=action_type.device) * (-1e9)

        # AlphaStar: If `action_type` does not involve selecting units, this head is ignored.

        # select_unit_mask: [batch_size x 1]
        # note select_unit_mask should be bool type to make sure it is a right whether mask
        assert len(action_type.shape) == 2  

        select_unit_mask = L.action_involve_selecting_units_mask(action_type).bool()
        no_select_units_index = ~select_unit_mask.squeeze(dim=1)
        print("no_select_units_index:", no_select_units_index) if debug else None

        #autoregressive_embedding[no_select_units_index] = original_ae[no_select_units_index]
        units_logits[no_select_units_index] = (-1e9)  # a magic number

        # remove the EOF
        select_units_num = select_units_num - 1

        del selected_mask, select_unit_mask, no_select_units_index, mask, units_logits_list, key, key_avg

        return units_logits, units, autoregressive_embedding, select_units_num


def test():
    batch_size = 4
    autoregressive_embedding = torch.zeros(batch_size, AHP.autoregressive_embedding_size)
    action_type = torch.randint(low=0, high=SFS.available_actions, size=(batch_size, 1))
    action_type[0, 0] = 0  # no-op
    action_type[3, 0] = 168  # move-camera
    entity_embeddings = torch.randn(batch_size, AHP.max_entities, AHP.entity_embedding_size)
    entity_num = torch.tensor([1, 2, 3, 12])

    selected_units_head = SelectedUnitsHead()

    print("autoregressive_embedding:",
          autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:",
          autoregressive_embedding.shape) if debug else None

    units_logits, units, autoregressive_embedding, units_num = \
        selected_units_head.forward(
            autoregressive_embedding, action_type, entity_embeddings, entity_num)

    if units_logits is not None:
        print("units_logits:", units_logits) if debug else None
        print("units_logits.shape:", units_logits.shape) if debug else None
    else:
        print("units_logits is None!")

    if units is not None:
        print("units:", units) if debug else None
        print("units.shape:", units.shape) if debug else None
    else:
        print("units is None!")

    print("units_num:", units_num) if debug else None

    units_logits, _, autoregressive_embedding, _ = \
        selected_units_head.mimic_forward(
            autoregressive_embedding, action_type, entity_embeddings, entity_num, units, units_num)

    if units_logits is not None:
        print("units_logits:", units_logits) if debug else None
        print("units_logits.shape:", units_logits.shape) if debug else None
    else:
        print("units_logits is None!")

    print("autoregressive_embedding:",
          autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:",
          autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
