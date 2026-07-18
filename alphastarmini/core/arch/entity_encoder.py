#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Entity Encoder."

from time import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysc2.lib.actions import RAW_FUNCTIONS as RF
from pysc2.lib.units import Protoss, Neutral

from alphastarmini.lib.alphastar_transformer import Transformer
from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import MiniStar_Arch_Hyper_Parameters as MAHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

from alphastarmini.third import action_dict as AD

__author__ = "Ruo-Ze Liu"

debug = False
speed = False


class EntityEncoder(nn.Module):
    # below is value form max value of one-hot encoding
    max_entities = AHP.max_entities
    max_unit_type = SCHP.max_unit_type
    max_alliance = 5

    max_unit_attributes = 13

    max_health = 1500
    max_shield = 1000
    max_energy = 200

    max_cargo_space_used = 9
    max_cargo_space_maximum = 9

    max_display_type = 5
    max_cloakState = 5

    max_is_powered = 2
    max_is_hallucination = 2
    max_is_active = 2
    max_is_on_screen = 2
    max_is_in_cargo = 2

    max_current_minerals = 19
    max_current_vespene = 26

    max_mined_minerals = 1800
    max_mined_vespene = 2500

    max_assigned_harvesters = 25  # AlphaStar: 24. RuntimeError: index 24 is out of bounds for dimension 1 with size 24
    max_ideal_harvesters = 17

    max_weapon_cooldown = 32
    max_order_queue_length = 9

    max_order_progress = 10

    max_order_ids = SCHP.max_order_ids
    max_buffer_ids = SCHP.max_buffer_ids
    max_add_on_type = SCHP.max_add_on_type

    max_weapon_upgrades = 4
    max_armor_upgrades = 4
    max_shield_upgrades = 4

    max_was_selected = 2
    max_was_targeted = 2

    bias_value = 0.  # -1e9

    '''
    Inputs: entity_list
    Outputs:
        embedded_entity - A 1D tensor of the embedded entities 
        entity_embeddings - The embedding of each entity (as opposed to `embedded_entity`, which has one embedding for all entities)
    '''

    def __init__(self, dropout=0.0, original_256=AHP.original_256, 
                 original_1024=AHP.original_1024,
                 original_128=AHP.original_128):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedd = nn.Linear(AHP.embedding_size, original_256) # 对输入的实体进行编码，特征提取，把原始实体特征从"手工编码的稀疏高维空间"映射到 Transformer 所需的"固定维度稠密空间"。
        self.transformer = Transformer(d_model=original_256, d_inner=original_1024,
                                       n_layers=3, n_head=2, d_k=original_128, 
                                       d_v=original_128, 
                                       dropout=0.)  # make dropout=0 to make training and testing consistent
        self.conv1 = nn.Conv1d(original_256, original_256, kernel_size=1, stride=1,
                               padding=0, bias=True)
        self.fc1 = nn.Linear(original_256, original_256)

    @classmethod
    def preprocess_numpy(cls, entity_list, return_entity_pos=False, debug=False):
        entity_array_list, entity_pos_list = [], []

        t = time()
        for i, entity in enumerate(entity_list):
            if i >= cls.max_entities:
                break

            field_encoding_list = []

            unit_type_index = L.unit_tpye_to_unit_type_index(entity.unit_type)  # change to a fast version
            field_encoding_list.append(unit_type_index)
            del unit_type_index

            unit_attributes_encoding = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
            field_encoding_list.extend(unit_attributes_encoding)
            del unit_attributes_encoding

            field_encoding_list.append(entity.alliance)

            field_encoding_list.append(entity.display_type)

            field_encoding_list.append(entity.x)

            field_encoding_list.append(entity.y)

            entity_pos_list.append([entity.x, entity.y])

            # print('entity.x, entity.y', entity.x, entity.y)

            field_encoding_list.append(entity.health)

            field_encoding_list.append(entity.shield)

            field_encoding_list.append(entity.energy)

            field_encoding_list.append(entity.cargo_space_taken)

            field_encoding_list.append(entity.cargo_space_max)

            field_encoding_list.append(entity.build_progress)

            field_encoding_list.append(entity.health_ratio)

            field_encoding_list.append(entity.shield_ratio)

            field_encoding_list.append(entity.energy_ratio)

            field_encoding_list.append(entity.cloak)

            field_encoding_list.append(entity.is_powered)

            field_encoding_list.append(entity.hallucination)

            field_encoding_list.append(entity.active)

            field_encoding_list.append(entity.is_on_screen)

            field_encoding_list.append(entity.is_in_cargo)

            field_encoding_list.append(entity.mineral_contents)

            field_encoding_list.append(entity.vespene_contents)

            mined_minerals_encoding = 22  # TODO: change to another version
            field_encoding_list.append(mined_minerals_encoding)

            mined_vespene_encoding = 17  # TODO: change to another version
            field_encoding_list.append(mined_vespene_encoding)

            field_encoding_list.append(entity.assigned_harvesters)

            field_encoding_list.append(entity.ideal_harvesters)

            field_encoding_list.append(entity.weapon_cooldown)

            field_encoding_list.append(entity.order_length)

            print('entity.unit_type', L.get_unit_tpye_name_and_race(entity.unit_type)[0].name) if debug else None

            order_id_0 = entity.order_id_0
            field_encoding_list.append(order_id_0)
            del order_id_0
            #print('entity.order_id_0', RF[order_id_0].name) if entity.alliance == 1 and entity.order_id_0 != 0 and entity.order_id_1 != 0 else None

            order_id_1 = entity.order_id_1
            field_encoding_list.append(order_id_1)
            del order_id_1
            #print('entity.order_id_1', RF[order_id_1].name) if entity.alliance == 1 and entity.order_id_0 != 0 and entity.order_id_1 != 0 else None

            buff_id_0 = L.get_buff_index_fast(entity.buff_id_0)
            field_encoding_list.append(buff_id_0)
            del buff_id_0            
            # print('entity.buff_id_0', entity.buff_id_0) if entity.alliance == 1 and entity.buff_id_0 else None
            # print('entity.buff_id_0', buff_id_0) if entity.alliance == 1 and buff_id_0 else None

            if False:
                buff_id_1 = L.get_buff_index_fast(entity.buff_id_1)
                field_encoding_list.append(buff_id_1)
                del buff_id_1    
                print('entity.buff_id_0', entity.buff_id_1) if entity.alliance == 1 and entity.buff_id_1 else None
                print('entity.buff_id_0', buff_id_1) if entity.alliance == 1 and buff_id_1 else None

            field_encoding_list.append(entity.order_progress_0)
            field_encoding_list.append(entity.order_progress_0)

            field_encoding_list.append(entity.order_progress_1)
            field_encoding_list.append(entity.order_progress_1)

            field_encoding_list.append(entity.attack_upgrade_level)

            field_encoding_list.append(entity.armor_upgrade_level)

            field_encoding_list.append(entity.shield_upgrade_level)

            field_encoding_list.append(entity.is_selected)

            was_targeted_encoding = 0  # change to another
            field_encoding_list.append(was_targeted_encoding)
            del was_targeted_encoding

            entity_array_list.append(field_encoding_list)
            del field_encoding_list

        del entity_list
        print('preprocess_numpy, t1', time() - t) if speed else None
        t = time()

        entities_array = np.array(entity_array_list)

        field_encoding_list = []

        unit_type_encoding = L.np_one_hot(entities_array[:, 0].astype(np.int32), cls.max_unit_type)
        field_encoding_list.append(unit_type_encoding)
        del unit_type_encoding

        unit_attributes_encoding = entities_array[:, 1:14]
        field_encoding_list.append(unit_attributes_encoding)
        del unit_attributes_encoding

        alliance_encoding = L.np_one_hot(entities_array[:, 14].astype(np.int32), cls.max_alliance)
        field_encoding_list.append(alliance_encoding)
        del alliance_encoding

        display_type_encoding = L.np_one_hot(entities_array[:, 15].astype(np.int32), cls.max_display_type)
        field_encoding_list.append(display_type_encoding)
        del display_type_encoding

        x_encoding = np.unpackbits(entities_array[:, 16:17].astype(np.uint8), axis=1)
        field_encoding_list.append(x_encoding)
        del x_encoding

        y_encoding = np.unpackbits(entities_array[:, 17:18].astype(np.uint8), axis=1)
        field_encoding_list.append(y_encoding)
        del y_encoding

        current_health_encoding = np.sqrt(np.minimum(entities_array[:, 18], cls.max_health))
        current_health_encoding = L.np_one_hot(current_health_encoding.astype(np.int32), int(cls.max_health ** 0.5) + 1)
        field_encoding_list.append(current_health_encoding)

        current_shield_encoding = np.sqrt(np.minimum(entities_array[:, 19], cls.max_shield))
        current_shield_encoding = L.np_one_hot(current_shield_encoding.astype(np.int32), int(cls.max_shield ** 0.5) + 1)
        field_encoding_list.append(current_shield_encoding)
        del current_shield_encoding   

        current_energy_encoding = np.sqrt(np.minimum(entities_array[:, 20], cls.max_energy))
        current_energy_encoding = L.np_one_hot(current_energy_encoding.astype(np.int32), int(cls.max_energy ** 0.5) + 1)
        field_encoding_list.append(current_energy_encoding)
        del current_energy_encoding

        cargo_space_used_encoding = L.np_one_hot(entities_array[:, 21].astype(np.int32), cls.max_cargo_space_used)
        field_encoding_list.append(cargo_space_used_encoding)
        del cargo_space_used_encoding

        cargo_space_maximum_encoding = L.np_one_hot(entities_array[:, 22].astype(np.int32), cls.max_cargo_space_maximum)
        field_encoding_list.append(cargo_space_maximum_encoding)
        del cargo_space_maximum_encoding

        field_encoding_list.append(entities_array[:, 23:24] * 0.01)

        field_encoding_list.append(entities_array[:, 24:27] * 0.0039215)  # / 255

        cloakState_encoding = L.np_one_hot(entities_array[:, 27].astype(np.int32), cls.max_cloakState)
        field_encoding_list.append(cloakState_encoding)
        del cloakState_encoding

        is_powered_encoding = L.np_one_hot(entities_array[:, 28].astype(np.int32), cls.max_is_powered)
        field_encoding_list.append(is_powered_encoding)
        del is_powered_encoding

        is_hallucination_encoding = L.np_one_hot(entities_array[:, 29].astype(np.int32), cls.max_is_hallucination)
        field_encoding_list.append(is_hallucination_encoding)
        del is_hallucination_encoding

        is_active_encoding = L.np_one_hot(entities_array[:, 30].astype(np.int32), cls.max_is_active)
        field_encoding_list.append(is_active_encoding)
        del is_active_encoding

        is_on_screen_encoding = L.np_one_hot(entities_array[:, 31].astype(np.int32), cls.max_is_on_screen)
        field_encoding_list.append(is_on_screen_encoding)
        del is_on_screen_encoding

        is_in_cargo_encoding = L.np_one_hot(entities_array[:, 32].astype(np.int32), cls.max_is_in_cargo)
        field_encoding_list.append(is_in_cargo_encoding)
        del is_in_cargo_encoding

        current_minerals_encoding = L.np_one_hot((entities_array[:, 33] * 0.01).astype(np.int32), cls.max_current_minerals)
        field_encoding_list.append(current_minerals_encoding)
        del current_minerals_encoding

        current_vespene_encoding = L.np_one_hot((entities_array[:, 34] * 0.01).astype(np.int32), cls.max_current_vespene)
        field_encoding_list.append(current_vespene_encoding)
        del current_vespene_encoding

        mined_minerals_encoding = L.np_one_hot(entities_array[:, 35].astype(np.int32), int(cls.max_mined_minerals ** 0.5) + 1)
        field_encoding_list.append(mined_minerals_encoding)
        del mined_minerals_encoding

        mined_vespene_encoding = L.np_one_hot(entities_array[:, 36].astype(np.int32), int(cls.max_mined_vespene ** 0.5) + 1)
        field_encoding_list.append(mined_vespene_encoding)
        del mined_vespene_encoding

        assigned_harvesters_encoding = L.np_one_hot(np.minimum(entities_array[:, 37], 24).astype(np.int32), cls.max_assigned_harvesters)
        field_encoding_list.append(assigned_harvesters_encoding)
        del assigned_harvesters_encoding

        ideal_harvesters_encoding = L.np_one_hot(entities_array[:, 38].astype(np.int32), cls.max_ideal_harvesters)
        field_encoding_list.append(ideal_harvesters_encoding)
        del ideal_harvesters_encoding

        weapon_cooldown_encoding = L.np_one_hot(np.minimum(entities_array[:, 39], 31).astype(np.int32), cls.max_weapon_cooldown)
        field_encoding_list.append(weapon_cooldown_encoding)
        del weapon_cooldown_encoding

        order_queue_length_encoding = L.np_one_hot(np.minimum(entities_array[:, 40], 8).astype(np.int32), cls.max_order_queue_length)
        field_encoding_list.append(order_queue_length_encoding)
        del order_queue_length_encoding

        idx = 41
        order_id_0_encoding = entities_array[:, idx].astype(np.int32)
        print('order_id_0_encoding before', order_id_0_encoding) if debug else None

        # we transform the act to general act
        order_id_0_encoding = AD.ACT_TO_GENERAL_ACT_ARRAY[order_id_0_encoding]
        print('order_id_0_encoding after', order_id_0_encoding) if debug else None

        order_id_0_encoding = L.np_one_hot(order_id_0_encoding, cls.max_order_ids)
        field_encoding_list.append(order_id_0_encoding)
        del order_id_0_encoding
        idx += 1

        order_id_1_encoding = entities_array[:, idx].astype(np.int32)
        order_id_1_encoding = AD.ACT_TO_GENERAL_ACT_ARRAY[order_id_1_encoding]
        order_id_1_encoding = L.np_one_hot(order_id_1_encoding, cls.max_order_ids)
        field_encoding_list.append(order_id_1_encoding)
        del order_id_1_encoding
        idx += 1

        buff_id_0_encoding = entities_array[:, idx].astype(np.int32)
        buff_id_0_encoding = np.minimum(buff_id_0_encoding, cls.max_buffer_ids - 1)
        buff_id_0_encoding = L.np_one_hot(buff_id_0_encoding, cls.max_buffer_ids)
        field_encoding_list.append(buff_id_0_encoding)
        del buff_id_0_encoding
        idx += 1    

        if False:
            buff_id_1_encoding = entities_array[:, idx].astype(np.int32)
            buff_id_1_encoding = np.minimum(buff_id_1_encoding, cls.max_buffer_ids - 1)
            buff_id_1_encoding = L.np_one_hot(buff_id_1_encoding, cls.max_buffer_ids)
            field_encoding_list.append(buff_id_1_encoding)
            del buff_id_1_encoding
            idx += 1    

        order_progress_0_encoding = entities_array[:, idx:idx + 1] * 0.01
        order_progress_0_encoding_onehot = L.np_one_hot(np.minimum(entities_array[:, idx + 1] * 0.1, 9).astype(np.int32), cls.max_order_progress)
        field_encoding_list.append(order_progress_0_encoding)
        field_encoding_list.append(order_progress_0_encoding_onehot)
        del order_progress_0_encoding, order_progress_0_encoding_onehot
        idx += 2

        order_progress_1_encoding = entities_array[:, idx:idx + 1] * 0.01
        order_progress_1_encoding_onehot = L.np_one_hot(np.minimum(entities_array[:, idx + 1] * 0.1, 9).astype(np.int32), cls.max_order_progress)
        field_encoding_list.append(order_progress_1_encoding)
        field_encoding_list.append(order_progress_1_encoding_onehot)
        del order_progress_1_encoding, order_progress_1_encoding_onehot
        idx += 2

        weapon_upgrades_encoding = L.np_one_hot(entities_array[:, idx].astype(np.int32), cls.max_weapon_upgrades)
        field_encoding_list.append(weapon_upgrades_encoding)
        del weapon_upgrades_encoding
        idx += 1

        armor_upgrades_encoding = L.np_one_hot(entities_array[:, idx].astype(np.int32), cls.max_armor_upgrades)
        field_encoding_list.append(armor_upgrades_encoding)
        del armor_upgrades_encoding
        idx += 1 

        shield_upgrades_encoding = L.np_one_hot(entities_array[:, idx].astype(np.int32), cls.max_shield_upgrades)
        field_encoding_list.append(shield_upgrades_encoding)
        del shield_upgrades_encoding
        idx += 1 

        was_selected_encoding = L.np_one_hot(entities_array[:, idx].astype(np.int32), cls.max_was_selected)
        field_encoding_list.append(was_selected_encoding)
        del was_selected_encoding
        idx += 1 

        was_targeted_encoding = L.np_one_hot(entities_array[:, idx].astype(np.int32), cls.max_was_targeted)
        field_encoding_list.append(was_targeted_encoding)
        del was_targeted_encoding
        idx += 1 

        all_entities_array = np.concatenate(field_encoding_list, axis=1)
        del field_encoding_list

        # count how many real entities we have
        real_entities_size = all_entities_array.shape[0]

        # we use a bias of 0 for any of the 512 entries that doesn't refer to an entity.
        if all_entities_array.shape[0] < cls.max_entities:
            bias_length = cls.max_entities - all_entities_array.shape[0]
            bias = np.zeros((bias_length, AHP.embedding_size))
            bias[:, :] = cls.bias_value
            all_entities_array = np.concatenate([all_entities_array, bias], axis=0)
            del bias

        all_entities_array = all_entities_array.astype(np.float32)
        print('preprocess_numpy, all_entities_array', time() - t) if speed else None
        t = time()

        del entities_array

        if return_entity_pos:
            return all_entities_array, entity_pos_list

        return all_entities_array

    def forward(self, x, debug=False, return_unit_types=False):
        '''
        x: 输入游戏的实体信息（实体（单位、建筑等）），shape (batch, 实体数量，embedding_size)
        '''
        # refactor by reference mostly to https://github.com/opendilab/DI-star
        # some mistakes for transformer are fixed
        batch_size = x.shape[0] # 批量训练的尺寸的大小，每个batch之间不相关
        entities_size = x.shape[1] # 获取选择的实体数量

        # calculate there are how many real entities in each batch
        # tmp_x: [batch_seq_size x entities_size]
        tmp_x = torch.mean(x, dim=2, keepdim=False) # 对实体维度求平均值，如果不存在实体，使用0替代，所以取平均值后会出现

        # tmp_y: [batch_seq_size x entities_size]
        tmp_y = (tmp_x != self.bias_value) # 找到有多少个真实的实体，通过不等于0的操作

        # entity_num: [batch_seq_size]
        entity_num = torch.sum(tmp_y, dim=1, keepdim=False) # 每个batch对应的实体数量
        del tmp_x, tmp_y

        # make sure we can have max to AHP.max_entities - 2 (510)
        # this is we must use a 512 one-hot to represent the entity_nums
        # so we have 0 to 511 entities, meanwhile, the 511 entity we use as a none index
        # so we at most have 510 entities.
        # 限制组打的实体数量，余下的1个位置有如下作用，必须保证真实实体数量 ≤ 510，这样索引 511 永远是空闲的，可以安全地用作 none 标记：
        # "空"/"无实体" 的特殊标记（none index）
        entity_num_numpy = np.minimum(AHP.max_entities - 2, entity_num.cpu().numpy())
        entity_num = torch.tensor(entity_num_numpy, dtype=entity_num.dtype, device=entity_num.device)
        del entity_num_numpy

        # this means for each batch, there are how many real enetities
        print('entity_num:', entity_num) if debug else None

        # generate the mask for transformer 制作蒙版，估计作用就是仅对最大实体数量内的数据有效，标记哪些是实体，哪些是padding
        mask = torch.arange(0, self.max_entities).float()
        mask = mask.repeat(batch_size, 1)

        device = next(self.parameters()).device
        mask = mask.to(device)

        # mask: [batch_size, max_entities]
        # 通过对比实体数量，位置编号小于实体数量的则为有效的实体，大于实体数量的则为padding
        # 生成一个bool矩阵
        mask = mask < entity_num.unsqueeze(dim=1)

        '''
        # x:              (batch, 512, embedding_size)
        # mask.unsqueeze: (batch, 512, 1)  广播到 (batch, 512, embedding_size)
        # 结果: padding 实体的所有特征被清零
        '''
        masked_x = x * mask.unsqueeze(-1)
        # x中，每个实体潜入编码中的前 max_unit_type 维就是 实体类型（unit type，表示是建筑还是农民还是机枪兵之类的单位）
        # 的 one-hot 编码（在 preprocess_numpy 中第 238 行拼接在最前面）。这里把它单独切出来，shape 为 (batch, 512, max_unit_type)。
        # 这里就可以快速知道每个实体的种类
        unit_types = masked_x[:, :, :SCHP.max_unit_type]
        del masked_x

        unit_types_one_list = []
        for i, batch in enumerate(unit_types): # 这里遍历batch中的每个样本
            unit_types_one = torch.nonzero(batch, as_tuple=True)[-1] # 使用torch.nonzero获取元素中非0索引位置的方式，找到 one-hot 中1的位置
            unit_types_one = unit_types_one.reshape(1, -1) # 将所有元素1的位置展平

            # entities_size（每个样本的最大样本数量） - entity_num[i]（每个样本的真实的实体数量） = 获取占位符数量
            # 这里像是构建非实体占位符的潜入编码
            placeholder = torch.ones(entities_size - entity_num[i], dtype=unit_types_one.dtype)
            placeholder = (placeholder * SCHP.max_unit_type).to(device).reshape(1, -1) # 将创建的非实体全1矩阵（每个1代表一个非实体）转换为 SCHP.max_unit_type
            # placeholder = torch.full((entities_size - entity_num[i],), SCHP.max_unit_type, ...)
            # 为什么占位符要填 max_unit_type？
            # 因为上面的one-hot编码中，有一个 max_unit_type 个种类，其中第max_unit_type表示的就是无兵种，表示没有任何选中
            # max_unit_type 是在兵种编码表里专门多出来的一个"不是任何兵种"的特殊 token。当某个实体槽位是 padding（没有真实单位），它的 unit type 不能填 0（0 是第一个真实兵种，比如 Probe），必须填一个超出所有真实兵种范围的占位值——也就是 max_unit_type。
 
            # 将实体和非实体组合在一起
            '''
            unit_types_one: [3, 7, 14, 2, 5, max_unit_type, max_unit_type, ..., max_unit_type]
                 ↑──────────────────────↑  ↑───────────────────────────────────↑
                 5 个真实实体的兵种索引        507 个 padding 槽位，统一填 max_unit_type
            '''
            unit_types_one = torch.cat([unit_types_one, placeholder], dim=1) 
            unit_types_one_list.append(unit_types_one)

            del placeholder

        unit_types_one = torch.cat(unit_types_one_list, dim=0) #  shape 是 (batch_size, max_entities)，即 (batch, 512)，每个位置要么是真实兵种的整数编号，要么是 max_unit_type（≈ 空）。
        del unit_types, unit_types_one_list

        # assert the input shape is : batch_seq_size x entities_size x embeding_size
        # note: because the feature size of entity is not equal to 256, so it can not fed into transformer directly.
        # thus, we add a embedding layer to transfer it to right size.
        # x is batch_entities_tensor (dim = 3). Shape: batch_seq_size x entities_size x embeding_size
        x = self.embedd(x) # x shape is (batch, 512, original_256)
        print('x.shape:', x.shape) if debug else None

        # mask for transformer need a special format
        mask_seq_len = mask.shape[-1] # 由于mask的shape是(batch, 512），那么它这里获取的512，也就是理论上的最大实体数量
        tran_mask = mask.unsqueeze(1) # shape变成(batch, 1, 512)

        # tran_mask: [batch_seq_size x max_entities x max_entities]
        tran_mask = tran_mask.repeat(1, mask_seq_len, 1) # 这里的shape就变成了（batch, 512, 512)
        # 最终 tran_mask 是一个三维张量。对每个 batch 样本来说，它是一个 512×512 的矩阵，且每一行完全相同：
        '''
        tran_mask[0] =
        [[T,T,T,T,T, F,F,F,...,F],    ← 第 0 行（实体 0 能看到哪些实体）
        [T,T,T,T,T, F,F,F,...,F],    ← 第 1 行（实体 1 能看到哪些实体）
        [T,T,T,T,T, F,F,F,...,F],    ← 第 2 行
        ...
        [T,T,T,T,T, F,F,F,...,F]]    ← 第 511 行（完全相同！）
        '''

        # out: [batch_seq_size x entities_size x embeding_size]
        # out 表示 (b, lq/实体数量 512, dim)
        out = self.transformer(x, mask=tran_mask)
        print('out.shape:', out.shape) if debug else None

        # entity_embeddings: [batch_seq_size x entities_size x conv1_output_size]
        entity_embeddings = F.relu(self.conv1(F.relu(out).transpose(1, 2))).transpose(1, 2)
        print('entity_embeddings.shape:', entity_embeddings.shape) if debug else None

        # AlphaStar: The mean of the transformer output across across the units (masked by the missing entries) 
        # is fed through a linear layer of size 256 and a ReLU to yield `embedded_entity`
        masked_out = out * mask.unsqueeze(dim=2)

        # sum over across the units
        # masked_out: [batch_seq_size x entities_size x embeding_size]
        # z: [batch_size, embeding_size]
        z = masked_out.sum(dim=1, keepdim=False)

        # here we should dived by the entity_num, not the cls.max_entities
        # z: [batch_size, embeding_size]
        z = z / entity_num.unsqueeze(dim=1)

        # note, dim=1 means the mean is across all entities in one timestep
        # The mean of the transformer output across across the units  
        # is fed through a linear layer of size 256 and a ReLU to yield `embedded_entity`
        # embedded_entity: [batch_size, fc1_output_size]
        embedded_entity = F.relu(self.fc1(z))
        del tran_mask, mask, masked_out, x, out, z

        if return_unit_types:
            return entity_embeddings, embedded_entity, entity_num, unit_types_one

        return entity_embeddings, embedded_entity, entity_num


class Entity(object):

    def __init__(self, unit_type=1,
                 unit_attributes=[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], alliance=0,
                 health=10, shield=20, energy=50,
                 cargo_space_taken=0, cargo_space_max=0, build_progress=0,
                 health_ratio=0.4, shield_ratio=0.5, energy_ratio=0.7,
                 health_max=100, shield_max=50, energy_max=40,
                 display_type=1, x=43, y=23, cloak=3, is_powered=True, hallucination=False, active=True,
                 is_on_screen=True, is_in_cargo=False, mineral_contents=1000, vespene_contents=1500, mined_minerals=500,
                 mined_vespene=300, assigned_harvesters=8, ideal_harvesters=14, weapon_cooldown=5.0, orders=[0, 1, 3, 0],
                 attack_upgrade_level=2, armor_upgrade_level=1, shield_upgrade_level=0, is_selected=True, is_targeted=False,
                 order_length=4, order_id_0=1, order_id_1=0, order_id_2=3, order_id_3=2, order_progress_0=50, 
                 order_progress_1=95, buff_id_0=12, buff_id_1=8, addon_unit_type=4, tag=0):
        super().__init__()
        self.unit_type = unit_type
        self.unit_attributes = unit_attributes
        self.alliance = alliance
        self.health = health
        self.shield = shield
        self.energy = energy
        self.cargo_space_taken = cargo_space_taken
        self.cargo_space_max = cargo_space_max
        self.build_progress = build_progress
        self.health_ratio = health_ratio
        self.shield_ratio = shield_ratio
        self.energy_ratio = energy_ratio
        self.health_max = health_max
        self.shield_max = shield_max
        self.energy_max = energy_max
        self.display_type = display_type
        self.x = x
        self.y = y 
        self.cloak = cloak
        self.is_powered = is_powered
        self.hallucination = hallucination
        self.active = active
        self.is_on_screen = is_on_screen
        self.is_in_cargo = is_in_cargo
        self.mineral_contents = mineral_contents
        self.vespene_contents = vespene_contents
        self.mined_minerals = mined_minerals
        self.mined_vespene = mined_vespene
        self.assigned_harvesters = assigned_harvesters
        self.ideal_harvesters = ideal_harvesters
        self.weapon_cooldown = weapon_cooldown
        self.attack_upgrade_level = attack_upgrade_level
        self.armor_upgrade_level = armor_upgrade_level
        self.shield_upgrade_level = shield_upgrade_level
        self.is_selected = is_selected
        self.is_targeted = is_targeted
        self.order_length = order_length
        self.order_id_0 = order_id_0
        self.order_id_1 = order_id_1
        self.order_id_2 = order_id_2
        self.order_id_3 = order_id_3
        self.order_progress_0 = order_progress_0
        self.order_progress_1 = order_progress_1
        self.buff_id_0 = buff_id_0
        self.buff_id_1 = buff_id_1
        self.addon_unit_type = addon_unit_type
        self.tag = tag

    def __str__(self):
        return 'unit_type: ' + str(self.unit_type) + ', alliance: ' + str(self.alliance) + ', health: ' + str(self.health)


def benchmark(e_list):
    # benchmark test
    benchmark_start = time()

    for i in range(1000):
        entities_tensor = torch.tensor(EntityEncoder.preprocess_numpy(e_list))

    elapse_time = time() - benchmark_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Preprocess time {}".format(elapse_time)) if debug else None

    # benchmark test
    benchmark_start = time()

    # for i in range(1000):
    #     entities_tensor = EntityEncoder.preprocess_in_tensor(e_list)

    elapse_time = time() - benchmark_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Preprocess time {}".format(elapse_time))

    print('entities_tensor:', entities_tensor) if debug else None
    print('entities_tensor.shape:', entities_tensor.shape) if debug else None
    # entities_tensor (dim = 2): entities_size x embeding_size


def test(debug=False):
    print(torch.tensor(np.unpackbits(np.array([25], np.uint8)))) if debug else None
    batch_size = 10

    e_list = []
    e1 = Entity(115, [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], 0, 100, 60, 50, 4, 8, 95, 0.2, 0.0, 0.0, 140, 60, 100,
                1, 123, 218, 3, True, False, True, True, False, 0, 0, 0, 0, 0, 0, 3.0, [2, 3], 2, 1, 0, True, False)
    e2 = Entity(1908, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], 2, 1500, 0, 200, 0, 4, 15, 0.5, 0.8, 0.5, 1500, 0, 250,
                2, 69, 7, 3, True, False, False, True, False, 0, 0, 0, 0, 10, 16, 0.0, [1], 1, 1, 0, False, False)
    e_list.append(e1)
    e_list.append(e2)

    encoder = EntityEncoder()

    # test use preproces in numpy array
    entities_tensor = torch.tensor(EntityEncoder.preprocess_numpy(e_list))

    # test use preproces in torch tensor
    # entities_tensor = EntityEncoder.preprocess_in_tensor(e_list)

    entities_tensor = entities_tensor.unsqueeze(0)

    batch_entities_tensor = entities_tensor

    # for i in range(batch_size):
    #     entities_tensor_copy = entities_tensor.detach().clone()
    #     batch_entities_tensor = torch.cat([batch_entities_tensor, entities_tensor_copy], dim=0)   

    # print('batch_entities_tensor:', batch_entities_tensor) if 1 else None    
    # print('batch_entities_tensor.shape:', batch_entities_tensor.shape) if 1 else None

    batch_entities_tensor = batch_entities_tensor.float()

    entity_embeddings, embedded_entity, entity_num = encoder.forward(batch_entities_tensor)

    print('entity_embeddings.shape:', entity_embeddings.shape) if debug else None
    print('embedded_entity.shape:', embedded_entity.shape) if debug else None
    print('entity_num.shape:', entity_num.shape) if debug else None

    if debug:
        print("This is a test!")


if __name__ == '__main__':
    test()
