#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Location Head."

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import kaiming_uniform, normal

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import MiniStar_Arch_Hyper_Parameters as MAHP

from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

from alphastarmini.lib import utils as L

__author__ = "Ruo-Ze Liu"

debug = False


class LocationHead(nn.Module):
    '''
    Inputs: autoregressive_embedding, action_type, map_skip
    Outputs:
        target_location_logits - The logits corresponding to the probabilities of targeting each location
        target_location - The sampled target location
    '''

    def __init__(self, autoregressive_embedding_size=AHP.autoregressive_embedding_size, 
                 output_map_size=SCHP.world_size, is_sl_training=True, 
                 max_map_channels=AHP.location_head_max_map_channels,
                 temperature=AHP.temperature):
        super().__init__()
        self.use_improved_one = True

        self.is_sl_training = is_sl_training
        self.temperature = temperature

        mmc = max_map_channels
        self.ds_1 = nn.Conv2d(mmc + 4, mmc, kernel_size=1, stride=1,
                              padding=0, bias=True)
        self.film_blocks_num = 4

        if not self.use_improved_one:
            self.film_net = FiLM(n_resblock=self.film_blocks_num, 
                                 conv_hidden=mmc, 
                                 gate_size=autoregressive_embedding_size)
        else:
            self.film_net_mapskip = FiLMplusMapSkip(n_resblock=self.film_blocks_num, 
                                                    conv_hidden=mmc, 
                                                    gate_size=autoregressive_embedding_size)            

        self.us_1 = nn.ConvTranspose2d(mmc, int(mmc / 2), kernel_size=4, stride=2,
                                       padding=1, bias=True)
        self.us_2 = nn.ConvTranspose2d(int(mmc / 2), int(mmc / 4), 
                                       kernel_size=4, stride=2,
                                       padding=1, bias=True)
        self.us_3 = nn.ConvTranspose2d(int(mmc / 4), int(mmc / 8), 
                                       kernel_size=4, stride=2,
                                       padding=1, bias=True)
        self.us_4 = nn.ConvTranspose2d(int(mmc / 8), int(mmc / 16), 
                                       kernel_size=4, stride=2,
                                       padding=1, bias=True)
        self.us_4_original = nn.ConvTranspose2d(int(mmc / 8), 1, 
                                                kernel_size=4, stride=2,
                                                padding=1, bias=True)        

        # note: in mAS, we add a upsampling layer to transfer from 8x8 to 256x256
        self.us_5 = nn.ConvTranspose2d(int(mmc / 16), 1, kernel_size=4, stride=2,
                                       padding=1, bias=True)

        # note: when SCHP.world_size=64, we add a new upsampling layer
        self.us_6 = nn.ConvTranspose2d(int(mmc / 4), 1, kernel_size=4, stride=2,
                                       padding=1, bias=True)

        self.output_map_size = output_map_size

        self.softmax = nn.Softmax(dim=-1)
        self.is_rl_training = False

    def set_rl_training(self, staus):
        self.is_rl_training = staus

    def forward(self, autoregressive_embedding, action_type, map_skip, target_location=None):    
        '''
        LocationHead.forward 是一个**"条件化的编码器-解码器"：把 autoregressive_embedding（动作+局势意图）摊成小图、与最深层 map_skip 特征拼接 → 经 FiLM 调制的 ResBlock 组（逐层加回跳连）→ 逐级上采样 8×8→64×64 → 输出逐像素 logits**，采样解码成坐标 (x, y)。全程 tensor 从 [B,256] 的意图向量 + [B,32,8,8] 的地图特征，变成 [B,64,64] 的位置分布和 [B,2] 的坐标。
        Inputs:
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
            action_type: [batch_size x 1]
            map_skip: [batch_size x channel x height x width]
            autoregressive_embedding：游戏资源、地图信息等选择预测的动作+局势的嵌入+操作延迟（下一次什么时候在预测动作操作）+ 针对执行动作指令action_type是否需要立即执行的掩码信息 （batch， autoregressive_embedding_size），新加入了根据动作选择了要操作的实体单位的信息 
            action_type: 选择的动作（随机采样或者外部传入的专家动作）shape (batch, 1)
            map_skip is 一个list，每个元素shape是(batch, original_128, H / 8， W / 8) ，把地图特征从 64×64 压缩到 8×8 的过程中"攒下来的中间特征图"（U-Net 风格的跳连 / skip connections）
        Output:
            target_location_logits: [batch_size x self.output_map_size x self.output_map_size]
            location_out: [batch_size x 2 (x and y)]
        '''

        # AlphaStar: `autoregressive_embedding` is reshaped to have the same height/width as the final skip in `map_skip` 
        # AlphaStar: (which was just before map information was reshaped to a 1D embedding) with 4 channels
        # sc2_imitation_learning: map_skip = list(reversed(map_skip))
        # sc2_imitation_learning: inputs, map_skip = map_skip[0], map_skip[1:]
        map_skip = list(reversed(map_skip)) # 逆序，讲最深的特征采样放在最前面
        x, map_skip = map_skip[0], map_skip[1:] # x是最深的特征采样，map_skip是逐步向浅层递进
        batch_size = x.shape[0] 

        reshap_size = x.shape[-1] 
        reshape_channels = int(AHP.autoregressive_embedding_size / (reshap_size * reshap_size))
        ar_map = autoregressive_embedding.reshape(batch_size, -1, reshap_size, reshap_size) # 讲全局信息进行reshape，看起来是要讲小地图的信息接入进来

        # AlphaStar: and the two are concatenated together along the channel dimension,
        # map skip shape: (-1, 128, 16, 16)
        # x shape: (-1, 132, 16, 16)
        x = torch.cat([ar_map, x], dim=1) # 将小地图信息和全局地图信息拼接起来
        print("x.shape:", x.shape) if debug else None

        # AlphaStar: passed through a ReLU, 
        # AlphaStar: passed through a 2D convolution with 128 channels and kernel size 1,    
        # AlphaStar: then passed through another ReLU.
        x = F.relu(self.ds_1(F.relu(x))) # 混有小地图的全局信息特征提取，相当于做特征融合 (-1, 132, 16, 16)

        if not self.use_improved_one:
            # AlphaStar: The 3D tensor (height, width, and channels) is then passed through a series of Gated ResBlocks 
            # AlphaStar: with 128 channels, kernel size 3, and FiLM, gated on `autoregressive_embedding`  
            # note: FilM is Feature-wise Linear Modulation, please see the paper "FiLM: Visual Reasoning with 
            # a General Conditioning Layer"
            # in here we use 4 Gated ResBlocks, and the value can be changed
            x = self.film_net(x, gate=autoregressive_embedding) # # （b， filter_size， 1， 1）

            # x shape (-1, 128, 16, 16)
            # AlphaStar: and using the elements of `map_skip` in order of last ResBlock skip to first.
            x = x + map_skip # 这里大概是将注意力权重增加回地图，这样可以让地图知道该注意什么
        else:
            # Referenced mostly from "sc2_imitation_learning" project in spatial_decoder
            assert len(map_skip) == self.film_blocks_num

            # use the new FiLMplusMapSkip class
            x = self.film_net_mapskip(x, gate=autoregressive_embedding, 
                                      map_skip=map_skip)

            # Compared to AS, we a relu, referred from "sc2_imitation_learning"
            x = F.relu(x)

        # AlphaStar: Afterwards, it is upsampled 2x by each of a series of transposed 2D convolutions 
        # AlphaStar: with kernel size 4 and channel sizes 128, 64, 16, and 1 respectively 
        # AlphaStar: (upsampled beyond the 128x128 input to 256x256 target location selection).
        # 逐级上采样 8×8 → 64×64
        # ConvTranspose2d(k=4, s=2, p=1) 每次把边长翻倍：H_out = (H−1)×2 − 2 + 4。
        x = F.relu(self.us_1(x))
        x = F.relu(self.us_2(x))

        if SCHP.world_size == 64:
            # if world_size is (64, 64), we can make the output size to be 64 x 64
            x = self.us_6(x)
        else:
            x = F.relu(self.us_3(x))
            if AHP == MAHP:
                x = F.relu(self.us_4(x))
                # only in mAS, we need one more upsample step
                # x = F.relu(self.us_5(x))
                # Note: in the final layer, we don't use relu
                x = self.us_5(x)
            else:
                x = self.us_4_original(x)
        # 注意最后一步不加 ReLU——logits 需要正负值，ReLU 会把一半信息砍掉。

        del ar_map, map_skip, autoregressive_embedding

        # AlphaStar: Those final logits are flattened and sampled (masking out invalid locations using `action_type`, 
        # AlphaStar: such as those outside the camera for build actions) with temperature 0.8 
        # AlphaStar: to get the actual target position.
        # x shape: (-1, 1, 256, 256)
        # 将上采样后的全局特征融合信息进行reshape，得到一个目标点的logits分布
        # 看来是决定要重点关注哪里的预测
        target_location_logits = x.reshape(batch_size, 1 * self.output_map_size * self.output_map_size)

        temperature = self.temperature if self.is_rl_training else 1
        target_location_logits = target_location_logits / temperature
        print("target_location_logits:", target_location_logits) if debug else None
        print("target_location_logits.shape:", target_location_logits.shape) if debug else None

        # AlphaStar: If `action_type` does not involve targetting location, this head is ignored.
        # Note, maks sure the mask should be booll type, otherwise ~target_location_mask will output -2 when mask is 1.
        # 看来这是又是根据不同的动作决定选择的目标是否可以被动作执行的
        # 检查动作参数里有没有 world 参数——"这个动作需不需要目标位置"（no_op、纯单位目标技能等不需要）。
        target_location_mask = L.action_involve_targeting_location_mask(action_type).bool()
        no_target_location_mask = ~target_location_mask.squeeze(dim=1)

        # AlphaStar: (masking out invalid locations using `action_type`, such as those outside 
        # the camera for build actions)
        # TODO: use action to decide the mask
        # referenced from lib/utils.py function of masked_softmax()

        # mask = torch.zeros(batch_size, 1 * self.output_map_size * self.output_map_size, device=device)
        # mask = L.get_location_mask(mask)
        # mask_fill_value = -1e32  # a very small number
        # target_location_logits = target_location_logits.masked_fill((1 - mask).bool(), mask_fill_value)

        device = next(self.parameters()).device
        if target_location is None: # 没有传入专家数据
            target_location_probs = self.softmax(target_location_logits)
            location_id = torch.multinomial(target_location_probs, num_samples=1, replacement=True)

            target_location = np.zeros([batch_size, 2]) # 构建采样的目标位置的矩阵
            for i, idx in enumerate(location_id): # 遍历每一个采样点
                # 计算出横纵坐标的位置，因为上面的采样是展平后后的采样分布
                row_number = idx // self.output_map_size
                col_number = idx - self.output_map_size * row_number

                target_location_y = row_number
                target_location_x = col_number

                # note! sc2 and pysc2 all accept the position as [x, y], so x be the first, y be the last!
                # below is right! so the location point map to the point in the matrix!
                # target_location[i] = np.array([target_location_x.item(), target_location_y.item()])
                target_location[i] = np.array([target_location_x.item(), target_location_y.item()])

            del location_id
            target_location = torch.tensor(target_location, device=device).long()
            # 如果动作不需要目标位置，那么将对应的样本采样数据的目标位置设置为地图尺寸的边缘
            # 一是特定表示，二是即使执行了也不会有影响
            # 与 units=511、target_unit=511 同一套哨兵思路，这里是二维版 (63, 63)；
            target_location[no_target_location_mask] = torch.tensor([self.output_map_size - 1, self.output_map_size - 1], device=device)

        # 清除不需要目标位置动作的logits值
        target_location_logits = target_location_logits.reshape(-1, self.output_map_size, self.output_map_size)
        target_location_logits = target_location_logits * target_location_mask.float().unsqueeze(-1)

        del action_type, x, target_location_mask, no_target_location_mask

        # 一句话：LocationHead.forward = "意图摊平成图 → 与最深地图特征拼接 → FiLM 条件调制 4 层（每层补一条跳连）→ 三级反卷积上采样 8×8→64×64 → 展平采样出 (x,y) 坐标，不需要位置的动作用哨兵坐标和清零 logits 兜底"——它把自回归链的最后一环"打到哪里"变成了一个受动作/局势条件化的逐像素分类问题。
        '''
        target_location_logits: (B, 64(map_size), 64(map_size)), 地图每个像素的分数（用于 loss / RL 采样）
        target_location: [B, 2] 采样出的坐标 [x, y]
        '''
        return target_location_logits, target_location


class ResBlockFiLM(nn.Module):
    # some copy from https://github.com/rosinality/film-pytorch/blob/master/model.py
    def __init__(self, filter_size):
        super().__init__()

        self.conv1 = nn.Conv2d(filter_size, filter_size, kernel_size=[1, 1], stride=1, padding=0)
        self.conv2 = nn.Conv2d(filter_size, filter_size, kernel_size=[3, 3], stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(filter_size, affine=False) # affine=False 这里将bn的可学习参数给去掉了，纯粹的进行归一化，主要是为了避免和 gamma * out + beta 冲突

        self.reset()

    def forward(self, x, gamma, beta):
        # out: (-1, 132, 16, 16) x
        # film[i * 2]: （batch, conv_hidden) gamma
        # film[i * 2 + 1]: （batch, conv_hidden) beta

        out = self.conv1(x) #  # 1×1 卷积：通道混合
        resid = F.relu(out) # # 存一个残差分支
        out = self.conv2(resid) # # 3×3 卷积：空间处理
        out = self.bn(out) # # 批归一化
        
        gamma = gamma.unsqueeze(2).unsqueeze(3) # （batch, conv_hidden, 1, 1)
        beta = beta.unsqueeze(2).unsqueeze(3) # （batch, conv_hidden, 1, 1)

        out = gamma * out + beta # 注意力混合，这里单纯进行对应位置的数字进行混合

        out = F.relu(out) # 
        out = out + resid

        return out # （b， filter_size， 8， 8）

    def reset(self):
        # deprecated, should try to find others
        # kaiming_uniform(self.conv1.weight)
        # self.conv1.bias.data.zero_()
        # kaiming_uniform(self.conv2.weight)
        pass


class FiLM(nn.Module):
    # some copy from https://github.com/rosinality/film-pytorch/blob/master/model.py
    '''
    FiLM（Feature-wise Linear Modulation，特征级线性调制）是一种**"让一路输入去控制另一路特征处理方式"的技术：用条件向量（这里的 autoregressive_embedding）动态生成两组参数 γ 和 β，对中间特征图做逐通道的"缩放 + 平移"**——相当于给网络装了一套"由局势控制的旋钮"，决定哪些特征通道该放大、哪些该抑制。
    '''
    def __init__(self, n_resblock=4, conv_hidden=128, gate_size=1024):
        super().__init__()
        self.n_resblock = n_resblock
        self.conv_hidden = conv_hidden

        self.resblocks = nn.ModuleList()
        for i in range(n_resblock):
            self.resblocks.append(ResBlockFiLM(conv_hidden))

        self.film_net = nn.Linear(gate_size, conv_hidden * 2 * n_resblock)

    def reset(self):
        # deprecated, should try to find others
        # kaiming_uniform(self.film_net.weight)
        # self.film_net.bias.data.zero_()
        pass

    def forward(self, x, gate):
        '''
        这里函数就有点像注意力机制，通过将混有小地图的全局信息和没有小地图的全局信息作比较
        输出一个带有应该注意哪些信息的矩阵
        x: (-1, 132, 16, 16)
        gate: （batch， autoregressive_embedding_size）

        想象一个调音台：

        一张特征图有 32 个通道，可以看成 32 路音轨；
        γ（gamma）：每路音轨的"音量旋钮"——放大或压低这一路的信息；
        β（beta）：每路音轨的"偏移旋钮"——整体加一个基准值；
        关键：怎么拧这些旋钮，不是写死的，而是由"当前局势"决定的——局势说"现在要放技能"，旋钮就拧到适合生成技能落点的档位；局势说"现在要造建筑"，旋钮就换一组档位。

        把分享标题也换成萌宠探趣岛

        而 γ 和 β 本身是由条件向量算出来的：
        γ, β = film_net(gate)        # gate = autoregressive_embedding
        '''
        out = x

        # self.film_net(gate): （batch， conv_hidden * 2 * n_resblock）
        # .chunk(self.n_resblock * 2, 1)：（batch, 2 * n_resblock, conv_hidden)
        film = self.film_net(gate).chunk(self.n_resblock * 2, 1)

        for i, resblock in enumerate(self.resblocks):
            out = resblock(out, film[i * 2], film[i * 2 + 1])

        return out # # （b， filter_size， 1， 1）


class FiLMplusMapSkip(nn.Module):
    # Thanks mostly from https://github.com/metataro/sc2_imitation_learning in spatial_decoder
    def __init__(self, n_resblock=4, conv_hidden=128, gate_size=1024):
        super().__init__()
        self.n_resblock = n_resblock
        self.conv_hidden = conv_hidden

        self.resblocks = nn.ModuleList()
        for i in range(n_resblock):
            self.resblocks.append(ResBlockFiLM(conv_hidden))

        self.film_net = nn.Linear(gate_size, conv_hidden * 2 * n_resblock)

    def reset(self):
        # deprecated, should try to find others
        # kaiming_uniform(self.film_net.weight)
        # self.film_net.bias.data.zero_()
        pass

    def forward(self, x, gate, map_skip):
        out = x
        film = self.film_net(gate).chunk(self.n_resblock * 2, 1)

        for i, resblock in enumerate(self.resblocks):
            out = resblock(out, film[i * 2], film[i * 2 + 1])
            out = out + map_skip[i]

        # TODO: should we add a relu?

        return out


def test():
    batch_size = 2
    autoregressive_embedding = torch.randn(batch_size, AHP.autoregressive_embedding_size)
    action_type_sample = 65  # func: 65/Effect_PsiStorm_pt (1/queued [2]; 2/unit_tags [512]; 0/world [0, 0])
    action_type = torch.randint(low=0, high=SFS.available_actions, size=(batch_size, 1))

    map_skip = []
    if AHP == MAHP:
        for i in range(5):
            map_skip.append(torch.randn(batch_size, AHP.location_head_max_map_channels, 8, 8))
    else:
        for i in range(5):
            map_skip.append(torch.randn(batch_size, AHP.location_head_max_map_channels, 16, 16))

    location_head = LocationHead()

    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    target_location_logits, target_location = \
        location_head.forward(autoregressive_embedding, action_type, map_skip)

    if target_location_logits is not None:
        print("target_location_logits:", target_location_logits) if debug else None
        print("target_location_logits.shape:", target_location_logits.shape) if debug else None
    else:
        print("target_location_logits is None!")

    if target_location is not None:
        print("target_location:", target_location) if debug else None
        # print("target_location.shape:", target_location.shape) if debug else None
    else:
        print("target_location is None!")

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
