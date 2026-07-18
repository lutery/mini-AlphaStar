''' Define the sublayers in encoder/decoder layer '''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__original__author__ = "Yu-Hsiang Huang"

# modified refernect mostly to https://github.com/opendilab/DI-star in distar.model.alphastar.module_utils.Transformer
__modified__ = "Ruo-Ze Liu"


debug = True


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, bias_value=-1e9):
        super().__init__()
        self.temperature = temperature # self.temperature 的作用是防止点积注意力分数随向量维度增大而膨胀，从而避免 softmax 梯度消失。它的值是 √(d_k)。
        self.dropout = nn.Dropout(attn_dropout)
        self.biasval = bias_value # 按照mask，将padding部分设置为极小值

    def forward(self, q, k, v, mask=None):
        # mask shape is (b, 1, lq/lk/lv, lq/lk/lv)

        # q: (b, n, lq, dk)
        # k: (b, n, lk, dk)
        # 这里除以self teamareture 除了可以是 self.temperature，还可以是 除以 √(d_k)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) # atten (b, n, lq, lk)

        # atten: (b, n, lq, lk),
        if mask is not None:
            attn = attn.masked_fill(mask == 0, self.biasval) # 按照mask，将padding部分设置为极小值
            del mask

        attn = self.dropout(F.softmax(attn, dim=-1)) # 利用softmax，将最后一个维度变成一个概率分布，大概的意思就是每个token需要重点关注哪些哪些token的重要权重

        # v: (b, n, lv, dv)
        # r: (b, n, lq, dv)
        r = torch.matmul(attn, v) # 利用权重合并计算整个序列每个token之和，加权求和的词嵌入

        # r shape is (b, n, lq/lk/lv, dv/dk/dq), attn shape is (b, n, lq/lk/lv, lk/lq/lv)
        return r, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k # k张量中，每个头的维度
        self.d_v = d_v # d张量中，每个头的维度

        # pre-attention projection
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=True)

        # after-attention projection
        self.fc = nn.Linear(n_head * d_v, d_model, bias=True)

        # attention
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, q, k, v, mask=None):
        # q: (b, lq, dm)
        # k: (b, lk, dm)
        # v: (b, lv, dm)
        # mask: (b, lq/lk/lv, lq/lk/lv)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        size_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # pass through the pre-attention projection
        # separate different heads

        # after that q: (b, lq, n, dk) 常规的多头转换
        q = self.w_qs(q).view(size_b, len_q, n_head, d_k)

        k = self.w_ks(k).view(size_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(size_b, len_v, n_head, d_v)

        # transpose for attention dot product: (b, n_head, lq, dk)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting. mask shape (b, 1, lq/lk/lv, lq/lk/lv)

        # q shape is (b, n_head, lq/lk/lv, dv/dk/dq), attn shape is (b, n_head, lq/lk/lv, lk/lq/lv)
        q, attn = self.attention(q, k, v, mask=mask)

        # q: (b, n, lq, dk), k: (b, n, lk, dk), atten = q \matmul k^t = (b, n, lq, lk),
        # v: (b, n, lv, dv), assert lk = lv
        # atten \matmul v = (b, n, lq, dv)

        # transpose to move the head dimension back: (b, lq, n, dv)
        # combine the last two dimensions to concatenate all the heads together: (b, lq, (n*dv))
        q = q.transpose(1, 2).contiguous().view(size_b, len_q, -1)

        # q: (b, lq, (n*dv)) \matmul ((n*dv), dm) = (b, lq, dm)
        # note, q has the same shape as when it enter in
        q = self.fc(q)

        del mask, k, v, 

        # q shape is (b, lq, dm), attn shape is (b, n_head, lq, lq)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        return x
