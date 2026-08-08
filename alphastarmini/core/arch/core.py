#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Core."

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

__author__ = "Ruo-Ze Liu"

debug = False


class Core(nn.Module):
    '''
    Inputs: prev_state, embedded_entity, embedded_spatial, embedded_scalar
    Outputs:
        next_state - The LSTM state for the next step
        lstm_output - The output of the LSTM
    '''

    def __init__(self, embedding_dim=AHP.original_1024, hidden_dim=AHP.lstm_hidden_dim, 
                 batch_size=AHP.batch_size,
                 sequence_length=AHP.sequence_length,
                 n_layers=AHP.lstm_layers, drop_prob=0.0):
        super(Core, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Unfortunately, PyTorch doesn't have a LayerNorm RNN cell or LayerNorm LSTM class, so we use the ordinary one.
        # please see https://github.com/pytorch/pytorch/issues/6760
        # TODO: change it to a LayerNorm one (using OpenDI)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, 
                            dropout=drop_prob, batch_first=True)

        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def forward(self, embedded_scalar, embedded_entity, embedded_spatial, 
                batch_size=None, sequence_length=None, hidden_state=None):
        '''
        embedded_scalar: 游戏资源嵌入编码
        embedded_entity： 游戏中可看到的实体嵌入编码
        embedded_spatial： 地图嵌入编码，方便了解当前全局局势

        hidden_state：隐藏层，如果没有传入则说明是从头开始；如果要继续上一次的计算结果，需要将上一次的计算得到的隐藏层信息传入
        '''
        # note: the input_shape[0] is batch_seq_size, we only transfrom it to [batch_size, seq_size, ...]
        # before input it into the lstm
        # shapes of embedded_entity, embedded_spatial, embedded_scalar are all [batch_seq_size x embedded_size]
        batch_seq_size = embedded_scalar.shape[0]

        batch_size = batch_size if batch_size is not None else self.batch_size
        sequence_length = sequence_length if sequence_length is not None else self.sequence_length
        input_tensor = torch.cat([embedded_scalar, embedded_entity, embedded_spatial], dim=-1) # 组合了嵌入编码、实体编码、地图编码的tensor
        del embedded_scalar, embedded_entity, embedded_spatial # 节省空间

        # note, before input to the LSTM
        # we transform the shape from [batch_seq_size, embedding_size] 
        # to the actual [batch_size, seq_size, embedding_size] 
        embedding_size = input_tensor.shape[-1] 
        input_tensor = input_tensor.reshape(batch_size, sequence_length, embedding_size)

        if hidden_state is None:
            # hidden_state size ((n_layers, batch_size, hidden_dim)， (n_layers, batch_size, hidden_dim))
            hidden_state = self.init_hidden_state(batch_size=batch_size)

        # lstm_output: (batch_size, sequence_length, hidden_dim) 
        # hidden_state: 一个元组 (h_n, c_n)，分别是隐藏状态和细胞状态。
        lstm_output, hidden_state = self.forward_lstm(input_tensor, hidden_state)
        # 为什么 flatten？ 因为下游的 ActionTypeHead、DelayHead 等模块不需要区分"这是第几个时间步"——它们接收的是展平后的 (batch*seq, hidden_dim)，每个样本独立产生动作
        # LSTM 已经把时间依赖编码进了各个时刻的隐状态里，reshape 后每个时刻的输出就是一份独立的、富含上下文的状态表示。
        # lstm_output shape is (batch_size * sequence_length, self.hidden_dim)
        lstm_output = lstm_output.reshape(batch_size * sequence_length, self.hidden_dim)
        del input_tensor

        return lstm_output, hidden_state

    def forward_lstm(self, x, hidden):
        '''
        x: shape is [batch_size, seq_size, embedding_size] ，保存着游戏资源、实体对象、地图信息的嵌入编码
        hidden：隐藏层信息， ((n_layers, batch_size, hidden_dim)， (n_layers, batch_size, hidden_dim))
        '''
        # note: No projection is used.
        # note: The outputs of the LSTM are the outputs of this module.
        '''
        返回值 1：lstm_out
        项目	说明
        形状	(batch_size, sequence_length, hidden_dim)
        MiniStar	(batch_size, seq_len, 128)
        完整 AlphaStar	(batch_size, seq_len, 384)
        含义	LSTM 最后一层在每一个时间步的输出

        ---

        hidden 是一个元组 (h_n, c_n)，分别是隐藏状态和细胞状态。
        返回值	形状（MiniStar）	形状（完整 AlphaStar）	含义
        h_n	(1, batch_size, 128)	(3, batch_size, 384)	LSTM 所有层在最后时刻的隐状态
        c_n	(1, batch_size, 128)	(3, batch_size, 384)	LSTM 所有层在最后时刻的细胞状态
        lstm_out 的最后一列和 h_n 的最后一层在数学上是同一个向量（不考虑数值精度差异）
        '''
        lstm_out, hidden = self.lstm(x, hidden)

        return lstm_out, hidden

    def init_hidden_state(self, batch_size=1):
        '''
        TODO: use learned hidden state ?
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        or 
        device = next(self.parameters()).device
        self.hidden = nn.Parameter(torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        self.cell_state = nn.Parameter(torch.zeros(self.n_layers, batch_size, self.hidden_dim))        
        nn.init.uniform_(self.hidden, b=1./ self.hidden_dim)
        nn.init.uniform_(self.cell_state, b=1./ self.hidden_dim)
        '''

        device = next(self.parameters()).device
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device), 
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

        return hidden


def test():

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
