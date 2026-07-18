# param for some configs, for ease use of changing different servers
# also ease of use for experiments

'''whether is running on server, on server meaning use GPU with larger memoary'''
on_server = False # 本次小规模调试测试
#on_server = True # 服务器大规模运行测试

'''The replay path'''
replay_path = "data/Replays/filtered_replays_1/" # 已经采集的训练数据
#replay_path = "/home/liuruoze/data4/mini-AlphaStar/data/filtered_replays_1/"
#replay_path = "/home/liuruoze/mini-AlphaStar/data/filtered_replays_1/"

'''The mini scale used in hyperparameter'''
Batch_Scale = 16 # 控制训练时 batch size 的缩小比例：batch_size = int(16 × 8 / Batch_Scale)，为了小显存而设计
Seq_Scale = 16 # 控制 sequence_length（LSTM 展开的时间步数）的缩小比例，sequence_length = int(16 × 8 / Seq_Scale) ，主要也是小显存设计，多次训练梯度合在一起进行反向传播
Select_Scale = 4 #  控制 可选中单位的最大数量 的缩小比例（这个应该是策略游戏的特性，能够同时多少个单位）

handle_cuda_error = False # 和强化学习无关

# 在 forward() 和 mimic_forward() 中，entity_embeddings 和 embedded_entity 立即被清零。这意味着模型后续的所有计算（spatial encoder、action heads 等）都收不到实体信息（如单位类型、血量、位置等），变成一个"盲打"模型。
skip_entity_list = False # 消融实验开关，快速验证 entity encoder 到底对模型性能有多大贡献。
# 正常 AlphaStar 架构中，各个 action head 是按顺序执行的（action_type → delay → queue → select_units → target_unit），前一个 head 的输出会作为
skip_autoregressive_embedding = False #  消融实验开关——断开 action head 之间的自回归依赖链。快速验证 action head 之间的自回归依赖是否有用，或者用于调试时隔离某个 head 的问题。
