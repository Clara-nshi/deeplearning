# 导包
import torch
# 模拟 Xt 数据      32个样本（句子），每个样本有5个词，每个词有10个特征
# 1. todo 注意：循环层Xt的维度为：（句子中词的个数，句子数，词向量维度）
Xt = torch.randn(5, 32, 128)

# 2. todo 模拟Ht-1上一个隐藏层状态值
# 注意：由于是第一个，默认初始全0， 隐藏层状态值Ht-1的维度为：（句子数，隐藏层维度）
h0 = torch.zeros(size=(1, 32, 256))

# 3. todo 创建循环层
# input_size：输入的维度, hidden_size：隐藏层维度, num_layers：循环层数
rnn = torch.nn.RNN(input_size=128, hidden_size=256, num_layers=1, batch_first=False)

# todo 4. 循环层前向传播
# 输入数据 Xt, 初始隐藏层状态值 Ht-1
# 输出隐藏层状态值 Ht, 输出特征值 Yt
y, h_n = rnn(Xt, h0)
print(y.shape)
print(h_n.shape)
print("==========================================================")
# 假设 x 是（句子数batch， 句子词个数seq_len, 词向量维度）
x = torch.randn(size=(32, 5, 128))
# 2. todo 模拟Ht-1上一个隐藏层状态值
# 注意：由于是第一个，默认初始全0， 隐藏层状态值Ht-1的维度为：（句子数，隐藏层维度）
h0 = torch.zeros(size=(1, 32, 256))

# 3. todo 创建循环层
# input_size：输入的维度, hidden_size：隐藏层维度, num_layers：循环层数
# todo batch_first=True, 接收的数据维度为：（句子数batch， 句子词个数seq_len, 词向量维度）
rnn = torch.nn.RNN(input_size=128, hidden_size=256, num_layers=1, batch_first=True)

# todo 4. 循环层前向传播
# 输入数据 Xt, 初始隐藏层状态值 Ht-1
# 输出隐藏层状态值 Ht, 输出特征值 Yt
y, h_n = rnn(x, h0)
print(y.shape)
print(h_n.shape)