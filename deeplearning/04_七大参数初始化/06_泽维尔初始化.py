# 导包
import torch
import torch.nn as nn
print("++++++++++++++++++泽维尔随即均匀初始化（只能初始化权重）++++++++++++++++++++")
# 模拟网络层
fc = nn.Linear(2, 3)
print(f"初始化之前权重：{fc.weight}")
# 模拟初始化权重参数
nn.init.xavier_uniform_(fc.weight, 1.5)
print(f"初始化之后权重：{fc.weight}")

print("++++++++++++++++++泽维尔正态分布初始化（不需要手动传入均值和标准差）++++++++++++++++++++")
# 模拟网络层
fc = nn.Linear(2, 3)
print(f"初始化之前权重：{fc.weight}")
# 模拟初始化权重参数
nn.init.xavier_normal_(fc.weight, 1.5)
print(f"初始化之后权重：{fc.weight}")

