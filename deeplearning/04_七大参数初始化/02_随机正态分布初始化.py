# 导包
import torch
import torch.nn.functional as F
import torch.nn as nn

# 模拟网络层
fc = torch.nn.Linear(2, 3)
print(f"初始化之前权重：{fc.weight}")
# 模拟初始化权重参数
nn.init.normal_(fc.weight, mean=0, std=1)
print(f"初始化之后权重：{fc.weight}")
print("+++++++++++++++++++++++++++++")
# 模拟初始化偏置参数
print(f"初始化之前偏置：{fc.bias}")
nn.init.normal_(fc.bias)
print(f"初始化之后偏置：{fc.bias}")

