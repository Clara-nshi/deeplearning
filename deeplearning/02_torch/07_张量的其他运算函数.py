# 常见的运算函数
import math
import torch
# math数学库
print(math.sqrt(4))
print(math.pow(2, 3))
print(math.sin(math.pi / 2))
print(math.cos(math.pi / 2))
print(math.tan(math.pi / 4))
print(math.log(2, 10))
print(math.log10(2))
print(math.log2(2))
print(math.pi)
print(math.e)
print('***************')
# pytorch 运算函数
t1 = torch.tensor([4, 9])
print(torch.sqrt(t1))
print(torch.pow(t1, 2))
print(torch.log2(t1))
print(torch.log10(t1))
print(torch.log(t1))
print(torch.cos(t1))
print(torch.sin(torch.tensor(math.pi / 2)))

"""
注意：聚合函数中有dim属性，指定聚合维度
"""
t2 = torch.tensor([[[1, 2], [4, 5]], [[2, 1], [1, 4]]])
print(t2)
print(torch.sum(t2))
print(torch.sum(t2, dim=0))
print(torch.sum(t2, dim=1))
print(torch.sum(t2, dim=2))
print(torch.sum(t2, dim=-3))
print(torch.sum(t2, dim=-2))
print(torch.sum(t2, dim=-1))
t2 = torch.tensor([[1, 2], [4, 5]])
print(t2)
"""
(维度1， 维度2， 维度3)
  0      1      2
  -3    -2     -1
"""
print(torch.sum(t2))
print(torch.sum(t2, dim=0))
print(torch.sum(t2, dim=1))
print(torch.sum(t2, dim=-2))
print(torch.sum(t2, dim=-1))