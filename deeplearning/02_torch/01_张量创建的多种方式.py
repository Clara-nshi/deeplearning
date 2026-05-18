# 张量创建的多种方式
import torch

# TODO 1. 基础方式
"""
torch.tensor 默认int64类型
int类型 8/16/32/64
"""
t1 = torch.tensor(data=[1, 2, 3])
# shape和size效果一样
print(type(t1), t1.dtype, t1.shape, t1.ndim, t1.size())
print(t1)

t1 = torch.tensor(data=[[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
# shape和size效果一样
print(type(t1), t1.dtype, t1.shape, t1.ndim, t1.size())
print(t1)

"""
torch.Tensor 默认float32类型
int类型 8/16/32/64
"""
t2 = torch.Tensor([1, 2, 3])
print(t2, type(t2), t2.dtype, t2.shape, t2.ndim, t2.size())
t2 = torch.Tensor(size=(2, 2))
print(t2, type(t2), t2.dtype, t2.shape, t2.ndim, t2.size())
t2 = torch.IntTensor(size=(2, 2))

"""
int32
"""
print(t2, type(t2), t2.dtype, t2.shape, t2.ndim, t2.size())
print('*****************************************')

# TODO 线性张量
"""
左闭右开
开始，结束，步长
"""
r1 = torch.arange(0, 10.1, 2)
print(r1, type(r1), r1.dtype, r1.shape, r1.ndim, r1.size())

"""
左闭右闭
4个等步长的数
等差序列
"""
r2 = torch.linspace(0, 10.1, 4)
print(r2, type(r2), r2.dtype, r2.shape, r2.ndim, r2.size())

# TODO 随机张量
"""
.rand生成0-1的随机浮点数
"""
r3 = torch.rand(size=(2, 2))
print(r3, type(r3), r3.dtype, r3.shape, r3.ndim, r3.size())

"""
.randn生成标准正态分布的随机数
（2， 3， 3）两个3行3列的
"""
r4 = torch.randn(size=(2, 3, 3))
print(r4, type(r4), r4.dtype, r4.shape, r4.ndim, r4.size())

"""
randint生成指定范围内的随机整数，左闭右开
配合随机种子的使用
initial_seed查看自己设置是随机种子数
"""
torch.manual_seed(6)
r5 = torch.randint(1, 100, size=(2, 2))
print(r5, type(r5), r5.dtype, r5.shape, r5.ndim, r5.size())
print(torch.initial_seed())

# TODO 指定值的张量: 全0，全1，指定值
"""
全0，全1，指定值
"""
d1 = torch.zeros(size=(2, 2))
print(d1, type(d1), d1.dtype, d1.shape, d1.ndim, d1.size())
d2 = torch.ones(size=(2, 2))
print(d2, type(d2), d2.dtype, d2.shape, d2.ndim, d2.size())
d3 = torch.full(size=(2, 2), fill_value=6)
print(d3, type(d3), d3.dtype, d3.shape, d3.ndim, d3.size())
print('****************************')

"""
xx_like(张量)：参考指定张量的形状
"""
d4 = torch.tensor(data=[[1, 2, 3], [4, 5, 6]])
print(d4.ndim, d4.size(), d4.shape)
d5 = torch.zeros_like(d4)
print(d5, type(d5), d5.dtype, d5.shape, d5.ndim, d5.size())
d6 = torch.ones_like(d4)
print(d6, type(d6), d6.dtype, d6.shape, d6.ndim, d6.size())
d7 = torch.full_like(d4, fill_value=7)
print(d7, type(d7), d7.dtype, d7.shape, d7.ndim, d7.size())


