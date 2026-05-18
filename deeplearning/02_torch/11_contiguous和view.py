# 导包
import torch
# TODO 1. view：和view几乎一致，但是唯一不同的是不能修改非连续张量
# 创建张量
torch.manual_seed(6)
t = torch.randint(low=1, high=10, size=(12, ))
print("原始数据及形状\n", t.shape, t.ndim)
t1 = t.view(2, 6)
print("view后数据及形状\n", t1.shape, t1.ndim)
t2 = t.view(6, 2)
print("view后数据及形状\n", t2.shape, t2.ndim)
t3 = t.view(3, 4)
print("view后数据及形状\n", t3.shape, t3.ndim)
t4 = t.view(4, 3)
print("view后数据及形状\n", t4.shape, t4.ndim)
t5 = t.view(1, 12)
print("view后数据及形状\n", t5.shape, t5.ndim)
t6 = t.view(12, 1)
print("view后数据及形状\n", t6.shape, t6.ndim)

t7 = t.view(1, 1, 12)
print("三维view后数据及形状\n", t7, t7.shape, t7.ndim)
t8 = t.view(12, 1, 1)
print("三维view后数据及形状\n", t8, t8.shape, t8.ndim)

# todo: is_contiguous: 判断是否是连续张量
print(t, t.shape, t.ndim)
print(t.is_contiguous())
print(t8.is_contiguous())

print("===========================")
torch.manual_seed(6)
t = torch.randint(low=1, high=10, size=(2, 3, 4))
print(t, t.shape, t.ndim)
print(t.is_contiguous())
# todo transpose: 转置一次交换两个
t1 = t.transpose(0, 1).transpose(1, 2)
print(t1, t1.shape, t1.ndim, t1.is_contiguous())
# todo permute(): 一次交换多个
t2 = t.permute(1, 2, 0)
print(t2, t2.shape, t2.ndim, t2.is_contiguous())
print("===========================")

print(t2.reshape(3, 8))
new_t2 = t2.contiguous()
print(new_t2.shape, new_t2.ndim, new_t2.is_contiguous())

#
