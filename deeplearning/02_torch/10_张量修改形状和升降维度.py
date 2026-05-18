# 导包
import torch
# TODO 1. shape和reshape
# 创建张量
torch.manual_seed(6)
t = torch.randint(low=1, high=10, size=(12, ))
print("原始数据及形状\n", t.shape, t.ndim)
t1 = t.reshape(2, 6)
print("reshape后数据及形状\n", t1.shape, t1.ndim)
t2 = t.reshape(6, 2)
print("reshape后数据及形状\n", t2.shape, t2.ndim)
t3 = t.reshape(3, 4)
print("reshape后数据及形状\n", t3.shape, t3.ndim)
t4 = t.reshape(4, 3)
print("reshape后数据及形状\n", t4.shape, t4.ndim)
t5 = t.reshape(1, 12)
print("reshape后数据及形状\n", t5.shape, t5.ndim)
t6 = t.reshape(12, 1)
print("reshape后数据及形状\n", t6.shape, t6.ndim)

t7 = t.reshape(1, 1, 12)
print("三维reshape后数据及形状\n", t7, t5.shape, t5.ndim)
t8 = t.reshape(12, 1, 1)
print("三维reshape后数据及形状\n", t8, t6.shape, t6.ndim)

# 删除维度为1的
new_t8 = t8.squeeze()
print("删除维度为1的数据及形状\n", new_t8, new_t8.shape, new_t8.ndim)
new_t7 = t7.squeeze()
print("删除维度为1的数据及形状\n", new_t7, new_t7.shape, new_t7.ndim)
new_t6 = t6.squeeze()
print("删除维度为1的数据及形状\n", new_t6, new_t6.shape, new_t6.ndim)

# 增加维度：每次都只能增加一个
new_t = t.unsqueeze(0)
print("增加维度后数据及形状\n", new_t, new_t.shape, new_t.ndim)
new_new_t = new_t.unsqueeze(0)
print("增加维度后数据及形状\n", new_new_t, new_new_t.shape, new_new_t.ndim)
# 把原始的t变成3维（12， 1， 1）
print("***************第一种方式**************")
new_t = t.unsqueeze(dim=-1).unsqueeze(dim=-1)
print("把原始的t变成3维（12， 1， 1）增加维度后数据及形状\n", new_t, new_t.shape, new_t.ndim)
print("***************第二种方式**************")
new_t = t.unsqueeze(dim=1)
new_t = new_t.unsqueeze(dim=2)
print("把原始的t变成3维（12， 1， 1）增加维度后数据及形状\n", new_t, new_t.shape, new_t.ndim)
print("***************第三种方式：第二种方式的变形**************")
new_t = t.unsqueeze(dim=1).unsqueeze(dim=2)
print("把原始的t变成3维（12， 1， 1）增加维度后数据及形状\n", new_t, new_t.shape, new_t.ndim)

new_t1 = t1.unsqueeze(0)
print("增加维度后数据及形状\n", new_t1, new_t1.shape, new_t1.ndim)\

print("-----------------------------------")
t = torch.arange(12)
print("原始数据及形状\n", t, t.shape, t.ndim)
t1 = t.unsqueeze(dim=-1).unsqueeze(dim=0)
print("增加维度后数据及形状\n", t1, t1.shape, t1.ndim)
