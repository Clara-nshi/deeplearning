# # todo 张量索引格式：张量名[行索引, 列索引]
# import torch
#
# # 创建张量
# # 提前设置随机种子
# torch.manual_seed(6)
# t1 = torch.randint(low=0, high=10, size=(4, 4))
# print(t1)
# # todo 1. 单索引方式
# # 需求：获取第一行数据
# print(t1[0])
# # 需求：获取第一列数据
# print(t1[:, 0])
# # 需求：获取第1， 2， 3行数据
# print(t1[0])
# print(t1[1])
# print(t1[2])
#
# print("获取第1~3行数据", t1[0:3])
# print("获取第1、第3行数据", t1[[0, 2]])
# # 需求：获取第1， 2， 3列数据
# print(t1[:0])
# print(t1[:1])
# print(t1[:2])
# print("获取第1~3列数据", t1[:, 0:3])
# print("获取第1、第3列数据", t1[:, [0, 2]])
#
# # todo 布尔索引方式：条件按判断得到布尔索引
# # 需求：获取小于5的数据
# t2 = t1 < 5
# print(t1[t2])
#
# # 需求：获取第一行数据大于0的数据
# print("获取第一行数据大于0的数据", t1[0, t1[0] > 0])
# print("获取第一行数据大于0的数据", t1[0][[0] > 0])
#

import torch
# 设置随机种子
# 保证每次数据一致
torch.manual_seed(6)
t = torch.randint(low=0, high=10, size=(4, 4))
print("原始数据\n", t)
print("第一行数据\n", t[0])
print("第一列数据\n", t[:, 0])

print("前两行数据\n", t[[0, 2]])
print("前两列数据\n", t[:, [0, 2]])

print("第1、3行数据\n", t[[0, 2]])
print("第1、3列数据\n", t[:, [0, 2]])

print("大于5的数据\n", t[t > 5])
print("第2行大于6的数据\n", t[1][t[1] > 6])
print("第2行大于6的数据\n", t[1, t[1] > 6])

print("第3列大于6的数据\n", t[:, 2][t[:, 2] > 6])
# print("第3列大于6的数据\n", t[2, t[1] > 6])
print("第3列大于6的数据\n", t[t[:, 2] > 6, 2])

