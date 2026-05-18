# 导包
import torch


# 模拟准备数据
x = torch.randint(0, 10, size=(1, 4), dtype=torch.float)

# 模拟线形层
linear = torch.nn.Linear(4, 5)
# 模拟前向传播
y = linear(x)
print("随即失活前：", y)
# todo 模拟创建随机失活层   0.4 只是一个参考，每个数有40%的概率失活，其他没有被置为0的数，会自动乘以1/1-0.4
dropout = torch.nn.Dropout(0.4)
# todo 开始随机失活
y = dropout(y)
print("随即失活后：", y)
