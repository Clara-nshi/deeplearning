# softmax: 把预测多分类的分数转换为概率
# 后续开发应用：softmax一般配合多分类交叉熵损失函数内部自动使用(torch.nn.CrossEntropyLoss())
import torch
# 模拟一个样本属于9个类别的预测结果
y = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
# todo softmax把预测分数转化为概率（和为1）
softmax_y = torch.softmax(y, dim=-1)
print(softmax_y)
print(softmax_y.sum())

# 模拟一个样本属于9个类别的预测结果
y = torch.tensor([[3, 4, 5], [1, 2, 6]], dtype=float)
print(y)
# todo softmax把预测分数转化为概率（和为1）
softmax_y = torch.softmax(y, dim=0)
print(softmax_y)
print(softmax_y.sum())


""""""