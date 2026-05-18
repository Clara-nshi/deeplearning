import torch
#
# todo 模拟准备真实3分类标签
labels = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
# todo 模拟模型预测分数
logits = torch.tensor([[5, 2], [1.5, 7]])
print("分数：", logits)
# todo 此处的分数需要自己提前转概率
probs = torch.sigmoid(logits)
print("概率：", probs)
# todo 准备多分类交叉熵损失函数(先softmax再算损失), 默认取平均：reduction="mean"
loss_fn = torch.nn.BCELoss()
# todo 计算损失
loss = loss_fn(probs, labels)
print(loss)
