import torch
#
# todo 模拟准备真实3分类标签
# 方式1：one-hot分类独热编码方式
#                            A  B  C    A  B  C
#                            0  1  2    0  1  2
# labels = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.float)
# 方式2：索引方式(C的可行性最大[索引是2]，过来是B[索引是1]，二分类：)
labels = torch.tensor([2, 1], dtype=torch.long)
# todo 模拟模型预测分数(默认最大分数对应位置就是预测结果)
logits = torch.tensor([[1.5, 2.5, 5], [5, 5.5, 0.5]])
print("分数：", logits)
# todo 此处的分数不要自己提前转概率，因为后续softmax损失函数自动调用
# probs = torch.softmax(logits, dim=-1)
# print("概率：", probs)
# todo 准备多分类交叉熵损失函数(先softmax再算损失), 默认取平均：reduction="mean"
loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
# todo 计算损失
loss = loss_fn(logits, labels)
print(loss)