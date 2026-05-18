# 导包
import torch
# 假设： 反例 -> 良性（0）；正例 -> 恶性（1）
# 模拟有两个样本[[第一个样本特征], [第二个样本特征]]
# todo 模拟准备样本对应的真实标签
labels = torch.tensor([1, 1], dtype=torch.float)
# todo 模拟准备模型预测分数
logits = torch.tensor([1.5, 2])
print("分数：", logits)
# todo 转换为概率
probs = torch.sigmoid(logits)
print("概率：", probs)
"""
结果:
([0.8176, 0.8808])
说明第一个样本特征是正例的概率是0.8176， 反例是1-0.8176
第二个样本特征同理
"""
# todo 创建交叉熵二分类损失函数
loss_fn = torch.nn.BCELoss()
# todo 计算损失
loss = loss_fn(probs, labels)
print(loss)