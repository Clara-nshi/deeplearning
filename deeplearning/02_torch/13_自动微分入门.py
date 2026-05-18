import torch

# 模拟初始化权重参数
t1 = torch.tensor(3, requires_grad=True, dtype=torch.float32)
t2 = torch.tensor(data=[3, 4], requires_grad=True, dtype=torch.float32)
# 模拟已知损失函数
loss = 2*t1**2
loss2 = 2*t2**2
# todo 反向传播中自动计算梯度
loss.backward()
loss2.mean().backward()
# loss2.sum().backward()
# todo backward()把计算的梯度存储到了t1.grad属性中，此处获取它的梯度
print(t1.grad)
print(t2.grad)
# todo 参数更新：新参数=旧参数-学习率*梯度
# 此处固定学习率
print("更新前的参数为：", {t1.data})
lr = 0.01
t1.data = t1.data - lr*t1.grad
print(t1)
print("更新后的参数为：", {t1.data})

print("t2更新前的参数为：", {t2.data})
lr = 0.01
t2.data = t2.data - lr*t2.grad
print(t2)
print("更新后的参数为：", {t2.data})
