# 导包
import torch

torch.manual_seed(666)
# 模拟 x 真实特征
x = torch.ones(size=(5, 3))
# 模拟 y 真实标签
y = torch.zeros(size=(2, 3))
# 模拟初始化 w 数据
w = torch.randn(size=(2, 5), requires_grad=True)
# 模拟初始化 b 数据
b = torch.randn(3, requires_grad=True)
# 计算 z 预测标签
z = torch.matmul(w, x) + b
print(z.shape)
# 计算损失（提前选择损失函数）
loss_fn = torch.nn.MSELoss()
loss = loss_fn(z, y)
print(f"损失值", loss)
# 反向传播（计算梯度）
loss.backward()
print("w梯度", w.grad)
print("b梯度", b.grad)
# 参数更新：新 = 旧 - 学习率*梯度
print(f"w更新参数前：\n{w}")
w = w - 0.01 * w.grad
print(f"w更新参数后：\n{w}")
print(f"b更新参数前：\n{w}")
b = b - 0.01 * b.grad
print(f"b更新参数后：\n{w}")