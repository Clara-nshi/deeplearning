import torch
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

x = torch.linspace(-10, 10, 1000)
y = torch.relu(x)
_, axes = plt.subplots(1, 2)
axes[0].plot(x, y)
axes[0].set_title('tanh的原函数')
axes[0].grid()

x = torch.linspace(-10, 10, 1000, requires_grad=True)
y = torch.relu(x)
y.sum().backward()
y = x.grad
axes[1].plot(x.detach(), y)
axes[1].set_title('tanh的梯度')
axes[1].grid()

plt.show()