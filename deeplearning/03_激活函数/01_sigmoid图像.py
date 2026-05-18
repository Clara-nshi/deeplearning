# 导包
import torch
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 绘制图像
# 创建画布和坐标轴
# TODO 为了把值域图像和导数图像画到一个画布上，创建1行2列的画布
_, axes = plt.subplots(1, 2)
# 准备数据 x, y
x = torch.linspace(start=-10, end=10, steps=1000, requires_grad=False)
y = torch.sigmoid(x)
# 绘制图像
axes[0].plot(x, y)
axes[0].set_title("sigmoid值域图像")
axes[0].grid()

print("======================")
# todo 注意x开启自动微分，y本质是梯度
x = torch.linspace(start=-20, end=20, steps=1000, requires_grad=True)
torch.sigmoid(x).mean().backward()
y = x.grad
axes[1].grid()

# 绘制图像
axes[1].plot(x.detach(), y)
axes[1].set_title("sigmoid导数图像")


plt.show()

