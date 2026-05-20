import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

matplotlib.use("TkAgg")

# 1.准备彩色numpy数据
array = plt.imread("./data/bizhi_r.jpg")

# 2. 由于卷积层处理的是pytorch的张量，所以需要将numpy数据转换成张量
t1 = torch.tensor(array)
print(type(array), array.shape)
# 2.1 转张量，交换维度
t2 = t1.permute(2, 0, 1)
print(type(t2), t2.shape)
# 2.3 升维
t3 = t2.unsqueeze(dim=0)
print(type(t3), t3.shape)
# 3. 创建卷积层并提取特征
conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=2)
# 3.1 提取特征
out = conv(t3.float())
print(type(out), out.shape)

# 由于plt操作的数据必须是numpy, 所以需要将张量转换成numpy
out1 = out.squeeze(dim=0)
print(type(out1), out1.shape)
out2 = out1.permute(1, 2, 0)
print(type(out2), out2.shape)
out = out2.detach().numpy()
print(type(out), out.shape)
plt.imshow(out)
plt.axis("off")
plt.savefig("./savedata/bizhi_conv11.jpg")
plt.show()


