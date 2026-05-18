import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# numpy中的图像：(H, W, C)
# todo 绘制全 0 的图像
# 1. 现有数据
m1 = np.zeros(shape=(200, 200, 3))
plt.imshow(m1)
plt.show()

# todo 绘制全 255 的图像
m2 = np.full(shape=(200, 200, 3), fill_value=255)
plt.imshow(m2)
plt.show()


# todo 绘制全 任意rgb 的图像
m2 = np.full(shape=(200, 200, 3), fill_value=(223, 223, 223))
plt.imshow(m2)
plt.show()

