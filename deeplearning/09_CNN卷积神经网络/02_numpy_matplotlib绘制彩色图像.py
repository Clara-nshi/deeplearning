import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# 准备彩色 numpy 数据，直接读取图片获取像素数据
array = plt.imread('./data/bizhi_r.jpg')
print(type(array), array.shape)
print(array)

# 再绘制
plt.imshow(array)
plt.savefig('./data/img_save.jpg')
# 3. 展示图片
plt.show()

