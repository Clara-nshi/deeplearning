# 导包
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
# todo 模拟一个月的天和温度数据
x_day = torch.arange(1, 31)
print(x_day)
torch.manual_seed(66)
y_temp = torch.rand(30) * 20 + 5
print(y_temp)


# todo 指数加权平均后线性图
print("====================指数加权平均直线图=======================")
y_iwm = []
beida = 0.9
# 遍历原始温度，计算指数加权平均值
for i, t in enumerate(y_temp):
    if i == 0:
        y_iwm.append(t)
    else:
        y_iwm.append(y_iwm[-1] * beida + (1-beida)*t)

# todo 模拟原始温度及指数加权平均的线性图
plt.subplot(121)
plt.scatter(x_day, y_temp)
plt.plot(x_day, y_temp)
plt.subplot(122)
plt.scatter(x_day, y_temp)
plt.plot(x_day, y_iwm)
plt.show()

