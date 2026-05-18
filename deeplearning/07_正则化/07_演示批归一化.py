"""
梯度下降优化算法
"""
# 初始数据准备
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def oirgin_data():
    ELEMENT_NUMBER = 30
    # 固定随机种子
    torch.manual_seed(1)
    # 随机生成温度
    temp = torch.rand(size=[ELEMENT_NUMBER, ]) * 20
    # 绘制平均温度
    day = torch.arange(1, ELEMENT_NUMBER + 1)
    # 作图
    plt.plot(day, temp, "o")
    plt.scatter(day, temp)
    plt.show()


if __name__ == '__main__':

