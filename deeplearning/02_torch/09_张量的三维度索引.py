# 导包
import torch
# 设置随机种子
torch.manual_seed(1)
# 创建张量
x = torch.randint(low=1, high=10, size=(3, 4, 5))
print("原始数据\n", x)

# 获取0轴的第一个面（三维得到的索引：降维都是面）
print("0轴的第一个\n", x[0])
print("0轴的第一个\n", x[0, :, :])

print("1轴的第一个\n", x[:, 0, :])
print("1轴的第一个\n", x[:, 0])
print("2轴的第一个\n", x[:, :, 0])
