# 导包
import torch
# todo 单通道池化操作
input1 = torch.tensor([[[2, 1, 3],
                        [1, 2, 3],
                        [5, 6, 7]]], dtype=torch.float32)
# 创建池化层
pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
output1 = pool1(input1)
print("最大池化层：", output1, output1.shape)
pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
output1 = pool1(input1)
print("平均池化层：", output1, output1.shape)
print("==============================================================")
# todo 多通道池化操作
input2 = torch.tensor([[[5, 6, 7], [2, 1, 3], [1, 2, 3]],
                       [[2, 1, 3], [1, 2, 3], [5, 6, 7]],
                       [[3, 4, 5], [5, 6, 7], [1, 2, 3]]], dtype=torch.float32)

# 创建池化层
pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
output2 = pool2(input2)
print("最大池化层：", output2, output2.shape)
pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
output2 = pool2(input2)
print("平均池化层：", output2, output2.shape)
print("==============================================================")