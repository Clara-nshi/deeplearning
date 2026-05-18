# 导包
import torch
# 模拟准备线性的真实数据
y_true = torch.tensor([1, 2, 3, 4, 5])
print(y_true)
# 模拟准备线性的预测数据
y_pred = torch.tensor([0.6, 1.7, 3.0, 4.2, 5.3])
print(y_pred)
loss_fn = torch.nn.L1Loss()
loss_l1 = loss_fn(y_pred, y_true)
print("L1Loss（MAE）损失：\n", loss_l1)
loss_fn = torch.nn.MSELoss()
loss_mse = loss_fn(y_pred, y_true)
print("mseLoss（MSE）损失：\n", loss_mse)
loss_fn = torch.nn.SmoothL1Loss()
loss_smooth = loss_fn(y_pred, y_true)
print("SmoothL1Loss（固定参数为1的HuberLoss）损失：\n", loss_smooth)
