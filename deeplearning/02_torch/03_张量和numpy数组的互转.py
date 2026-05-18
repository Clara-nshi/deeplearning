# TODO 1. 张量和numpy数组互转
import numpy as np
import torch

# 创建numpy数组
my_numpy = np.array([[1, 2, 3], [4, 5, 6]])
print(my_numpy, type(my_numpy))
# torch.from_numpy(): 将numpy转换为张量, 会共享内存
t1 = torch.from_numpy(my_numpy)
print(t1, type(t1))
my_numpy[0] = 100
print(t1, type(t1))

# TODO torch.tensor：不会共享内存
print("************************")
t2 = torch.tensor(my_numpy)
my_numpy[1] = 200
print(t2, type(t2))


# 创建张量
print("***************")
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(my_tensor, type(my_tensor))
# numpy()：将张量转换为numpy数组，会共享内存
n1 = my_tensor.numpy()
print(t2, type(t2))
my_tensor[0] = 9
print(n1, type(n1))

# TODO numpy().copy()：将张量转换为numpy数组，不会共享内存
n2 = my_tensor.numpy().copy()
print(n2, type(n2))
my_tensor[1] = 9
print(n2, type(n2))
