# 通用方式
import torch

t1 = torch.tensor([1, 2, 2])
t2 = torch.tensor([2, 3, 4])
re = torch.matmul(t1, t2)
print(re)

t1 = torch.tensor([[1, 2], [2, 3], [4, 5]])
t2 = torch.tensor([[2, 3], [5, 6], [3, 4]])
t3 = torch.tensor([[1, 2, 3], [4, 5, 6]])

ree = t1.T @ t2
print(ree)
ree1 = t1 @ t2.T
print(ree1)
ree2 = t1 @ t3
print(ree2)
print("**********")
re = torch.matmul(t1.T, t2)
re1 = torch.matmul(t1, t2.T)
re2 = torch.matmul(t1, t3)
print(re)
print(re1)
print(re2)


