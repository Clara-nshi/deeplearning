import torch
# 创建标量
a = 1
print(a, type(a))
# 标量转张量
a1 = torch.tensor(a)
print(a1, type(a1))

# TODO： 重要张量转标量
print("**************")
t = torch.tensor(1)
print(t, type(t), t.ndim)
t1 = t.item()
print(t1, type(t1))

t = torch.tensor([1])
print(t, type(t), t.ndim)
t1 = t.item()
print(t1, type(t1))

t = torch.tensor([[1]])
print(t, type(t), t.ndim)
t1 = t.item()
print(t1, type(t1))

t = torch.tensor([1, 1, 2, 3])
print(t, type(t), t.ndim)
t1 = t.sum().item()
print(t1, type(t1))
