# 张量基础运算
import torch
a1 = torch.tensor([1, 2, 3], dtype=torch.float32)
"""a1不变"""
print(a1 + 2)
print(a1 - 2)
print(a1 * 2)
print(a1 / 2)
print(a1 % 2)
print(a1 ** 2)
print(a1 // 2)
"""
a1不变
"""
print(torch.add(a1, 2))
print(torch.sub(a1, 2))
print(torch.mul(a1, 2))
print(torch.div(a1, 2))
print(torch.remainder(a1, 2))
print(torch.pow(a1, 2))
print(torch.floor_divide(a1, 2))
print(torch.sqrt(a1))
print(torch.log(a1))
print(torch.exp(a1))
"""每次算完之后a1会发生改变"""
a1.add_(2)
print(a1)
a1.sub_(2)
print(a1)
a1.mul_(2)
print(a1)
a1.div_(2)
print(a1)
"""对元素正负取反"""
print(torch.neg(a1))
a1.neg_()
print(a1)
"""对元素取绝对值"""
print(torch.abs(a1))
a1.abs_()
print(a1)

z1 = torch.tensor([[1, 2, 3], [4, 5, 6]] )
z1_sum = torch.sum(z1)
print(z1_sum)

























