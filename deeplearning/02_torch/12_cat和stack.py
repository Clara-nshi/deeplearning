import torch

# 设置种子
torch.manual_seed(6)
# todo cat(): 拼接两个张量，维度数不变，指定维度位置拼接
t1 = torch.randint(low=1, high=5, size=(3, 4))
t2 = torch.randint(low=1, high=5, size=(3, 4))
print(t1, t1.ndim, t1.shape)
print(t2, t2.ndim, t2.shape)

cat1 = torch.cat([t1, t2], dim=0)
print(cat1, cat1.ndim, cat1.shape)
cat2 = torch.cat([t1, t2], dim=1)
print(cat2, cat2.ndim, cat2.shape)

# todo：注意:cat(): 除了拼接维度外，其他维度必须相同
t3 = torch.randint(low=1, high=5, size=(3, 5))
cat3 = torch.cat([t1, t3], dim=1)
print(cat3, cat3.ndim, cat3.shape)

t4 = torch.randint(low=1, high=5, size=(5, 4))
cat4 = torch.cat([t1, t4], dim=0)
print(cat4, cat4.ndim, cat4.shape)


print("****************************************************************************************************************")
# todo stack(): 拼接两个张量，维度数不变，指定维度位置拼接，两个张量形状必须完全一样
t1 = torch.randint(low=1, high=5, size=(3, 4))
t2 = torch.randint(low=1, high=5, size=(3, 4))
print(t1, t1.ndim, t1.shape)
print(t2, t2.ndim, t2.shape)
stack1 = torch.stack([t1, t2], dim=0)
print(stack1, stack1.ndim, stack1.shape)
stack2 = torch.stack([t1, t2], dim=1)
print(stack2, stack2.ndim, stack2.shape)
stack3 = torch.stack([t1, t2], dim=2)
print(stack3, stack3.ndim, stack3.shape)


