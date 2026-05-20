import torch


# todo 如果张量设置了自动微分，就不能直接用numpy()转换了
t1 = torch.tensor([1, 2, 3])
print(type(t1))
t2 = t1.numpy()
print(type(t2))
print("======================================")
# todo 如果张量设置了自动微分，就不能直接用numpy()转换了
# todo detch()作用是：拷贝张量内容，但是自动关闭自动微分，没有开辟新内存
t1 = torch.tensor([1, 2, 3], requires_grad=True, dtype=torch.float)
print(type(t1))
print("11", t1[0])
t2 = t1.detach().numpy()
print(type(t2))
print("======================================")
# todo 如果张量设置了自动微分，就不能直接用numpy()转换了
t1 = torch.tensor([1, 2, 3])
print(type(t1))
t2 = t1.detach().numpy()
print(type(t2))
print("======================================")
# todo clone()的作用的：拷贝张量内容，但是自动打开自动微分，开辟了新内存