# 导包
import torch

# 创建张量
t1 = torch.zeros(size=(2, 6))
print(t1, type(t1), t1.dtype)

# TODO:元素类型转换
# 方式1
print(t1.to(torch.int64).dtype)
print(t1.to(torch.int32).dtype)
print(t1.to(torch.int16).dtype)
print(t1.to(torch.int8).dtype)
print(t1.to(torch.float16).dtype)
print(t1.to(torch.float32).dtype)
print(t1.to(torch.float64).dtype)
print(t1.to(torch.double).dtype)
print(t1.to(torch.uint8).dtype)
print(t1.to(torch.bool).dtype)
print(t1.to(torch.complex64).dtype)
print(t1.to(torch.complex128).dtype)

# 方式2: 直接指定类型转换
print(t1.byte().dtype)
print(t1.short().dtype)
print(t1.int().dtype)
print(t1.long().dtype)
print(t1.half().dtype)
print(t1.float().dtype)
print(t1.double().dtype)

# 方式3: type（指定类型）
print(t1.type(torch.int8).dtype)
print(t1.type(torch.int16).dtype)
print(t1.type(torch.int32).dtype)
print(t1.type(torch.int64).dtype)
print(t1.type(torch.float16).dtype)
print(t1.type(torch.float32).dtype)
print(t1.type(torch.float64).dtype)

# 方式4：方式2和方式3的结合
print(t1.type(torch.int).dtype)
print(t1.type(torch.short).dtype)
print(t1.type(torch.long).dtype)
print(t1.type(torch.half).dtype)
print(t1.type(torch.float).dtype)
print(t1.type(torch.double).dtype)