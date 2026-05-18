# 导包
import torch


def test_SGD():
    w = torch.tensor([1.0], requires_grad=True)
    # todo 准备优化器
    # 方式1：原始的SGD    问题：梯度消失和梯度震荡
    optimizer = torch.optim.SGD([w], lr=0.01)
    # todo 反向传播更新参数
    # 循环迭代 5 次
    for i in range(5):
        # 模拟计算损失
        loss = 0.5 * w ** 2
        print("损失", loss)
        # 梯度会累加,所以一定要在参数更新前梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        print("梯度", w.grad)
        # 更新参数
        optimizer.step()
        print("参数w", w)


def test_SGD_Momentum():
    w = torch.tensor([0.1], requires_grad=True)
    # 准备优化器
    optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)
    for i in range(5):
        # 优化器梯度清零
        optimizer.zero_grad()
        # 模拟计算损失
        loss = 0.5*w**2
        print("损失", loss)
        # 计算梯度
        loss.backward()
        print("梯度", w.grad)
        # 更新参数
        optimizer.step()
        print("更新后的参数", w)


if __name__ == '__main__':
    # TODO 使用原生SGD优化器   问题：容易梯度鞍点停滞不前和梯度震荡
    test_SGD()
    # TODO 使用SGD+动量优化法优化器   优势：引入指数加权平均思想，缓解梯度鞍点停滞不前和梯度震荡，让梯度更加平滑
    test_SGD_Momentum()

