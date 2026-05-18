import torch


def adaGrad():
    # 准备初始权重数据
    w = torch.tensor([1.0], requires_grad=True)
    # 准备优化器
    # todo 原始的 Adagrad  问题：受梯度平方和单调递增影响，学习率过早的衰减，导致参数无法更新到最优
    optimizer = torch.optim.Adagrad([w], lr=0.01)
    for i in range(5):
        # 模拟损失函数
        loss = 0.5 * w ** 2
        print("损失值：", loss)
        # 梯度清零
        optimizer.zero_grad()
        # 开始反向传播
        loss.backward()
        print("梯度值：", w.grad)
        # 计算参数
        optimizer.step()
        print("更新后的参数：", w)


def rmsprop():
    # 准备初始权重数据
    w = torch.tensor([1.0], requires_grad=True)

    # 准备优化器
    # todo 引入指数加权平均思想   缓解了学习率下降过快的问题，参数更新速度更加平滑
    optimizer = torch.optim.RMSprop([w], lr=0.01, alpha=0.9)
    for i in range(5):
        # 模拟损失函数
        loss = 0.5 * w ** 2
        print("损失值：", loss)
        # 梯度清零
        optimizer.zero_grad()
        # 开始反向传播
        loss.backward()
        print("梯度值：", w.grad)
        # 计算参数
        optimizer.step()
        print("更新后的参数：", w)





if __name__ == '__main__':
    adaGrad()
    print("+++++++++++++++++++++++")
    rmsprop()