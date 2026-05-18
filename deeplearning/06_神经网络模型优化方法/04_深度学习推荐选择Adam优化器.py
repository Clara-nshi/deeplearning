import torch


def test_Adam():
    # 初始化权重数据
    w = torch.tensor([1.0], requires_grad=True)
    # 准备优化器
    # todo 动量法 + RMSprop    同时优化梯度和学习率  betas=（动量法的系数，RMSprop系数）
    optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.999))
    for i in range(5):
        # 损失函数
        loss = 0.5*w**2
        print("损失：", loss)
        optimizer.zero_grad()
        loss.backward()
        print("梯度：", w.grad)
        optimizer.step()
        print("参数：", w)


if __name__ == '__main__':
    test_Adam()
