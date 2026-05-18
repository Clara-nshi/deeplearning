# 导包
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def test_scheduler_lr(scheduler_lr):
    # todo 1.开始训练之前，提前创建两个列表，分别存储轮次和学习率，方便后续绘图
    epoch_list, lr_list = [], []
    # todo 外层循环控制轮次 200 轮
    for epoch in range(0, 200):
        # todo 2.存储轮次和学习率
        lr_list.append(scheduler_lr.get_last_lr())
        epoch_list.append(epoch)
        # todo 内层循环控制批次(可以不要，更新参数的，为了保证流程完整性)
        for batch in range(0, 10):
            # 模拟损失函数
            loss = 0.5 * (w * x - y) ** 2
            # 默认梯度会累加，所以一定在参数更新前要梯度清零
            optimizer.zero_grad()
            # 开始反向传播（计算梯度）
            loss.backward()
            # 参数更新
            optimizer.step()
            # print(f"第{epoch}轮，第{batch}批次，损失值:{loss},参数值:{w}")
        # todo 学习率更新
        scheduler_lr.step()
    # todo 3. 绘制学习率曲线
    plt.plot(epoch_list, lr_list, c='blue')
    plt.title("学习率曲线")
    plt.xlabel("轮次")
    plt.ylabel("学习率衰减")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # todo 模拟准备数据
    x = torch.tensor([1.0])
    y = torch.tensor([0])
    w = torch.tensor([1.0], requires_grad=True)
    # todo 准备优化器
    optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)
    # todo 创建学习率衰减对象
    # 方式1：等间隔学习率衰减
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    test_scheduler_lr(scheduler_lr)
    # 方式2：指定学习衰减
    scheduler_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.5, milestones=[50, 125, 160])
    test_scheduler_lr(scheduler_lr)
    # 方式3：指数衰减学习率
    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    test_scheduler_lr(scheduler_lr)





