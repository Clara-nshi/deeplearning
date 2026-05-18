# 导包
import torch


# todo 1. 构建神经网络
# 核心步骤： 1个继承2个重写
class my_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print("init执行了")
        # 创建网络层（默认初始化 w 和 b）
        self.fc1 = torch.nn.Linear(3, 3)
        self.fc2 = torch.nn.Linear(3, 2)
        self.out = torch.nn.Linear(2, 2)
        # 手动初始化 w 权重
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        # 手动初始化 b 偏置
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # self.fc1加权求和激活函数
        x = torch.sigmoid(self.fc1(x))
        # self.fc2加权求和激活函数
        x = torch.relu(self.fc2(x))
        # self.out加权求和激活函数
        logits = self.out(x)
        print("分数", self.out(x))
        prob = torch.softmax(logits, dim=1)
        # 返回结果
        return prob


if __name__ == '__main__':
    # todo 2. 调用神经网络，拿到预测结果
    # todo 2.1 先准备数据
    torch.manual_seed(55)
    x = torch.randn(5, 3)
    print(x)
    # todo 2.2 创建模型
    model = my_model()
    # todo 2.3 前向传播
    preds = model(x)
    print(preds)

