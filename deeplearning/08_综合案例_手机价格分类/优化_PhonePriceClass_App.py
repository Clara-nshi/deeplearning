import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time


def step1_get_data(data_path):
    # 读取原始数据
    df = pd.read_csv(data_path)
    # print(df.head())
    # print(df.shape)
    # 数据预处理
    df = df.dropna(how="any")
    # print(df.shape)
    # 分别获取x: 特征和y: 标签
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    # todo 提前把x转换为浮点数
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    # print("去重：", y.unique())
    # print(x.shape, y.shape)
    # 切割数据
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # 特征标准化
    ss = StandardScaler()
    x_train = ss.fit_transform(X_train)
    x_test = ss.transform(X_test)
    return x_train, x_test, y_train, y_test


# 构建数据加载器
def step2_get_dataloader(batch_size, x_train, x_test, y_train, y_test):
    # # 先构建 dataset数据集
    # 构建 dataset数据集 —— 正确写法，无警告
    train_dataset = TensorDataset(
        torch.tensor(x_train),  # 自动复制，解决只读
        torch.tensor(y_train.values)
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test),
        torch.tensor(y_test.values)
    )
    # print(train_dataset)
    # 再构建 data_loader 数据加载器, 把完整的数据进行分批操作
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
    # print(train_dataloader)
    # print(test_dataloader)
    # for i in train_dataloader:
    #     print(i)
    #     break
    # 返回数据加载器
    return train_dataloader, test_dataloader


class PhoneModel(torch.nn.Module):
    # 重写 init 魔法方法，创建模型的时候会自动调用
    def __init__(self, input_num, output_num):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_num, 128)
        self.linear2 = torch.nn.Linear(128, 256)
        self.linear3 = torch.nn.Linear(256, 256)
        self.out = torch.nn.Linear(256, output_num)

    # 重写 forward 魔法方法，定义网络结构
    def forward(self, x):
        # 加权求和+激活函数
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        # 因为后续要用多分类交叉熵损失函数，所以这里不加softmax函数
        x = self.out(x)
        return x


def train_model(data_loader, model, lr, model_path):
    # 设置模型为训练模式
    model.train()
    # todo 1. 准备数据（此处已经传入）
    # todo 2. 准备模型（此处已经传入）
    # todo 3. 创建损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    # todo 4. 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # todo 5. 外层循环控制轮次
    for epoch in range(1, epochs+1):
        # 为了打印日志，提前记录总损失和轮次和时间
        total_loss, batch_cnt, start_time = 0, 0, time.time()
        print(f"第{epoch}轮训练...")
        # todo 6. 内层循环控制批次
        for batch_x, batch_y in data_loader:
            # todo 7. 前向传播
            logits_y = model(batch_x)
            # todo 8. 计算损失
            loss = loss_fn(logits_y, batch_y)
            total_loss += loss.item()
            batch_cnt += 1
            # todo 9. 梯度清零
            optimizer.zero_grad()
            # todo 10. 反向传播
            loss.backward()
            # todo 11. 参数更新
            optimizer.step()
        #  打印日志
        print(f"第{epoch}轮，平均损失：{total_loss / batch_cnt:.4f}, 用时：{time.time() - start_time:.4f}秒")
    # todo 12. 模型保存
    torch.save(model.state_dict(), model_path)


# todo 模型评估
def eval_model(data_loader, modelpath):
    # todo 1. 准备数据（此处已传参）
    # todo 2. 准备模型（创建一个新模型，加载已经训练好的参数）
    model = PhoneModel(input_num, output_num)
    model.load_state_dict(torch.load(modelpath))
    # todo 3. 设置模型为评估模式
    model.eval()    # 禁用 dropout，同时改变 BN 计算方式
    # 禁用梯度计算以减少内存和加速
    with torch.no_grad():
        # todo 3. 遍历数据加载器
        # 提前设置 pred_true_cnt记录预测正确的数量
        pre_true_cnt = 0
        for batch_x, batch_y in data_loader:
            # todo 4. 前向传播
            logits_y = model(batch_x)
            print(logits_y)
            # todo 5. 获取预测结果
            pre_y = logits_y.argmax(dim=1)
            # todo 统计预测正确的数量
            pre_true_cnt += (pre_y == batch_y).sum()
            print(pre_true_cnt)
        # todo 5. 计算准确率    预测对的预测数
        acc = pre_true_cnt / len(data_loader.dataset)
        print(f"准确率{acc:.4f}")


if __name__ == '__main__':
    # todo 优化1：特征标准化
    # todo 优化2：增加网络深度
    # todo 优化3：修改优化器
    # todo 优化4：降低学习率
    # todo 优化5：调整批次大小
    # todo 优化6：调整轮次大小
    # 各种超参数/路径
    datapath = "./data/手机价格预测.csv"
    batch_size = 16
    model_path = "./model/手机价格分类模型.pth"
    epochs = 50
    lr = 0.0001
    # 提前获取数据并切割数据
    x_train, x_test, y_train, y_test = step1_get_data(datapath)
    # 提前构建数据加载器  todo 自动把原始的整体数据分成多批数据
    train_data_loader, test_data_loader = step2_get_dataloader(batch_size, x_train, x_test, y_train, y_test)
    # 提前准备模型
    input_num = x_train.shape[1]
    output_num = len(y_train.unique())
    model = PhoneModel(input_num, output_num)
    # TODO: 模型训练（传入数据加载器和模型）
    train_model(train_data_loader, model, lr, model_path)
    # for x, y in train_data_loader:
    #     out = model(x)
    #     print("输出结果：", out)
    #     break
    # 模型评估
    eval_model(test_data_loader, model_path)


