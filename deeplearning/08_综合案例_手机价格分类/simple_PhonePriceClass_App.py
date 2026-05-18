import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# ===================== 1. 数据读取与预处理 =====================
def step1_get_data(data_path):
    # 读取原始数据
    df = pd.read_csv(data_path)

    # 去除缺失值
    df = df.dropna(how="any")

    # 分离特征 X 和标签 y
    x = df.iloc[:, 0:-1].values  # 直接转numpy
    y = df.iloc[:, -1].values

    # 统一数据类型（适配PyTorch）
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 划分训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

    # 特征标准化
    ss = StandardScaler()
    x_train = ss.fit_transform(X_train)
    x_test = ss.transform(X_test)

    return x_train, x_test, y_train, y_test


# ===================== 2. 构建数据加载器 =====================
def step2_get_dataloader(batch_size, x_train, x_test, y_train, y_test):
    # 构建数据集
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.int64)
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.int64)
    )

    # 构建数据加载器（分批加载）
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


# ===================== 3. 定义神经网络模型 =====================
class PhoneModel(torch.nn.Module):
    def __init__(self, input_num, output_num):
        super().__init__()
        # 全连接层
        self.linear1 = torch.nn.Linear(input_num, 128)
        self.linear2 = torch.nn.Linear(128, 256)
        self.out = torch.nn.Linear(256, output_num)

    def forward(self, x):
        # 前向传播
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.out(x)
        return x


# ===================== 4. 训练与评估函数 =====================
def train_model(model, train_loader, test_loader, epochs=50, lr=0.001):
    # 定义损失函数（多分类任务）
    criterion = torch.nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    for epoch in range(epochs):
        model.train()  # 训练模式
        train_loss = 0.0
        correct = 0
        total = 0

        # 遍历训练批次
        for x, y in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(x)
            # 计算损失
            loss = criterion(outputs, y)
            # 反向传播 + 更新参数
            loss.backward()
            optimizer.step()

            # 累计损失和准确率
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        # 计算平均损失和准确率
        avg_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        # ========== 每轮结束后评估测试集 ==========
        model.eval()  # 评估模式
        test_correct = 0
        test_total = 0
        with torch.no_grad():  # 关闭梯度计算
            for x, y in test_loader:
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()
        test_acc = 100 * test_correct / test_total

        # 打印日志
        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"训练损失: {avg_loss:.4f} | "
              f"训练准确率: {train_acc:.2f}% | "
              f"测试准确率: {test_acc:.2f}%")


# ===================== 主程序入口 =====================
if __name__ == '__main__':
    # 配置参数
    datapath = "./data/手机价格预测数据集.csv"
    batch_size = 8
    epochs = 50
    learning_rate = 0.001

    # 1. 数据预处理
    x_train, x_test, y_train, y_test = step1_get_data(datapath)

    # 2. 构建数据加载器
    train_loader, test_loader = step2_get_dataloader(batch_size, x_train, x_test, y_train, y_test)

    # 3. 初始化模型
    input_feature_num = x_train.shape[1]  # 特征数量
    class_num = len(np.unique(y_train))  # 分类类别数
    model = PhoneModel(input_feature_num, class_num)

    # 4. 训练 + 评估
    print("=" * 50)
    print("开始训练模型...")
    print("=" * 50)
    train_model(model, train_loader, test_loader, epochs=epochs, lr=learning_rate)

    print("\n训练完成！")
