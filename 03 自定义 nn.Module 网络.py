import torch
import torch.nn as nn
from torchsummary import summary  # 用于计算和查看模型参数量、结构


# ===================== 1. 搭建神经网络 =====================
class ModelDemo(nn.Module):
    """
    自定义神经网络类，必须继承 nn.Module。
    网络结构：输入层(3) → 隐藏层1(3, sigmoid) → 隐藏层2(2, ReLU) → 输出层(2, 无激活/回归)
    """

    def __init__(self):
        super().__init__()  # 调用父类 nn.Module 的初始化方法（必须写）

        # --- 定义网络的层（搭积木）---
        # 隐藏层1：输入特征数 3，输出特征数 3
        self.Linear1 = nn.Linear(in_features=3, out_features=3)
        # 隐藏层2：输入特征数 3，输出特征数 2
        self.Linear2 = nn.Linear(in_features=3, out_features=2)
        # 输出层：输入特征数 2，输出特征数 2（回归任务，直接输出数值）
        self.output = nn.Linear(in_features=2, out_features=2)

        # --- 参数初始化（让训练更稳定）---
        # 隐藏层1：Xavier 初始化权重（适合 sigmoid），偏置初始化为 0
        nn.init.xavier_normal_(self.Linear1.weight)
        nn.init.zeros_(self.Linear1.bias)
        # 隐藏层2：Kaiming 初始化权重（适合 ReLU），偏置初始化为 0
        nn.init.kaiming_normal_(self.Linear2.weight)
        nn.init.zeros_(self.Linear2.bias)
        # 输出层：正态分布初始化权重，偏置为 0
        nn.init.normal_(self.output.weight, mean=0, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        """
        前向传播函数：定义数据在网络中的流动路径。
        注意：函数名必须是 forward，PyTorch 会自动调用它。
        """
        # 第1层：线性计算 + sigmoid 激活函数（引入非线性）
        x = torch.sigmoid(self.Linear1(x))
        # 第2层：线性计算 + ReLU 激活函数（把负数变成 0）
        x = torch.relu(self.Linear2(x))
        # 输出层：直接输出（回归任务不需要 softmax，softmax 用于分类）
        x = self.output(x)
        return x


# ===================== 2. 真正的训练函数 =====================
def train():
    # --- 2.1 选择设备（有 GPU 用 GPU，否则用 CPU）---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'当前使用设备: {device}')

    # --- 2.2 实例化模型并移动到设备 ---
    my_model = ModelDemo()
    my_model.to(device)

    # --- 2.3 构造训练数据（回归任务）---
    # 固定随机数种子
    torch.manual_seed(3)
    # 输入：5 个样本，每个样本 3 个特征（随机生成）
    x_train = torch.randn(size=(5, 3)).to(device)
    # 标签：5 个样本，每个样本对应 2 个真实值（随机生成，仅作演示）
    y_true = torch.randn(size=(5, 2)).to(device)

    print(f'\n输入数据形状: {x_train.shape}')  # 期望: (5, 3)

    # --- 2.4 定义损失函数和优化器 ---
    criterion = nn.MSELoss()  # 均方误差损失（回归任务标配）
    optimizer = torch.optim.SGD(my_model.parameters(), lr=0.01)  # 随机梯度下降

    # --- 2.5 训练循环 ---
    epochs = 100
    for epoch in range(epochs):
        # 前向传播：把输入扔进模型，得到预测值
        y_pred = my_model(x_train)

        # 计算损失
        loss = criterion(y_pred, y_true)

        # 反向传播 + 参数更新（标准四步）
        optimizer.zero_grad()  # 1. 清空旧梯度
        loss.backward()  # 2. 反向传播，计算新梯度
        optimizer.step()  # 3. 更新参数

        # 每 20 轮打印一次损失
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    print(f'\ny_true = {y_true}')
    print(f'y_pred = {y_pred}')

    # --- 2.6 训练后查看模型参数 ---
    print('\n========================= 模型参数 =========================')
    for name, param in my_model.named_parameters():
        print(f'参数名: {name}, \n参数值: {param}\n')

    # --- 2.7 用 torchsummary 查看网络结构（注意 input_size 只传特征维度）---
    print('\n========================= 网络结构 =========================')
    # 注意：summary 期望的 input_size 是单个样本的形状，即 (3,)
    summary(my_model, input_size=(3,), device=str(device))


# ===================== 3. 主程序入口 =====================
if __name__ == '__main__':
    train()
