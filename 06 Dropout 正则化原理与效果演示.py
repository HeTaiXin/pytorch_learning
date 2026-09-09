"""
案例：
    演示（随机）丢弃法正则化（torch.nn.Dropout()）→ 防止神经网络过拟合

1. Dropout 核心思想：
   在神经网络的训练阶段，它会以给定的概率 p 随机地将某些神经元的输出（激活值）置为 0。
   为了保证训练和测试时该层输出的总期望不变，Dropout 会将**未被丢弃**的神经元输出值缩放 1/(1-p) 倍。

   举个例子：假设一个神经元输出是 2，p=0.5。
   - 训练时：有 50% 概率输出 0，有 50% 概率输出 2 / (1-0.5) = 4。
   - 期望输出：0 * 0.5 + 4 * 0.5 = 2，与没有 Dropout 时一致。

2. 关键参数：
   - p：丢弃概率（默认 0.5）。即训练时每个神经元有 50% 的概率被临时“关闭”。
   - 被关闭的神经元既不参与前向传播，也不参与反向传播（梯度不会流向它）。

3. 训练模式 vs 评估模式（⚠️ 核心重点）：
   - model.train()：启用 Dropout，进行随机丢弃和缩放。
   - model.eval()：禁用 Dropout，所有神经元都参与计算，不再丢弃和缩放。
     （测试时需要确定的输出，不能带随机性）

4. 使用建议：
   - 通常放在全连接层或激活函数之后（本例放在激活函数之后）。
   - 常用 p 值范围：0.2 ~ 0.5。过高（如 0.8）可能导致欠拟合，过低（如 0.1）正则化效果不明显。
"""

import torch
import torch.nn as nn


# 1. 搭建神经网络，继承 nn.Module
class ModelDemo(nn.Module):
    def __init__(self, dropout_prob1=0.5, dropout_prob2=0.5):
        super().__init__()

        # --- 定义网络层 ---
        self.Linear1 = nn.Linear(3, 3)  # 隐藏层1
        self.dropout1 = nn.Dropout(p=dropout_prob1)  # Dropout 层1

        self.Linear2 = nn.Linear(3, 2)  # 隐藏层2
        self.dropout2 = nn.Dropout(p=dropout_prob2)  # Dropout 层2

        self.output = nn.Linear(2, 2)  # 输出层

        # --- 参数初始化（为了演示，全部初始化为 1）---
        # 这样经过 Linear 层后，输出值 = 输入和 + 1，方便观察 Dropout 的缩放效果
        nn.init.ones_(self.Linear1.weight)
        nn.init.ones_(self.Linear1.bias)
        nn.init.ones_(self.Linear2.weight)
        nn.init.ones_(self.Linear2.bias)
        nn.init.ones_(self.output.weight)
        nn.init.ones_(self.output.bias)

    def forward(self, x):
        # 隐藏层 1：线性变换 → Sigmoid 激活 → Dropout 正则化
        x = self.Linear1(x)
        x = torch.sigmoid(x)
        print(f'hidden layer 1 (without dropout): {x}')
        x = self.dropout1(x)  # 训练时：随机置零 + 缩放；评估时：原样输出
        print(f'hidden layer 1 (with dropout): {x}')

        # 隐藏层 2：线性变换 → ReLU 激活 → Dropout 正则化
        x = self.Linear2(x)
        x = torch.relu(x)
        print(f'hidden layer 2 (without dropout): {x}')
        x = self.dropout2(x)
        print(f'hidden layer 2 (with dropout): {x}')

        # 输出层：线性变换 → Softmax（得到概率分布）
        x = torch.softmax(self.output(x), dim=-1)
        return x


# 2. 模型训练与评估演示
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}\n')

    # 创建模型，设置 Dropout 概率为 0.5
    my_model = ModelDemo(dropout_prob1=0.5, dropout_prob2=0.5).to(device)

    # 创建输入数据：5 个样本，每个样本 3 个特征
    dataset = torch.linspace(1, 15, 15).reshape(5, 3).to(device)
    #     torch.manual_seed(2)
    #     dataset = torch.randn(5, 3, dtype=torch.float32).to(device)
    print(f'输入数据:\n{dataset}\n')

    # ========== 训练模式（Dropout 生效）==========
    my_model.train()  # 切换到训练模式
    print("=" * 50)
    print("训练模式 (model.train()) → Dropout 生效")
    print("=" * 50)
    train_output = my_model(dataset)
    print(f'\n最终输出 (训练模式，带随机性):\n{train_output}\n')

    # ========== 评估模式（Dropout 不生效）==========
    my_model.eval()  # 切换到评估模式
    print("=" * 50)
    print("评估模式 (model.eval()) → Dropout 不生效")
    print("=" * 50)

    # torch.no_grad() 关闭梯度计算，节省内存，是评估时的标准做法
    with torch.no_grad():
        eval_output = my_model(dataset)
        print(f'\n最终输出 (评估模式，确定性的):\n{eval_output}\n')


if __name__ == '__main__':
    train()