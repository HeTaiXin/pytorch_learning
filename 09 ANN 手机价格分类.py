"""
案例：
    ANN（人工神经网络）实现手机价格区间分类（多分类任务）。

数据集说明：
    输入：20 个手机特征（电池、内存、像素等）。
    输出：价格区间（4 类，如 0=低价, 1=中低价, 2=中高价, 3=高价）。
    本例演示标准的 训练/验证/测试 三阶段流程。
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time

# 全局设备（GPU 优先，否则 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 工具函数导入（假设 toolkit.py 已存在）====================
# 如果你的环境没有 toolkit.py，可以直接用 nn.BatchNorm1d / nn.Dropout 替代，见下方 Net 类的注释
from toolkit import get_batch_norm, get_dropout

bn_input_params = {'use_bn': False, 'dim': 1, 'bn_momentum': 0.1, 'affine': True}
bn_params = {'use_bn': True, 'dim': 1, 'bn_momentum': 0.1, 'affine': True}
dropout_param1 = {'use_dropout': False, 'dropout_prob': 0.1, 'dim': None}
dropout_param2 = {'use_dropout': False, 'dropout_prob': 0.4, 'dim': None}


# ==================== 1. 构建数据集 ====================
def create_dataset():
    """
    读取 CSV，切分为 训练集/验证集/测试集（比例 64%/16%/20%）。
    返回三个 TensorDataset 以及输入特征数和类别数。
    """
    # 读取数据（请确保路径正确）
    data = pd.read_csv('data/phone_price/手机价格预测.csv')

    # 假设标签列名为 'price_range'（Kaggle 经典数据集），如果不是请修改这里
    target_col = 'price_range'
    if target_col not in data.columns:
        # 兜底：取最后一列作为标签
        target_col = data.columns[-1]

    # 特征 X，标签 y
    x = data.drop(columns=[target_col]).astype('float32').values
    y = data[target_col].astype('int64').values  # 分类标签用长整型

    # 切分：先分出测试集（20%），剩下 80% 再分训练/验证
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.2, random_state=4
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, train_size=0.8, random_state=41
    )

    # 封装成 TensorDataset（输入特征用 float32，标签用 long 类型）
    train_data = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_data = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    test_data = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    input_dim = x.shape[1]  # 输入特征维度（20）
    output_dim = len(np.unique(y))  # 类别数（4）
    return train_data, val_data, test_data, input_dim, output_dim


# ==================== 2. 搭建神经网络 ====================
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 输入层批归一化（可选）
        self.bn_input = get_batch_norm.get_batch_norm(input_dim, **bn_input_params)

        # 隐藏层 1
        self.linear1 = nn.Linear(input_dim, 300)
        self.bn1 = get_batch_norm.get_batch_norm(300, **bn_params)
        self.dropout1 = get_dropout.get_dropout(**dropout_param1)

        # 隐藏层 2
        self.linear2 = nn.Linear(300, 260)
        self.bn2 = get_batch_norm.get_batch_norm(260, **bn_params)
        self.dropout2 = get_dropout.get_dropout(**dropout_param2)

        # 输出层（分类头）：输出维度=类别数
        # ⚠️ 注意：分类任务输出层不加 Softmax！因为 CrossEntropyLoss 内部已包含 Softmax
        self.output = nn.Linear(260, output_dim)

    def forward(self, x):
        # 输入层
        x = self.bn_input(x)
        # 隐藏层 1：线性 → BN → 激活(ReLU) → Dropout
        x = self.linear1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        # 隐藏层 2：线性 → BN → 激活(ReLU) → Dropout
        x = self.linear2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        # 输出层（原始 logits，不经过 softmax）
        return self.output(x)


# ==================== 3. 模型训练 ====================
def train(model, train_data, val_data):
    print('=' * 60)
    print('模型训练')
    print('=' * 60)

    # 数据加载器：训练集打乱，验证集不打乱
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

    # 损失函数：交叉熵（已内含 Softmax，标签需为类别索引）
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # 优化器：SGD（也可用 Adam）
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    epochs = 50
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # -------- 训练阶段 --------
        model.train()  # ⚠️ 启用 Dropout 和 BatchNorm 的训练行为
        total_loss, batch_num = 0.0, 0
        start = time.time()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            y_pred = model(x_batch)  # 前向传播
            loss = criterion(y_pred, y_batch)  # 计算损失

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()  # 累加本 epoch 所有 batch 的损失
            batch_num += 1

        avg_loss = total_loss / batch_num
        end = time.time()

        # -------- 验证阶段（每个 epoch 后评估一次）--------
        val_acc = evaluate_accuracy(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'data/phone_price/best_model.pth')  # 保存最佳模型

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch [{epoch:02d}/{epochs}] | Loss: {avg_loss:.4f} | '
                  f'Val Acc: {val_acc:.4f} | Time: {end - start:.2f}s')

    print(f'\n训练结束，最佳验证集准确率: {best_val_acc:.4f}')
    return best_val_acc


# ==================== 4. 评估准确率（通用函数）====================
@torch.no_grad()  # 关闭梯度计算，节省内存
def evaluate_accuracy(model, data_loader):
    """
    在给定数据集上计算分类准确率。
    配合 @torch.no_grad() 和 model.eval() 使用。
    """
    model.eval()  # ⚠️ 关闭 Dropout，BatchNorm 用移动均值/方差
    correct, total = 0, 0

    for x_data, y_data in data_loader:
        x_data = x_data.to(device, non_blocking=True)
        y_data = y_data.to(device, non_blocking=True)

        y_pred = model(x_data)  # 前向传播（输出 logits）
        print(f'y_pred（每个类别的得分）：{y_pred.detach()}\n')

        y_pred_class = torch.argmax(y_pred, dim=1)  # 取概率最大的类别
        correct += (y_pred_class == y_data).sum().item()
        total += y_data.size(0)

    return correct / total if total > 0 else 0.0


# ==================== 5. 测试函数（独立、清晰）====================
def test(model, test_data):
    print('=' * 60)
    print('模型测试（最终评估）')
    print('=' * 60)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    # 加载训练过程中保存的最佳模型参数
    model.load_state_dict(torch.load('data/phone_price/best_model.pth', map_location=device))
    test_acc = evaluate_accuracy(model, test_loader)
    print(f'测试集准确率: {test_acc:.4f}')
    print('-' * 60)


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # 1. 准备数据集
    train_data, val_data, test_data, input_dim, output_dim = create_dataset()
    print(f'输入特征维度: {input_dim}, 类别数: {output_dim}')
    print(f'使用设备: {device}\n')

    # 2. 构建模型并移到设备
    model = Net(input_dim=input_dim, output_dim=output_dim).to(device)

    # 3. 训练（内含验证，自动保存最佳模型）
    train(model, train_data, val_data)

    # 4. 在独立测试集上做最终评估
    test(model, test_data)
