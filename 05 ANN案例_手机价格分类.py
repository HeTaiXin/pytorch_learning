"""
案例：
    ANN（人工神经网络）案例：手机价格分类案例。

背景：
    基于手机的 20 例特征 → 预测手机的价格区间（4 个区间），可以用机器学习做，也可以用 深度学习做（推荐）。
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# todo 1：构建数据集
def create_dataset(device):
    data = pd.read_csv('data/phone_price/手机价格预测.csv')
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x = x.astype('float32')  # 输入特征转换为浮点型
    # 切分训练集和测试集
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    # 从原训练集中切分训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, train_size=0.8, random_state=41)
    # 把数据封装为张量数据集，思路：数据 → tensor → 数据集对象（TensorDataset）→ 数据加载器（DataLoader）
    train_data = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    val_data = TensorDataset(torch.tensor(x_val.values), torch.tensor(y_val.values))
    test_data = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    # x.shape[1] → 神经网络输入层特征数, len(np.unique(y)) → 神经网络输出层标签数
    return train_data, val_data, test_data, x.shape[1], len(np.unique(y))

# todo：批归一化函数
from toolkit import get_batch_norm
bn_input_params = {'use_bn': False, 'dim': 1, 'bn_momentum': 0.1, 'affine': True}
bn_params = {'use_bn': True, 'dim': 1, 'bn_momentum': 0.1, 'affine': True}
# todo：Dropout函数
from toolkit import get_dropout
dropout_param1 = {'use_dropout': False, 'dropout_prob': 0.1, 'dim': None}
dropout_param2 = {'use_dropout': False, 'dropout_prob': 0.4, 'dim': None}

# todo 2: 搭建神经网络
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 输入层批归一化
        self.bn_input = get_batch_norm.get_batch_norm(input_dim, **bn_input_params)
        # 隐藏层 1 准备
        self.linear1 = nn.Linear(in_features=input_dim, out_features=300)  # 加权求和（线性变换）
        self.bn1 = get_batch_norm.get_batch_norm(300, **bn_params)  # 批归一化处理
        self.dropout1 = get_dropout.get_dropout(**dropout_param1)   # 数据非线性变换(激活函数)后的随机失活
        # 隐藏层 2 准备
        self.linear2 = nn.Linear(in_features=300, out_features=260)
        self.bn2 = get_batch_norm.get_batch_norm(260, **bn_params)
        self.dropout2 = get_dropout.get_dropout(**dropout_param2)
        # 输出层准备
        self.output = nn.Linear(in_features=260, out_features=output_dim)

        # 参数初始化
        # nn.init.xavier_uniform_(self.linear1.weight)
        # nn.init.xavier_uniform_(self.linear2.weight)
        # nn.init.xavier_uniform_(self.linear2.bias)
        # nn.init.xavier_uniform_(self.output.weight)
        # nn.init.xavier_uniform_(self.output.bias)

    def forward(self, x):
        # 输入层计算
        x = self.bn_input(x)  # 批归一化
        # 隐藏层 1 计算
        x = self.linear1(x)  # 加权求和
        x = self.bn1(x)  # 批归一化
        x = torch.relu(x)  # 非线性变换
        x = self.dropout1(x)  # 随机失活
        # 隐藏层 2 计算
        x = self.linear2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        # 输出层计算
        output = self.output(x)

        return output

# todo 3：模型训练
def train(model, train_data):
    print('=' * 30, '模型训练', '=' * 30)
    # 1. 数据加载
    """pin_memory=Ture：可锁页内存，加速 CPU → GPU 的数据复制。"""
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, pin_memory=True)
    # 2. 定义损失函数
    # 交叉熵公式：-ln(sofemax(x)) → -P_true * ln(P_pred)（其中 P_true = 1, P_pred = softmax(x)）
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # 3. 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.9), eps=1e-9)
    ## 查看每层神经元的初始参数
    # for name, param in model.named_parameters():
    #     print(f'name: {name}, param.shape: {param.shape}\n')
    # 4. 模型训练
    epochs, loss_list = 50, []
    for epoch in range(1, epochs + 1):
        total_loss, batch_num = 0.0, 0
        start = time.time()
        for x_train, y_train in train_loader:
            model.train()  # 训练模式
            """
            · 加速模型训练的策略：用 DataLoader 在 CPU 并行加载和预处理数据，在将处理好的批次传输到 GPU 完成模型计算。
            · 不建议将整个数据集直接放到 GPU，因为会消耗大量显存，导致 GPU 长时间空闲等待 CPU 数据。
            · 使用 non_blocking=True 可实现异步传输，加速数据处理。
            """
            x_train = x_train.to(device, non_blocking=True)
            y_train = y_train.to(device, non_blocking=True)

            y_pred = model(x_train)  # 模型预测
            loss = criterion(y_pred, y_train)  # 定义损失函数
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新

            total_loss = loss.item()  # 把本轮的每批次的平均损失累计起来
            batch_num += 1  # 把本轮的批次数累计起来
        end = time.time()

        loss_list.append(total_loss / batch_num)
        if epoch % 10 == 0:
            print(f'Epoch[{epoch:02d}/{epochs}], Loss: {total_loss/batch_num:.4f}, Time: {end-start:.4f}s')

    print(f'\n------- 第 {epochs} 轮的最后一个批次的训练结果 -------\n')
    print(f'y_pred（每个类对应的得分）：{y_pred.detach()}\n')
    y_pred = torch.argmax(y_pred, dim=1)
    print(f'y_pred（预测出的类）：{y_pred.detach()}\n')
    print(f'y_train（正确的分类）：{y_train.detach()}\n')
    correct_sum = (y_pred == y_train).sum()
    train_sum = len(y_train)
    correct_rate = correct_sum / train_sum
    print(f'correct_sum: {correct_sum}, train_sum: {train_sum}, correct_rate: {correct_rate}\n')
    print('-'*50)
    print('-' * 68, '\n')

    # 5. 训练结束，保存模型参数
    # print(f'\n模型参数信息：\n{model.state_dict()}\n')
    torch.save(model.state_dict(), 'data/phone_price/phone_eval.pth')
    # 6. 返回每 epoch 训练的损失值 → 用于绘图
    # print(f'loss_list: {[round(x, 4) for x in loss_list]}')
    return loss_list

# todo 4：模型验证
def validate(model, val_data):
    print('='*30,'模型验证','='*30)
    # 1. 数据加载
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, pin_memory=True)
    # 2. 加载模型参数
    model.load_state_dict(torch.load('data/phone_price/phone_val.pth'))
    # 3. 预测每批次的验证集
    for x_val, y_val in val_loader:
        model.eval()  # 切换到测试模式（关闭 dropout。BatchNorm 的参数采用训练集参数的移动平均）
        x_val = x_val.to(device, non_blocking=True)
        y_val = y_val.to(device, non_blocking=True)

        y_pred = model(x_val)
        # print(f'y_pred（每个类别的得分）：{y_pred.detach()}\n')

        # argmax() → 最大值索引（按此方法的话，多分类数据的 target 必须为 0, 1, 2,... → 可见在实际计算中无大小关系）
        # (dim=1：逐行比较。dim=0：逐列比较)
        y_pred = torch.argmax(y_pred, dim=1)
        # 无法将 cuda:0 设备类型张量转为 numpy。需使用 tensor.cpu() 将张量复制到主机内存
        # y_pred = y_pred.detach().numpy()  # 张量（tensor）转 numpy()
        y_pred = y_pred.detach()
        # print(f'y_pred（预测出的类）: {y_pred}\n')
        # print(f'y_true（正确的分类）: {y_val.detach()}\n')

        correct_sum = (y_val == y_pred).sum()
        val_sum = len(y_val)
        correct_rate = correct_sum / val_sum
        print(f'correct_sum: {correct_sum}, val_sum: {val_sum}, correct_rate: {correct_rate}')
        # print(f'y_pred == y_val 的结果展示: \n{y_pred == y_val}\n')

        print('-' * 68, '\n')

# todo 5：模型测试
def evaluate(model, test_data):
    print('=' * 30, '模型测试', '=' * 30)
    # 1. 数据加载
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, pin_memory=True)
    # 2. 加载模型参数
    model.load_state_dict(torch.load('data/phone_price/phone_eval.pth'))
    # 3. 预测每批次的测试集
    for x_test, y_test in test_loader:
        model.eval()  # 切换到测试模式（关闭 dropout。BatchNorm 的参数采用训练集参数的移动平均）
        x_test = x_test.to(device, non_blocking=True)
        y_test = y_test.to(device, non_blocking=True)

        y_pred = model(x_test)
        # print(f'y_pred（每个类的得分）: {y_pred.detach().numpy()}\n')

        # argmax() → 最大值索引（按此方法的话，多分类数据的 target 必须为 0, 1, 2,... → 可见在实际计算中无大小关系）
        # (dim=1：逐行比较。dim=0：逐列比较)
        y_pred = torch.argmax(y_pred, dim=1)
        # print(f'y_pred（预测出的类）: {y_pred.detach()}\n')
        # print(f'y_true（正确的分类）: {y_test.detach()}\n')

        correct_sum = (y_test == y_pred).sum()
        test_sum = len(y_test)
        correct_rate = correct_sum / test_sum
        print(f'correct_sum: {correct_sum}, test_sum: {test_sum}, correct_rate: {correct_rate}')
        # print(f'y_pred == y_test 的结果展示: \n{y_pred==y_test}\n')

        print('-' * 68, '\n')


# todo 5：测试
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. 准备数据集
    train_data, val_data, test_data, input_dim, output_dim = create_dataset(device)
    train_val_data = ConcatDataset([train_data, val_data])
    all_data = ConcatDataset([train_data, val_data, test_data])
    # 2. 构建神经网络模型
    model = Net(input_dim=input_dim, output_dim=output_dim).to(device)
    # 3. 模型训练（train_data训练 → phone_val.pth。 train_val_data训练 → phone_eval.pth）
    loss_list = train(model, train_val_data)
    # 4. 模型验证
    validate(model, val_data)
    # 5. 模型测试
    evaluate(model, test_data)