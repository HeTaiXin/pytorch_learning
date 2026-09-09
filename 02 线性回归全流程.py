# ===================== 导入必要的库 =====================
import torch
from torch.utils.data import TensorDataset, DataLoader  # 用于构建数据管道
from torch import nn       # 神经网络模块（包含线性层和损失函数）
from torch import optim    # 优化器模块（用于更新参数）
from sklearn.datasets import make_regression  # 用于生成线性回归数据集
import matplotlib.pyplot as plt  # 用于绘图可视化

# 设置 matplotlib 正常显示中文标签和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 1. 创建数据集 =====================
def create_dataset():
    """
    使用 sklearn 的 make_regression 生成带噪声的线性数据。
    数据关系：y = x * coef + bias + noise
    """
    x, y, coef = make_regression(
        n_samples=100,     # 生成 100 个样本
        n_features=1,      # 每个样本只有 1 个特征（一维线性回归）
        coef=True,         # 返回真实的回归系数（权重 w）
        bias=14.5,         # 真实的偏置（截距 b）
        noise=10,          # 添加标准差为 10 的噪声，模拟真实数据
        random_state=3     # 固定随机种子，保证每次运行结果一致
    )

    # 将 numpy 数组转换为 PyTorch 张量，并指定数据类型为 float32
    x = torch.tensor(x, dtype=torch.float32)
    # ✅ 提前将 y 的形状从 (100,) 转为 (100, 1) 的二维列向量
    #    这样在训练时就不需要每个 batch 都 reshape 了
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    coef = torch.tensor(coef, dtype=torch.float32)

    return x, y, coef


# ===================== 2. 训练模型 =====================
def train(x, y):
    """
    使用 mini-batch 梯度下降训练线性回归模型。
    返回训练好的模型和每轮的平均损失记录。
    """
    # -------- 超参数设置（集中管理，方便统一调整）--------
    batch_size = 16   # 每批样本数量
    lr = 0.01         # 学习率
    epochs = 100      # 训练轮数（遍历整个数据集的次数）

    # -------- 构建数据管道 --------
    # TensorDataset：将 x 和 y 打包成一个数据集，支持按索引取样本
    dataset = TensorDataset(x, y)
    # DataLoader：将数据集按 batch_size 分批，shuffle=True 表示每个 epoch 打乱数据顺序
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -------- 定义模型 --------
    # nn.Linear(1, 1) 定义了一个线性层：y = x * w + b
    # 其中 in_features=1（输入特征数），out_features=1（输出特征数）
    model = nn.Linear(1, 1)
    # 手动初始化权重和偏置为 3（默认是随机初始化）
    nn.init.constant_(model.weight, val=3)
    nn.init.constant_(model.bias, val=3)

    # -------- 定义优化器和损失函数 --------
    # optim.SGD：随机梯度下降优化器，传入模型所有可学习参数（w 和 b）
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # nn.MSELoss：均方误差损失函数，reduction='mean' 表示对 batch 内所有样本求平均
    criterion = nn.MSELoss(reduction='mean')

    # 用于记录每一轮的平均损失
    loss_list = []

    # -------- 训练循环 --------
    for epoch in range(epochs):
        total_loss = 0.0      # 累加当前轮所有 batch 的损失
        total_samples = 0     # 累加当前轮处理的 batch 数量

        # 遍历 DataLoader，每次取出一个 batch 的数据
        for train_x, train_y in dataloader:
            # ✅ 标准步骤1：清空上一轮积累的梯度（防止梯度累加导致更新方向错误）
            optimizer.zero_grad()

            # ✅ 标准步骤2：前向传播，计算模型预测值
            # model(train_x) 等价于 train_x @ model.weight.T + model.bias
            y_pred = model(train_x)

            # ✅ 标准步骤3：计算当前 batch 的损失值
            loss = criterion(y_pred, train_y)

            # ✅ 标准步骤4：反向传播，自动计算所有可学习参数的梯度
            # loss 是标量，直接调用 backward() 即可
            loss.backward()

            # ✅ 标准步骤5：优化器根据梯度更新参数
            # 更新规则：w = w - lr * w.grad
            optimizer.step()

            # 累加损失和样本数，用于计算本轮平均损失
            total_loss += loss.item()   # .item() 将单元素张量转为 Python 数值
            total_samples += 1

        # 计算并记录本轮平均损失
        avg_loss = total_loss / total_samples
        loss_list.append(avg_loss)

    # 训练结束，返回模型对象和损失记录
    return model, loss_list


# ===================== 3. 主程序 =====================
if __name__ == '__main__':
    # 生成数据
    x, y, coef = create_dataset()

    # 训练模型
    model, loss_list = train(x, y)

    # 提取训练后的参数（.item() 将单元素张量转为 Python 数值）
    final_w = model.weight.item()
    final_b = model.bias.item()

    # 打印真实参数和训练得到的参数，进行对比
    print(f'正确的权重 w = {coef.item():.4f}, 正确的偏置 b = 14.5')
    print(f'训练后的权重 w = {final_w:.4f}, 训练后的偏置 b = {final_b:.4f}')

    # ✅ 推理阶段使用 torch.no_grad() 上下文管理器
    #    告诉 PyTorch 不需要构建计算图，节省内存/显存
    with torch.no_grad():
        y_pred_all = model(x).numpy()   # 对整个数据集进行预测，并转为 numpy 数组

    # ===================== 4. 可视化结果 =====================
    fig = plt.figure(figsize=(8, 7), dpi=100, facecolor='white')

    # 子图1：原始数据散点图 + 拟合直线
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(x.numpy(), y.numpy(), marker='o', s=20, c='deepskyblue', label='真实值')
    ax1.plot(x.numpy(), y_pred_all, '-', c='red', linewidth=1, label='预测直线')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.legend()

    # 子图2：损失下降曲线
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(range(1, len(loss_list) + 1), loss_list, '-', linewidth=1, c='green', label='Loss')
    ax2.set_xlabel('训练轮数', fontsize=12)
    ax2.set_ylabel('损失值', fontsize=12)
    ax2.legend()

    # 子图3：预测值 vs 真实值散点图（越靠近对角线说明拟合越好）
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(y_pred_all, y.numpy(), s=20, c='red')
    y_min, y_max = y.min().item(), y.max().item()
    ax3.plot([y_min, y_max], [y_min, y_max], '--', c='green', linewidth=1, label='理想线')
    ax3.set_xlabel('predicted value', fontsize=12)
    ax3.set_ylabel('true value', fontsize=12)
    ax3.legend()

    plt.show()
