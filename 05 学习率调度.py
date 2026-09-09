"""
案例：
    演示 PyTorch 中的学习率衰减（Learning Rate Decay）策略。

背景知识：
    在深度学习中，学习率（Learning Rate）控制着参数更新的步长。固定学习率往往存在两难：
        - 学习率过大：训练初期可能收敛快，但后期容易在最优解附近震荡甚至发散。
        - 学习率过小：训练初期收敛极慢，且容易陷入局部最优。
    因此，理想的策略是“先大后小”：初期用较大学习率快速接近最优区域，后期用较小学习率精细调整。

学习率调整的两大类别：
    1. 自适应优化器（如 AdaGrad, RMSprop, Adam）：
       它们**内部**会根据历史梯度信息自动调整每个参数的有效学习率，无需外部干预。
    2. 学习率调度器（Scheduler，如本例所示）：
       它们基于**预设的规则**（如等间隔、指定间隔、指数衰减）从外部手动调整优化器的学习率。
       注意：两者可以结合使用（例如 Adam + StepLR），但本例为了清晰演示，使用 SGD + Momentum。

PyTorch 学习率调整核心接口：
    - `torch.optim.lr_scheduler` 模块提供了各种调度器。
    - 使用模式：先创建优化器（Optimizer），再将优化器传给调度器（Scheduler）。
    - 关键方法：`scheduler.step()` 必须在每个 epoch（或 iteration）结束后调用，以更新学习率。
    - 查看当前学习率：`scheduler.get_last_lr()` 返回一个列表（因为优化器可能管理多个参数组）。

本例演示的三种衰减方法：
    1. 等间隔衰减（StepLR）：每隔固定步数（step_size）将学习率乘以衰减系数（gamma）。
    2. 指定间隔衰减（MultiStepLR）：在指定的里程碑（milestones）轮次将学习率乘以 gamma。
    3. 指数衰减（ExponentialLR）：每个 epoch 都将学习率乘以 gamma，即 lr = lr * (gamma ** epoch)。
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置 matplotlib 正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train_with_scheduler(scheduler_type, w, optimizer, scheduler, epochs, iterations):
    """
    公共训练函数：使用给定的调度器训练模型，并记录学习率和损失值。
    返回：epoch_list, lr_list, loss_list
    """
    # 构造极简数据集：目标 y_true=0，特征 x=1，初始权重 w=1
    # 损失函数为 loss = (w*x - y_true)^2 = w^2，最优解显然是 w=0
    y_true = torch.tensor([0.0], dtype=torch.float32)
    x = torch.tensor([1.0], dtype=torch.float32)

    epoch_list, lr_list, loss_list = [], [], []

    for epoch in range(epochs):
        # 记录当前 epoch 的学习率（get_last_lr() 返回列表，取第0个元素）
        current_lr = scheduler.get_last_lr()[0]
        lr_list.append(current_lr)
        epoch_list.append(epoch + 1)

        # 模拟一个 epoch 内的多次迭代（batch）
        for _ in range(iterations):
            y_pred = w * x
            loss = (y_pred - y_true) ** 2

            optimizer.zero_grad()  # 清空历史梯度
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 优化器根据当前学习率更新参数

        # 记录当前 epoch 结束后的损失值（detach 断开计算图）
        loss_list.append(loss.detach().item())

        # 关键：调用调度器的 step() 方法，更新学习率（为下一个 epoch 做准备）
        scheduler.step()

    return epoch_list, lr_list, loss_list


def plot_results(epoch_list, lr_list, loss_list, title):
    """绘制学习率和损失曲线"""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, lr_list, '-', c='blue', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(f'{title} - 学习率变化', fontsize=13)

    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, loss_list, '-', c='red', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{title} - 损失变化', fontsize=13)

    plt.tight_layout()
    plt.show()


# 1. 演示：等间隔学习率衰减（StepLR）
def dm01_step_lr():
    lr, epochs, iterations = 0.1, 15, 1
    w = torch.tensor([10.0], requires_grad=True, dtype=torch.float32)
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)
    # 等间隔衰减：每 50 个 epoch，学习率乘以 0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    epoch_list, lr_list, loss_list = train_with_scheduler(
        'StepLR', w, optimizer, scheduler, epochs, iterations
    )
    plot_results(epoch_list, lr_list, loss_list, '等间隔衰减 (StepLR)')


# 2. 演示：指定间隔学习率衰减（MultiStepLR）
def dm02_multistep_lr():
    lr, epochs, iterations = 0.1, 15, 1
    w = torch.tensor([10.0], requires_grad=True, dtype=torch.float32)
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)
    # 指定间隔衰减：在 epoch 50, 125, 160 时，学习率乘以 0.5
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[4, 10], gamma=0.5
    )

    epoch_list, lr_list, loss_list = train_with_scheduler(
        'MultiStepLR', w, optimizer, scheduler, epochs, iterations
    )
    plot_results(epoch_list, lr_list, loss_list, '指定间隔衰减 (MultiStepLR)')


# 3. 演示：指数学习率衰减（ExponentialLR）
def dm03_exponential_lr():
    lr, epochs, iterations = 0.1, 15, 1
    w = torch.tensor([10.0], requires_grad=True, dtype=torch.float32)
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)
    # 指数衰减：每个 epoch 学习率乘以 0.95，即 lr = initial_lr * (0.95 ** epoch)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    epoch_list, lr_list, loss_list = train_with_scheduler(
        'ExponentialLR', w, optimizer, scheduler, epochs, iterations
    )
    plot_results(epoch_list, lr_list, loss_list, '指数衰减 (ExponentialLR)')


if __name__ == '__main__':
    dm01_step_lr()
    dm02_multistep_lr()
    dm03_exponential_lr()

