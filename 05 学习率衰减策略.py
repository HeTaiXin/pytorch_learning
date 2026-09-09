"""
案例：
    演示学习率衰减策略。

学习率衰减策略介绍：
    目的：
        较之于AdaGrad, RMSprop, Adam方式，我们可以通过 等间隔，指定间隔，指数等方式来手动控制学习率的调整。

    为什么要调整学习率：
        学习率控制着模型参数更新的步长。在模型训练中，固定学习率存在明显局限性：
            ·初始学习率过大：可能导致参数更新幅度过大，模型损失震荡不收敛，甚至出现梯度爆炸。
            ·初始学习率过小：参数更新缓慢，训练周期大幅度延长，且易陷入局部最优解，无法达到理想性能。
        理想的学习率策略应满足“先快后慢”：训练初期用较大学习率让模型快速逼近最优解区域，后期用较小学习率精细调整参数，
        确保稳定收敛到全局最优。

    Pytorch 学习率调整核心接口：
        optim.lr_scheduler模块提供统一的学习率调整接口，所有调度器配合优化器（如 Adam、SGD）使用，且需在每个epoch（或
        迭代）后调用scheduler.step()更新学习率。

    学习率衰减方法分类：
        · 等间隔学习率衰减
        · 指定间隔学习率衰减
        · 指数学习率衰减

    等间隔学习率衰减：
        step_size：间隔的轮数（epoch），即多少轮调整一次学习率
        gamma：学习率衰减系数，即：lr新 = lr旧 * gamma
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 1. 定义函数，演示：等间隔学习率衰减
def dm01():
    # 1. 定义变量，记录初始的 学习率，训练的轮数，每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10
    # 2. 创建数据集：y_true, x, w
    y_true = torch.tensor([0.0], dtype=torch.float32)
    x = torch.tensor([1.0], dtype=torch.float32)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 3. 创建优化器对象，动量法 → 加速模型的收敛，减少震荡
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)
    # 4. 创建学习率衰减对象。
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # 5. 循坏遍历训练轮数，进行具体的训练。
    epoch_list, lr_list, loss_list = [], [], []  # 记录训练轮数（epoch_list）、每轮训练的学习率（lr_list）、每轮训练后的损失值（loss_list）
    for epoch in range(epochs):
        epoch_list.append(epoch + 1)
        lr_list.append(scheduler.get_last_lr())
        #         print(f'第 {epoch+1} 轮: scheduler.get_last_lr() = {scheduler.get_last_lr()}')
        # 循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            y_pred = w * x
            loss = (y_pred - y_true) ** 2
            optimizer.zero_grad()  # 梯度清零
            loss.sum().backward()  # 反向传播 → 损失函数自动微分
            optimizer.step()  # 参数更新
        loss_list.append(loss.detach())
        scheduler.step()  # 学习率更新（等间隔学习率衰减更新 → 每50轮（step_size=50） 更新一次学习率（lr * gamma））
    # 6. 打印结果
    #     print(f'lr_list: {lr_list} \n')
    #     print(f'loss_list: {loss_list}')
    # 7. 可视化
    plt.figure(figsize=(6, 4))
    plt.plot(epoch_list, lr_list, '-', c='blue', linewidth=1.5, label='等间隔学习率衰减')
    plt.xlabel('epoch', fontsize=15, weight='normal')
    plt.ylabel('Learn Rate', fontsize=15, weight='normal')
    plt.legend()
    plt.show()


#     plt.figure(figsize=(6,4))
#     plt.plot(epoch_list, loss_list, '-', c='red', linewidth=1.5, label='等间隔学习率衰减')
#     plt.xlabel('epoch', fontsize=15, weight='normal')
#     plt.ylabel('Loss value', fontsize=15, weight='normal')
#     plt.legend()
#     plt.show()

# 2. 定义函数，演示：指定间隔学习率衰减
def dm02():
    # 1. 定义变量，记录初始的 学习率，训练的轮数，每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10
    # 2. 创建数据集：y_true, x, w
    y_true = torch.tensor([0.0], dtype=torch.float32)
    x = torch.tensor([1.0], dtype=torch.float32)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 3. 创建优化器对象，动量法 → 加速模型的收敛，减少震荡
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)
    # 4. 创建学习率衰减对象。
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 125, 160], gamma=0.5)
    # 5. 循坏遍历训练轮数，进行具体的训练。
    epoch_list, lr_list, loss_list = [], [], []  # 记录训练轮数（epoch_list）、每轮训练的学习率（lr_list）、每轮训练后的损失值（loss_list）
    for epoch in range(epochs):
        epoch_list.append(epoch + 1)
        lr_list.append(scheduler.get_last_lr())
        #         print(f'第 {epoch+1} 轮: scheduler.get_last_lr() = {scheduler.get_last_lr()}')
        # 循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            y_pred = w * x
            loss = (y_pred - y_true) ** 2
            optimizer.zero_grad()  # 梯度清零
            loss.sum().backward()  # 反向传播 → 损失函数自动微分
            optimizer.step()  # 参数更新
        loss_list.append(loss.detach())
        scheduler.step()  # 学习率更新（指定间隔学习率衰减更新 → 第50、125、160轮（milestones=[50, 125, 160]） 更新一次学习率）
    # 6. 打印结果
    #     print(f'lr_list: {lr_list} \n')
    #     print(f'loss_list: {loss_list}')
    # 7. 可视化
    plt.figure(figsize=(6, 4))
    plt.plot(epoch_list, lr_list, '-', c='blue', linewidth=1.5, label='指定间隔学习率衰减')
    plt.xlabel('epoch', fontsize=15, weight='normal')
    plt.ylabel('Learn Rate', fontsize=15, weight='normal')
    plt.legend()
    plt.show()


#     plt.figure(figsize=(6,4))
#     plt.plot(epoch_list, loss_list, '-', c='red', linewidth=1.5, label='指定间隔学习率衰减')
#     plt.xlabel('epoch', fontsize=15, weight='normal')
#     plt.ylabel('Loss value', fontsize=15, weight='normal')
#     plt.legend()
#     plt.show()

# 3. 定义函数，演示：指数学习率衰减
def dm03():
    # 1. 定义变量，记录初始的 学习率，训练的轮数，每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10
    # 2. 创建数据集：y_true, x, w
    y_true = torch.tensor([0.0], dtype=torch.float32)
    x = torch.tensor([1.0], dtype=torch.float32)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 3. 创建优化器对象，动量法 → 加速模型的收敛，减少震荡
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)
    # 4. 创建学习率衰减对象。
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # 5. 循坏遍历训练轮数，进行具体的训练。
    epoch_list, lr_list, loss_list = [], [], []  # 记录训练轮数（epoch_list）、每轮训练的学习率（lr_list）、每轮训练后的损失值（loss_list）
    for epoch in range(epochs):
        epoch_list.append(epoch + 1)
        lr_list.append(scheduler.get_last_lr())
        #         print(f'第 {epoch+1} 轮: scheduler.get_last_lr() = {scheduler.get_last_lr()}')
        # 循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            y_pred = w * x
            loss = (y_pred - y_true) ** 2
            optimizer.zero_grad()  # 梯度清零
            loss.sum().backward()  # 反向传播 → 损失函数自动微分
            optimizer.step()  # 参数更新
        loss_list.append(loss.detach())
        scheduler.step()  # 学习率更新（指数学习率衰减更新 → 学习率按公式 lr = lr * gamma^epoch 更新学习率）
    # 6. 打印结果
    #     print(f'lr_list: {lr_list} \n')
    #     print(f'loss_list: {loss_list}')
    # 7. 可视化
    plt.figure(figsize=(6, 4))
    plt.plot(epoch_list, lr_list, '-', c='blue', linewidth=1.5, label='指数学习率衰减')
    plt.xlabel('epoch', fontsize=15, weight='normal')
    plt.ylabel('Learn Rate', fontsize=15, weight='normal')
    plt.legend()
    plt.show()


#     plt.figure(figsize=(6,4))
#     plt.plot(epoch_list, loss_list, '-', c='red', linewidth=1.5, label='指数学习率衰减')
#     plt.xlabel('epoch', fontsize=15, weight='normal')
#     plt.ylabel('Loss value', fontsize=15, weight='normal')
#     plt.legend()
#     plt.show()

# 4. 测试
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()