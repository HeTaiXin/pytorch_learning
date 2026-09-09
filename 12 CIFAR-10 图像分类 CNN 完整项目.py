"""
案例：
    演示 CNN 的综合案例，图像分类。

回顾：深度学习项目的步骤
    1. 准备数据集
        这里我们用计算机视觉模块torchvision自带的 CIFAR10 数据集，包含6W张(32, 32, 3)的图片
        ，5W张训练集，1W张测试集，10个分类。每个分类有6k张图片。
    2. 搭建（卷积）神经网络
    3. 模型训练
    4. 模型测试
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary


# todo 1. 准备数据集
def create_dataset():
    """"""
    """
    root：存放数据的路径。
    train：True/False → 选择训练集/测试集。
    download：如果数据不存在则下载。
    transform：数据预处理。
        原始的图像文件（如 JPEG、PNG）通常以 PIL Image 或 Numpy数组的形式存在，取值范围为 [0, 255]，形状为 (H, W, C)。
        而 PyTorch 模型期望的输入是：
            数据类型：torch.Tensor
            数值范围：通常归一化到 [0, 1] 或 [-1, 1] → 目的是加速收敛、避免梯度饱和/消失......
            形状顺序：(C, H, W)
        transform.ToTensor()数据预处理函数：将图像文件转换为 [0., 1.] 的torch.FloatTensor，形状顺序调整为(C, H, W)。
    """
    # 1.1 数据预处理 (重点学习，必要时写一个可调用的函数)
    # 组合函数 Compose()：将多个变换组合成一个流水线，按顺序一次执行。
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),  # 随机裁剪，相当于加了 padding 后随机截取 32×32
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor()  # ToTensor处理
        , transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # 归一化处理
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
        , transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    # 1.2 获取训练集和测试集
    train_data = datasets.CIFAR10(root='./data/CIFAR10', train=True, transform=transform_train, download=True)
    test_data = datasets.CIFAR10(root='./data/CIFAR10', train=False, transform=transform_test, download=True)
    # 1.3 通过 classes 属性获取分类的类别数量 → 即输出层神经元个数（cnn 输入层神经元个数为 3 （RGB三通道））
    # train_data.classes：['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_classes = train_data.classes
    nums_classes = len(train_classes)
    # 1.4 返回数据集
    return train_data, test_data, nums_classes


# 批归一化函数
from toolkit import get_batch_norm

bn_params_conv = {'use_bn': False, 'dim': 2, 'bn_eps': 1e-5, 'bn_momentum': 0.1, 'affine': True}
bn_params_linear = {'use_bn': True, 'dim': 1, 'bn_eps': 1e-5, 'bn_momentum': 0.1, 'affine': True}
# Dropout函数
from toolkit import get_dropout

dropout_param = {'use_dropout': False, 'dropout_prob': 0.5, 'dim': 2}


# todo 2. 搭建卷积神经网络
class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        # 层 1 准备：卷积层
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn1 = get_batch_norm.get_batch_norm(32, **bn_params_conv)
        # 层 2 准备：卷积层
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = get_batch_norm.get_batch_norm(64, **bn_params_conv)
        # 层 3 准备：卷积层
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = get_batch_norm.get_batch_norm(128, **bn_params_conv)
        # 定义池化层，作用在每一个卷积层。
        self.pool = nn.MaxPool2d((2, 2), stride=2, padding=0)
        # 层 4 准备：全连接层
        self.linear1 = nn.Linear(128 * 4 * 4, 1024)
        self.bn4 = get_batch_norm.get_batch_norm(1024, **bn_params_linear)
        # 层 5 准备：输出层
        self.output = nn.Linear(1024, output_dim)
        # 定义Dropout层，按需添加。
        self.dropout = get_dropout.get_dropout(dropout_param)

    def forward(self, x):
        # 层 1 计算：卷积层
        x = self.conv1(x)  # 卷积核计算
        x = self.bn1(x)  # 批归一化处理
        x = torch.relu(x)  # 激活处理 → shape：(N, C, H, W)
        x = self.pool(x)  # 池化计算
        x = self.dropout(x)  # 随机失活
        # 层 2 计算：卷积层
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        # 层 3 计算：卷积层
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        # 层 4 计算：全连接层
        # 全连接层只能处理二维数据（卷积层输出的数据形状：(N,C,H,W)），要将数据展开为 (N, C*H*W)
        # x.size(0)：样本数（N）。 -1：自动推断该维度的大小（C*H*W）
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout(x)
        # 层 5 计算：输出层
        output = self.output(x)
        return output


# todo 3. 模型训练
def train(model, train_data):
    # pin_memory=True：可锁页内存，加速 CPU → GPU 的数据复制。
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, pin_memory=True)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    """
    知识点：SGD的 weight_decay 参数
        · 权重衰减（weight_decay）与 L2 正则化类似，但会对模型所有的参数都进行正则化处理。
        · 经验和文献共识：几乎所有经典模型（ResNet、VGG等）的训练配方中，weight_decay 都只施加于权重，
          而 偏置项 和 BatchNorm 的可学习参数通常被设置为 weight_decay=0 （即不做正则化处理）。
        · 对模型 偏置项 和 BatchNorm项 参数进行正则化处理可能造成的后果：
            · 使模型难以学习到合适的偏置值，尤其是当偏置需要远离 0 时；
            · 轻微降低模型容量和最终精度（尤其在精细调参时可见）
        最佳实践是只对权重施加衰减，偏置项 和 归一化层 参数不衰减。可通过自定义 param_groups 实现：
    """
    from toolkit import weight_decay
    param_groups = weight_decay.weight_decay(model, wd=1e-4)
    optimizer = optim.SGD(param_groups, lr=0.01, momentum=0.9)
    # 创建学习率衰减对象
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 85], gamma=0.1)

    epochs, loss_list = 100, []
    for epoch in range(epochs):
        model.train()
        total_loss, batch_nums, correct_epoch = 0, 0, 0
        start = time.time()
        for x_train, y_train in train_loader:
            """
            · 加速模型训练的策略：用 DataLoader 在 CPU 并行加载和预处理数据，在将处理好的批次传输到 GPU 完成模型计算。
            · 不建议将整个数据集直接放到 GPU，因为会消耗大量显存，导致 GPU 长时间空闲等待 CPU 数据。
            · 使用 non_blocking=True 可实现异步传输，加速数据处理。
            """
            x_train = x_train.to(device, non_blocking=True)
            y_train = y_train.to(device, non_blocking=True)

            y_pred = model(x_train)  # 每个类别的得分
            y_pred_class = torch.argmax(y_pred, dim=1)  # 预测出的类
            correct_epoch += (y_train == y_pred_class).sum()

            loss = criterion(y_pred, y_train)
            total_loss += loss.item()
            batch_nums += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end = time.time()
        scheduler.step()

        loss_list.append(total_loss / batch_nums)
        correct_rate_epoch = correct_epoch / len(train_data)  # 每轮训练后的正确率
        print(f'epoch: {epoch:03d}, loss: {total_loss / batch_nums:.3f}'
              f', time: {end - start:.4f}s, correct_rate: {correct_rate_epoch:.4f}')

    # 训练结束，保持模型参数
    torch.save(model.state_dict(), './data/image_classification_params.pth')
    return loss_list


# todo 4. 模型测试
def evaluate(model, test_data):
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, pin_memory=True)
    model.load_state_dict(torch.load('./data/image_classification_params.pth'))
    start = time.time()
    correct_total = 0
    for x_test, y_test in test_loader:
        model.eval()
        x_test = x_test.to(device, non_blocking=True)
        y_test = y_test.to(device, non_blocking=True)

        y_pred = model(x_test)
        y_pred = torch.argmax(y_pred, dim=1)

        y_pred = y_pred.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()
        # print(f'y_pred（预测出的类）: {y_pred}')
        # print(f'y_true（正确的分类）: {y_test}')

        correct_sum = (y_pred == y_test).sum()
        test_sum = len(y_test)
        correct_rate = correct_sum / test_sum
        print(f'correct_sum: {correct_sum}, test_sum: {test_sum}, correct_rate: {correct_rate}')
        correct_total += correct_sum
    end = time.time()
    print(f'\ncorrect_total: {correct_total}, test_total: {len(test_data)}, '
          f'correct_rate_total: {correct_total / len(test_data)}, time: {end - start:.4f}s')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # todo 1. 准备数据集
    train_data, test_data, nums_classes = create_dataset()
    """
    知识点:
        在 torchvision.datasets.CIFAR10 数据集中，train_dataset.data 是一个Numpy数组，它存储了数据集中所有原始图像的像素值，
        不受任何 transform 的影响。（transform 只会在通过 train_dataset[idx] 获取具体样本是被动态应用，而不会修改 .data 属性。）
    """
    # # 查看 .data 的属性
    # print(type(train_data.data))  # <class 'numpy.ndarray'>
    # print(train_data.data.shape)  # (50000, 32, 32, 3)
    # print(train_data.data.dtype)  # uint8
    # print(train_data.data.min(), train_data.data.max(), '\n')  # 0 255
    #
    # # 对比通过索引获取的图像（已经过 transform）
    # img_tensor, label = train_data[0]
    # print(type(img_tensor))  # <class 'torch.Tensor'>
    # print(img_tensor.shape)  # torch.Size([3, 32, 32])
    # print(img_tensor.dtype)  # torch.float32
    # print(img_tensor.min(), img_tensor.max(), '\n')  # tensor(0.) tensor(1.)

    # todo 2. 搭建卷积神经网络
    model = CNN(output_dim=nums_classes).to(device)

    # todo 3. 模型训练
    loss_list = train(model, train_data)

    # todo 4. 模型测试
    """
    image_classification_params01：bn_conv=True, batch=100, optim=SDG(lr=0.01, momentum=0.9)
    image_classification_params02：bn_conv=False, batch=100, optim=SDG(lr=0.01, momentum=0.9)
    image_classification_params03：bn_conv=True, batch=100, optim=lr_scheduler(lr=0.01*0.1, [25,50,85])
    """
    # evaluate(model, test_data)
