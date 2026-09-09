import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 通用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


# 1. 定义函数，绘制：全黑，全白图
def dm01():
    # 1. 定义图片：像素点越接近 0 越黑，越接近 255 越白。
    # HWC：H → 高度（纵轴每列有h个像素点）， W → 宽度（横轴每行有w个像素点）， C → 通道（每个像素点的RGB值（c=3的情况））。
    img1 = np.array(
        [[[89., 100., 200.], [150., 250., 5.], [100., 1., 200.], [150., 250., 5.], [0., 1., 200.]]
            , [[255., 0., 0.], [0., 255., 0.], [0., 0., 255.], [0., 200., 100.], [100., 0., 200.]]
            , [[200., 0., 100.], [0., 200., 100.], [100., 0., 200.], [205., 100., 207.], [95., 203., 180.]]
            , [[120., 120., 120.], [205., 100., 207.], [95., 203., 180.], [150., 250., 5.], [0., 0., 0.]]]
        , dtype=int)
    print(f'img1.shape：{img1.shape}')
    print(f'img1：{img1}')
    print(img1[0, 1, 2])

    # 2. 绘制图片
    plt.imshow(img1)
    plt.show()


def dm02():
    img1 = plt.imread('./data/images/a.jpg')
    img2 = torch.tensor(img1, dtype=torch.float)
    img2 = img2.permute(2, 0, 1)  # 把图像的形状从 HWC → CHW
    print(f'img2.shape：{img2.shape}')
    print(f'img2：{img2}\n')

    """ 
    PyTorch中，二维卷积层 nn.Conv2d 的输入必须是四维张量。 → （N, C, H, W)
        N：Batch size，批次数，一次喂给模型的图像数量。
        C：Channels，通道数，图像或特征图的深度。
        H：Height，高度，即行数。
        W：Width，宽度，即列数。
    """
    # 这里只有 1 张图，增加一个表示图像张数的维度：(C, H, W) → (1, C, H, W)：1张 3 通道的 H×W像素的图像。
    img3 = img2.unsqueeze(dim=0)  # unsqueeze()函数作用是在指定的位置插入一个大小为 1 的新维度，从而改变张量的形状。
    print(f'img3: {img3}, shape: {img3.shape}\n')
    # 创建卷积层对象。
    conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
    # 查看卷积核 → 即卷积层需训练的参数（可自己设置初始化参数）
    print(f'filter (convolution kernel weight): {conv.weight}\n'
          f'filter (convolution kernel bias): {conv.bias}\n')
    # 卷积计算 卷积核滑动移动提取特征图片
    # (N, C, H, W) → (1, 4, 4, 5)
    conv_img = conv(img3)
    print(f'conv_img: {conv_img}, shape: {conv_img.shape}\n')
    # 卷积计算获得的特征矩阵 (N, C, H, W) → (1, 4, 4, 5)
    # 实则是 4（C） 个 4×5（H×W） 的矩阵张量。
    # 查看该 4 个特征图
    feat_imgs = conv_img[0].detach().numpy()  # (C, H, W) → (4, 4, 5)
    print(f'feat_imgs: {feat_imgs}\n')
    feat_img1 = feat_imgs[0]
    feat_img2 = feat_imgs[1]
    feat_img3 = feat_imgs[2]
    feat_img4 = feat_imgs[3]

    # 对卷积层进行激活处理 → 插入激活层
    relu_img = torch.relu(conv_img)
    print(f'relu_img: {relu_img}, shape: {relu_img.shape}\n')  # shape: torch.Size([1, 4, 737, 920])

    relu_imgs = relu_img[0].detach().numpy()
    relu_imgs1 = relu_imgs[0]
    relu_imgs2 = relu_imgs[1]
    relu_imgs3 = relu_imgs[2]
    relu_imgs4 = relu_imgs[3]

    # 对卷积层进行池化
    pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
    pool_operate = pooling(relu_img)
    print(f'pool_img: {pool_operate}, shape: {pool_operate.shape}\n')

    pool_imgs = pool_operate[0].detach().numpy()
    pool_img1 = pool_imgs[0]
    pool_img2 = pool_imgs[1]
    pool_img3 = pool_imgs[2]
    pool_img4 = pool_imgs[3]

    """
    可视化特征图：
        当 imshow 接受一个二维数组时，它实则是将每个数值映射到一个色带上的一种颜色。默认为 'viridis' 色带（翠绿色）。
        色带范围默认为特征矩阵的最小值到最大值。
        可通过 cmap, vmin, vmax 参数来设置色带以及色带范围
    """
    plt.imshow(feat_img1, cmap='gray')
    plt.show()
    plt.imshow(relu_imgs1, cmap='gray')
    plt.show()
    plt.imshow(pool_img1, cmap='gray')
    plt.show()

    plt.imshow(feat_img2, cmap='hot')
    plt.show()
    plt.imshow(relu_imgs2, cmap='hot')
    plt.show()
    plt.imshow(pool_img2, cmap='hot')
    plt.show()

    plt.imshow(feat_img3, cmap='PuBu')
    plt.show()
    plt.imshow(relu_imgs3, cmap='PuBu')
    plt.show()
    plt.imshow(pool_img3, cmap='PuBu')
    plt.show()

    plt.imshow(feat_img4)
    plt.show()
    plt.imshow(relu_imgs4)
    plt.show()
    plt.imshow(pool_img4)
    plt.show()


if __name__ == '__main__':
    # dm01()
    dm02()

