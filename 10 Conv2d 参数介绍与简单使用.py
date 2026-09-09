import torch
from torch import nn
import matplotlib.pyplot as plt

conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
"""
参数说明：
    in_channels：输入通道数，RGB 图片一般是3。
    out_channels：输出通道数，也可以理解为卷积核 kernel 的数量。
    kernel_size：卷积核的高和宽设置，一般为3，5，7，...。如kernel_size=3 → 3×3的卷积核。kernel_size=(3,4) → 3×4的卷积核。
    stride：卷积核移动的步长
        整数stride：表示在所有维度上使用相同的步长。如 stride=2 表示在水平和垂直方向上每次移动2个像素。
        元组stride：允许在不同维度上设置不同的步长。如 stride=(2, 1) 表示在水平方向上步长为2，在垂直方向上步长为1。
    padding：在四周加入 padding 的数量，默认为0。
        padding = 0：不进行填充。
        padding = 1：在每个维度上填充 1 个像素。
        padding = 输入形状大小 - 输出形状大小：表示输出尺寸与输入相同。
        padding = 'same'：从 PyTorch 1.9+ 开始支持，让输出特征图的尺寸与输入保持一致。PyTorch 会自动计算需要的填充量。
                          padding必须等于1，不支持跨行，因为不然计算padding时可能出现小数。
        padding = kernel_size - 1：Full Padding 完全填充。
"""

def test01():
    # 1. 读取图像
    img = plt.imread('./data/images/a.jpg')
    print(f'img shape: {img.shape}')
    # print(f'img: {img}')
    print(f'img type: {type(img)}')

    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # 2. 定义卷积层
    myconv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
    print(f'myconv2d: {myconv2d}')

if __name__ == '__main__':
    test01()
