"""
### 批归一化（Batch Normalization, BN）

#### 具体做法
对于神经网络中某一层的输入（通常是该层线性变换后、激活函数之前的数据），在一个小批量（mini-batch）上计算其均值和方差，然后将其标准化为均值为 0、方差为 1 的标准分布。最后，为了恢复模型的表达能力，引入两个可学习的参数：缩放参数 γ 和平移参数 β。

#### 数学表达式
对每个特征维度独立计算：
    μ_B = (1/m) * Σ x_i           # 当前 mini-batch 在该特征上的均值
    σ_B² = (1/m) * Σ (x_i - μ_B)² # 当前 mini-batch 在该特征上的方差（有偏估计）
    x̂_i = (x_i - μ_B) / √(σ_B² + ε)   # 标准化
    y_i = γ * x̂_i + β              # 缩放和平移

参数说明：
    x_i     : 该层对一个样本的原始输出（某一特征维度上的值）
    μ_B, σ_B² : 当前 mini-batch 所有样本在该特征上的均值和方差
    ε       : 小常数（默认 1e-5），防止除零
    x̂_i     : 标准化后的值（近似均值为 0、方差为 1）
    γ, β    : 可学习的参数。γ 控制缩放（初始为 1），β 控制平移（初始为 0）。训练时通过反向传播更新。
    y_i     : BN 层的最终输出，送入下一层（通常是激活函数）

注意：在训练模式下，BN 使用当前 mini-batch 的统计量进行标准化；在评估（推理）模式下，BN 使用训练过程中累积的移动平均统计量（running_mean 和 running_var），以保证推理的稳定性。

#### 为什么使用 BN？
BN 主要解决深层神经网络训练中的“内部协变量偏移”问题，使训练更快、更稳定。

1. **缓解梯度消失/爆炸**
   深层网络中，激活函数（如 Sigmoid）的饱和区会导致梯度消失。BN 将输入拉回激活函数的非饱和区（如 0 附近），梯度保持较大，训练更顺畅。

2. **允许使用更大的学习率**
   BN 限制了每层输入的范围，避免了权重更新对后续层输入的剧烈放大或缩小，因此可以使用更大的学习率加速收敛。

3. **轻微正则化效果**
   BN 的均值和方差基于当前 mini-batch 计算，不同 batch 的统计量有微小差异，相当于引入噪声，轻微抑制过拟合。

4. **降低对参数初始化的敏感度**
   BN 在每一层重新规范化分布，即使初始化不佳，早期训练也能有较健康的输入分布。

#### 直观比喻
流水线中，每个工人（网络层）处理上游传来的零件（特征）。没有 BN 时，上游输出的零件尺寸形状各异，下游难以加工；有 BN 后，每个工人前设置“标准调节器”（BN 层），强制校准零件为标准尺寸，流水线效率提升。

#### 注意事项
- **依赖 batch size**：BN 的效果依赖足够大的 batch（通常 ≥ 32）。batch 太小时，统计量噪声过大，效果变差（此时可考虑 Layer Norm 或 Instance Norm）。
- **RNN 等序列模型**：由于序列长度可变，BN 不常用，常改用 Layer Norm。

#### 总结
BN 通过对每一层输入进行标准化，让网络训练更稳定、收敛更快，并允许使用更大的学习率和更鲁棒的初始化，已成为现代深度学习（尤其是 CNN）的标配技术。
"""

import torch
import torch.nn as nn

# 设置随机种子以便结果可复现
torch.manual_seed(42)

def demo_bn2d():
    """演示二维批归一化（用于图像数据，形状通常为 (N, C, H, W)）"""
    # 1. 创建图像样本数据：batch_size=4, 通道数=3, 高=3, 宽=4
    input_2d = torch.normal(mean=10, std=5, size=(4, 3, 3, 4))
    print("原始 2D 数据 (batch=4, C=3, H=3, W=4):")
    print(f"  shape: {input_2d.shape}")
    # 打印每个通道的均值和标准差（在 N,H,W 维度上求平均）
    print("  各通道均值:   ", input_2d.mean(dim=(0,2,3)).detach().numpy().round(4))
    print("  各通道标准差:", input_2d.std(dim=(0,2,3)).detach().numpy().round(4))
    print()

    # 2. 创建二维批量归一化层
    bn2d = nn.BatchNorm2d(num_features=3, eps=1e-5, momentum=0.1, affine=True)
    bn2d.train()  # 确保处于训练模式（使用当前 batch 的统计量）

    # 3. 对数据进行批归一化处理
    output_2d = bn2d(input_2d)
    print("经过 BatchNorm2d 后的数据:")
    print(f"  shape: {output_2d.shape}")
    print("  各通道均值:   ", output_2d.mean(dim=(0,2,3)).detach().numpy().round(4))
    print("  各通道标准差:", output_2d.std(dim=(0,2,3)).detach().numpy().round(4))
    print("  (注意: 各通道均值应接近 0，标准差应接近 1)")
    print()

def demo_bn1d():
    """演示一维批归一化（用于全连接层输出，形状通常为 (N, D)）"""
    # 1. 创建样本数据：batch_size=8, 特征数=3
    input_1d = torch.normal(mean=5, std=5, size=(8, 3))
    print("原始 1D 数据 (batch=8, features=3):")
    print(f"  shape: {input_1d.shape}")
    print("  各特征均值:   ", input_1d.mean(dim=0).detach().numpy().round(4))
    print("  各特征标准差:", input_1d.std(dim=0).detach().numpy().round(4))
    print()

    # 2. 创建线性层
    linear1 = nn.Linear(in_features=3, out_features=4)
    # 线性变换（加权求和）
    output_1d = linear1(input_1d)
    print("线性层输出 (未经过 BN):")
    print(f"  shape: {output_1d.shape}")
    print("  各特征均值:   ", output_1d.mean(dim=0).detach().numpy().round(4))
    print("  各特征标准差:", output_1d.std(dim=0).detach().numpy().round(4))
    print()

    # 3. 创建一维批归一化层
    bn1d = nn.BatchNorm1d(num_features=4)
    bn1d.train()  # 训练模式

    # 4. 对线性输出进行批归一化
    output_bn1d = bn1d(output_1d)
    print("经过 BatchNorm1d 后的输出:")
    print(f"  shape: {output_bn1d.shape}")
    print("  各特征均值:   ", output_bn1d.mean(dim=0).detach().numpy().round(4))
    print("  各特征标准差:", output_bn1d.std(dim=0).detach().numpy().round(4))
    print("  (注意: 各特征均值应接近 0，标准差应接近 1)")
    print()

if __name__ == '__main__':
    print("="*50)
    print("二维批归一化演示 (适用于图像卷积层)")
    print("="*50)
    demo_bn2d()
    print("="*50)
    print("一维批归一化演示 (适用于全连接层)")
    print("="*50)
    demo_bn1d()
