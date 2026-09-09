import torch
import torch.nn as nn

"""
    返回一个 BatchNorm 层，如果不使用 BN 则返回 Identity（不做任何操作）

    :param num_features: 特征维度（C 或 H*W，实际为通道数）
    :param use_bn: True → 返回 BN 层。False → 返回 Identity
    :param dim: 1 对应 nn.BatchNorm1d，2 对应 nn.BatchNorm2d，3 对应 nn.BatchNorm3d
    :param bn_eps: 防止除零的小常数
    :param bn_momentum: 动量参数
    :param affine: 控制是否学习缩放 γ 和偏置 β 参数。默认为 True，若为 False，则只做标准化，不学习变换。
"""
def get_batch_norm(num_features, use_bn=True, dim=1, bn_eps=1e-5, bn_momentum=0.1, affine=True):

    if not use_bn:
        return nn.Identity()  # 直接跳过 BN
    """较冗余的表达："""
    # if dim == 1:
    #     return nn.BatchNorm1d(num_features=num_features, eps=bn_eps, momentum=bn_momentum)
    # elif dim == 2:
    #     return nn.BatchNorm2d(num_features=num_features, eps=bn_eps, momentum=bn_momentum)
    # elif dim == 3:
    #     return nn.BatchNorm3d(num_features=num_features, eps=bn_eps, momentum=bn_momentum)
    # else:
    #     raise ValueError('dim must be 1 or 2 or 3')

    """简介表达（推荐）"""
    bn_class = {
        1: nn.BatchNorm1d,
        2: nn.BatchNorm2d,
        3: nn.BatchNorm3d,
    }.get(dim)

    if bn_class is None:
        raise ValueError('dim must be 1 or 2 or 3')

    # 等价于：return nn.BatchNorm_d(num_features, eps=bn_eps, momentum=bn_momentum, affine=affine)
    return bn_class(num_features, eps=bn_eps, momentum=bn_momentum, affine=affine)
