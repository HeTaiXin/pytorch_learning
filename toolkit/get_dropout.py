import torch
import torch.nn as nn

"""
    参数：
        · use_dropout：True → 返回 Dropout 层。False → 返回 Identity
        · dropout_prob：dropout 概率（丢弃神经元的概率）
        · dim：1 对应 Dropout1d，2 对应 Dropout2d， 3 对应 Dropout3d
    
    返回：
        nn.Module: Dropout 层或 Identity 层
"""
def get_dropout(use_dropout=True, dropout_prob=0.5, dim=None):

    if not use_dropout or dropout_prob == 0:
        return nn.Identity()

    if not 0 <= dropout_prob <= 1:
        raise ValueError(f'dropout_prob must be between [0, 1], which is actually {dropout_prob}.')

    dropout_class = {
        1: nn.Dropout1d,
        2: nn.Dropout2d,
        3: nn.Dropout3d
    }.get(dim, nn.Dropout)  # 若 dim 不是 1、2、3，则使用 nn.Dropout

    return dropout_class(p=dropout_prob)
