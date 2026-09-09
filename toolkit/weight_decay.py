""""""
"""
知识点：SGD的 weight_decay 参数
    · 权重衰减（weight_decay）与 L2 正则化类似，但会对模型所有的参数都进行正则化处理。
    · 经验和文献共识：几乎所有经典模型（ResNet、VGG等）的训练配方中，weight_decay 都只施加于权重，
      而 偏置项 和 BatchNorm 的可学习参数通常被设置为 weight_decay=0 （即不做正则化处理）。
    · 对模型 偏置项 和 BatchNorm项 参数进行正则化处理可能造成的后果：
        · 使模型难以学习到合适的偏置值，尤其是当偏置需要远离 0 时；
        · 轻微降低模型容量和最终精度（尤其在精细调参时可见）
    · 最佳实践是只对权重施加衰减，偏置项 和 归一化层 参数不衰减。可通过自定义 param_groups 实现：
"""


def weight_decay(model, wd=1e-4):
    decay_params = []  # 需要 weight_decay 的参数（通常是卷积/全连接的 weight）
    no_decay_params = []  # 不需要 weight_decay 的参数（bias, BatchNorm weight/bias）

    for name, param in model.named_parameters():
        # 跳过那些不需要计算梯度（即冻结）的参数，不将它们加入优化器的参数组中。
        if not param.requires_grad:
            continue  # continue：跳出本次循环。 break：跳出整个循环。

        # 'bias' 和 'bn'/'norm' 中的参数都不衰减
        if 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': wd},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    return param_groups
