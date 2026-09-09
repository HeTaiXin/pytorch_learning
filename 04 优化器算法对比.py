"""
案例：
    演示 梯度下降及其优化方法（Momentum, AdaGrad, RMSprop, Adam）

梯度下降相关介绍:
    概述：
        梯度下降是结合 本次损失函数的导数（作为梯度）基于学习率 来更新权重的。
    公式：
        Wt = Wt-1 - 学习率 * Gt
        （其中 Gt 是损失函数对 Wt-1 的梯度）
    存在问题：
        1. 遇到平缓区域，梯度下降（权重更新）可能会变慢。
        2. 可能会遇到 鞍点（梯度为0，导致更新停滞）。
        3. 可能会遇到 局部最小值（非全局最优）。
    解决思路：
        从上述的 学习率 或者 梯度 入手，进行优化，于是有了：
        动量法 Momentum、自适应学习率 AdaGrad、RMSProp，
        以及综合两者优势的自适应矩估计：Adam。

    1. 动量法 Momentum（SGD with Momentum）
        理论公式:
            Wt = Wt-1 - 学习率 * Mt
            Mt = β * Mt-1 + (1 - β) * Gt
            ⇔  Mt = (1-β)Gt + β(1-β)Gt-1 + ... + β^(t-1)(1-β)G1
        解释：
            Mt：   本次的指数移动加权平均梯度（动量）。
            β：    动量衰减系数（通常取0.9）。β越大，历史梯度权重越大，惯性越强。
            Gt：   本次计算的梯度。
        ⚠️ PyTorch 实现差异：
            PyTorch 的 optim.SGD(momentum=β) 内部使用：v = β*v + g，然后 W = W - lr*v。
            这里省略了 (1-β) 系数，直接累加梯度 g。因此代码中的动量项等价于 Mt = β*Mt-1 + Gt。
        总结：
            鞍点 ✔（惯性帮助冲过鞍点）
            缓坡：有惯性驱使，可加快过缓坡；但若当前梯度权重小，随着惯性衰减可能停滞。
            局部最小值 ✔（惯性可能帮助冲出较低的局部极小值）
            悬崖 ❌（惯性加持，可能加剧跳过悬崖，导致更新步长过大）
            最小值附近震荡 ✔（惯性方向与梯度方向相反，能平滑震荡）

    2. 自适应学习率：AdaGrad
        公式:
            Wt = Wt-1 - 学习率/sqrt(Vt + σ) * Gt
            Vt = Vt-1 + Gt²
            ⇔  Vt = Gt² + Gt-1² + ... + G1²
        解释：
            Vt：   累计平方梯度（无衰减的累加）。
            σ：   微小值（防止分母为0，PyTorch默认1e-10）。
        缺点：
            Vt 为梯度的纯平方累加 → 学习率只会单调变小 → 当缓坡足够长时，学习率趋近于0，训练几乎停滞。
        总结：
            鞍点 ❌    缓坡（过长）❌    局部最小值 ❌    悬崖 ✔    最小值附近震荡 ✔

    3. 自适应学习率：RMSprop
        公式：
            Wt = Wt-1 - 学习率/(sqrt(Vt) + σ) * Gt
            Vt = β*Vt-1 + (1-β)*Gt²
            ⇔  Vt = (1-β)Gt² + β(1-β)Gt-1² + ... + β^(t-1)(1-β)G1
        解释：
            Vt：平方梯度的指数移动平均值（引入衰减β，避免无限累加）。
        总结：
            RMSprop 通过引入权重系数 β 让历史信息指数级衰减，从而避免了学习率持续单调下降的问题。

    4. 自适应矩估计（Adaptive Moment Estimation）：Adam
        说明：
            Adam 融合了动量法（一阶矩）和 RMSProp（二阶矩）的优势，为每个参数自适应调整学习率。
        关键步骤：
            1. 更新一阶矩（动量项）：
                Mt = β1 * Mt-1 + (1 - β1) * Gt   （梯度均值的估计）
            2. 更新二阶矩（梯度平方的移动平均）：
                Vt = β2 * Vt-1 + (1 - β2) * Gt²  （梯度方差的估计）
            3. 偏差校正（因为 M0, V0 初始化为0，训练初期会偏小）：
                M_hat_t = Mt / (1 - β1^t)
                V_hat_t = Vt / (1 - β2^t)
            4. 参数更新：
                Wt = Wt-1 - 学习率/(sqrt(V_hat_t) + σ) * M_hat_t
"""

import torch
import torch.optim as optim


# 1. 演示：动量法（Momentum）
def dm01_momentum():
    print("=" * 50)
    print("动量法 Momentum 搜寻损失函数的最小值点")
    print("=" * 50)
    # 1. 初始化权重参数（requires_grad=True 表示需要计算梯度）
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2. 创建优化器：SGD + momentum
    #    momentum=0.9 表示 β=0.9
    #    PyTorch 内部更新规则：v = 0.9*v + g,  w = w - lr*v
    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)

    # 3. 训练循环（模拟多次参数更新）
    for epoch in range(11):  # 第0步 + 10步更新
        # 定义损失函数：loss = w^2 / 2，其梯度为 w
        loss = w ** 2 / 2

        # 标准三步：清零梯度 → 反向传播 → 更新参数
        optimizer.zero_grad()  # 清空历史梯度（防止累加）
        loss.backward()  # 反向传播，计算当前梯度 w.grad
        optimizer.step()  # 优化器根据动量法更新 w.data

        if epoch == 0:
            print(f"初始状态: w.data = {w.data.item():.4f}, w.grad = {w.grad.item():.4f}")
        else:
            print(f"第 {epoch:2d} 次更新: w.data = {w.data.item():.6f}, w.grad = {w.grad.item():.6f}")
    print()


# 2. 演示：自适应学习率 AdaGrad
def dm02_AdaGrad():
    print("=" * 50)
    print("自适应学习率法 AdaGrad 搜寻损失函数最小值点")
    print("=" * 50)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # AdaGrad 优化器，lr=0.01
    optimizer = optim.Adagrad(params=[w], lr=0.01)

    for epoch in range(11):
        loss = w ** 2 / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 0:
            print(f"初始状态: w.data = {w.data.item():.4f}, w.grad = {w.grad.item():.4f}")
        else:
            print(f"第 {epoch:2d} 次更新: w.data = {w.data.item():.6f}, w.grad = {w.grad.item():.6f}")
    print()


# 3. 演示：自适应学习率 RMSprop
def dm03_RMSprop():
    print("=" * 50)
    print("自适应学习率法 RMSprop 搜寻损失函数最小值点")
    print("=" * 50)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # RMSprop 优化器，lr=0.01, alpha=0.9（对应公式中的 β）
    optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.9)

    for epoch in range(11):
        loss = w ** 2 / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 0:
            print(f"初始状态: w.data = {w.data.item():.4f}, w.grad = {w.grad.item():.4f}")
        else:
            print(f"第 {epoch:2d} 次更新: w.data = {w.data.item():.6f}, w.grad = {w.grad.item():.6f}")
    print()


# 4. 演示：自适应矩估计 Adam
def dm04_Adam():
    print("=" * 50)
    print("自适应矩估计 Adam 搜寻损失函数最小值点")
    print("=" * 50)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # Adam 优化器，lr=0.01, betas=(β1, β2)
    # 这里为了演示，将 β2 也设为 0.9（默认是 0.999），方便观察与 RMSprop 的区别
    optimizer = optim.Adam(params=[w], lr=0.1, betas=(0.9, 0.9))

    for epoch in range(11):
        loss = w ** 2 / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 0:
            print(f"初始状态: w.data = {w.data.item():.4f}, w.grad = {w.grad.item():.4f}")
        else:
            print(f"第 {epoch:2d} 次更新: w.data = {w.data.item():.6f}, w.grad = {w.grad.item():.6f}")
    print()


if __name__ == '__main__':
    # 调用四个演示函数
    dm01_momentum()
    dm02_AdaGrad()
    dm03_RMSprop()
    dm04_Adam()
