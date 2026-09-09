"""
例：神经网络输入、输出、损失函数
输出层的输入：Yi = Wi * Xi-1 + b  （输出层为第 i 层）
输出层的输出：Xi = softmax(Yi)
输出层的损失函数：Loss(Xi) = MLE(xi)  （最大似然估计 = 交叉熵）
神经网络训练的参数 Wi, b 并不是唯一的，有大有小。 → 可以这么理解：损失函数是帮助神经网络优化网格参数，以达到拟合训练集的目的，符合拟合要求的参数不是唯一的，就像
                                                     能穿过所有数据点的函数不是唯一的，但是它们对未知数据的拟合能力却是不同的，也就是泛化误差不同。
诉求：
    我们希望获得值尽量小的一组 Wi 参数（偏置 b 不与输入数据相乘，不会缩放误差和噪音） → 这是因为预测的新输入数据含义一定的误差和噪音，过大的参数 Wi 会进一步放大该误差和噪音。
    · 权重参数 Wi 决定模型对输入特征的敏感度。如果 Wi 很大，输入稍微变化一点，输出就会剧烈变化，导致决策边界分厂曲折、复杂，容易过拟合。
    · 偏置参数 b 只是将整个决策平面整体平移，它不随输入的变化而变化。
    惩罚（限制范围 → 正则化）偏置参数 b 会损害模型表达力：
    · 无偏置约束：模型可以自由调整 b，让决策边界移动到最合适的位置（比如完美分割两类数据的中间）。
    · 有偏置约束：比如强行让 b 接近 0。如果真实的最优 b = 100，你强迫它只能取 0 附近的数，模型旧无法正确平移决策边界，导致欠拟合。
沿此思路提出问题：
    如何获得值尽量小的一组 Wi 参数？ → 答：拉格朗日乘数法
措施：
    L2正则化（岭回归）：
        min_J(W, b) 约束条件：∥W∥₂² - C ≤ 0 （C 代表参数平方的上界，这个约束明确设定了一个“范围值”）
        表达式：Loss(W, b, λ) = min_J(w, b) + λ(∥W∥₂² - C)  λ ≥ 0
        表达式解析：KKT条件（最优性必要条件）中的一个关键条件是互补松弛条件：
                        λ(∥W∥₂² - C) = 0
                这意味着：
                    · 如果约束不起作用（即最优解 W* 满足 ∥W*∥₂² ＜ C），那么 λ = 0，问题退化为无约束条件。
                    · 如果约束起作用（即∥W*∥₂² ＝ C），那么 λ ＞ 0。
                注意：在求解过程中，我们是通过对 W 求梯度，并考虑 λ 的取值来寻找满足 KKT 条件的点。此时 C 仍然在方程中。
        在实际应用中为什么看起来 C 消失了呢？
            在实际机器学习应用中，我们并不直接求解上面那个固定 C 的约束问题。我们求解的是它的拉格朗日对偶问题。
            对于凸问题（如岭回归），在约束作用的情况下，存在一个一一对应关系：
             · 每一个约束半径 C 对应一个最优的拉格朗日乘子 λ*。（当我们从“带约束问题”转换到“无约束的正则化问题”时，C 和 λ 之间存在一个函数关系 λ ＝ f(C)）
             · 而这个带约束的原问题，等价于下面这个无约束的优化问题（正则化形式）：
                    min[J(w, b) + λ*(∥W∥₂²)]  （在实际代码中，我们直接指定 λ* ，因为调节 λ 比调节 C 更方便，并且两者是等价的（对凸问题而言））
            关键来了：在这个（看似）无约束形式中，C 消失了，却而代之的是正则化系数 λ*。但是 λ* 的取值是与 C 紧密相关的（通常 C 越小， λ* 越大）。
        为什么实践中我们只指定 λ 而不指定 C ?
            原因是：我们放弃了直接控制约束半径 C，转而控制对偶变量 λ。
            C：物理含义是“参数允许的最大平方和（∥W*∥₂²）”，但调节 C 不够直观，因为它的范围依赖与数据尺度。
            λ：物理含义是“参数增长的惩罚强度”，调节 λ 更方便、更平滑、更符合直觉（0 到正无穷，越大越稀疏/约束范围越小）。

    L1正则化（Lasso）：
        原理和L2正则化相似。
        表达式：Loss(W, b, λ) = min[J(w, b) + λ*(∥W∥₁)]  λ* ≥ 0

    L1正则化 和 L2正则化的特点及其适用场景：
        L1正则化特点：
            1. 稀疏性：函数解中的大量系数精确为 0，自动做特征选择；
            2. 解的不唯一性：当特征数 ＞ 样本数（矩阵不满秩）时，解可能不唯一（但常用最小角度回归获得某个解）；
            3. 非光滑性：在 0 点不可导，需用次梯度或近端梯度等算法；
            4. 解的路径：随 λ 变化，系数路径分段线性，有折点；
            5. 对异常值：相对稳健（比 L2 对异常值敏感度低）；
            6. 组效应：对高度相关特征，通常只选其中一个，不鼓励同时保留。
        L1正则化适用场景：
            1. 特征维度非常高（如 特征数 >> 样本数），需要降维或特征选择；
            2. 希望模型可解释性强，只有少数特征起主要作用；
            3. 已知真实模型稀疏（如基因选择、文本分类中某些词无用）；
            4. 需要去除冗余或噪声特征；
            5. 计算资源有限，希望最终模型存储小（因为很多系数为 0 ）。

        L2正则化特点：
            1. 非稀疏性：系数趋于 0 但不会精确为 0，所有特征都被保留；
            2. 解的唯一性：严格凸，解唯一且稳定；
            3. 光滑性：处处可导，可用标准梯度下降；
            4. 解的路径：随 λ 变化，系数路径光滑曲线；
            5. 对异常值敏感（平方放大了大误差和噪音的影响）
            6. 组效应：对高度相关特征，它们的系数会相互接近，一起保留或一起缩小。
        L2正则化适用场景：
            1. 特征数适中，不需要特征选择；
            2. 所有特征都可能有用，不希望丢失信息；
            3. 存在多重共线性（特征高度相关）→ L2能稳定系数估计；
            4. 关注模型预测精度胜过可解释性。

        L1 L2正则化混合使用（Elastic Net 弹性网络）场景：
            希望既特征稀疏（可解释）又希望处理高度相关的特征（组效应），且特征数远多于样本数。
            则使用 Elastic Net（弹性网络）：
                        Loss(W,b,λ1,λ2) = min[J(W,b) + λ1∥W∥₁ + λ2∥W∥₂²]

        一句话总结：
            L1：想要一个简单、稀疏、可解释的模型 → 用它
            L2：想要一个稳定、不丢弃任何特征、所有特征都可能有点用的模型 → 用它
            都不确定：先试 L2（稳定），再试 L1（稀疏），必要时 Elastic Net。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# 搭建神经网络（输出层不再包含softmax，因为CrossEntropyLoss会自动计算log_softmax）
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层：输入3维，输出4维，使用sigmoid激活
        self.linear1 = nn.Linear(in_features=3, out_features=4)
        # 第二层：输入4维，输出2维（logits），不添加激活函数
        self.output = nn.Linear(in_features=4, out_features=2)
        # 使用默认初始化（而非全1），避免对称性问题
        # 如果有特殊需求可以自行修改

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.output(x)  # 输出logits
        return x


# 构建 L1 / L2 / Elastic Net 正则化项（只对权重，不对偏置）
def l_regularization(model, l='l2', l_lambda=1.0, l1_ratio=0.5):
    """
    计算模型的正则化损失项。

    参数:
        model: 神经网络模型
        l: 正则化类型，可选 'l1', 'l2', 'elasticnet'
        l_lambda: 正则化强度系数（总惩罚系数）
        l1_ratio: 当 l='elasticnet' 时，L1 部分的比例（0~1），其余为 L2 部分

    返回:
        一个标量张量，表示正则化损失（已乘以系数）
    """
    reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if 'weight' in name:  # 只对权重参数进行正则化，忽略偏置
            if l == 'l1':
                reg_loss = reg_loss + param.abs().sum()
            elif l == 'l2':
                reg_loss = reg_loss + param.pow(2).sum()
            elif l == 'elasticnet':
                reg_loss = reg_loss + l1_ratio * param.abs().sum() + (1 - l1_ratio) * param.pow(2).sum()
            else:
                raise ValueError(f"Unsupported regularization type: {l}")
    return l_lambda * reg_loss


def train(device, inputs, targets, num_epochs=100, l='l2', l_lambda=0.01):
    """
    训练一个简单的神经网络分类器（二分类）。

    参数:
        device: 使用的设备（cpu 或 cuda）
        inputs: 输入特征张量，形状 (样本数, 特征数)
        targets: 标签张量，形状 (样本数,)，每个元素为类别索引（0 或 1）
        num_epochs: 训练轮数
        l: 正则化类型
        l_lambda: 正则化强度

    返回:
        loss_list: 每个epoch的损失值列表
    """
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()  # 内部包含log_softmax，因此模型输出不需要softmax
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # 适当增大学习率以加快演示

    # 创建数据集对象并封装为DataLoader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=15, shuffle=True)

    loss_list = []
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            # 前向传播
            logits = model(batch_x)
            # 计算交叉熵损失 + 正则化项
            loss = criterion(logits, batch_y) + l_regularization(model, l=l, l_lambda=l_lambda)
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)  # 累加本批总损失
        # 计算本epoch平均损失
        avg_loss = epoch_loss / len(dataset)
        loss_list.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f}")

    # 简单评估（训练集上的准确率）
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        pred = torch.argmax(logits, dim=1)
        print(f'y_pred: {pred.detach()}\ny_true: {targets}\n')
        acc = (pred == targets).float().mean().item()
    print(f"Training Accuracy: {acc:.2%}")
    return loss_list


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构造一个简单的二分类数据集（类别交替，可能不是线性可分的，仅用于演示）
    # 输入：40个样本，每个3个特征（值在1~3之间）
    inputs = torch.linspace(1.0, 3.0, 120).reshape(40, 3).to(device)
    # 标签：类别0和1交替出现，转换为LongTensor
    targets = torch.tensor([i % 2 for i in range(40)], dtype=torch.long).to(device)

    # 训练模型（使用L2正则化，系数0.01）
    loss_list = train(device, inputs, targets, num_epochs=100, l='l2', l_lambda=0.1)
    print("Final loss list (last 5):", loss_list[-5:])
