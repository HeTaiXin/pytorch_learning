# PyTorch 深度学习实践代码库

> 系统性学习 PyTorch 的学习路径与代码笔记，覆盖从**张量基础 → 数据管道 → 模型搭建 → 优化算法 → 正则化 → 归一化 → 卷积神经网络**的完整链路，最终落地到 **CIFAR-10 图像分类完整项目**。

本仓库记录了我从零学习深度学习工程化的过程，每个脚本都配有详尽的中文注释（原理推导 + 代码演示），可作为入门参考资料，也是我投递**自动驾驶算法岗位**的技术能力佐证。

---

## 📖 目录

- [一、项目概览](#一项目概览)
- [二、目录结构](#二目录结构)
- [三、学习路径与知识点](#三学习路径与知识点)
- [四、环境依赖](#四环境依赖)
- [五、快速开始](#五快速开始)
- [六、核心算法笔记](#六核心算法笔记)
- [七、CIFAR-10 完整项目说明](#七cifar-10-完整项目说明)
- [八、代码规范与设计思路](#八代码规范与设计思路)
- [九、与自动驾驶的结合](#九与自动驾驶的结合)
- [十、后续计划](#十后续计划)
- [十一、参考资源](#十一参考资源)
- [十二、许可证](#十二许可证)

---

## 一、项目概览

| 项目 | 说明 |
|------|------|
| **仓库名** | `pytorch_learning` |
| **定位** | PyTorch 系统性学习代码库（含原理注释 + 可运行 Demo） |
| **语言/框架** | Python 3 / PyTorch / TorchVision |
| **覆盖主题** | 张量、线性回归、nn.Module、优化器、学习率调度、正则化、Dropout、BN、CNN |
| **最终落地** | CIFAR-10 图像分类（训练/验证/测试全流程） |
| **学习方式** | 每个知识点独立成脚本，由浅入深、可单文件运行 |

**设计原则**：
1. **原理优先**：每个优化器/正则化/归一化算法都先讲公式和适用场景，再给可运行代码；
2. **可视化驱动**：训练曲线、特征图、拟合效果均用 matplotlib 绘图验证；
3. **工程规范**：采用 `TensorDataset + DataLoader`、`torch.no_grad()`、`.eval()`/`.train()` 切换等工业级写法。

---

## 二、目录结构

```
pytorch_learning/
├── 01 张量基础.py                     # 张量创建、索引、逐元素运算；Tensor ↔ NumPy 互转（共享内存）
├── 02 线性回归全流程.py               # sklearn 造数据 → DataLoader → nn.Linear → SGD → 可视化
├── 03 自定义 nn.Module 网络.py        # 继承 nn.Module、forward、Xavier/Kaiming 初始化、torchsummary
├── 04 优化器算法对比.py               # Momentum / AdaGrad / RMSprop / Adam 原理 + 演示
├── 05 ANN案例_手机价格分类.py         # （旧版）ANN 手机价格分类
├── 05 学习率衰减策略.py               # StepLR / MultiStepLR / ExponentialLR
├── 05 学习率调度.py                   # 学习率调度公共训练框架
├── 06 Dropout 正则化原理与效果演示.py  # Dropout 缩放原理、train/eval 模式
├── 07 L1_L2_Elastic Net 正则化.py     # L1/L2/ElasticNet 拉格朗日推导 + 实现
├── 08 批归一化 BN.py                  # BatchNorm1d / BatchNorm2d 原理与演示
├── 09 ANN 手机价格分类.py             # （完整版）手机价格 4 分类：训练/验证/测试三阶段
├── 10 Conv2d 参数介绍与简单使用.py     # Conv2d 参数详解（kernel/stride/padding/same）
├── 11 卷积_激活_池化特征图可视化.py     # 卷积→ReLU→MaxPool 特征图灰度可视化
├── 12 CIFAR-10 图像分类 CNN 完整项目.py # CNN 完整项目：数据增强 + BN + Dropout + 权重衰减 + 学习率调度
├── toolkit/                           # 公共工具（get_batch_norm / get_dropout / weight_decay）
├── MNIST/raw/                         # MNIST 数据集（如本地缓存）
├── .gitignore
└── README.md
```

> ⚠️ 说明：脚本按学习顺序编号 01~12，`05` 有三个文件是不同阶段的学习笔记（学习率、调度、手机价格 ANN），保留可体现学习过程；如需整理可合并。

---

## 三、学习路径与知识点

### 阶段一：PyTorch 基础（01-02）

| 脚本 | 核心知识点 |
|------|-----------|
| `01 张量基础.py` | `torch.randn`、索引 `a[1,2]`、逐元素运算、`.sum()/.max()`；**Tensor ↔ NumPy 互转**（`.numpy()` 共享内存 / `torch.from_numpy` / `.copy()` 断开共享） |
| `02 线性回归全流程.py` | `make_regression` 造数据、`TensorDataset + DataLoader`（mini-batch、`shuffle`）、`nn.Linear`、`nn.init.constant_`、`optim.SGD`、`nn.MSELoss`、`torch.no_grad()`、matplotlib 三子图可视化 |

### 阶段二：模型与初始化（03）

| 脚本 | 核心知识点 |
|------|-----------|
| `03 自定义 nn.Module 网络.py` | 继承 `nn.Module`、`__init__` + `forward`、`nn.Linear` 堆叠、**Xavier（配 sigmoid）/ Kaiming（配 ReLU）初始化**、`torchsummary` 查看参数量、`device = cuda if available` |

### 阶段三：优化算法（04-05）

| 脚本 | 核心知识点 |
|------|-----------|
| `04 优化器算法对比.py` | **Momentum**（冲过鞍点/局部极小，但悬崖风险）、**AdaGrad**（学习率单调降，长缓坡停滞）、**RMSprop**（指数移动平均修复 AdaGrad）、**Adam**（一阶矩 + 二阶矩 + 偏差校正）；用 `loss = w²/2` 演示每步更新 |
| `05 学习率衰减策略.py` | **StepLR / MultiStepLR / ExponentialLR** 三种调度器；"先大后小"策略；`scheduler.step()` 时机；可视化学习率/损失曲线 |
| `05 学习率调度.py` | 抽取公共训练函数 `train_with_scheduler`，统一记录 lr 与 loss |

### 阶段四：正则化与归一化（06-08）

| 脚本 | 核心知识点 |
|------|-----------|
| `06 Dropout 正则化原理与效果演示.py` | Dropout 缩放 `1/(1-p)` 保持期望、`.train()` vs `.eval()`、`nn.Dropout`、推理用 `torch.no_grad()` |
| `07 L1_L2_Elastic Net 正则化.py` | **从拉格朗日乘数法推导 L2**（岭回归、KKT 条件、为何用 λ 不用 C）；**L1**（稀疏/特征选择）、**L2**（稳定/组效应）、**Elastic Net**；代码只对 **weight** 正则化、忽略 bias |
| `08 批归一化 BN.py` | BN 数学公式（μ/σ/γ/β）、**训练用 batch 统计量、推理用 running_mean**、缓解梯度消失、允许更大学习率；`BatchNorm1d` / `BatchNorm2d` 演示（均值→0、方差→1） |

### 阶段五：完整项目（09-12）

| 脚本 | 核心知识点 |
|------|-----------|
| `09 ANN 手机价格分类.py` | **train/val/test 三阶段**、`train_test_split`(64/16/20)、`CrossEntropyLoss`（内含 softmax）、保存最佳模型 `best_model.pth`、验证集监控、`@torch.no_grad()` + `.eval()` |
| `10 Conv2d 参数介绍与简单使用.py` | `in/out_channels`、`kernel_size`、`stride`、`padding`（含 `'same'`）、输出尺寸计算 |
| `11 卷积_激活_池化特征图可视化.py` | HWC → CHW → NCHW（`permute` + `unsqueeze`）、卷积核可视化、ReLU、`MaxPool2d`、特征图 cmap 灰度显示 |
| `12 CIFAR-10 图像分类 CNN 完整项目.py` | **端到端 CNN**：数据增强（`RandomHorizontalFlip` + `Normalize`）、3 个 Conv+BN+ReLU+Pool+Dropout 块、`weight_decay` 只对 weight、**`MultiStepLR` 学习率调度**、`non_blocking` 异步 GPU 传输、`pin_memory` |

---

## 四、环境依赖

- **Python**：3.8+
- **PyTorch**：2.0+（含 TorchVision）
- **CUDA**：可选，自动检测（`device = torch.device('cuda' if ...)`）

安装依赖：

```bash
pip install torch torchvision matplotlib numpy scikit-learn pandas
# 如需 torchsummary
pip install torchsummary
```

或将以下内容保存为 `requirements.txt`：

```
torch>=2.0
torchvision>=0.15
matplotlib
numpy
scikit-learn
pandas
torchsummary
```

---

## 五、快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/HeTaiXin/pytorch_learning.git
cd pytorch_learning
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行单个脚本（推荐按编号顺序学习）

```bash
# 线性回归全流程（自动造数据、训练、画图）
python "02 线性回归全流程.py"

# 优化器算法对比（Momentum / AdaGrad / RMSprop / Adam）
python "04 优化器算法对比.py"

# CIFAR-10 CNN 完整训练（首次运行会下载数据集，约 170MB）
python "12 CIFAR-10 图像分类 CNN 完整项目.py"
```

> 💡 脚本间相互独立，均可单独运行；`12` 号脚本会自动下载 CIFAR-10 到 `./data/CIFAR10`。

---

## 六、核心算法笔记

### 6.1 优化器对比速查

| 优化器 | 核心公式 | 优点 | 缺陷 | 适用 |
|--------|---------|------|------|------|
| **SGD** | `w = w - lr * g` | 简单、稳定 | 鞍点/局部极小易停滞 | 基线 |
| **Momentum** | `v=βv+g; w=w-lr*v` | 冲过鞍点/局部极小 | 悬崖可能加剧 | 配合 LR 调度 |
| **AdaGrad** | `V=V+g²; w=w-lr*g/(√V+ε)` | 自适应学习率 | 长缓坡学习率→0 停滞 | 稀疏数据 |
| **RMSprop** | `V=βV+(1-β)g²` | 修复 AdaGrad 单调降 | — | RNN 常用 |
| **Adam** | `m=β₁m+(1-β₁)g; v=β₂v+(1-β₂)g²; 偏差校正` | 一阶矩+二阶矩，兼顾两者 | 可能不收敛到最优点（调 β₂） | **通用首选** |

### 6.2 正则化对比速查

| 类型 | 惩罚项 | 特点 | 场景 |
|------|--------|------|------|
| **L1** | `λ‖w‖₁` | **稀疏**、自动特征选择、系数精确为 0 | 高维/特征选择（基因、文本） |
| **L2** | `λ‖w‖₂²` | 非稀疏、解唯一稳定、所有特征保留 | 多重共线性、重精度 |
| **Elastic Net** | `λ₁‖w‖₁ + λ₂‖w‖₂²` | 兼顾稀疏与组效应 | 高维 + 相关特征 |

> 🔑 **关键认知**（来自 `07` 号脚本推导）：**只对权重 w 做正则化、偏置 b 不做**——偏置仅平移决策边界，惩罚它会损害模型表达力。

### 6.3 训练/评估标准范式

```python
# 训练阶段
model.train()                    # 启用 Dropout、BN 训练行为
for x, y in dataloader:
    optimizer.zero_grad()        # ① 清空梯度
    y_pred = model(x)            # ② 前向
    loss = criterion(y_pred, y)  # ③ 计算损失
    loss.backward()              # ④ 反向传播
    optimizer.step()             # ⑤ 更新参数

# 评估阶段
model.eval()                     # 关闭 Dropout，BN 用 running 统计量
with torch.no_grad():            # 关闭计算图，省显存
    for x, y in dataloader:
        y_pred = model(x)
```

---

## 七、CIFAR-10 完整项目说明

`12 CIFAR-10 图像分类 CNN 完整项目.py` 是本仓库的**集大成者**，覆盖深度学习项目的完整步骤：

### 7.1 项目结构

```
1. 准备数据集（torchvision CIFAR10，5w 训练 / 1w 测试，10 类）
2. 搭建 CNN 网络
3. 模型训练（含验证、保存最佳）
4. 模型测试
```

### 7.2 网络架构

```
输入 (3×32×32)
  └─ Conv2d(3→32) → BN → ReLU → MaxPool → Dropout
  └─ Conv2d(32→64) → BN → ReLU → MaxPool → Dropout
  └─ Conv2d(64→128) → BN → ReLU → MaxPool → Dropout
  └─ Flatten → Linear(128*4*4 → 1024) → BN → ReLU → Dropout
  └─ Linear(1024 → 10) → logits
```

### 7.3 工程技巧（面试加分点）

| 技巧 | 实现 |
|------|------|
| **数据增强** | `RandomHorizontalFlip` + `Normalize`（CIFAR 均值方差） |
| **权重衰减最佳实践** | 自定义 `param_groups`，**仅权重衰减，偏置/BN 不衰减** |
| **学习率调度** | `MultiStepLR(milestones=[25,50,85], gamma=0.1)` |
| **GPU 加速** | `pin_memory=True` + `non_blocking=True` 异步传输 |
| **模型保存** | 按验证集准确率保存 `best_model.pth` |

---

## 八、代码规范与设计思路

- **注释详尽**：每个脚本顶部/关键步骤都有中文原理说明，优化器脚本甚至含完整公式推导；
- **函数封装**：训练逻辑抽取为函数（如 `train`、`evaluate_accuracy`、`l_regularization`）；
- **可复现性**：固定随机种子（`torch.manual_seed`）、`reduction='mean'`；
- **可视化验证**：每个阶段用 matplotlib 验证（拟合曲线、损失下降、特征图、学习率曲线）；
- **配置分离**：超参数集中在函数开头，便于调参；
- **工具模块**：`toolkit/`（BN、Dropout、weight_decay 工厂函数）供多个脚本复用。

---

## 九、与自动驾驶的结合

本仓库的深度学习基础，为我另一个 ROS2 规划项目（`autonomous_driving_ws`）提供算法支撑：

| PyTorch 技能 | 在自动驾驶中的应用 |
|--------------|-------------------|
| **CNN / 特征提取** | 感知模块：车道线检测、目标识别、语义分割 |
| **优化器（Adam/SGD）** | 轨迹优化、参数调优 |
| **正则化（L1/L2/Dropout）** | 防止感知/预测模型过拟合 |
| **BN** | 加速感知网络收敛 |
| **数据管道（DataLoader）** | 处理传感器数据流 |
| **训练/验证/测试范式** | 模型评估标准流程 |

> 🔗 关联项目：[`autonomous_driving_ws`](https://github.com/HeTaiXin/autonomous_driving_ws) — ROS2 自动驾驶规划系统（Frenet 坐标转换、OSQP 参考线 QP 平滑、工厂模式、TF/URDF）。

---

## 十、后续计划

- [ ] 补充 **RNN / LSTM / Transformer**（时序预测，用于轨迹预测）
- [ ] 实现 **目标检测（YOLO）** 与 **语义分割（UNet）**（感知方向）
- [ ] 用 **PyTorch 实现规划相关算法**（如基于学习的轨迹生成）
- [ ] 将 MNIST/CIFAR 脚本统一为 `train.py + config.yaml` 工程化结构
- [ ] 添加 **GitHub Actions CI**（自动化测试）
- [ ] 整理 `05` 系列重复脚本（合并学习率相关文件）

---

## 十一、参考资源

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [《动手学深度学习》(d2l)》](https://d2l.ai/)
- [CS231n: CNN for Visual Recognition](http://cs231n.stanford.edu/)
- 吴恩达《深度学习》课程（优化器、正则化章节）

---

## 十二、许可证

本项目仅供学习交流使用。如需引用或商用，请联系作者。

---

## ✍️ 作者

**HeTaiXin**

- GitHub: [@HeTaiXin](https://github.com/HeTaiXin)
- 关联仓库：[`autonomous_driving_ws`](https://github.com/HeTaiXin/autonomous_driving_ws)（ROS2 自动驾驶规划）

---

⭐ 如果这个仓库对你的 PyTorch 学习有帮助，欢迎 Star / Fork！
