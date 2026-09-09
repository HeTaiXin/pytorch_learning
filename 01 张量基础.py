import torch

# 设置数据类型和设备
dtype = torch.float  # 张量数据类型为浮点型
device = torch.device("cpu")  # 本次计算在 CPU 上进行

# 创建并打印两个随机张量 a 和 b
a = torch.randn(2, 3, device=device, dtype=dtype)  # 创建一个 2x3 的随机张量
b = torch.randn(2, 3, device=device, dtype=dtype)  # 创建另一个 2x3 的随机张量

print("张量 a:")
print(a)

print("张量 b:")
print(b)

# 逐元素相乘并输出结果
print("a 和 b 的逐元素乘积:")
print(a * b)

# 输出张量 a 所有元素的总和
print("张量 a 所有元素的总和:")
print(a.sum())

# 输出张量 a 中第 2 行第 3 列的元素（注意索引从 0 开始）
print("张量 a 第 2 行第 3 列的元素:")
print(a[1, 2])

# 输出张量 a 中的最大值
print("张量 a 中的最大值:")
print(a.max())

import torch
import numpy as np

# 张量 —> numpy
def dm01():
    t1 = torch.tensor([1, 2, 3, 4, 5])
    n1 = t1.numpy()  # 和 t1 共享内存
    cn1 = t1.numpy().copy()  # 和 t1 不共享内存
    t1[0] = 100
    print(f't1: {t1}, type: {type(t1)}')
    print(f'n1: {n1}, type: {type(n1)}')
    print(f'cn1: {cn1}, type: {type(cn1)}')

# numpy —> 张量
def dm02():
    n1 = np.array([1, 2, 3, 4, 5])
    t1 = torch.tensor(n1)  # 和 n1 不共享内存
    t2 = torch.from_numpy(n1)  # 和 n1 共享内存
    t3 = t2.type(torch.float32)  # int32 转为 float32，和 n1 不共享内存
    n1[0] = 100
    print(f'n1: {n1}, type: {type(n1)}')
    print(f't1: {t1}, type: {type(t1)}')
    print(f't2: {t2}, type: {type(t2)}')
    print(f't3: {t3}, type: {type(t3)}')

# 测试
if __name__ == '__main__':
#     dm01()
    dm02()
