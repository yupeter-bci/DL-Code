"""
案例:
    演示张量 和 numpy之间如何相互转换, 以及 如何从标量张量中 提取其内容.

涉及到的API:
    场景1: 张量 -> numpy  nd数组对象
        张量对象.numpy()            共享内存
        张量对象.numpy().copy()     不共享内存,  链式编程写法.
    场景2: numpy nd数组 -> 张量.
        from_numpy()              共享内存
        torch.tensor(nd数组)       不共享内存
    场景3: 从标量张量中 提取其内容.
        标量张量.item()

掌握:
    张量 -> numpy:   张量对象.numpy()
    numpy -> 张量:   torch.tensor(nd数组)
    从标量张量中 提取其内容:  标量张量.item()
"""
import torch
import numpy as np


# 1. 定义函数, 演示: 张量 -> numpy
def dm01():
    # 1. 创建张量.
    t1 = torch.tensor([1, 2, 3, 4, 5])
    print(f't1: {t1}, type: {type(t1)}')

    # 2. 张量 -> numpy.
    # n1 = t1.numpy()           # 共享内存.
    n1 = t1.numpy().copy()      # 不共享内存.
    print(f'n1: {n1}, type: {type(n1)}')

    # 3. 演示上述方式 是否共享内存.
    n1[0] = 100
    print(f'n1: {n1}')  # [100, 2, 3, 4, 5]
    print(f't1: {t1}')  # [?, 2, 3, 4, 5]

# 2. 定义函数, 演示: numpy -> 张量
def dm02():
    # 1. 创建numpy数组.
    n1 = np.array([11, 22, 33])
    print(f'n1: {n1}, type: {type(n1)}')

    # 2. 把上述的numpy数组, 转换成张量.
    # t1 = torch.from_numpy(n1).type(torch.float32)   # 转换 ＋ 转类型.
    t1 = torch.from_numpy(n1)               # 共享内存.
    print(f't1: {t1}, type: {type(t1)}')

    t2 = torch.tensor(n1)                   # 不共享内存
    print(f't2: {t2}, type: {type(t2)}')

    # 3. 演示上述方式 是否共享内存.
    n1[0] = 100
    print(f'n1: {n1}')  # 100, 22, 33
    print(f't1: {t1}')  # 100, 22, 33
    print(f't2: {t2}')  # 11, 22, 33


# 3. 定义函数, 演示: 从标量张量(只有1个值的张量) 中 提取其内容.
def dm03():
    # 1. 创建张量.
    t1 = torch.tensor(100)              # 可以
    # t1 = torch.tensor([100, ])        # 可以
    # t1 = torch.tensor([100, 200])     # 不可以.
    print(f't1: {t1}, type: {type(t1)}')

    # 2. 从张量中提取内容.
    a = t1.item()
    print(f'value: {a}, type: {type(a)}')

# 4. 测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()