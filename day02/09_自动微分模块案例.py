"""
案例:
    演示自动微分模块, 循环实现 计算梯度, 更新参数.

需求:
    求 y = x**2 + 20 的极小值点 并打印y是最小值时 w的值(梯度)

解题步骤:
    1. 定义点 w=10 requires_grad=True  dtlosspe=torch.float32
    2. 定义函数 loss = w**2 + 20
    3. 利用梯度下降法 循环迭代1000 求最优解
    3.1 正向计算(前向传播)
    3.2 梯度清零 w.grad.zero_()
    3.3 反向传播
    3.4 梯度更新 w.data = w.data - 0.01 * w.grad

回顾:
    W新 = W旧 - 学习率 * 梯度
"""
# 导包
import torch

# 1. 定义点 w=10 requires_grad=True  dtlosspe=torch.float32
# 参1: 初始值, 参2: 自动微分(求导), 参3: 数据类型, 浮点型.
w = torch.tensor(10, requires_grad=True, dtype=torch.float32)

# 2. 定义函数 loss = w**2 + 20
loss = w ** 2 + 20          # 求导: loss' = 2w

# 3. 利用梯度下降法 循环迭代100 求最优解
print(f'开始 权重初始值: {w}, (0.01 * w.grad): 无,  loss: {loss}')   # 10, 无, 120

# 迭代100次, 求最优解.
for i in range(1, 101):
    # 3.1 正向计算(前向传播)
    loss = w ** 2 + 20

    # 3.2 梯度清零 w.grad.zero_()   默认梯度会累加.
    # 至此(第一次的时候), 还没有计算梯度, 所以w.grad = None, 要做非空判断.
    if w.grad is not None:
        w.grad.zero_()

    # 3.3 反向传播
    loss.sum().backward()

    # 3.4 梯度更新 w.data = w.data - 0.01 * w.grad
    # print(f'梯度值为: {w.grad}')
    w.data = w.data - 0.01 * w.grad

    # 3.5 打印本次 梯度更新后的 权重参数结果.
    print(f'第 {i} 次, 权重初始值: {w}, (0.01 * w.grad): {0.01 * w.grad:.5f}, loss: {loss:.5f}')

# 4. 打印最终结果.
print(f'最终结果 权重: {w}, 梯度: {w.grad:.5f}, loss: {loss:.5f}')
