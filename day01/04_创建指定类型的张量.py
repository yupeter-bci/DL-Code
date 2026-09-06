"""
案例:
    创建指定类型的张量.

涉及到的函数:
    type(torch支持的数据类型)
    half()/double()/float()/short()/int()/long()

你要掌握的函数:
    type()
"""

# 导包
import torch

# 场景1: 直接创建指定类型的张量.
t1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)  # 默认是: float32
print(f't1: {t1}, (元素)类型: {t1.dtype}, (张量)类型: {type(t1)}')  # float32
print('-' * 30)

# 场景2: 创建好张量后 -> 做类型转换.
# 思路1: type()函数, 推荐掌握.
t2 = t1.type(torch.int16)
print(f't2: {t2}, (元素)类型: {t2.dtype}, (张量)类型: {type(t2)}')  # int16
print('-' * 30)


# 思路2: half()/double()/float()/short()/int()/int()
print(t2.half())        # float16
print(t2.float())       # float32, 默认
print(t2.double())      # float64
print(t2.short())       # int16
print(t2.int())         # int32
print(t2.long())        # int64, 默认