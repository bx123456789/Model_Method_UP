'''
1 Dropout 是一种正则化技术，通过在训练过程中随机丢弃一些神经元来防止过拟合。在每次训练迭代中，
  Dropout 会以一定概率（drop_prob）将一些神经元设置为零，而不会对它们的梯度进行更新。为了补偿被丢弃的神经元，保留的神经元的输出需要进行缩放。

2 DropPath 是 Dropout 的一种变体，通常用于残差网络中。它以一定的概率丢弃整个路径（而不是单个神经元），这意味着某些残差块在训练过程中被随机禁用。
'''

import torch
import torch.nn as nn

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        # div 就是除法的意思
        return x.div(keep_prob) * random_tensor

if __name__ == '__main__':
    # 测试 DropPath 模块
    drop_path = DropPath(drop_prob=0.2)
    drop_path.train()  # 设置为训练模式

    # 创建一个示例输入张量
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # 计算输出
    output = drop_path(x)
    print("输入张量：")
    print(x)
    print("输出张量：")
    print(output)
