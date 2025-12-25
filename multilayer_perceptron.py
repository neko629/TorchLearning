import torch
from torch import nn
from d2l import torch as d2l
from practice.softmax.softmax_practice import loss

def manual_mlp():
    """手动实现的多层感知机"""
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    W1 = nn.Parameter(  # nn.Parameter 函数将一个张量转换为模型参数, 并且会自动将其添加到模型的参数列表中
        torch.randn(num_inputs, num_hiddens, requires_grad = True)  # 第一层权重
    )
    b1 = nn.Parameter(
        torch.zeros(num_hiddens, requires_grad = True)  # 第一层偏置
    )
    W2 = nn.Parameter(
        torch.randn(num_hiddens, num_outputs, requires_grad = True)  # 第二层权重
    )
    b2 = nn.Parameter(
        torch.zeros(num_outputs, requires_grad = True)  # 第二层偏置
    )

    params = [W1, b1, W2, b2]

    def relu(X):  # 激活函数
        a = torch.zeros_like(X)  # zero_like 返回一个与 X 形状相同的全零张量
        return torch.max(X, a)  # 返回 X 和 a 中对应位置的最大值

    # 定义模型
    def net(X):
        X = X.reshape((-1, num_outputs))  # 将输入展平为二维张量, -1 表示自动计算该维度的大小
        H = relu(X @ W1 + b1)  # 第一层的输出, @ 表示矩阵乘法
        return (H @ W2) + b2  # 第二层的输出

    loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    num_epochs, lr = 20, 0.1
    trainer = torch.optim.SGD(params, lr = lr)  # 随机梯度下降优化算法
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


def mlp_using_nn():
    """使用 PyTorch 内置的 nn 模块实现的多层感知机"""
    net = nn.Sequential(
        nn.Flatten(), #  # 将输入展平为二维张量
        nn.Linear(784, 256), # 第一层线性层
        nn.ReLU(), # 激活函数
        nn.Linear(256, 10) # 第二层线性层
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std = 0.01) # 使用正态分布初始化权重, 标准差为0.01

    net.apply(init_weights) # 初始化模型参数
    batch_size, lr, num_epochs = 256, 0.1, 20
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr = lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

"""
nn 仅仅简略了模型的定义和参数初始化部分, 训练和评估部分与手动实现的多层感知机是一样的
"""

