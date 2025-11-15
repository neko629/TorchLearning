# 这里获取现成的数据，所以不需要生成数据集
# 所以只需要执行以下步骤
# 1. 读取数据集
# 2. 定义模型
# 3. 定义损失函数
# 4. 定义优化算法
# 5. 训练模型
# 6. 评估模型


import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l.torch import Accumulator
from animator import Animator

# transforms 是 pytorch 中专门用来做图像与处理的模块


def get_dataloader_workers():
    return 4 # 使用4个进程来读取数据

# 读取数据集
def load_data_fashion_mnist(batch_size, resize = None):
    trans = [transforms.ToTensor()] # 定义 trans，是处理图像各个步骤的列表，先把图像转为张量的工具 ToTensor 放进去
    if resize: # 如果需要调整图像大小，则把调整图像大小的步骤放到最前面
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root = "../data", train = True, transform = trans, download = True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root = "../data", train = False, transform = trans, download = True
    )
    # 返回一个元组，包含训练集和测试集的迭代器
    return (
        data.DataLoader(mnist_train, batch_size, shuffle = True, num_workers = get_dataloader_workers()),
        data.DataLoader(mnist_test, batch_size, shuffle = False, num_workers = get_dataloader_workers())
    )

# 定义 softmax 回归模型
softmax_net = torch.nn.Sequential( # 顺序容器，将各个层按顺序组合在一起
    torch.nn.Flatten(), # 将输入的图像展平为一个向量
    torch.nn.Linear(784, 10) # 784 = 28 * 28, 输入特征的维度是 784，输出特征的维度是 10
)

def accuracy(y_hat, y):
    # 兼容 logits/probs (batch,num_classes) 与 label (batch,)
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    if y.ndim > 1 and y.shape[1] > 1:
        y = y.argmax(dim=1)
    y_hat = y_hat.reshape(-1).long()
    y = y.reshape(-1).long()
    return float((y_hat == y).sum().item())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)  # [正确数, 样本总数]
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]

#%%
# Softmax 回归的训练
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

#%%
# 训练行数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics


# 定义 softmax 损失函数
softmax_loss = torch.nn.CrossEntropyLoss(reduction='none') # 交叉熵损失函数

# 定义优化算法, 使用小批量随机梯度下降
lr = 0.1 # 学习率
softmax_trainer = torch.optim.SGD(softmax_net.parameters(), lr)

# 训练模型, 入参依次是：模型，训练集迭代器，测试集迭代器，损失函数，优化算法，训练轮数
num_epochs = 10

def train_softmax_model(net, train_iter, test_iter, loss, trainer, num_epochs):
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
train_softmax_model(softmax_net, train_iter, test_iter, softmax_loss, softmax_trainer, num_epochs)

# 在非 Jupyter 环境下，需要显示并保持图形窗口
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式
plt.show()  # 显示所有图形
