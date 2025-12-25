"""
训练模型的步骤:
1. 准备数据集
2. 读取数据集
3. 定义模型
4. 定义损失函数
5. 定义优化算法
6. 训练模型
7. 评估模型
"""

import torch
from torch import nn # 导入神经网络模块, nn is short for neural network
from d2l import torch as d2l

# 1. 准备数据集
# 使用 fashion-mnist 数据集
# 2. 读取数据集, 读取内容包括训练集和测试集
# 定义每个批次的样本数
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 1024), nn.ReLU(), nn.Dropout(p=0.5),
                    nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(p=0.5),
                    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.5),
                    nn.Linear(256, 10))
# nn.Flatten() 将输入展平为一维向量
# nn.Linear(784, 10) 定义一个线性层，输入特征数为 784(28x28)，输出特征数为 10
# Sequential 函数的作用是将各个层按顺序组合在一起, 这里会先把数据展平, 然后输入到线性层中

# 3 定义模型
def init_weights(m):
    if type(m) == nn.Linear: # 如果是线性层
        nn.init.normal_(m.weight, std=0.01) # 使用正态分布初始化权重, 标准差为0.01

net.apply(init_weights) # 初始化模型参数, apply 函数会递归地将 init_weights 应用到 net 的每一层

# 4 定义损失函数
loss = nn.CrossEntropyLoss(reduction = 'none') # 交叉熵损失函数, 适用于多分类问题, reduction='none' 表示不对损失进行任何归约操作, 如果是 sum 或 mean 则会对损失进行求和或求平均

# 5 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr = 0.1) # 随机梯度下降优化算法, lr 是学习率
scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=30, gamma=0.1)
num_epochs = 100 # 训练轮数

# 6 训练模型
# 计算分类准确率, y_hat 是预测值, y 是真实值, 此函数并没有对比预测值和真实值是否相等, 只是计算预测正确的样本数
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: # 如果 y_hat 是二维张量且第二维度大于1 ,说明是多分类问题
        y_hat = y_hat.argmax(axis = 1) # argmax 返回最大值的索引, axis=1 表示按行取最大值, 进过 argmax 后, y_hat 变成了一维张量, 每个元素是预测的类别索引
    cmp = y_hat.type(y.dtype) == y # 得到的 cmp 是一个布尔张量, 表示预测正确与否
    return float(cmp.type(y.dtype).sum()) # 将布尔张量转换为数值张量并求和, 得到预测正确的样本数

# 评估模型在数据集上的准确率
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module): # 如果 net 是一个 PyTorch 模型
        net.eval() # 将模型设置为评估模式, 这会关闭 dropout 和 batch normalization 等训练时特有的行为
        metric = d2l.Accumulator(2) # 创建一个累加器, 用于累加正确预测的样本数和总样本数
        with torch.no_grad(): # 不需要计算梯度.
            for X, y in data_iter: # 遍历数据集
                y_hat = net(X)
                accuracy_result = accuracy(y_hat, y)
                metric.add(accuracy_result, y.numel()) # y.numel() 返回 y 中元素的总数
        return metric[0] / metric[1] # 返回准确率
# 单个迭代周期的训练函数
def train_epoch(net, train_iter, loss, updater):
    """
    训练模型一个迭代周期
    """
    if isinstance(net, torch.nn.Module):  # 如果 net 是一个 PyTorch 模型
        net.train() # 将模型设置为训练模式
    metric = d2l.Accumulator(3) # 创建一个累加器, 有三个变量, 分别用于累加训练损失, 训练正确的样本数, 总样本数
    for X, y in train_iter: # 遍历训练集
        y_hat = net(X) # 向前传播, 计算预测值
        l = loss(y_hat, y) # 计算损失
        # 反向传播和参数更新
        if isinstance(updater, torch.optim.Optimizer): # 如果 updater 是一个 PyTorch 优化器
            updater.zero_grad() # 清零梯度,防止梯度累加
            # 反向传播, 计算梯度
            l.mean().backward() # 这里对损失取均值, 因为 loss 函数返回的是每个样本的损失值, 需要对其取均值后再进行反向传播
            updater.step() # 更新参数
        else: # 如果是自定义的优化器
            l.sum().backward() # 这里对损失求和, 因为自定义优化器通常需要总损失值
            updater(X.shape[0]) # 传入批量大小, 这里需要根据具体的自定义优化器实现来调整
        # 累加训练损失, 训练正确的样本数, 总样本数
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]

# 训练函数
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    print(f"训练开始, 训练轮数: {num_epochs}")

    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater) # 训练一个周期
        test_acc = evaluate_accuracy(net, test_iter) # 评估在测试集上的准确率
        scheduler.step()
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.3f}, '
              f'train acc {train_metrics[1]:.3f}, '
              f'test acc {test_acc:.3f}')
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    print(f'\n训练完成！')

# 执行训练
train(net, train_iter, test_iter, loss, num_epochs, trainer)
# 7 评估模型, 已经包含在训练函数中
# 训练完成后, 可以使用 evaluate_accuracy 函数在测试集上评估模型的准确率









