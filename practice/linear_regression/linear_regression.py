import random
import torch
from d2l import torch as d2l
from torch.utils import data
from torch import nn # nn 是 neural network 的缩写，包含了很多神经网络相关的组件

# 一个训练的完整过程需要包含一下几个组成部分：
# 1. 生成数据集
# 2. 读取数据集
# 3. 定义模型
# 4. 定义损失函数
# 5. 定义优化算法
# 6. 训练模型
# 7. 评估模型

# 生成数据集函数，w 是权重，类型是张量，b 是偏差，类型是标量，num_examples 是样本数，类型是整数
def synthetic_data(w, b, num_examples):
    # 生成特征矩阵 X，服从均值为 0，标准差为 1 的正态分布，形状为 (num_examples, len(w))
    # len 函数返回张量的第 0 维的大小，即张量的行数，权重通常是一个列向量
    # 当 w 是一个列向量时，X 是一个形状为 num_examples 行 len(w) 列的矩阵
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 生成标签 y，matmul 函数用于矩阵乘法，相当于 numpy 的 dot 函数
    # 此计算相当于单层线性神经网络的前向计算
    # 此时 y 的行数等于 num_examples，列数等于 1
    # 因为矩阵乘法的结果是 X 的行数和 w 的列数
    # +b 是广播机制，将标量 b 加到每一行上
    y = torch.matmul(X, w) + b
    # 加入一个噪声，服从均值为 0，标准差为 0.01 的正态分布
    noise = torch.normal(0, 0.01, y.shape)
    y += noise
    # reshape 的时候如果某一维度是 -1，表示这一维度的大小由其他维度推断出来
    # 即 如果一个 2x2的举证 变形成 -1 行 1 列，那么 -1 就表示 2x2/1=4，所以变成 4 行 1 列
    # X 的每一行是一个样本，可能有多个标量特征，y 的每一行是对应样本的标签
    return X, y.reshape(-1, 1) # 将 y 变成列向量返回

# 设置一个权重，如果使用了行向量，那么最后需要把 y reshape 成列向量
# 如果使用列向量，那么 y 本身就是列向量
true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, num_examples=1000)
# print('features:', features[0],'\nlabel:', labels[0])
# print('features shape:', features.shape)
# print('labels shape:', labels.shape)
#
# # 画图
# d2l.set_figsize() # 设置图的大小
# d2l.plt.scatter(features[:, 1] # (1) 表示取第二列
#                 , labels
#                 , 1) # 点的大小为 1
# d2l.plt.show()

# 截止到这里，我们已经生成了一个数据集
# 总结一下，生成数据集，需要指定权重，偏差和样本数量，生成的结果包括特征矩阵和标签向量，要保证特征和标签的行数一致


# 读取数据集
def data_iter(batch_size, features, labels):
    # 获取样本数量
    num_examples = len(features)
    # 生成一个包换所有元素下标的 list
    indices = list(range(num_examples))
    # 打乱
    random.shuffle(indices)
    # 每次取 batch_size 个样本
    for i in range(0, num_examples, batch_size): # 这里 range 的第一个参数是开区间，第二个参数是闭区间，第三个参数是步长
        start_idx = i
        end_idx = min(start_idx + batch_size, num_examples)
        # 此次取出的样本下标
        batch_indices = torch.tensor(indices[start_idx:end_idx])
        # 使用 yeild 返回当前批次的特征和标签，并保存当前函数的状态
        yield features[batch_indices], labels[batch_indices]

batch_size = 10 # 批量大小
# 遍历第一步生成的数据集
#gen = data_iter(batch_size, features, labels) # 生成一个迭代器
# for X, y in gen:
#     print('X:', X, '\ny:', y)

# 读取数据集的代码已经完成
# 总结一下，读取数据集需要指定批量大小，特征矩阵和标签向量
# 读取数据集的结果是一个生成器，每次迭代返回一个批次的特征和标签

# 定义模型
# 定义模型即设定模型的参数，包括权重和偏差
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True) # 2 行 1 列的张量, 特征需要和他相乘，所以特征必须是 2 列
b = torch.zeros(1, requires_grad=True) # zeros 表示所有元素都是 0 的张量，第一个参数 1 表示张量的形状是 1 维，只有一个元素，如果需要 m 行 n 列的张量，则需要传入 (m, n)
# 这里的 w 和 b 都是需要梯度的，因为我们需要通过梯度下降来更新它们，是我们最终需要优化的参数

# 定义线性回归模型, 即由特征、权重和偏差计算预测值
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# 定义损失函数
# 这里使用均方误差作为损失函数, 这个值越大表示预测值和真实值的差距越大，我们的目标是最小化这个值
def squared_loss(y_hat, y): # y_hat 是预测值，y 是真实值
    # 这里的 y_hat 和 y 的形状都是 (batch_size, 1)
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # reshape 保证 y 和 y_hat 形状一致

# 定义优化算法
# 这里使用小批量随机梯度下降作为优化算法
def sgd(params, lr, batch_size): # params 是模型的参数，即 w 和 b，lr 是学习率, 表示每次更新的步长，batch_size 是批量大小
    with torch.no_grad(): # 这里不要计算梯度，使用 with 可以在执行代码块时临时设置，执行完后回复
        for param in params:
            # 更新参数
            grad = param.grad # 获取参数的梯度, 注意 param.grad 是一个张量
            param -= lr * grad / batch_size # 注意这里要除以 batch_size，因为我们计算的梯度是一个批次的梯度和
            param.grad.zero_() # 将梯度清零，避免累加

# 训练模型
# 设置超参数
lr = 0.03
num_epochs = 10
net = linreg # 使用我们定义的线性回归模型, 即神经网络
loss = squared_loss # 使用我们定义的均方误差损失函数

# 每一轮次需要做的事情是：
# 1. 通过数据迭代器获取一个批次的数据
# 2. 通过模型计算预测值
# 3. 通过损失函数计算损失值
# 4. 对损失值进行反向传播，计算梯度
# 5. 将参数传入优化算法，更新参数
def full_training_loop():
    for epoch in range(num_epochs): # 迭代轮次
        for X, y in data_iter(batch_size, features, labels): # 迭代每个批次
            y_hat = net(X, w, b) # 计算预测值
            l = loss(y_hat, y) # 计算损失值
            l.sum().backward() # 反向传播，计算梯度，注意这里要对损失值求和，因为 loss 返回的是一个批次的损失值
            sgd([w, b], lr, batch_size) # 使用小批量随机梯度下降更新参数
        with torch.no_grad(): # 不计算梯度
            train_l = loss(net(features, w, b), labels) # 计算所有样本的损失值
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}') # 打印当前轮次的平均损失值

# 使用 pytorch 内置的组件来实现同样的功能
# 读取数据集使用 DataLoader
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays) # * 表示解包，将 data_arrays 里面的元素依次传入 TensorDataset
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # shuffle 表示是否打乱数据. 这里我们在训练时打乱数据，在测试时不打乱数据
# 每次都是全量训练，仅仅是打乱数据
batch_size = 10
# 获取 dataloader
data_loader = load_array((features, labels), batch_size)
# 生成一个迭代器
data_iter = iter(data_loader)
#print(next(data_iter)) # next 用于获取迭代器的下一个元素
linear2 = nn.Linear(2, 1) # 得到一个线性回归模型，2 表示输入特征的维度，1 表示输出特征的维度
net2 = nn.Sequential(linear2) # 使用 Sequential 将线性回归模型包装成一个神经网络
# 初始化模型参数
net2[0].weight.data.normal_(0, 0.01) # 使用正态分布初始化权重
net2[0].bias.data.fill_(0) # 使用 0 初始化偏差
loss2 = nn.MSELoss() # 使用均方误差作为损失函数
# 使用小批量随机梯度下降作为优化算法
trainer = torch.optim.SGD(net2.parameters(), lr=0.03) # net2.parameters() 返回模型的所有参数, lr 是学习率

def pytorch_training_loop():
    for epoch in range(num_epochs):
        for X, y in data_loader:
            y_hat = net2(X)
            l = loss2(y_hat, y)
            trainer.zero_grad() # 清零梯度
            l.backward() # 反向传播
            trainer.step() # 更新参数, 这里不需要传入参数，因为 trainer 已经知道要更新哪些参数
        with torch.no_grad():
            train_l = loss2(net2(features), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


pytorch_training_loop()

# 可以看到使用 pytorch 内置的组件实现的训练过程和我们手动实现的训练过程是一样的
# 总结一下，使用 pytorch 内置的组件可以大大简化代码量
# 可以直接更加便利的部分包括：
# 1. 读取数据集： 使用 DataLoader
# 2. 定义模型： 使用 nn.Linear 定义线性回归模型
# 3. 定义损失函数： 使用 nn.MSELoss 定义均方误损失函数
# 4. 定义优化算法： 使用 torch.optim.SGD 定义小批量随机梯度下降优化算法
# 5. 训练模型： 使用 trainer.step() 更新参数
# 这些组件都是 pytorch 提供的高层接口，封装了很多底层细节，使得我们可以更加专注于模型的设计和训练过程


