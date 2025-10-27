### 函数总结
```
a = torch.arange(x) 生成一个 0 到 x - 1 的一维张量
a.shape 输出 a 各维度的大小
a.numel 输出 a 里元素的数量
a = a.reshape(3, 4) 把 a 重新排列成一个 3 行 4 列的张量
a = torch.zeros(2, 3, 4) 生成一个三维张量，各维度长度为 2,3,4，每个元素的值都是 0
a = torch.ones(2, 3, 4) 生成一个三维张量，各维度长度为 2,3,4，每个元素的值都是 1
a = torch.tensor([1,2,...]) 直接定义一个张量
torch.cat((x, y), dim = 0) 行维度拼接张量
torch.cat((x, y), dim = 1) 列维度拼接张量
a.sum() 返回各个元素的和
id(a) 取内存地址
b = a.numpy() 转为 numpy 数组
c = torch.tensor(b) 再转回 torch 数组
pandas 的操作
pd.read_csv('file.csv') 读取 csv 文件为 DataFrame
data.iloc[行, 列] 通过行列索引访问 DataFrame 元素
data.loc[行标签, 列标签] 通过行列标签访问 DataFrame 元素
inputs.fillna() 填充缺失值
inputs.mean() 计算均值 (numeric_only = 1 时只计算数值列)
pd.get_dummies(data, columns = [...]) 独热编码
torch.tensor(inputs.values) 将 DataFrame 转为张量
torch.sum(a, dim = 0) 按行求和，结果是每列的和
torch.sum(a, dim = 1) 按列求和，结果是每行的和
bb.view(-1, num_features) 将 bb 重塑为行数不变，列数为 num_features 的张量
```

### 概念总结
#### 广播机制
```
单行或者单列的张量，在计算需要时会自动复制自身的行或者列升为二维
```
#### 访问元素
```
a[1,2] 准确的数字表示访问单个元素
a[0:2, 3] 冒号表示范围，这里是 0 行和 1 行，前开后闭
a[:2, 3] and a[1: , 2] 、
冒号前方省略表示起始在开头，冒号后省略表示结束至末尾
```
#### 内存分配和原地操作
```
a = a + 1 会改变 id(a)
a += 1 不会改变 id(a)
f[:] = a + 1 元素赋值不会改变 id(f)
```