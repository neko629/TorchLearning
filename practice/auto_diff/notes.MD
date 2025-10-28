### 函数总结
```
x.requires_grad_(True) 让张量 x 支持自动求导
x.grad 获取张量 x 的梯度 (即导数)
y = f(x) 计算 y 关于 x 的函数
y.backward() 计算 y 关于其依赖的张量的梯度，backward 意为反向传播
x.grad 获取 x 的梯度
x.grad.zero_() 清零 x 的梯度, 以免累加
z = y.detach() 创建一个新的张量 z，与 y 共用内存，但不记录梯度信息, 即脱离计算图
torch.randn(size=()) 生成一个 0 维张量，即标量
```