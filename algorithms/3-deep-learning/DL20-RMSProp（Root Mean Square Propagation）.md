# DL20 RMSProp（Root Mean Square Propagation）

## 描述

给定一组训练数据，使用RMSProp算法实现线性回归模型。
你的任务是编写一个函数，接受特征矩阵和目标值、学习率和衰减率，并返回训练好的模型参数。
损失函数为均方误差：$MSE = \frac{1}{2m} \sum(y_{pred} - y)^2$
训练方式是批量梯度下降，即每次迭代使用所有样本。
参数更新时，使用1e-8防止分母为0。   

## 输入描述：
- 第一行包含两个整数 m，n，表示训练样本的数量和特征的数量。
- 第二行包含一个整数，表示迭代次数。
- 接下来的 m 行，每行包含 n 个浮点数，表示特征矩阵 X 的一行。
- 接下来一行包含 m 个浮点数，表示目标值 y。
- 最后一行包含两个浮点数，表示学习率和衰减率。

## 输出描述：
- 输出一行，包含 n 个浮点数，表示训练好的模型参数，保留两位小数。

## 示例
```
输入：
9 4
25
0.09 0.25 0.65 0.8
0.6 0.4 0.97 0.82
0.54 0.5 0.76 0.32
0.6 0.89 0.61 0.28
0.14 0.87 0.97 0.14
0.66 0.56 0.79 0.39
0.83 0.3 0.13 0.63
0.18 0.75 0.54 0.9
0.22 0.41 0.11 0.2
0.93 0.8 0.54 0.11 0.86 0.35 0.37 0.17 0.6
0.02 0.93
输出：
0.06 0.16 0.36 0.29
```

## 算法步骤

RMSProp（Root Mean Square Propagation）是一种自适应学习率的优化算法，常用于训练神经网络和其他机器学习模型。其主要思想是通过调整每个参数的学习率来加速收敛，特别是在处理非平稳目标时。以下是RMSProp的基本过程：

1. 初始化参数：
   - 初始化参数 $\theta$ 和均方根平方和 $s$。
   - 初始化学习率 $\alpha$ 和衰减率 $\gamma$。

2. 计算梯度：
   - 计算损失函数对参数 $\theta$ 的梯度 $g$。

3. 更新均方根平方和：
   - 计算均方根平方和 $s = \gamma * s + (1 - \gamma) * g^2$。

4. 更新参数：
   - 更新参数 $\theta = \theta - \alpha * g / \sqrt{s + \epsilon}$，其中 $\epsilon$ 是一个很小的正数（如 $1e-8$），以防止分母为零。

5. 重复步骤 2-4，直到达到预定的迭代次数或满足停止条件。

RMSProp通过动态调整学习率，使得在梯度较大的方向上学习率较小，而在梯度较小的方向上学习率较大，从而加速收敛。


```python
import numpy as np

def rmsprop_linear_regression(X, y, learning_rate, decay_rate, epochs):
    m, n = X.shape
    theta = np.zeros((n, 1))
    s = np.zeros((n, 1))
    for _ in range(epochs):
        y_pred = X@theta
        error = y_pred - y
        gradient = 1/m * X.T@error
        s = decay_rate * s + (1 - decay_rate) * gradient**2
        theta = theta - learning_rate * gradient / np.sqrt(s+1e-8)
    return np.round(theta.flatten(), 2).tolist()


if __name__ == "__main__":
    m, n = map(int, input().split())
    epochs = int(input())
    X = np.array([input().split() for _ in range(m)]).astype(float)
    y = np.array(input().split()).astype(float).reshape(-1, 1)
    learning_rate, decay_rate = map(float, input().split())
    theta = rmsprop_linear_regression(X, y, learning_rate, decay_rate, epochs)
    print(" ".join(map(str, np.round(theta, 2))))
```

RMSProp 的核心思想是动态调整学习率，使其在训练过程中保持稳定且适应不同的参数。具体来说，RMSProp 使用指数加权移动平均来累积梯度的平方，从而避免学习率下降过快。它通过以下步骤实现：

1. **累积梯度平方的指数加权移动平均**：
   使用一个衰减系数（通常记为 $\gamma$）来对梯度的平方进行指数加权移动平均。这使得较新的梯度平方对学习率的影响更大，而较旧的梯度平方的影响逐渐减小。

2. **调整学习率**：
   使用累积的梯度平方的均值来调整学习率，使得学习率在每次迭代中都能自适应地变化。

3. **更新参数**：
   使用调整后的学习率和当前梯度来更新模型参数。
   
### 优点

1. **避免学习率过快下降**：通过指数加权移动平均，RMSProp 避免了 Adagrad 中学习率过快下降的问题。
2. **自适应学习率**：能够根据每个参数的历史梯度动态调整学习率，适应不同参数的更新需求。
3. **训练稳定**：在训练过程中表现出良好的稳定性和收敛速度。

### 缺点

1. **超参数选择**：需要手动调整学习率和衰减系数 $\gamma$，不同的任务可能需要不同的参数设置。
2. **计算复杂度**：需要维护每个参数的梯度平方的累积值，增加了计算和存储开销。
