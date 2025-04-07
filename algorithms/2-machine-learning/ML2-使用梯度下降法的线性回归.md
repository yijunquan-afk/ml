# ML2 使用梯度下降的线性回归
[牛客网](https://www.nowcoder.com/practice/e9f12bb403f44847b44e287d5a71e56c?tpId=379&tqId=11118316&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3DAI%25E7%25AF%2587%26topicId%3D379)

## 描述
编写一个使用梯度下降执行线性回归的 Python 函数。该函数应将 NumPy 数组 X（具有一列截距的特征）和 y（目标）作为输入，以及学习率 alpha 和迭代次数，并返回一个 NumPy 数组，表示线性回归模型的系数。
## 输入描述：
第1行输入X，第2行输入y，第3行输入alpha，第4行输入迭代次数。

## 输出描述：
输出线性回归模型的系数，四舍五入到小数点后四位。返回类型是List类型。

```
输入:
[[1, 1], [1, 2], [1, 3], [1, 4]]
[2, 3, 4, 5]
0.01
1000

输出: 
[0.8678 1.045 ]
```


```python
import numpy as np
def linear_regression_gradient_descent(X, y, alpha, iterations):
    # 补全代码
    m,n = X.shape
    theta = np.zeros((n,1)) # 为了和答案一致
    for _ in range(iterations):
        y_predict = X@theta
        errors = y_predict - y
        discent = X.T@(errors)/m
        theta = theta - alpha * discent
    return np.round(theta.flatten(), 4)

# 主程序
if __name__ == "__main__":
    # 输入矩阵和向量
    matrix_inputx = input()
    array_y = input()
    alpha = input()
    iterations = input()

    # 处理输入
    import ast
    matrix = np.array(ast.literal_eval(matrix_inputx))
    y = np.array(ast.literal_eval(array_y)).reshape(-1,1)
    alpha = float(alpha)
    iterations = int(iterations)

    # 调用函数计算逆矩阵
    output = linear_regression_gradient_descent(matrix,y,alpha,iterations)
    
    # 输出结果
    print(output)


```

    [0.8678 1.045 ]


## 梯度下降求解

梯度下降是一种计算局部最小值的一种方法。梯度下降思想就是给定一个初始值𝜃，每次沿着函数梯度下降的方向移动𝜃：

$$
\theta^{(t+1)} := \theta^{(t)} - \alpha \nabla_{\theta} J(\theta^{(t)})
$$


在梯度为零或趋近于零的时候收敛
$$
J(\theta)=\frac{1}{2n}\sum^n_{i=1}(x_i^T\theta-y_i)^2
$$
对损失函数求偏导可得到 (n个样本，每个样本p维)
$$
x_i=(x_{i,0},...,x_{i,p})^T\\
x_{ij}表示第i个样本的第j个分量\\
\frac{\partial}{\theta_j}\frac{1}{2n}(x_i^T\theta-y_i)^2=\frac{\partial}{\theta_j}\frac{1}{2n}(\sum^p_{j=0}x_{i,j}\theta_j-y_i)^2=\frac{1}{n}(\sum^p_{j=0}x_{i,j}\theta_j-y_i)x_{i,j}=\frac{1}{n}(f(x_i)-y_i))x_{i,j}
\\
\nabla_\theta J=\begin{bmatrix}
\frac{J}{\theta_0}\\
\frac{J}{\theta_1}\\...\\
\frac{J}{\theta_p}
\end{bmatrix}
$$
对于只有一个训练样本的训练组而言，每走一步，𝜃𝑗(𝑗= 0,1,…,𝑝)的更新公式就可以写成：
$$
\theta_j^{(t+1)} := \theta_j^{(t)} - \alpha \frac{\partial}{\partial \theta_j} J(\theta_j^{(t)}) = \theta_j^{(t)} - \alpha \frac{1}{n} (f(x_i) - y_i) x_{i,j}
$$
因此，当有 n 个训练实例的时候（批处理梯度下降算法），该公式就可以写为：
$$
\theta_j^{(t+1)}:=\theta_j^{(t)}-\alpha\frac{1}{n}\sum^n_{i=1}(f(x_i)-y_i)x_{i,j}
$$
这样，每次根据所有数据求出偏导，然后根据特定的步长𝛼，就可以不断更新𝜃𝑗，直到其收敛。当<mark>梯度为0或目标函数值不能继续下降的时候</mark>，就可以说已经收敛，即目标函数达到局部最小值。

具体过程可以归纳如下

> :one: 初始化𝜃（随机初始化）
>
> :two: 利用如下公式更新𝜃
> $$
> \theta_j^{(t+1)}:=\theta_j^{(t)}-\alpha \frac{1}{n}\sum^n_{i=1}(f(x_i)-y_i)x_{i,j}\\
> \theta^{(t+1)}:=\theta^{(t)}-\alpha \frac{1}{n}\sum^n_{i=1}(f(x_i)-y_i)x_{i}
> $$
> 其中α为步长
>
> :three: 如果新的𝜃能使𝐽(𝜃)继续减少，继续利用上述步骤更新𝜃，否则收敛，停止迭代。
>


