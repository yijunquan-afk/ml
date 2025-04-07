# ML1 使用正规方程的线性回归

## 描述
编写一个使用正规方程执行线性回归的 Python 函数。
函数输入是一个矩阵 X（特征）和向量 y（目标），返回线性回归模型的系数。
最后的答案四舍五入保留小数点后四位。

## 输入描述：
第1行输入矩阵 X，第2行输入向量 y。

## 输出描述：
输出线性回归模型的系数。函数返回类型是列表类型，第一个是权重，第二个是偏置。


```
输入: 
[[1, 1], [1, 2], [1, 3]]
[2, 2, 3]

输出:

[1.3333, 0.5]
```


```python
import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # 实现代码
    theta = (np.linalg.inv(x.T@x)@x.T@y).T.round(4).reshape(-1).tolist()
    return theta


if __name__ == "__main__":
    import ast
    x = np.array(ast.literal_eval(input()))
    y = np.array(ast.literal_eval(input())).reshape(-1, 1)

    # Perform linear regression
    coefficients = linear_regression_normal_equation(x, y)

    # Print the coefficients
    print(coefficients)

```

## 正规方程法求解

将训练数据表示成矩阵形式
$$
\mathbf{X} = 
\begin{bmatrix}
\mathbf{x}_1^T \\
\mathbf{x}_2^T \\
\vdots \\
\mathbf{x}_n^T \\
\end{bmatrix}
=
\begin{bmatrix}
x_{1,0} & x_{1,1} & \cdots & x_{1,p} \\
x_{2,0} & x_{2,1} & \cdots & x_{2,p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n,0} & x_{n,1} & \cdots & x_{n,p} \\
\end{bmatrix}
\quad
\mathbf{Y} = 
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n \\
\end{bmatrix}
$$

$$
\mathbf{x_1}^T=[1,x_{1,1},x_{1,2},...,x_{1,p}]
$$

损失函数$J(\theta)$可变为
$$
J(\theta)=\frac{1}{2}||Y-\hat Y||^2=\frac{1}{2}||Y-X\theta||^2
$$
使用矩阵表达形式转化损失函数
$$
\begin{equation}
\begin{split}
J(\theta)&=\frac{1}{2}||Y-X\theta||^2\\
&=\frac{1}{2}(X\theta-Y)^T(X\theta-Y)\\
&=\frac{1}{2}(\theta ^TX^TX\theta-2Y^TX\theta+Y^TY)(利用了a^Tb=b^Ta求导结果)
\end{split}
\end{equation}
$$
最小化损失函数𝐽(𝜃)，可通过令梯度= 𝟎（𝑝+1维的零向量）实现:

<img src="https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721704.png" alt="image-20220317095048551" style="zoom:67%;"/>

利用公式
$$
\frac{\partial x^TBx}{\partial x}=(B+B^T)x\\
\frac{\partial x^Ta}{\partial x}=\frac{\partial a^Tx}{\partial x}=a
$$
可得到
$$
\nabla_\theta J(\theta)=X^TX\theta-(Y^TX)^T=X^TX\theta-X^TY=0
$$
:mag:<mark>正规方程为：</mark>$X^TX\theta=X^TY$，得到解：$\theta^*=(X^TX)^{-1}X^TY$(假设$X^TX$可逆)


