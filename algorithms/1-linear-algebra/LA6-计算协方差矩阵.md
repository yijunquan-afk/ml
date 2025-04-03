# LA6 计算协方差矩阵

题目来源：牛客网

## 描述
编写一个 Python 函数来计算给定向量集的协方差矩阵。该函数应该采用一个列表列表，其中每个内部列表代表一个特征及其观察结果，并返回一个协方差矩阵。

## 输入描述：
输入给定向量集。

## 输出描述：
输出协方差矩阵。

# 示例1

输入：
```
[[7, 8, 9], [10, 11, 12]]
```
输出：
```
[[1.0, 1.0], [1.0, 1.0]]
```

$$
\Sigma_{ij} = \frac{1}{m-1} \sum_{k=1}^{m} (X_{ik} - \mu_i)(X_{jk} - \mu_j)
$$


```python
import numpy as np

def calculate_covariance_matrix(vectors):
    # 补全代码
    features = len(vectors)
    samples = len(vectors[0])
    samples_mean = [sum(sublist)/len(sublist) for sublist in vectors]
    covariance_matrix = [[0 for _ in range(features)] for _ in range(features)]
    for i in range(features):
        for j in range(i, features):
            covariance =sum ((vectors[i][k] - samples_mean[i]) * (vectors[j][k]-samples_mean[j]) for k in range(samples)) / (samples-1)
            covariance_matrix[i][j] = covariance
            covariance_matrix[j][i] = covariance
    # return np.cov(vectors, rowvar=True).tolist() # 精度不太一样
    return covariance_matrix
 

# 主程序
if __name__ == "__main__":
    # 输入
    ndarrayA = input()

    # 处理输入
    import ast
    A = ast.literal_eval(ndarrayA)

    # 调用函数计算
    output = calculate_covariance_matrix(A)
    
    # 输出结果
    print(output)
```

## 协方差
协方差（Covariance）是衡量两个随机变量之间线性关系的统计量。它描述了两个变量如何一起变化。

- **正值**：如果协方差为正，说明当 $X$增大时，$Y$也倾向于增大；当 $X$减小时，$Y$也倾向于减小。这表示 $X$和 $Y$之间存在**正相关关系**。
- **负值**：如果协方差为负，说明当 $X$增大时，$Y$倾向于减小；当 $X$减小时，$Y$倾向于增大。这表示 $X$和 $Y$之间存在**负相关关系**。
- **零值**：如果协方差为零，说明 $X$和 $Y$之间**没有线性关系**。但这并不一定意味着 $X$和 $Y$完全独立，因为协方差只能衡量线性关系，不能捕捉非线性关系。

## 协方差矩阵

协方差矩阵是一个用来描述多维随机变量之间协方差的矩阵。对于一个$n$维随机变量$\mathbf{X} = (X_1, X_2, \ldots, X_n)$，其协方差矩阵$\Sigma$是一个$n \times n$的矩阵，其中第$i$行第$j$列的元素是$X_i$和$X_j$的协方差，即：

$\Sigma_{ij} = \text{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]$

其中$\mu_i = E[X_i]$是$X_i$的期望值。

协方差矩阵具有以下性质：
- 它是一个对称矩阵，即$\Sigma_{ij} = \Sigma_{ji}$。
- 它的对角线元素是各个随机变量的方差，即$\Sigma_{ii} = \text{Var}(X_i)$。

如果用数据样本计算协方差矩阵，假设我们有$m$个样本点$\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_m$，每个样本点是一个$n$维向量，那么样本协方差矩阵$\Sigma$可以表示为：

$\Sigma = \frac{1}{m-1} \sum_{k=1}^{m} (\mathbf{x}_k - \mathbf{\mu})(\mathbf{x}_k - \mathbf{\mu})^T$

其中$\mathbf{\mu}$是样本均值向量，定义为：

$\mathbf{\mu} = \frac{1}{m} \sum_{k=1}^{m} \mathbf{x}_k$

## 协方差矩阵的具体案例说明

假设我们有三个随机变量 $X_1, X_2, X_3$，并且有以下4个样本观测值：

| $X_1$ | $X_2$ | $X_3$ |
|---------|---------|---------|
| 2       | 3       | 5       |
| 4       | 6       | 8       |
| 6       | 9       | 11      |
| 8       | 12      | 14      |

### 1. 计算每个随机变量的均值

首先，计算每个随机变量的均值：

$$
\mu_1 = \frac{2 + 4 + 6 + 8}{4} = 5
$$

$$
\mu_2 = \frac{3 + 6 + 9 + 12}{4} = 7.5
$$

$$
\mu_3 = \frac{5 + 8 + 11 + 14}{4} = 10
$$

### 2. 计算每个随机变量与其均值的偏差

接下来，计算每个样本点与均值的偏差：

| $X_1 - \mu_1$ | $X_2 - \mu_2$ | $X_3 - \mu_3$ |
|-----------------|-----------------|-----------------|
| $2 - 5 = -3$  | $3 - 7.5 = -4.5$ | $5 - 10 = -5$ |
| $4 - 5 = -1$  | $6 - 7.5 = -1.5$ | $8 - 10 = -2$ |
| $6 - 5 = 1$   | $9 - 7.5 = 1.5$  | $11 - 10 = 1$ |
| $8 - 5 = 3$   | $12 - 7.5 = 4.5$ | $14 - 10 = 4$ |

### 3. 计算协方差矩阵

协方差矩阵的每个元素 $\Sigma_{ij}$ 是通过计算 $(X_i - \mu_i)$ 和 $(X_j - \mu_j)$ 的乘积的平均值来得到的。具体公式为：

$$
\Sigma_{ij} = \frac{1}{m-1} \sum_{k=1}^{m} (X_{ik} - \mu_i)(X_{jk} - \mu_j)
$$

其中 $m$ 是样本数量，这里是4。

#### 计算 $\Sigma_{11}$（$X_1$ 和 $X_1$ 的协方差，即 $X_1$ 的方差）：

$$
\Sigma_{11} = \frac{1}{4-1} [(-3)^2 + (-1)^2 + 1^2 + 3^2] = \frac{1}{3} [9 + 1 + 1 + 9] = \frac{20}{3} \approx 6.67
$$

#### 计算 $\Sigma_{12}$（$X_1$ 和 $X_2$ 的协方差）：

$$
\Sigma_{12} = \frac{1}{4-1} [(-3)(-4.5) + (-1)(-1.5) + 1 \times 1.5 + 3 \times 4.5] = \frac{1}{3} [13.5 + 1.5 + 1.5 + 13.5] = \frac{30}{3} = 10
$$

#### 计算 $\Sigma_{13}$（$X_1$ 和 $X_3$ 的协方差）：

$$
\Sigma_{13} = \frac{1}{4-1} [(-3)(-5) + (-1)(-2) + 1 \times 1 + 3 \times 4] = \frac{1}{3} [15 + 2 + 1 + 12] = \frac{30}{3} = 10
$$

#### 计算 $\Sigma_{22}$（$X_2$ 和 $X_2$ 的协方差，即 $X_2$ 的方差）：

$$
\Sigma_{22} = \frac{1}{4-1} [(-4.5)^2 + (-1.5)^2 + 1.5^2 + 4.5^2] = \frac{1}{3} [20.25 + 2.25 + 2.25 + 20.25] = \frac{45}{3} = 15
$$

#### 计算 $\Sigma_{23}$（$X_2$ 和 $X_3$ 的协方差）：

$$
\Sigma_{23} = \frac{1}{4-1} [(-4.5)(-5) + (-1.5)(-2) + 1.5 \times 1 + 4.5 \times 4] = \frac{1}{3} [22.5 + 3 + 1.5 + 18] = \frac{45}{3} = 15
$$

#### 计算 $\Sigma_{33}$（$X_3$ 和 $X_3$ 的协方差，即 $X_3$ 的方差）：

$$
\Sigma_{33} = \frac{1}{4-1} [(-5)^2 + (-2)^2 + 1^2 + 4^2] = \frac{1}{3} [25 + 4 + 1 + 16] = \frac{46}{3} \approx 15.33
$$

### 4. 构建协方差矩阵

将上述计算结果填入矩阵中，得到协方差矩阵：

$$
\Sigma = \begin{pmatrix}
6.67 & 10 & 10 \\
10 & 15 & 15 \\
10 & 15 & 15.33
\end{pmatrix}
$$

这个矩阵描述了三个随机变量之间的协方差关系。对角线上的元素是每个变量的方差，非对角线上的元素是变量之间的协方差。
