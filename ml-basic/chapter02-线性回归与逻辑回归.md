[TOC]

# 第二课线性回归与逻辑回归

## 线性回归

机器学习中的线性回归

![image-20220316224043829](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721669.png)

当输入变量𝑥的特征/属性数目变为𝑝时，线性回归模型表示为：
$$
f(x)=\theta_0+\theta_1x_1+...+\theta_px_p
$$
向量积表达形式为
$$
f(\pmb x)=\sum^p_{i=0}\theta_ix_i=\theta^Tx=x^T\theta\\
\theta=\begin{bmatrix}
\theta_0\\
\theta_1\\...\\
\theta_p
\end{bmatrix}
\qquad x^T=[(x_0=1),x_1,x_2,...,x_p]
$$
多元线性回归的目标是选择一个最优的𝜃，使得预测输出𝑓(𝑥)与实际输出𝑦之间的差别尽可能小

使用均方误差（MSE, Mean Squred Error）和来衡量损失(假设一共有n个样本)
$$
J(\theta)=\frac{1}{2}\sum^n_{i=1}(x_i^T\theta-y_i)^2
$$
> 没有对误差进行归一化，因此损失函数的值会随着样本数量 *n* 的增加而增大。它更多地用于优化过程中，而不是直接用于模型性能的评估。
>
> 标准的MSE是 $1/n$

线性回归目标：求解参数𝜃使损失函数$𝐽(𝜃)$的值达到最小。

几何意义：试图找到一个超平面，使所有样本到超平面上的欧式距离之和最小

**隐含假设**：误差（预测$𝜃^𝑇 𝑥_𝑖 $与真实输出𝑦𝑖 差异）服从服从**独立同分布的高斯分布**

### 梯度下降法求解

#### 复习梯度

梯度的本意是一个向量（矢量），表示某一函数在该点处的方向导数沿着该方向取得最大值，即函数在该点处沿着该方向（此梯度的方向）变化最快，变化率最大（为该梯度的模）。

设二元函数$z=f(x,y)$在平面区域D上具有一阶连续偏导数，则对于每一个点P（x，y）都可定出一个向量,该函数就称为函数$z=f(x,y)$在点P（x，y）的梯度，记作gradf（x，y）或$\nabla f(x,y)$,即有：

$$
gradf(x,y)=\nabla f(x,y)=\{\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}\}=f_x(x,y)\vec{i}+f_y(x,y)\vec{j}
$$
其中$\nabla = \frac{\partial}{\partial x}\vec{i}+\frac{\partial}{\partial y}\vec{j}$称为（二维的）向量[微分算子](https://baike.baidu.com/item/微分算子)或[Nabla算子](https://baike.baidu.com/item/Nabla算子/22786343)，$\nabla f = \frac{\partial f}{\partial x}\vec{i}+\frac{\partial f}{\partial y}\vec{j}$

#### 梯度下降求解

梯度下降是一种计算局部最小值的一种方法。梯度下降思想就是给定一个初始值𝜃，每次沿着函数梯度下降的方向移动𝜃：

<img src="https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721698.png" alt="image-20220316231646546" style="zoom: 67%;"/>

$$
\theta^{(t+1)} := \theta^{(t)} - \alpha \nabla_{\theta} J(\theta^{(t)})
$$


在梯度为零或趋近于零的时候收敛
$$
J(\theta)=\frac{1}{2}\sum^n_{i=1}(x_i^T\theta-y_i)^2
$$
对损失函数求偏导可得到 (n个样本，每个样本p维)
$$
x_i=(x_{i,0},...,x_{i,p})^T\\
x_{ij}表示第i个样本的第j个分量\\
\frac{\partial}{\theta_j}\frac{1}{2}(x_i^T\theta-y_i)^2=\frac{\partial}{\theta_j}\frac{1}{2}(\sum^p_{j=0}x_{i,j}\theta_j-y_i)^2=(\sum^p_{j=0}x_{i,j}\theta_j-y_i)x_{i,j}=(f(x_i)-y_i))x_{i,j}
\\
\nabla_\theta J=\begin{bmatrix}
\frac{J}{\theta_0}\\
\frac{J}{\theta_1}\\...\\
\frac{J}{\theta_p}
\end{bmatrix}
$$
对于只有一个训练样本的训练组而言，每走一步，𝜃𝑗(𝑗= 0,1,…,𝑝)的更新公式就可以写成：
$$
\theta_j^{(t+1)} := \theta_j^{(t)} - \alpha \frac{\partial}{\partial \theta_j} J(\theta_j^{(t)}) = \theta_j^{(t)} - \alpha (f(x_i) - y_i) x_{i,j}
$$
因此，当有 n 个训练实例的时候（批处理梯度下降算法），该公式就可以写为：
$$
\theta_j^{(t+1)}:=\theta_j^{(t)}-\alpha\sum^n_{i=1}(f(x_i)-y_i)x_{i,j}
$$
这样，每次根据所有数据求出偏导，然后根据特定的步长𝛼，就可以不断更新𝜃𝑗，直到其收敛。当<mark>梯度为0或目标函数值不能继续下降的时候</mark>，就可以说已经收敛，即目标函数达到局部最小值。

具体过程可以归纳如下

> :one: 初始化𝜃（随机初始化）
>
> :two: 利用如下公式更新𝜃
> $$
> \theta_j^{(t+1)}:=\theta_j^{(t)}-\alpha\sum^n_{i=1}(f(x_i)-y_i)x_{i,j}
> $$
> 其中α为步长
>
> :three: 如果新的𝜃能使𝐽(𝜃)继续减少，继续利用上述步骤更新𝜃，否则收敛，停止迭代。
>
> > 如何判断收敛？<img src="https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721701.png" alt="image-20220316233856578" style="zoom:67%;"/>

**学习率α的影响**

> 小的𝛼值可以帮助找到最优解，但是收敛速度很慢
>
> 大的𝛼值一开始会使损失下降的较快，但会导致“锯齿”现象。每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛
>
> ![image-20220317092324162](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721671.png)

#### 特征缩放

在我们面对多维特征问题的时候，我们要保证这些特征都具有相近的尺度，这将帮助梯度下降算法更快地收敛。以房价问题为例，假设我们使用两个特征，房屋的尺寸和房间的数量，尺寸的值为0-2000平方英尺，而房间数量的值则是0-5，以两个参数分别为横纵坐标，绘制代价函数的等高线图能，看出图像会显得很扁，梯度下降算法需要非常多次的迭代才能收敛。

解决的方法是尝试将所有特征的尺度都尽量缩放到-1到1之间。最简单的方法最小最小缩放和平均值缩放。

#### 随机梯度下降法

梯度下降算法有一个最大的问题：每次更新，都要利用所有的数据，当数据量十分大的时候，这会使效率变得特别低。因此，又出现了增强梯度下降（随机梯度下降算法），<mark>每次只用训练集中的一个数据</mark>更新𝜃，即：
$$
\theta_j^{(t+1)}:=\theta_j^{(t)}-\alpha(f(x_i)-y_i)x_{i,j}
$$
在深度神经网络学习中，<mark>小批量梯度下降(mini-batch gradient decent)应</mark>用的非常广泛：把数据分为若干个批，按批来更新参数。一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。



### 正规方程法求解

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
<img src="https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721702.png" alt="image-20220317093150438" style="zoom:67%;"/>

损失函数$𝐽(𝜃)$可变为
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

对θ再求一次梯度可得到$\nabla_\theta^2J(\theta)=X^TX$，这是<mark>半正定的</mark>，因此,若$\nabla_{\theta}J(\theta^*)=0$，则$J(\theta)$在𝜃∗处取到最小值

> :bookmark: 正定矩阵与半正定矩阵
>
> **正定矩阵**：设M是n阶方阵，如果对任何非零向量z，都有$z^TMz>0$，其中$z^T$表示z的转置，就称M为正定矩阵。正定矩阵的行列式恒为正。
>
> **半正定矩阵**：设M是n阶方阵，如果对任何非零向量z，都有$z^TMz\ge0$，其中$z^T$表示z的转置，就称M为半正定矩阵。矩阵与其转置的矩阵积是一个半正定矩阵。

当训练样本的数目𝑛大于训练样本的维度𝑝+1时，$𝑋^𝑇𝑋$在实际中通常是可逆的

当$𝑋^𝑇𝑋$可逆时，表明该矩阵是正定矩阵，求得的参数𝜃∗是全局最优解

> 矩阵可逆的充要条件：行列式不为0

当$𝑋^𝑇𝑋$不可逆时，可解出多个𝜃。可使用正则化给出一个“归纳偏好”解

> 保留所有特征，但减少θ的大小，通常使其接近于0

### 三种求解方法的比较

| 方法名称     | 表达式                                                       | 优点               | 缺点                                               |
| ------------ | ------------------------------------------------------------ | ------------------ | -------------------------------------------------- |
| 正规方程     | $\theta^*=(X^TX)^{-1}X^TY$                                   | 有闭式解，实现简单 | 计算量大：需计算矩阵乘积及矩阵的逆，矩阵有可能奇异 |
| 梯度下降     | $\theta_j^{(t+1)}:=\theta_j^{(t)}-\alpha\sum^n_{i=1}(f(x_i)-y_i)x_{i,j}$ | 收敛、实现简单     | 通常收敛速度较慢                                   |
| 随机梯度下降 | $\theta_j^{(t+1)}:=\theta_j^{(t)}-\alpha(f(x_i)-y_i)x_{i,j}$ | 迭代成本低         | 不一定收敛到最优解                                 |

> :bookmark: 奇异矩阵：对应的行列式等于0的方阵
>
> 如果A(n×n)为奇异矩阵（singular matrix）<=> A的秩Rank(A)<n.

## 多项式回归

线性回归并不意味着我们只能处理线性关系

![image-20220317103703134](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721675.png)

![img](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721676.png)

$$
\theta^*=(X^TX)^{-1}X^T\vec y\qquad\Rightarrow\qquad \theta^*=(\varphi^T\varphi)^{-1}\varphi^T\vec y\\
\varphi(x):=[1,x,x^2]^T
$$
一旦基给定，参数𝜃的学习仍然是线性问题（𝜃中包含的分量个数发生了变化）

**多项式回归包含两个主要问题**：

:one: <mark>参数学习</mark>

> 线性回归几乎一样，仅仅是将𝑥→𝜑(𝑥)
>
> 参数仍然是基函数的线性组合

:two: <mark>确定多项式的阶数（模型评估）</mark>

> 选择不好会引起欠拟合(under-fitting)或过拟合问题(over-fitting)
>
> **欠拟合：训练误差较大；过拟合：训练误差几乎为0**
>
> ![image-20220317104922123](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721677.png)

## 模型评估

:question: 对于给定的标记数据，既要训练，又要测试，该如何做

### 留出法

:one: 随机挑选一部分标记数据作为测试集（空心点）

:two: 其余的作为训练集（实心点），计算回归模型

:three: 使用测试集对模型评估：MSE =2.4

![image-20220317110813810](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721678.png)

:book: 测试集不能太大，也不能太小。2 <= n:m <=4

### 交叉验证

问题：没有足够的数据留出一个测试集

方案：每一个数据既被当作训练样本也被当作测试样本

![image-20220317111039287](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721679.png)

**线性回一维例子**：3折交叉验证

> 将数据分成紫、绿、蓝三个子集
>
> 对于蓝色划分：使用紫、绿数据训练线性模型，使用蓝色数据计算均方误差
>
> 对于绿色划分：使用紫、蓝数据训练线性模型，使用绿色数据计算均方误差
>
> 对于紫色划分：使用绿、蓝数据训练线性模型，使用紫色数据计算均方误差
>
> ![image-20220317111411016](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721681.png)

## 性能度量

### 均方误差

对于线性回归模型，其损失函数使用平方差误差来度量
$$
J_{train}(\theta)=\frac{1}{2}\sum^n_{i=1}(x_i^T\theta-y_i)^2
$$
最小化平方和误差，利用正规方程得到
$$
\theta^*=argminJ_{train}(\theta)=(X_{train}^TX_{train})^{-1}X_{train}^T\vec{Y}_{train}
$$

> $argminJ_{train}(\theta)$就是使$J_{train}(\theta)$达到最小值时的θ的取值

在测试集上报告 MSE（mean square error）误差

![image-20220318151425256](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721682.png)

### 错误率与精度

分类任务性能度量：错误率与精度

**错误率**是分类错误的样本数占样本总数的比例

**精度**是分类正确的样本数占样本总数的比例

### accuracy、precision与recall

对于二分类问题，有分类结果混淆矩阵

![image-20220303111021487](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721684.png)

> True Positive，即正确预测出的正样本个数
>
> False Positive，即错误预测出的正样本个数（本来是负样本，被我们预测成了正样本）
>
> True Negative，即正确预测出的负样本个数
>
> False Negative，即错误预测出的负样本个数（本来是正样本，被我们预测成了负样本）

**准确率(Accuracy)**＝(TP + TN)/总样本

> 定义是: 对于给定的测试数据集，分类器正确分类的样本数与总样本数之比

**精确率(Precision)**＝ TP /(TP + FP)，又称为查准率

> 预测为正的样本中有多少是真正的正样本，是**针对我们预测结果**而言的

**召回率(Recall)**＝ TP /(TP + FN)，又称为查全率

> 它表示：样本中的正例有多少被预测正确了，是**针对我们原来的样本**而言的

**F1-score**：F1-score 是精确率和召回率的调和平均数，计算公式为：
$$
\text{F1-score}=2×\cfrac{Precision+Recall}{Precision×Recall}
$$


F1-score 的取值范围是 [0, 1]，值越高表示模型的分类效果越好。F1-score 考虑了精确率和召回率的平衡，因此**在正负样本分布不均的情况下，F1-score 是一个比准确率（Accuracy）更合适的评价指标**。



## 逻辑回归

线性回归模型中，假设输出标记𝑦是连续值。如果输出标记𝑦= 0,1 （即二分类任务），怎么将分类标记与线性回归模型联系起来？

最理想的情况—单位阶跃函数
$$
\hat y=f_{\theta}(z)=\begin{cases}
0,&z<0\\
0.5,&z=0\\
1,&z>0
\end{cases}
$$
预测值小于0判为反例，大于0判为正例，临界值0任意判别

但单位阶跃函数不可导（很难学习参数𝜃）

这时可以使用**逻辑函数（logistic/sigmoid function，对数几率函数）**来进行替代

> 单调递增、任意阶可微
>
> 输出值在𝑧= 0 附近变化很快

$$
y=\frac{1}{1+e^{-z}}
$$

<img src="https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721705.png" alt="image-20220318153401929" style="zoom:67%;" />

不难发现$f_{\theta}(x)\subseteq(0,1)$,因此可将$f_{\theta}(x)$视为样本 𝑥 作为正例的可能性， 则$1-f_{\theta}(x)$ 是反例可能性

> $\hat y$越接近1，认为是正例的可能性越大
>
> $\hat y$越接近0，认为是反例的可能性越大

![image-20220318154143225](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721685.png)

合并以上两个公式，可得到伯努利公式
$$
P(y|x;\theta)=(f_{\theta}(x))^y(1-f_{\theta}(x))^{1-y}
$$
logistic回归可以被看做一种**概率模型**，且y发生的概率与 回归参数Θ有关

问题转化为求Logistic回归的最佳系数。

### 极大似然估计

极大似然估计法是参数点估计中的一个重要方法,它是**利用总体的分布函数的表达式及样本所提供的信息建立未知参数估计量的方法**.它的直观解释为:设一个随机试验有若干个可能结果，如果在某次试验中，某一结果A出现了，那么，我们认为在诸多可能的试验条件中，应该是使事件 A 发生的概率为最大的那一种条件，这就是极大似然估计的基本思想。即从**若干个参数集合中选取使样本出现概率最大的参数**。

**定义**

设总体X为连续型随机变量，其密度函数为$f(x;\theta_1,\theta_2,\theta_3,...,\theta_t)$,其中$\theta_1,\theta_2,\theta_3,...,\theta_t$为位置参数，则如下公式为参数$\theta_1,\theta_2,\theta_3,...,\theta_t$的**似然函数**
$$
L(\theta_1,\theta_2,...,\theta_t)=\prod_{i=1}^nf(x_i;\theta_1,\theta_2,...,\theta_t)
$$
如果$\hat{\theta_j}(x_1,x_2,...,x_n),j=1,2,...,t$满足使似然函数，则称其为${\theta_j}(x_1,x_2,...,x_n),j=1,2,...,t$的极大似然估计值，相应的统计量$\hat{\theta_j}(X_1,X_2,...,X_n),j=1,2,...,t$为极大似然估计量

极大似然估计法就是当得到样本值$(x_1,x_2,x_3,...,x_n)$时选取$(\hat{\theta_1},\hat{\theta_2},...,\hat{\theta_t})$ 使得似然函数最大，因为lnx是x的单调递增函数，因此$\ln L(\theta_1,\theta_2,\theta_3,...,\theta_t)$与$L(\theta_1,\theta_2,\theta_3,...,\theta_t)$有相同的极大值点，故极大似然估计$(\hat{\theta_1},\hat{\theta_2},...,\hat{\theta_t})$ 可等价定义为
$$
\ln L(\hat{\theta_1},\hat{\theta_2},...,\hat{\theta_t})=\underset{(\theta_1,\theta_2,,...,\theta_t)\in \Theta}{max}\ln L(\theta_1,\theta_2,...,\theta_t)
$$
很多情况下，似然函数$L(\theta_1,\theta_2,...,\theta_t)$和对数似然函数关于$\theta_1,\theta_2,...,\theta_t$的偏导数存在，这时$\hat{\theta_1},\hat{\theta_2},...,\hat{\theta_t}$可以从方程组
$$
\frac{\partial L(\theta_1,\theta_2,...,\theta_t)}{\partial \theta_j}=0\ (j=1,2,...,t)
$$
或者
$$
\frac{\partial\ln L(\theta_1,\theta_2,...,\theta_t)}{\partial \theta_j}=0\ (j=1,2,...,t)
$$
解得

### 系数求解

对Θ进行极大似然 估计，使得观测数据发生的概率最大：
$$
L(\theta) = \prod_{i=1}^{n} P(y_i|x_i; \theta) = \prod_{i=1}^{n} (f_{\theta}(x_i))^{y_i} (1 - f_{\theta}(x_i))^{1-y_i}
$$
转换为对数似然，有
$$
\ln L(\theta) = \sum_{i=1}^{n} \left( y_i \ln(f_{\theta}(x_i)) + (1 - y_i) \ln(1 - f_{\theta}(x_i)) \right) \\
= \sum_{i=1}^{n} \left( (1 - y_i)(-\theta^T * x_i) - \ln(1 + e^{-\theta^T * x_i}) \right)
$$


极大似然法要求最大值，所以使用梯度上升来求

> 在求极值的问题中，有梯度上升和梯度下降两个最优化方法。
>
> 梯度上升用于求极大值，梯度下降用于求极小值。如logistic回归的目标函数 是求参向量𝜃，使得对数似然函数达到最大值，所以使用梯度上升算法。
>
> 而线性回归的代价函数：代表的则是误差，我们想求误差最小值，所以用 梯度下降算法。

梯度上升公式如下：
$$
\theta^{(t+1)}=\theta^{(t)}+\alpha \nabla_\theta \ln({L(\theta^{(t)})})
$$
**求梯度如下**：

<img src="https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721707.png" alt="image-20220318163554005" style="zoom:80%;" />

得到：

![image-20220318163833614](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721689.png)

与线性回归的公式很相似   $\theta_j^{(t+1)}:=\theta_j^{(t)}-\alpha\sum^n_{i=1}(f(x_i)-y_i)x_{i,j}$

## 逻辑回归与线性回归的比较

| 类别                   | 逻辑回归                                                     | 线性回归                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **输出**               | $y\in\{0,1\}$                                                | $y\in R$                                                     |
|                        | 线性二分类 ：决策边界 $𝜃^𝑇 𝑥 = 0$                            | 线性拟合 ： $𝑦 = 𝜃^𝑇 𝑥$                                      |
|                        | ![image-20220318164310873](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721690.png) | ![image-20220318164319006](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721692.png) |
| **自变量和因变量关系** | 无要求                                                       | 线性关系                                                     |
| **数据假设**           | P(y\|x)服从伯努利分布                                        | 𝑦 − 𝑓(𝑥)服从iid高斯分布                                      |
| **建模**               | 极大似然方法                                                 | 最小二乘方法                                                 |
| **求解**               | 梯度上升                                                     | 梯度下降                                                     |

## 线性回归的概率解释

假设输入输出之间满足以下关系

![image-20220318164551455](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721693.png)

假设  𝜀 服从高斯分布𝑁(0, 𝜎)，那么
$$
p(y_i \mid \mathbf{x}_i; \theta) = \frac{1}{\sqrt{2\pi\sigma}} \exp\left(-\frac{(y_i - \theta^T \mathbf{x}_i)^2}{2\sigma^2}\right)
$$


假设样本独立同分布，于是可通过极大似然法(MLE)估 计𝜃。构造似然函数：

![image-20220318164627356](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721695.png)

通过“对数似然”求解

![image-20220318164644563](https://raw.githubusercontent.com/yijunquan-afk/img-bed-1/main/img4/1695721696.png)

最大化$\ln (L(\theta))$等价于  $\underset{\theta}{min}\sum^n_{i=1}(y_i-\theta^Tx_i)^2=J(\theta)$

在假设误差服从独立同分布的高斯分布情况下，极大似然估计等价于最小平方和误差损失函数

## 参考资料

[1]庞善民.西安交通大学机器学习导论2022春PPT

[2]周志华.机器学习.北京:清华大学出版社,2016

[3]王宁.概率论与数理统计:西安交通大学出版社,2017

[3\][多项式回归（Polynomial regression）](https://www.jianshu.com/p/19f870e4cb5a)

[4]百度百科

[5\][机器学习中的Accuracy和Precision的区别](https://www.cnblogs.com/zzai/p/15750322.html)

