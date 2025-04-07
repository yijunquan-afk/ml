# ML6 鸢尾花分类

## 描述
鸢尾花（Iris）数据集是一个经典的机器学习数据集，包含三种鸢尾花的特征和标签。
数据集包含150个样本，每个样本有4个特征：
花萼长度（sepal length）
花萼宽度（sepal width）
花瓣长度（petal length）
花瓣宽度（petal width）
标签有三种：
Iris-setosa
Iris-versicolor
Iris-virginica
你的任务是用sklearn的LogisticsRegression模型对鸢尾花类别进行预测，模型请指定参数max_iter=200以减少训练用时。
1.读入数据集
2.对数据集进行一次随机洗牌
3.数据分成训练集与测试集

## 输入描述：
输入一个整数n，表示测试样本的数量，直接截取最后数据集的最后n个作为测试集。
鸢尾花数据集请用sklearn库中的load_iris()函数加载

## 输出描述：
输出n行，每行包含两个数据，用空格分隔，第一个数据是预测类别，第二个数据是最大概率
概率保留两位小数

```
输入：
3
输出：
virginica 0.60
setosa 0.99
setosa 0.98
```

鸢尾花分类在机器学习中是经典的分类问题，通常被初学者用来学习分类算法。

鸢尾花数据集包含150个样本，每个样本有4个特征：花萼长度、花萼宽度、花瓣长度和花瓣宽度。该数据集用于分类任务，标签分为三类：Iris-setosa、Iris-versicolor和Iris-virginica。

逻辑回归是一种常用的分类算法，用于解决二分类问题。它通过计算输入特征的线性组合，并应用一个sigmoid函数来输出概率。 鸢尾花分类其实是多分类，但是为了简化算法与实现，我们使用逻辑回归算法进行训练，并输出每个测试样本的预测类别及其最大概率。并且为了简化算法与实现，我们使用sklearn库中的LogisticRegression类进行实现，并且为了防止过拟合以及加快收敛速度，我们设置max_iter=200。




```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import random 

iris = load_iris()
X = iris.data
y = iris.target

n = int(input())

random.seed(42)
indices=list(range(len(X)))
random.shuffle(indices)
X = X[indices]
y = y[indices]
X_train = X[:-n]
X_test = X[-n:]
y_train = y[:-n]
y_test = y[-n:]
model = LogisticRegression(max_iter = 200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
for i in range(len(y_pred)):
    print(f'{iris.target_names[y_pred[i]]} {np.max(y_prob[i]):.2f}')
```

    virginica 0.60
    setosa 0.99
    setosa 0.98


鸢尾花分类在机器学习中是经典的分类问题，通常被初学者用来学习分类算法。

鸢尾花数据集包含150个样本，每个样本有4个特征：花萼长度、花萼宽度、花瓣长度和花瓣宽度。该数据集用于分类任务，标签分为三类：Iris-setosa、Iris-versicolor和Iris-virginica。

逻辑回归是一种常用的分类算法，用于解决二分类问题。它通过计算输入特征的线性组合，并应用一个sigmoid函数来输出概率。 鸢尾花分类其实是多分类，但是为了简化算法与实现，我们使用逻辑回归算法进行训练，并输出每个测试样本的预测类别及其最大概率。并且为了简化算法与实现，我们使用sklearn库中的LogisticRegression类进行实现，并且为了防止过拟合以及加快收敛速度，我们设置max_iter=200。


