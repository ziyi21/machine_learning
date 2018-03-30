#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.datasets.samples_generator import make_regression
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_hastie_10_2

# rand(d0, d1, ..., dn) 用来生成d0xd1x...dn维的数组。数组的值在[0,1]之间
print(np.random.rand(2,3,2))
# randn((d0, d1, ..., dn), 也是用来生成d0xd1x...dn维的数组。不过数组的值服从N(0,1)的标准正态分布。
print(np.random.randn(2,3,2))
# 如果需要服从N(μ,σ2)的正态分布，只需要在randn上每个生成的值x上做变换σx+μ即可，例如：
print(2*np.random.randn(2,3,2) + 1)
# randint(low[, high, size])，生成随机的大小为size的数据，size可以为整数，为矩阵维数，或者张量的维数。值位于半开区间 [low, high)。
print(np.random.randint(1, 5, size=[3,2])) #返回维数为2x3的数据。取值范围为[3,6).
print(np.random.randint(2, size=[3,2,4])) # 指定最大值
print(np.random.random_integers(2, size=[3,2,4]))
# random_integers(low[, high, size]),和上面的randint类似，区别在与取值范围是闭区间[low, high]。
# random_sample([size]), 返回随机的浮点数，在半开区间 [0.0, 1.0)。如果是其他区间[a,b),可以加以转换(b - a) * random_sample([size]) + a
print((6-2)*np.random.random_sample(4)+2)

plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
# 在同一个画布中生产3行2列6个子画布

# 1、生成回归数据样本
plt.subplot(321)
plt.title("生成回归数据样本", fontsize='small')
# 生成500个样本数据，其中X表示样本特征，每个样本有一个特征，y表示样本取值， coef为样本回归系数，
X0, Y0, coef = make_regression(n_samples=500, n_features=1 ,noise=30, coef=True)
plt.scatter(X0, Y0,  color='purple',marker='*')
plt.plot(X0, X0*coef, color='yellow',linewidth=3, label='回归模型')

# 2、生成2分类数据样本
plt.subplot(322)
plt.title("生成多分类数据样本", fontsize='small')
# # X1为样本特征，Y1为样本类别输出， 共500个样本，每个样本3个有效特征2个冗余特征1个重复特征，输出有3个类别
X1, Y1 = make_classification(n_samples=500 , n_features=6, n_redundant=2, n_informative=3,n_clusters_per_class=1,n_repeated=1,n_classes=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='*', c=Y1)
csv_records = pd.DataFrame(X1,Y1)
csv_records.to_csv('../data/classes_data.csv',header=False)

# 3、生成多分类数据样本
plt.subplot(323)
plt.title("hastie生成2分类数据样本",fontsize='small')
X2, Y2 = make_hastie_10_2(n_samples=500)
plt.scatter(X2[:, 0], X2[:, 1], marker='+', c=Y2)


# 4、生成聚类数据样本
plt.subplot(324)
plt.title("生成聚类数据样本", fontsize='small')
# X3为样本特征，Y3为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，并依次给出簇位置和方差
X3, Y3 = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [1,1], [2,2],[3,3]], cluster_std=[0.2, 0.7, 0.5,0.3],)
plt.scatter(X3[:, 0], X3[:, 1], marker='+', c=Y3)

# 5、生成正太分布数据样本
# plt.subplot(325)
# plt.title("正态分布数据样本", fontsize='small')
# X4, Y4 = make_gaussian_quantiles(n_features=2, n_classes=3)
# plt.scatter(X4[:, 0], X4[:, 1], marker='o', c=Y4, s=50, edgecolor='y')
# # 2、生成2分类数据样本
plt.subplot(325)
plt.title("生成单分类数据样本", fontsize='small')
# # X1为样本特征，Y1为样本类别输出， 共800个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
X4, Y4 = make_classification(n_samples=500 , n_features=2, n_redundant=0, n_informative=1,n_clusters_per_class=1,n_repeated=0,n_classes=1)
plt.scatter(X4[:, 0],X4[:, 1], marker='*', c=Y4)

# 6、生成正太分布数据样本
plt.subplot(326)
plt.title("正态分布数据样本", fontsize='small')
#生成2维正态分布，生成的数据按分位数分成3组，1000个样本,2个样本特征均值为1和2，协方差系数为2
X5, Y5 = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, mean=[1,2],cov=2)
plt.scatter(X5[:, 0], X5[:, 1], marker='*', c=Y5)
# 数据样本可视化
plt.show()

# rand(d0, d1, ..., dn) 用来生成d0xd1x...dn维的数组。数组的值在[0,1]之间
np.random.rand(3,2,2)
# randn((d0, d1, ..., dn), 也是用来生成d0xd1x...dn维的数组。不过数组的值服从N(0,1)的标准正态分布。
np.random.randn(3,2)
# 如果需要服从N(μ,σ2)的正态分布，只需要在randn上每个生成的值x上做变换σx+μ即可，例如：
2*np.random.randn(3,2) + 1
# randint(low[, high, size])，生成随机的大小为size的数据，size可以为整数，为矩阵维数，或者张量的维数。值位于半开区间 [low, high)。
np.random.randint(3, 6, size=[2,3]) #返回维数为2x3的数据。取值范围为[3,6).
np.random.randint(3, size=[2,3,4])
# random_integers(low[, high, size]),和上面的randint类似，区别在与取值范围是闭区间[low, high]。
# random_sample([size]), 返回随机的浮点数，在半开区间 [0.0, 1.0)。如果是其他区间[a,b),可以加以转换(b - a) * random_sample([size]) + a
(5-2)*np.random.random_sample(3)+2