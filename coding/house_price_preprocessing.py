#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random
boston_house_price = datasets.load_boston()
boston_house_price_X = boston_house_price.data
boston_house_price_y = boston_house_price.target

#随机擦痒25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test=train_test_split(boston_house_price_X, boston_house_price_y, random_state=0, test_size=0.25)
# 查看回归目标值的特征
print("最高房价值为：",np.max(boston_house_price_y))
print("最低房价值为：",np.min(boston_house_price_y))
print("房价的均值为",np.mean(boston_house_price_y))
# 数据初始化
boston_house_price = datasets.load_boston()
house_price_X = boston_house_price.data
house_price_y = boston_house_price.target
print(house_price_X.shape)
house_price_X_norm1 = preprocessing.scale(house_price_X)
house_price_X_norm = preprocessing.MinMaxScaler().fit_transform(house_price_X)
print(house_price_X_norm.shape)
# 拆分测试集和训练集
house_price_train_X , house_price_test_X, house_price_train_y ,house_price_test_y = train_test_split(house_price_X_norm, house_price_y, test_size=0.2,random_state=0)
# 图片颜色选择列表
colors = ['red','yellow','blue','green','purple','grey']
# 依次画出13个属性与房屋价格之间的关系
for i in range(house_price_X_norm.shape[1]):
    plt.ion()
    plt.scatter(house_price_X[:,i], house_price_y, c=random.choice(colors), alpha=1, marker='+')
    plt.title('第{}个属性与房屋价格的关系图'.format(i+1))
    plt.xlabel('属性 {}'.format(i+1))
    plt.ylabel('房屋价格')
    plt.savefig('../all_pictures/第{}个属性与房屋价格的关系图'.format(i+1))
    plt.pause(2)
    plt.close()
