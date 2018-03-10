#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'
# 计算画图模块和数据的导入
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn import datasets
# LinearModile算法调用
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# 数据初始化
house_price_X, house_price_y = mglearn.datasets.load_extended_boston()
print(house_price_X.shape)
boston_house_price = datasets.load_boston()
boston_house_price_X = boston_house_price.data
boston_house_price_y = boston_house_price.target
print(boston_house_price_X.shape)
# 拆分测试集和训练集
house_price_train_X , house_price_test_X, house_price_train_y ,house_price_test_y = train_test_split(house_price_X, house_price_y, test_size=0.5,random_state=0)
# 训练多元线性回归模型
lr = LinearRegression().fit(house_price_train_X ,house_price_train_y)
# 预测测试集房价结果
predict_result_lr = lr.predict(house_price_test_X)
# print(predict_result_lr)

# 计算预测的准确性
print('训练集模型准确率', lr.score(house_price_train_X, house_price_train_y))
print('测试集模型准确率', lr.score(house_price_test_X, house_price_test_y))
print("最高房价值为：",np.max(boston_house_price_y))
print("最低房价值为：",np.min(boston_house_price_y))
print("房价的均值为",np.mean(boston_house_price_y))

v = np.power(house_price_train_X, 0.5)
print(v)
print(np.var(house_price_train_X))