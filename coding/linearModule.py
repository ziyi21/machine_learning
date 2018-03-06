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
boston_house_price = datasets.load_boston()
boston_house_price_X = boston_house_price.data
boston_house_price_y = boston_house_price.target
# 拆分测试集和训练集
house_price_train_X , house_price_test_X, house_price_train_y ,house_price_test_y = train_test_split(house_price_X, house_price_y, test_size=0.1,random_state=0)
# 训练多元线性回归模型
lr = LinearRegression().fit(house_price_train_X ,house_price_train_y)
# 预测测试集房价结果
predict_result  = lr.predict(house_price_test_X)
print(predict_result)
print(house_price_test_y)
# 计算预测的准确性
print(lr.score(house_price_test_X, house_price_test_y))