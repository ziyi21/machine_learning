#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy
from sklearn.model_selection import train_test_split
boston_house_price = datasets.load_boston()
boston_house_price_X = boston_house_price.data
boston_house_price_y = boston_house_price.target
#随机擦痒25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test=train_test_split(boston_house_price_X, boston_house_price_y, random_state=0, test_size=0.25)
# 查看回归目标值的特征
print("最高房价值为：",np.max(boston_house_price_y))
print("最低房价值为：",np.min(boston_house_price_y))
print("房价的均值为",np.mean(boston_house_price_y))