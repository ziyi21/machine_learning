#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'

import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
# 数据预处理模块导入
from sklearn import preprocessing
# 房屋数据初始化
boston_house_price = datasets.load_boston()
house_price_X = boston_house_price.data
house_price_y = boston_house_price.target
print(house_price_X.shape)
# 进行数据的标准化和归一化处理
house_price_X_norm1 = preprocessing.scale(house_price_X)
house_price_X_norm = preprocessing.MinMaxScaler().fit_transform(house_price_X)
print(house_price_X_norm.shape)
# 拆分测试集和训练集
house_price_train_X , house_price_test_X, house_price_train_y ,house_price_test_y = train_test_split(house_price_X_norm, house_price_y, test_size=0.5,random_state=0)



# 岭回归模型的导入
from sklearn.linear_model import Ridge
# 普通拉索回归模型的导入
from sklearn.linear_model import Lasso
# 实现交叉检验拉索回归模型的导入
from sklearn.linear_model import LassoCV

# 按照不同的参数训练岭回归模型和拉索模型
for i in [0,0.0001,0.5,1,5,10]:
    from sklearn.linear_model import Ridge
    # 岭回归模型的导入
    ridge = Ridge(alpha=float('{}'.format(i))).fit(house_price_train_X,house_price_train_y)# 默认的lamda的参数为i
    # 岭回归模型训练的准确率
    predict_result_ridge = ridge.predict(house_price_test_X)
    predict_result_ridge1 = ridge.score(house_price_train_X, house_price_train_y)
    predict_result_ridge0 = ridge.score(house_price_test_X, house_price_test_y)
    print('岭回归惩罚参数为 {} ，训练集的准确率:'.format(i),predict_result_ridge1)
    print('岭回归惩罚参数为 {} ，测试集的准确率:'.format(i), predict_result_ridge0)

    # 普通拉索回归模型的导入
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=np.float('{}'.format(i)) ,max_iter=1000000).fit(house_price_train_X, house_price_train_y)  # 默认的lamda的参数为i
    # 拉索模型训练的准确率
    predict_result_lasso = lasso.predict(house_price_test_X)
    predict_result_lasso1 = lasso.score(house_price_train_X, house_price_train_y)
    predict_result_lasso0 = lasso.score(house_price_test_X, house_price_test_y)
    print('拉索回归惩罚参数为 {} ，训练集的准确率:'.format(i), predict_result_lasso1)
    print('拉索回归惩罚参数为 {} ，测试集的准确率:'.format(i), predict_result_lasso0)
    print('拉索回归惩罚参数为 {}，使用的特征属性有：{}'.format(i,np.sum(lasso.coef_ != 0)))

    # 实现交叉检验拉索回归模型的导入
    from sklearn.linear_model import LassoCV
    lasso_cv = LassoCV(alphas=np.logspace(-3,1,2,50) ,max_iter=1000000).fit(house_price_train_X, house_price_train_y)  # 默认的lamda的参数为i
    # 交叉检验拉索模型训练的准确率
    predict_result_lasso_cv = lasso_cv.predict(house_price_test_X)
    predict_result_lasso_cv1 = lasso_cv.score(house_price_train_X, house_price_train_y)
    predict_result_lasso_cv0 = lasso_cv.score(house_price_test_X, house_price_test_y)
    print('交叉检验拉索回归 训练集的准确率:', predict_result_lasso_cv1)
    print('交叉检验拉索回归 测试集的准确率:', predict_result_lasso_cv0)

from sklearn.linear_model import LinearRegression
# 训练多元线性回归模型
lr = LinearRegression().fit(house_price_train_X ,house_price_train_y)
# 预测测试集房价结果
predict_result_lr = lr.predict(house_price_test_X)
# 模型训练的准确率
predict_result_lr1 = lr.score(house_price_train_X, house_price_train_y)
predict_result_lr0 = lr.score(house_price_test_X, house_price_test_y)
print('线性回归训练集的准确率:',predict_result_lr1)
print('线性回归测试集的准确率:',predict_result_lr0)