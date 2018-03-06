#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'
# KNN调用
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
iris = datasets.load_iris()
print(iris.data.shape)
# 拆分属性数据
iris_X = iris.data
# 拆分类别数据
iris_y = iris.target

print(len(iris_y))
# plt.plot(iris_X)
# plt.show()
a = np.unique(iris_y)
print(np.unique(iris_y))
print(np.zeros(a.shape))#按照原矩阵格式以0填充所有数据
# 方法一：拆分测试集和训练集,并进行预测
iris_train_X , iris_test_X, iris_train_y ,iris_test_y = train_test_split(iris_X, iris_y, test_size=0.2,random_state=0)
knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(iris_train_X, iris_train_y)
knn1.predict(iris_test_X)
print (iris_test_y)

# 方法二：拆分测试集和训练集
np.random.seed(0)
# permutation随机生成0-150的系列
indices = np.random.permutation(len(iris_y))
iris_X_train = iris_X[indices[:-30]]
iris_y_train = iris_y[indices[:-30]]
iris_X_test = iris_X[indices[-30:]]
iris_y_test = iris_y[indices[-30:]]
knn = KNeighborsClassifier()
# 提供训练集进行顺利
knn.fit(iris_X_train, iris_y_train)
# 预测测试集数据鸢尾花类型
predict_result = knn.predict(iris_X_test)
print(predict_result)
print(knn.score(iris_X_test, iris_y_test))
# precision, recall, thresholds = precision_recall_curve(iris_y_test, knn.predict(iris_X_test))
# answer = knn.predict_proba(iris_X_test)[:,1]
# print(classification_report(iris_y_test, answer ))

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                     weights='uniform')