#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import datasets

def plot_origin(X):
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 按照三个类别取出数据并且绘图表示
    plt.scatter(X[:50, 0], X[:50, 1],color='b', marker='x', label='山鸢尾花')#setosa
    plt.scatter(X[50:100, 0], X[50:100, 1],color='g', marker='*', label='变色鸢尾花')#versicolor
    plt.scatter(X[100:, 0], X[100:, 1],color='r', marker='+', label='维尼亚鸢尾')#Virginica
    plt.xlabel('花瓣的宽度')#petal width
    plt.ylabel('萼片的宽度')#sepal width
    # 显示图例
    plt.legend(loc='upper right')
    plt.show()

def Logistic_regression(X,Y):
    logistic = LogisticRegression(C=1e5)
    logistic.fit(X, Y)
    return logistic

def test_data(logistic):
    # 生成一组测试数据集
    # meshgrid函数生成两个网格矩阵
    h = .02
    x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
    y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
    # print(x_max,x_min,y_max,y_min)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # print(xx.ravel(),yy.ravel())
    # pcolormesh函数将xx,yy两个网格矩阵和对应的预测结果Z绘制在图片上
    Z = logistic.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z)
    plt.figure(1, figsize=(6, 4))
    # plt.scatter(xx,yy,color='purple',marker='.')
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
    plot_origin(X)

if __name__ == '__main__':
    iris = datasets.load_iris()
    x = pd.DataFrame(iris.data)
    X = x.iloc[:, [1, 3]].values  # 取出2个特征，并把它们用Numpy数组表示
    Y = pd.DataFrame(iris.target)
    logistic = Logistic_regression(X, Y)
    test_data(logistic)
    # plot_origin(X)
    # df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases//iris.data', header=None) # 加载Iris数据集作为DataFrame对象