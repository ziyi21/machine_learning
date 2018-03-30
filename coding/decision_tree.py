#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'

# 决策树演示
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

# 导入路径
os.environ["PATH"] += os.pathsep + 'D:\\program\\graphviz\\bin\\'

# 导入数据
# weather_data_o = pd.get_dummies(pd.read_csv('../data/weather_data.csv',encoding='gbk'))
# weather_data_o = pd.read_csv('../data/weather_data.csv',encoding='gbk')


def getTrainSet():
    # 获取数据集
    trainSet = pd.read_csv ('../data/naivebayes_data.csv')
    # print(trainSet.values)
    trainSet_dum = pd.get_dummies(trainSet.loc[:,['x1','x2']]) # 获取两个特征数据并哑原化处理
    print(trainSet_dum)
    trainall = pd.concat([trainSet_dum,trainSet['Y']],axis=1)
    print(trainall)
    # trainSet_all = pd.concat([trainSet,trainSet_dum],axis=1)
    # print(list(trainSet_dum.columns)[::-1])
    # 转换数据类型
    trainData = np.array (trainSet_dum)
    labels = np.array(trainSet['Y'])
    # print(trainData,labels)
    # 获取男生的特征属性专业和身高
    # trainData = trainSetNP[:, 1:trainSetNP.shape[1] - 1]
    # 获取是否有女朋友的结果，0表示没有1表示有
    # labels = trainSetNP[:, trainSetNP.shape[1] - 1]
    # print(trainData)
    return trainSet_dum,trainData, labels
# # print(weather_data_o)
# weather_data = np.array(weather_data_o)
# # print('numpy类型',weather_data)
# trainData = preprocessing.MinMaxScaler().fit_transform(weather_data[:, 1:weather_data.shape[1]])
# print('属性',trainData)
# labels = weather_data[:, 0:1]
# # print('标签',labels)
iris = load_iris()
# print(iris.feature_names)
# print(iris.target_names)
# print(np.array(weather_data_o.columns)[1:])
# 构建模型
trainSet_dum,trainData, labels = getTrainSet()
decision_tree = tree.DecisionTreeClassifier(max_depth=4)
decision_tree = decision_tree.fit(trainData, labels)

# from IPython.display import Image
# dot_data = tree.export_graphviz(tree, out_file=None,feature_names=list(trainSet_dum.columns)[,class_names=['0','1'],filled=True, rounded=True,special_characters=True)
# graph1 = pydotplus.graph_from_dot_data(dot_data)
# Image(graph1.create_png())

# 保存模型
with open("boyfriends.dot", 'w') as f:
    f = tree.export_graphviz(decision_tree, out_file=f)

# 画图，保存到pdf文件
# 设置图像参数
dot_data = tree.export_graphviz(decision_tree, out_file=None,feature_names=list(trainSet_dum.columns),class_names=['yes','no'],filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
# 保存图像到pdf文件或pdf图片
# graph.write_pdf("weather.pdf")
graph.write_png("../all_pictures/boyfriends.png")

# # 导入数据
# iris = load_iris()
# # 构建模型
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)
#
# # 保存模型
# with open("iris.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)
#
# # 画图，保存到pdf文件
# # 设置图像参数
# dot_data = tree.export_graphviz(clf, out_file=None,
#                          feature_names=iris.feature_names,
#                          class_names=iris.target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
#
# from IPython.display import Image
# dot_data = tree.export_graphviz(clf, out_file=None,
#                          feature_names=iris.feature_names,
#                          class_names=iris.target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph1 = pydotplus.graph_from_dot_data(dot_data)
# Image(graph1.create_png())