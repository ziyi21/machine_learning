#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'

from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
import os
import pandas as pd
import numpy as np

# 导入路径
os.environ["PATH"] += os.pathsep + 'D:\\program\\graphviz\\bin\\'
# 获取数据集
def getTrainSet():
    trainSet = pd.read_csv ('../data/naivebayes_data.csv')
    trainSet_dum = pd.get_dummies(trainSet.loc[:,['x1','x2']]) # 获取两个特征数据并哑原化处理
    # trainall = pd.concat([trainSet_dum,trainSet['Y']],axis=1)
    # trainSet_all = pd.concat([trainSet,trainSet_dum],axis=1) # 将处理后的数据进行拼接
    # 转换数据类型并的得到特征属性和分类结果
    trainData = np.array(trainSet_dum)
    labels = np.array(trainSet['Y'])
    data_columns = list(trainSet_dum.columns)
    return data_columns,trainData, labels
#
# def yuanweiData():
#     iris = load_iris ()
#     return iris

# 构建模型
def tree_module(trainData,labels):
    # data_columns,trainData, labels = getTrainSet()
    # iris = yuanweiData()
    decision_tree = tree.DecisionTreeClassifier(max_depth=4)
    decision_tree_one = decision_tree.fit(trainData, labels)
    # decision_tree_flower = decision_tree.fit(iris.data, iris.target)
    return decision_tree_one

# 绘制决策树图

def pic_tree(decision_tree,filename,columns,theclass,save_path):
    # 保存模型
    with open (filename, 'w') as f:
        f = tree.export_graphviz (decision_tree, out_file=f)

    dot_data = tree.export_graphviz (decision_tree, out_file=None, feature_names=columns,
                                     class_names=theclass, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data (dot_data)
    # 保存图像到pdf文件或pdf图片
    # graph.write_pdf("weather.pdf")
    graph.write_png (save_path)

if __name__ == '__main__':
    data_columns, trainData, labels = getTrainSet()
    iris = load_iris ()
    boyfriends = tree_module(trainData,labels)
    flower = tree_module(iris.data,iris.target)
    filename = ['boyfriends.dot','flower.dot']
    columns = [data_columns,iris.feature_names]
    print(columns[0])
    classes = [['yes','no'],iris.target_names]
    save_path = ['../all_pictures/boyfriends.png','../all_pictures/flowers.png']
    pic_tree(boyfriends,filename[0],columns[0],classes[0],save_path[0])
    pic_tree (flower, filename[1], columns[1], classes[1], save_path[1])