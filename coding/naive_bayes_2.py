#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'
#coding:utf-8
#朴素贝叶斯算法   贝叶斯估计， λ=1  K=2， S=3； λ=1 拉普拉斯平滑
import pandas as pd
import numpy as np

class NavieBayesB(object):
    def __init__(self):
        self.A = 1    # 即λ=1
        self.K = 2
        self.S = 3

    def getTrainSet(self):
        # 获取数据集
        trainSet = pd.read_csv('../data/naivebayes_data.csv')
        # 转换数据类型
        trainSetNP = np.array(trainSet)
        # 获取男生的特征属性专业和身高
        trainData = trainSetNP[:,0:trainSetNP.shape[1]-1]
        # 获取是否有女朋友的结果，0表示没有1表示有
        labels = trainSetNP[:,trainSetNP.shape[1]-1]
        return trainData, labels

    def naive_bayes(self, trainData, labels, features):
        labels = list(labels)    #转换为list类型
        # 计算先验概率
        P_y = {}
        for label in labels:
            P_y[label] = (labels.count(label) + self.A) / float(len(labels) + self.K*self.A)
        # 计算条件概率
        P = {}
        for y in P_y.keys():
            # 区分不同的分类，此数据中分为0,1两类
            y_index = [i for i, label in enumerate(labels) if label == y]
            # 计算每种类型出现的次数
            y_count = labels.count(y)
            for j in range(len(features)):
                pkey = str(features[j]) + '|' + str(y)
                # 区分不同属性集，专业和身高的一一对应
                x_index = [i for i, x in enumerate(trainData[:,j]) if x == features[j]]
                # 计算特征x和对应的类别出现的次数
                xy_count = len(set(x_index) & set(y_index))
                # 计算条件概率
                P[pkey] = (xy_count + self.A) / float(y_count + self.S*self.A)
        #features所属类
        F = {}
        for y in P_y.keys():
            F[y] = P_y[y]
            for x in features:
                F[y] = F[y] * P[str(x)+'|'+str(y)]
        # 得到概率最大的类别
        features_y = max(F, key=F.get)
        return features_y


if __name__ == '__main__':
    nb = NavieBayesB()
    # 训练数据
    trainData, labels = nb.getTrainSet()
    # 测试数据集
    features_all = [['art','short'],['art','medium'],['art','tall'],['computer','short'],['computer','medium'],['computer','tall'],['mathematica','short'],['mathematica','medium'],['mathematica','tall'],]
    for features in features_all:
        # 该特征应属于哪一类
        result = nb.naive_bayes(trainData, labels, features)
        if result == 1 :
            print ('专业为:',features[0],'身高为',features[1],'的男生有女朋友')
        else:
            print('专业为:', features[0], '身高为：', features[1], '的男生没有女朋友')



# #coding:utf-8
# # 极大似然估计  朴素贝叶斯算法
# class NaiveBayes(object):
#     def getTrainSet(self):
#         dataSet = pd.read_csv('../data/naivebayes_data.csv')
#         dataSetNP = np.array(dataSet)  #将数据由dataframe类型转换为数组类型
#         trainData = dataSetNP[:,0:dataSetNP.shape[1]-1]   #训练数据x1,x2
#         labels = dataSetNP[:,dataSetNP.shape[1]-1]        #训练数据所对应的所属类型Y
#         return trainData, labels
#
#     def classify(self, trainData, labels, features):
#         #求labels中每个label的先验概率
#         labels = list(labels)    #转换为list类型
#         P_y = {}       #存入label的概率
#         for label in labels:
#             P_y[label] = labels.count(label)/float(len(labels))   # p = count(y) / count(Y)
#
#         #求label与feature同时发生的概率
#         P_xy = {}
#         for y in P_y.keys():
#             y_index = [i for i, label in enumerate(labels) if label == y]  # labels中出现y值的所有数值的下标索引
#             for j in range(len(features)):      # features[0] 在trainData[:,0]中出现的值的所有下标索引
#                 x_index = [i for i, feature in enumerate(trainData[:,j]) if feature == features[j]]
#                 xy_count = len(set(x_index) & set(y_index))   # set(x_index)&set(y_index)列出两个表相同的元素
#                 pkey = str(features[j]) + '*' + str(y)
#                 P_xy[pkey] = xy_count / float(len(labels))
#
#         #求条件概率
#         P = {}
#         for y in P_y.keys():
#             for x in features:
#                 pkey = str(x) + '|' + str(y)
#                 P[pkey] = P_xy[str(x)+'*'+str(y)] / float(P_y[y])    #P[X1/Y] = P[X1Y]/P[Y]
#
#         #求[2,'S']所属类别
#         F = {}   #[2,'S']属于各个类别的概率
#         for y in P_y:
#             F[y] = P_y[y]
#             for x in features:
#                 F[y] = F[y]*P[str(x)+'|'+str(y)]     #P[y/X] = P[X/y]*P[y]/P[X]，分母相等，比较分子即可，所以有F=P[X/y]*P[y]=P[x1/Y]*P[x2/Y]*P[y]
#
#         features_label = max(F, key=F.get)  #概率最大值对应的类别
#         return features_label
#
#
# if __name__ == '__main__':
#     nb = NaiveBayes()
#     # 训练数据
#     trainData, labels = nb.getTrainSet()
#     # x1,x2
#     features = [2,'S']
#     # 该特征应属于哪一类
#     result = nb.classify(trainData, labels, features)
