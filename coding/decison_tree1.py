#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'

#---author:朱元禄---

import pandas
data = pandas.read_csv(
'file:///Users/apple/Desktop/jacky_reinvest.csv',
encoding='GBK'
)

#调用Map方法进行可比较大小虚拟变量的转换
productDict={
'12个月定存':4,'6个月定存':3,'3个月定存':2,'1个月定存':1
}
data['产品Map']=data['金融产品'].map(productDict)

#调用get_dummyColumns方法进行可比较大小虚拟变量的转换
dummyColumns = ['初始心里预期','客户类别',]
for column in dummyColumns:
    data[column]= data[column].astype('category')

dummiesData=pandas.get_dummies(
    data,
    columns=dummyColumns,prefix=dummyColumns
    ,prefix_sep='_',dummy_na=False,drop_first=False)

#挑选可以建模的变量 featureData
fData = dummiesData[[
    '购买金额','产品Map','初始心里预期_复投','客户类别_VIP用户'
]]

#设定目标变量 targetData
tData = dummiesData['复投模式']

#生成决策树
from sklearn.tree import DecisionTreeClassifier

#设置最大叶子数为8
dtModel = DecisionTreeClassifier(max_leaf_nodes=8)

'''
#模型检验－交叉验证法
from sklearn.model_selection import cross_val_score

cross_val_score(
    dtModel,
    fData,tData,cv=10
)
'''
#训练模型
dtModel=dtModel.fit(fData,tData)

#模型可视化
import pydotplus
from sklearn.externals.six import StringIO  #生成StringIO对象
from sklearn.tree import export_graphviz

dot_data = StringIO() #把文件暂时写在内存的对象中

export_graphviz(
    dtModel,
    out_file=dot_data,
    class_names=['复投','不复投'],
    feature_names=['购买金额','产品Map',
                   '初始心里预期_不复投','客户类别_VIP用户'],
    filled=True,rounded=True,special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('shujudata.png')