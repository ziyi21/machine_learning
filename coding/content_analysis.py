#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'
import pandas as pd
import numpy as np
# 导入结巴分词的模块并且将自定义的字典加入其中
import jieba
from jieba import analyse
jieba.load_userdict('factors_line.txt')

# 根据分类构建用户自己的字典，
def create_user_dict(filepath):
    user_dict_origin = pd.read_excel(filepath)
    for i in range(user_dict_origin.shape[1]):
        one_factor = pd.DataFrame(user_dict_origin.iloc[:,i])
        # notnull_factor = one_factor.dropna().values  # 结果每一个值是一个列表，然后归入每一列的总列表中
        notnull_factor =  np.array(one_factor.dropna()).tolist() # 将DataFrame的值转化为series类型，再转换为列表类型
        # 依次将每个word按行追加到文件后面，并且指定编码方式为utf-8
        for word in notnull_factor:
            with open('factors_line.txt', 'a',encoding='utf-8') as f:
                f.write(word[0] + '\n')

# 获取每篇财务报告的内容
def cut_content(filepath,stopwords):
    contents = pd.read_excel(filepath)
    print('数据的规模', contents.shape)
    cut_report_results = []
    for report_index in range(contents.shape[0]):

        cut_report_results.append([contents.iloc[report_index,0],contents.iloc[report_index, 1],contents.iloc[report_index, 2]])
    print(cut_report_results[1:3])

def statistics_factors(filepath):
    factors = pd.read_excel(filepath)
    # print(all_factor)
    # with open('factors.txt', 'w') as f:
    #     f.write(repr(all_factor))  # 使用repr关键字将列表转化为原生的字符串
    # all_factor.to_csv('xgboost.txt', header=True, index=False, sep='\t', mode='a')

    #
    # table1 = pd.read_excel(filepath, header=None)# 表示自动添加一行索引的表头，所有数据从第一行开始读取
    # # table1 = table1.dropna() # 去掉有空值的行
    # # table1 = table1.replace(0,np.nan)
    # table1 = table1.fillna(0) # 替换空值的内容
    # print('数据的规模',table1.shape)
    # print('数据的前五条内容',table1.head(2))# 单个数字n表示获取前n条数据
    # # print('',table.get('战略计划').dropNa) # 获取数据某一列的数据和介绍
    # print('总的空值的个数',table1.isnull().sum().sum())
    # # print('每列空值的个数', table.isnull().sum())
    # print('获取某一列的值', table1.iloc[:,1])



    # print(type(table))
    # table1 = pd.read_excel(filepath, header=None)# 表示自动添加一行索引的表头，所有数据从第一行开始读取
    # # table1 = table1.dropna() # 去掉有空值的行
    # # table1 = table1.replace(0,np.nan)
    # table1 = table1.fillna(0) # 替换空值的内容

    # print('数据的前五条内容',table1.head(2))# 单个数字n表示获取前n条数据
    # # print('',table.get('战略计划').dropNa) # 获取数据某一列的数据和介绍
    # print('总的空值的个数',table.isnull().sum().sum())
    # # print('每列空值的个数', table.isnull().sum())
    # print('获取某一列的值', table.iloc[:,1])

if __name__ == '__main__':
    factors_filepath = r'C:\Users\Think\Desktop\content_analysis\linguistic_feature.xlsx'
    content_filepath = r'C:\Users\Think\Desktop\content_analysis\mda.xlsx'
    # 创建用户自己的分词字典
    # create_user_dict(factors_filepath)
    stopwords = [line.rstrip() for line in open('stopwords_gbk.txt', 'r')]
    # 使用自定义停用词集合
    analyse.set_stop_words("stopwords_utf-8.txt")
    # 切分文章内容
    financial_content = cut_content(content_filepath, stopwords)