#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'
# -*- coding: utf-8 -*-

# import re
# from operator import contains
import datetime
import re
from collections import Counter, defaultdict

import jieba
# import numpy as np
import pandas as pd
from dateutil.parser import parse
from jieba import analyse
from pandas import Series
from pymongo import MongoClient

tfidf = analyse.extract_tags
jieba.load_userdict('index_words.txt')


def read_from_mongodb():
    # 建立MongoDB数据库连接
    # connection = MongoClient('116.62.237.194', 27017)
    connection = MongoClient('localhost', 27017)
    db = connection['mmwz']  # 获取或创建一个数据库
    table = db['mmwz_enterprises_news']  # 获取或创建一个数据集
    return table


def count_medias(table):
    medias = []
    for u in table.find():
        if u.get('news_source'):
            medias.append(u.get('news_source').strip().strip('来源：來源: &nbsp').strip())
    return medias


def cut_words(table, stopwords):
    # 把数据库中的文章进行切词
    news_content_cut_all = []
    n = 0
    for u in table.find():  # .find 取得数据集中的内容{'enterprises_name': '蒋锡培'}
        if 'news_content' in u.keys():
            n = n + 1
            news_content = u.get('news_content')  # 如果键不存在可以设为默认的值
            news_content_cut = jieba.lcut(news_content, cut_all=False)  # lcut返回的是一个完整的列表cut_all=False 表示的精确切词
            for news_word in news_content_cut:
                if news_word not in stopwords:
                    news_content_cut_all.append(news_word)
    print(n)
    return news_content_cut_all  # return 返回的是一个列表


def get_keywords(table):
    keywords_all = []
    keywords_array = []
    for u in table.find():  # .find 取得数据集中的内容{'enterprises_name': '蒋锡培'}
        if 'news_content' in u.keys():
            news_content = u.get('news_content')  # 如果键不存在可以设为默认的值
            keywords = tfidf(news_content, topK=150)
            # print(keywords)
            for one in keywords:
                keywords_all.append(one)
            keywords = [u.get('news_title')] + [u.get('news_url')] + keywords
            keywords_array.append(keywords)
    return keywords_all, keywords_array  # return 返回的是一个列表


def count_words(words):
    count = Counter(words)  # 直接统计词频的方法
    # print(count)
    count = dict(count)
    print(count)
    # count_paixu = sorted(count, key=lambda k:count[k], reverse=True)#count[k]表示的是值，操作的是字典
    count_paixu = sorted(count.items(), key=lambda e: e[1], reverse=True)
    return count_paixu


def get_index_words(filepath):
    words_index_dataframe = pd.read_excel(filepath)
    words_index_dict = {}
    for col_name in words_index_dataframe.columns:
        words_index_dict[col_name] = list(set(words_index_dataframe[col_name].dropna().values.tolist()))
    return words_index_dict


def classify_words(words_index_dict, words_count):
    classify_words_results = defaultdict(list)
    for word in words_count:
        n = 0
        for key, values in words_index_dict.items():
            n += 1
            if word[0] in values:
                classify_words_results[key].append(word)
                break
                # print(word, n)
    return classify_words_results


def text_save(content, filename, mode='w'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    file.write('[')
    n = 0
    for i in content:
        n += 1
        if n < len(content):
            # print(i[0], i[1])
            file.write('{' + '"name":' + '"' + i[0] + '"' + ',' + '"value":' + str(i[1]) + '}' + ',')
        else:
            file.write('{' + '"name":' + '"' + i[0] + '"' + ',' + '"value":' + str(i[1]) + '}')
    file.write(']')
    file.close()


def text_save2(content, filename, mode='w'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    file.write('[')
    n = 0
    for i in content:
        n += 1
        if n < len(content):
            # print(i[0], i[1])
            if i[0] in ['legend', 'date']:
                file.write('{' + '"data":' + str(i[1]) + '}' + ',')
            elif i[0] in ['影响力指数']:
                file.write('{' + '"name":' + '"' + i[
                    0] + '"' + ',"type":"line","smooth":true,"symbolSize": 10,"symbol": "star6","data":' + str(
                    i[1]) + '}' + ',')
            elif i[0] in ['每年新闻报道总量', '每年新闻评论总量']:
                file.write('{' + '"name":' + '"' + i[
                    0] + '"' + ',"type":"line","smooth":true,"symbolSize": 6,"symbol": "circle","data":' + str(i[1]) + '}' + ',')
            else:
                file.write('{' + '"name":' + '"' + i[0] + '"' + ',type:"line",smooth:true,"data":' + str(i[1]) + '}' + ',')
        else:
            if i[0] in ['每年新闻报道总量', '每年新闻评论总量']:
                file.write('{' + '"name":' + '"' + i[
                    0] + '"' + ',"type":"line","smooth":true,"symbolSize": 6,"symbol": "circle","data":' + str(i[1]) + '}')
            elif i[0] in ['影响力指数']:
                file.write('{' + '"name":' + '"' + i[
                    0] + '"' + ',"type":"line","smooth":true,"symbolSize": 10,"symbol": "star6","data":' + str(
                    i[1]) + '}')
            else:
                file.write('{' + '"name":' + '"' + i[0] + '"' + ',type:"line",smooth:true,"data":' + str(i[1]) + '}')
    file.write(']')
    file.close()


def total_count(table, medias, filepath):
    # 统计指标一的各统计量,和新闻评论的时间序列

    all_list = []

    news_total_amount = table.find().count()
    all_list.append(('全网信息总量', news_total_amount))

    sina_comments = table.find({'spider': '新浪新闻'})
    comments_time = []
    comments_as = []
    comments_amount = 0
    for u in sina_comments:
        if u.get('news_comment_count'):
            comments_amount += int(u.get('news_comment_count')['show'])
            # 获得评论的时间序列
            if u.get('news_pubtime'):
                time = u.get('news_pubtime')
                if isinstance(time, datetime.datetime):
                    comments_time.append(time.date())
                    comments_as.append(int(u.get('news_comment_count')['show']))
                else:
                    # print(time)
                    standard_time = rowtime_processed(time)
                    if standard_time:
                        comments_time.append(standard_time.date())
                        comments_as.append(int(u.get('news_comment_count')['show']))
    other_comments = table.find({"$or": [{'spider': '东方财富网'}, {'spider': '一点资讯'}, {'spider': '今日头条'}]})
    for u in other_comments:
        if u.get('news_comment_count'):
            comments_amount += int(u.get('news_comment_count'))
            if u.get('news_pubtime'):
                time = u.get('news_pubtime')
                if isinstance(time, datetime.datetime):
                    comments_time.append(time.date())
                    comments_as.append(int(u.get('news_comment_count')))
                else:
                    # print(time)
                    standard_time = rowtime_processed(time)
                    if standard_time:
                        comments_time.append(standard_time.date())
                        comments_as.append(int(u.get('news_comment_count')))
    comments_ts_list = []
    print('comments_as:', comments_as)
    comments_ts = Series(comments_as, index=pd.DatetimeIndex(comments_time))
    print('comments_ts:', comments_ts)
    comments_count = timeseries_process(comments_ts)
    data_list = [int(str(i)) for i in list(comments_count.index)]
    comments_ts_list.append(('date', data_list))  # 把横坐标页加进去
    comments_ts_list.append(('每年新闻评论总量', list(comments_count)))
    text_save2(comments_ts_list, filename=r'analysis_data\comments_ts.txt')

    all_list.append(('全网评论总量', comments_amount))

    medias_amount = len(Counter(medias))
    all_list.append(('信息来源网站总量', medias_amount))

    file = open(filepath, mode='r')
    site_set = file.read()
    site_amount = len(site_set.split(',{')) + 2  # 加2表示加上北京和上海 其实不加也OK
    all_list.append(('信息散布的地级市总量', site_amount))
    text_save(all_list, filename=r'analysis_data\total.txt')


def top10_medias(count_list_sort):
    # 统计指标二的统计量
    other_values = 0
    for i in count_list_sort[10:]:
        other_values += i[1]
    top10_other = count_list_sort[:10] + [('其他', other_values)]
    text_save(top10_other, filename=r'analysis_data\top10_medias.txt')


def rowtime_processed(time):
    time = time.strip().strip('发布时间：公告日期：時間:   中国经济网  人民日报[]')
    try:
        standard_time = datetime.datetime.strptime(time, '%Y年%m月%d日%H:%M')
    except ValueError:
        try:
            standard_time = datetime.datetime.strptime(time, '%Y年%m月%d日 %H:%M')
        except ValueError:
            try:
                standard_time = parse(time)
            except ValueError:
                print('valueerror', time)
                standard_time = ''
    return standard_time


def timeseries_process(news_ts, freq='A-DEC'):
    # 生成一个10年1月1号到今天的全时间序列
    news_ts_all = Series(int(0), index=pd.date_range('1/1/2010', datetime.datetime.now().date(), freq='D'))
    # 下面可以完成补齐缺失时间
    news_ts_merge = news_ts_all + news_ts
    # 两个series中不是共有的元素相加得到NAN，下面是将空值用0代替
    news_ts = news_ts_merge.fillna(int(0))
    news_ts = news_ts[datetime.date(2010, 1, 1):datetime.datetime.now().date()]
    news_ts = news_ts.to_period(freq)  # 'M'代表月，'W'代表周，'Q-DEC'代表正常季度划分,'A-DEC'代表每年指定十二月最后一个日历日为年末
    # print(news_ts)
    # 对非唯一的时间戳进行聚合
    grouped = news_ts.groupby(level=0)
    print(type(grouped))
    news_count = grouped.sum()  # 求均值mean() 计数count()等
    print(grouped.sum())
    return news_count


def news_timeseries(table, count_list_sort=[], top_n=10, freq='A-DEC'):
    # 统计指标三的统计量
    news_ts_list = []
    ts_index = []
    if top_n and count_list_sort:
        timeseries = defaultdict(list)
        for i in range(top_n):
            time_table = table.find({'news_source': re.compile('{}'.format(count_list_sort[i][0]))})
            for u in time_table:
                if u.get('news_pubtime'):
                    time = u.get('news_pubtime')
                    if isinstance(time, datetime.datetime):
                        timeseries[count_list_sort[i][0]].append(time.date())
                    else:
                        # print(time)
                        standard_time = rowtime_processed(time)
                        if standard_time:
                            timeseries[count_list_sort[i][0]].append(standard_time.date())
        print(timeseries)
        legend = []
        judge = 0  #
        for key, values in timeseries.items():
            legend.append(key)
            judge += 1
            # 将时间重新排序
            values.sort()
            # 将原始数据生成时间序列
            news_ts = Series(int(1), index=pd.DatetimeIndex(values))
            # 随时间序列进行缺失处理
            news_count = timeseries_process(news_ts, freq)
            news_ts_list.append((key, list(news_count)))
            if judge == 1:
                data_list = [int(str(i)) for i in list(news_count.index)]
                ts_index.append(('date', data_list))  # 把横坐标页加进去
        news_ts_list = [('legend', legend)] + ts_index + news_ts_list
        text_save2(news_ts_list, filename=r'analysis_data\medias_ts.txt')
    else:
        time_table = table.find()
        timeseries = []
        for u in time_table:
            if u.get('news_pubtime'):
                time = u.get('news_pubtime')
                if isinstance(time, datetime.datetime):
                    timeseries.append(time.date())
                else:
                    # print(time)
                    standard_time = rowtime_processed(time)
                    if standard_time:
                        timeseries.append(standard_time.date())
        timeseries.sort()
        # 将原始数据生成时间序列
        news_ts = Series(int(1), index=pd.DatetimeIndex(timeseries))
        # 对时间序列进行缺失处理
        news_count = timeseries_process(news_ts, freq)
        data_list = [int(str(i)) for i in list(news_count.index)]
        news_ts_list.append(('date', data_list))  # 把横坐标页加进去
        news_ts_list.append(('每年新闻报道总量', list(news_count)))
        text_save2(news_ts_list, filename=r'analysis_data\news_ts.txt')


def main():
    stopwords = [line.rstrip() for line in open('stopwords_gbk.txt', 'r')]
    # 使用自定义停用词集合
    analyse.set_stop_words("stopwords_utf-8.txt")
    # 读取数据库的集合
    table = read_from_mongodb()
    # 切分所有的文章，得到所有切分到的词
    news_content_cut_all = cut_words(table, stopwords)
    # 统计所有切分到的词
    all_words_count = count_words(news_content_cut_all)
    # print(all_words_count)
    # 得到关键词
    keywords = get_keywords(table)
    # 将得到的关键词转换为dataframe格式
    # keywords_dataframe = DataFrame(keywords[1])
    # keywords_dataframe.to_csv('../keywords_content.csv') # 存储
    # 统计得到的关键词
    keywords_count = count_words(keywords[0])
    print(keywords_count)
    # 读取指标文件，转换为字典便于后面对词分类
    words_index_dict = get_index_words(filepath='words_index.xlsx')
    # 将得到的所有关键词按指标进行分类
    classify_words_results = classify_words(words_index_dict, keywords_count)
    n = 0
    index_five = []
    for key, values in classify_words_results.items():
        print(key, 'values:', values)
        n += 1
        text_save(values, filename='index_set\{name}指标.txt'.format(name=key))

        if len(values) > 3:
            index_five = index_five + values[:3]
        else:
            index_five = index_five + values
    # 统计指标五的统计量
    text_save(index_five, filename=r'analysis_data\wordcloud.txt')
    print('指标个数：', n)
    print('classify_words_results:', classify_words_results)
    # 统计所有新闻来源
    medias = count_medias(table)
    count_list_sort = count_words(medias)
    print(count_list_sort)
    n = 0
    for i in count_list_sort:
        if i[1] < 5:
            n += 1
    print(n)
    text_save(count_list_sort[:20], filename='news_source.txt')

    # 统计指标一的各统计量
    total_count(table, medias, filepath='index_set\市级地点指标.txt')
    # 统计指标二的统计量
    top10_medias(count_list_sort)
    # 统计指标三的统计量
    news_timeseries(table, count_list_sort=count_list_sort, top_n=10,
                    freq='A-DEC')  # 'M'代表月，'W'代表周，'Q-DEC'代表正常季度划分,'A-DEC'代表每年指定十二月最后一个日历日为年末
    # 统计指标十一中的新闻总量
    news_timeseries(table, top_n=0, freq='A-DEC')


if __name__ == '__main__':
    main()
