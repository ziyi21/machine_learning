#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'
import matplotlib.pyplot as plt
import matplotlib
def sigmoid(h):
    return 1.0 / (1.0 + np.exp(-h))
if __name__ == '__main__':
    matplotlib.rcParams['axes.unicode_minus']=False
    x = np.arange(-20, 20, 0.1) # 定义x的范围，像素为0.1
    h2 = np.arange(-10, 10, 0.1)
    f_x = sigmoid(x) # sigmoid为上面定义的函数
    plt.figure(1)
    plt.subplot(211)
    plt.axvline(0.0, color='k') # 在坐标轴上加一条竖直的线，0.0为竖直线在坐标轴上的位置
    plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted') # 加水平间距通过坐标轴
    plt.axhline(y=0.5, ls='dotted', color='k') # 加水线通过坐标轴
    plt.yticks([0.0, 0.5, 1.0]) # 加y轴刻度
    plt.ylim(-0.1, 1.1) # 加y轴范围
    plt.xlabel('x')
    plt.ylabel('$f(x)$')
    plt.plot(x, f_x,label='Sigmoid')
    plt.legend(loc='upper left')
    plt.subplot(212)
    plt.xlim(-10,10)
    plt.axvline(0.0, color='k') # 在坐标轴上加一条竖直的线，0.0为竖直线在坐标轴上的位置
    plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted') # 加水平间距通过坐标轴
    plt.axhline(y=0.5, ls='dotted', color='k') # 加水线通过坐标轴
    plt.yticks([0.0, 0.5, 1.0]) # 加y轴刻度
    plt.ylim(-0.1, 1.1) # 加y轴范围
    plt.xlabel('x')
    plt.ylabel('$f(x)$')
    plt.plot(x,f_x,label='small',color='r')
    plt.legend(loc='lower right')
    plt.show()


