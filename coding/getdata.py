#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ziyi'
import pandas as pd
pd_house_price = pd.read_csv(r'D:\anaconda python\pkgs\scikit-learn-0.19.0-np113py36_0\Lib\site-packages\sklearn\datasets\data\boston_house_prices.csv')
pd_house_price_X = pd_house_price.loc[14]
pd_house_price_y = pd_house_price.loc[:-1]
print(pd_house_price_X)