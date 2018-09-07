#-*- coding:utf-8 -*- 
__author__ = 'Oct. Chen'
"""
"""
from data import DataSet
from gbdt import GBDT
import numpy as numpy

if __name__ == '__main__':
    data_file = './train.csv'
    dateset = DataSet(data_file, index_col = 'Id', target_col = 'SalePrice')
    dateset.drop_feature_by_missed_rate()
    dateset.fix_na()
    dateset.feature_discrete()
    gbdt = GBDT(10000, 0.1, 0.3, 5, learn_rate = 0.007)
    gbdt.fit(dateset)