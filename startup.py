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
    dateset.correct_na()
    dateset.feature_transfer()
    gbdt = GBDT(2000, 0.1, 0.1, 0.1)
    gbdt.fit(dateset)