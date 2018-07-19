#-*- coding:utf-8 -*- 
__author__ = 'Oct. Chen'
"""
"""
from data import DataSet
from gbdt import GBDT
import numpy as np
def log_change(arr):
    return np.log(arr + 1.)
if __name__ == '__main__':
    data_file = './train.csv'
    dateset = DataSet(data_file, data_trans= log_change)
    gbdt = GBDT()
    gbdt.fit(dateset)