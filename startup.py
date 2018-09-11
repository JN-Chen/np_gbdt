#-*- coding:utf-8 -*- 
__author__ = 'Oct. Chen'
"""
"""
from data import DataSet
from gbdt import GBDT
import numpy as numpy

if __name__ == '__main__':
    data_file = './train_1.csv'
    dataset = DataSet(data_file, index_col = 'Id', target_col = 'SalePrice')
    #dataset = DataSet(data_file, index_col = 'ID', target_col = 'target')
    dataset.drop_feature_by_missed_rate()
    dataset.fix_na()
    dataset.feature_discrete(skip_cols = ['Id', 'SalePrice'])
#    dataset.feature_discrete(skip_cols = ['ID', 'target'])
    dataset.divide_data()
    dataset.hold_feat_by_corr(0.22)
    #asdsadad
    dataset.target_trasfer()
    gbdt = GBDT(4000, 0.1, 0.3, 5, learn_rate = 0.01)
    gbdt.fit(dataset)