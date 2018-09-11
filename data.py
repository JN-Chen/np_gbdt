import pandas as pd
import numpy as np
import time
from feature import FeatTransfer

def normal_change(arr):
    return arr
def log_change(arr):
    return np.log1p(np.array(arr, dtype=np.float32))

class DataBase(object):
    def __init__(self, train_file, index_col = 'ID', drop_col_list = None):
        self.sample_count = 0
        self.feat_count = 0
        self.data_df, self.orgin_idx = self.drop_feature(train_file, index_col, drop_col_list)
    def drop_feature(self, train_file, index_col, drop_col_list):#drop drop_duplicates feature and useless feature
        data_df = pd.read_csv(train_file, index_col=index_col)
        orgin_idx = data_df.index
        unique_df = data_df.nunique().reset_index()#get unique value of feature.
        unique_df.columns = ["col_name", "unique_count"]
        constant_df = unique_df[unique_df["unique_count"]==1]#get useless feature which num of value is 1. 
        #the useless feature only have one value
        constant_cols = list(constant_df['col_name'].values)
        data_df.drop(constant_cols, axis=1, inplace=True)#drop useless feature.
        if(drop_col_list != None):
            for drop_col in drop_col_list:
                data_df.drop(drop_col, axis=1, inplace=True)#drop specify feature
        data_df = data_df.T.drop_duplicates().T#drop_duplicates feature.
        return data_df, orgin_idx

class DataTransfer(DataBase):
    def __init__(self, train_file, index_col = 'ID', drop_col_list = None):
        DataBase.__init__(self, train_file, index_col, drop_col_list)
        self.FeatTransArray = []
    def find_na_feat(self, miss_value_rate = 0.):#ratio of na > threshold will be droped
        na_count = self.data_df.isnull().sum().sort_values(ascending=False)
        na_rate = na_count / len(self.data_df)
        return na_rate[na_rate > miss_value_rate].index

    def fillna_map_func(self, na_feature):
        max_freq_value = self.data_df[na_feature].value_counts().index[0]
        self.data_df[na_feature] = self.data_df[na_feature].fillna(max_freq_value)
        return na_feature

    def get_na_count(self):
        return np.sum(self.data_df.isnull().sum().values)#left feature with na in the value will be fixed
    def drop_feature_by_missed_rate(self, miss_value_rate = 0.35):
        drop_idx = self.find_na_feat(miss_value_rate)
        na_count = self.get_na_count()
        print("na count=%d" % na_count)
        self.data_df.drop(drop_idx, axis=1, inplace=True)#
        print('###drop feature###')
        print(drop_idx.values)
    def fix_na(self):# fill na
        print('###left feature###')
        print(self.data_df.columns.values)
        na_count = self.get_na_count()
        print("na count=%d" % na_count)
        fix_idx = self.find_na_feat()
        print('###left na feature###')
        print(fix_idx.values)
        list(map(self.fillna_map_func, fix_idx.values))#fill na with max freq value
        na_count = self.get_na_count()
        print("na count=%d" % na_count)
    def discrete_map_func(self, feature):
        if feature in self.skip_cols:
            return
        feat_transfer = FeatTransfer()
        feat_transfer.cst_trans(self.data_df[feature])
        self.data_df[feature] = feat_transfer.transfer(self.data_df[feature])
    def feature_discrete(self, skip_cols = []):
        print('feature_discrete')
        features = self.data_df.columns.values
        self.skip_cols = skip_cols
        list(map(self.discrete_map_func, features))
        print('feature_discrete done')

class DataSet(DataTransfer):
    def __init__(self, train_file, index_col = 'ID', target_col = 'target',  valid_rate = 0.2,\
     drop_col_list = None):
        DataTransfer.__init__(self, train_file, index_col, drop_col_list)
        self.target_df = None
        self.valid_rate = valid_rate
        self.train_ids = None
        self.valid_ids = None
        self.target_col = target_col
    def get_train_target(self, data_df, valid_rate, target_col):
        self.target_df = data_df[target_col]
        self.data_df = data_df.drop(target_col, axis=1, inplace=False)
        self.update_count()
        return self.data_df, self.target_df
    def update_count(self):
        self.sample_count = len(self.data_df.values)
        self.feat_count = len(self.data_df.values.T)
    def divide_data(self):
        self.get_train_target(self.data_df, self.valid_rate, self.target_col)
        self.splite_train_valid()

    def target_trasfer(self, function = log_change):
        #print(self.target_df)
        index = self.target_df.index
        name = self.target_df.name
        self.target_df = pd.Series(function(self.target_df.values), index = index, name = name)
        #print(self.target_df)
        #adsfsadfasdfa
    def get_all_sample(self):
        data = self.data_df.values
        target = self.target_df.values
        data_ids = self.get_sample_ids()
        feature_ids = self.get_feature_ids()
        return data, target, data_ids, feature_ids
        
    def get_sample_count(self):
        return self.sample_count
    def get_feature_count(self):
        return self.feat_count
    def get_sample_ids(self):
        return np.array([i for i in range(self.get_sample_count())])
    def get_feature_ids(self):
        return np.array([i for i in range(self.get_feature_count())])
    def splite_train_valid(self):
        ids = self.get_sample_ids()
        np.random.shuffle(ids)
        splite_len = int(len(ids) * (1 - self.valid_rate))
        self.train_ids = ids[:splite_len]
        self.valid_ids = ids[splite_len:]

    def get_train_sample(self, sample = 0, feat = 0):
        #print(int(time.time()))
        data, target, _, _ = self.get_all_sample()
        np.random.seed(int(time.time()))
        np.random.shuffle(self.train_ids)
        if(sample != 0):
            train_id = self.train_ids[:sample]
        else:
            train_id = self.train_ids[:]
        feature_ids = self.get_feature_ids()
        np.random.shuffle(feature_ids)
        if(feat == 0):
            feat_id = feature_ids[:]
        else:
            feat_id = feature_ids[:feat]
        rand_train_data = data[train_id, :].T[feat_id, :].T
        rand_train_target = target[train_id]
        return rand_train_data, rand_train_target, train_id, feat_id
    def get_valid_sample(self, sample = 0):
        data, target, _, _ = self.get_all_sample()
        np.random.shuffle(self.valid_ids)
        if(sample != 0):
            valid_ids = self.valid_ids[:sample]
        else:
            valid_ids = self.valid_ids[:]
        rand_valid_data = data[valid_ids, :]
        rand_valid_target = target[valid_ids]
        return rand_valid_data, rand_valid_target, valid_ids
    def corr(self, feature):
        s_target = pd.Series(list(self.target_df.values))
        s_feature = pd.Series(list(self.data_df[feature].values))
        corr = s_feature.corr(s_target)
        return corr
    def hold_feat_by_corr(self, importance_hold):
        features = self.data_df.columns
        for feature in features:
            value_corr = self.corr(feature)
            if(value_corr < 0):
                value_corr = 0. - value_corr
            if(value_corr < importance_hold):
                self.data_df.drop(feature, axis=1, inplace=True)#
                print('drop feature %s' % feature)
        self.update_count()

    

if __name__ == "__main__":
    from sys import argv
    data = DataTransfer('train_1.csv', 'Id')
    #data.correct_na()
    #data.feature_transfer()