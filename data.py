import pandas as pd
import numpy as np
import time
from feat_trans import FeatTransfer

def normal_change(arr):
    return arr
def log_change(arr):
    return np.log1p(np.array(arr, dtype=np.float32))

class DataBase(object):
    def __init__(self, train_file, test_file, index_col , target_col , drop_col_list = None):
        self.__target_df = None
        self.__sample_count = 0
        self.__test_count = 0
        self.__feat_count = 0
        self.__data_df, self.__test_df = self.__get_data_and_test(train_file, test_file, index_col, drop_col_list)
        self.__get_train_target(self.get_data_df(), target_col)
    def __get_train_target(self, data_df, target_col):
        self.__target_df = data_df[target_col]
        data_df.drop(target_col, axis=1, inplace=True)
        self._update_count()
        return self.get_data_df(), self.__target_df
    def _update_count(self):
        self.__sample_count = len(self.get_data_df().values)
        self.__feat_count = len(self.get_data_df().values.T)
        self.__test_count = len(self.get_test_df().values)

    def _drop_feature(self, cols):
        self.__data_df.drop(cols, axis=1, inplace=True)#drop useless feature.
        self.__test_df.drop(cols, axis=1, inplace=True)
        self._update_count()
    def __get_data_and_test(self, train_file, test_file, index_col, drop_col_list):#drop drop_duplicates feature and useless feature
        data_df = pd.read_csv(train_file, index_col=index_col)
        test_df = pd.read_csv(test_file, index_col=index_col)

        #orgin_idx = data_df.index
        unique_df = data_df.nunique().reset_index()#get unique value of feature.
        unique_df.columns = ["col_name", "unique_count"]
        constant_df = unique_df[unique_df["unique_count"]==1]#get useless feature which num of value is 1. 
        #the useless feature only have one value
        constant_cols = list(constant_df['col_name'].values)
        data_df.drop(constant_cols, axis=1, inplace=True)#drop useless feature.
        test_df.drop(constant_cols, axis=1, inplace=True)

        if(drop_col_list != None):
            data_df.drop(drop_col_list, axis=1, inplace=True)#drop specify feature
            test_df.drop(drop_col_list, axis=1, inplace=True)#drop specify feature
        data_df = data_df.T.drop_duplicates().T#drop_duplicates feature.
        test_df = test_df.T.drop_duplicates().T#drop_duplicates feature.
        return data_df, test_df
    def _get_na_count(self):
        return np.sum(self.__data_df.isnull().sum().values) \
        + np.sum(self.__test_df.isnull().sum().values)#left feature with na in the value will be fixed
    def get_data_df(self):
        return self.__data_df
    def get_target_df(self):
        return self.__target_df
    def set_target_df(self, target_df):
        self.__target_df = target_df
    def get_sample_count(self):
        return self.__sample_count
    def get_feature_count(self):
        return self.__feat_count
    def get_test_df(self):
        return self.__test_df
    def get_test_count(self):
        return self.__test_count

class DataTransfer(DataBase):
    def __init__(self, train_file, test_file, index_col , target_col, drop_col_list = None):
        DataBase.__init__(self, train_file, test_file, index_col, target_col, drop_col_list)
        self.FeatTransArray = []
    def __find_feat_by_miss(self, data_df, miss_rate = 0.):#ratio of na > threshold will be droped
        na_count = data_df.isnull().sum().sort_values(ascending=False)
        na_rate = na_count / len(data_df)
        return na_rate[na_rate > miss_rate].index
    def __map_fill_data_na_func(self, na_feature):
        max_freq_value = self.get_data_df()[na_feature].value_counts().index[0]
        self.get_data_df()[na_feature] = self.get_data_df()[na_feature].fillna(max_freq_value)
        return na_feature
    def __map_fill_test_na_func(self, na_feature):
        max_freq_value = self.get_data_df()[na_feature].value_counts().index[0]
        self.get_test_df()[na_feature] = self.get_test_df()[na_feature].fillna(max_freq_value)
        return na_feature
    def __discrete_map_func(self, feature):
        if feature in self.skip_cols:
            return
        feat_transfer = FeatTransfer()
        feat_transfer.cst_trans(pd.concat([self.get_data_df()[feature], self.get_test_df()[feature]]))
        self.get_data_df()[feature] = feat_transfer.transfer(self.get_data_df()[feature])
        self.get_test_df()[feature] = feat_transfer.transfer(self.get_test_df()[feature])
    def drop_feature_by_missed_rate(self, miss_value_rate = 0.35):
        data_df = self.get_data_df()
        drop_idx = self.__find_feat_by_miss(data_df, miss_value_rate)
        na_count = self._get_na_count()
        print("na count=%d" % na_count)
        #data_df.drop(drop_idx, axis=1, inplace=True)#
        self._drop_feature(drop_idx)
        print('###drop feature###')
        print(drop_idx.values)
    def fix_na(self):# fill na
        data_df = self.get_data_df()
        test_df = self.get_test_df()
        print('###left feature###')
        print(data_df.columns.values)
        na_count = self._get_na_count()
        print("na count=%d" % na_count)
        fix_idx = self.__find_feat_by_miss(data_df)
        print('###left na feature###')
        print(fix_idx.values)
        list(map(self.__map_fill_data_na_func, fix_idx.values))#fill na with max freq value
        na_count = self._get_na_count()
        print("na count=%d" % na_count)
        
        fix_idx = self.__find_feat_by_miss(test_df)
        print('###left na feature###')
        print(fix_idx.values)
        list(map(self.__map_fill_test_na_func, fix_idx.values))#fill na with max freq value
        na_count = self._get_na_count()
        print("na count=%d" % na_count)
    def feature_discrete(self, skip_cols = []):
        print('feature_discrete')
        features = self.get_data_df().columns.values
        self.skip_cols = skip_cols
        list(map(self.__discrete_map_func, features))
        print('feature_discrete done')
class DataAna():
    def __init__(self):
        pass
class DataSet(DataTransfer, DataAna):
    def __init__(self, train_file, test_file, index_col = 'ID', target_col = 'target',  valid_rate = 0.2,\
     drop_col_list = None):
        DataTransfer.__init__(self, train_file, test_file, index_col, target_col, drop_col_list)
        DataAna.__init__(self)
        self.valid_rate = valid_rate
        self.train_ids = None
        self.valid_ids = None

    def target_trasfer(self, function = log_change):
        target_df = self.get_target_df()
        index = target_df.index
        name = target_df.name
        self.set_target_df(pd.Series(function(target_df.values), index = index, name = name))
    def get_all_sample(self):
        data = self.get_data_df().values
        target = self.get_target_df().values
        data_ids = self.get_sample_ids()
        feature_ids = self.get_feature_ids()
        return data, target, data_ids, feature_ids
    
    def get_sample_ids(self):
        return np.array([i for i in range(self.get_sample_count())])
    def get_test_ids(self):
        return np.array([i for i in range(self.get_test_count())])
    def get_feature_ids(self):
        return np.array([i for i in range(self.get_feature_count())])
    def divide_data_ids(self):
        ids = self.get_sample_ids()
        np.random.shuffle(ids)
        splite_len = int(len(ids) * (1 - self.valid_rate))
        self.train_ids = ids[:splite_len]
        self.valid_ids = ids[splite_len:]
    def get_test_data(self):
        return self.get_test_df().values, self.get_test_ids()
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
        s_target = pd.Series(list(self.get_target_df().values))
        s_feature = pd.Series(list(self.get_data_df()[feature].values))
        corr = s_feature.corr(s_target)
        return corr
    def hold_feat_by_corr(self, importance_hold):
        features = self.get_data_df().columns
        for feature in features:
            value_corr = self.corr(feature)
            if(value_corr < 0):
                value_corr = 0. - value_corr
            if(value_corr < importance_hold):
                self._drop_feature(feature)
                print('drop feature %s' % feature)
        self._update_count()
