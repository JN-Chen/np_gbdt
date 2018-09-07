import pandas as pd
import numpy as np
import time

def normal_change(arr):
    return arr
def log_change(arr):
    return np.log1p(np.array(arr, dtype=np.float32))

class DataBase(object):
    def __init__(self, filename, index_col = 'ID', drop_col_list = None):
        self.data_df, self.orgin_idx = self.drop_feature(filename, index_col, drop_col_list)
    def drop_feature(self, filename, index_col, drop_col_list):#drop drop_duplicates feature and useless feature
        train_df = pd.read_csv(filename, index_col=index_col)
        orgin_idx = train_df.index
        unique_df = train_df.nunique().reset_index()#get unique value of feature.
        unique_df.columns = ["col_name", "unique_count"]
        constant_df = unique_df[unique_df["unique_count"]==1]#get useless feature which num of value is 1. 
        #the useless feature only have one value
        constant_cols = list(constant_df['col_name'].values)
        train_df.drop(constant_cols, axis=1, inplace=True)#drop useless feature.
        if(drop_col_list != None):
            for drop_col in drop_col_list:
                train_df.drop(drop_col, axis=1, inplace=True)#drop specify feature
        train_df = train_df.T.drop_duplicates().T#drop_duplicates feature.
        return train_df, orgin_idx

class DiscreteTrans():
    def __init__(self, name, value = 0):
        self.featname = name
        self.dic = {}
        self.value = value
    def set(self, name):
        self.dic[name] = self.value
        self.value = self.value + 1
    def get(self, name):
        return self.dic[name]
    def find(self, name):
        return name in self.dic
    def transfer(self, df):
        df = df.map(self.dic)
        return df
class ContinuationTrans():
    def __init__(self, name, value = 0):
        pass
    def transfer(self, df):
        pass

class DataTransfer(DataBase):
    def __init__(self, filename, index_col = 'ID', drop_col_list = None):
        DataBase.__init__(self, filename, index_col, drop_col_list)
        self.FeatTransArray = []
    def na_detect(self, threshold = 0.):#ratio of na > threshold will be droped
        na_count = self.data_df.isnull().sum().sort_values(ascending=False)
        na_rate = na_count / len(self.data_df)
        return na_rate[na_rate > threshold].index

    def transfer_map(self, feature):
        value = self.data_df[feature].values[0]
        if(isinstance(value, (int, float)) == False):
            featrans = DiscreteTrans(feature)
            for value in self.data_df[feature].values:
                if(featrans.find(value) == False):
                    featrans.set(value)
            self.FeatTransArray.append(featrans)
            self.data_df[feature] = featrans.transfer(self.data_df[feature])
        return feature

    def fill_na_map(self, na_feature):
        max_freq_value = self.data_df[na_feature].value_counts().index[0]
        self.data_df[na_feature] = self.data_df[na_feature].fillna(max_freq_value)
        return na_feature

    def get_na_count(self):
        return np.sum(self.data_df.isnull().sum().values)#left feature with na in the value will be fixed
    def drop_feature_by_missed_rate(self, miss_value_rate = 0.35):
        drop_idx = self.na_detect(miss_value_rate)
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
        fix_idx = self.na_detect()
        print('###left na feature###')
        print(fix_idx.values)
        list(map(self.fill_na_map, fix_idx.values))#fill na with max freq value
        na_count = self.get_na_count()
        print("na count=%d" % na_count)

    def feature_discrete(self):
        features = self.data_df.columns.values
        list(map(self.transfer_map, features))#fill na with max freq value
        #print(self.data_df)

class DataSet(DataTransfer):
    def __init__(self, filename, index_col = 'ID', target_col = 'target',  valid_rate = 0.2,\
     data_trans = normal_change, target_trans = log_change, drop_col_list = None):
        #self.data_df = self.drop_feature(filename, index_col)
        DataTransfer.__init__(self, filename, index_col, drop_col_list)
        self.train_df = None
        self.target_df = None
        self.valid_rate = valid_rate
        self.train_ids = None
        self.valid_ids = None
        self.data_trans = data_trans
        self.target_trans = target_trans
        self.target_col = target_col
        #print(self.train_df.head())
    def get_train_target(self, data_df, valid_rate, target_col):
        target_df = data_df[target_col]
        train_df = data_df.drop(target_col, axis=1, inplace=False)
        return train_df, target_df
    def get_all_sample(self):
        self.train_df, self.target_df = self.get_train_target(self.data_df, self.valid_rate, self.target_col)
        self.splite_train_valid()
        train_data = self.data_trans(self.train_df.values)
        target_data = self.target_trans(self.target_df.values)
        data_ids = np.array([i for i in range(self.get_total_count())])
        feature_ids = np.array([i for i in range(self.get_feature_count())])
        return train_data, target_data, data_ids, feature_ids
        
    def get_total_count(self):
        return len(self.target_df.values)
    def get_feature_count(self):
        return len(self.train_df.values.T)
        
    def splite_train_valid(self):
        ids = np.array([i for i in range(self.get_total_count())])
        np.random.shuffle(ids)
        splite_len = int(len(ids) * (1 - self.valid_rate))
        self.train_ids = ids[:splite_len]
        self.valid_ids = ids[splite_len:]

    def get_train_sample(self, data, target, sample = 0, feat = 0):
        #print(int(time.time()))
        np.random.seed(int(time.time()))
        np.random.shuffle(self.train_ids)
        if(sample != 0):
            train_id = self.train_ids[:sample]
        else:
            train_id = self.train_ids[:]
        feature_ids = np.array([i for i in range(self.get_feature_count())])
        np.random.shuffle(feature_ids)
        if(feat == 0):
            feat_id = feature_ids[:]
        else:
            feat_id = feature_ids[:feat]
        rand_train_data = data[train_id, :].T[feat_id, :].T
        rand_train_target = target[train_id]
        return rand_train_data, rand_train_target, train_id, feat_id
    def get_valid_sample(self, data, target, sample = 0):
        np.random.shuffle(self.valid_ids)
        if(sample != 0):
            valid_ids = self.valid_ids[:sample]
        else:
            valid_ids = self.valid_ids[:]
        rand_valid_data = data[valid_ids, :]
        rand_valid_target = target[valid_ids]
        return rand_valid_data, rand_valid_target, valid_ids


if __name__ == "__main__":
    from sys import argv
    data = DataTransfer('train_1.csv', 'Id')
    #data.correct_na()
    #data.feature_transfer()