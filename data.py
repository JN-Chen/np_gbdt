import pandas as pd
import numpy as np
import time

def normal_change(arr):
    return arr
def log_change(arr):
    return np.log(arr)
class DataSet:
    def __init__(self, filename, index_col = 'ID', valid_rate = 0.2, data_trans = log_change):
        self.train_df = self.drop_feature(filename, index_col)
        self.train_df, self.target_df = self.splite_train_df(self.train_df, valid_rate)
        self.valid_rate = valid_rate
        self.train_ids = None
        self.valid_ids = None
        self.data_trans = data_trans
        #print(self.train_df.head())
    def drop_feature(self, filename, index_col):
        train_df = pd.read_csv('train.csv', index_col=index_col)
        unique_df = train_df.nunique().reset_index()#get unique value of feature.
        unique_df.columns = ["col_name", "unique_count"]
        constant_df = unique_df[unique_df["unique_count"]==1]#get useless feature which num of value is 1.
        constant_cols = list(constant_df['col_name'].values)
        train_df.drop(constant_cols, axis=1, inplace=True)#drop useless feature.
        train_df = train_df.T.drop_duplicates().T#drop_duplicates feature.
        #print(train_df)
        return train_df
    def splite_train_df(self, train_df, valid_rate):
        target_df = train_df['target']
        train_df.drop('target', axis=1, inplace=True)
        return train_df, target_df
    def get_all_sample(self):
        self.splite_train_valid()
        return self.data_trans(self.train_df.values), \
        self.data_trans(self.target_df.values), \
        np.array([i for i in range(len(self.target_df.values))]), \
        np.array([i for i in range(len(self.target_df.values.T))])
    def get_total_count(self):
        return len(self.target_df.values)
    def get_total_feature(self):
        return len(self.target_df.values.T)
        
    def splite_train_valid(self):
        ids = np.array([i for i in range(len(self.target_df.values))])
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
        feature_ids = np.array([i for i in range(len(self.target_df.values.T))])
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
    data = DataSet(argv[1], 'ID')