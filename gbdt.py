from random import sample
import numpy as np
from tree import Res_Tree
from tree import Xgb_Tree
class GBDT:
    def __init__(self, max_iter = 1000, sample_rate = 0.5, feature_rate = 0.5,max_depth = 10, learn_rate = 0.05):
        self.max_iter = max_iter
        self.sample_rate = sample_rate
        self.max_depth = max_depth
        self.learn_rate = learn_rate
        self.feature_rate = feature_rate
        self.dts = []
        self.steps = []
    def update_res(self, target_y, F):
        return np.array(target_y) - np.array(F)

    def fit(self, dataset):
        dataset.divide_data_ids()

        data_x, data_y, sample_id, _ = dataset.get_all_sample()#get total samples
        test_x, test_id = dataset.get_test_data()
    
        total_data_count = dataset.get_sample_count()
        total_feat_count = dataset.get_feature_count()
        total_test_count = dataset.get_test_count()

        Fm_sample = [0. for i in range(total_data_count)]#record Fmֵ, m=0,1,2,3,4,5......max_iter
        sample_predict = np.array([0. for i in range(total_data_count)])#get predict buffer
        
        Fm_test = [0. for i in range(total_test_count)]
        test_predict = np.array([0. for i in range(total_test_count)])

        sample_count = int(total_data_count*self.sample_rate)
        feature_count = int(total_feat_count*self.feature_rate)
        _, valid_target, valid_id = dataset.get_valid_sample()
        #print(valid_data)
        valid_rsme = 10.
        step = self.learn_rate
        for iter in range(self.max_iter):
            train_data, sample_target, train_id, feat_id = dataset.get_train_sample(sample_count, feature_count)
            res = self.update_res(data_y, Fm_sample)#get residual of all samples
            train_target = res[train_id]#get residual of train samples
            dt = Res_Tree(train_data.T, train_target, feat_id, self.max_depth)
            dt.construct()
            dt.get_output(data_x.T, sample_id, sample_predict)#use a single tree to predict all data
            Fm_sample = Fm_sample + sample_predict*step#caculate final output

            dt.get_output(test_x.T, test_id, test_predict)#use a single tree to predict all data
            Fm_test = Fm_test + test_predict*step#caculate final output
            #self.dts.append(dt)
            #self.steps.append(step)
            
            train_output = Fm_sample[train_id]
            train_res = sample_target - train_output
            train_rsme = np.sqrt(np.square(train_res).mean())
            valid_output = Fm_sample[valid_id]
            valid_res = valid_target - valid_output
            valid_rsme = np.sqrt(np.square(valid_res).mean())
            print("step %d train rsme=%f valid rsme = %f" % (iter, train_rsme, valid_rsme))
        return Fm_test