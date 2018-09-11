from random import sample
import numpy as np
from tree import DTree
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
        data_x, data_y, sample_id, _ = dataset.get_all_sample()#get total samples
        total_data_count = dataset.get_sample_count()
        total_fea_count = dataset.get_feature_count()
        Fm = [0. for i in range(total_data_count)]#record Fmֵ, m=0,1,2,3,4,5......max_iter
        predict = np.array([0. for i in range(total_data_count)])#get predict buffer
        sample_count = int(total_data_count*self.sample_rate)
        feature_count = int(total_fea_count*self.feature_rate)
        _, valid_target, valid_id = dataset.get_valid_sample()
        #print(valid_data)
        valid_rsme = 10.
        for iter in range(self.max_iter):
            train_data, sample_target, train_id, feat_id = dataset.get_train_sample(sample_count, feature_count)
            res = self.update_res(data_y, Fm)#get residual of all samples
            train_target = res[train_id]#get residual of train samples
            dt = DTree(train_data.T, train_target, feat_id, self.max_depth)
            dt.construct()
            dt.get_output(data_x.T, sample_id, predict)#use a single tree to predict all data
            #step = self.caculate_step(predict[train_id], res[train_id], valid_rsme)
            step = self.learn_rate
            Fm = Fm + predict*step#caculate final output
            self.dts.append(dt)
            self.steps.append(step)
            
            train_output = Fm[train_id]
            train_res = sample_target - train_output
            train_rsme = np.sqrt(np.square(train_res).mean())
            valid_output = Fm[valid_id]
            valid_res = valid_target - valid_output
            valid_rsme = np.sqrt(np.square(valid_res).mean())
            #print("%f:%f" % (train_id.mean(), feat_id.mean()))
            print("step %d train rsme=%f valid rsme = %f" % (iter, train_rsme, valid_rsme))
            #_ = valid_res.mean()
            #err_idx = np.argwhere(valid_res > 0.1).flatten()
            #err_res = res[valid_id[err_idx]]
            #print(valid_id[err_idx])
            #print('valid len %d err len %d err res mean %f' %(len(valid_id), len(err_idx), err_res.mean()))
            #break