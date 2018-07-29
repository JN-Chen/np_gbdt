from random import sample
import numpy as np
from tree import DTree
class GBDT:
    def __init__(self, max_iter = 1000, sample_rate = 0.5, feature_rate = 0.1,max_depth = 10, learn_rate = 0.2):
        self.max_iter = max_iter
        self.sample_rate = sample_rate
        self.max_depth = max_depth
        self.learn_rate = learn_rate
        self.feature_rate = feature_rate
        self.dts = []
        self.steps = []
    def update_res(self, target_y, F):
        return np.array(target_y) - np.array(F)
    def caculate_step(self, train_predict, train_target, rsme):
        term1 = np.multiply(train_predict, train_target)
        term2 = np.square(train_predict)
        '''if(rsme >= 1.7):
            return 0.1
        elif(rsme >= 1.5 and rsme < 1.7):
            return 0.05
        else:
            return 0.01'''
        #return term1.sum()/(term2.sum() + 0.001)
        return 0.05
    #def predict(self, )
    def fit(self, dataset):
        total_data_count = dataset.get_total_count()
        total_fea_count = dataset.get_total_feature()
        Fm = [0 for i in range(total_data_count)]#record Fmֵ, m=0,1,2,3,4,5......max_iter
        predict = np.array([0 for i in range(total_data_count)])#get predict buffer
        sample_count = int(total_data_count*self.sample_rate)
        feature_count = int(total_fea_count*self.feature_rate)
        data_x, data_y, sample_id, feature_id = dataset.get_all_sample()#get total samples
        valid_data, valid_target, valid_id = dataset.get_valid_sample(data_x, data_y)
        valid_rsme = 10.
        for iter in range(self.max_iter):
            train_data, sample_target, train_id, feat_id = dataset.get_train_sample(data_x, data_y, sample_count, feature_count)
            res = self.update_res(data_y, Fm)#get residual of all samples
            train_target = res[train_id]#get residual of train samples
            dt = DTree(train_data.T, train_target, feat_id, 20)
            dt.construct()
            dt.get_output(data_x.T, sample_id, predict)#use a single tree to predict all data
            step = self.caculate_step(predict[train_id], res[train_id], valid_rsme)
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
            print("train rsme=%f valid rsme = %f" % (train_rsme, valid_rsme))
            #break