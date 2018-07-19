from random import sample
import numpy as np
from tree import DTree
class GBDT:
    def __init__(self, max_iter = 1000, sample_rate = 0.8, max_depth = 10, learn_rate = 0.2):
        self.max_iter = max_iter
        self.sample_rate = sample_rate
        self.max_depth = max_depth
        self.learn_rate = learn_rate
        self.dts = []
        self.steps = []
    def update_res(self, target_y, F):
        return np.array(target_y) - np.array(F)
    def caculate_step(self, train_predict, train_target):
        term1 = np.multiply(train_predict, train_target)
        term2 = np.square(train_predict)
        return term1.sum()/(term2.sum() + 0.001)
    #def predict(self, )
    def fit(self, dataset):
        total_data_count = dataset.get_total_count()
        Fm = [0 for i in range(total_data_count)]#record Fmֵ, m=0,1,2,3,4,5......max_iter
        predict = np.array([0 for i in range(total_data_count)])#get predict buffer
        sample_count = int(total_data_count*self.sample_rate)
        data_x, data_y, id_sample = dataset.get_all_sample()#get total samples
        for iter in range(self.max_iter):
            train_data, sample_target, train_id = dataset.get_train_sample(data_x, data_y, sample_count)
            valid_data, valid_target, valid_id = dataset.get_valid_sample(data_x, data_y)
            res = self.update_res(data_y, Fm)#get residual of all samples
            train_target = res[train_id]#get residual of train samples
            dt = DTree(train_data.T, train_target, 2)
            dt.construct()
            dt.get_output(data_x.T, id_sample, predict)#use a single tree to predict all data
            step = self.caculate_step(predict[train_id], res[train_id])
            Fm = Fm + predict*step#caculate final output
            self.dts.append(dt)
            self.steps.append(step)
            
            train_output = Fm[train_id]
            train_res = sample_target - train_output
            train_rsme = np.sqrt(np.square(train_res).mean())
            valid_output = Fm[valid_id]
            valid_res = valid_target - valid_output
            valid_rsme = np.sqrt(np.square(valid_res).mean())
            print("train rsme=%f valid rsme = %f" % (train_rsme, valid_rsme))