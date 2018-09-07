import numpy as np
import gc
import time
class DTree:
    def __init__(self, data, target, feat_id, depth):
        self.data = data
        self.target = target
        self.depth = depth
        self.feat_id = feat_id
        self.splite_feat_idx = -1
        self.rnext = None
        self.lnext = None
        self.leaf = 0
        self.output = 0
        self.feature_id = -1
        self.splite_point = 0
    def clear_mem(self):
        del self.data
        del self.target

    def splite_feature(self, feature):
        min_E = 1000000
        splite_points = set(feature[:])
        d1 = 0
        d2 = 0
        sp_return = 0.
        for s_point in splite_points:#iterate for all splite points.
            idx_left = np.argwhere(feature < s_point).flatten()
            idx_right = np.argwhere(feature >= s_point).flatten()
            if(len(idx_left) == 0):
                continue
            else:
                target_left = self.target[idx_left]
                average_left = target_left.mean()
                d1 = target_left - average_left
            if(len(idx_right) == 0):
                continue
            else:
                target_right = self.target[idx_right]
                average_right = target_right.mean()
                d2 = target_right - average_right
            e = np.abs(d1).sum() + np.abs(d2).sum()
            if(e < min_E):
                sp_return = s_point
                min_E = e
            #print(e)
        return [min_E,  sp_return]
    def map_first_element(self, ele):
        return ele[0]
    def splite_data(self):
        En_data = list(map(self.splite_feature, self.data))
        Entropys = np.array(list(map(self.map_first_element, En_data)))
        splite_idx = np.argmin(Entropys)
        splite_idx = np.array(splite_idx).min()
        self.feature_id = self.feat_id[splite_idx]
        self.splite_feat_idx = splite_idx
        self.splite_point = En_data[splite_idx][1]
        left_idx = np.argwhere(self.data[splite_idx] < self.splite_point).flatten()
        right_idx = np.argwhere(self.data[splite_idx] >= self.splite_point).flatten()
        #print("l : r = %d : %d", len(left_idx), len(right_idx))
        return left_idx, right_idx
    def construct(self, ret = 0):
        if (ret == 1 or self.depth <= 0 or\
         len(self.data) == 1 or len(self.data.T) == 1 ):#leaf node, return mean of target.
            self.leaf = 1
            self.output = self.target.mean()
            self.clear_mem()
            #print("leafnode output=%f" % self.output)
            return
        left_idx, right_idx = self.splite_data()
        drop_feature_data = np.delete(np.array(self.data), self.splite_feat_idx, axis = 0)
        drop_feature_ids = np.delete(np.array(self.feat_id), self.splite_feat_idx)
        ret = 0
        if(len(left_idx) == 0 or len(right_idx) == 0):
            ret = 1
        if(len(left_idx) > 0):
            self.lnext = DTree(drop_feature_data[:,left_idx], self.target[left_idx], drop_feature_ids, self.depth - 1)
            self.lnext.construct(ret)
        if(len(right_idx) > 0):
            self.rnext = DTree(drop_feature_data[:,right_idx], self.target[right_idx], drop_feature_ids, self.depth - 1)
            self.rnext.construct(ret)
        if(len(left_idx) == 0 and len(right_idx) == 0):
            #print()
            print(self.feature_id)
            print(self.splite_point)
            raise RuntimeError("construct panic!")
        self.clear_mem()
    def get_output(self, data, ids, output):
        if(self.leaf == 1):
            output[ids] = self.output
            return
        #print("get feat id = %d" % self.feature_id)
        #print(len(data))
        feature = data[self.feature_id, :]
        idx_left = np.argwhere(feature < self.splite_point).flatten()
        idx_right = np.argwhere(feature >= self.splite_point).flatten()
        if(self.lnext != None and self.rnext != None):
            if(len(idx_left) > 0):
                self.lnext.get_output(data[:,idx_left], ids[idx_left], output)
            if(len(idx_right) > 0):
                self.rnext.get_output(data[:,idx_right], ids[idx_right], output)
        elif(self.lnext != None and self.rnext == None):
            self.lnext.get_output(data[:,:], ids[:], output)
        elif(self.lnext == None and self.rnext != None):
            self.rnext.get_output(data[:,:], ids[:], output)
        else:
            raise RuntimeError("panic!")
        pass