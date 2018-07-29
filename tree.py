import numpy as np
class DTree:
    def __init__(self, data, target, feat_id, depth):
        self.data = data
        self.target = target
        self.depth = depth
        self.feat_id = feat_id
        self.min_feat_idx = -1
        self.rnext = None
        self.lnext = None
        self.leaf = 0
        self.output = 0
        self.feature_id = -1
        self.splite_point = 0
    def splite_feature(self, feature):
        min_E = 1000000
        splite_points = set(feature[:])
        left_idx = []
        right_idx = []
        d1 = 0
        d2 = 0
        for s_point in splite_points:#iterate for all splite points.
            idx_left = np.argwhere(feature < s_point).flatten()
            idx_right = np.argwhere(feature >= s_point).flatten()
            if(len(idx_left) == 0):
                d1 = 0
            else:
                target_left = self.target[idx_left]
                average_left = target_left.mean()
                d1 = target_left - average_left
            if(len(idx_right) == 0):
                d2 = 0
            else:
                target_right = self.target[idx_right]
                average_right = target_right.mean()
                d2 = target_right - average_right
            e = np.sqrt(np.square(d1).sum()) + np.sqrt(np.square(d2).sum())
            if(e < min_E):
                self.splite_point = s_point
                #self.feature_id = iter
                left_idx = idx_left[:]
                right_idx = idx_right[:]
                min_E = e
        return min_E, left_idx, right_idx
    def map_first_element(self, ele):
        return ele[0]
    def splite_data(self):
        En_data = list(map(self.splite_feature, self.data))
        Entropys = np.array(list(map(self.map_first_element, En_data)))
        min_idx = np.argmin(Entropys)
        min_idx = np.array(min_idx).min()
        self.feature_id = self.feat_id[min_idx]
        self.min_feat_idx = min_idx
        left_idx = En_data[min_idx][1]
        right_idx = En_data[min_idx][2]
        return left_idx, right_idx
    def construct(self):
        if (self.depth <= 0):#leaf node, return mean of target.
            self.leaf = 1
            self.output = self.target.mean()
            #print("leafnode output=%f" % self.output)
            return
        left_idx, right_idx = self.splite_data()
        drop_feature_data = np.delete(np.array(self.data), self.min_feat_idx, axis = 0)
        drop_feature_ids = np.delete(np.array(self.feat_id), self.min_feat_idx)
        #print("drop feat id = %d" % self.feat_id[self.min_feat_idx])
        #print(drop_feature_ids.shape)
        if(len(left_idx) > 0):
            self.lnext = DTree(drop_feature_data[:,left_idx], self.target[left_idx], drop_feature_ids, self.depth - 1)
            self.lnext.construct()
        if(len(right_idx) > 0):
            self.rnext = DTree(drop_feature_data[:,right_idx], self.target[right_idx], drop_feature_ids, self.depth - 1)
            self.rnext.construct()

    def get_output(self, data, ids, output):
        if(self.leaf == 1):
            output[ids] = self.output
            return
        feature = data[self.feature_id, :]
        #print("get feat id = %d" % self.feature_id)
        idx_left = np.argwhere(feature < self.splite_point).flatten()
        idx_right = np.argwhere(feature >= self.splite_point).flatten()
        drop_feature_data = np.delete(data, self.feature_id, axis = 0)
        if(self.lnext != None and self.rnext != None):
            if(len(idx_left) > 0):
                self.lnext.get_output(drop_feature_data[:,idx_left], ids[idx_left], output)
            if(len(idx_right) > 0):
                self.rnext.get_output(drop_feature_data[:,idx_right], ids[idx_right], output)
        elif(self.lnext != None and self.rnext == None):
            self.lnext.get_output(drop_feature_data[:,:], ids[:], output)
        elif(self.lnext == None and self.rnext != None):
            self.rnext.get_output(drop_feature_data[:,:], ids[:], output)
        else:
            raise RuntimeError("panic!")
        pass