import pandas as pd
import numpy as np

BIN = 20
TYPE_PROJECT = 1
TYPE_MAP = 2

class DiscreteTrans():
    def __init__(self):
        self.dic = {}
        self.value = 0
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
    def __init__(self, bin = BIN):
        self.min = 0.
        self.max = 0.
        self.bin = bin
        self.label = []
    def set(self):
        for i in range(self.bin):
            self.label.append(i)
    def transfer(self, df):
        return pd.cut(df, self.bin, labels = self.label)

class FeatTransfer():
    def __init__(self):
        self.pro = ContinuationTrans()
        self.map = DiscreteTrans()
        self.type = 0
    def cst_continuation_trans(self, df):
        self.pro.set()
    def cst_discrete_trans(self, df):
        for value in df.values:
            if(self.map.find(value) == False):
                self.map.set(value)
    def cst_trans(self, df):
        count = df.nunique()
        value = df.values[0]
        if(isinstance(value, (int, float)) == True and count >= BIN):
            self.type = TYPE_PROJECT
            self.cst_continuation_trans(df)
        else:
            self.type = TYPE_MAP
            self.cst_discrete_trans(df)
    def transfer(self, df):
        if(self.type == TYPE_MAP):
            return self.map.transfer(df)
        elif(self.type == TYPE_PROJECT):
            return self.pro.transfer(df)