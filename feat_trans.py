import pandas as pd
import numpy as np

BIN = 20
TYPE_PROJECT = 1
TYPE_MAP = 2

class DiscreteTrans():
    def __init__(self):
        self.__dic = {}
        self.__value = 0
#protected method
    def _setup_dic(self, name):
        self.__dic[name] = self.__value
        self.__value = self.__value + 1
    def _find(self, name):
        return name in self.__dic
    def _map_transfer(self, df):
        df = df.map(self.__dic)
        return df

class ContinuationTrans():
    def __init__(self, bin = BIN):
        self._bin = bin
        self.__label = []

#protected method
    def _setup_bin(self):
        for i in range(self._bin):
            self.__label.append(i)
    def _project_transfer(self, df):
        return pd.cut(df, self._bin, labels = self.__label)

class FeatTransfer(ContinuationTrans, DiscreteTrans):
    def __init__(self):
        #self.pro = ContinuationTrans()
        #self.map = DiscreteTrans()
        ContinuationTrans.__init__(self)
        DiscreteTrans.__init__(self)
        self.__type = 0

#private method
    def __cst_continuation_trans(self, df):
        self._setup_bin()
    def __cst_discrete_trans(self, df):
        for value in df.values:
            if(self._find(value) == False):
                self._setup_dic(value)
#public method
    def cst_trans(self, df):
        count = df.nunique()
        value = df.values[0]
        if(isinstance(value, (int, float)) == True and count >= self._bin):
            self.__type = TYPE_PROJECT
            self.__cst_continuation_trans(df)
        else:
            self.__type = TYPE_MAP
            self.__cst_discrete_trans(df)
    def transfer(self, df):
        if(self.__type == TYPE_MAP):
            return self._map_transfer(df)
        elif(self.__type == TYPE_PROJECT):
            return self._project_transfer(df)