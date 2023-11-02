import os
import numpy as np
import random
from Task1.preprocessing import *
class perceptron:
    def __init__(self,data,ephocs,learningrate,bais):
        self.data=data
        self.ephocs = ephocs
        self.learningrate = learningrate
        self.bais = bais

    def Perceptron(self):
        if self.bais:
            bais = random.random()
        else:
            bais=0
        w = np.random.randn(2)
        for n in range(self.ephocs):
            for i in range(self.data.x_train.shape[0]):
                y = np.sign(np.dot(self.data.x_train[i, :], w.T) + self.bais)
                if y != self.data.y_train.iloc[i]:
                    loss = self.data.y_train.iloc[i] - y
                    w = w + (np.dot(loss * self.learningrate, self.data.x_train.iloc[i, :]))
        print(w,self.data.x_test)
        return np.dot(w,self.data.x_test)


"""obj=PreProcessing()
obj.read_data("Dry_Bean_Dataset.csv",["Perimeter","Area"],["BOMBAY","CALI"])
print(obj.x)
obj.x_train=obj.x.iloc[:40,:]
obj.y_train=obj.y.iloc[:40,:]
obj.x_test=obj.x.iloc[41:,:]
obj.y_test=obj.y.iloc[41:,:]
print(obj.x_train)
print(obj.x_test)
print(obj.y_train)
print(obj.y_test)"""

obj=PreProcessing()
obj.read_data("Dry_Bean_Dataset.csv",["Perimeter","Area"],["BOMBAY","CALI"])
obj.split_data(40)
obj.null_handel()
obj.normalize_train_data()
obj.normalize_test_data()
o=perceptron(obj,10,.01,1)
print(o.Perceptron())