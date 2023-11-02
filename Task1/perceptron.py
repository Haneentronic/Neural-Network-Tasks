import numpy as np
import random

class perceptron:
    def init(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.ephocs = None
        self.learningrate = None
        self.bais = None
    def Perceptron(self):
        if self.bais:
            bais = random.random()
        else:
            bais=0
        w = np.random.randn(2)
        for n in range(self.ephocs):
            for i in range(self.x_train.shape[0]):
                y = np.sign(np.dot(self.x_train.iloc[i, :], w.T) + bais)
                if y != self.y_train[i]:
                    loss = self.y_train[i] - y
                    w = w + (np.dot(loss * self.learningrate, self.x_train.iloc[i, :]))
        return np.dot(w,self.x_test)