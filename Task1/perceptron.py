import numpy as np
import random
import pandas as pd

class Perceptron:
    def __init__(self, data, ephocs, learningrate, bais):
        self.data = data
        self.ephocs = ephocs
        self.learningrate = learningrate
        self.bais = bais
        self.w = None
        self.prediction = None

    def perceptron_train(self):
        if self.bais:
            self.bais = random.random()
        else:
            self.bais = 0
        self.w = np.random.randn(2)
        for n in range(self.ephocs):
            for i in range(self.data.x_train.shape[0]):
                y = np.sign(np.dot(self.data.x_train.iloc[i, :], self.w.T) + self.bais)
                if y != self.data.y_train.iloc[i, -1]:
                    loss = self.data.y_train.iloc[i, -1] - y
                    self.w = self.w + (np.dot(loss * self.learningrate, self.data.x_train.iloc[i, :]))

    def perceptron_test(self):
        self.prediction = np.sign(np.dot(self.data.x_test, self.w.T))

