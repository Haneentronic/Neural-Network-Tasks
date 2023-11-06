import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import PreProcessing
from test import test
from test import ploting
from test import confusion_matrix
from test import accuracy_score


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

    def perceptron_testt(self):
        #self.prediction = np.array(np.sign(np.dot(self.data.x_test, self.w.T)))
        test(self)


    def plotingg(self):
        ploting(self)
        #plt.scatter(self.data.x_train.iloc[:, 0], self.data.x_train.iloc[:, 1], c=self.data.y_train.iloc[:, -1])
        #x = np.linspace(self.data.x_train.iloc[:, 0].min(), self.data.x_train.iloc[:, 0].max(), 100)
        #y = -(self.w[0] * x + self.bais) / self.w[1]
        #plt.plot(x, y, color='red')
        #plt.show()

    def confusion_matrixx(self):
        confusion_matrix(self)
        # Get the unique classes
        #classes = np.unique(np.concatenate((np.array(self.data.y_test.iloc[:,0]), self.prediction)))

        # Initialize the confusion matrix
        #cm = np.zeros((len(classes), len(classes)))

        # Fill the confusion matrix
        #for i in range(len(np.array(self.data.y_test.iloc[:,0]))):
         #   true_class = np.where(classes == np.array(self.data.y_test.iloc[:,0])[i])[0][0]
         #  pred_class = np.where(classes == self.prediction[i])[0][0]
         #   cm[true_class][pred_class] += 1

        #return cm

    def accuracy_scoree(self):
        accuracy_score(self)
        # Calculate the number of correct predictions
        #correct = sum(np.array(self.data.y_test.iloc[:,0]) == self.prediction)

        # Calculate the total number of predictions
        #total = len(np.array(self.data.y_test.iloc[:,0]))

        # Calculate the accuracy score
        #accuracy = correct / total

        #return accuracy


preprocessing = PreProcessing()
preprocessing.read_data("Dry_Bean_Dataset.csv", ["Perimeter", "Area"], ["BOMBAY", "CALI"])
preprocessing.split_data(40)
preprocessing.null_handel()
o = Perceptron(preprocessing, 100, 0.1, 0)
o.perceptron_train()
o.perceptron_testt()
o.confusion_matrixx(o.data.y_test,o.prediction)
print(type(np.array(o.data.y_test.loc[:,0])))
# print(type(o.prediction))
# print(o.confusion_matrix())
# print(o.accuracy_score())
# o.ploting()
