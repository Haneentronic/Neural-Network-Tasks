import numpy as np


class Evaluate:
    def __init__(self, prediction, actual, num_classes):
        self.prediction = prediction
        self.actual = actual
        self.num_classes = num_classes
        self.confusion_matrix = None

    def calculate_confusion_matrix(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for i in range(len(self.prediction)):
            ind1 = list(self.prediction[i]).index(max(self.prediction[i]))
            ind2 = list(self.actual[i]).index(max(self.actual[i]))
            self.confusion_matrix[ind2][ind1] += 1

    def calculate_accuracy(self):
        sum = 0
        for i in range(self.num_classes):
            sum += self.confusion_matrix[i][i]
        return sum / len(self.prediction)
