import numpy as np


class Evaluate:
    def __init__(self, prediction, actual, num_classes):
        self.prediction = prediction
        self.actual = actual
        self.num_classes = num_classes
        self.confusion_matrix = None

    def calculate_confusion_matrix(self, predicted):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        #print(self.actual[0].tolist())
        for a, p in zip(self.actual[0].tolist(), predicted):
            self.confusion_matrix[a][np.argmax(p)] += 1

        return self.confusion_matrix

    def calculate_accuracy(self):
        sum = 0
        for i in range(self.num_classes):
            sum += self.confusion_matrix[i][i]
        return sum / len(self.prediction)