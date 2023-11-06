from preprocessing import PreProcessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import test


class Adaline:
    def __init__(self):
        self.weights = []
        self.mse = 0

    def train(self, x_train, y_train, bias, m, threshold):
        # create weights
        num_samples, num_features = x_train.shape
        self.weights = np.random.rand(num_features)

        # bias checking
        if bias:
            self.weights = np.random.rand(num_features + 1)  # +1 for bias
            x_train.insert(0, 'x0', 1)

        # adaline algorithm
        iterations = 1000
        while True:
            for i in range(0, num_samples):
                # get actual class
                actual_class = y_train.iloc[i].values

                # get prediction value
                values = x_train.iloc[i].values
                prediction = np.dot(self.weights, values)

                # calc error
                error = actual_class - prediction

                # update weights
                self.weights = self.weights + (m * error * values)

            # calc MSE
            sum_error = 0
            for i in range(0, num_samples):
                # get actual class
                actual_class = y_train.iloc[i].values

                # get prediction value
                values = x_train.iloc[i].values
                prediction = np.dot(self.weights, values)

                # calc error
                error = actual_class - prediction
                error = (error ** 2) / 2
                sum_error += error

            mse = (1 / num_samples) * sum_error
            mse = np.round(mse, 2)
            print("mse", mse)
            self.mse = mse
            if mse <= threshold or iterations == 0:
                break

            iterations -= 1

    def adaline_test(self):
        test.test(self)

    def ploting(self):
        test.ploting(self)

    def confusion_matrix(self):
        test.confusion_matrix(self)

    def accuracy_score(self):
        test.accuracy_score(self)


# CONSOLE TEST
# ft = ["Area", "Perimeter"]
# cls = ["BOMBAY", "CALI"]
# pre = PreProcessing()
# pre.read_data("Dry_Bean_Dataset.csv", ft, cls)
#
# pre.split_data(40)
# pre.normalize_train_data()
#
# x_train = pre.x_train
# y_train = pre.y_train
#
# x_test = pre.x_test
# y_test = pre.y_test
#
# ad = Adaline()
# ad.train(x_train, y_train, 1, 0.1, 0.01)
# print(ad.mse)
