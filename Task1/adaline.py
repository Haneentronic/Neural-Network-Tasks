from preprocessing import PreProcessing
from main import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Adaline:
    def __init__(self):
        self.weights = []
        self.wrong_classified = 0
        self.mse = 0

    def train(self, x_train, y_train, bias, m, threshold):
        # create weights  [PROBLEM] when bias is 0, it will be updated
        num_samples, num_features = x_train.shape
        self.weights = np.zeros(num_features + 1)  # +1 for bias
        if bias:
            self.weights[0] = 1

        # create new column of ones in the features
        x_train.insert(0, 'x0', 1)

        # y_train classes
        unique_values = y_train.unique()
        mapping = {unique_values[0]: 1, unique_values[1]: -1}
        y_actual = y_train.map(mapping)

        iterations = 100
        # adaline algorithm
        while iterations:
            for i in range(0, num_samples):
                # get actual class
                actual_class = y_actual[i]

                # get prediction value
                values = x_train.iloc[i].to_numpy()
                prediction = np.dot(self.weights, values)

                # get predicted class using signum function
                predicted_class = np.where(prediction >= 0, 1, -1)

                # calc error
                error = actual_class - predicted_class

                # update weights
                self.weights = self.weights + (m * error * values)

            # calc MSE
            sum_error = 0
            cnt = 0
            for i in range(0, num_samples):
                # get actual class
                actual_class = y_actual[i]

                # get prediction value
                values = x_train.iloc[i].to_numpy()
                prediction = np.dot(self.weights, values)

                # get predict class using signum function
                predicted_class = np.where(prediction >= 0, 1, -1)

                # calc error
                error = actual_class - predicted_class
                error = error * error
                sum_error += error

            mse = 0.5 * (sum_error / 2)
            self.mse = mse
            if mse < threshold:
                break

            iterations -= 1

    def test(self, x_test, y_test):
        pass
        # Mariam, code here. Delete pass and do art


# Just for testing and will be REMOVED
pre = PreProcessing()
pre.read_data("Dry_Bean_Dataset.csv", ['Area', 'roundnes'],
              ['BOMBAY', 'CALI'])

adaline = Adaline()
adaline.train(pre.x, pre.y, 2, 0.5, 0.1)
print(adaline.weights)
print(adaline.mse)


# plt.scatter(pre.df['Area'], pre.df['roundnes'])
# plt.xlabel('Area')
# plt.ylabel('Perimeter')
# plt.title('Scatter Plot between Feature1 and Feature2')
# plt.show()
