from Task1.preprocessing import PreProcessing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

            self.mse = mse
            if mse <= threshold or iterations == 0:
                break

            iterations -= 1

    def test(self, x_test, y_test, bias):
        num_of_samples = x_test.shape[0]
        y_pred = []

        if bias:
            x_test.insert(0,'x0',1)

        for i in range(num_of_samples):
            values = x_test.iloc[i].values
            pred = np.dot(self.weights, values)
            y_pred.append(1 if pred >= 0 else -1)

        y_pred = np.array(y_pred)
        y_test = y_test.values

        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        # calculate the confusion matrix

        for i,j in zip(y_test,y_pred):
            if i == 1 & j == 1:
                true_positive += 1
                continue
            if i == -1 & j == 1:
                false_negative += 1
                continue
            if i == -1 & j == -1:
                true_negative += 1
                continue
            if i == 1 & j == -1:
                false_negative += 1
                continue

        confusion_matrix = np.array([[true_positive, false_positive], [false_negative, true_negative]])

        # CONSOLE
        accuracy = (true_positive + true_negative) / num_of_samples
        print("Adaline Accuracy", accuracy)

        # GUI -> Confusion matrix plotting
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Reds',)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Adaline Confusion Matrix')
        plt.show()

