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
        iterations = 700
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

        print("Adaline Training Ends with MSE = ", self.mse)
        if bias and x_train['x0'].iloc[0] == 1:
            # print("Bias is 1. Deleting the inserted column.")
            x_train.drop('x0', axis=1, inplace=True)

    def plot_decision_boundary(self, x_train, y_train, bias):
        x_train1 = x_train.iloc[:, 0]
        x_train2 = x_train.iloc[:, 1]
        y_train_values = y_train.iloc[:, -1].values

        # Create a scatter plot
        plt.scatter(x_train1[y_train_values == 1], x_train2[y_train_values == 1], label='Class 1', marker='o')
        plt.scatter(x_train1[y_train_values == -1], x_train2[y_train_values == -1], label='Class -1', marker='x')

        # Set labels and title
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Adaline Decision Boundary')

        # plot decision boundary
        x = np.linspace(x_train1.min() - 1, x_train1.max() + 1, 100)
        if bias:
            y = -(self.weights[1] * x + self.weights[0]) / self.weights[2]
        else:
            y = -(self.weights[0] * x + 0) / self.weights[1]

        plt.plot(x, y, color='yellow')
        plt.show()

    def test(self, x_test, y_test, bias):
        num_samples, num_features = x_test.shape

        if bias:
            x_test.insert(0,'x0',1)

        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for i in range(num_samples):
            values = x_test.iloc[i].values
            prediction = np.dot(self.weights, values)
            predicted_class = 1 if prediction >= 0 else -1
            actual_class = y_test.iloc[i].values

            if predicted_class == actual_class:
                if actual_class == 1:
                    true_positive += 1
                else:
                    true_negative += 1

            elif actual_class == 1 and predicted_class == -1:
                false_negative += 1

            elif actual_class == -1 and predicted_class == 1:
                false_positive += 1

        # calc accuracy
        accuracy = (true_positive + true_negative) / num_samples
        print("Adaline Accuracy", accuracy)

        conf_matrix = np.array([[true_negative, false_positive], [false_negative, true_positive]])

        # Plot the confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Actual Negative', 'Actual Positive'],
                    yticklabels=['Predicted Negative', 'Predicted Positive'])
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Adaline Confusion Matrix')
        plt.show()


    def predict(self, x, b):
        if b:
            x = np.append(x, 1)
        # print(x)
        # print(self.weights)
        prediction = np.dot(self.weights, x)
        predicted_class = 1 if prediction >= 0 else -1
        return predicted_class


# CONSOLE TEST
# ft = ["Perimeter", "MajorAxisLength"]
# cls = ["BOMBAY", "SIRA"]
# pre = PreProcessing()
# pre.read_data("Dry_Bean_Dataset.csv", ft, cls)
#
# pre.split_data(40)
# pre.null_handel()
# pre.normalize_train_data()
# pre.normalize_test_data()
#
# x_train = pre.x_train
# y_train = pre.y_train
#
# ad = Adaline()
# ad.train(x_train, y_train, 1, 0.01, 0.01)
# ad.plot_decision_boundary(x_train, y_train, 1)
# ad.test(pre.x_test, pre.y_test, 1)