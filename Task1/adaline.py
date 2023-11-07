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
        iterations = 500
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

        x_train1 = x_train.iloc[:, 0]
        x_train2 = x_train.iloc[:, 1]
        y_train_values = y_train.iloc[:, -1].values  # Convert to a NumPy array

        class_0_mask = y_train_values == 1
        class_1_mask = y_train_values == -1

        plt.plot(x_train1[class_0_mask], x_train2[class_0_mask], label='Class 0', marker='o', c='red')
        plt.plot(x_train1[class_1_mask], x_train2[class_1_mask], label='Class 1', marker='x', c='blue')

        x = np.linspace(x_train1.min() - 1, x_train1.max() + 1, 100)
        if bias:
            y = -(self.weights[1] * x + self.weights[0]) / self.weights[2]
        else:
            y = -(self.weights[0] * x + 0) / self.weights[1]

        plt.plot(x, y, color='yellow')
        # Add labels and legend

        plt.show()

    def test(self, x_test, y_test, bias):
        num_of_samples = x_test.shape[0]
        y_pred = []

        if bias:
            x_test.insert(0,'x0',1)

        for i in range(num_of_samples):
            values = x_test.iloc[i].values
            # print(values)
            pred = np.dot(self.weights, values)
            y_pred.append(1 if pred >= 0 else -1)

        y_pred = np.array(y_pred)
        y_test = np.array(y_test.iloc[:, -1])

        # y_test = y_test.values


        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        # calculate the confusion matrix

        for actual, predicted in zip(y_test,y_pred):
            if predicted == 1 & actual == 1:
                true_positive += 1
                continue
            if predicted == -1 & actual == 1:
                false_negative += 1
                continue
            if predicted == -1 & actual == -1:
                true_negative += 1
                continue
            if predicted == 1 & actual == -1:
                false_positive += 1
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

    def confusion_matrix(self, y_true, prediction):
        # Get the unique classes excluding NaN
        unique_classes = np.unique(np.concatenate((y_true[~np.isnan(y_true)], prediction[~np.isnan(prediction)])))

        # Initialize the confusion matrix
        cm = np.zeros((len(unique_classes), len(unique_classes)))

        # Fill the confusion matrix
        for i in range(len(y_true)):
           if not np.isnan(y_true[i]) and not np.isnan(prediction[i]):
                true_class = np.where(unique_classes == y_true[i])[0][0]
                pred_class = np.where(unique_classes == prediction[i])[0][0]
                cm[true_class][pred_class] += 1
        # print(self.y_true)
        # print(self.prediction)
        return cm

    def plot_confusion_matrix(self, confusion_matrix, class_names):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Perceptron Confusion Matrix')
        plt.show()


# ft = ["Area", "Perimeter"]
# cls = ["BOMBAY", "CALI"]
# pre = PreProcessing()
# pre.read_data("Dry_Bean_Dataset.csv", ft, cls)
#
# pre.split_data(40)
# pre.null_handel()
# pre.normalize_train_data()
# ad = Adaline()
# # ad.train(pre.x_train, pre.y_train, 1, 0.1, 0.01)
#
# # ad.plotting(pre.x_train, pre.y_train)
#
# X_train = pre.x_train
# y_train = pre.y_train
#
# print(X_train.columns)
#
#
# # class_1 = X_train[y_train == 1]
# # class_minus1 = X_train[y_train == -1]
# #
# # # Create a scatter plot for each class
# # plt.scatter(class_1['feature1'], class_1['feature2'], label='Class 1', c='b', marker='o')
# # plt.scatter(class_minus1['feature1'], class_minus1['feature2'], label='Class -1', c='r', marker='x')
# #
# # # Add labels and legend
# # plt.xlabel('Feature 1')
# # plt.ylabel('Feature 2')
# # plt.legend()
# #
# # # Show the plot
# # plt.show()
#
# # Show the plot
# plt.show()