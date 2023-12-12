import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Perceptron:
    def __init__(self, data, epochs, learning_rate, bias):
        self.data = data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.bias = bias
        self.w = None
        self.prediction = None
        self.y_true = None

    def perceptron_train(self):
        if self.bias:
            self.bias = np.random.uniform(0, 1, 1)
        else:
            self.bias = 0
        self.w = np.random.uniform(0, 1, 2)
        for n in range(self.epochs):
            for i in range(self.data.x_train.shape[0]):
                y = np.dot(self.data.x_train.iloc[i].values, self.w.T) + self.bias
                if y >= 0:
                    y = 1
                else:
                    y = -1
                t = self.data.y_train.iloc[i].values
                if y != t:
                    loss = t - y
                    self.w = self.w + (loss * self.learning_rate * self.data.x_train.iloc[i].values)
                    if self.bias:
                        self.bias = self.bias + (loss * self.learning_rate)

    def perceptron_test(self):
        y = np.dot(self.data.x_test, self.w.T) + self.bias
        self.prediction = np.where(y >= 0, 1, -1)
        self.y_true = np.array(self.data.y_test.iloc[:, -1])

    def confusion_matrix(self):
        # Get the unique classes excluding NaN
        # from sklearn.metrics import confusion_matrix
        # cm_test = confusion_matrix(self.y_true, self.prediction)
        # print("cm test built in")
        # print(cm_test)
        # train_pred = np.dot(self.data.x_train, self.w.T)
        # train_pred = np.where(train_pred >= 0, 1, -1)
        # cm_train = confusion_matrix(np.array(self.data.y_train.iloc[:, -1]),train_pred)
        # print("cm train built in")
        # print(cm_train)
        unique_classes = np.unique(np.concatenate((self.y_true[~np.isnan(self.y_true)], self.prediction[~np.isnan(self.prediction)])))

        # Initialize the confusion matrix
        cm = np.zeros((len(unique_classes), len(unique_classes)))

        # Fill the confusion matrix
        for i in range(len(self.y_true)):
            if not np.isnan(self.y_true[i]) and not np.isnan(self.prediction[i]):
                true_class = np.where(unique_classes == self.y_true[i])[0][0]
                pred_class = np.where(unique_classes == self.prediction[i])[0][0]
                cm[true_class][pred_class] += 1
        # print(self.y_true)
        # print(self.prediction)
        return cm

    def accuracy_score(self):
        # Calculate the number of correct predictions
        correct = sum(self.y_true == self.prediction)

        # Calculate the total number of predictions
        total = len(self.y_true)

        # Calculate the accuracy score
        accuracy = correct / total

        return accuracy

    def plotting(self):
        plt.scatter(self.data.x_train.iloc[:, 0], self.data.x_train.iloc[:, 1], c=self.data.y_train.iloc[:, -1])
        x = np.linspace(self.data.x_train.iloc[:, 0].min(), self.data.x_train.iloc[:, 0].max(), 100)
        y = -(self.w[0] * x + self.bias) / self.w[1]
        plt.plot(x, y, color='red')
        plt.title('Train')
        plt.show()

        plt.scatter(self.data.x_test.iloc[:, 0], self.data.x_test.iloc[:, 1], c=self.data.y_test.iloc[:, -1])
        x = np.linspace(self.data.x_test.iloc[:, 0].min(), self.data.x_test.iloc[:, 0].max(), 100)
        y = -(self.w[0] * x + self.bias) / self.w[1]
        plt.plot(x, y, color='blue')
        plt.title('Test')
        plt.show()



    def plot_confusion_matrix(self, confusion_matrix, class_names):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Perceptron Confusion Matrix')
        plt.show()

    def predict(self, x, b):
        # print(x)
        # print(self.w)
        prediction = np.dot(self.w.T, x) + b
        predicted_class = 1 if prediction >= 0 else -1
        return predicted_class