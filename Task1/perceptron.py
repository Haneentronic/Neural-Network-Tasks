import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from Task1.preprocessing import PreProcessing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
class Perceptron:
    def __init__(self, data, ephocs, learningrate, bais):
        self.data = data
        self.ephocs = ephocs
        self.learningrate = learningrate
        self.bais = bais
        self.w = None
        self.prediction = None
        self.y_true=None

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
        self.prediction = np.array(np.sign(np.dot(self.data.x_test, self.w.T)))
        self.y_true = np.array(self.data.y_test.iloc[:, -1])



    def confusion_matrix(self):
        # Get the unique classes excluding NaN
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

    def ploting(self):
        plt.scatter(self.data.x_test.iloc[:, 0], self.data.x_test.iloc[:, 1], c=self.data.y_test.iloc[:, -1])
        x = np.linspace(self.data.x_test.iloc[:, 0].min(), self.data.x_test.iloc[:, 0].max(), 100)
        y = -(self.w[0] * x + self.bais) / self.w[1]
        plt.plot(x, y, color='red')
        plt.show()

    def plot_confusion_matrix(self,confusion_matrix, class_names):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()


# preprocessing = PreProcessing()
# preprocessing.read_data("Dry_Bean_Dataset.csv", ["Perimeter", "MinorAxisLength"], ["BOMBAY", "CALI"])
# preprocessing.split_data(40)
# preprocessing.null_handel()
# preprocessing.normalize_train_data()
# preprocessing.normalize_test_data()
# # Calculate the correlation matrix
# # Load your dataset into a DataFrame
# data = pd.read_csv('Dry_Bean_Dataset.csv')
# le = LabelEncoder()
# data.iloc[:,-1] = le.fit_transform(data.iloc[:,-1])
# correlation_matrix = data.corr()
#
# # Create a heatmap of the correlation matrix
# # plt.figure(figsize=(12, 8))
# # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# # plt.title("Feature Correlation Matrix")
# # plt.show()
#
# o = Perceptron(preprocessing, 5000, 0.1, 1)
# o.perceptron_train()
# o.perceptron_testt()
# #
# # print(type(np.array(o.data.y_test.loc[:,0])))
# # # print(type(o.prediction))
# # print(o.confusion_matrixx())
# # print(o.accuracy_scoree())
# # o.plotingg()
# # classes=['BOMBAY', 'CALI']
# o.plot_confusion_matrix(o.confusion_matrixx(),classes)