import numpy as np
import matplotlib.pyplot as plt

def perceptron_test(self):
    self.prediction = np.array(np.sign(np.dot(self.data.x_test, self.w.T)))


def ploting(self):
    plt.scatter(self.data.x_train.iloc[:, 0], self.data.x_train.iloc[:, 1], c=self.data.y_train.iloc[:, -1])
    x = np.linspace(self.data.x_train.iloc[:, 0].min(), self.data.x_train.iloc[:, 0].max(), 100)
    y = -(self.w[0] * x + self.bais) / self.w[1]
    plt.plot(x, y, color='red')
    plt.show()


def confusion_matrix(self):
    # Get the unique classes
    classes = np.unique(np.concatenate((np.array(self.data.y_test.iloc[:,0]), self.prediction)))
    # Initialize the confusion matrix
    cm = np.zeros((len(classes), len(classes)))

    # Fill the confusion matrix
    for i in range(len(np.array(self.data.y_test.iloc[:,0]))):
        true_class = np.where(classes == np.array(self.data.y_test.iloc[:,0])[i])[0][0]
        pred_class = np.where(classes == self.prediction[i])[0][0]
        cm[true_class][pred_class] += 1


def accuracy_score(self):
    # Calculate the number of correct predictions
    correct = sum(np.array(self.data.y_test.iloc[:,0]) == self.prediction)

    # Calculate the total number of predictions
    total = len(np.array(self.data.y_test.iloc[:,0]))

     # Calculate the accuracy score
    accuracy = correct / total

    return accuracy