import numpy as np
from preprocessing import PreProcessing
from evaluate import Evaluate

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
class NeuralNetwork:
    def __init__(self, layers, learning_rate, epochs, add_bias, activation_function):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.add_bias = add_bias
        self.weights = []
        self.activation_function = activation_function
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(1, len(self.layers)):
            if self.add_bias:
                w = np.random.randn(self.layers[i - 1] + 1, self.layers[i])
            else:
                w = np.random.randn(self.layers[i - 1], self.layers[i])
            self.weights.append(w)
    def sigmoid(x, derivative=False):
        if derivative:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))

    def tanh(x, derivative=False):
        if derivative:
            return 1 - np.tanh(x) ** 2
        else:
            return np.tanh(x)
    def add_bias_unit(self, X):
        if self.add_bias:
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, :-1] = X
            return X_new
        else:
            return X

    def forward(self, X):
        z = [X]
        for w in self.weights:
            if self.add_bias:
                z[-1] = self.add_bias_unit(z[-1])
            z.append(self.activation_function(np.dot(z[-1], w)))
        return z

    def backward(self, z, y):
        error = y - z[-1]
        deltas = [error * self.activation_function(z[-1], derivative=True)]

        for i in reversed(range(len(deltas), len(self.weights))):
            deltas.insert(0, np.dot(deltas[0], self.weights[i].T) * self.activation_function(z[i], derivative=True))

        for i in range(len(self.weights)):
            layer = np.atleast_2d(z[i])
            delta = np.atleast_2d(deltas[i])
            self.weights[i] += layer.T.dot(delta) * self.learning_rate

    def fit(self, X, y):
        X = self.add_bias_unit(X)
        for _ in range(self.epochs):
            z = self.forward(X)
            self.backward(z, y)

    def predict(self, X):
        X = self.add_bias_unit(X)
        return np.argmax(self.forward(X)[-1], axis=1)

preprocessing = PreProcessing()
# Assuming data is in a CSV file
data = pd.read_csv('Dry_Bean_Dataset.csv')

# Selecting the first 5 features and the target variable
X = data.iloc[:, :5].values
y = data.iloc[:, -1].values
# preprocessing.read_data(r"Dry_Bean_Dataset.csv",
#                         ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'],
#                         ['CALI', 'BOMBAY', 'SIRA'])
# preprocessing.split_data(40)
# preprocessing.null_handel()
# # User Input
# Encoding the classes
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

layers = [5, 4, 3]  # number of neurons in each layer
learning_rate = 0.01  # learning rate
epochs = 500  # number of epochs
add_bias = True  # add bias or not
activation_function = NeuralNetwork.sigmoid  # activation function
# Scaling the features

obj = NeuralNetwork(layers, learning_rate, epochs, add_bias, activation_function)
obj.fit(X_train, y_train)
p = obj.predict(X_test)
# ev = Evaluate(p, preprocessing.y_test, obj.neurons_hidden[-1])
# ev.calculate_confusion_matrix(p)
# print(ev.calculate_accuracy())
