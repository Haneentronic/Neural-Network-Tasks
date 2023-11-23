import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, neurons_hidden, activation, learning_rate, epochs):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.neurons_hidden = neurons_hidden
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = self._initialize_weights_and_bias()

    def initialize_weights_and_bias(input_size, hidden_layers, neurons_hidden):
        weights = []
        bias = []

        # Input to first hidden layer
        weights.append(np.random.rand(input_size, neurons_hidden[0]))
        bias.append(np.random.rand(neurons_hidden[0]))

        # Hidden layers
        for i in range(len(neurons_hidden) - 1):
            weights.append(np.random.rand(neurons_hidden[i], neurons_hidden[i + 1]))
            bias.append(np.random.rand(neurons_hidden[i + 1]))

        return weights, bias

    # Activation functions and their derivatives
    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)

    def _activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2

    def _forward_propagation(self, x):
       pass

    def _backward_propagation(self, x, y):
        layers = self._forward_propagation(x)
        errors = [None] * len(self.hidden_layers)
        deltas = [None] * len(self.hidden_layers)

        output_error = layers[-1] - y
        output_delta = output_error * self._activation_derivative(layers[-1])
        errors[-1] = output_error
        deltas[-1] = output_delta

        for i in range(len(self.hidden_layers) - 1, 0, -1):
            errors[i - 1] = deltas[i].dot(self.weights[i].T)
            deltas[i - 1] = errors[i - 1] * self._activation_derivative(layers[i])

        return deltas

    def train(self, X_train, y_train):
        for epoch in range(self.epochs):
            for i, x in enumerate(X_train):
                deltas = self._backward_propagation(x, y_train[i])

                for j in range(len(self.hidden_layers)):
                    if j == 0:
                        layer_input = x
                    else:
                        layer_input = layers[j - 1]

                    self.weights[j] += self.learning_rate * layer_input.reshape(-1, 1) * deltas[j]
                    self.bias[j] += self.learning_rate * deltas[j]

    def predict(self, X_test):
        pass

