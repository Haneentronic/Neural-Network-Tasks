import numpy as np
from Task2.preprocessing import PreProcessing
from Task2.evaluate import Evaluate


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, neurons_hidden, activation, learning_rate, epochs):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.neurons_hidden = neurons_hidden
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = self._initialize_weights_and_bias()

    def _initialize_weights_and_bias(self):
        weights = []
        bias = []

        layer_sizes = [self.input_size] + self.neurons_hidden

        for i in range(len(layer_sizes) - 2):
            weights.append(np.random.rand(layer_sizes[i], layer_sizes[i + 1]))
            bias.append(np.random.rand(layer_sizes[i + 1]))
        weights.append(np.random.rand(self.neurons_hidden[self.hidden_layers - 1]))
        bias.append(np.random.rand(self.neurons_hidden[self.hidden_layers]))

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
        output = []
        activated = x
        for layer_number in range(self.hidden_layers + 1):
            net = np.dot(activated, np.array(self.weights[layer_number])) + self.bias[layer_number]
            activated = self._activation_function(net)
            output.extend(activated)
        return output

    def _backward_propagation(self, x, y):
        layers = self._forward_propagation(x)
        print(layers)
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

    def train(self, x_train, y_train):
        for epoch in range(self.epochs):
            for i, x in enumerate(x_train):
                deltas = self._backward_propagation(x_train[i:i+1], y_train[i])

                for j in range(len(self.hidden_layers)):
                    if j == 0:
                        layer_input = x
                    else:
                        layer_input = layers[j - 1]

                    self.weights[j] += self.learning_rate * layer_input.reshape(-1, 1) * deltas[j]
                    self.bias[j] += self.learning_rate * deltas[j]

    def predict(self, x_test):
        predictions = []
        for i in range(len(x_test)):
            output = self._forward_propagation(x_test[i:i+1])
            labels = output[-1*self.neurons_hidden[-1]:]
            predictions.append(labels.index(max(labels)))
        return predictions


preprocessing = PreProcessing()
preprocessing.read_data(r"C:\Users\hb\PycharmProjects\Neural-Project\Task2\Dry_Bean_Dataset.csv",
                        ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'],
                        ['CALI', 'BOMBAY', 'SIRA'])
preprocessing.split_data(40)
preprocessing.null_handel()

obj = NeuralNetwork(5, 1, [2, 3], "sigmoid", 1, 1)
# obj.train(preprocessing.x_train, preprocessing.y_train) # Error
p = obj.predict(preprocessing.x_test)
ev = Evaluate(p, preprocessing.y_test, obj.neurons_hidden[-1])
ev.calculate_confusion_matrix()
ev.calculate_accuracy()
