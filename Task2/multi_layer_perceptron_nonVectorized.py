import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, num_layers, hidden_size, output_size, learning_rate, epochs, activation, bias_enable):
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.bias_enable = bias_enable

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

    def init_weights(self):
        layer_sizes = sum([self.input_size] + self.hidden_size)


    def fw_propagation(self):
        pass

    def bw_propagation(self):
        pass

    def update_weights(self):
        pass

    def predict(self):
        pass
