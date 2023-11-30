import numpy as np
import pandas as pd

from Task2.preprocessing import PreProcessing
from Task2.evaluate_old import Evaluate


class Layer:
    def __init__(self, num_neurons, bias_enable, activation):
        self.activation = activation
        self.num_neurons = num_neurons
        self.layer_output = None
        self.layer_weights = None
        self.layer_bias = None
        self.bias_enable = bias_enable
        self.change = None

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

    def fw_propagation(self, layer_input):
        self.layer_weights = np.random.rand(layer_input.shape[1], self.num_neurons)
        if self.bias_enable == 1:
            self.layer_bias = np.random.rand(1, self.num_neurons)
        else:
            self.layer_bias = np.zeros([1, self.num_neurons])
        net = np.dot(layer_input, self.layer_weights) + self.layer_bias
        self.layer_output = self._activation_function(net)
        return self.layer_output

    def bw_propagation_hidden(self, layer_input, prev_layer_weights):
        error = np.dot(layer_input, np.array(prev_layer_weights).T)
        delta = error * self._activation_derivative(self.layer_output)
        self.change = delta
        return delta

    def bw_propagation_output(self, target):
        output_error = target.iloc[0:] - self.layer_output
        output_delta = output_error * self._activation_derivative(np.array(self.layer_output))
        self.change = output_delta
        return output_delta


class NeuralNetwork:
    def __init__(self, input_size, num_layers, hidden_neurons_list, output_size, learning_rate, epochs, activation, bias_enable):
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_neurons_list = hidden_neurons_list
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.bias_enable = bias_enable
        self.layers_list = self.init_layers()

    def init_layers(self):
        layers_list = []
        for i in range(self.num_layers):
            layer = Layer(self.hidden_neurons_list[i], self.bias_enable, self.activation)
            layers_list.append(layer)
        layer = Layer(self.output_size, self.bias_enable, self.activation)
        layers_list.append(layer)
        return layers_list

    def train(self, x_train, y_train):
        for e in range(self.epochs):
            for i in range(len(x_train)):
                activated = x_train[i:i+1]
                for j in range(self.num_layers + 1):
                    activated = self.layers_list[j].fw_propagation(activated)

                activated = self.layers_list[-1].bw_propagation_output(y_train[i:i+1])
                for j in range(self.num_layers, 0, -1):
                    activated = self.layers_list[j - 1].bw_propagation_hidden(activated, self.layers_list[j].layer_weights)

                for j in range(self.num_layers + 1):
                    if j == 0:
                        layer_input = np.array(x_train[i:i+1])
                    else:
                        layer_input = self.layers_list[j - 1].layer_output

                    self.layers_list[j].layer_weights += self.learning_rate * np.dot(layer_input.T, self.layers_list[j].change)
                    self.layers_list[j].layer_bias += self.learning_rate * self.layers_list[j].change

    def predict(self, x_test):
        predictions = []
        for i in range(len(x_test)):
            for j in range(self.num_layers + 1):
                self.layers_list[j].fw_propagation(x_test[i:i + 1])
            labels = self.layers_list[-1].layer_output
            max_index = labels.argmax()
            labels = [0, 0, 0]
            labels[max_index] = 1
            predictions.append(labels)
        return predictions


# preprocessing = PreProcessing()
# preprocessing.read_data(r"C:\Users\hb\PycharmProjects\Neural-Project\Task2\Dry_Bean_Dataset.csv",
#                         ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'],
#                         ['CALI', 'BOMBAY', 'SIRA'])
# preprocessing.split_data(40)
# preprocessing.null_handel()
# preprocessing.normalize_train_data()
# preprocessing.normalize_test_data()
#
# obj = NeuralNetwork(5, 2, [2,4], 3, 10, 1000, "sigmoid", 1)
# obj.train(preprocessing.x_train, preprocessing.y_train)
# tr = obj.predict(preprocessing.x_train)
# ev = Evaluate(tr, preprocessing.y_train, obj.output_size)
# ev.calculate_confusion_matrix()
# # print("Train confusion_matrix ", ev.confusion_matrix)
# # print("Train accuracy: ", ev.calculate_accuracy())
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(preprocessing.y_train, tr)
# print(accuracy)
# p = obj.predict(preprocessing.x_test)
# evv = Evaluate(p, preprocessing.y_test,  obj.output_size)
# evv.calculate_confusion_matrix()
# # print("Test confusion_matrix ", evv.confusion_matrix)
# # print("Test accuracy: ", evv.calculate_accuracy())
# accuracy = accuracy_score(preprocessing.y_test, p)
# print(accuracy)


