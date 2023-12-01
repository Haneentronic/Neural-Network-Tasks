import numpy as np
from Task2.preprocessing import PreProcessing
from Task2.evaluate_old import Evaluate


class MultiLayerPerceptron:
    def __init__(self, num_inputs, hidden_layers, num_outputs, activation, bias_enable):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.activation = activation

        # layers
        self.layers = [self.num_inputs] + self.hidden_layers + [self.num_outputs]
        self.layers_number = len(self.layers)

        # initiate weights. create matrix weight for each pair of layers
        self.weights = []
        for i in range(self.layers_number - 1):
            w = np.random.rand(self.layers[i], self.layers[i + 1])
            # print(w)
            # print(w.shape)
            self.weights.append(w)

        # self.weights = []
        # w = [
        #     [0.21, -0.4],
        #     [0.15, 0.1]
        # ]
        # w = np.array(w)
        # self.weights.append(w)
        # w = [
        #     [-0.2],
        #     [0.3]
        # ]
        # w = np.array(w)
        # self.weights.append(w)

        # initiate bias for each layer [from first hidden layer to output layer]
        biases = []
        if bias_enable == 1:
            for i in range(1, self.layers_number):
                b = np.random.rand(self.layers[i])
                biases.append(b)
        else:
            for i in range(1, self.layers_number):
                b = np.zeros(self.layers[i])
                biases.append(b)
        self.biases = biases

        # self.biases = []
        # b = [-0.3, 0.25]
        # b = np.array(b)
        # self.biases.append(b)
        # b = [-0.4]
        # b = np.array(b)
        # self.biases.append(b)

        #
        # for w in self.weights:
        #     print(w)
        #     print(w.shape)
        #
        # for b in self.bias:
        #     print(b)
        #     print(b.shape)

        # for b in self.biases:
        #     print(b)

        # activations for each layer
        activations = []
        for i in range(self.layers_number):
            a = np.zeros(self.layers[i])
            activations.append(a)
        self.activations = activations

        # errors
        errors = []
        for i in range(self.layers_number):
            e = np.zeros(self.layers[i])
            errors.append(e)
        self.errors = errors

    def activation_function(self, net):
        ans = 0
        if self.activation == "sigmoid":
            ans = 1 / (1 + np.exp(-net))
        elif self.activation == "tanh":
            ans = np.tanh(net)
        return ans

    def forward_propagate(self, inputs):
        activations_tmp = inputs
        self.activations[0] = inputs
        for i in range(len(self.weights)):
            nets = np.dot(activations_tmp, self.weights[i]) + self.biases[i]
            activations_tmp = self.activation_function(nets)
            self.activations[i + 1] = activations_tmp

        return activations_tmp
        # print("activations")
        # for a in self.activations:
        #     print(a)

    def back_propagate(self, act_output):
        # print("errors: ")
        i = self.layers_number - 1
        while i > 0:
            if i == self.layers_number - 1:
                self.errors[i] = (act_output - self.activations[i]) * (self.activations[i] * (1 - self.activations[i]))
            else:
                self.errors[i] = np.dot(self.weights[i], self.errors[i + 1]) * self.activations[i] * (1 - self.activations[i])
            # print(self.errors[i])
            i -= 1

    def update_weights(self, eta):
        # update weights
        for i in range(len(self.weights)):
            inputs = self.activations[i]
            errors = self.errors[i + 1]
            # print("inputs", inputs)
            # print("errors", errors)
            # print("original w\n", self.weights[i])
            inputs_matrix = inputs.reshape((-1, 1))
            errors_matrix = errors.reshape((-1, 1))
            update_matrix = np.outer(inputs_matrix, errors_matrix.T)
            self.weights[i] += update_matrix * eta
            # print("updated w\n", self.weights[i])

            # update bias
            bias = self.biases[i]
            # print("old bias", bias)
            self.biases[i] += errors * eta
            # print("new bias", bias)
            # print("---------")

    def train(self, inputs, targets, epochs, eta, bias_enable):
        for epoch in range(epochs):
            for (input, target) in zip(inputs, targets):
                # forward propagation
                self.forward_propagate(input)

                # back propagation
                self.back_propagate(target)

                # update weights
                self.update_weights(eta)

    def converter(self, x):
        max_indices = np.argmax(x, axis=1)
        result_array = np.zeros_like(x)
        result_array[np.arange(result_array.shape[0]), max_indices] = 1
        result_array = result_array.astype(int)
        return result_array

    def predict(self, x_test):
        prediction = self.forward_propagate(x_test)
        prediction = self.converter(prediction)
        return prediction


def extract_input_and_output(x, y):
    num_samples, num_features = x.shape
    INPUTS = []
    OUTPUTS = []
    for i in range(num_samples):
        values = x.iloc[i].values
        targets = y.iloc[i].values
        values = np.array(values)
        targets = np.array(targets)
        INPUTS.append(values)
        OUTPUTS.append(targets)

    INPUTS = np.array(INPUTS)
    OUTPUTS = np.array(OUTPUTS)

    return INPUTS, OUTPUTS


# activation1 = "sigmoid"
# activation2 = "tanh"
# mlp = MultiLayerPerceptron(5, [3, 5], 3, activation1, 0)
#
# preprocessing = PreProcessing()
# preprocessing.read_data("Dry_Bean_Dataset.csv",
#                         ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'],
#                         ['CALI', 'BOMBAY', 'SIRA'])
# preprocessing.handel_all_outliers()
# preprocessing.split_data(40)
# preprocessing.null_handel()
# preprocessing.normalize_train_data()
# preprocessing.normalize_test_data()

# num_samples, num_features = preprocessing.x_train.shape
# INPUTS = []
# OUTPUTS = []
# for i in range(num_samples):
#     values = preprocessing.x_train.iloc[i].values
#     targets = preprocessing.y_train.iloc[i].values
#     values = np.array(values)
#     targets = np.array(targets)
#     INPUTS.append(values)
#     OUTPUTS.append(targets)
#
# INPUTS = np.array(INPUTS)
# OUTPUTS = np.array(OUTPUTS)

# train_input, train_expected_output = extract_input_and_output(preprocessing.x_train, preprocessing.y_train)
# mlp.train(train_input, train_expected_output, 30, 0.5, 1)
# train_prediction = mlp.predict(preprocessing.x_train)
# train_evaluator = Evaluate(train_prediction, train_expected_output, mlp.num_outputs)
# train_evaluator.calculate_confusion_matrix()
# print("Train Confusion Matrix: ")
# print(train_evaluator.confusion_matrix)
# print("Train Accuracy: ", train_evaluator.calculate_accuracy())
#
# test_input, test_expected_output = extract_input_and_output(preprocessing.x_test, preprocessing.y_test)
# test_prediction = mlp.predict(preprocessing.x_test)
# test_evaluator = Evaluate(test_prediction, test_expected_output, mlp.num_outputs)
# test_evaluator.calculate_confusion_matrix()
# print("Test Confusion Matrix: ")
# print(test_evaluator.confusion_matrix)
# print("Test Accuracy: ", test_evaluator.calculate_accuracy())

# import pandas as pd
# data = {'Area': [114004],
#         'Perimeter': [1279.356],
#         'MajorAxisLength': [451.3612558],
#         'MinorAxisLength': [323.7479961],
#         'roundnes': [0.875280258]}
# sample = pd.DataFrame(data)
# sample = preprocessing.normalize_sample(sample)
# sample_prediction = mlp.predict(sample)
# print("Sample Prediction: ", sample_prediction)
# convert weights on the last layer to 1 (mx), 0(others)
# def converter(x):
#     max_indices = np.argmax(x, axis=1)
#     result_array = np.zeros_like(x)
#     result_array[np.arange(result_array.shape[0]), max_indices] = 1
#     result_array = result_array.astype(int)
#     return result_array


# train_prediction = mlp.forward_propagate(preprocessing.x_train)
# train_prediction = mlp.converter(train_prediction)
# correct_classified = 0
# for act, pred in zip(OUTPUTS, train_prediction):
#     if np.array_equal(act, pred):
#         correct_classified += 1
#
# training_accuracy = (correct_classified / num_samples)
# print("training accuracy:", training_accuracy)

# num_samples, num_features = preprocessing.x_test.shape
# INPUTS = []
# OUTPUTS = []
# for i in range(num_samples):
#     values = preprocessing.x_test.iloc[i].values
#     targets = preprocessing.y_test.iloc[i].values
#     values = np.array(values)
#     targets = np.array(targets)
#     INPUTS.append(values)
#     OUTPUTS.append(targets)
# test_prediction = mlp.forward_propagate(preprocessing.x_test)
# test_prediction = mlp.converter(test_prediction)
# correct_classified = 0
# for act, pred in zip(OUTPUTS, test_prediction):
#     if np.array_equal(act, pred):
#         correct_classified += 1
#
# testing_accuracy = (correct_classified / num_samples)
# print("testing accuracy:", testing_accuracy)


# # XOR inputs and outputs
# inputs = np.array([
#     [0, 0], [0, 1], [1, 0], [1, 1]
# ])
# outputs = np.array([[0], [1], [1], [0]])
