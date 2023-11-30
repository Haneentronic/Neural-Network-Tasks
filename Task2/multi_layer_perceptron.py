import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# PreProcessing class definition (use your provided code)
from preprocessing import PreProcessing
from evaluate import Evaluate
# Evaluate class definition (use your provided code)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs , activation):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation=activation

        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))

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

    def forward_propagation(self, x):
        # Input to hidden layer
        self.hidden_output = self._activation_function(np.dot(x, self.weights_input_hidden) + self.bias_input_hidden)
        # Hidden to output layer
        self.predicted_output = self._activation_function(
            np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output)

    def backward_propagation(self, x, y):
        # Output to hidden layer
        output_error = y - self.predicted_output
        d_output = output_error * self._activation_derivative(self.predicted_output)

        # Hidden to input layer
        hidden_error = d_output.dot(self.weights_hidden_output.T)
        d_hidden = hidden_error * self._activation_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_hidden_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += x.T.dot(d_hidden) * self.learning_rate
        self.bias_input_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, x, y):
        for epoch in range(self.epochs):
            # Forward propagation
            self.forward_propagation(x)

            # Backward propagation
            self.backward_propagation(x, y)

    def predict(self, x):
        self.forward_propagation(x)
        return self.predicted_output

# Using PreProcessing and NeuralNetwork classes to load data, train the model, and print confusion matrix and accuracy
data_processor = PreProcessing()

# Read data and specify features and classes
data_processor.read_data(r"Dry_Bean_Dataset.csv",
                        ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'],
                        ['CALI', 'BOMBAY', 'SIRA'])
# Split the data into training and testing sets
data_processor.split_data(40)  # Adjust split rate as needed

# Handle missing values in the training set
data_processor.null_handel()
# Normalize the training and testing data
data_processor.normalize_train_data()
data_processor.normalize_test_data()

# Instantiate the neural network
# input_size = len(data_processor.x_train.columns)
# ... (same as previous code)

# Instantiate the neural network
hidden_size = 128  # Adjusted hidden layer size
neural_net = NeuralNetwork(5, hidden_size, 3, 0.1, 500, 'tanh')  # Adjusted learning rate and epochs

# Train the neural network
neural_net.train(data_processor.x_train.values, data_processor.y_train.values)

# Make predictions on the test set
predictions = neural_net.predict(data_processor.x_test.values)

# Instantiate Evaluate class
evaluator = Evaluate(predictions.flatten(), data_processor.y_test, 3)

# Calculate confusion matrix
confusion_matrix = evaluator.calculate_confusion_matrix(predictions)
print("Confusion Matrix:")
print(confusion_matrix)

# Calculate accuracy
accuracy = evaluator.calculate_accuracy()
print(f"Accuracy: {accuracy * 100:.2f}%")

