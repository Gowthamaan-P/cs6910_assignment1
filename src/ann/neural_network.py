import numpy as np

from .neural_layer import NeuralLayer
from .objective_functions import ObjectiveFunction

class NeuralNetwork:
    def __init__(self, config):
        """
        Initialize a neural network based on the provided configuration.

        Args:
        - config (dict): Configuration dictionary containing network parameters.
        """
        def get_value(key, default):
            """
            Helper function to get a value from config dictionary or return default.

            Args:
            - key (str): Key to look up in the config dictionary.
            - default: Default value to return if key is not found.

            Returns:
            - Value from config dictionary if key is found, otherwise returns default.
            """
            return config[key] if key in config else default

        self.layers = []

        self.criterion = get_value('criterion', 'cross_entropy')
        self.weight_initialization = get_value('weight_initialization', 'random')

        self.c = ObjectiveFunction(method=self.criterion)

        self.add_layers(config['input_size'],
                         config['num_layers'],
                         config['output_size'],
                         config['hidden_size'],
                         config['activation'],
                         config['output_activation']
                        )

    def forward(self, x):
        """
        Perform forward pass through the neural network.

        Args:
        - x (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Output of the neural network.
        """
        for layer in self.layers:
            x = layer.activate(x)
        return x

    def backward(self, x, y, y_hat):
        """
        Perform backward pass (backpropagation) through the neural network.

        Args:
        - x (numpy.ndarray): Input data.
        - y (numpy.ndarray): True labels.
        - y_hat (numpy.ndarray): Predicted labels.
        """
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer == self.layers[-1]:
                layer.error = self.c.get_derivative(y, y_hat)
                output_derivative_matrix = []
                for i in range(y_hat.shape[0]):
                    output_derivative_matrix.append(np.matmul(
                        self.c.get_derivative(y[i], y_hat[i]),
                        layer.apply_activation_derivative(y_hat[i])
                    ))
                layer.delta = np.array(output_derivative_matrix)
            else:
                next_layer = self.layers[i + 1]
                layer.error = np.matmul(next_layer.delta, next_layer.weights.T)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.activation)


        for i in range(len(self.layers)):
            layer = self.layers[i]
            activation = np.atleast_2d(x if i == 0 else self.layers[i - 1].activation)
            layer.d_weights = np.matmul(activation.T, layer.delta)/y.shape[0]
            layer.d_bias = np.sum(layer.delta, axis=0)/y.shape[0]

    def add_layers(self, input_size, hidden_layers, output_size, neurons, activation, output_activation):
        """
        Add layers to the neural network.

        Args:
        - input_size (int): Number of input features.
        - hidden_layers (int): Number of hidden layers.
        - output_size (int): Number of output neurons.
        - neurons (int): Number of neurons in hidden layers.
        - activation (str): Activation function for hidden layers.
        - output_activation (str): Activation function for output layer.
        """
        for i in range(0, hidden_layers+1):
            n_input = input_size if i==0 else neurons
            n_neurons = output_size if i==hidden_layers else neurons
            self.layers.append(NeuralLayer(
                index=i+1,
                n_input=n_input,
                n_neurons=n_neurons,
                function= output_activation if i==hidden_layers else activation,
                method=self.weight_initialization
                )
            )