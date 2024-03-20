import numpy as np

class NeuralLayer:
    """
    This class represents a layer in a neural network.
    """
    def __init__(self, index, n_input, n_neurons, function=None, weights=None, bias=None, method="random"):
        """
        Initialize a layer with specified parameters.

        Args:
        - index (int): Index of the layer.
        - n_input (int): Number of input neurons.
        - n_neurons (int): Number of neurons in the layer.
        - function (str, optional): Activation function type. Default is 'sigmoid'.
        - weights (numpy.ndarray, optional): Weight matrix for connections from input neurons. 
                                              If None, weights will be initialized based on the method.
        - bias (numpy.ndarray, optional): Bias values for neurons in the layer. 
                                           If None, biases will be initialized randomly.
        - method (str, optional): Method for weight initialization. Default is 'random'.

        """
        self.index = index
        self.function = function if function is not None else 'sigmoid'
        self.weights = weights if weights is not None else self.initialize_weights(method, n_input, n_neurons)
        self.bias = bias if bias is not None else np.random.randn(n_neurons)
        self.activation = None

        self.error = None
        self.delta = None

        self.d_weights = np.zeros([n_input, n_neurons])
        self.d_bias = np.zeros(n_neurons)

        self.h_weights = np.zeros([n_input, n_neurons])
        self.h_bias = np.zeros(n_neurons)
        self.m_weights = np.zeros([n_input, n_neurons])
        self.m_bias = np.zeros(n_neurons)

    def initialize_weights(self, method, n_input, n_neurons):
        """
        Initialize weights for connections from input neurons.

        Args:
        - method (str): Method for weight initialization.
        - n_input (int): Number of input neurons.
        - n_neurons (int): Number of neurons in the layer.

        Returns:
        - numpy.ndarray: Initialized weight matrix.
        """
        if method == "Xavier":
            limit = np.sqrt(2 / (n_input + n_neurons))
            return np.random.uniform(-limit, limit, size=(n_input, n_neurons))
        return np.random.randn(n_input, n_neurons)

    def activate(self, x):
        """
        Compute the activation of the layer given input values.

        Args:
        - x (numpy.ndarray): Input values to the layer.

        Returns:
        - numpy.ndarray: Activation values of the layer.
        """
        z = np.dot(x, self.weights) + self.bias
        self.activation = self._apply_activation(z)
        return self.activation

    def _apply_activation(self, r):
        """
        Apply the activation function to the given input.

        Args:
        - z (numpy.ndarray): Input to which activation function is applied.

        Returns:
        - numpy.ndarray: Output of the activation function.
        """
        if self.function == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        elif self.function == 'tanh':
            return np.tanh(r)
        elif self.function == 'ReLu':
            return np.maximum(0, r)
        elif self.function == 'softmax':
            max_r = np.max(r, axis=1)
            max_r = max_r.reshape(max_r.shape[0], 1)
            exp_r = np.exp(r - max_r)
            return exp_r / np.sum(exp_r, axis=1).reshape(exp_r.shape[0], 1)
        return r

    def apply_activation_derivative(self, z):
        """
        Compute the derivative of the activation function applied to the given input.

        Args:
        - z (numpy.ndarray): Input to which activation function is applied.

        Returns:
        - numpy.ndarray: Derivative of the activation function with respect to the input.
        """
        if self.function == 'sigmoid':
            return z * (1 - z)
        elif self.function == 'tanh':
            return (z - z**2)
        elif self.function == 'ReLu':
            return np.where(z > 0, 1, 0)
        elif self.function == 'softmax':
            return np.diag(z) - np.outer(z, z)
        return np.ones(z.shape)

    def __str__(self):
        """
        Return a string representation of the neural layer.

        Returns:
        - str: String representation of the neural layer.
        """
        return f'Neural Layer: {self.index}, {self.weights.shape} , {self.function}'