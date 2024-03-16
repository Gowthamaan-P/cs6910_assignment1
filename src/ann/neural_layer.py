import numpy as np

# Represents a layer (hidden or output) in the neural network.
class NeuralLayer:
    
    # Init the layer
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        self.activation = activation if activation is not None else 'relu'
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons)
        self.velocity = np.zeros_like(self.weights)
        self.bias = bias if bias is not None else np.random.randn(n_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None
    
    # Activate the neural layer
    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    # Activation function
    def _apply_activation(self, r):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'relu':
            return r if r>0 else 0
        elif self.activation == 'softmax':
            return (np.exp(r).T / np.sum(np.exp(r),axis=1)).T
        return r

    # Derivative of the Activation function (will be used in backpropagation)
    def apply_activation_derivative(self, r):
        if self.activation == 'sigmoid':
            return r * (1 - r)
        elif self.activation == 'tanh':
            return (1 - r**2)
        elif self.activation == 'relu':
            return 1 if r>0 else 0
        elif self.activation == 'softmax':
            return np.diag(r) - np.outer(r, r)
        return r