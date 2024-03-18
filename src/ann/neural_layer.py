import numpy as np

class NeuralLayer: 
    def __init__(self, index, n_input, n_neurons, function=None, weights=None, bias=None, method="random"):
        self.index = index
        self.function = function if function is not None else 'sigmoid'
        self.weights = weights if weights is not None else self.initialize_weights(method, n_input, n_neurons)
        self.bias = bias if bias is not None else np.random.randn(n_neurons)
        self.activation = None
        
        self.error = None
        self.delta = None
        
        self.d_weights = None
        self.d_bias = None

    def initialize_weights(self, method, n_input, n_neurons):
        if method == "xavier":
            limit = np.sqrt(2 / (n_input + n_neurons))
            return np.random.randn(n_input, n_neurons) * limit
        return np.random.randn(n_input, n_neurons)

    def activate(self, x):
        z = np.dot(x, self.weights) + self.bias
        self.activation = self._apply_activation(z)
        return self.activation

    def _apply_activation(self, r):
        if self.function == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        elif self.function == 'tanh':
            return np.tanh(r)
        elif self.function == 'relu':
            return np.maximum(0, r)
        elif self.function == 'softmax':
            max_r = np.max(r, axis=1)
            max_r = max_r.reshape(max_r.shape[0], 1)
            exp_r = np.exp(r - max_r)
            return exp_r / np.sum(exp_r, axis=1).reshape(exp_r.shape[0], 1)
        return r

    def apply_activation_derivative(self, z):
        if self.function == 'sigmoid':
            return z * (1 - z)
        elif self.function == 'tanh':
            return (z - z**2)
        elif self.function == 'relu':
            return np.where(z > 0, 1, 0)
        elif self.function == 'softmax':
            return np.diag(z) - np.outer(z, z)
        return np.ones(z.shape)
    
    def __str__(self):
        return f'Neural Layer: {self.index}, {self.weights.shape} , {self.function}'