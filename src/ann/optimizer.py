import numpy as np
from .neural_network import NeuralNetwork

class Optimizer:
    def __init__(self, nn:NeuralNetwork, config=None):
        """
        Initialize the optimizer.

        Args:
        - nn (NeuralNetwork): Neural network instance to optimize.
        - config (dict): Configuration dictionary containing optimizer parameters.
        """
        self.nn, self.lr, self.optimizer = nn, config['learning_rate'], config['optimizer']
        self.beta, self.epsilon, self.beta1, self.beta2= config['beta'], config['epsilon'], config['beta1'], config['beta2']
        self.timestep = 0
        self.decay = config['weight_decay']

    def step(self):
        """
        Perform optimization step based on the selected optimizer.
        """
        if(self.optimizer == "sgd"):
            self.sgd()
        elif(self.optimizer == "momentum"):
            self.momentum()
        elif(self.optimizer == "nag"):
            self.nag()
        elif(self.optimizer == "rmsprop"):
            self.rmsprop()
        elif(self.optimizer == "adam"):
            self.adam()
        elif (self.optimizer == "nadam"):
            self.nadam()
    
    def sgd(self):
        """
        Perform stochastic gradient descent optimization.
        """
        for layer in self.nn.layers:
            layer.weights -= self.lr*(layer.d_weights + self.decay*layer.weights)
            layer.bias -= self.lr*(layer.d_bias + self.decay*layer.bias)
    
    def momentum(self):
        """
        Perform momentum optimization.
        """
        for layer in self.nn.layers:
            layer.h_weights = self.beta*layer.h_weights + layer.d_weights
            layer.h_bias = self.beta*layer.h_bias + layer.d_bias
            layer.weights -= self.lr*(layer.h_weights + self.decay*layer.weights)
            layer.bias -= self.lr*(layer.h_bias + self.decay*layer.bias)
    
    def nag(self):
        """
        Perform Nesterov accelerated gradient (NAG) optimization.
        """
        for layer in self.nn.layers:
            layer.h_weights = self.beta*layer.h_weights + layer.d_weights
            layer.h_bias = self.beta*layer.h_bias + layer.d_bias
            layer.weights -= self.lr * (self.beta * layer.h_weights + layer.d_weights + self.decay * layer.weights)
            layer.bias -= self.lr * (self.beta * layer.h_bias + layer.d_bias + self.decay * layer.bias)
            
    def rmsprop(self):
        """
        Perform RMSprop optimization.
        """
        for layer in self.nn.layers:
            layer.h_weights = self.beta * layer.h_weights + (1 - self.beta) * layer.d_weights**2
            layer.h_bias = self.beta * layer.h_bias + (1 - self.beta) * layer.d_bias**2
            layer.weights -= (self.lr / (np.sqrt(layer.h_weights) + self.epsilon)) * layer.d_weights + self.decay * layer.weights * self.lr
            layer.bias -= (self.lr / (np.sqrt(layer.h_bias) + self.epsilon)) * layer.d_bias + self.decay * layer.bias * self.lr
        
    def adam(self):
        """
        Perform Adam optimization.
        """
        for layer in self.nn.layers:
            layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.d_weights
            layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.d_bias
            layer.h_weights = self.beta2 * layer.h_weights + (1 - self.beta2) * layer.d_weights**2
            layer.h_bias = self.beta2 * layer.h_bias + (1 - self.beta2) * layer.d_bias**2
            correction_term1 = 1/(1 - self.beta1**(self.timestep + 1))
            correction_term2 = 1/(1 - self.beta2**(self.timestep + 1))
            weights_hat1 = layer.m_weights * correction_term1 
            bias_hat1 = layer.m_bias * correction_term1
            weights_hat2 = layer.h_weights * correction_term2
            bias_hat2 = layer.h_bias * correction_term2
            layer.weights -= self.lr * (weights_hat1 / ((np.sqrt(weights_hat2)) + self.epsilon)) + self.decay * layer.weights * self.lr
            layer.bias -= self.lr * (bias_hat1 / ((np.sqrt(bias_hat2)) + self.epsilon)) + self.decay * layer.bias * self.lr
    
    def nadam(self):
        """
        Perform NAdam (Nesterov Adam) optimization.
        """
        for layer in self.nn.layers:
            layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.d_weights
            layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.d_bias
            layer.h_weights = self.beta2 * layer.h_weights + (1 - self.beta2) * layer.d_weights**2
            layer.h_bias = self.beta2 * layer.h_bias + (1 - self.beta2) * layer.d_bias**2
            correction_term1 = 1/(1 - self.beta1**(self.timestep + 1))
            correction_term2 = 1/(1 - self.beta2**(self.timestep + 1))
            weights_hat1 = layer.m_weights * correction_term1 
            bias_hat1 = layer.m_bias * correction_term1
            weights_hat2 = layer.h_weights * correction_term2
            bias_hat2 = layer.h_bias * correction_term2
            combined_weight_update = self.beta1 * weights_hat1 + ((1 - self.beta1) / (1 - self.beta1 ** (self.timestep + 1))) * layer.d_weights
            combined_bias_update = self.beta1 * bias_hat1 + ((1 - self.beta1) / (1 - self.beta1 ** (self.timestep + 1))) * layer.d_bias
            layer.weights -= self.lr * (combined_weight_update / ((np.sqrt(weights_hat2)) + self.epsilon)) + self.decay * layer.weights * self.lr
            layer.bias -= self.lr * (combined_bias_update / ((np.sqrt(bias_hat2)) + self.epsilon)) + self.decay * layer.bias * self.lr