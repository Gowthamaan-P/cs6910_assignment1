from neural_network import NeuralNetwork

class Optimizer:
    def __init__(self, nn:NeuralNetwork, config=None):
        # Initialize parameters
        self.nn, self.lr, self.optimizer = nn, config['learning_rate'], config['optimizer']
        self.beta, self.epsilon, self.beta1, self.beta2= config['beta'], config['epsilon'], config['beta1'], config['beta2']
        self.timestep = 0
        self.decay = config['decay']

    def step(self):
        if(self.optimizer == "sgd"):
            self.sgd()
    
    def sgd(self):
        for layer in self.nn.layers:
            layer.weights -= self.lr * (layer.d_weights + self.decay*layer.weights)
            layer.bias -= self.lr * (layer.d_bias + self.decay*layer.bias)