import numpy as np

class ObjectiveFunction:
    def __init__(self, method):
        self.method = method
    
    def get_loss(self, y, y_hat):
        if self.method == "cel":
            return self.cross_entropy_loss(y, y_hat)
        elif self.method == "mse":
            return self.mean_square_error(y, y_hat)
    
    def get_derivative(self, y, y_hat):
        if self.method == "cel":
            return self.cross_entropy_loss_derivative(y, y_hat)
        elif self.method == "mse":
            return self.mean_square_error_derivative(y, y_hat)
    
    def mean_square_error(self, y, y_hat):
        return np.sum((y - y_hat) ** 2) / 2
    
    def mean_square_error_derivative(self, y, y_hat):
        return y_hat - y
    
    def cross_entropy_loss(self, y, y_hat):
        return -np.sum(y * np.log(y_hat))
    
    def cross_entropy_loss_derivative(self, y, y_hat):
        return -y/y_hat