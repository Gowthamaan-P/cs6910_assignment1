import numpy as np

class ObjectiveFunction:
    def __init__(self, method):
        """
        Initialize the objective function with the specified method.

        Args:
        - method (str): Method specifying the type of loss function ('cel' for cross-entropy, 'mse' for mean squared error).
        """
        self.method = method

    def get_loss(self, y, y_hat):
        """
        Calculate the loss given the true values and predicted values.

        Args:
        - y (numpy.ndarray): True values.
        - y_hat (numpy.ndarray): Predicted values.

        Returns:
        - float: Loss value.
        """
        if self.method == "cross_entropy":
            return self.cross_entropy_loss(y, y_hat)
        elif self.method == "mean_squared_error":
            return self.mean_square_error(y, y_hat)

    def get_derivative(self, y, y_hat):
        """
        Calculate the derivative of the loss function with respect to predicted values.

        Args:
        - y (numpy.ndarray): True values.
        - y_hat (numpy.ndarray): Predicted values.

        Returns:
        - numpy.ndarray: Derivative of the loss function.
        """
        if self.method == "cross_entropy":
            return self.cross_entropy_loss_derivative(y, y_hat)
        elif self.method == "mean_squared_error":
            return self.mean_square_error_derivative(y, y_hat)

    def mean_square_error(self, y, y_hat):
        """
        Calculate the mean squared error loss.

        Args:
        - y (numpy.ndarray): True values.
        - y_hat (numpy.ndarray): Predicted values.

        Returns:
        - float: Mean squared error loss.
        """
        return np.sum((y - y_hat) ** 2) / 2

    def mean_square_error_derivative(self, y, y_hat):
        """
        Calculate the derivative of the mean squared error loss.

        Args:
        - y (numpy.ndarray): True values.
        - y_hat (numpy.ndarray): Predicted values.

        Returns:
        - numpy.ndarray: Derivative of the mean squared error loss.
        """
        return y_hat - y

    def cross_entropy_loss(self, y, y_hat):
        """
        Calculate the cross-entropy loss.

        Args:
        - y (numpy.ndarray): True values.
        - y_hat (numpy.ndarray): Predicted values.

        Returns:
        - float: Cross-entropy loss.
        """
        return -np.sum(y * np.log(y_hat))

    def cross_entropy_loss_derivative(self, y, y_hat):
        """
        Calculate the derivative of the cross-entropy loss.

        Args:
        - y (numpy.ndarray): True values.
        - y_hat (numpy.ndarray): Predicted values.

        Returns:
        - numpy.ndarray: Derivative of the cross-entropy loss.
        """
        return -y/y_hat