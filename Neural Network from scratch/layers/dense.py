import numpy as np
from .layer import Layer

class Dense(Layer):
    def __init__(self, n_hidden_units):
        self.n_hidden_units = n_hidden_units
        self.weights = None
        self.bias = None
    
    def forward_propagation(self, X):
        self.X = X
        if self.weights is None:
            self.weights = np.random.rand(self.X.shape[1], self.n_hidden_units)

        if self.bias is None:
            self.bias = np.random.rand(1, self.n_hidden_units)

        return np.dot(self.X, self.weights) + self.bias

    def backward_propagation(self, output_error, learning_rate):
        
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.X.T, output_error)
        bias_error = np.sum(output_error, axis = 0)

        # update parameters
        self.weights -= learning_rate * weights_error / self.X.shape[0]
        self.bias -= learning_rate * bias_error / self.X.shape[0]
        return input_error
