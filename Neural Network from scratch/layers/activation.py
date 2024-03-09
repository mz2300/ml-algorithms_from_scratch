import numpy as np
from dataclasses import dataclass
from typing import Callable

from .layer import Layer


#-------sigmoid---------
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))
    
#-------relu---------
def relu(z):
    return np.where(z < 0, 0, z)

def relu_derivative(z):
    return np.where(z < 0, 0, 1)
    
    
@dataclass
class ActivationFunc:
    func: Callable[[np.ndarray], np.float16]
    derivative: Callable[[np.ndarray], np.float16]

activtion_map = {'sigmoid' : ActivationFunc(func = sigmoid, derivative = sigmoid_derivative),
                 'relu' : ActivationFunc(func = relu, derivative = relu_derivative)}
                 
                 
                 
class ActivationLayer(Layer):
    def __init__(self, func_name):
        self.activation = activtion_map[func_name.lower()]

    def forward_propagation(self, input_data):
        self.input_data = input_data
        return self.activation.func(input_data)

    def backward_propagation(self, output_error, learning_rate):
        return self.activation.derivative(self.input_data) * output_error