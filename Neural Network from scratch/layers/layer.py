from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward_propagation(self, input_data):
        pass

    @abstractmethod
    def backward_propagation(self, output_error, learning_rate):
        pass
