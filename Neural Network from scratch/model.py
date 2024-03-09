import numpy as np

class Model():
    def __init__(self, layers = None, learning_rate = 1e-5):
        self.layers = layers
        self.learning_rate= learning_rate


    def add(self, layer):
        if not self.layers:
            self.layers = []
            
        self.layers.append(layer)


    def fit(self, X, y, epoches = 10):
        loss = []

        # Stochastic Gradient Descent
        for _ in range(epoches):
            epoch_loss = 0
            for i in range(X.shape[0]):
                x_i = X[i].reshape(1, -1)
                y_i = y[i]
                
                # forward propagation
                output = x_i
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                    
                epoch_loss += self.mse(y_i, output)
                                       
                # backward propagation
                err = self.mse_derivative(y_i, output)
                for layer in self.layers[::-1]:
                    err = layer.backward_propagation(err, self.learning_rate)
            loss.append(epoch_loss/X.shape[0])
        
        return loss
            
    
    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output


    @staticmethod
    def mse(y_true, y_pred):
        y_true = np.asarray(y_true).reshape((-1, 1))
        y_pred = np.asarray(y_pred).reshape((-1, 1))
        return (2 / y_true.shape[1]) * np.sum((y_true - y_pred) ** 2)


    @staticmethod
    def mse_derivative(y_true, y_pred):
        y_true = np.asarray(y_true).reshape((-1, 1))
        y_pred = np.asarray(y_pred).reshape((-1, 1))
        return (1 / y_true.shape[1]) * (y_pred - y_true) 