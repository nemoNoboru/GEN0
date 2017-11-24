import numpy as np
import copy

class DenseLayer():
    def __init__(self, size, activation):
        #self.weights = np.random.rand(output_dim, input_dim)
        self.activation = activation
        self.output_dim = size
        #self.input_dim = input_dim
        #self.size = input_dim * output_dim
        #self.reset()

    def getWeights(self):
        return np.copy(self.weights.flatten()).tolist()

    def feedForward(self, i):
        output = np.matmul(self.weights, i)
        return self.activation(output)

    def setWeights(self, weights):
        self.weights = np.reshape(weights, (self.output_dim, self.input_dim))

    def reset(self):
        self.weights = np.random.rand(self.output_dim, self.input_dim)

    def setInputAndCreate(self, input_dim):
        self.input_dim = input_dim
        self.size = self.input_dim * self.output_dim
        self.reset()
        return self
