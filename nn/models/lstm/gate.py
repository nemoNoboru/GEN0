import numpy as np


class Gate():
    def __init__(self, activation, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = 0
        self.reset()
        self.activation = activation
        self.size = self.input_dim * self.output_dim

    def getWeights(self):
        return np.copy(self.weights.flatten()).tolist()

    def feedForward(self, i):
        output = np.matmul(self.weights, i)
        return self.activation(output)

    def setWeights(self, weights):
        a = self.weights
        self.weights = np.reshape(weights, (self.output_dim, self.input_dim))
        b = self.weights
        if np.array_equal(a,b):
            print("SAME GENES!!")

    def reset(self):
        gen = np.random.normal(size=self.output_dim * self.input_dim)
        self.setWeights(gen)
