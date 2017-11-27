# feedForwards input to all layers Sequentialy
import numpy as np
from mutable import Mutable


class Sequential(Mutable):
    def __init__(self, mutationFunction):
        self.layers = []
        self.mutationFunction = mutationFunction

    def first(self, layer, input_size):
        self.layers.append(layer.setInputAndCreate(input_size))

    def add(self, layer):
        output_dim = self.layers[-1].output_dim
        self.layers.append(layer.setInputAndCreate(output_dim))

    def run(self, i):
        i = np.array([i])
        out = i.transpose()
        for layer in self.layers:
            out = layer.feedForward(out)
        return out

    def new(self):
        new = Sequential(self.mutationFunction)
        new.layers = self.layers
        for layer in new.layers:
            layer.reset()
        return new
