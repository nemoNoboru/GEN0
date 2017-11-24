# feedForwards input to all layers Sequentialy
import copy
import numpy as np

class Sequential():
    def __init__(self, mutationFunction):
        self.layers = []
        self.mutationFunction = mutationFunction

    def first(self, layer, size):
        self.layers.append(layer.setInputAndCreate(size))

    def add(self, layer):
        output_dim = self.layers[-1].output_dim
        self.layers.append(layer.setInputAndCreate(output_dim))

    def run(self, i):
        i = np.array([i])
        out = i.transpose()
        for layer in self.layers:
            out = layer.feedForward(out)
        return out

    def getGen(self):
        gen = []
        for layer in self.layers:
            gen += layer.getWeights()
        return gen

    def mutateWith(self, lover):
        genSelf = self.getGen()
        genLover = lover.getGen()
        self.setGen(self.mutationFunction(genSelf, genLover))

    def setGen(self, gen):
        for layer in self.layers:
            layer.setWeights(gen[:layer.size])
            del gen[:layer.size]
        if len(gen) > 0:
            print("error, some genes were not used")

    def new(self):
        new = Sequential(self.mutationFunction)
        new.layers = self.layers
        for layer in new.layers:
            layer.reset()
        return new
