# defines a mutable class


class Mutable():
    def __init__(self):
        self.layers = []

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
            print("Layer added")
        if len(gen) > 0:
            print("error, some genes were not used")
