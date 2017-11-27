# long short term memory layer
# source : http://colah.github.io/posts/2015-08-Understanding-LSTMs/
import numpy as np
from gate import Gate
from nn.models.mutable import Mutable

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


v_sigmoid = np.vectorize(sigmoid)


class LSTM(Mutable):
    def __init__(self, input_dim, output_dim, mutation):
        self.mutation = mutation
        self.input_dim = input_dim + output_dim
        self.output_dim = output_dim
        self.ht = np.ones(output_dim)
        self.ct = np.ones(output_dim)

        self.forget = Gate(v_sigmoid, self.input_dim, output_dim)
        self.input = Gate(v_sigmoid, self.input_dim, output_dim)
        self.cand = Gate(np.tanh, self.input_dim, output_dim)
        self.output = Gate(v_sigmoid, self.input_dim, output_dim)
        self.layers = [self.forget, self.input, self.cand, self.output]

    def run(self, x):
        i = np.concatenate((self.ht, x))

        ft = self.forget.feedForward(i)
        it = self.input.feedForward(i)
        nct = self.cand.feedForward(i)
        ot = self.output.feedForward(i)

        self.ct = (ft * self.ct) + (it * nct)
        self.ht = ot * np.tanh(self.ct)
        return self.ht

    def mutateWith(self, lover):
        self.ht = np.ones(self.output_dim)
        self.ct = np.ones(self.output_dim)
        genSelf = self.getGen()
        genLover = lover.getGen()
        self.setGen(self.mutation(genSelf, genLover))

    def new(self):
        new = LSTM(self.input_dim - self.output_dim, self.output_dim, self.mutation)
        new.layers = self.layers
        for layer in new.layers:
            layer.reset()
        return new
