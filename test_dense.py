from nn.dense import DenseLayer
from nn.models.sequential import Sequential
from nn.activation_functions import tanh
from mutation.mutators import cross_with_mutation
from pool import GeneticPool

def makeModel():
    sq = Sequential(mutationFunction=cross_with_mutation)
    sq.first(DenseLayer(size=90, activation=tanh), input_size=8)
    sq.add(DenseLayer(size=90, activation=tanh))
    sq.add(DenseLayer(size=2, activation=tanh))
    return sq

def test_pool():
    def mock(agent):
        return 1
    s = makeModel()

    pool = GeneticPool(model=s, env=mock, poolSize=10)
    pool.improve()
