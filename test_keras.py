# create a keras model
from keras.models import Sequential
from keras.layers import Dense
from keras_support.keras_wrapper import KerasWrapper
from mutation.mutators import cross_with_mutation
from pool import GeneticPool
import pytest
import numpy as np


@pytest.fixture
def model():
    s = Sequential()
    s.add(Dense(5, input_dim=1))
    s.add(Dense(3))
    yield KerasWrapper(s, cross_with_mutation)


def test_mutation(model):
    one = np.array([1])
    a = model.model.predict(one)
    model.mutateWith(model.new())
    b = model.model.predict(one)
    assert not np.array_equal(a, b)


def test_pool(model):
    def mock(agent):
        return 1

    pool = GeneticPool(model, mock, 10)
    pool.improve()
