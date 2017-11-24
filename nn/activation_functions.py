import numpy as np


def relu(i):
    return np.vectorize(np.maximum(i, 0, i))

def tanh(i):
    return np.tanh(i)
