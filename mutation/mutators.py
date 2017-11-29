import numpy as np


def cross_with_mutation(genA, genB):
    for i in range(len(genA)):
        if np.random.random() > 0.5:
            genA[i] = genB[i]

        if np.random.random() > 0.99:
            genA[i] = np.random.normal()
    return genA
