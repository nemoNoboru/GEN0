import numpy as np


def cross_with_mutation(genA, genB):
    g = []
    for i in range(len(genA)):
        if np.random.random() > 0.5:
            g.append(genA[i])
        else:
            g.append(genB[i])

        if np.random.random() > 0.99:
            g.pop()
            g.append(np.random.normal())
    return g
