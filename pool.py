# 2019
# manages a pool of neural nets
import numpy as np


class GeneticPool():
    def __init__(self, model, env, poolSize):
        self.poolRange = range(poolSize)
        self.individuals = [model.new() for _ in self.poolRange]
        self.fitnesses = [0 for _ in self.poolRange]
        self.env = env

    def measure_performance(self, agent):
        return self.env(agent)

    def improve(self):
        # Measure fitness of all individuals
        for i in self.poolRange:
            self.fitnesses[i] = self.measure_performance(self.individuals[i])
            print(self.fitnesses[i])
            #print ("reward of {} is :{}".format(i, self.fitnesses[i]))

        # Get the winner of the generation
        winner = np.argmax(self.fitnesses)
        max_reward = self.fitnesses[winner]

        # Cross all individuals with the winner
        for individual in self.individuals:
            individual.mutateWith(self.individuals[winner])

        # return max reward of the generation
        return max_reward
