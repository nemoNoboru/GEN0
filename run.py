from nn.dense import DenseLayer
from nn.models.sequential import Sequential
from nn.activation_functions import tanh
from mutation.mutators import cross_with_mutation
from pool import GeneticPool
from server import Gym

sq = Sequential(mutationFunction=cross_with_mutation)
sq.first(DenseLayer(size=10, activation=tanh), input_size=7)
sq.add(DenseLayer(size=9, activation=tanh))
sq.add(DenseLayer(size=4, activation=tanh))

env = Gym()


def measure_unity(agent):
    total_reward = 0
    state = env.step([0, 0, 0, 0])['state']
    for _ in range(1000):
        #env.render()
        action = agent.run(state)
        print(action)
        observation = env.step(action.tolist())
        total_reward += observation['reward']
        if observation['done']:
            break
        state = observation['state']
    return total_reward


pool = GeneticPool(model=sq, env=measure_unity, poolSize=10)
for generation in range(500):
    max_reward = pool.improve()
    print('Max reward of generation {} is {}'.format(generation, max_reward))
