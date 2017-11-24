from nn.dense import DenseLayer
from nn.sequential import Sequential
from nn.activation_functions import tanh
from mutation.mutators import cross_with_mutation
from pool import GeneticPool
import gym

sq = Sequential(mutationFunction=cross_with_mutation)
sq.first(DenseLayer(size=90, activation_function=tanh), input_dim=8)
sq.add(DenseLayer(size=90, activation_function=tanh))
sq.add(DenseLayer(size=2, activation_function=tanh))

env = gym.make('LunarLanderContinuous-v2')


def measure(agent):
    state = env.reset()
    total_reward = 0

    for _ in range(1000):
        #env.render()
        action = agent.run(state)
        state, reward, done, info = env.step(action.transpose()[0])
        total_reward += reward
        if done:
            break
    return total_reward


pool = GeneticPool(model=sq, env=measure, poolSize=1000)
for generation in range(500):
    max_reward = pool.improve()
    print('Max reward of generation {} is {}'.format(generation, max_reward))
