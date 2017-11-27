from nn.lstm.lstm import LSTM
from mutation.mutators import cross_with_mutation
from pool import GeneticPool
import gym



env = gym.make('LunarLanderContinuous-v2')

sq = LSTM(8, 2, cross_with_mutation)


def measure(agent):
    state = env.reset()
    total_reward = 0

    for _ in range(1000):
        #env.render()
        action = agent.run(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


pool = GeneticPool(model=sq, env=measure, poolSize=10000)
for generation in range(500):
    max_reward = pool.improve()
    print('Max reward of generation {} is {}'.format(generation, max_reward))
