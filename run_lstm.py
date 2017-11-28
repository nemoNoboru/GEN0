from nn.models.lstm.lstm import LSTM
from mutation.mutators import cross_with_mutation
from pool import GeneticPool
from server import Gym

sq = LSTM(7, 1, cross_with_mutation)
env = Gym()


def measure_unity(agent):
    total_reward = 0
    state = env.step([0, 0, 0, 0])['state']
    while True:
        #env.render()
        action = agent.run(state)
        #print(action)
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
