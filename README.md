# GEN0 by MOBGEN:LAB
Neural Network Genetic Algorithm framework written in python, based in numpy

Genetic algorithms are relatively simple to understand and debug and can give
pretty good results in Reinforcement Learning environments.

# Creating a model
i followed the same interface that keras uses to creating their models
```python
from nn.dense import DenseLayer
from nn.sequential import Sequential
from nn.activation_functions import tanh
from mutation.mutators import cross_with_mutation

# create a sequential neural network
nn = Sequential(mutationFunction=cross_with_mutation)
# add the very first layer
nn.first(DenseLayer(size=90, activation=tanh), input_dim=8)
# add the second layer
nn.add(DenseLayer(size=200, activation=tanh))
# add the third and last layer
nn.add(DenseLayer(size=3, activation=tanh))
```

# Fitness Measure function
in order to improve the neural networks the genetic pool needs a way of measure how well a individual works. that is, how much it "fits" in his environment
this is a example of a measuring function that uses an open ai gym environment
```python
import gym

env = gym.make('LunarLanderContinuous-v2')

def measure(individual):
  state = env.reset()
  total_reward = 0

  for _ in range(1000):
    action = individual.run(state)
    state, reward, done, info = env.step(action.transpose()[0])
    total_reward += reward
    if done:
      break
  return total_reward
```

# Create the genetic pool
A genetic pool is a collection of random initialised neural networks that automatically improves their genes by natural selection and crossbreeding

```python
from pool import GeneticPool

# Create a genetic pool of 1000 individuals, using random initialized clones of nn and using the fitness measuring function measure
pool = GeneticPool(model=nn, env=measure, poolSize=1000)
```
notice how we pass the measuring function created before, automatically the poolSize will pass to this function every individual and keep track of his fitness score

# Improving the genetic pool
simply call pool.improve() and it will get fitness score, crossbreed and mutate all individuals in the pool.
it returns the max fitness score for each generation.
Every call to improve() is seen as a generation
```python
for generation in range(500):
  max_reward = pool.improve()
  print('Max reward of generation {} is {}'.format(generation, max_reward))
```

# Work in progress
- Add more types of layers, mutation and activation functions
- Improve the docs

Made by Felipe Vieira, MOBGEN:LAB 2017
