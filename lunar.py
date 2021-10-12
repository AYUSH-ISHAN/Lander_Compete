import tensorflow
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

import gym
env = gym.make('LunarLander-v2')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    print(env.observation_space.shape)  # output is -- (8,)
    print(env.action_space.shape)  # output is -- ()

env.close()