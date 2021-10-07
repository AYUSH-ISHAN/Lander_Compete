## Serial Version :

#%%time   # uncomment them when using in jupyter notebook

import gym
import numpy as np
import time

def run1():
    env = gym.make("LunarLander-v2")
    env.reset()
    steps = []
    for _ in range(1):
      obs, reward, done, info = env.step(env.action_space.sample())
      
      if (done):
        break

    return len(steps)


for i in range(1):
  result = run1()

# Parallel Version

#%%time   # uncomment them when using jupyter notebook

def run2():
    env = gym.make("LunarLnader-v2")
    env.reset()
    steps = []

    for _ in range(1):  # for nice training make it to 100
      obs, reward, done, info = env.step(env.action_space.sample())
      if(done):
        break
    
    return len(steps)

for i in range(1):
  result = run2()