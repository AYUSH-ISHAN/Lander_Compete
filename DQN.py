
import torch
import torch.nn as nn
from collections import deque
import random
import numpy as np

'''ALSO MAKE A CUSTOM CALLBACK AS DID IN SEMANTIC SEGMENTATION'''

EPISODES = 25000
EPSILON = 1e-4
GAMMA = 0.1
LANDING_REWARD = 100
COMPLETION_REWARD = 200
PENALTY_CRASH = -100
PENALTY_MAIN_ENGINE = -0.3
PENALTY_SIDE_ENGINE = -0.03
REWARD_LEG_CONT = 10
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1000
VERBOSE = 1
num_action_spaces = 4  # if this shows error then do env.action_space.n

class DQN(nn.Module):
    def __init__(self):

        super(DQN,self).__init__()
        self.nets = nn.Sequential(
            nn.Linear(8, 512),
            nn.ReLU(),    # nn.LeakyReLU()
            nn.Linear(512, 256),
            nn.ReLU(),    # nn.LeakyReLU()
            # nn.Linear(),
            # nn.ReLU(),    # nn.LeakyReLU()
            # nn.Linear(),
            # nn.ReLU(),    # nn.LeakyReLU()
            nn.Linear(256, 4)
        )

    def Forward(self, state):
        state = torch.tensor(state)
        q_val = self.nets(state)
        return q_val

# nets = DQN()
# print(nets.Forward([1, 2, 3]))


##  MAY BE WE DON'T NEED THIS CLASS AS WE MAY USE TRAINING AND REPLAY IN ONE CLASS ONLY.

class Replay_Memory():

    '''We will pass the q_values and store them in a Deque 
    type storage and check its validity here.'''

    def __init__(self):
        self.replay_memory = deque(REPLAY_MEMORY_SIZE)

    def update_replay(self, states):
        self.replay.append(states)



## may use a random sampling or shuffling to reduce variance or biaseness.
        


class Agent():

    '''Here, is the main process of training of agent will take place
    '''
    def __init__(self):
        self.replay_memory = deque(REPLAY_MEMORY_SIZE)

    def update_replay(self, states):
        self.replay_memory.append(states)


    ## selecting action - as prescribed in Q learning after Bellman Update.

    def Action_selection(self, state):

        ''' We will pass the state to the Qnets to get the Q-value of action'''

        para = random.random()  ## check difference between random.random() and random.randint()

        if para < EPSILON:
            action = torch.tensor([random.randrange(num_action_spaces)])

        else:
            state_tensor = torch.tensor([state])
            action = np.argmax(DQN.Forward(state_tensor))

        return action


    def BE_update_optimise(self):

        # firstly, we should have some descent amount of data.
        if self.replay_memory < MIN_REPLAY_MEMORY_SIZE:
            return

        '''Take the states or the transitions and then feed them to DQN Net to take 
        output. --> take in the actions and then optimise it.'''

        trasitions = torch.tensor([state])







'''Make the epoch running part out of any class and make it in default part
that is without any function or class.'''

class Training():

    ''' Also make a custom callback.'''

    def __init__(self):
        self.replay_memory = deque(REPLAY_MEMORY_SIZE)

    def update_replay(self, states):
        self.replay.append(states)

    for epoch in EPISODES:

        '''we will trainin here the full batch size of deque and call the 
        functions which optimise it and makes our model learn more'''
        pass
    


'''
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers
in state vector. Reward for moving from the top of the screen to landing pad and
zero speed is about 100..140 points. If lander moves away from landing pad it loses
reward back. Episode finishes if the lander crashes or comes to rest, receiving
additional -100 or +100 points. Each leg ground contact is +10. Firing main engine
is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is
possible. Fuel is infinite, so an agent can learn to fly and then land on its 
first attempt. Four discrete actions available: do nothing, fire left orientation
engine, fire main engine, fire right orientation engine.
'''
