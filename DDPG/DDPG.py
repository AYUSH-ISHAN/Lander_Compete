import gym
import keras.backend as K
from matplotlib.pyplot import step
from numpy import random
from utility.Actor import Actor
from utility.Critic import Critic   # importing actor and critic model from the file.
from utility.utils import *
from collections import deque
from tqdm import tqdm
import random

env = gym.make('LunarLander-v2')
Actor = Actor()
Critic = Critic()
Orn_Uhl = Ornstein_Uhlenbeck()

REPLAY_MEMORY_SIZE = 50_000
EPISODES = 25_000
NUM_TRANS = 10
MIN_Replay = 5_000
MINI_BATCH_SIZE = 1000


class DDPG():

    def __init__(self):

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)   # to initialise a replay buffer.
        self.Critic = Critic.C_model()
        self.Actor = Actor.A_model()
        self.target_Critic = Critic.C_model()
        self.target_Actor = Actor.A_model()
        self.trans_per_iter = NUM_TRANS
        self.min_replay = MIN_Replay

    def update_memory(self, transitions):

        self.replay_memory.append(transitions)

    def Policy(state):  
        '''
        # argument is state at a given parameter theta.
        # also add the Ornstein_Uhlenbeck random data
        '''

    def A2C_update(loss, state, action):

        '''
        Update critic by minimising the loss.
        Update policy using the sampled policy gradient.
        '''

    def train(self):

        '''
        From here I will Monitor the whole update thing and helper funtions calling.
        '''
        if len(self.replay_memory) < self.min_replay:
            return

        miniBatch = random.sample(self.replay_memory, MINI_BATCH_SIZE)
        # miniBatch is of format DDPG.update_mamory() in the episode running step.

        state = [trans[0] for trans in miniBatch]
        action = [trans[1] for trans in miniBatch]
        reward = [trans[2] for trans in miniBatch]
        future_state = [trans[3] for trans in miniBatch]
        



DDPG = DDPG()

for episode in tqdm(range(1, EPISODES), ascii=True, unit='episodes'):

    steps = 1  # initilising the steps per episodes or not.
    act_noise = Orn_Uhl.OU(steps)

    episode_reward = 0
    current_state = env.reset()

    done = False
    while not done:

        action = DDPG.Policy(current_state) + act_noise

        new_state, reward, done, _ = env.step(action)
        DDPG.update_memory([current_state, action, reward, new_state])

        

        











