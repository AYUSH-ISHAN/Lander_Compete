import gym
from keras.engine import training
from keras.layers.merge import Average
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from numpy import random
from utility.Actor import Actor
from utility.Critic import Critic   # importing actor and critic model from the file.
from utility.utils import *
from collections import deque
from tqdm import tqdm
import random
from tensorflow.keras.optimizers import Adam

env = gym.make('LunarLander-v2')
Actor = Actor()
Critic = Critic()
Orn_Uhl = Ornstein_Uhlenbeck()

REPLAY_MEMORY_SIZE = 50_000
EPISODES = 25_000
NUM_TRANS = 10
MIN_Replay = 5_000
MINI_BATCH_SIZE = 1000
GAMMA = 0.98
tau = 0.005

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

    def Policy(self, state, noise):  
        '''
        # argument is state at a given parameter theta.
        # also add the Ornstein_Uhlenbeck random data
        '''
        sampled_actions = tf.squeeze(self.Actor(state))
        sampled_actions = sampled_actions.numpy() + noise     
                        # converted tensorflow tensor to numpy array
        action = np.clip(sampled_actions, -1, 1)   # clipping between the lower and upper bound

        return action

    def A2C_update(loss, state, action):

        '''
        Update critic by minimising the loss.
        Update policy using the sampled policy gradient.
        '''

    def train_and_update(self):

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
        
        ''' If any error occurs put the trainable = True in these cases.'''


        with tf.GradientTape() as tape:

            target_action = self.target_Actor(future_state)
            Y = reward + GAMMA*self.target_Critic(future_state, target_action)
            critic_val = self.Critic([state, action])
            Loss = tf.math.reduce_mean(tf.math.square(Y, critic_val))
            
            '''
            Update the critic after minimising this loss.
            '''

        C_gradient = tape.gradient(Loss, self.Critic.trainable_variables)

        Adam.apply_gradients(zip(C_gradient, self.Critic.trainable_variables))

        ''' Training and Updating the Actor Model '''

        with tf.GradientTape() as tape:

            A_actions = self.Actor(state)
            critic_val = self.Critic([state, A_actions])

            A_Loss = -tf.math.reduce_mean(critic_val)

        A_grad = tape.gradient(A_Loss, self.Actor.trainable_variables)
        Adam.apply_gradients(zip(A_grad, self.Actor.trainable_variables))

        ''' Updating the weights '''

        for (t_w, w) in zip(self.target_Actor.weights, self.Actor.weights):

            t_w.assign(w * tau + (1 - tau) * t_w)
        
        for (t_w, w) in zip(self.target_Critic.weights, self.Critic.weights):

            t_w.assign(w * tau + (1 - tau) * t_w)


DDPG = DDPG()
episode_list = []
avg_reward_list = []

for episode in tqdm(range(1, EPISODES), ascii=True, unit='episodes'):

    steps = 1  # initilising the steps per episodes or not.
    act_noise = Orn_Uhl.OU(steps)

    episode_reward = 0
    current_state = env.reset()

    done = False
    while not done:

        action = DDPG.Policy(current_state, act_noise)

        new_state, reward, done, _ = env.step(action)
        DDPG.update_memory([current_state, action, reward, new_state])

        ''' Training and Updating the Critic Model '''

        episode_reward += reward
        DDPG.train_and_update()
        


    episode_list.append(episode_reward)
    averages_reward = np.mean(episode_list[-100:])  # average the reward per 100 epochs or last 100 epochs
    print(f'Episode of {episode} EPISODE -> {episode_reward}\nAVERAGE REWARD -> {averages_reward}')
    avg_reward_list.append(averages_reward)


plt.pyplot(episode_list)
plt.xlabel("Episodes")
plt.ylabel("Episodic Reward")
plt.show()

plt.pyplot(avg_reward_list)
plt.xlabel("Episodes")
plt.ylabel("Avg Episodic Reward (pre 100 episodes)")
plt.show()

