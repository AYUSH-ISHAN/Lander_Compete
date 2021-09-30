###  first implement in tensorflow and then go for pytorch
from hashlib import new
from os import terminal_size
from keras.layers import Activation, Flatten
import random
from keras import Sequential
import numpy as np
import tensorflow
import tensorflow.keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from collections import deque
from keras.layers import Dense, Conv2D
from tqdm import tqdm
import gym

env = gym.make('LunarLander-v2')
##   according to the conditions
DISCOUNT = 0.99
EPISODES = 25000
EPSILON = 1
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.99975
GAMMA = 0.1
# LANDING_REWARD = 100
# COMPLETION_REWARD = 200
# PENALTY_CRASH = -100
# PENALTY_MAIN_ENGINE = -0.3
# PENALTY_SIDE_ENGINE = -0.03
# REWARD_LEG_CONT = 10
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1000
VERBOSE = 1
num_action_spaces = 4  
MINI_BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
AGGREGATE_STATS_EVERY = 50  # saving stats after 50 epochs

EP_REWARD = []  # In this we will store the rewards from each episodes.

callbacks = [
             ModelCheckpoint("model.h5", save_best_only=True, verbose=1),
             ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, min_lr=1e-6, verbose=1),
             EarlyStopping(monitor='val_loss', patience=5, verbose=1)
]

class DQNAgent():
    def __init__(self):
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.model = self.DQN()
        self.target_model = self.DQN()
        self.target_update_counter = 0

    def update_memory(self, transition):
        self.replay_memory.append(transition)

    def DQN(self):
        '''
        Here, we can do Input shapes = env.n_observation  or action_space.n
        This will help us in getting the direct result of observation space and 
        action space dimensions.
        '''

        '''
        May be we can include some dropouts, Pooling layers and all for better results 
        '''
        model = Sequential()
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        # model.Flatten()
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(4))
        return model

    def Q_values(self, state):

        return self.model(np.array(state).reshape(-1, state.shape)/255)[0]

    def train(self, step, terminal_state):
        
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINI_BATCH_SIZE)

        current_states = np.array([transitions[0] for transitions in minibatch])/255
        current_q_list = self.model.predict(current_states)

        new_states = np.array([transitions[3] for transitions in minibatch])/255
        next_q_list = self.target_model.predict(new_states)

        X = []
        y = []

        for index, (current_state, action, reward, next_current_state, done) in enumerate(minibatch):

            if not done:
                future_q_max = np.max(next_q_list[index])  # max Q of action that is to be taken
                next_q = reward + DISCOUNT*future_q_max
            else:
                next_q = reward

            ## update the qs values to the current Q table 

            current_qs = current_q_list[index]
            current_qs[action] = next_q
                
            X.append(current_state)
            y.append(current_qs)

            '''Also introduce its own custom callback -- as did in Semantic Seg.'''
            self.model.fit(np.array(X)/255, np.array(y), batch_size=MINI_BATCH_SIZE, shuffle=False, callbacks=callbacks)

            '''we will set the weights of target model as that of original model
            used and then comparing it'''

            if terminal_state:
                self.target_update_counter += 1

            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0
        
    def fetch_Qs(self, state):
        
        '''just print and see what is the return of predict'''
        print(self.model.predict(np.array(state).reshape(-1, *state.shape)/255))
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

##  after this we will run episodes to train our model.

agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii='True', unit='episodes'):
        
        '''
        1. Look at Lunar Lander github code to see how its coded.
        2. Look at Coursera RL assignemts and see how they have the Lunar Lander Model
        '''

        episode_reward = 0
        step = 1

        current_state = env.reset()

        done = False  ## this waits till terminal step
        while not done:

            if np.random.random() > EPSILON:
                action = np.argmax(agent.fetch_Qs(current_state))
            else:
                action = np.random.randint(0, 4)

            new_state, reward, done, _ = env.step(action)

            episode_reward += reward

            if episode % 100:
                env.render()  
                ''' You can use SHOW_PREVIEW or similarly
                AGGREGATE STATS EVERY more detailing
                 '''
            
            agent.update_memory([current_state, action, reward, new_state, done])
            agent.train(step, done)
            current_state = new_state
            step+=1
        
        EP_REWARD.append(episode_reward)

        '''
            In this subsection update the Logs + Savce the model.
            Below this we will do the gradient descent step.
        '''
        agent.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        # Epsilon secay step:

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)
            

'''
I THINK I AM DONE !!
'''
###########################################################################################################################










