# In this file we will declare the actor-critic model to feed in the DDPG.
from keras.layers.core import Flatten
from tensorflow.keras import Sequential
from keras.layers import Dense,Input, Activation, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.models import Model

class Critic():

    def __init__(self):
        self.state_dim = 8  # in lunar lander we have 8 states
        self.action_dim = 1   # in lunar lander we have 1 action.

    def C_model(self):
        
        state = Input((self.state_dim))
        action = Input((self.action_dim))
        x = Dense(500)(state)  # x = Dense(500, activation="relu")
        x = concatenate([Flatten()(x), action])
        x = Dense(100)(x)
        x = Activation('relu')(x)
        output = Dense(1)(x)

        return Model([state, action], output)

    def Bellman_update(self, state, action):
        '''
        Update the CommNet using BellMan's euqation.
        '''














