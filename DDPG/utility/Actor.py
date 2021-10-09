# In this file we will declare the actor-critic model to feed in the DDPG.
from keras.layers.core import Flatten
from tensorflow.keras import Sequential
from keras.layers import Dense,Input, Activation, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.models import Model

class Actor():

    def __init__(self):
        self.state_dim = 8  # in lunar lander we have 8 states
        self.action_dim = 1   # in lunar lander we have 1 action.
        self.model = self.A_model()

    def A_model(self):
        
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """

        x = Input(self.state_dim)
        x = Dense(500)(x)  # or can do   x = Dense(500, activation='relu')
        x = Activation('relu')(x)
        x = Dense(100)(x)  # or can do   x = Dense(500, activation='relu')
        x = Activation('relu')(x)
        output = Dense(1, activation="relu")(x)  # or can do   x = Dense(500, activation='relu')

        return output

    def update_Act(self):
        pass







