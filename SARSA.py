import numpy as np
from collections import deque
from tqdm import tqdm
import gym

########   THINK OF A NEW POLICY DERIVED FROM Q - as discussed in the book.   ########                

env = gym.make('LunarLander-v2')
DISCOUNT = 0.99
EPISODES = 25000
EPSILON = 1
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.99975
GAMMA = 0.1
REPLAY_MEMORY_SIZE = 1
MIN_REPLAY_MEMORY_SIZE = 1
VERBOSE = 1
num_action_spaces = 4  
ALPHA = 0.65
MINI_BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
AGGREGATE_STATS_EVERY = 50  # saving stats after 50 epochs

EP_REWARD = []  # In this we will store the rewards from each episodes.

current_q_list = [0,0,0,0]

class SARSAAgent():
    def __init__(self):
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def update_memory(self, transition):
        self.replay_memory.append(transition)

    def Q_values(self, state):

        return self.model(np.array(state).reshape(-1, 8)/255)[0]

    def Q_update(self, current_q_list, terminal_state):
        current_states = np.array([transitions[0] for transitions in self.replay_memory])/255  
        # or can do self.replay_memory[0] - as only one element is there in the array. 

        new_states = np.array([transitions[3] for transitions in self.replay_memory])/255

        for index, (current_state, action, reward, next_current_state, done) in enumerate(self.replay_memory):

            if not terminal_state:
                # max Q of action that is to be taken
                next_q = [reward + DISCOUNT*Q_val for Q_val in current_q_list]
            else:
                next_q = [0,0,0,0]

            for i in range(3):
                current_q_list[i] = current_q_list[i] + ALPHA*(next_q[i] - current_q_list[i])
        
        return current_q_list

    def Policy(self, current_state):   #####  Using Epsilon greedy algorithm as Policy
        
        if np.random.random() > EPSILON:
            action = agent.fetch_Qs(current_state)
        else:
            action = np.random.randint(0, 4)

        return action

##  after this we will run episodes to train our model.

agent = SARSAAgent()

for episode in tqdm(range(1, EPISODES+1), ascii='True', unit='episodes'):

    '''
    1. Look at Lunar Lander github code to see how its coded.
    2. Look at Coursera RL assignemts and see how they have the Lunar Lander Model
    '''
    episode_reward = 0

    current_state = env.reset()

    done = False  # this waits till terminal step
    while not done:

        action = agent.Policy(current_state)

        new_state, reward, done, _ = env.step(action)

        episode_reward += reward
        if episode % 100:
            env.render()
            ''' You can use SHOW_PREVIEW or similarly
                AGGREGATE STATS EVERY more detailing
                 '''

        agent.update_memory([current_state, action, reward, new_state, done])
        new_q_list = agent.Q_update(current_q_list, done)
        current_state = new_state

    EP_REWARD.append(episode_reward)

    '''
            In this subsection update the Logs + Savce the model.
            Below this we will do the gradient descent step.
        '''
    
    # Epsilon decay step:

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
        EPSILON = max(MIN_EPSILON, EPSILON)


'''
I THINK I AM DONE !!
'''
###########################################################################################################################
