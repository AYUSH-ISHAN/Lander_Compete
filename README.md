# Lander Compete

<h2>Introduction :</h2>
The main purpuse of this project to do some experiment on the three well knowns Reinforcement Algorithms DQN, DDPG and SARSA in terms of their learning pace,
reward, average reward over a certain number of episodes, senstivity to the change in state and ...... # To think

<h3>The agent and environment :</h3>
 The agent is this case was the lunar lander which was attempting to have a safe landing on the moon's surface. It is an OpenAI Gym whose description can be found 
 in the link given - <a href = "https://gym.openai.com/envs/LunarLander-v2/">Lunar Lander with discete action</a> and <a href="https://gym.openai.com/envs/LunarLanderContinuous-v2/">Lunar Lander with continous action</a>.
<h3>Models Used :</h3>
<ul>
 
 <li><B>DQN -> </B>Deep Q Networks (DQN) are neural networks (and/or related tools) that utilize deep Q learning in order to provide models such as the simulation of intelligent video game play. Rather than being a specific name for a specific neural network build, Deep Q Networks may be composed of convolutional neural networks and other structures that use specific methods to learn about various processes.</li>
 <li><B>DDPG -> </B>
 Deep Deterministic Policy Gradient (DDPG) is a model-free off-policy algorithm for learning continous actions.

It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network). It uses Experience Replay and slow-learning target networks from DQN, and it is based on DPG, which can operate over continuous action spaces.</li>
 
 <li><B>SARSA -> </B>State–action–reward–state–action (SARSA) is an algorithm for learning a Markov decision process policy, used in the reinforcement learning area of machine learning. It was proposed by Rummery and Niranjan in a technical note with the name "Modified Connectionist Q-Learning" (MCQ-L).</li>
 
</ul>

<h2>Results :</h2>
The following are the rewards plots of the models. Above is the reward plot for 20 episodes while following that we have reward averages over last 5 episodes.
<h3><B>DDPG :</B></h3>
<img src ="https://github.com/AYUSH-ISHAN/Lander_Compete/blob/main/DDPG_reward.png" height = "400" width = "400" align="center"/><img src ="https://github.com/AYUSH-ISHAN/Lander_Compete/blob/main/DDPG_avg_reward.png" height = "400" width = "400" align="center"/>
<br>
<h3><B>DQN :</B></h3>
<img src ="https://github.com/AYUSH-ISHAN/Lander_Compete/blob/main/DQN_reward.png" height = "400" width = "400" align="center"/><img src ="https://github.com/AYUSH-ISHAN/Lander_Compete/blob/main/DQN_avg_reward.png" height = "400" width = "400" align="center"/>
<h3><B>SARSA :</B></h3>
<img src ="https://github.com/AYUSH-ISHAN/Lander_Compete/blob/main/Sarsa_reward.png" height = "400" width = "400" align="center"/><img src ="https://github.com/AYUSH-ISHAN/Lander_Compete/blob/main/Sarsa_avg_reward.png" height = "400" width = "400" align="center"/>






 
