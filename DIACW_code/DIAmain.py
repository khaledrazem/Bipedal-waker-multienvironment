# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:56:20 2022

@author: khale
"""

import torch
import gym
import numpy as np
from ai import dqn
from matplotlib import pyplot as plt


#################### Hyperparameters ####################

env_name = ["BipedalWalker-v3","BipedalWalkerHardcore-v3"]
current_env=1          # environment index for env_name
log_interval = 10           # print avg reward after interval         
Frame_Skips=4               # number of frames to use the same action
expl_noise = 0.4            # strength of random actions
randomstart=10              # number of frames to play randomly at the begining of an episode
gamma = 0.99                # discount for future rewards
batch_size = 200            # num of transitions sampled from replay buffer
lr = 0.0001                 # Learning rate for actor 
train_episodes = 1000       # max num of training episodes
eval_episodes=100           # number of episodes for testing
targetupdate=10000          # timesteps before updating target network
max_timesteps = 6000        # max timesteps in one episode
visual=True                 # boolean to view game environment
Load=True                   # boolean to load pretrained model
Save=True                   # boolean to save new model
Loadindex=0                 # which game environment model to load

##########################################################

env = gym.make(env_name[current_env])
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

brain = dqn(gamma,batch_size,lr, state_dim, action_dim)
target_brain=dqn(gamma,batch_size,lr, state_dim, action_dim)
target_brain.setsdictionary(brain.getsdictionary())
target_brain.seteval()


#load pretrained
if Load:
    brain.load(env_name[Loadindex])
    train_episodes=0
    expl_noise=0
    randomstart=0
    target_brain.setsdictionary(brain.getsdictionary())


#score storage
avg_reward = 0
ep_reward = 0
all_episodes=train_episodes+eval_episodes
scores=np.zeros(all_episodes)


# training procedure:
for episode in range(1, all_episodes):
    state = env.reset()
    expl_noise *= 0.9995

    for t in range(max_timesteps):

        
        if visual:
            env.render()
        
        # select action and add exploration noise:
        if t>randomstart and t%Frame_Skips==0:
            action = brain.select_action(state)
            action = action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])
            action = action.clip(-1, 1)
        elif t<=randomstart:
            action=env.action_space.sample()
            
        # take action in env:
        next_state, reward, done, _ = env.step(action)
        
         #update reward
        avg_reward += reward
        ep_reward += reward
        
        if reward==-100:
            reward=-25

        brain.update(state, action, reward, next_state, float(done),t,target_brain)
        state = next_state
        
       
        
        #update target network
        if t % targetupdate == 0: 
            target_brain.setsdictionary(brain.getsdictionary())
            
        # if episode is done then update brain:
        if done or t==(max_timesteps-1):
            break
    
    scores[episode]=ep_reward
    ep_reward = 0
    
    # if episode==train_episodes:
    #     expl_noise=0

    
    # print avg reward every log interval:
    if episode % log_interval == 0:
        avg_reward = int(avg_reward / log_interval)
        print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
        avg_reward = 0

if Save and not Load:
    print("SAVING")
    brain.save( env_name[current_env])

print("evaluation average"+str(sum(scores[-eval_episodes:])/eval_episodes))
print(scores)
a, b = np.polyfit(range(all_episodes),scores, 1)
plt.plot(scores)
plt.plot(a*range(all_episodes)+b)   
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()
print("Gradient: "+str(a))


env.close()