# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:56:20 2022

@author: khale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple,deque
import random
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self):
        self.buffer = []
        self.max_size = int(500000)
        self.size = 0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state","done"])
    
    def add(self, state, action, reward, new_state, done):
        self.size +=1
        e = self.experience(state, action, reward, new_state,done)
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        experiences = random.sample(self.buffer, k=batch_size)

        state, action, reward, next_state, done = [], [], [], [], []
        for i in range(len(experiences)):

            state.append(np.array(experiences[i].state, copy=False))
            action.append(np.array(experiences[i].action, copy=False))
            reward.append(np.array(experiences[i].reward, copy=False))
            next_state.append(np.array(experiences[i].next_state, copy=False))
            done.append(np.array(experiences[i].done, copy=False))
        
        _batchaction=np.array(action)
        batchstate = torch.FloatTensor(np.array(state)).to(device)
        batchaction = torch.FloatTensor(_batchaction).to(device)
        batchreward = torch.FloatTensor(np.array(reward)).reshape((batch_size,1)).to(device)
        batchnextstate = torch.FloatTensor(np.array(next_state)).to(device)
        done = torch.FloatTensor(np.array(done)).reshape((batch_size,1)).to(device)
            
        return _batchaction,batchstate, batchaction, batchreward, batchnextstate, done
    


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
       
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a
        
    



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 500)
        self.l2 = nn.Linear(500, 300)
        self.l3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        
        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
    
class dqn:
    def __init__(self,gamma, batch_size, lr, state_dim, action_dim):
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=lr)
        
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_optimizer = optim.RMSprop(self.critic_1.parameters(), lr=0.001)
        
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_optimizer = optim.RMSprop(self.critic_2.parameters(), lr=0.001)
        
        self.batch_size=batch_size
        self.gamma=gamma
        self.polyak=0.995
        self.replaymemory = ReplayMemory()
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def learn(self,_batchaction, batchstate,batchnextstate,batchreward,batchaction,done,steps,targetmodel):
          
        # Select next action according to target policy:

            next_action = (targetmodel.actor(batchnextstate))
            next_action = next_action.clamp(-1, 1)
            
            # Compute target Q-value:
            target_Q1 = targetmodel.critic_1(batchnextstate, next_action)
            target_Q2 = targetmodel.critic_2(batchnextstate, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = batchreward + ((1-done) * self.gamma * target_Q).detach()
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(batchstate, batchaction)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            if loss_Q1>20:
                print("HIGH LOSS Q1: "+str(loss_Q1))
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(batchstate, batchaction)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            if loss_Q2>20:
                print("HIGH LOSS Q2: "+str(loss_Q2))
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            
           
            if steps%2==0:
                # Compute actor loss:
                actor_loss = -self.critic_1(batchstate, self.actor(batchstate)).mean()
                
                if actor_loss>20:
                    print("HIGH LOSS Actor: "+str(actor_loss))
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                

    def update(self,state, action, reward, next_state, done,steps,target_brain ):
        
        
        # Sample a batch of transitions from replay buffer:
        self.replaymemory.add(state, action, reward, next_state,done)
        
        if self.replaymemory.size>self.batch_size:
            _batchaction,batchstate, batchaction, batchreward, batchnextstate, done = self.replaymemory.sample(self.batch_size)
            self.learn(_batchaction,batchstate,batchnextstate,batchreward, batchaction,done,steps,target_brain)
    
    def getsdictionary(self):
        return ([self.actor.state_dict(),self.critic_1.state_dict(),self.critic_2.state_dict()])
    def setsdictionary(self,modellist):
        self.actor.load_state_dict(modellist[0])
        self.critic_1.load_state_dict(modellist[1])
        self.critic_2.load_state_dict(modellist[2])
    def seteval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
    def save(self,name):
        torch.save({'statedict':self.actor.state_dict(),
                    'optimizer':self.actor_optimizer.state_dict(),
                    },name+'actor.pth')
        torch.save({'statedict':self.critic_1.state_dict(),
                    'optimizer':self.critic_1_optimizer.state_dict(),
                    },name+'critic1brain.pth')
        torch.save({'statedict':self.critic_2.state_dict(),
                    'optimizer':self.critic_2_optimizer.state_dict(),
                    },name+'critic2brain.pth')
        
        
    def load(self,name):
        print(name)
        succ=0
        if os.path.isfile(str(name)+'actor.pth'):
            print("loading file")
            checkpoint = torch.load(str(name)+'actor.pth')
            self.actor.load_state_dict(checkpoint['statedict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer'])
            print("done")
            succ=1

        else:
            print("no load ):")
            
            
            
        succ=0
        if os.path.isfile(str(name)+'critic1brain.pth'):
            print("loading file")
            checkpoint = torch.load(str(name)+'critic1brain.pth')
            self.critic_1.load_state_dict(checkpoint['statedict'])
            self.critic_1_optimizer.load_state_dict(checkpoint['optimizer'])
            print("done")
            succ=1

        else:
            print("no load ):")
            
        succ=0
        if os.path.isfile(str(name)+'critic2brain.pth'):
            print("loading file")
            checkpoint = torch.load(str(name)+'critic2brain.pth')
            self.critic_2.load_state_dict(checkpoint['statedict'])
            self.critic_2_optimizer.load_state_dict(checkpoint['optimizer'])
            print("done")
            succ=1

        else:
            print("no load ):")
            
        
        return succ
    
        
 
  
      
        
