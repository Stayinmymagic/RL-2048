# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:42:04 2021

@author: notfu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from NN_2048 import CNN_Net
from ReplayBuffer_2048 import ReplayBuffer
import math
#%%
class DQN():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #hyperparameter
    batch_size = 128
    lr = 1e-3
    epsilon = 0.15   
    Net_update_iteration = 200 # same as episilon decay timing
    soft_update_theta = 0.2
    train_interval = 10
    gamma = 0.99
    clip_norm_max = 1
    conv_size = (64, 128)
    fc_size = (128, 64)
    conv_size=(256, 512)
    fc_size=(1024, 256)
    
    def __init__(self, num_state, num_action):
    
        self.num_state = num_state
        self.num_action = num_action
        
        #create behavior net to interact with env, and update parameters of target net by behavior net.
        self.behavior_net = CNN_Net(self.num_state, self.num_action, self.conv_size, self.fc_size).to(self.device)
        self.target_net = CNN_Net(self.num_state, self.num_action, self.conv_size, self.fc_size).to(self.device)
        self.behavior_net.apply(self.weights_init_uniform)   
        self.target_net.apply(self.weights_init_uniform)   
        #set-up
        self.learn_step_counter = 0
        self.initial_epsilon = self.epsilon
        self.optimizer = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr)
        
    # takes in a module and applies the specified weight initialization
    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        #if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
        if classname == "linear" or classname == "Conv2d":
            m.weight.data.uniform_(0.0, 0.1)
            m.bias.data.fill_(0)

       
    def select_action(self, state, random = False):
        state = torch.FloatTensor(state).to(self.device)
        #greedy policy
        if not random and np.random.random() > self.episilon:
            #use Network to calculate Q-value in each action
            action_value = self.behavior_net(state)
            #choose max Q-value as the action
            action = torch.max(action_value.reshape(-1, 4), 1)[1].data.cpu().numpy()
        #random policy
        else:
            action = np.random.randint(0, self.num_action)
        
        return action
    
    def update(self, buffer):
        
        #soft update parameters, update target network
        if self.learn_step_counter % self.Net_update_iteration ==0 and self.learn_step_counter:
            for p_e, p_t in zip(self.behavior_net.parameters(), self.target_net.parameters()):
                p_t.data = self.soft_update_theta * p_e.data + (1 - self.soft_update_theta) * p_t.data
                
        self.learn_step_counter+=1
        
        #sample a batchsize from buffer
        b_states, b_actions, b_next_states, b_done, b_rewards = buffer.get_sample(self.batch_size)
        b_states = torch.FloatTensor(b_states).to(self.device)
        b_next_states = torch.FloatTensor(b_next_states).to(self.device)
        b_actions = torch.FloatTensor(b_actions).to(self.device)
        b_rewards = torch.FloatTensor(b_rewards).to(self.device)
        #print(b_rewards.shape)
        
        #send batch samples to behavior net and calculate Q-value at state t
        Q_value_total = self.behavior_net(b_states)
        #get actions from values
        Q_behavior= Q_value_total.gather(1, b_actions.to(torch.int64).view(1, 128))
        #send batch samples to target net and calculate Q-value at state t+1
        next_Q_value_total = self.target_net(b_next_states).detach()
        #print('value', next_Q_value_total.shape)
        Q_max = next_Q_value_total.max(1)[0].view(1, self.batch_size)
        #print('max', Q_max.shape)
        Q_target = b_rewards + self.gamma * Q_max
        #print(Q_target.shape)
        #calculate loss by the difference between 2 Q-networks output
        loss = F.mse_loss(Q_behavior, Q_target)
        #print(Q_behavior)
        #print(Q_target)
        self.optimizer.zero_grad()
        #update behavior network
        loss.backward()
        #avoid gradient vanishing
        nn.utils.clip_grad_norm_(self.behavior_net.parameters(), self.clip_norm_max)
        self.optimizer.step()

        return loss
    def epsilon_decay(self, episode, total_episode):
        self.epsilon = self.initial_epsilon * (1 - episode / total_episode)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        