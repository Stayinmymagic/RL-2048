# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:36:35 2021
@author: notfu
"""
from gym_2048 import Game2048Env, stack
from DQN_2048 import DQN
from ReplayBuffer_2048 import ReplayBuffer
import torch
import numpy as np
import time
import math
#%%hyperparameters
train_episodes = 5000
#test_episodes = 5000
eval_interval = 100
epsilon_decay = 200
buffer_size = 100000
episodes = train_episodes
max_reward = 0
agent = DQN(num_state=16, num_action=4)
env = Game2048Env()
buffer = ReplayBuffer(buffer_size=buffer_size, obs_space=
(buffer_size, env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2]))

for i in range(episodes):
    state, reward, done, info = env.reset()
   
    loss = None
    total_reward = 0
    #New game start
    while True:

        #create samples
        if buffer.size <= buffer_size:
            action = agent.select_action(state, random = True)
        else:
            action = agent.select_action(state)
        #interact with env
        next_state, reward, done, info = env.step(action)

        #store smaples
        buffer.add_transition(state, action, next_state, done, reward)
        
        #make board move to next state
        state = next_state
        #calculate total reward in the episode
        total_reward += reward
        #use samples to update target network or not, train_intervals = 間隔多少次才update Q Network; buffer.size > 10000才開始update target
        if buffer.size % agent.train_interval == 0 and buffer.size >= 10000:
            loss = agent.update(buffer)
        # if game over
        if done:
            if i % epsilon_decay == 0 and i:#到了要衰減episilon的時刻
                agent.epsilon_decay(i, episodes)
                print('Epiosdes: ', i,  '| Ep_reward: ', round(total_reward, 2), 'loss: ', loss)
            
            if total_reward > max_reward:
                max_reward = total_reward
                print("current_max_reward {}".format(max_reward))
            # 保存模型
            #torch.save(dqn.behaviour_model, "2048.pt")
            break

