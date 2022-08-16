# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:54:13 2021

@author: notfu
"""
import numpy as np
from random import sample, randint

class ReplayBuffer:
    def __init__(self, buffer_size, obs_space):
        self.s1 = np.zeros(obs_space, dtype=np.float32)
        self.s2 = np.zeros(obs_space, dtype=np.float32)
        self.a = np.zeros(buffer_size, dtype=np.int32)
        self.r = np.zeros(buffer_size, dtype=np.float32)
        self.done = np.zeros(buffer_size, dtype=np.float32)

        # replaybuffer size
        self.buffer_size = buffer_size
        self.size = 0
        self.pos = 0

    # store data into buffer
    def add_transition(self, s1, action, s2, done, reward):
        self.s1[self.pos] = s1
        self.a[self.pos] = action
        if not done:
            self.s2[self.pos] = s2
        self.done[self.pos] = done
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    # random sample a batchsize
    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        
        return self.s1[i], self.a[i], self.s2[i], self.done[i], self.r[i]
    