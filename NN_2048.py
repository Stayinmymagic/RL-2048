# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:41:14 2021

@author: notfu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
class CNN_Net(nn.Module):
    def __init__(self, input_len, output_num, conv_size=(256, 512), fc_size=(1024, 256), out_softmax=False):
        super(CNN_Net, self).__init__()
        self.input_len = input_len
        self.output_num = output_num
        self.out_softmax = out_softmax 

        self.conv = nn.Sequential(
            nn.Conv2d(input_len, conv_size[0], kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_size[0], conv_size[1], kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True),
            )
        self.linear = nn.Sequential(
            nn.Linear(conv_size[1] * int(np.sqrt(self.input_len)), fc_size[0]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_size[0], fc_size[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_size[1], self.output_num)
            )
            
        
     # takes in a module and applies the specified weight initialization

        
    def forward(self, x):
        #print(x.shape)
        #x = x.reshape(-1,16,int(np.sqrt(self.input_len)), int(np.sqrt(self.input_len)))
        x = torch.reshape(x, (-1,16,int(np.sqrt(self.input_len)), int(np.sqrt(self.input_len))))
        #print(x.shape)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        output = self.linear(x)
        if self.out_softmax:
            output = F.softmax(output, dim=1)   #值函数估计不应该有softmax
        #print(output.shape)
        return output