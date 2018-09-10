#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:57:34 2018

@author: acis
"""
import gym_sim_to_real
import gym
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import namedtuple
import torch.optim as optim
from itertools import count
import random, math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        self.i = nn.Linear(4,10)
        self.h1 = nn.Linear(10,100)
        self.h2 = nn.Linear(100,50)
        self.h3 = nn.Linear(50,10)
        self.o = nn.Linear(10, 3)
         
    def forward(self, x):
        #x = F.leaky_relu(self.i(x))
#        x = F.leaky_relu(self.i(x))
#        x = F.leaky_relu(self.h1(x))
#        x = F.leaky_relu(self.h2(x))
#        x = F.leaky_relu(self.h3(x))
        
#        x = F.rrelu(self.i(x))
#        x = F.rrelu(self.h1(x))
#        x = F.rrelu(self.h2(x))
#        x = F.rrelu(self.h3(x))

        x = F.relu(self.i(x))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))

        x = self.o(x)
        
        return x
        
        
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        #self.hidden_size = hidden_size

        self.i2h1 = nn.Linear(14, 10)
        #self.i2h2 = nn.Linear(5, 10)
        
        self.i2o1 = nn.Linear(14, 50)
        self.i2o2 = nn.Linear(50, 30)
        self.i2o3 = nn.Linear(30, 3)
        
        #self.softmax = nn.LogSoftmax(dim=1)
         
    def forward(self, x, last_hidden):
        
        combined = torch.cat((x, last_hidden), 1)
        hidden = self.i2h1(combined)
        #hidden = self.i2h2(hidden)
        
        output = F.leaky_relu(self.i2o1(combined))
        output = F.leaky_relu(self.i2o2(output))
        output = self.i2o3(output)
        
        #output = self.softmax(output)
        
        return output, hidden

    #def initHidden(self):
        #return torch.zeros(1, self.hidden_size)
        
        
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.i = nn.Linear(4,10)
        self.h1 = nn.Linear(10,100)
        self.h2 = nn.Linear(100,50)
        self.h3 = nn.Linear(50,10)
        self.o = nn.Linear(10, 4)
         
    def forward(self, x):
        #x = F.leaky_relu(self.i(x))
#        x = F.leaky_relu(self.i(x))
#        x = F.leaky_relu(self.h1(x))
#        x = F.leaky_relu(self.h2(x))
#        x = F.leaky_relu(self.h3(x))
        
#        x = F.rrelu(self.i(x))
#        x = F.rrelu(self.h1(x))
#        x = F.rrelu(self.h2(x))
#        x = F.rrelu(self.h3(x))

        x = F.relu(self.i(x))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))

        x = self.o(x)
        
        return x
        