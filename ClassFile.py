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