#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 19:38:24 2018

@author: acis
"""

import environment as envo
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
import ClassFile


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
plt.ion()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

#env = envo.Environment("sim")
env = envo.Environment("sim", "localhost", 18001)
env.reset()


total_episodes = 100
steps_done = 0
episode_durations = []



#target_net.load_state_dict('target_net.pth')
policy_net = ClassFile.RNN_disc().to(device)
policy_net = torch.load('disc.pth', map_location='cpu')

policy_net.eval()
hidden_size = 10;
last_hidden = torch.zeros(1, hidden_size, device = device, dtype = dtype)
        
def select_action(state):
    global steps_done, last_hidden
    
    steps_done += 1
    action, last_hidden = policy_net(state, last_hidden)
    return action.max(1)[1].view(1, 1)
    
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Inference')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    

    plt.plot(durations_t.numpy())
    
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
for i_episode in range(total_episodes):
    # Initialize the environment and state
    state = env.reset()
    env.render()
    state = torch.tensor([state], dtype = dtype, device=device)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        next_state, reward, done, info = env.step(action.item())
        reward = torch.tensor([reward], dtype = dtype, device=device)
        next_state = torch.tensor([next_state], dtype = dtype, device=device)
        # Observe new state
        if not done:
            next_state = next_state
        else:
            next_state = None

        
        # Move to the next state
        state = next_state

        
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network

env.render()
env.close()
plt.ioff()
plt.show()
