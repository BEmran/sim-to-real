#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 23:20:31 2018

@author: kashishg
disc
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
import ClassFile

#Graphs seperate from ipython console terminal
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
plt.ion()

#Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        


eps_t = []
rand_t = []

def plot_thresh():
    plt.figure(1)
    plt.clf()
    eps_thresh = torch.tensor(eps_t, dtype=torch.float)
    rand_val = torch.tensor(rand_t, dtype=torch.float)
    plt.xlabel('Episode*Duration')
    plt.ylabel('ThreshValue')
    

    plt.plot(eps_thresh.numpy())
    #plt.plot(rand_val.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

        
def select_action(state):
    global steps_done, last_hidden
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    eps_t.append(eps_threshold)
    #plot_thresh()
    if sample > eps_threshold:
        rand_t.append(eps_threshold)
        with torch.no_grad():
            policy_action, last_hidden = policy_net(state, last_hidden)
            policy_action = policy_action.max(1)[1].view(1, 1)
            return policy_action, last_hidden
    else:
        rand_t.append(0)
        random_action = torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)
        return random_action, last_hidden

mean_dur = 5

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    

    plt.plot(durations_t.numpy())
    
    #Take 100 episode averages and plot them too
    if len(durations_t) >= mean_dur:
        
        means = durations_t.unfold(0, mean_dur, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(mean_dur-1), means))
        plt.plot(means.numpy(), 'r', linewidth=2.5)
    plt.legend(['Episode Duration (Reward)','100 Episode Mean Reward'])  
    
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

        

def optimize_model():
    global steps_done
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    hidden_batch = torch.cat(batch.last_hidden)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    #last_hidden = torch.zeros(BATCH_SIZE, 10, device = device, dtype = dtype)
    
    state_action_values, _ = policy_net(state_batch, hidden_batch)
    state_action_values = state_action_values.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #temp, _ = target_net(non_final_next_states, last_hidden)
    temp, _ = target_net(state_batch, hidden_batch)
    #next_state_values[non_final_mask] = temp.max(1)[0].detach()
    next_state_values = temp.max(1)[0].detach()
    #next_state_values = next_state_values[non_final_mask]
    # Compute the expected Q values
    gamma = GAMMA_END - (GAMMA_START - GAMMA_END) * \
        math.exp(-1. * steps_done / GAMMA_INC)
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #for param in policy_net.parameters():
        #param.grad.data.clamp_(-1, 1)
    optimizer.step()
    


#Hyperparameters
Transition = namedtuple('Transition',
                        ('state', 'last_hidden', 'action', 'next_state', 'reward'))
BATCH_SIZE = 64

GAMMA_END = 0.9
GAMMA_START = 0.5
GAMMA_INC = 200

EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 200

TARGET_UPDATE = 10
num_episodes = 25
hidden_size = 10
last_hidden = torch.zeros(1, hidden_size, device = device, dtype = dtype)

policy_net = ClassFile.RNN_disc().to(device)
target_net = ClassFile.RNN_disc().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

    
env = gym.make('QubeMotorAngle-v0')

env.reset()

episode_durations = []       


for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    env.render()
    state = torch.tensor([state], dtype = dtype, device=device)
    for t in count():
        # Select and perform an action
        action, hidden = select_action(state)
        env.render()
        next_state, reward, done, info = env.step(action.item())
        reward = torch.tensor([reward], dtype = dtype, device=device)
        next_state = torch.tensor([next_state], dtype = dtype, device=device)
        
        
        #hidden = torch.tensor([hidden], dtype = dtype, device=device)
        # Observe new state
        if not done:
            next_state = next_state
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, hidden, action, next_state, reward)
        #print (action.item())
        
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
#plot_thresh()

env.render()
env.close()
plt.ioff()
plt.show()

