#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 19:00:46 2018

@author: acis
"""

import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import ClassFile

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float




num_epochs = 500
batch_size = 64
learning_rate = 1e-3


def add_noise(data):
    noise = -0.5+torch.rand(4).to(device)
    noisy_data = data + noise
    return noisy_data




env = gym.make('Qube-v0')

env.reset()

episode_durations = []       


model = ClassFile.autoencoder_pend().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

def select_action(state):
    global steps_done

    random_action = torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)
    return random_action

for epoch in range(num_epochs):
    
    state = env.reset()
    env.render()
    state = torch.tensor([state], dtype = dtype, device=device)
    for t in count():
        # Select and perform an action
        action= select_action(state)
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.tensor([next_state], dtype = dtype, device=device)
        
        #hidden = torch.tensor([hidden], dtype = dtype, device=device)
        # Observe new state
        if not done:
            next_state = next_state
            data = add_noise(next_state)
        else:
            break
            next_state = None      
        # Move to the next state

        
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        MSE_loss = nn.MSELoss()(output, data)
        print(loss)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
