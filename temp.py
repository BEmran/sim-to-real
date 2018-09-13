#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 19:23:57 2018

@author: bemran
"""
import gym_sim_to_real
import gym
import numpy as np


gym.logger.set_level(40)
env = gym.make('QubeMotorAngle-v0')
env.reset()

total_episodes = 1
steps_done = 0
episode_durations = []
 
for i_episode in range(total_episodes):
    # Initialize the environment and state
    state = env.reset()
    env.render()
    for t in np.arange(100):
        next_state, reward, done, info = env.step(-1)
        print("t: {}, s{}".format(t,next_state))
        env.render()
        if done:
            break

env.close()

