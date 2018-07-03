# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:33:49 2018
@author: Barae
"""

import gym_sim_to_real
import gym
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    gym.logger.set_level(40)
    env = gym.make('Qube-v0')
    n = 400;
    x = np.zeros([n,4])
    for i_episode in range(1,2):
        observation = env.reset()
        x[0] =  observation
        for t in range(1,n):
            env.render()
            print(observation)
            action = 1;
            observation, reward, done, info = env.step(action)
            x[t] =  observation
            if done==1:
                print("Episode finished after {} timesteps".format(t+1))
                break
            
    print(env.action_space)
    time = np.arange(n)/100
    plt.plot(time[:t], x[:t,0], time[:t], x[:t,1])    
    plt.legend(("phi", "theta"))        
    env.close()