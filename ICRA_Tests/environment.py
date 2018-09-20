#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:46:37 2018

@author: kashishg & bemran
inference piece

"""
import gym_sim_to_real
import gym
import tcpip as com
import numpy as np
class Environment():
    AVAIL_TORQUE = [0.0, +1.0, -1.0]
    def __init__(self, sys = "sim", local_ip = "localhost", local_port = 18001):
        self.init = True
        if sys == "sim":
            self.sim = True
            print("run test as simulation >>>>>>>>>>>>")
        elif sys == "real":
            self.sim = False
            print("run test on qube >>>>>>>>>>>>>>>>>>")
        else:
            raise Exception("Error: please select a crroect system")
        if self.sim:            
            gym.logger.set_level(40)
            self.env = gym.make('QubeMotorAngle-v0')
            self.env.reset()
        else:
            self.tcpip = com.TCPIP(local_ip,local_port,"",0, False)
            self.tcpip.init_server()
            self.tcpip.server_listen()
            
    def reset(self):
        if self.sim:
            self.state = self.env.reset()
        else:
            input("please balance the system and press enter ....")
            self.state = self.read(0)
            self.tcpip.FlushListen()
        return self.state

    def render(self):
        if self.sim:
            self.env.render()
    
    def step(self, action):
        if self.sim:           
            self.state, self.reward, self.done, self.info = self.env.step(action)
        else:
            a = self.AVAIL_TORQUE[action]
            self.state = self.read(a)
            self.done = self.terminal(self.state)
            self.reward = 1.0 if not self.done else 0.0
            self.info = []
            #print(self.state)
        print (self.state)
        return self.state, self.reward, self.done, self.info

    def read(self, a = 0):
        data, dt = self.tcpip.recieve(1024*10, a)
        #print("data: {}".format(data))
        last = len(data) - list(reversed(data)).index(4545.0) -1           
        d = data[last-4:last]
        s = d[0:2]
        print("data: {}".format(s))
        return s
        
    def terminal(self,s):
        MAX_ANG = np.pi*2
        s = self.state
        result = bool(s[0] > MAX_ANG or s[0] < -MAX_ANG)
        if result:
            print("Terminated ang={} vel={}".format(s[0], s[1]))                        
        return result 
    
    def close(self):
        if self.sim:
            self.env.close()
        else:
            self.tcpip.close()
    