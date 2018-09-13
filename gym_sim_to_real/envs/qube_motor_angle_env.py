# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 19:23:57 2018

@author: bemran
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint

__copyright__ = "" #TODO
__credits__ = ["Bara Emran"] #TODO
__license__ = "BSD 3-Clause" #TODO
__author__ = "Bara Emran <bara.emran@gmail.com>" #TODO

class QubeMotorAngleEnv(gym.Env):

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 30
    }

    dt = 0.002

    # Parameters:
    TF_NUMERATOR = 220
    TF_DENOMINATOR = 9.15
 
    MAX_ANG = 4 * np.pi
    MAX_VEL = 100.0
    MAX_VOL = 1

    AVAIL_TORQUE = [0.0, +1.0, -1.0]

    def __init__(self):
        print("Qube Motor Angle")
        self.viewer = None       
        high = np.array([self.MAX_ANG, self.MAX_VEL,])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-high, high=high)
        self.state = None
        self.seed()
	self.v0 = 0
        self.u0 = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        return self._get_ob()
    
    def step(self, a):
        
        u = self.AVAIL_TORQUE[a]
	v = self._poly(u)
        self.state = self.rk4(v)
        terminal = self._terminal()
        reward = self._reward();
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        s = self.state
        return np.array([s[0], s[1]])

    def _poly(self, u):
	v = v0 + u - 0.9998*u0
	self.v0 = v
	self.u0 = u
        return v 
        
    def _terminal(self):
        s = self.state
        result = bool(s[0] > self.MAX_ANG or s[0] < -self.MAX_ANG)
        if result:
            print("Terminated ang={} vel={}".format(s[0], s[1]))                        
        return result 

    def _dsdt(self, s, u, t):
        
        n = self.TF_NUMERATOR
        d = self.TF_DENOMINATOR

        ang, vel = self.state
        dvel = - d * vel + n * u
        
        return (vel, dvel,)

    def rk4(self, a):
        s = self.state
        dt  = self.dt
        dt2 = self.dt/2
        k1 = np.asarray(self._dsdt(s           , a, 0  ))
        k2 = np.asarray(self._dsdt(s + dt2 * k1, a, dt2))
        k3 = np.asarray(self._dsdt(s + dt2 * k2, a, dt2))
        k4 = np.asarray(self._dsdt(s + dt  * k3, a, dt ))
        yout = s + self.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return yout

    def _reward(self):
        s = self.state
        return np.exp(-1.5*((s[0]/0.5)**2))

    def render(self, mode='human'):
    
        s = self.state
        w_size = 8
        
        if self.viewer is None:        
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,250)
            self.viewer.set_bounds(-w_size, w_size, -w_size/2, w_size/2)
            # define colors
            rgb_dred = [0.5, 0,0]
            rgb_gray = [0.5, 0.5, 0.5]
            
            # create transform
            self.transform = rendering.Transform()
            # circle   
            circle = self.viewer.draw_circle(1)
            circle.set_color(*rgb_dred) 
            circle.add_attr(self.transform)     
            self.viewer.add_geom(circle)
            # indicator line
            line = rendering.make_capsule(0.5, .2)
            line.set_color(*rgb_gray) 
            line.add_attr(self.transform)
            self.viewer.add_geom(line)

            
        # rotoate
        self.transform.set_rotation(s[0]) # pointing up is zero

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
