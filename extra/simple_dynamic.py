#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:29:10 2018

@author: emran
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
class Dynamic:
    def __init__(self, s0, dsdt, dt = 0.01, int_type = "rk4"):
        ns = len(s0)
        self.dt = dt
        self.state = np.asarray(s0)
        self.intType = int_type
        self.dsdt = dsdt
        self.ds = np.zeros(ns)
        
    def step(self, a):
        s = np.append(self.state,a)
        
        if (self.intType == "forward"):
            ns = self.forward(s, 0)
        if (self.intType == "euler"):
            ns = self.euler(s, 0)
        elif (self.intType == "rk4"):
            ns = self.rk4(s, 0)
            
        self.state = ns[:-1]
        return (self.state)
    
    def get_states(self):      
        return (self.state)
    
    def rk4(self, y0, a):
        dt  = self.dt
        dt2 = self.dt/2
        k1 = np.asarray(self.dsdt(y0           , 0))
        k2 = np.asarray(self.dsdt(y0 + dt2 * k1, dt2))
        k3 = np.asarray(self.dsdt(y0 + dt2 * k2, dt2))
        k4 = np.asarray(self.dsdt(y0 + dt  * k3, dt))
        yout = y0 + self.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return yout

    def forward(self, y0, t):
        yout = y0 + self.dt * np.asarray(self.dsdt(y0, self.dt))
        return yout

    def euler(self, y0, t):
        dy = np.append(self.ds,0)
        ndy = np.asarray(self.dsdt(y0, self.dt))
        yout = y0 + self.dt * (ndy + dy) / 2.0;
        
        self.ds = ndy[:-1]
        return yout
    
def dsdt (s, t):   
    x1 = s[0]
    x2 = s[1]
    a = s[2]
    dx = x2
    ddx = -10 * x1 - 4 * x2 + 10 * a
    return [dx, ddx, 0]


if __name__ == "__main__":
    dt = 0.1;
    s0 = [0.0, -10.0]
    sysf = Dynamic(s0, dsdt, dt,"forward")
    sysr = Dynamic(s0, dsdt, dt,"rk4")
    syse = Dynamic(s0, dsdt, dt,"rk4")
    imax = 100;
    time = np.arange(0,imax*dt,dt)

    Xsf = np.zeros([imax,2])
    Xsr = np.zeros([imax,2])
    Xse = np.zeros([imax,2])
    Xso = np.zeros([imax,2])
    Xsf[0] = np.asarray(sysf.get_states())    
    Xsr[0] = np.asarray(sysr.get_states())
    Xse[0] = np.asarray(syse.get_states())
    Xso[0] = np.asarray(s0)
    
    for t in range(1,len(time)):
        a = np.random.randint(-10,10)
        a= 1
        sysf.step(a)
        sysr.step(a)
        syse.step(a)
        Xsf[t] = sysf.get_states()
        Xsr[t] = sysr.get_states()
        Xse[t] = sysr.get_states()
        Xso[t] = odeint(dsdt, list(np.asarray(Xso[t-1],a)), [0, dt])[1,:-1]
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(time, Xsf[:,0],
             time, Xsr[:,0],
             time, Xse[:,0],
             time, Xso[:,0])
    plt.legend(("forward", "rk4", "euler", "odint"))
    plt.subplot(212)
    plt.plot(time, np.sqrt((Xso[:,0]-Xsf[:,0])**2),
             time, np.sqrt((Xso[:,0]-Xsr[:,0])**2),
             time, np.sqrt((Xso[:,0]-Xse[:,0])**2))    
    plt.legend(("forward", "rk4", "euler"))