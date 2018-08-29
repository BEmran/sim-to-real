# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 19:09:07 2018
@author: Bara Emran
"""

"""classic Furuta Pendulum task"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import odeint

__copyright__ = "" #TODO
__credits__ = ["Bara Emran"] #TODO
__license__ = "BSD 3-Clause" #TODO
__author__ = "Bara Emran <bara.emran@gmail.com>" #TODO

# SOURCE:
#TODO

class QubeEnv(gym.Env):

    """
    **Definition:**
    - Furuta Pendulum is a 2-link system with two links (an arm and a pendulum)
    with the arm only actuated
    - Intitially, arm joint point arbitrarily while pendulum links point upwards.
    - The goal is to keep the pendulum at upward position (angle ~= PI).
    - Both links can swing freely.
    **STATE:**
    The state consists of the two rotational joint angles and the joint angular
    velocities : [phi theta phiDot1 thetaDot].
        phi   = s[0]  # joint angle of the arm (first link)
        theta = s[1]  # joint angle of the pendulum (second link)
    For the first link, an angle of 0 corresponds to the link pointing at the middle.
    The 0 angle of the second link corresponds to the pendulum pointing downward.
    **ACTIONS:**
    The action is either applying -1, 0 or +1 torque on arm joint.
    **REFERENCE:**
    """
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 15
    }

    dt = 0.01

    # Parameters:
    LINK_MASS_p = 0.024     #: [kg] mass of the pendulum
    LINK_LENGTH_a = 0.085   #: [m] the length of the arm
    LINK_LENGTH_p = 0.129   #: [m] distance from the pendulum center of mass to the pivot
    LINK_MOI_a = 0.0000571  #: moments of inertia of the arm with respect to the axis of rotation
    LINK_MOI_p = 0.0000328  #: moments of inertia of the pendulum with respect to its pivot

    MAX_VEL_1 = 10.0 * np.pi
    MAX_VEL_2 = 10.0 * np.pi

    AVAIL_TORQUE = [-1.0, 0.0, +1.0]

    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self):
        print("Qube servo without motor")
        self.viewer = None
        high = np.array([np.pi, np.pi, self.MAX_VEL_1, self.MAX_VEL_2])
        self.observation_space = spaces.Box(low=-high, high=high)
        self.action_space = spaces.Discrete(3)
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        self.state[1] += np.pi
        return self._get_ob()

    def step(self, a):
        torque = self.AVAIL_TORQUE[a]

        ns = self.rk4(torque)

        ns[0] = self.wrap (ns[0], -np.pi,   np.pi)
        ns[1] = self.wrap (ns[1], 0     , 2*np.pi)
        ns[2] = self.bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = self.bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)

        self.state = ns
        terminal = self._terminal()
        reward = 1.0 if not terminal else 0.0
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        s = self.state
        return np.array([s[0], s[1], s[2], s[3]])

    def _terminal(self):
        s = self.state
        result = bool(s[0] <-2*np.pi/3 or s[0] > 2*np.pi/3 or
                      s[1] <   np.pi/2 or s[1] > 3*np.pi/2)
        if result:
            print ('Terminate: arm angle s[0]=',s[0],'pend. angle s[1]=',s[1])
        return result
    

    def _dsdt(self, s, T, t):

        Mp = self.LINK_MASS_p
        Jp = self.LINK_MOI_p
        Ja = self.LINK_MOI_a
        l = self.LINK_LENGTH_p
        r = self.LINK_LENGTH_a
        g = 9.8

        phi = s[0]  # rotational angle of the arm (first link)
        the = s[1]  # rotational angle of the pendulum (second link)
        dphi = s[2]
        dthe = s[3]
        #eq1 = + ((Jp * sin(2 * the) * dphi**2)/2 + Mp * g * l * sin(the)) * (Jp * sin(the)**2 + Ja) - Mp * l * r * cos(the) * (Mp * l * r * sin(the) * dthe**2 + Jp * dphi * sin(2 * the) * dthe - T)
        #eq2 = + (Mp * l * r * sin(2 * the) * (Jp * cos(the) * dphi**2 + Mp * g * l))/2 - Jp * Mp * l * r * sin(the) * dthe**2 - Jp**2 * dphi * sin(2 * the) * dthe + Jp * Jp * T)

        eq2 = + Jp * Mp * dphi**2 * l * r * cos(the)**2 * sin(the) + Mp**2 * g * l**2 * r * cos(the) * sin(the) - 2 * Jp**2 * dphi * dthe * cos(the) * sin(the) - Jp * Mp * dthe**2 * l * r * sin(the) + Jp * T

        eq1 = + (Ja + Jp * sin(the)**2) * Jp * dphi**2 * cos(the) * sin(the) + (Ja + Jp * sin(the)**2) * Mp * g * l * sin(the) - 2 * Jp * Mp * dphi * dthe * l * r * cos(the)**2 * sin(the)  - Mp**2 * dthe**2 * l**2 * r**2 * cos(the) * sin(the) + Mp * T * l * r * cos(the)

        det = + Ja * Jp + Jp**2 * sin(the)**2 - (Mp * l * r * cos(the))**2


        ddphi = eq2 / det
        ddthe = eq1 / det

        return (dphi, dthe, ddphi, ddthe)

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

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state
        w_size = 8
        if self.viewer is None:
            self.viewer = rendering.Viewer(500,250)
            self.viewer.set_bounds(-w_size, w_size, -w_size/2, w_size/2)

        if s is None: return None

        box_l = 0.5;

        # define center points and angles
        arm_xy = np.array([ 4.0, 0.0])
        pen_xy = np.array([-4.0, 0.0])
        thetas = [s[0]+ np.pi, s[1]+ np.pi] 

        # define colors
        rgb_dred = [0.5, 0,0]
        rgb_lred = [0.9, 0,0]
        rgb_gray = [0.5, 0.5, 0.5]
        rgb_light = [0.9, 0.9, 0.9]

        self.viewer.draw_line(( 0, -w_size), ( 0,  w_size))
        self.viewer.draw_line((arm_xy[0], -w_size), (arm_xy[0],  w_size)).set_color(*rgb_light)
        self.viewer.draw_line((pen_xy[1], -w_size), (pen_xy[1],  w_size)).set_color(*rgb_light)

        l, r, t, b = -0.2, 0.2, 3, 0    # link dimintions
        # ARM

        arm_box = self.viewer.draw_polygon([(-box_l,  box_l), ( box_l,  box_l),
                                            ( box_l, -box_l), (-box_l, -box_l)])
        arm_box.set_color(0, 0, 0)
        arm_bos_tra = rendering.Transform(rotation=0,
                                          translation=(arm_xy[0], arm_xy[1]))
        arm_box.add_attr(arm_bos_tra)

        arm = self.viewer.draw_polygon([(l, t), (r, t), (r, b), (l, b)])
        arm_tra = rendering.Transform(rotation=thetas[0],
                                      translation=(arm_xy[0], arm_xy[1]))
        arm.add_attr(arm_tra)
        arm.set_color(*rgb_gray)

        arm_circ = self.viewer.draw_circle(.2)
        arm_circ.add_attr(arm_tra)
        arm_circ.set_color(*rgb_dred)

        # PENDULUM + box_l + .2
        pen_box = self.viewer.draw_polygon([(-box_l,  box_l), ( box_l,  box_l),
                                            ( box_l, -box_l), (-box_l, -box_l)])
        pen_box_tra = rendering.Transform(rotation=0,
                                          translation=(pen_xy[0], pen_xy[1]))
        pen_box.add_attr(pen_box_tra)

        pen_bar = self.viewer.draw_polygon([(-box_l, -0.1), ( box_l, -0.1),
                                            ( box_l, -0.2), (-box_l, -0.2)])
        pen_bar.add_attr(pen_box_tra)
        pen_bar.set_color(*rgb_lred)

        pen = self.viewer.draw_polygon([(l, t), (r, t),(r, b), (l, b)])
        pen_tra = rendering.Transform(rotation=thetas[1],
                                      translation=(pen_xy[0], pen_xy[1] + box_l + 0.2))
        pen.add_attr(pen_tra)
        pen.set_color(*rgb_dred)

        pen_circ = self.viewer.draw_circle(.2)
        pen_circ.add_attr(pen_tra)
        pen_circ.set_color(*rgb_gray)


        return self.viewer.render()

    def wrap(self, x, m, M):
        """
        :param x: a scalar
        :param m: minimum possible value in range
        :param M: maximum possible value in range
        wraps x around the coordinate system defined by m and M.
        For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
        """
        diff = M - m
        while x > M:
            x = x - diff
        while x < m:
            x = x + diff
        return x

    def bound(self, x, m, M):
        """
        :param x: scalar
        returns m <= x <= M
        """
        # bound x between min (m) and Max (M)
        return min(max(x, m), M)

    def close(self):
        if self.viewer: self.viewer.close()