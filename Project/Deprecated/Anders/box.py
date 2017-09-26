# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:58:10 2017

@author: domin
"""

import numpy as np
#import matplotlib.pyplot as plt
#import mpl_toolkits.mplot3d.axes3d as p3
#import matplotlib.animation as animation
import time
start = time.clock()
#np.set_printoptions(threshold=np.nan)
np.random.seed(666)

def box(t, T, L, hs, N, n):

    #Function for simulating box. t = total time, T = Temperature of gas, L = Size of box, hs = size of hole
    #N = Number of particles, n = number of timesteps.

    m = 3.35*10**-27                #Mass of hydrogen molecule
    k = 1.38064852*10**-23          #Boltzman constant
    sigma = np.sqrt(k*T/m)
    dt = t/n

    pos = np.random.uniform(0,L,(N,3))
    vel = np.random.normal(0,sigma,(N,3))

    sumpart = 0; f = 0; f_esc = 0

    #mean_kinE = (np.sum(vel**2)*m/2)/N

    for i in xrange(n):
        x = np.ma.filled(np.ma.masked_where((pos[:,0] < L), vel[:,0]), fill_value=0)
        y = np.ma.filled(np.ma.masked_where(((0 < pos[:,1]) & ( pos[:,1] < hs)), x), fill_value=0)
        z = np.ma.filled(np.ma.masked_where(((0 < pos[:,2]) & ( pos[:,2] < hs)), y), fill_value=0)
        v_new = np.ma.filled(np.ma.masked_where(((pos > L) | (pos < 0)), vel), fill_value=(vel*-1))
        p_new = np.ma.filled(np.ma.masked_where((z > 0) , pos[:,0]), fill_value=0)
        pos[:,0] = p_new
        pos = pos + v_new*dt
        vel = v_new
        sumpart += np.count_nonzero(z)
        f += (2*np.sum(x)*m/dt)
        f_esc += (2*np.sum(z[z > 0])*m/dt)

    return sumpart/t, f_esc/n

L = 10**-6              #Size of box
Temp = 10000.           #Temperature in Kelvin
N = 10**5               #Number of particles
n = 1000                #Number of timesteps
t = 10**-9              #Time to simulate
hs = L/2                #Size of hole

s, f = box(t, Temp, L, hs, N, n)

print s, f
