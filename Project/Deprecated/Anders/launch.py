# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 12:58:15 2017

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

    return pos, vel, sumpart, f/n, f_esc/n

def velRock(f, fuel, n, pps, boxes):
    #Function for calculating velocity of rocket.
    #f = force, fuel = weight of fuel n = number of timesteps, boxes = number of boxes
    #pps = particles pr second,
    from ast2000solarsystem import AST2000SolarSystem

    seed = 28631
    system = AST2000SolarSystem(seed)

    MS = 1.980*10**30                       #Mass of earth sun in kg
    G = 6.67*10**-11                        #Gravitational constant
    HPmass = system.mass[0]*MS              #Mass of Home Planet in kg
    r = system.radius[0]*1000               #Radius of Home Planet in m
    v = 0
    dt = 1200./n
    v_esc = np.sqrt(2*G*HPmass/r)
    counter = 0

    while v < v_esc:
        M = 1000 + fuel
        seconds = 0
        r = system.radius[0]*1000
        fuel_left = fuel
        for i in xrange(n):
            dv = -(G*HPmass/(r**2))
            a = boxes*f/M
            M -= dt*boxes*pps*(3.35*10**-27)
            fuel_left -= dt*boxes*pps*(3.35*10**-27)
            if fuel_left <= 0:
                break
            v = v + a*dt + dv*dt
            if v <= 0:
                v = 0
            r += v*dt
            v_esc = np.sqrt(2*G*HPmass/r)
            seconds += 1
            if v > v_esc:
                break
        if fuel_left <= 0:
            fuel = fuel*1.001
        counter += 1
        if counter > 5000:
            print "Simulation will never reach Escape Velocity"
            break
        boxes = boxes*1.001

    return v, r, boxes, fuel, (fuel - fuel_left)

L = 10**-6              #Size of box
Temp = 10000.           #Temperature in Kelvin
N = 10**5               #Number of particles
n = 1000                #Number of timesteps
t = 10**-9              #Time to simulate
k = 1.38064852*10**-23  #Boltzmann constant
M = 1000                #Weight of rocket
T = 20*60               #20 minutes
V_esc = 17367.0539787   #Escape velocity of home planet
m = 3.35*10**-27        #Mass of hydrogen molecule
hs = L/2                #Size of hole

#pos, vel, sumpart, f, f_esc = box(t, Temp, L, hs, N, n)

sumpart = 77129; f_esc = 4.34605960845*10**-9

#A = L**2; V = L**3
#P1 = N*k*T/V; P2 = f/(n*A)

v1 = T*f_esc/M

intboxes = V_esc/v1                #Initial number of boxes needed to reach escape velocity
pps = sumpart/t                 #Initial particles through hole pr second
intfuel = intpps*T*m*intboxes      #Initial fuel needed for escape velocity

v, r, boxes, fuel, fuel_left = velRock(f_esc, intfuel, 1200, pps, intboxes)
incbox = boxes - intboxes

#print "Escape velocity: %5.2f m/s    Increased boxes: %e   Fuel used: %5.2f kg" %(v, incbox, fuel)


def trajectory(f, boxes, pps, M, x):
    from ast2000solarsystem import AST2000SolarSystem

    seed = 28631
    system = AST2000SolarSystem(seed)

    AU = 149597871                                          #AU in km

    HPpos_x0 = system.x0[0]                                 #Home planet pos in x-direction
    HPpos_y0 = system.y0[0]                                 #Home planet pos in y-direction
    HPvel_vx0 = system.vx0[0]                               #Planet velocity in y-direction
    HPvel_vy0 = system.vy0[0]                               #Planet velocity in y-direction
    HPperiod = system.period[0]*24*60*60                    #Rotation period in seconds
    HPrad_AU = system.radius[0]/AU                          #Home planet radius in AU
    HPspeed = 2*np.pi*system.radius[0]*1000/HPperiod        #Rotational speed in m/s
    x = init_sat_pos = (system.x0[0] + HPrad_AU, system.y0[0])

    _, _, _, _, f = box(t, T, L, hs, N, n)

    for i in range(590):


    system.engine_settings(f, boxes, pps, fuel, T_launch, init_sat_pos, 0)

    pos_after_launch = (cx, cy)

    system.mass_needed_launch(pos_after_launch, test=True)

trajectory(f_esc, boxes, pps, fuel, r)

end = time.clock()
print "Runtime for program: %2.1f s" %(end - start)
