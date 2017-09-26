# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:00:02 2017

@author: domin
"""
import numpy as np
from ast2000solarsystem_27_v3 import AST2000SolarSystem

seed = 45355
system = AST2000SolarSystem(seed)

AU = 149597871.                                         #AU in km

HPpos_x0 = system.x0[0]                                 #Home planet pos in x-direction
HPpos_y0 = system.y0[0]                                 #Home planet pos in y-direction
HPvel_vx0 = system.vx0[0]                               #Planet velocity in y-direction in AU/Yrs
HPvel_vy0 = system.vy0[0]                               #Planet velocity in y-direction in AU/Yrs
HPperiod = system.period[0]/365                         #Rotation period in
HPrad_AU = system.radius[0]/AU                          #Home planet radius in AU
HPspeed = 2*np.pi*system.radius[0]/(AU*HPperiod)        #Rotational speed in AU/Yrs
init_sat_pos = (system.x0[0] + HPrad_AU, system.y0[0])

totvel_y = HPvel_vy0 + HPspeed                          #Total velocity in y-direction

x = system.x0[0] + 12657251.9221/(AU*1000)
y = system.y0[0] + totvel_y*(742./(60*60*24*365))

f = 4.34605960845e-09; boxes = 3.88518473438e13 ; pps = 7.7129e13; fuel = 8431.05244654; T_launch = 2000

system.engine_settings(f, boxes, pps, fuel, T_launch, init_sat_pos, 0)

pos_after_launch = (x, y)

#system.mass_needed_launch(pos_after_launch, test=True)

print system.psi[5], system.omega[5]
