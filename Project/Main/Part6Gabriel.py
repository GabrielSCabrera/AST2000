from ast2000solarsystem_27_v5 import AST2000SolarSystem as A2000
import matplotlib.pyplot as plt
import functions as fx
import classes as cl
import numpy as np
import numba as nb
from numba import jit
from numpy import linalg as LA
seed= 45355
system= A2000(seed)

#INPUT
r0= 0.003
r1= 0.0001
theta0= 0.75*np.pi
v0= np.array([1.,1.])

#AUTOMATIC
P= cl.Planet('jevelan')
R= cl.Rocket()
theta1= (theta0 + np.pi) % 2.*np.pi
x0= np.array([r0*np.cos(theta0), r0*np.sin(theta0)])
x1= np.array([r1*np.cos(theta1), r1*np.sin(theta1)])
orbit_data= R.get_orbit_data(x0, x1, start_at_apoapsis= True, k= R.G*P.mass)
print orbit_data['v_a']
dv= orbit_data['v_a'] - v0
x2= np.array([r1*np.cos(theta0), r1*np.sin(theta0)])
orbit_data2= R.get_orbit_data(x1, x2, start_at_apoapsis= True, k= R.G*P.mass)
print orbit_data2['v_a']

@jit(cache= True)
def Fd(rho, A, v, Cd= 1):
    '''
        rho, A and Cd should be floats, and v should be an ndarray
    '''
    return -0.5*rho*Cd*A*(LA.norm(v)**2.)*(v/LA.norm(v))

def get_parachute_area(rho, lander, planet):
    '''
        rho should be a float, lander should be a Lander object and planet
        should be a Planet object
    '''
    m= lander.mass
    A0= lander.cs_area
    G= 6.674e-11
    M= planet.mass*1.99e30
    R= planet.radius*1e3
    return (2.*G*M*m)/(9.*R**2*rho)
