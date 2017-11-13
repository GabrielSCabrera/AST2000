import numpy as np
from ast2000solarsystem_27_v5 import AST2000SolarSystem as A2000
from scipy import interpolate as INTER
import matplotlib.pyplot as plt
from numba import jit

system= A2000(45355)

mu= 36.7
T0= 418.
m_H= 1.67e-27
solar_mass= 1.9891e30
m_planet= 2.05903678173e-06*solar_mass
k= 1.38064852e-23
G= 6.67408e-11
planet_radius= system.radius[1]*1000
g= G*m_planet/planet_radius**2
rho0= system.rho0[1]
P0= (rho0*k*T0)/(m_H*mu)
gamma= 1.4


def temp(h, h0):
    return T0*(1 - (gamma - 1)*h/(gamma*h0))

def gravity(h, M):
    return G*M/(planet_radius + h)**2

def pressure(h, h0):
    return P0*(1 - h/(3.5*h0))**(3.5)

def pressure_t(h, h0):
    return P0*np.exp(-h/h0)

def rho(h, h0):
    return rho0*(1 - h/(3.5*h0))**(2.5)

def rho_t(h, h0):
    return rho0*np.exp(-h/h0)


@jit(cache= True, nopython= True)
def integrate_rho():
    dh= 0.05
    h= 0
    T= T0
    P= P0
    Rho= np.array(rho0)
    Rho_n= rho0
    H= np.array(0)
    M= m_planet
    i= 0
    while Rho_n > 1e-2:
        g= gravity(h, M)
        h0= k*T/(mu*g*m_H)
        M+= (4./3)*Rho_n
        if T > T0/2.:
            T= temp(h, h0)
            P= pressure(h, h0)
            Rho_n= rho(h, h0)
        elif T <= T0/2.:
            T= T0/2.
            P= pressure_t(h, h0)
            Rho_n= rho_t(h, h0)
        i+= 1
        h+= dh
        Rho= np.vstack((Rho, Rho_n))
        H= np.vstack((H, i*dh))
    return H[:,0], Rho[:,0]

h, rho= integrate_rho()
np.save('height.npy', h)
np.save('density.npy', rho)
rho_inter= INTER.interp1d(h, rho, bounds_error= False, fill_value= 'extrapolate')

plt.plot(rho_inter(h), h)
plt.show()
