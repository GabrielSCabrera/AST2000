import numpy as np
import numba as nb
from numba import jit
import matplotlib.pyplot as plt
import classes as cs
import functions as fx

class Planet_Landing(object):

    def __init__(self, planet, lander, rho0):
        self.planet= planet
        self.lander= lander
        self.period= planet.period*86400.
        self.m= lander.mass
        self.M= planet.mass*1.99e30
        self.G= 6.674e-11
        self.R= planet.radius*1e3
        self.rho0
        self.A_parachute= (2.*self.G*self.M*self.m)/(9.*self.R**2*rho0)

    def atmospheric_density(self, r):
        '''
            --TO BE COMPLETED--
            Returns the atmospheric density in kg/m^3 at an altitude r
        '''
        return None

    def wind_speed(self, r, x):
        v_abs= r/(self.period*2.*np.pi)
        return v_abs*fx.unit_vector(fx.rotate_vector(x, -np.pi/2.))

    def gravitational_force(self, r, x):
        return (-self.G*self.m*self.M*x)/(r**3)

    def drag_force(self, r, x, v, A):
        '''
            Note -- The drag coefficient Cd is set to 1
        '''
        rho= self.atmospheric_density(r)
        v+= self.wind_speed(r, x)
        return -0.5*rho*A*(LA.norm(v)*v)

    def sum_of_forces(self, x, v):
        return self.gravitational_force(r, x) + self.drag_force(r, x, v)

    def land(self, x0, v0, deploy_time):
        x= [x0.copy()]
        r= LA.norm(x)
        v= [v0.copy()]
        dt= 1.
        t= 0.
        deployed_chute= False
        while x > self.radius:
            t+= dt
            if deployed_chute == False and deploy_time is not None and t > deploy_time:
                deployed_chute= True
            if deployed_chute == True:
                Fd= self.drag_force(r, x, v, self.A_parachute)
            if deployed_chute == False:
                Fd= self.drag_force(r, x, v, self.lander.cs_area)
            if Fd > 2.5e5:
                return x, v
            Fg= self.gravitational_force(r, x)
            a= (Fd + Fg)/self.m
            v.append(v[-1] + a*dt)
            x.append(x[-1] + v*dt)
        return x, v

    def find_best_initial_conditions(self):
        pass

P = cs.Planet('jevelan')
print P.temperature
