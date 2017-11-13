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
        self.rho0= planet.rho0
        self.A_parachute= (2.*self.G*self.M*self.m)/(9.*self.R**2*rho0)
        self.initialize_atmospheric_density()

    def initialize_atmospheric_density(self):
        mu= 36.7
        T0= 418.
        m_H= 1.67e-27
        m_planet= self.M
        k= 1.38064852e-23
        G= self.G
        planet_radius= self.R
        g= G*m_planet/planet_radius**2
        rho0= self.rho0
        P0= (rho0*k*T0)/(m_H*mu)
        gamma= 1.4

        @jit(cache= True, nopython=True)
        def integrate_rho():
            dh= 0.01
            h= 0
            T= T0
            P= P0
            Rho= [rho0]
            Rho_n= rho0
            H= [0.]
            M= m_planet
            i= 0
            while Rho_n > 1e-2:
                g= G*M/(planet_radius + h)**2
                h0= k*T/(mu*g*m_H)
                M+= (4./3)*Rho_n
                if T > T0/2.:
                    T= T0*(1 - (gamma - 1)*h/(gamma*h0))
                    P= P0*(1 - h/(3.5*h0))**(3.5)
                    Rho_n= rho0*(1 - h/(3.5*h0))**(2.5)
                elif T <= T0/2.:
                    T= T0/2.
                    P= P0*np.exp(-h/h0)
                    Rho_n= rho0*np.exp(-h/h0)
                i+= 1
                h+= dh
                Rho.append(Rho_n)
                H.append(h)
            return np.array(H), np.array(Rho)

        h, rho= integrate_rho()
        self.rho_inter= INTER.interp1d(h, rho, bounds_error= False, fill_value= 'extrapolate')

    def atmospheric_density(self, r):
        '''
            Returns the atmospheric density in kg/m^3 at an altitude r
        '''
        return self.rho_inter(r)

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
