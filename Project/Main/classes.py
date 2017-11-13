from ast2000solarsystem_27_v5 import AST2000SolarSystem as A2000
import functions as fx
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import linalg as LA
from scipy import interpolate as INTER
import sympy.geometry as GEO
import ui_tools as ui
import time, sys, os, math
from multiprocessing import Process
try:
    from numba import jit
    import numba as nb
except ImportError:
    string= "User must install module <numba> to use module <classes.py>"
    fx.error(ImportError, string)

'''
Steinn 52772
Simen 47566
Lars 78826
Ulrik 82275
Gabriel 45355
Anders 28631
Ivar 81402
Torstein 12827'''

class Planet(object):                                   #FIX LEAPFROG ALGORITHM

    def __init__(self, name, dt= None, frames_max= 100000, seed= 45355):

        self.numerical_complete= False
        self.analytical_complete= False
        self.seed= seed
        np.random.seed(self.seed)
        self.sun= Sun(seed= self.seed)
        self.system= A2000(self.seed)
        planet_data, planet_order= fx.get_planet_data(seed= self.seed, return_order= True)

        name= name.lower()
        if name in planet_data:
            for prop, val in planet_data[name].iteritems():
                self.__dict__[prop]= val
                self.index= planet_order.index(name)
                if len(name) > 1:
                    self.name= name[0].upper() + name[1:]
                else:
                    self.name= name.upper()
        else:
            fx.error(NameError, 'Invalid Planet Name')

        #CONSTANTS
        self.G= 4.*np.pi**2.
        self.G_SI= 6.67408e-11
        self.mu= self.G*self.sun.mass

        #CALCULATED DATA
        self.b= self.a*np.sqrt(1 - self.e**2)
        self.mu_hat= (self.mass*self.sun.mass)/(self.mass+self.sun.mass)
        self.T= 2.*np.pi*np.sqrt((self.a**3.)/(self.G*self.sun.mass))
        self.F= self.sun.L/(4.*np.pi*(((1.+self.e)*self.a*1.496e11)**2.))
        self.temperature= (self.F/self.sun.sigma)**0.25
        self.apoapsis= self.a*(1. + self.e)
        self.periapsis= self.a*(1. - self.e)

        #INTEGRATION PARAMETERS
        self.dt= self.T/5e6
        self.frames= min(frames_max, int(self.T/self.dt))
        self.final_dt= float(self.T)/float(self.frames)

    #CALCULATIONS

    def get_analytical_position(self, theta, unit= False):
        p= self.a*(1. - self.e**2.)
        r= p/(1. + self.e*np.cos(theta + np.pi - self.psi))
        ur= np.array([np.cos(theta), np.sin(theta)])
        if unit == False:
            return np.abs(r)*ur
        else:
            return ur

    def get_analytical_velocity(self, theta, unit= False):
        k= self.sun.mass*self.G
        p= self.a*(1. - self.e**2.)
        ur= np.array([np.cos(theta), np.sin(theta)])
        utheta= fx.rotate_vector(ur, np.pi/2.)
        r= p/(1. + self.e*np.cos(theta + np.pi - self.psi))
        vabs= np.sqrt(self.mu*((2./r)-(1./self.a)))
        if unit == False:
            return vabs*utheta
        else:
            return utheta

    def get_analytical_orbit(self):
        theta= np.linspace(0, 2.*np.pi, 10000)
        x= self.get_analytical_position(theta)
        self.analytical= {'theta': theta, 'x': x}
        self.analytical_complete= True

    def get_numerical_orbit(self, T= None, dt= None, frames= None, return_vals= False):
        #USES LEAPFROG INTEGRATION
        if T == None:
            T= self.T
        if dt == None:
            dt= self.dt
        if frames == None:
            frames= self.frames
        x0, y0, vx0, vy0= self.x0, self.y0, self.vx0, self.vy0
        G, sun_mass= self.G, self.sun.mass
        @jit(cache= True)
        def jit_integrate(T,dt,frames,x0,y0,vx0,vy0,G,sun_mass):
            t= np.linspace(0., T, frames)
            x= np.zeros((frames, 2))
            v= x.copy()
            a= x.copy()

            x[0,0], x[0,1]= x0, y0
            v[0,0], v[0,1]= vx0, vy0
            r_magnitude= np.sqrt(np.sum(x[0]**2))
            ur= np.divide(x[0], r_magnitude)
            a[0]= (-(G*sun_mass)/(r_magnitude**2.))*ur
            t_now= dt
            x_now= x[0].copy()
            v_now= v[0].copy()# + 0.5*get_acceleration(x_now)*dt
            a_now= a[0].copy()
            save_index= 1

            while t_now <= T + dt:
                r_magnitude= np.sqrt(np.sum(x_now**2.))
                ur= np.divide(x_now, r_magnitude)
                a_now= (-(G*sun_mass)/(r_magnitude**2.))*ur
                v_now += a_now*dt
                x_now += v_now*dt

                if t_now >= t[save_index]:
                    x[save_index]= x_now
                    v[save_index]= v_now
                    a[save_index]= a_now
                    save_index += 1
                t_now += dt
            return t, x, v, a

        t, x, v, a= jit_integrate(T, dt, frames, x0, y0, vx0, vy0, G, sun_mass)

        if return_vals == False:
            self.numerical= {'t': t, 'x': x, 'v': v, 'a': a}
            self.numerical_complete= True
            self.interpolated= INTER.interp1d(t, x, axis= 0, bounds_error= False,
            fill_value= 'extrapolate')
            self.velocity_from_time= INTER.interp1d(t, v, axis= 0)
            angles= np.arctan2(x[:,1], x[:,0])
            angles[np.where(angles == np.min(angles))[0][0]]= 0.
            angles[np.where(angles == np.max(angles))[0][0]]= 2*np.pi
            angles[np.where(angles < 0)] += 2*np.pi
            self.time_from_angle= INTER.interp1d(angles, t, axis= 0,
            bounds_error= False, fill_value= 2*np.pi)
            self.angle_from_time= INTER.interp1d(t, angles, axis= 0)
        else:
            return {'t': t, 'x': x, 'v': v, 'a': a}

    def get_2_body_numerical_orbit(self, T= None, dt= None, frames= None):
        if T == None:
            T= self.T
        if dt == None:
            dt= self.dt
        if frames == None:
            frames= self.frames
        x0, y0, vx0, vy0= self.x0, self.y0, self.vx0, self.vy0
        G, sun_mass, mass= self.G, self.sun.mass, self.mass
        @jit(cache= True)
        def jit_integrate(T,dt,frames,x0,y0,vx0,vy0,G,sun_mass,mass):
            t= np.linspace(0., T, frames)
            x= np.zeros((frames, 2))
            v= x.copy()
            a= x.copy()

            x[0,0], x[0,1]= x0, y0
            v[0,0], v[0,1]= vx0, vy0
            r_magnitude= np.sqrt(np.sum(x[0]**2.))
            ur= np.divide(x[0], r_magnitude)
            a[0]= (((G*mass)/(r_magnitude**2.))-((G*sun_mass)/(r_magnitude**2.)))*ur
            t_now= dt
            x_now= x[0].copy()
            v_now= v[0].copy()# + 0.5*get_acceleration(x_now)*dt
            a_now= a[0].copy()
            save_index= 1

            while t_now <= T + dt:
                r_magnitude= np.sqrt(np.sum(x_now**2.))
                ur= np.divide(x_now, r_magnitude)
                a_now= (((G*mass)/(r_magnitude**2.))-((G*sun_mass)/(r_magnitude**2.)))*ur
                v_now += a_now*dt
                x_now += v_now*dt
                if t_now >= t[save_index]:
                    x[save_index]= x_now
                    v[save_index]= v_now
                    a[save_index]= a_now
                    save_index += 1
                t_now += dt
            return t, x, v, a

        t, x, v, a= jit_integrate(T,dt,frames,x0,y0,vx0,vy0,G,sun_mass,mass)
        return {'t': t, 'x': x, 'v': v, 'a': a}

    def get_area(self, t0= 0, dt= None):
        self._check_numerical()
        t0= t0%self.T

        con1= t0 - 0.5*self.final_dt < self.numerical['t']
        con2= t0 + 0.5*self.final_dt > self.numerical['t']
        t0_index= np.where(con1 & con2)[0][0]

        if dt == None:
            dt= self.T-self.final_dt

        t1= t0 + dt
        if dt > self.T:
            fx.error(ValueError, 'Cannot calculate area over a time larger than the orbital period')
        elif dt < self.final_dt:
            return 0
        elif t1 > self.T:
            dt2= t1 - self.T
            t1= self.T
            A2= self.get_area(t0= 0, dt= dt2)
        else:
            A2= 0

        con3= t1 - 0.5*self.final_dt < self.numerical['t']
        con4= t1 + 0.5*self.final_dt > self.numerical['t']
        t1_index= np.where(con3 & con4)[0][0]

        x_slice= self.numerical['x'][t0_index:t1_index]
        r= LA.norm(x_slice[:-1], axis= 1)
        arc_lengths= LA.norm(np.diff(x_slice, axis= 0), axis= 1)

        A1= np.sum(np.multiply(r, arc_lengths))
        return A1 + A2

    def get_arc_length(self, t0= 0, dt= None):
        self._check_numerical()
        t0= t0%self.T

        con1= t0 - 0.5*self.final_dt < self.numerical['t']
        con2= t0 + 0.5*self.final_dt > self.numerical['t']
        t0_index= np.where(con1 & con2)[0][0]

        if dt == None:
            dt= self.T-self.final_dt

        t1= t0 + dt
        if dt > self.T:
            fx.error(ValueError, 'Cannot calculate area over a time larger than the orbital period')
        elif dt < self.final_dt:
            return 0
        elif t1 > self.T:
            dt2= t1 - self.T
            t1= self.T
            L2= self.get_arc_length(t0= 0, dt= dt2)
        else:
            L2= 0

        con3= t1 - 0.5*self.final_dt < self.numerical['t']
        con4= t1 + 0.5*self.final_dt > self.numerical['t']
        t1_index= np.where(con3 & con4)[0][0]

        x_slice= self.numerical['x'][t0_index:t1_index]
        arc_lengths= LA.norm(np.diff(x_slice, axis= 0), axis= 1)

        L1= np.sum(arc_lengths)
        return L1 + L2

    def get_mean_velocity(self, t0= 0, dt= None, raw= False):
        self._check_numerical()
        t0= t0%self.T

        con1= t0 - 0.5*self.final_dt < self.numerical['t']
        con2= t0 + 0.5*self.final_dt > self.numerical['t']
        t0_index= np.where(con1 & con2)[0][0]

        if dt == None:
            dt= self.T-self.final_dt

        t1= t0 + dt
        if dt > self.T:
            fx.error(ValueError, 'Cannot calculate area over a time larger than the orbital period')
        elif dt < self.final_dt:
            return 0
        elif t1 > self.T:
            dt2= t1 - self.T
            t1= self.T
            v2= self.get_mean_velocity(t0= 0, dt= dt2, raw= True)
        else:
            v2= None

        con3= t1 - 0.5*self.final_dt < self.numerical['t']
        con4= t1 + 0.5*self.final_dt > self.numerical['t']
        t1_index= np.where(con3 & con4)[0][0]

        x_slice= self.numerical['x'][t0_index:t1_index]
        arc_lengths= LA.norm(np.diff(x_slice, axis= 0), axis= 1)

        v1= arc_lengths/self.final_dt
        if raw == True:
            return v1
        else:
            if v2 == None:
                return np.mean(v1)
            else:
                v3= np.zeros(len(v1)+len(v2))
                v3[:len(v1)], v3[len(v1):]= v1, v2
                return np.mean(v3)

    def get_escape_velocity(self, h= 0.):
        '''Takes and returns SI-units'''
        return np.sqrt(max(2.*self.G_SI*1.99e30*self.mass/(self.radius*1000. + h), 0.))

    def get_gravitational_acceleration(self, h= 0.):
        '''Takes and returns SI-units'''
        return -self.G_SI*1.980e30*self.mass/((self.radius*1000. + h)**2.)

    def get_time_to_angle_from_angle(self, start_angle, end_angle):
        self._check_numerical()
        if end_angle == 'periapsis':
            end_angle= self.psi + np.pi
        elif end_angle == 'apoapsis':
            end_angle= self.psi
        t= self.get_time_from_angle(end_angle)\
        - self.get_time_from_angle(start_angle)%self.T
        if t < 0:
            t= self.T + t
        return t

    def get_time_to_angle_from_time(self, start_time, end_angle):
        self._check_numerical()
        if end_angle == 'periapsis':
            end_angle= self.psi + np.pi
        elif end_angle == 'apoapsis':
            end_angle= self.psi
        t= self.get_time_from_angle(end_angle)\
        - start_time%self.T
        if t < 0:
            t= self.T + t
        return t

    def get_position_from_time(self, t):
        self._check_numerical()
        t= np.mod(t, self.T)
        return self.interpolated(t)

    def get_position_from_angle(self, theta):
        self._check_numerical()
        t= self.get_time_from_angle(theta)
        return self.get_position_from_time(t)

    def get_velocity_from_time(self, t):
        self._check_numerical()
        t= np.mod(t, self.T)
        if abs(t) < 1e-14 or abs(t - self.T) < 1e-14:
            return np.array([self.vx0, self.vy0])
        else:
            return self.velocity_from_time(t)

    def get_angle_from_time(self, t):
        self._check_numerical()
        t= np.mod(t, self.T)
        return self.angle_from_time(t)

    def get_time_from_angle(self, theta):
        self._check_numerical()
        multiplier= np.floor_divide(theta, 2*np.pi)
        angle= np.mod(theta, 2*np.pi)
        return self.time_from_angle(angle) + np.multiply(self.T, multiplier)

    def get_r_from_time(self, t):
        self._check_numerical()
        return LA.norm(self.get_position_from_time(t))

    def get_r_from_angle(self, theta):
        self._check_numerical()
        return LA.norm(self.get_position_from_time(self.get_time_from_angle(theta)))

    #TESTS

    def _check_numerical(self):
        if self.numerical_complete == False:
            self.get_numerical_orbit()

    def _check_analytical(self):
        if self.analytical_complete == False:
            self.get_analytical_orbit()

    #DATA VISUALIZATION

    def plot(self, analytical= True, numerical= False, axes= True):
        legend= ['Sun', 'Planet %s'%(self.name)]
        sun= plt.Circle(xy= (0.,0.), radius= self.sun.radius*6.68459e-9,
        color= 'y')
        planet= plt.Circle(xy= (self.x0,self.y0), radius= self.radius*6.68459e-9,
        color= 'b')
        fig, ax= plt.subplots()
        ax.set(aspect=1)
        ax.add_artist(sun)
        ax.add_artist(planet)
        plt.plot(0,0,'oy',ms=1)
        plt.plot(self.x0,self.y0,'xm',ms=5)

        if analytical == True:
            self._check_analytical()
            legend.append('Analytical Orbit')
            x_analytical= self.analytical['x']
            plt.plot(x_analytical[0], x_analytical[1], '-r')

        if numerical == True:
            self._check_numerical()
            legend.append('Numerical Orbit')
            x_numerical= self.numerical['x']
            plt.plot(x_numerical[:,0], x_numerical[:,1], '-b')

        if axes == True:
            x_a= [0., (1.+self.e)*self.a*np.cos(self.psi)]
            y_a= [0., (1.+self.e)*self.a*np.sin(self.psi)]
            x_b= [0., -(1.-self.e)*self.b*np.cos(self.psi)]
            y_b= [0., -(1.-self.e)*self.b*np.sin(self.psi)]
            plt.plot(x_a, y_a, '-g')
            plt.plot(x_b, y_b, '-m')
            legend += ['Semi-Major Axis', 'Semi-Minor Axis']

        plt.title('The Orbit of Planet %s'%(self.name))
        plt.legend(legend, loc= 1)
        plt.xlabel('x in AU')
        plt.ylabel('y in AU')
        plt.show()

    def plot_2_body(self, T= None):
        data= self.get_2_body_numerical_orbit()
        t= data['t']
        x, y= data['x'][:,0], data['x'][:,1]

        legend= ['Sun']
        sun= plt.Circle(xy= (0.,0.), radius= self.sun.radius*6.68459e-9,
        color= 'y')
        fig, ax= plt.subplots()
        ax.set(aspect=1)
        ax.add_artist(sun)
        plt.plot(0,0,'oy',ms=1)
        plt.plot(x, y)
        plt.title('The Orbit of Planet %s'%(self.name))
        plt.legend(legend)
        plt.xlabel('x in AU')
        plt.ylabel('y in AU')
        plt.show()

    def plot_velocity_curve(self, peculiar_velocity= (0,0), i= np.pi/2.,
    two_body= False, T= None, noise= True):
        if two_body == False:
            self._check_numerical()
            t= self.numerical['t']
            v= self.numerical['v']
        else:
            data= self.get_2_body_numerical_orbit(T= T)
            t= data['t']
            v= data['v']
        if not isinstance(peculiar_velocity, np.ndarray):
            peculiar_velocity= np.array(peculiar_velocity)
        v_max= np.max(LA.norm(v))*np.sin(i)
        v= v_max*np.cos((2*np.pi*t)/self.T) + LA.norm(peculiar_velocity)

        if noise == True:
            noisiness= np.random.normal(loc= 0.0, scale= v_max/5., size= len(t))
            v += noisiness

        plt.plot(t,v)
        plt.title('Velocity Curve of a System with an Inclination of %.2g rads\
        \n and a Peculiar Velocity of (%.2g,%.2g) AU/yr'%(i, peculiar_velocity[0],
        peculiar_velocity[1]))
        plt.xlabel('Observed Velocity (AU/Yr)')
        plt.ylabel('Time (Yr)')
        plt.axis([t[0], t[-1], min(v) - 0.1*abs(max(v) - min(v)),
        max(v) + 0.1*abs(max(v) - min(v))])
        plt.show()

    def plot_light_curve(self, T= None, steps= 1e3, noise= True):

        sun_area= np.pi*(self.sun.radius**2.)
        planet_area= np.pi*(self.radius**2.)
        max_flux= sun_area
        min_flux= max_flux - planet_area
        max_flux /= sun_area
        min_flux /= sun_area
        d_flux= max_flux - min_flux
        v= LA.norm(np.array([self.vx0, self.vy0]))
        cross_time= 2.*6.68459e-9*self.radius/v
        min_time= 2.*6.68459e-9*self.sun.radius/v - cross_time
        t_tot= 2.*cross_time + 3.*min_time
        dt= t_tot/steps
        ct_frames= cross_time/dt
        mt_frames= min_time/dt

        ct_x1= np.linspace(min_flux, max_flux, ct_frames)
        ct_x0= np.copy(ct_x1)[::-1]
        mt_x= min_flux*np.ones(int(mt_frames))
        x0= np.ones_like(mt_x)

        new= np.concatenate((x0, ct_x0, mt_x, ct_x1, x0))

        if noise == True:
            noisiness= np.random.normal(loc= 0.0, scale= 0.2, size= len(new))
            new += noisiness

        t= 8760.*np.linspace(0., t_tot, len(new))
        plt.plot(t, new)
        plt.title('Light Curve of Planet %s Eclipsing its Sun'%(self.name))
        plt.xlabel('Time in Hours')
        plt.ylabel('Relative Flux')
        plt.axis([t[0], t[-1], min(new) - 0.1*abs(max(new) - min(new)),
        max(new) + 0.1*abs(max(new) - min(new))])
        plt.show()

    #MISC FUNCTIONS

    def convert_AU(self, val, convert_to= 'm'):
        if convert_to == 'm':
            return 149597870700.*val
        elif convert_to == 'km':
            return 149597870.700*val
        elif convert_to == 'earth radii':
            return val/4.25875e-5
        elif convert_to == 'solar radii':
            return val/215.
        else:
            fx.error(KeyError, 'Invalid Conversion Unit <%s>'%(convert_to))

    def convert_AU_per_year(self, val, convert_to= 'm/s'):
        if convert_to == 'm/s':
            return val*4743.717360825723
        elif convert_to == 'km/h':
            return val*17066.0582
        elif convert_to == 'km/s':
            return val*4.74372
        else:
            fx.error(KeyError, 'Invalid Conversion Unit <%s>'%(convert_to))

    def convert_year(self, val, convert_to= 's'):
        if convert_to == 's':
            return val*3.154e+7
        elif convert_to == 'm':
            return val*525600.
        elif convert_to == 'h':
            return val*8760.
        elif convert_to == 'd':
            return val*365.2422
        else:
            fx.error(KeyError, 'Invalid Conversion Unit <%s>'%(convert_to))

    def convert_solar_masses(self, val, convert_to= 'kg'):
        if convert_to == 'kg':
            return val*1.99e30
        elif convert_to == 'earth masses':
            return val*332946.
        else:
            fx.error(KeyError, 'Invalid Conversion Unit <%s>'%(convert_to))

    #CLASS OPERATORS

    def __str__(self):
        properties= ['a', 'e', 'radius', 'omega', 'psi', 'mass', 'period', 'x0', 'y0',
        'vx0', 'vy0', 'rho0']
        string= 'Seed: %d\nPlanet Name: %s\nPlanet Index: %d'%(self.seed, self.name, self.index)
        string += '\n\nOrbital Parameters:'
        string += '\n\tSemi-Major Axis: %gAU\n\tSemi-Minor Axis: %gAU'%(self.a, self.b)
        string += '\n\tEccentricity: %g\n\tAngle of Semi-Major Axis: %grad'%(self.e, self.psi)
        string += '\n\tStarting Angle of Orbit: %grad'%(self.omega)
        string += '\n\tOrbital Period: %gYr'%(self.T)
        string += '\n\tStarting Coordinates: (%g, %g) AU'%(self.x0, self.y0)
        string += '\n\tStarting Velocity: (%g, %g) AU/Yr'%(self.vx0, self.vy0)
        string += ', (%g,%g)km/s'%(self.convert_AU_per_year(self.vx0, 'km/s'),
        self.convert_AU_per_year(self.vy0, 'km/s'))
        string += '\n\nPlanet Properties:'
        string += '\n\tRadius: %gkm, %g Earth Radii'\
        %(self.radius, self.radius/6371.)
        string += '\n\tRotational Period: %g Days, %g Hours'%(self.period, self.period*24.)
        string += '\n\tMass: %g Solar Masses, %g Earth Masses, %gkg'\
        %(self.mass, self.convert_solar_masses(self.mass, 'earth masses'),
        self.convert_solar_masses(self.mass))
        string += '\n\tAtmospheric Density at Surface: %gkg/m^3'%(self.rho0)
        return string

    def __eq__(self, p):
        if not isinstance(p, Planet):
            return False
        elif self.seed != p.seed or self.name != p.name:
            return False
        else:
            return True

class Solar_System(object):

    def __init__(self, dt= 5e-7, frames_max= 100000, seed= 45355):
        self.numerical_complete= False
        self.analytical_complete= False
        self.seed= seed
        self.sun= Sun(seed= seed)
        planet_data, self.planet_order= fx.get_planet_data(seed= self.seed, return_order= True)
        self.planets= {}
        self.orbits= {}
        self.defaults= (self.planet_order[0], self.planet_order[1])

        self.dt= dt
        self.frames_max= frames_max

        for planet in self.planet_order:
            self.planets[planet]= Planet(name= planet, dt= self.dt,
            frames_max= self.frames_max, seed= self.seed)

        self.number_of_planets= len(self.planet_order)
        self.system= A2000(self.seed)
        self.G= 4.*np.pi**2.

    #CALCULATIONS

    def get_analytical_orbits(self):
        for p in self.planets:
            self.planets[p].get_analytical_orbit()
        self.analytical_complete= True

    def get_numerical_orbits(self):
        print "Calculating Orbits for %d Planets:"\
        %(self.number_of_planets)
        for n,p in enumerate(self.planets):
            t0= time.time()
            sys.stdout.write("[%d/%d]\tPlanet %s"%(n+1,self.number_of_planets,
            self.planets[p].name))
            self.planets[p].get_numerical_orbit()
            self.orbits[p]= self.planets[p].interpolated
            sys.stdout.flush()
            print ", Done - %.2fs"%(time.time()-t0)
        self.numerical_complete= True

    def get_numerical_orbits_custom(self, T= None, dt= None, frames_max= None):
        if dt == None:
            dt= self.dt
        if frames_max == None:
            frames_max= self.frames_max
        if T == 'min' or T == None:
            T= self.get_min(parameter= 'T', only_return_value= True)
        elif T == 'max':
            T= self.get_max(parameter= 'T', only_return_value= True)

        frames= min(frames_max, int(T/dt))

        orbits= {}
        print "Calculating Custom Orbits for %d Planets (Total Loops: %g):"\
        %(self.number_of_planets, self.number_of_planets*(T/dt))
        for n,p in enumerate(self.planet_order):
            t0= time.time()
            sys.stdout.write("[%d/%d]\tPlanet %s"%(n+1,self.number_of_planets,
            self.planets[p].name))
            sys.stdout.flush()
            orbits[p]= self.planets[p].get_numerical_orbit(T= T, dt= dt,
            frames= frames, return_vals= True)
            print ", Done - %.2fs"%(time.time()-t0)

        return orbits

    def get_positions_from_time(self, t):
        x= np.zeros((self.number_of_planets, 2))
        for n, p in enumerate(self.planets.itervalues()):
            x[n]= p.get_position_from_time(t)
        return x

    #DATA VISUALIZATION

    def plot(self, T= None, dt= None, frames_max= None):
        legend= ['Sun']
        sun= plt.Circle(xy= (0.,0.), radius= self.sun.radius*6.68459e-9,
        color= 'y')
        fig, ax= plt.subplots()
        ax.set(aspect=1)
        ax.add_artist(sun)
        plt.plot(0,0,'oy',ms=1)
        if T == None:
            self._check_analytical()
            for p in self.planet_order:
                x_analytical= self.planets[p].analytical['x']
                plt.plot(x_analytical[0], x_analytical[1])
                legend.append(self.planets[p].name)
        else:
            if dt == None:
                dt= self.dt
            if frames_max == None:
                frames_max= self.frames_max
            orbits= self.get_numerical_orbits_custom(T= T, dt= dt,
            frames_max= frames_max)
            for p in self.planet_order:
                x_numerical= orbits[p]['x']
                plt.plot(x_numerical[:,0], x_numerical[:,1])
                legend.append(self.planets[p].name)
        for p,d in self.planets.iteritems():
            planet= plt.Circle(xy= (d.x0,d.y0), radius= d.radius*6.68459e-9)
            ax.add_artist(planet)
        plt.title('The Orbits of Solar System %d'%(self.seed))
        plt.legend(legend)
        plt.xlabel('x in AU')
        plt.ylabel('y in AU')
        plt.show()

    #TESTS

    def _check_numerical(self):
        if self.numerical_complete == False:
            self.get_numerical_orbits()

    def _check_analytical(self):
        if self.analytical_complete == False:
            self.get_analytical_orbits()

    #DATA EXTRACTION

    def __str__(self):
        string=  'Seed: %d, Number of Planets: %d'\
        %(self.seed, self.number_of_planets)
        return string

    def __call__(self, parameter, array= False):
        if array == False:
            op= {}
            for n,p in self.planets.iteritems():
                op[n]= p.__dict__[parameter]
        else:
            op= np.zeros(self.number_of_planets)
            for n,p in enumerate(self.planets.itervalues()):
                op[n]= p.__dict__[parameter]
        return op

    def get_max(self, parameter, only_return_value= False):
        op= self.__call__(parameter)
        highest_value= max(op.values())
        if only_return_value == True:
            return highest_value
        all_max= {}
        for n,d in op.iteritems():
            if d == highest_value:
                all_max[n]= d
        return all_max

    def get_min(self, parameter, only_return_value= False):
        op= self.__call__(parameter)
        lowest_value= min(op.values())
        if only_return_value == True:
            return lowest_value
        all_min= {}
        for n,d in op.iteritems():
            if d == lowest_value:
                all_min[n]= d
        return all_min

    def get_ordered_list(self, names= False):
        ordered= []
        for p in self.planet_order:
            if names == True:
                ordered.append((self.planets[p].name,self.planets[p]))
            else:
                ordered.append(self.planets[p])
        return ordered

    def save_XML(self, T= None, dt= None, frames_max= None, save= True):
        if dt == None:
            dt= self.dt
        if frames_max == None:
            frames_max= self.frames_max
        if T == None:
            p_0= self.planet_order[0]
            T= 21.*self.planets[p_0].T

        numerical_orbit_data= self.get_numerical_orbits_custom(T= T, dt= dt,
        frames_max= frames_max)
        x= np.zeros([2, self.number_of_planets,
        len(numerical_orbit_data[self.planet_order[0]]['x'][:,0])])

        for n, name in enumerate(self.planet_order):
            t= numerical_orbit_data[name]['t']
            x[0,n]= numerical_orbit_data[name]['x'][:,0]
            x[1,n]= numerical_orbit_data[name]['x'][:,1]
        if save == True:
            self.system.orbit_xml(x,t)
        return x, t

    def confirm_planet_positions(self):
        '''For a dt= 5e-7, takes approximately 54s per planet, or 7.2 mins
        With @jit, takes approximately 3.6s per planet, or 30s,
        The biggest relative deviation was detected at planet 7,
        which drifted 0.3873 percent from its actual position '''
        x, t= self.save_XML(save= True)
        frames= x.shape[2]
        self.system.check_planet_positions(x, t[-1], frames/t[-1])

class Sun(object):

    def __init__(self, seed= 45355):
        self.seed= seed
        data= fx.get_sun_data(seed= self.seed)
        self.mass= data['mass']
        self.radius= data['radius']
        self.temperature= data['temperature']
        self.sigma= 5.6703e-8
        self.L= self.sigma*(self.temperature**4.)*(4.*np.pi*((self.radius*1e3)**2.))

    def __str__(self):
        string=  'Sun Data\n'
        string += 'Seed: %d\n'%(self.seed)
        string += 'Mass: %g Solar Masses or %g kg'%(self.mass, self.mass*1.99e30)
        string += '\nRadius: %g Solar Radii or %g km'%(self.radius/6.957e5, self.radius)
        string += '\nTemperature: %g K, Luminosity: %g W'%(self.temperature, self.L)
        return string

class Gas_Box(object):

    def __init__(self, temperature= 1e4, time= 1e-9, steps= 1e3, L= 1e-6,
    nozzle= None, number_of_particles= 1e5, particle_mass= 3.3474472e-27,
    seed= 45355):

        #CONSTANTS
        self.k= 1.38064852e-23                            #Boltzmann's Constant

        #PHYSICAL VARIABLES
        self.T= float(temperature)                       #Temperature in Kelvin
        self.L= float(L)                                  #Box Length in meters
        self.N= int(number_of_particles)                   #Number of particles
        self.m= particle_mass                #Mass of individual particle in kg

        if nozzle == None:
            nozzle= self.L/2.

        self.nozzle= nozzle                              #Size of Rocket Nozzle

        #SIMULATION VARIABLES
        self.time= float(time)                   #Simulation Runtime in Seconds
        self.steps= int(steps)             #Number of Steps Taken in Simulation
        self.dt= self.time/self.steps        #Simulation Step Length in Seconds
        self.seed= seed
        np.random.seed(self.seed)
        self.particles_per_second, self.force_per_second= self.burn()
        self.box_mass= self.particles_per_second*self.m

    def burn(self):
        sigma= np.sqrt(self.k*self.T/self.m)
        x= np.random.uniform(low= 0., high= self.L, size= (self.N, 3))
        v= np.random.normal(loc= 0.0, scale= sigma, size= (self.N, 3))
        exiting= 0.
        low_bound= 0.25*self.L
        high_bound= 0.75*self.L
        f= 0.
        i= 0
        for i in range(self.steps):
            x += v*self.dt
            v_exiting= np.abs(v[:,2])

            collision_points= np.logical_or(np.less(x, 0), np.greater(x, self.L))
            x_exit_points= np.logical_and(np.greater(x[:,0], low_bound),
            np.less_equal(x[:,0], high_bound))
            y_exit_points= np.logical_and(np.greater(x[:,1], low_bound),
            np.less_equal(x[:,1], high_bound))

            exit_points= np.logical_and(x_exit_points, y_exit_points)
            exit_points= np.logical_and(np.less(x[:,2], 0), exit_points)
            exit_indices= np.where(exit_points == True)

            collisions_indices= np.where(collision_points == True)
            exiting += len(exit_indices[0])
            sign_matrix= np.ones_like(x)
            sign_matrix[collisions_indices]= -1.
            sign_matrix[:,2][exit_indices]= 1.

            f += (2.*np.sum(v_exiting[exit_indices])*self.m/self.dt)
            x[:,2][exit_indices]= 0.99*self.L
            v= np.multiply(v, sign_matrix)

        return exiting/self.time, f/self.steps

class Rocket(object):

    def __init__(self, T= 1200., steps= 1e5, accuracy= 1e-4, planet= None,
    target= None, gas_box= None, payload_mass= 1.1e3, seed= 45355,
    final_height= 9e7):
        self.seed= seed
        self.launch_window_calculated= False
        self.gas_box_calculated= False
        self.chambers_calculated= False
        self.initial_conditions_calculated= False
        self.liftoff_calculated= False
        self.transfer_calculated= False
        self.circularization_calculated= False

        self.final_height= final_height
        self.gas_box= gas_box
        self.T= float(T)
        self.steps= int(steps)
        self.accuracy= float(accuracy)
        self.payload_mass= float(payload_mass)

        self.solar_system= Solar_System(seed= self.seed)
        self.G= self.solar_system.G

        if planet is None:
            planet_name= self.solar_system.defaults[0]
        else:
            planet_name= planet

        self.planet= self.solar_system.planets[planet_name.lower()]

        if target is None:
            target_name= self.solar_system.defaults[1]
        else:
            target_name= target

        self.target= self.solar_system.planets[target_name.lower()]

        if self.planet == self.target:
            fx.error(NameError, 'Must launch rocket from one planet to another')

        self.k= self.planet.sun.mass*self.planet.G

        self.burns= []

    #MAIN FUNCTIONS

    def run(self):
        self.calculate_circularization()
        print '\n', self.__str__()
        #self.plot_liftoff()
        #self.plot_intercept()
        self.engine_settings_array= np.array([self.box_force, self.chambers,
        self.gas_box.particles_per_second, self.total_fuel, self.T,
        self.launch_position[0]*6.68459e-12, self.launch_position[1]*6.68459e-12,
        self.launch_time])
        np.save('engine_settings_array', self.engine_settings_array)

    def calculate_launch_window(self):
        t0= time.time()
        sys.stdout.write('Calculating Launch Window')
        sys.stdout.flush()

        self.intercept_data= self.get_intercept_data()

        t1= time.time()
        print ' -  Done (%.2fs)'%(t1-t0)
        self.launch_window_calculated= True

    def calculate_gas_box(self):
        t0= time.time()
        sys.stdout.write('Simulating Gas Box')
        sys.stdout.flush()
        if isinstance(self.gas_box, Gas_Box):
            pass
        elif self.gas_box is None:
            self.gas_box= Gas_Box(seed= self.seed)
        else:
            fx._error(TypeError, 'Argument <gas_box> must be of type <Gas_Box>')
        self.box_mass= self.gas_box.box_mass
        self.box_force= self.gas_box.force_per_second
        t1= time.time()
        print ' -  Done (%.2fs)'%(t1-t0)
        self.gas_box_calculated= True

    def calculate_chambers(self):
        self._check_gas_box_calculated()
        t0= time.time()
        sys.stdout.write('Determining Ideal Number of Chambers')
        sys.stdout.flush()
        self.chambers= self.get_gas_boxes_required()
        self.fuel_mass= self.chambers*self.T*self.box_mass
        t1= time.time()
        print ' -  Done (%.2fs)'%(t1-t0)
        self.chambers_calculated= True

    def calculate_initial_conditions(self):
        self._check_launch_window_calculated()
        t0= time.time()
        sys.stdout.write('Getting Initial Velocities and Positions')
        sys.stdout.flush()
        self.x0, self.v0, self.u0, self.u_theta, self.v_rot= self.get_initial_conditions()
        t1= time.time()
        print ' -  Done (%.2fs)'%(t1-t0)
        self.initial_conditions_calculated= True

    def calculate_liftoff(self):
        self._check_initial_conditions_calculated()
        self._check_chambers_calculated()
        t0= time.time()
        sys.stdout.write('Launching First Stage')
        sys.stdout.flush()
        self.liftoff_data= self.launch_lifter(self.x0, self.v0, self.u0,
        self.u_theta, self.v_rot)
        t1= time.time()
        print ' -  Done (%.2fs)'%(t1-t0)
        self.liftoff_calculated= True

    def calculate_transfer(self):
        self._check_liftoff_calculated()
        t0= time.time()
        sys.stdout.write('Calculating Orbital Transfer')
        sys.stdout.flush()
        self.transfer_data= self.run_transfer()
        t1= time.time()
        print ' -  Done (%.2fs)'%(t1-t0)
        circ= self.circularize_sun(self.transfer_data['x'][-1], self.transfer_data['v'][-1])
        circ['t']= self.transfer_data['t'][-1]
        t2= time.time()
        sys.stdout.write('Performing Correctional Maneuvers')
        sys.stdout.flush()
        self.final_intercept_data= self.get_final_intercept_data(circ)
        t3= time.time()
        print ' -  Done (%.2fs)'%(t3-t2)
        self.transfer_calculated= True

    def calculate_circularization(self):
        self._check_transfer_calculated()
        t0= time.time()
        sys.stdout.write('Circularizing Orbit Around Target')
        sys.stdout.flush()
        self.circularization_data= self.get_circularization_data()
        self.target_orbit_data= self.circularize()
        self.intercept_time= self.target_orbit_data['t_end']
        t1= time.time()
        print ' -  Done (%.2fs)'%(t1-t0)
        self.circularization_calculated= True

    #ORBITAL DATA FUNCTIONS

    def get_interception_a(self, t):
        return (self.planet.sun.mass*(t - np.pi*np.sqrt((self.target.a**3.)/\
        (self.k))))**(1./3.)

    def get_eccentricity(self, r_a, r_p):
        return (r_a - r_p)/(r_a + r_p)

    def get_periapsis_velocity(self, a, e, k= None):
        if k == None:
            k= self.k
        return np.sqrt((k/a)*(1. + e)/(1. - e))

    def get_apoapsis_velocity(self, a, e, k= None):
        if k == None:
            k= self.k
        p= a*(1.-e)
        return np.sqrt((k/p)*(2. + e**2.))

    def get_orbital_period(self, a, k= None):
        if k == None:
            k= self.k
        return 2.*np.pi*np.sqrt(a**3./k)

    def get_semi_major_axis(self, apoapsis, e):
        return apoapsis/(1. + e)

    def get_orbit_data(self, x_a, x_p, start_at_apoapsis, k= None):
        '''Parameters <x_a> and <x_p> refer to the orbit's apoapsis and periapsis
        coordinates in the xy-plane. start_at_apoapsis defines whether we begin
        at the apoapsis or periapsis'''
        r_a= LA.norm(x_a)
        r_p= LA.norm(x_p)
        e= self.get_eccentricity(r_a, r_p)
        a= self.get_semi_major_axis(r_a, e)
        T= self.get_orbital_period(a, k)
        v_p= self.get_periapsis_velocity(a, e, k)
        v_a= self.get_apoapsis_velocity(a, e, k)
        if start_at_apoapsis == True:
            u= fx.unit_vector(x_a)
        elif start_at_apoapsis == False:
            u= fx.unit_vector(x_p)
        u_tangent= fx.rotate_vector(u, np.pi/2.)
        v_p= v_p * u_tangent
        v_a= v_a * u_tangent
        psi= np.arctan2(x_a[1], x_a[0])
        if psi < 0:
            psi += 2*np.pi
        return {'x_a':x_a, 'x_p':x_p, 'r_a':r_a, 'r_p':r_p, 'e':e, 'a':a,
        'T':T, 'v_a':v_a, 'v_p':v_p, 'psi':psi, 'start_at_apoapsis':start_at_apoapsis}

    def get_intercept_data(self, intervals= None, accuracy= 1e-8,
    final_height= None):
        if final_height == None:
            final_height= self.final_height
        multiplier= 1e-3
        lowest_dx= None
        best_time= None
        best_x_target= None
        years= 2.*(self.target.T + self.planet.T)
        if intervals == None:
            intervals= max(15, int(years/2.))
        closest_altitude= (self.target.radius*1000. + final_height)*6.68459e-12

        loading_string= ' - Preparing...'
        ui.write(string= loading_string)

        def prep_orbit_data(t):
            x= self.planet.get_position_from_time(t)
            theta= np.arctan2(x[1], x[0])
            theta_intercept= theta + np.pi
            x_intercept= self.target.get_position_from_angle(theta_intercept)
            u= fx.unit_vector(x_intercept)
            x_intercept += closest_altitude*u

            if LA.norm(x) > LA.norm(x_intercept):
                apoapsis= x
                periapsis= x_intercept
                start_at_apoapsis= True
            else:
                apoapsis= x_intercept
                periapsis= x
                start_at_apoapsis= False

            return self.get_orbit_data(apoapsis, periapsis, start_at_apoapsis)

        for y in range(int(years*intervals)):
            last_dx= None
            last_x_target= None
            last_t= None
            dt= 0.1
            t0= y/float(intervals)
            t= t0
            dy= years/float(intervals)

            while True:
                orbit_data= prep_orbit_data(t)
                time_to_intercept= orbit_data['T']/2. + t
                target_x_intercept= self.target.get_position_from_time(time_to_intercept)
                if orbit_data['start_at_apoapsis'] == True:
                    x= orbit_data['x_p']
                else:
                    x= orbit_data['x_a']

                dx= LA.norm(x - target_x_intercept)

                if last_dx is not None and abs(last_dx - dx) <= accuracy:
                    break
                elif last_dx is None or last_dx > dx:
                    last_dx= dx
                    last_x_target= target_x_intercept
                    last_x= x
                    last_t= t
                    t += dt
                else:
                    t -= dt
                    dt *= multiplier
                    t += dt

            ui.delete_chars(len(loading_string))
            progress= float(y)/int(years*intervals)*100.
            loading_string= ' - Loading (%d%%)'%(progress)
            ui.write(string= loading_string)

            if lowest_dx is None or lowest_dx > last_dx:
                lowest_dx= last_dx
                best_x_target= last_x_target
                best_x= last_x
                best_time= last_t

        ui.delete_chars(len(loading_string))
        orbit_data= prep_orbit_data(best_time)
        data= {'t':best_time, 'h_final':lowest_dx - self.target.radius*6.68459e-9,
        'x_target':best_x_target, 'x_closest':best_x}
        data.update(orbit_data)
        return data

    def get_analytical_position(self, orbit_data, theta, unit= False):
        a= orbit_data['a']
        e= orbit_data['e']
        psi= orbit_data['psi']

        p= a*(1. - e**2.)
        r= p/(1. + e*np.cos(theta + np.pi - psi))
        ur= np.array([np.cos(theta), np.sin(theta)])
        if unit == False:
            return np.abs(r)*ur
        else:
            return ur

    def get_analytical_velocity(self, orbit_data, theta, unit= False):
        a= orbit_data['a']
        e= orbit_data['e']
        psi= orbit_data['psi']
        mu= self.planet.mu

        k= self.planet.sun.mass*self.planet.G
        p= a*(1. - e**2.)
        ur= np.array([np.cos(theta), np.sin(theta)])
        utheta= fx.rotate_vector(ur, np.pi/2.)
        r= p/(1. + e*np.cos(theta + np.pi - psi))
        vabs= np.sqrt(mu*((2./r)-(1./a)))
        if unit == False:
            return vabs*utheta
        else:
            return utheta

    def get_circularization_data(self):
        self._check_transfer_calculated()

        t0= self.final_intercept_data['t'][-1]
        x0= self.final_intercept_data['x'][-1]
        v0= self.final_intercept_data['v'][-1]

        xp0= self.target.get_position_from_time(t0)
        vp0= self.target.get_velocity_from_time(t0)

        r0= LA.norm(x0 - xp0)
        T= 0.01#2.*np.pi*np.sqrt((r0**3.)/(self.G*self.target.mass))

        v_orbit= np.sqrt(self.target.mass*self.planet.G/r0)
        uv_orbit= fx.rotate_vector(fx.unit_vector(x0 - xp0), np.pi/2.)
        v_orbit= v_orbit*uv_orbit

        return {'v1':vp0 + v_orbit, 'x':x0, 'v0':v0, 't':t0, 'r':r0, 'T':T}

    def get_final_intercept_data(self, circ, accuracy= 1e-8, final_height= None):
        if final_height == None:
            final_height= self.final_height
        lowest_dx= None
        years= 4.*(self.target.T + self.planet.T)
        closest_altitude= (self.target.radius*1000. + final_height)*6.68459e-12

        def prep_orbit_data(x, t):
            theta= np.arctan2(x[1], x[0])
            theta_intercept= theta + np.pi
            x_intercept= self.target.get_position_from_angle(theta_intercept)
            u= fx.unit_vector(x_intercept)
            x_intercept += closest_altitude*u

            if LA.norm(x) > LA.norm(x_intercept):
                apoapsis= x
                periapsis= x_intercept
                start_at_apoapsis= True
            else:
                apoapsis= x_intercept
                periapsis= x
                start_at_apoapsis= False

            return self.get_orbit_data(apoapsis, periapsis, start_at_apoapsis)

        v1= self.burn(circ['t'], circ['v0'], circ['v1'])
        trajectory= self.trajectory(circ['t'], circ['x'], v1, years, 1e4)
        t_array= trajectory['t']
        x_array= trajectory['x']
        v_array= trajectory['v']
        last_dx= None
        for i,(t,x,v) in enumerate(zip(t_array, x_array, v_array)):
            orbit_data= prep_orbit_data(x, t)
            time_to_intercept= orbit_data['T']/2. + t
            target_x_intercept= self.target.get_position_from_time(time_to_intercept)
            if orbit_data['start_at_apoapsis'] == True:
                x0= orbit_data['x_p']
            else:
                x0= orbit_data['x_a']

            dx= LA.norm(x0 - target_x_intercept)

            if last_dx is None or last_dx > dx:
                last_dx= dx
                last_i= i
                last_t= t
                last_x= x
                last_v= v

        orbit_data= prep_orbit_data(last_x, last_t)
        if orbit_data['start_at_apoapsis'] is True:
            v2= orbit_data['v_a']
        else:
            v2= orbit_data['v_p']
        v2= self.burn(last_t, last_v, v2)

        x_first= x_array[:last_i]
        v_first= v_array[:last_i]
        traj= self.trajectory(last_t, last_x, v2, orbit_data['T']/2., 1e4)

        self.intercept_time= traj['t_end']
        return {'t':np.linspace(circ['t'], last_t + orbit_data['T']/2.,
        len(x_first) + len(traj['x'])),
        'x':np.concatenate((x_first, traj['x'])),
        'v':np.concatenate((x_first, traj['v']))}

    #LIFTOFF DATA FUNCTIONS

    def get_gas_boxes_required(self):
        self._check_gas_box_calculated()
        chambers= 1.
        step= 1e15
        v= 0.
        x= 0.
        dt= self.T*self.accuracy
        while abs(self.planet.get_escape_velocity(0) - v) >= self.accuracy:
            while True:
                x_escape= None
                t_escape= None
                mass= (chambers*self.box_mass*self.T) + self.payload_mass
                dm= chambers*self.box_mass*dt
                force= chambers*self.box_force
                x= 0.
                v= 0.
                for i in range(int(1./self.accuracy)):
                    mass -= dm
                    a= (force/mass + self.planet.get_gravitational_acceleration(x))
                    v += a*dt
                    x += v*dt
                    if x_escape == None and v > self.planet.get_escape_velocity(x):
                        x_escape= x
                        t_escape= i*dt
                if v > self.planet.get_escape_velocity(0):
                    break

                chambers += step
            chambers -= step
            step /= 10.
        self.escape_altitude= x_escape
        self.escape_time= t_escape
        return chambers

    def get_initial_conditions(self):
        self._check_launch_window_calculated()
        t0= self.intercept_data['t'] - self.T*3.17098e-8
        self.launch_time= t0
        x= self.planet.get_position_from_time(t0)
        v= self.planet.get_velocity_from_time(t0)
        theta= self.planet.get_angle_from_time(t0)

        u_theta= self.planet.get_analytical_velocity(theta, unit= True)
        v_rot= (self.planet.radius*1000.*2.*np.pi)/(self.planet.period*24.*3600.)

        u= fx.unit_vector(x)
        return self.planet.convert_AU(x), self.planet.convert_AU_per_year(v),\
        u, u_theta, v_rot

    def get_orbit_stability(self, r, R):
        '''<R> is satellite distance from sun,
        <r> is satellite distance from target
        k <= 10 is undesirable'''
        M, m= self.planet.sun.mass, self.target.mass
        return ((R/r)**2.)*(m/M)

    #IN FLIGHT FUNCTIONS

    def burn(self, t, v0, v1):
        self.burns.append([t, v1 - v0])
        return v1

    #TRAJECTORIES

    def run_transfer(self, steps= 1e4):
        self._check_liftoff_calculated()
        x0= self.liftoff_data['x'][-1]*6.68459e-12
        v_start= self.liftoff_data['v'][-1]*0.000210805
        t0= self.liftoff_data['t_abs'][-1]*3.17098e-8

        data= self.intercept_data
        if data['start_at_apoapsis'] == True:
            v0= data['v_a']
        else:
            v0= data['v_p']

        x1= data['x_closest']

        T= data['T']/2.

        t_end= t0 + T
        dt= float(T)/steps

        t= np.linspace(t0, t_end, steps)

        planet_name= self.planet.name.lower()
        G= self.solar_system.G
        mass_p= self.solar_system.planets[planet_name].mass
        mass_s= self.planet.sun.mass
        c1= -G*mass_s*dt
        c2= -G*mass_p*dt
        x_p= self.solar_system.planets[planet_name].get_position_from_time(t)

        @jit(nopython= True, cache= True)
        def integrate(x_p, c1, c2, x0, v0, dt):
            x= x0.copy()
            v= v0.copy()
            for n in range(len(x_p)-1):
                v+= x*c1/(LA.norm(x)**3.) + \
                (x - x_p[n])*c2/(LA.norm(x - x_p[n])**3.)
                x+= v*dt
            return x

        @jit(cache= True)
        def thrust(x_p, c1, c2, x0, x1, v0, t0, dt):
            goal= LA.norm(x1)
            d_step= 0.1
            best_v= v0
            best= LA.norm(integrate(x_p, c1, c2, x0, v0, dt) - goal)
            step= 1e-3
            i= 1.

            while True:
                test= LA.norm(integrate(x_p, c1, c2, x0, v0*i, dt) - goal)
                if test < best:
                    best= test
                    best_v= v0*i
                    i+= step
                elif step < 1e-16:
                    break
                else:
                    step*= d_step
                    i+= step

            return best_v

        v0= self.burn(t0, v_start, thrust(x_p, c1, c2, x0, x1, v_start, t0, dt))

        return self.trajectory(t0, x0, v0, T, steps)

    def circularize_sun(self, x, v):
        v_orbit= np.sqrt(self.planet.sun.mass*self.planet.G/(LA.norm(x)))
        uv_orbit= fx.rotate_vector(fx.unit_vector(x), np.pi/2.)
        v_orbit= v_orbit*uv_orbit

        return {'v1':v_orbit, 'x':x, 'v0':v}

    def circularize(self, steps= 1e4):
        self._check_transfer_calculated()
        data= self.circularization_data
        t0= data['t']
        x0= data['x']
        v0= self.burn(t0, data['v0'], data['v1'])
        T= data['T']

        return self.trajectory(t0, x0, v0, T, steps)

    #INTEGRATORS

    def launch_lifter(self, x0, v0, u, u_theta, v_rot, chambers= None):
        self._check_chambers_calculated()
        if chambers == None:
            chambers= self.chambers
        dt= self.T/float(self.steps)
        self.launch_fuel_mass= (chambers*self.box_mass*self.T)
        mass= self.launch_fuel_mass + self.payload_mass
        dm= chambers*self.box_mass*dt
        F= chambers*self.box_force

        G= self.planet.G_SI
        planet_mass= self.planet.mass
        sun_mass= self.planet.sun.mass
        radius= self.planet.radius*1000.

        def unit_vector(v):
            a= LA.norm(v)
            if a == 0:
                return np.zeros_like(v)
            else:
                return v/float(a)

        def jit_gravity(G, mass, r):
            return -G*1.99e30*mass*unit_vector(r)/(LA.norm(r)**2.)

        def integration(G, planet_mass, radius, steps, mass, dm, F, T, dt, u,
        utheta, x0, v0, v_rot, sun_mass):
            x= np.zeros((steps, 2))
            x_p= np.copy(x)
            v= np.copy(x)
            v_p= np.copy(x)
            t= np.linspace(0, T, steps)
            t_abs= t + self.planet.convert_year(self.intercept_data['t'], 's')
            x[0]= x0 + radius*u
            v[0]= v0 + v_rot*u_theta
            x_p[0]= x0
            v_p[0]= v0
            v_esc= None
            x_esc= None
            t_esc= None
            for i in range(steps-1):
                mass -= dm

                a_p= jit_gravity(G, sun_mass, x_p[i])
                v_p[i+1]= v_p[i] + a_p*dt
                x_p[i+1]= x_p[i] + v0*dt

                u= unit_vector(x[i]-x_p[i])
                a= (F/mass)*u
                a += jit_gravity(G, planet_mass, x[i]-x_p[i])
                a += jit_gravity(G, sun_mass, x[i])

                v[i+1]= v[i] + a*dt
                x[i+1]= x[i] + v[i+1]*dt

                if v_esc is None and np.sqrt(np.sum((v[i+1] - v0 - v_rot*u_theta)**2.))\
                > self.planet.get_escape_velocity(np.sqrt(np.sum((x[i+1]-x_p[i+1]-radius*u)**2.))):
                    v_esc= v[i+1].copy()
                    x_esc= x[i+1].copy()
                    t_esc= t[i+1].copy()
                    i_esc= i+1

            return {'t':t, 'x':x, 'v':v, 'x_p':x_p, 'v_p':v_p, 't_esc':t_esc,
            'x_esc':x_esc, 'v_esc':v_esc, 'i_esc':i_esc, 't_abs':t_abs}

        self.launch_position= x0.copy()
        return integration(G, planet_mass, radius, self.steps, mass, dm, F,
        self.T, dt, u, u_theta, x0, v0, v_rot, sun_mass)

    def trajectory(self, t0, x0, v0, T, steps, inline_print= True):

        if inline_print == True:
            ui.write(' - ')
        loading_string= 'Preparing...'
        ui.write(string= loading_string)

        masses= self.solar_system('mass', array= True)
        radii= self.solar_system('radius', array= True)
        G= self.solar_system.G

        def get_gravity(x, t):
            x_p= self.solar_system.get_positions_from_time(t)
            r= x - x_p
            a0= np.divide(-G*masses, LA.norm(r, axis= -1)**3.)
            a= np.zeros((len(a0), 2))
            a[:,0], a[:,1]= a0, a0
            return np.sum(a*r)

        def get_sun_gravity(x):
            a= -G*self.planet.sun.mass/(LA.norm(x)**3.)
            return a*x

        t_end= t0 + T
        dt= float(T)/steps
        n= 0

        t= [t0]
        x= [x0]
        v= [v0]

        while t[n] < t_end:
            v.append(v[n] + get_sun_gravity(x[n])*dt + get_gravity(x[n], t[n])*dt)
            x.append(x[n] + v[n+1]*dt)
            t.append(t[n] + dt)

            progress= (float(t[n+1])/float(t_end))*100.
            new_string= ' Loading (%d%%)'%(int(progress))
            if loading_string != new_string:
                ui.delete_chars(len(new_string))
                loading_string= new_string
                ui.write(string= loading_string)
            n += 1

        ui.delete_chars(len(loading_string))

        return {'t':np.array(t), 'x':np.array(x), 'v':np.array(v), 't_end':t_end}

    def fast_trajectory(self, planet_name, t0, x0, v0, T, steps):
        t_end= t0 + T
        dt= float(T)/steps

        t= np.linspace(t0, T, steps)
        x= np.zeros((len(t), 2))
        x[0]= x0
        v= np.zeros_like(x)
        v[0]= v0

        G= self.solar_system.G
        mass_p= self.solar_system.planets[planet_name].mass
        mass_s= self.planet.sun.mass
        positions_p= self.solar_system.planets[planet_name].get_position_from_time(t)

        @jit(nopython= True, cache= True)
        def integrate(x_p, c1, c2, x, v, t):
            for n in range(len(x_p)-1):
                v[n+1]= v[n] + x[n]*c1/(LA.norm(x[n])**3.) - \
                (x[n] - x_p[n])*c2/(LA.norm(x[n] - x_p[n])**3.)
                x[n+1]= x[n] + v[n+1]*dt
            return x

        return integrate(positions_p, G*mass_s*dt, G*mass_p*dt, x, v, t)

    def find_burn_fuel(self, dt= 1e-4, chambers= None):
        if chambers == None:
            chambers= self.chambers
        F= chambers*self.box_force

        @jit(cache= True)
        def integrate(dv, F, dt, m, chambers, box_mass):
            t= 0.
            v= 0.
            while v < dv:
                t+= dt
                v+= (F/m)*dt
            return chambers*box_mass*t

        total_mass= self.payload_mass
        for b in reversed(self.burns):
            dv= LA.norm(b[1])
            total_mass+= integrate(dv, F, dt, total_mass, self.chambers,
            self.box_mass)
        return total_mass - self.payload_mass + self.launch_fuel_mass

    #PLOTTING FUNCTIONS

    def plot_liftoff(self):
        self._check_liftoff_calculated()
        radius= self.planet.radius*1000.
        data= self.liftoff_data
        x0= data['x_p'][0]

        planet= plt.Circle(xy= data['x_p'][0], radius= radius, color= 'b')
        planet2= plt.Circle(xy= data['x_p'][-1], radius= radius, color= 'b')
        fig, ax= plt.subplots()
        ax.set(aspect=1)
        ax.add_artist(planet)
        ax.add_artist(planet2)
        plt.plot(data['x_p'][:,0], data['x_p'][:,1], '--b')
        plt.plot(data['x'][:,0], data['x'][:,1], '-r')
        x= data['x_esc']
        plt.plot(x[0], x[1], 'xk', ms= 10)

        plt.legend(["Planet's Trajectory", "Rocket's Trajectory",
        "Point of Escape Velocity"])

        plt.title("Launching Rocket From Planet %s in Seed %d\n Target: Planet %s"\
        %(self.planet.name, self.planet.seed, self.target.name))

        plt.show()

    def plot_intercept(self, prediction= True, numerical= True):
        self._check_launch_window_calculated()

        self.planet._check_analytical()
        self.target._check_analytical()

        self.planet._check_numerical()
        self.target._check_numerical()

        #Planet and Target Orbits
        x_planet= self.planet.analytical['x']
        x_target= self.target.analytical['x']
        t0= self.intercept_data['t']
        if numerical is True:
            self._check_circularization_calculated()
            t1= self.intercept_time

        #Initial and Final Positions of Probe and Target Planet
        if LA.norm(self.planet.get_position_from_time(t0)) <\
        LA.norm(self.target.get_position_from_time(t0)):
            planet_x0= self.intercept_data['x_p']
            target_x0= self.target.get_position_from_time(t0)

            probe_x1= self.intercept_data['x_a']
        else:
            planet_x0= self.intercept_data['x_a']
            target_x0= self.target.get_position_from_time(t0)

            probe_x1= self.intercept_data['x_p']

        if numerical == True:
            target_x1= self.target.get_position_from_time(t1)
        else:
            target_x1= self.intercept_data['x_target']

        legend= ["Probe's Start Position", "Target's Start Position",
        "Probe's Final Position", "Target's Final Position",
        "Planet %s's Orbit"%(self.planet.name), "Planet %s's Orbit"%(self.target.name)]
        sun= plt.Circle(xy= (0.,0.), radius= self.planet.sun.radius*6.68459e-9,
        color= 'y')
        planet_0= plt.Circle(xy= planet_x0, radius= self.planet.radius*6.68459e-9,
        color= 'y')
        target_0= plt.Circle(xy= target_x0, radius= self.target.radius*6.68459e-9,
        color= 'y')
        target_1= plt.Circle(xy= target_x1, radius= self.target.radius*6.68459e-9,
        color= 'y')
        fig, ax= plt.subplots()
        ax.set(aspect= 1)
        ax.add_artist(sun)
        ax.add_artist(planet_0)
        ax.add_artist(target_0)
        ax.add_artist(target_1)

        plt.plot(planet_x0[0], planet_x0[1], 'xr', ms= 20)
        plt.plot(target_x0[0], target_x0[1], '*g', ms= 15)

        plt.plot(probe_x1[0], probe_x1[1], 'xb', ms= 20)
        plt.plot(target_x1[0], target_x1[1], '*k', ms= 15)

        plt.plot(x_planet[0], x_planet[1], '-.k')
        plt.plot(x_target[0], x_target[1], '--k')

        if numerical is True:
            x_t= self.transfer_data['x']
            x_f= self.final_intercept_data['x']
            x_c= self.target_orbit_data['x']
            x_n= np.concatenate((x_t,x_f,x_c))
            plt.plot(x_n[:,0], x_n[:,1], '-m')
            legend.append("Probe's True Trajectory")

        if prediction is True:
            theta= np.linspace(0., 2*np.pi, 10000)
            x_p= self.get_analytical_position(self.intercept_data, theta)
            plt.plot(x_p[0], x_p[1], '-k')
            legend.append("Probe's Predicted Trajectory")

        plt.legend(legend, loc= 9, bbox_to_anchor= (-0.2, 1))#, ncol= len(legend))
        plt.title("A Hohmann Transfer from Planet %s to Planet %s"\
        %(self.planet.name, self.target.name))
        plt.xlabel("AU")
        plt.ylabel("AU")

        plt.show()

    #CLASS OPERATIONS

    def __str__(self):
        string= 'Basic Info:'
        string += '\n\tSeed: %d\n\tPlanet Name: %s, Planet Index: %d'%(self.seed,
        self.planet.name, self.planet.index)
        string += '\n\tTarget Name: %s, Target Index: %d'%(self.target.name,
        self.target.index)

        if self.chambers_calculated == True:
            string += '\n\nRocket Stats:'
            string += '\n\tNumber of Chambers: %g'%(self.chambers)
            string += '\n\tTotal Thrust: %g kN'%(self.chambers*self.box_force/1e3)

        if self.launch_window_calculated == True:
            icd= self.intercept_data
            string += '\n\nLaunch Window Data:'
            string += '\n\tIdeal Launch Time: %g yrs'%(icd['t'])
            string += '\n\tClosest Approach: %g AU, %g km'\
            %(icd['h_final'], self.planet.convert_AU(icd['h_final'], 'km'))
            string += '\n\tRocket Launches at '
            if icd['start_at_apoapsis'] == True:
                string += 'Apoapsis'
            else:
                string += 'Periapsis'

            string += '\n\nIntercept Orbit Data:'
            string += '\n\tApoapsis: %g AU, Periapsis %g AU'\
            %(icd['r_a'], icd['r_p'])
            string += '\n\tEccentricity: %g, Orbital Period: %g yrs'\
            %(icd['e'], icd['T'])
            string += '\n\tSemi-Major Axis: %g AU, Angle of Semi-Major Axis: %g rad'\
            %(icd['a'], icd['psi'])
            string += '\n\tPeriapsis Velocity: %g AU/yr, %g km/s'\
            %(LA.norm(icd['v_p']),
            LA.norm(self.planet.convert_AU_per_year(icd['v_p'], 'km/s')))

        if self.liftoff_calculated == True:
            ld= self.liftoff_data
            string += '\n\nLiftoff Data:'
            string += '\n\tEscape Velocity: %s m/s'%(str(tuple(ld['v_esc'])))
            string += '\n\tEscape Position: %s m'%(str(tuple(ld['x_esc'])))
            string += '\n\tEscape Time: %s s'%(ld['t_esc'])

        if len(self.burns) > 0:
            self.total_fuel= self.find_burn_fuel()
            string += '\n\nInterception Burns (Delta-V):'
            for n,b in enumerate(self.burns):
                rad= np.arctan2(b[1][1], b[1][0])
                if rad < 0:
                    rad += 2*np.pi
                deg= np.rad2deg(rad)
                dv= self.planet.convert_AU_per_year(LA.norm(b[1]), 'm/s')
                d_sign= u'\u00b0'
                string += '\n[%d]\t%g m/s, %.3f%s at %g yrs'%(n+1, dv, deg, d_sign, b[0])
                string += ', or (%g, %g) AU/yr'%(b[1][0], b[1][1])
            string += '\n\tTotal Fuel Burnt: %g kg'%(self.total_fuel)

        return string

    def print_str(self):
        print self.__str__()

    #TEST FUNCTIONS

    def _check_launch_window_calculated(self):
        if self.launch_window_calculated == False:
            self.calculate_launch_window()

    def _check_gas_box_calculated(self):
        if self.gas_box_calculated == False:
            self.calculate_gas_box()

    def _check_chambers_calculated(self):
        if self.chambers_calculated == False:
            self.calculate_chambers()

    def _check_initial_conditions_calculated(self):
        if self.initial_conditions_calculated == False:
            self.calculate_initial_conditions()

    def _check_liftoff_calculated(self):
        if self.liftoff_calculated == False:
            self.calculate_liftoff()

    def _check_transfer_calculated(self):
        if self.transfer_calculated == False:
            self.calculate_transfer()

    def _check_circularization_calculated(self):
        if self.circularization_calculated == False:
            self.calculate_circularization()

    def test_launch_simple(self):
        self._check_chambers_calculated()

        m_to_AU= 1./149597870700.
        radius= self.planet.radius*1000.

        x0= np.array([self.planet.x0 + radius*m_to_AU, self.planet.y0])

        self.planet.system.engine_settings(self.box_force, self.chambers,
        self.gas_box.particles_per_second, self.fuel_mass, self.T,
        x0, 0.)

        x0= self.planet.convert_AU(x0)

        x_final= np.array([self.escape_altitude, 0.]) + x0

        v_rot= (radius*2.*np.pi)/(self.planet.period*24.*3600.)

        y_final= (self.planet.convert_AU_per_year(self.planet.vy0) + v_rot)*\
        np.array([0.,self.escape_time])

        x_final += y_final

        x_final *= m_to_AU

        real= self.planet.system.mass_needed_launch(x_final, test=True)[2]
        print self.planet.convert_AU(x_final - real)

    def test_launch(self):
        self._check_liftoff_calculated()
        m_to_AU= 1./149597870660.
        radius= self.planet.radius*1000.

        x0= np.array([self.planet.x0, self.planet.y0])
        u= fx.rotate_vector(np.array([1.,0.]), self.planet.omega)

        self.planet.system.engine_settings(self.box_force, self.chambers,
        self.gas_box.particles_per_second, self.fuel_mass, self.T,
        x0 + radius*m_to_AU*u, 0.)

        x0= self.planet.convert_AU(x0)

        v_rot= (radius*2.*np.pi)/(self.planet.period*24.*3600.)
        u_theta= fx.rotate_vector(np.array([0.,1.]), self.planet.omega)

        v0= np.array([self.planet.vx0, self.planet.vy0])
        v0= self.planet.convert_AU_per_year(v0)

        data= self.launch_lifter(x0= x0, v0= v0, u= u,
        u_theta= u_theta, v_rot= v_rot)

        i_esc= data['i_esc']

        planet= plt.Circle(xy= data['x_p'][0], radius= radius, color= 'b')
        planet2= plt.Circle(xy= data['x_p'][i_esc], radius= radius, color= 'b')
        planet3= plt.Circle(xy= data['x_p'][-1], radius= radius, color= 'r',
        alpha= 0.2)
        fig, ax= plt.subplots()
        ax.set(aspect=1)
        ax.add_artist(planet)
        ax.add_artist(planet2)
        ax.add_artist(planet3)
        plt.plot(data['x'][:,0], data['x'][:,1])
        plt.plot(data['x_p'][:,0], data['x_p'][:,1])
        plt.plot(data['x_p'][i_esc,0], data['x_p'][i_esc,1], 'xw', ms= 20)
        plt.plot(data['x_p'][-1,0], data['x_p'][-1,1], 'xy', ms= 20)
        plt.xlim(xmin= x0[0]-radius)
        plt.ylim(ymin= x0[1]-radius)

        x= data['x_esc']
        plt.plot(x[0], x[1], 'xg', ms= 10)
        plt.grid(True)
        x *= m_to_AU

        real= self.planet.convert_AU(self.planet.system.mass_needed_launch(x, test=True)[2])
        print "Target off by %s meters"%(str(tuple(self.planet.convert_AU(x)-real)))
        plt.plot(real[0], real[1], 'xr', ms= 10)
        plt.show()

class Satellite(object):

    def __init__(self, mass= 1100., cs_area= 15., seed= 45355):
        self.mass= mass
        self.cs_area= cs_area
        self.solar_system= Solar_System()
        self.seed= seed
        self.c= 299792458.
        self.k= 1.38064852e-23
        self.gases_dict= fx.get_gas_data()
        self.noise_sigma= np.load('noise_sigma.npy')
        self.spectrum= np.load('spectrum.npy')

    def take_pictures(self, a_phi= 70, a_theta= 70, theta0= (np.pi/2)):
        a_phi= np.deg2rad(a_phi)
        a_theta= np.deg2rad(a_theta)
        sky_sphere= np.load('himmelkule.npy')
        x_max= (2*np.sin(a_phi/2))/(1+np.cos(a_phi/2))
        x_min= -(2*np.sin(a_phi/2))/(1+np.cos(a_phi/2))
        y_max= (2*np.sin(a_theta/2))/(1+np.cos(a_theta/2))
        y_min= -(2*np.sin(a_theta/2))/(1+np.cos(a_theta/2))
        x= np.linspace(x_min,x_max,640)
        y= np.linspace(y_max,y_min,480)
        X, Y= np.meshgrid(x,y)
        XY= np.zeros((480,640,2))
        XY[:,:,0]= X; XY[:,:,1]= Y
        projections= np.zeros((360,480,640,3),dtype= np.uint8)
        for j in range(359):
            print j
            phi0= np.deg2rad(j)
            rho= np.sqrt(X**2 + Y**2)
            c= 2*np.arctan(rho/2)
            theta= np.pi/2 - np.arcsin(np.cos(c)*np.cos(theta0) + Y*np.sin(c)*np.sin(theta0)/rho)
            phi= phi0 + np.arctan(X*np.sin(c)/(rho*np.sin(theta0)*np.cos(c) - Y*np.cos(theta0)*np.sin(c)))
            for n,(i,v) in enumerate(zip(theta, phi)):
                for m,(k,w) in enumerate(zip(i,v)):
                    pixnum= A2000.ang2pix(k,w)
                    temp= sky_sphere[pixnum]
                    projections[j][n][m]= (temp[2], temp[3], temp[4])
        np.save('projections.npy', projections)

    def get_lambda_doppler(self, l, v):     #Need to add velocity relative to target planet
        return l/(1 + v/self.c)

    def get_fmin_lc(self, i):
        fmin= np.amin(self.spectrum[i-100:i+100,1])
        lc_i= int(np.where(self.spectrum[:,1] == fmin)[0])
        lc= float(self.spectrum[lc_i,0])
        return fmin, lc, lc_i

    def get_sigma(self, l, m, T):
        return (2*l/self.c)*np.sqrt(2*self.k*T/m)

    def get_full_comb(self, l, m, T0= 150, T1= 450, lc_lim=300):
        fmin, lc, lc_i= self.get_fmin_lc(l)
        fmin_a= np.linspace(fmin, 1, 30)
        sigma_a= np.linspace(self.get_sigma(lc, m, T0), self.get_sigma(lc, m, T1), 30)
        lc_a= self.spectrum[lc_i-lc_lim/2:lc_i+lc_lim/2, 0]
        lc_i= np.linspace(lc_i-lc_lim/2,lc_i+lc_lim/2,lc_lim+1, dtype=int)
        return np.array(np.meshgrid(fmin_a, sigma_a, lc_a)).T.reshape(-1,3), lc_i

    def get_flux_model(self, l, fmin, sigma, lc):
        fm= (1 + (fmin - 1)*np.exp((-(l - lc)**2)/(2*sigma**2)))
        return fm

    def get_flux_obv(self, i):
        if isinstance(i, np.ndarray):
            return self.spectrum[i[0]:i[-1],1]
        else:
            return self.spectrum[i,1]

    def get_noise_sigma(self, i):
        if isinstance(i, np.ndarray):
            return self.noise_sigma[i[0]:i[-1],1]
        else:
            return self.noise_sigma[i,1]

    def get_atmo_temp(self, h):
        return 450*np.exp(-h/10000.)

    def plot(self, l= None, limit= 150, lc= None):
        plt.figure(figsize=(25,15))
        lc_line= (np.abs(self.spectrum[:,0] - lc)).argmin()
        plt.plot(self.spectrum[l-limit:l+limit,0], self.spectrum[l-limit:l+limit,1])
        plt.axvline(x=self.spectrum[l,0], color='r')
        plt.axvline(x=self.spectrum[lc_line,0], color='g')
        plt.show()

    def check_for_line(self, mol):          #Need to add multiple lambdas
        comb, lc_i= self.get_full_comb(mol, m_N2O)
        df= np.zeros(len(comb))

        for i in xrange(len(comb)):
            df[i]= np.sum((1/self.get_noise_sigma(lc_i))*(self.get_flux_obv(lc_i) - \
            self.get_flux_model(self.spectrum[lc_i[0]:lc_i[-1],0]\
            , comb[i,0], comb[i,1], comb[i,2]))**2)

        if np.shape(np.where(df == np.amin(df)))[1] > 1:
            print "No line present"
        else:
            self.plot(l= N2O, lc= comb[np.where(df == np.amin(df)),2])

    def get_mean_mol_weight(self, l):
        mu= 0
        m_H= 1.67e-27
        for n,i in enumerate(l):
            mu += (1./len(i))*(self.gases_dict[i]['mass']/m_H)
        mu *= (1./len(i))
        return mu

    def get_orient(self):
        with open('sat_commands.txt', 'r') as infile:
            inputs= [elem for line in infile for elem in line.split()]
            for i, n in enumerate(inputs):
                if n == 'orient':
                    time= float(inputs[i+1])

        with open('planets.txt', 'w') as outfile:
            number_of_planets= self.solar_system.number_of_planets
            if self.seed == 45355:
                planets= ['sarplo', 'jevelan', 'calimno', 'sesena', 'corvee', 'bertela',
                'poppengo', 'trento']
            else:
                planets= list(string.ascii_lowercase[:number_of_planets])
            if number_of_planets % 2 == 0:
                pass
            else:
                del planets[-1]
            coords= np.zeros((len(planets) + 1, 2))
            self.solar_system.get_numerical_orbits()
            for i,n in enumerate(planets):
                try:
                    coords[i]= self.solar_system.orbits[n](time)
                    print "Interpolation of planet", n, "calculated correctly"
                except ValueError:
                    print "Interpolation of planet", n, "not calculated correctly."
            coords[-1]= [0,0]
            for item in planets:
                outfile.write(item)
                outfile.write(' ')
        np.save('coords.npy', coords)
        if os.name == 'nt':
            os.system("start /wait cmd /k python functions.py get_orient")
        elif os.name == 'posix':
            os.system('gnome-terminal -x sh -c "functions.py get_orient; bash"')

class Lander(object):

    def __init__(self, planet, mass= 90., cs_area= 6.2, watts= 40.,
    panel_efficiency= .12, seed= 45355):
        self.seed= seed
        self.mass= mass
        self.cs_area= cs_area
        self.watts= watts
        self.panel_efficiency= panel_efficiency
        self.sun= Sun(seed= self.seed)
        self.planet= Planet(name= planet.name, seed= self.seed)

    def get_min_panel_size(self):
        return self.watts/(self.planet.F*self.panel_efficiency)

    def get_parachute_area(rho0 = None):
        if rho0 == None:
            rho0= self.planet.rho0
        m= self.mass
        A0= self.cs_area
        G= 6.674e-11
        M= self.planet.mass*1.99e30
        R= self.planet.radius*1e3
        return (2.*G*M*m)/(9.*R**2*rho)

class Gaussian(object):

    def __init__(self, mu, sigma):
        self.mu= mu
        self.sigma= sigma
        self._getData()

    def __call__(self, x):
        return (1./(np.sqrt(2.*np.pi)*self.sigma))*np.exp(-0.5*(((x - self.mu)/self.sigma))**2.)

    def _getData(self):
        self.FWHM= 2.355*self.sigma
        self.maximum= self.__call__(self.mu)

    def point_probability(self, x, dx= 1e-2, dt= 2e-3):
        x= [float(x)-dx, float(x)+dx]
        return integrate_function_vectorized(f= self.__call__, a= x[0], b= x[1], dt= dt)

    def probability(self, a, b, dt= 2e-3, sigma_factor= 5.):
        infinity= sigma_factor * self.sigma
        if isinstance(a, types['text']) and a.lower() in inf_str:
            a= self.mu - infinity
        elif not isinstance(a, types['numbers']):
            _error(errortype= ValueError, msg= 'Invalid Type for Integration Bounds')
        if isinstance(b, types['text']) and b.lower() in inf_str:
            b= self.mu + infinity
        elif not isinstance(b, types['numbers']):
            _error(errortype= ValueError, msg= 'Invalid Type for Integration Bounds')
        return integrate_function_vectorized(f= self.__call__, a= a, b= b, dt= dt)

    def plot(self, title= 'Gaussian Plot', xlabel= 'Possible Outcomes', ylabel= 'Possible Outcome Density',
             sigma_factor= 4, dt= 2e-3):
        a= self.mu - sigma_factor * self.sigma
        b= self.mu + sigma_factor * self.sigma
        x= np.linspace(a, b, (b-a)/dt)
        plt.plot(x, self.__call__(x))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis([a, b, 0, 1.1*self.maximum])
        plt.show()
