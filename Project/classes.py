from ast2000solarsystem_27_v4 import AST2000SolarSystem as A2000
import functions as fx
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import time, sys, os
try:
    from numba import jit
    import numba as nb
except ImportError:
    string = "User must install module <numba> to use module <classes.py>"
    fx.error(ImportError, string)

'''
Steinn 52772
Simen 47566
Lars 78826
Ulrik 82275
Gabriel 45355
Anders 28631'''

class Planet(object):                                   #FIX LEAPFROG ALGORITHM

    def __init__(self, name, dt = 5e-7, frames_max = 100000, seed = 45355):

        self.numerical_complete = False
        self.analytical_complete = False
        self.seed = seed
        np.random.seed(self.seed)
        self.sun = Sun(seed = self.seed)
        planet_data, planet_order = fx.get_planet_data(seed = self.seed, return_order = True)

        name = name.lower()
        if name in planet_data:
            for prop, val in planet_data[name].iteritems():
                self.__dict__[prop] = val
                self.index = planet_order.index(name)
                if len(name) > 1:
                    self.name = name[0].upper() + name[1:]
                else:
                    self.name = name.upper()
        else:
            fx.error(NameError, 'Invalid Planet Name')

        #CONSTANTS
        self.G = 4.*np.pi**2.
        self.G_SI = 6.67408e-11

        #CALCULATED DATA
        self.b = self.a*np.sqrt(1 - self.e**2)
        self.mu_hat = (self.mass*self.sun.mass)/(self.mass+self.sun.mass)
        self.T = 2.*np.pi*np.sqrt((self.a**3.)/(self.G*self.sun.mass))
        self.F = self.sun.L/(4.*np.pi*(((1.+self.e)*self.a*1.496e11)**2.))
        self.temperature = (self.F/self.sun.sigma)**0.25

        #INTEGRATION PARAMETERS
        self.dt = dt
        self.frames = min(frames_max, int(self.T/self.dt))
        self.final_dt = float(self.T)/float(self.frames)

    #CALCULATIONS

    def get_analytical_position(self, theta):
        p = self.a*(1. - self.e**2.)
        r = p/(1. + self.e*np.cos(theta-self.omega))
        ur = np.array([-np.cos(theta-(self.omega-self.psi)), -np.sin(theta-(self.omega-self.psi))])
        return np.abs(r)*ur

    def get_analytical_orbit(self):
        theta = np.linspace(0, 2.*np.pi, self.frames)
        x = self.get_analytical_position(theta)
        self.analytical = {'theta': theta, 'x': x}
        self.analytical_complete = True

    def get_numerical_orbit(self, T = None, dt = None, frames = None, return_vals = False):
        #USES LEAPFROG INTEGRATION
        if T == None:
            T = self.T
        if dt == None:
            dt = self.dt
        if frames == None:
            frames = self.frames
        x0, y0, vx0, vy0 = self.x0, self.y0, self.vx0, self.vy0
        G, sun_mass = self.G, self.sun.mass
        @jit
        def jit_integrate(T,dt,frames,x0,y0,vx0,vy0,G,sun_mass):
            t = np.linspace(0., T, frames)
            x = np.zeros((frames, 2))
            v = x.copy()
            a = x.copy()

            x[0,0], x[0,1] = x0, y0
            v[0,0], v[0,1] = vx0, vy0
            r_magnitude = LA.norm(x[0])
            ur = np.divide(x[0], r_magnitude)
            a[0] = (-(G*sun_mass)/(r_magnitude**2.))*ur
            t_now = dt
            x_now = x[0].copy()
            v_now = v[0].copy()# + 0.5*get_acceleration(x_now)*dt
            a_now = a[0].copy()
            save_index = 1

            while t_now <= T + dt:
                r_magnitude = LA.norm(x_now)
                ur = np.divide(x_now, r_magnitude)
                a_now = (-(G*sun_mass)/(r_magnitude**2.))*ur
                v_now += a_now*dt
                x_now += v_now*dt

                if t_now >= t[save_index]:
                    x[save_index] = x_now
                    v[save_index] = v_now
                    a[save_index] = a_now
                    save_index += 1
                t_now += dt
            return t, x, v, a

        t, x, v, a = jit_integrate(T,dt,frames,x0,y0,vx0,vy0,G,sun_mass)

        if return_vals == False:
            self.numerical = {'t': t, 'x': x, 'v': v, 'a': a}
            self.numerical_complete = True
        else:
            return {'t': t, 'x': x, 'v': v, 'a': a}

    def get_2_body_numerical_orbit(self, T = None, dt = None, frames = None):
        if T == None:
            T = self.T
        if dt == None:
            dt = self.dt
        if frames == None:
            frames = self.frames
        x0, y0, vx0, vy0 = self.x0, self.y0, self.vx0, self.vy0
        G, sun_mass, mass = self.G, self.sun.mass, self.mass
        @jit
        def jit_integrate(T,dt,frames,x0,y0,vx0,vy0,G,sun_mass,mass):
            t = np.linspace(0., T, frames)
            x = np.zeros((frames, 2))
            v = x.copy()
            a = x.copy()

            x[0,0], x[0,1] = x0, y0
            v[0,0], v[0,1] = vx0, vy0
            r_magnitude = LA.norm(x[0])
            ur = np.divide(x[0], r_magnitude)
            a[0] = (((G*mass)/(r_magnitude**2.))-((G*sun_mass)/(r_magnitude**2.)))*ur
            t_now = dt
            x_now = x[0].copy()
            v_now = v[0].copy()# + 0.5*get_acceleration(x_now)*dt
            a_now = a[0].copy()
            save_index = 1

            while t_now <= T + dt:
                r_magnitude = LA.norm(x_now)
                ur = np.divide(x_now, r_magnitude)
                a_now = (((G*mass)/(r_magnitude**2.))-((G*sun_mass)/(r_magnitude**2.)))*ur
                v_now += a_now*dt
                x_now += v_now*dt
                if t_now >= t[save_index]:
                    x[save_index] = x_now
                    v[save_index] = v_now
                    a[save_index] = a_now
                    save_index += 1
                t_now += dt
            return t, x, v, a

        t, x, v, a = jit_integrate(T,dt,frames,x0,y0,vx0,vy0,G,sun_mass,mass)
        return {'t': t, 'x': x, 'v': v, 'a': a}

    def get_area(self, t0 = 0, dt = None):
        self._check_numerical()
        t0 = t0%self.T

        con1 = t0 - 0.5*self.final_dt < self.numerical['t']
        con2 = t0 + 0.5*self.final_dt > self.numerical['t']
        t0_index = np.where(con1 & con2)[0][0]

        if dt == None:
            dt = self.T-self.final_dt

        t1 = t0 + dt
        if dt > self.T:
            fx.error(ValueError, 'Cannot calculate area over a time larger than the orbital period')
        elif dt < self.final_dt:
            return 0
        elif t1 > self.T:
            dt2 = t1 - self.T
            t1 = self.T
            A2 = self.get_area(t0 = 0, dt = dt2)
        else:
            A2 = 0

        con3 = t1 - 0.5*self.final_dt < self.numerical['t']
        con4 = t1 + 0.5*self.final_dt > self.numerical['t']
        t1_index = np.where(con3 & con4)[0][0]

        x_slice = self.numerical['x'][t0_index:t1_index]
        r = LA.norm(x_slice[:-1], axis = 1)
        arc_lengths = LA.norm(np.diff(x_slice, axis = 0), axis = 1)

        A1 = np.sum(np.multiply(r, arc_lengths))
        return A1 + A2

    def get_arc_length(self, t0 = 0, dt = None):
        self._check_numerical()
        t0 = t0%self.T

        con1 = t0 - 0.5*self.final_dt < self.numerical['t']
        con2 = t0 + 0.5*self.final_dt > self.numerical['t']
        t0_index = np.where(con1 & con2)[0][0]

        if dt == None:
            dt = self.T-self.final_dt

        t1 = t0 + dt
        if dt > self.T:
            fx.error(ValueError, 'Cannot calculate area over a time larger than the orbital period')
        elif dt < self.final_dt:
            return 0
        elif t1 > self.T:
            dt2 = t1 - self.T
            t1 = self.T
            L2 = self.get_arc_length(t0 = 0, dt = dt2)
        else:
            L2 = 0

        con3 = t1 - 0.5*self.final_dt < self.numerical['t']
        con4 = t1 + 0.5*self.final_dt > self.numerical['t']
        t1_index = np.where(con3 & con4)[0][0]

        x_slice = self.numerical['x'][t0_index:t1_index]
        arc_lengths = LA.norm(np.diff(x_slice, axis = 0), axis = 1)

        L1 = np.sum(arc_lengths)
        return L1 + L2

    def get_mean_velocity(self, t0 = 0, dt = None, raw = False):
        self._check_numerical()
        t0 = t0%self.T

        con1 = t0 - 0.5*self.final_dt < self.numerical['t']
        con2 = t0 + 0.5*self.final_dt > self.numerical['t']
        t0_index = np.where(con1 & con2)[0][0]

        if dt == None:
            dt = self.T-self.final_dt

        t1 = t0 + dt
        if dt > self.T:
            fx.error(ValueError, 'Cannot calculate area over a time larger than the orbital period')
        elif dt < self.final_dt:
            return 0
        elif t1 > self.T:
            dt2 = t1 - self.T
            t1 = self.T
            v2 = self.get_mean_velocity(t0 = 0, dt = dt2, raw = True)
        else:
            v2 = None

        con3 = t1 - 0.5*self.final_dt < self.numerical['t']
        con4 = t1 + 0.5*self.final_dt > self.numerical['t']
        t1_index = np.where(con3 & con4)[0][0]

        x_slice = self.numerical['x'][t0_index:t1_index]
        arc_lengths = LA.norm(np.diff(x_slice, axis = 0), axis = 1)

        v1 = arc_lengths/self.final_dt
        if raw == True:
            return v1
        else:
            if v2 == None:
                return np.mean(v1)
            else:
                v3 = np.zeros(len(v1)+len(v2))
                v3[:len(v1)], v3[len(v1):] = v1, v2
                return np.mean(v3)

    def get_escape_velocity(self, h = 0.):
        '''Takes and returns SI-units'''
        return np.sqrt(2.*self.G_SI*1.99e30*self.mass/(self.radius*1000. + h))

    def get_gravitational_force(self, h = 0.):
        '''Takes and returns SI-units'''
        return -self.G_SI*self.mass/((self.radius*1000. + h)**2.)

    #TESTS

    def _check_numerical(self):
        if self.numerical_complete == False:
            self.get_numerical_orbit()

    def _check_analytical(self):
        if self.analytical_complete == False:
            self.get_analytical_orbit()

    #DATA VISUALIZATION

    def plot(self, analytical = True, numerical = False, axes = True):
        legend = ['Sun']
        sun = plt.Circle(xy = (0.,0.), radius = self.sun.radius*6.68459e-9,
        color = 'y')
        planet = plt.Circle(xy = (self.x0,self.y0), radius = self.radius*6.68459e-9,
        color = 'k')
        fig, ax = plt.subplots()
        ax.set(aspect=1)
        ax.add_artist(sun)
        ax.add_artist(planet)
        plt.plot(0,0,'oy',ms=1)

        if analytical == True:
            self._check_analytical()
            legend.append('Analytical Orbit')
            x_analytical = self.analytical['x']
            plt.plot(x_analytical[0], x_analytical[1], '-r')

        if numerical == True:
            self._check_numerical()
            legend.append('Numerical Orbit')
            x_numerical = self.numerical['x']
            plt.plot(x_numerical[:,0], x_numerical[:,1], '-b')

        if axes == True:
            x_a = [0., (1.+self.e)*self.a*np.cos(self.psi)]
            y_a = [0., (1.+self.e)*self.a*np.sin(self.psi)]
            x_b = [0., -(1.-self.e)*self.b*np.cos(self.psi)]
            y_b = [0., -(1.-self.e)*self.b*np.sin(self.psi)]
            plt.plot(x_a, y_a, '-g')
            plt.plot(x_b, y_b, '-m')
            legend += ['Semi-Major Axis', 'Semi-Minor Axis']

        plt.title('The Orbit of Planet %s'%(self.name))
        plt.legend(legend)
        plt.xlabel('x in AU')
        plt.ylabel('y in AU')
        plt.show()

    def plot_2_body(self, T = None):
        data = self.get_2_body_numerical_orbit()
        t = data['t']
        x, y = data['x'][:,0], data['x'][:,1]

        legend = ['Sun']
        sun = plt.Circle(xy = (0.,0.), radius = self.sun.radius*6.68459e-9,
        color = 'y')
        fig, ax = plt.subplots()
        ax.set(aspect=1)
        ax.add_artist(sun)
        plt.plot(0,0,'oy',ms=1)
        plt.plot(x, y)
        plt.title('The Orbit of Planet %s'%(self.name))
        plt.legend(legend)
        plt.xlabel('x in AU')
        plt.ylabel('y in AU')
        plt.show()

    def plot_velocity_curve(self, peculiar_velocity = (0,0), i = np.pi/2.,
    two_body = False, T = None, noise = True):
        if two_body == False:
            self._check_numerical()
            t = self.numerical['t']
            v = self.numerical['v']
        else:
            data = self.get_2_body_numerical_orbit(T = T)
            t = data['t']
            v = data['v']
        if not isinstance(peculiar_velocity, np.ndarray):
            peculiar_velocity = np.array(peculiar_velocity)
        v_max = np.max(LA.norm(v))*np.sin(i)
        v = v_max*np.cos((2*np.pi*t)/self.T) + LA.norm(peculiar_velocity)

        if noise == True:
            noisiness = np.random.normal(loc = 0.0, scale = v_max/5., size = len(t))
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

    def plot_light_curve(self, T = None, steps = 1e3, noise = True):

        sun_area = np.pi*(self.sun.radius**2.)
        planet_area = np.pi*(self.radius**2.)
        max_flux = sun_area
        min_flux = max_flux - planet_area
        max_flux /= sun_area
        min_flux /= sun_area
        d_flux = max_flux - min_flux
        v = LA.norm(np.array([self.vx0, self.vy0]))
        cross_time = 2.*6.68459e-9*self.radius/v
        min_time = 2.*6.68459e-9*self.sun.radius/v - cross_time
        t_tot = 2.*cross_time + 3.*min_time
        dt = t_tot/steps
        ct_frames = cross_time/dt
        mt_frames = min_time/dt

        ct_x1 = np.linspace(min_flux, max_flux, ct_frames)
        ct_x0 = np.copy(ct_x1)[::-1]
        mt_x = min_flux*np.ones(mt_frames)
        x0 = np.ones_like(mt_x)

        new = np.concatenate((x0, ct_x0, mt_x, ct_x1, x0))

        if noise == True:
            noisiness = np.random.normal(loc = 0.0, scale = 0.2*(max_flux-min_flux), size = len(new))
            new += noisiness

        t = 8760.*np.linspace(0., t_tot, len(new))
        plt.plot(t, new)
        plt.title('Light Curve of Planet %s Eclipsing its Sun'%(self.name))
        plt.xlabel('Time in Hours')
        plt.ylabel('Relative Flux')
        plt.axis([t[0], t[-1], min(new) - 0.1*abs(max(new) - min(new)),
        max(new) + 0.1*abs(max(new) - min(new))])
        plt.show()

    def __str__(self):
        properties = ['a', 'e', 'radius', 'omega', 'psi', 'mass', 'period', 'x0', 'y0',
        'vx0', 'vy0', 'rho0']
        string = 'Seed: %d\nPlanet Name: %s\nPlanet Index: %d'%(self.seed, self.name, self.index)
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

    #MISC FUNCTIONS

    def convert_AU(self, val, convert_to = 'm'):
        if convert_to == 'm':
            return 1.496e11*val
        elif convert_to == 'earth radii':
            return val/4.25875e-5
        elif convert_to == 'solar radii':
            return val/215.
        else:
            fx.error(KeyError, 'Invalid Conversion Unit <%s>'%(convert_to))

    def convert_AU_per_year(self, val, convert_to = 'm/s'):
        if convert_to == 'm/s':
            return val*4743.72
        elif convert_to == 'km/h':
            return val*17066.0582
        elif convert_to == 'km/s':
            return val*4.74372
        else:
            fx.error(KeyError, 'Invalid Conversion Unit <%s>'%(convert_to))

    def convert_year(self, val, convert_to = 's'):
        if convert_to == 'seconds':
            return val*3.154e+7
        elif convert_to == 'minutes':
            return val*525600.
        elif convert_to == 'hours':
            return val*8760.
        elif convert_to == 'days':
            return val*365.2422
        else:
            fx.error(KeyError, 'Invalid Conversion Unit <%s>'%(convert_to))

    def convert_solar_masses(self, val, convert_to = 'kg'):
        if convert_to == 'kg':
            return val*1.99e30
        elif convert_to == 'earth masses':
            return val*332946.
        else:
            fx.error(KeyError, 'Invalid Conversion Unit <%s>'%(convert_to))

class Solar_System(object):

    def __init__(self, dt = 5e-7, frames_max = 100000, seed = 45355):
        self.numerical_complete = False
        self.analytical_complete = False
        self.seed = seed
        self.sun = Sun(seed = seed)
        planet_data, self.planet_order = fx.get_planet_data(seed = self.seed, return_order = True)
        self.planets = {}

        self.dt = dt
        self.frames_max = frames_max

        for planet in self.planet_order:
            self.planets[planet] = Planet(name = planet, dt = self.dt,
            frames_max = self.frames_max, seed = self.seed)

        self.number_of_planets = len(self.planet_order)
        self.system = A2000(self.seed)

    #CALCULATIONS

    def get_analytical_orbits(self):
        for p in self.planets:
            self.planets[p].get_analytical_orbit()
        self.analytical_complete = True

    def get_numerical_orbits(self):
        print "Calculating Orbits for %d Planets:"\
        %(self.number_of_planets)
        for n,p in enumerate(self.planets):
            t0 = time.time()
            sys.stdout.write("[%d/%d]\tPlanet %s"%(n+1,self.number_of_planets,
            self.planets[p].name))
            self.planets[p].get_numerical_orbit()
            sys.stdout.flush()
            print ", Done - %.2fs"%(time.time()-t0)
        self.numerical_complete = True

    def get_numerical_orbits_custom(self, T = None, dt = None, frames_max = None):
        if dt == None:
            dt = self.dt
        if frames_max == None:
            frames_max = self.frames_max
        if T == 'min' or T == None:
            T = self.get_min(parameter = 'T', only_return_value = True)
        elif T == 'max':
            T = self.get_max(parameter = 'T', only_return_value = True)

        frames = min(frames_max, int(T/dt))

        orbits = {}
        print "Calculating Custom Orbits for %d Planets (Total Loops: %g):"\
        %(self.number_of_planets, self.number_of_planets*(T/dt))
        for n,p in enumerate(self.planet_order):
            t0 = time.time()
            sys.stdout.write("[%d/%d]\tPlanet %s"%(n+1,self.number_of_planets,
            self.planets[p].name))
            sys.stdout.flush()
            orbits[p] = self.planets[p].get_numerical_orbit(T = T, dt = dt,
            frames = frames, return_vals = True)
            print ", Done - %.2fs"%(time.time()-t0)

        return orbits

    #DATA VISUALIZATION

    def plot(self, T = None, dt = None, frames_max = None):
        legend = ['Sun']
        sun = plt.Circle(xy = (0.,0.), radius = self.sun.radius*6.68459e-9,
        color = 'y')
        fig, ax = plt.subplots()
        ax.set(aspect=1)
        ax.add_artist(sun)
        plt.plot(0,0,'oy',ms=1)
        if T == None:
            self._check_analytical()
            for p in self.planet_order:
                x_analytical = self.planets[p].analytical['x']
                plt.plot(x_analytical[0], x_analytical[1])
                legend.append(self.planets[p].name)
        else:
            if dt == None:
                dt = self.dt
            if frames_max == None:
                frames_max = self.frames_max
            orbits = self.get_numerical_orbits_custom(T = T, dt = dt,
            frames_max = frames_max)
            for p in self.planet_order:
                x_numerical = orbits[p]['x']
                plt.plot(x_numerical[:,0], x_numerical[:,1])
                legend.append(self.planets[p].name)
        for p,d in self.planets.iteritems():
            planet = plt.Circle(xy = (d.x0,d.y0), radius = d.radius*6.68459e-9)
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

    def __call__(self, parameter):
        op = {}
        for n,p in self.planets.iteritems():
            op[n] = p.__dict__[parameter]
        return op

    def get_max(self, parameter, only_return_value = False):
        op = self.__call__(parameter)
        highest_value = max(op.values())
        if only_return_value == True:
            return highest_value
        all_max = {}
        for n,d in op.iteritems():
            if d == highest_value:
                all_max[n] = d
        return all_max

    def get_min(self, parameter, only_return_value = False):
        op = self.__call__(parameter)
        lowest_value = min(op.values())
        if only_return_value == True:
            return lowest_value
        all_min = {}
        for n,d in op.iteritems():
            if d == lowest_value:
                all_min[n] = d
        return all_min

    def get_ordered_list(self, names = False):
        ordered = []
        for p in self.planet_order:
            if names == True:
                ordered.append((self.planets[p].name,self.planets[p]))
            else:
                ordered.append(self.planets[p])
        return ordered

    def save_XML(self, T = None, dt = None, frames_max = None, save = True):
        if dt == None:
            dt = self.dt
        if frames_max == None:
            frames_max = self.frames_max
        if T == None:
            p_0 = self.planet_order[0]
            T = 21.*self.planets[p_0].T

        numerical_orbit_data = self.get_numerical_orbits_custom(T = T, dt = dt,
        frames_max = frames_max)
        x = np.zeros([2, self.number_of_planets,
        len(numerical_orbit_data[self.planet_order[0]]['x'][:,0])])

        for n, name in enumerate(self.planet_order):
            t = numerical_orbit_data[name]['t']
            x[0,n] = numerical_orbit_data[name]['x'][:,0]
            x[1,n] = numerical_orbit_data[name]['x'][:,1]
        if save == True:
            self.system.orbit_xml(x,t)
        return x, t

    def confirm_planet_positions(self):
        '''For a dt = 5e-7, takes approximately 54s per planet, or 7.2 mins
        With @jit, takes approximately 3.6s per planet, or 30s,
        The biggest relative deviation was detected at planet 7,
        which drifted 0.3873 percent from its actual position '''
        x, t = self.save_XML(save = True)
        frames = x.shape[2]
        self.system.check_planet_positions(x, t[-1], frames/t[-1])

class Sun(object):

    def __init__(self, seed = 45355):
        data = fx.get_sun_data(seed = seed)
        self.mass = data['mass']
        self.radius = data['radius']
        self.temperature = data['temperature']
        self.sigma = 5.6703e-8
        self.L = self.sigma*(self.temperature**4.)*(4.*np.pi*((self.radius*1e3)**2.))

class Gas_Box(object):

    def __init__(self, temperature = 1e4, time = 1e-9, steps = 1e3, L = 1e-6,
    nozzle = None, number_of_particles = 1e5, particle_mass = 3.3474472e-27,
    seed = 45355):

        #CONSTANTS
        self.k = 1.38064852e-23                            #Boltzmann's Constant

        #PHYSICAL VARIABLES
        self.T = float(temperature)                       #Temperature in Kelvin
        self.L = float(L)                                  #Box Length in meters
        self.N = int(number_of_particles)                   #Number of particles
        self.m = particle_mass                #Mass of individual particle in kg

        if nozzle == None:
            nozzle = self.L/2.

        self.nozzle = nozzle                              #Size of Rocket Nozzle

        #SIMULATION VARIABLES
        self.time = float(time)                   #Simulation Runtime in Seconds
        self.steps = int(steps)             #Number of Steps Taken in Simulation
        self.dt = self.time/self.steps        #Simulation Step Length in Seconds
        self.seed = seed
        np.random.seed(self.seed)
        self.particles_per_second, self.force_per_second = self.burn()
        self.box_mass = self.particles_per_second*self.m

    def burn(self):
        sigma = np.sqrt(self.k*self.T/self.m)
        x = np.random.uniform(low = 0., high = self.L, size = (self.N, 3))
        v = np.random.normal(loc = 0.0, scale = sigma, size = (self.N, 3))
        exiting = 0.
        low_bound = 0.25*self.L
        high_bound = 0.75*self.L
        f = 0.
        i = 0
        for i in range(self.steps):
            x += v*self.dt
            v_exiting = np.abs(v[:,2])

            collision_points = np.logical_or(np.less(x, 0), np.greater(x, self.L))
            x_exit_points = np.logical_and(np.greater(x[:,0], low_bound),
            np.less_equal(x[:,0], high_bound))
            y_exit_points = np.logical_and(np.greater(x[:,1], low_bound),
            np.less_equal(x[:,1], high_bound))

            exit_points = np.logical_and(x_exit_points, y_exit_points)
            exit_points = np.logical_and(np.less(x[:,2], 0), exit_points)
            exit_indices = np.where(exit_points == True)

            collisions_indices = np.where(collision_points == True)
            exiting += len(exit_indices[0])
            sign_matrix = np.ones_like(x)
            sign_matrix[collisions_indices] = -1.
            sign_matrix[:,2][exit_indices] = 1.

            f += (2.*np.sum(v_exiting[exit_indices])*self.m/self.dt)
            x[:,2][exit_indices] = 0.99*self.L
            v = np.multiply(v, sign_matrix)

        return exiting/self.time, f/self.steps

class Rocket(object):

    def __init__(self, T = 1200., steps = 1e4, accuracy = 1e-3, planet = None,
    gas_box = None, payload_mass = 1e3, seed = 45355):
        self.seed = seed
        print 'Countdown Has Begun:\n'
        sys.stdout.write('[5] Simulating Gas Box')
        sys.stdout.flush()
        t0 = time.time()
        if isinstance(gas_box, Gas_Box):
            self.gas_box = gas_box
        elif gas_box == None:
            self.gas_box = Gas_Box(seed = self.seed)
        else:
            fx._error(TypeError, 'Argument <gas_box> must be of type <Gas_Box>')
        t1 = time.time()
        print ' - %.2fs'%(t1-t0)
        sys.stdout.write('[4] Initializing Planet and Variables')
        sys.stdout.flush()
        if isinstance(planet, Planet):
            self.planet = planet
        elif planet == None:
            self.planet = Planet(name = 'Sarplo', seed = self.seed)
        else:
            self.planet = Planet(seed = self.seed, name = planet)
        self.box_mass = self.gas_box.box_mass
        self.box_force = self.gas_box.force_per_second
        self.T = float(T)
        self.steps = int(steps)
        self.accuracy = float(accuracy)
        self.payload_mass = float(payload_mass)
        t2 = time.time()
        print ' -  %.2fs'%(t2-t1)
        sys.stdout.write('[3] Determining Ideal Number of Chambers')
        sys.stdout.flush()
        self.chambers = self.get_gas_boxes_required()
        t3 = time.time()
        print ' -  %.2fs'%(t3-t2)
        sys.stdout.write('[2] Getting Initial Velocities and Positions')
        sys.stdout.flush()
        self.x0, self.v0 = self.get_initial_conditions()
        t4 = time.time()
        print ' -  %.2fs'%(t4-t3)
        sys.stdout.write('[1] Launching Heavy Lifter')
        sys.stdout.flush()
        self.t, self.x, self.v = self.launch_lifter(self.x0, self.v0)
        t5 = time.time()
        print ' -  %.2fs'%(t5-t4)

    def get_gas_boxes_required(self):
        chambers = 1.
        step = 1e15
        v = 0.
        x = 0.
        dt = self.T/float(self.steps)
        while abs(self.planet.get_escape_velocity(x) - v) > 1e-4:
            while True:
                mass = (chambers*self.box_mass*self.T) + self.payload_mass
                dm = chambers*self.box_mass*dt
                force = chambers*self.box_force
                x = 0
                v = 0
                for i in range(self.steps-1):
                    mass -= dm
                    a = (force + self.planet.get_gravitational_force(x))/mass
                    v += a*dt
                    x += v*dt
                if v > self.planet.get_escape_velocity(x):
                    break
                chambers += step
            chambers -= step
            step /= 1e2
        return chambers

    def get_initial_conditions(self):
        x = np.array([self.planet.convert_AU(self.planet.x0),
        self.planet.convert_AU(self.planet.y0)])
        v_rot = (self.planet.radius*1000.*2.*np.pi)/(self.planet.period*24.*60.*60.)
        v = np.array([self.planet.convert_AU_per_year(self.planet.vx0),
        self.planet.convert_AU_per_year(self.planet.vy0) + v_rot])
        return x, v

    def launch_lifter(self, x0 = None, v0 = None, u = None):
        if x0 is None:
            x0 = np.zeros(2)
        if v0 is None:
            v0 = np.zeros(2)
        if u is None:
            u = np.array([0.,1.])
        dt = self.T/float(self.steps)
        mass = (self.chambers*self.box_mass*self.T) + self.payload_mass
        dm = self.chambers*self.box_mass*dt
        F = self.chambers*self.box_force*u
        x = np.zeros((self.steps, 2))
        v = np.copy(x)
        t = np.linspace(0, self.T, self.steps)
        x[0] = x0
        v[0] = v0
        g = np.zeros(2)
        for i in range(self.steps-1):
            mass -= dm
            g[1] = self.planet.get_gravitational_force(x[i][1])
            a = F*(1./mass) + g
            v[i+1] = v[i] + a*dt
            x[i+1] = x[i] + v[i+1]*dt
        return t, x, v

    def __str__(self):
        string = 'Rocket Stats:\n'
        string += '\tSeed: %d\n\tPlanet Name: %s\n\tPlanet Index: %d'%(self.seed,
        self.planet.name, self.planet.index)

class Satellite(object):

    def __init__(self, mass = 1100., cs_area = 15.):
        self.mass = mass
        self.cs_area = cs_area

class Lander(object):

    def __init__(self, planet, mass = 90., cs_area = 6.2, watts = 40.,
    panel_efficiency = .12, seed = 45355):
        self.seed = seed
        self.mass = mass
        self.cs_area = cs_area
        self.watts = watts
        self.panel_efficiency = panel_efficiency
        self.sun = Sun(seed = self.seed)
        self.planet = Planet(name = planet, seed = self.seed)

    def get_min_panel_size(self):
        return self.watts/(self.planet.F*self.panel_efficiency)

class Gaussian(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self._getData()

    def __call__(self, x):
        return (1./(np.sqrt(2.*np.pi)*self.sigma))*np.exp(-0.5*(((x - self.mu)/self.sigma))**2.)

    def _getData(self):
        self.FWHM = 2.355*self.sigma
        self.maximum = self.__call__(self.mu)

    def point_probability(self, x, dx = 1e-2, dt = 2e-3):
        x = [float(x)-dx, float(x)+dx]
        return integrate_function_vectorized(f = self.__call__, a = x[0], b = x[1], dt = dt)

    def probability(self, a, b, dt = 2e-3, sigma_factor = 5.):
        infinity = sigma_factor * self.sigma
        if isinstance(a, types['text']) and a.lower() in inf_str:
            a = self.mu - infinity
        elif not isinstance(a, types['numbers']):
            _error(errortype = ValueError, msg = 'Invalid Type for Integration Bounds')
        if isinstance(b, types['text']) and b.lower() in inf_str:
            b = self.mu + infinity
        elif not isinstance(b, types['numbers']):
            _error(errortype = ValueError, msg = 'Invalid Type for Integration Bounds')
        return integrate_function_vectorized(f = self.__call__, a = a, b = b, dt = dt)

    def plot(self, title = 'Gaussian Plot', xlabel = 'Possible Outcomes', ylabel = 'Possible Outcome Density',
             sigma_factor = 4, dt = 2e-3):
        a = self.mu - sigma_factor * self.sigma
        b = self.mu + sigma_factor * self.sigma
        x = np.linspace(a, b, (b-a)/dt)
        plt.plot(x, self.__call__(x))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis([a, b, 0, 1.1*self.maximum])
        plt.show()

if __name__ == '__main__':
    r = Rocket(seed = 45355, planet = 'sarplo')
