from ast2000solarsystem_27_v4 import AST2000SolarSystem as A2000
import functions as fx
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import time, sys, os

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
        self.sun = fx.get_sun_data(seed = self.seed)
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

        #CALCULATED DATA
        self.b = self.a*np.sqrt(1 - self.e**2)
        self.mu_hat = (self.mass*self.sun['mass'])/(self.mass+self.sun['mass'])
        self.T = 2.*np.pi*np.sqrt((self.a**3.)/(self.G*self.sun['mass']))

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

        t = np.linspace(0., T, frames)
        x = np.zeros((frames, 2))
        v = x.copy()
        a = x.copy()

        def get_acceleration(r):
            r_magnitude = LA.norm(r)
            ur = np.divide(r, r_magnitude)
            a = -(self.G*self.sun['mass'])/(r_magnitude**2.)
            return a*ur

        x[0,0], x[0,1] = self.x0, self.y0
        v[0,0], v[0,1] = self.vx0, self.vy0
        a[0] = get_acceleration(x[0])
        t_now = dt
        x_now = x[0].copy()
        v_now = v[0].copy()# + 0.5*get_acceleration(x_now)*dt
        a_now = a[0].copy()
        save_index = 1
        i = 0

        while t_now <= T + dt:
            a_now = get_acceleration(x_now)
            v_now += a_now*dt
            x_now += v_now*dt

            if t_now >= t[save_index]:
                x[save_index] = x_now
                v[save_index] = v_now
                a[save_index] = a_now
                save_index += 1
            t_now += dt
            i += 1

        if return_vals == False:
            self.numerical = {'t': t, 'x': x, 'v': v, 'a': a}
            self.numerical_complete = True
        else:
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
        sun = plt.Circle(xy = (0.,0.), radius = self.sun['radius']*6.68459e-9,
        color = 'y')
        fig, ax = plt.subplots()
        ax.set(aspect=1)
        ax.add_artist(sun)
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
        self.sun = fx.get_sun_data(seed = self.seed)
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
        for p in self.planets:
            self.planets[p].get_numerical_orbit()
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
        sun = plt.Circle(xy = (0.,0.), radius = self.sun['radius']*6.68459e-9,
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
        The biggest relative deviation was detected at planet 7,
        which drifted 0.3873 percent from its actual position '''
        x, t = self.save_XML(save = True)
        frames = x.shape[2]
        self.system.check_planet_positions(x, t[-1], frames/t[-1])

class Body(object):

    def __init__(self, mass, cs_area):
        self.mass = mass
        self.cs_area = cs_area

class Satellite(Body):

    def __init__(self, mass = 1100., cs_area = 15.):
        Body.__init__(self, mass = mass, cs_area = cs_area)

class Lander(Body):

    def __init__(self, mass = 90., cs_area = 6.2):
        Body.__init__(self, mass = mass, cs_area = cs_area)

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
    s = Solar_System(dt = 5e-7)
    s.plot()
