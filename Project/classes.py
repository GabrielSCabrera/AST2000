from ast2000solarsystem_27_v4 import AST2000SolarSystem as A2000
import functions as fx
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import time

'''
Steinn 52772
Simen 47566
Lars 78826
Ulrik 82275
Gabriel 45355
Anders 28631'''

class Planet_OLD(object):                           #TO BE REMOVED

    def __init__(self, name = None, seed = None, a = None, e = None, radius = None,
    omega = None, psi = None, mass = None, period = None, x0 = None, y0 = None,
    vx0 = None, vy0 = None, rho0 = None):

        properties = ['a', 'e', 'radius', 'omega', 'psi', 'mass', 'period',
         'x0', 'y0', 'vx0', 'vy0', 'rho0']
        planet_data = fx.get_planet_data(seed = seed)
        name = name.lower()
        self.solar_mass = 1.99e30
        self.AU_in_m = 1.496e11
        self.year_in_s = 3.1558e7
        if name == None:
            for planet_property in properties:
                try:
                    exec('self.%s = float(%s)'%(planet_property, planet_property))
                except TypeError:
                    fx.error(ValueError, 'Invalid <Planet> initialization argument <%s>'%(planet_property))
        elif name in planet_data:
            for planet_property in planet_data[name]:
                exec('self.%s = %s'%(planet_property, planet_data[name][planet_property]))
            self.mass *= self.solar_mass
            self.radius *= 1000.
            self.period *= 86400.
            self.a *= self.AU_in_m
            self.x0 *= self.AU_in_m
            self.y0 *= self.AU_in_m
            self.vx0 *= self.AU_in_m/self.year_in_s
            self.vy0 *= self.AU_in_m/self.year_in_s
        else:
            fx.error(NameError, 'Invalid Planet Name')
        self.circumference = 2.*self.radius*np.pi
        self.v_surface = self.circumference/self.period
        self.b = self.a*np.sqrt(1 - self.e**2)
        self.planet_name = name[0].upper() + name[1:]
        self.sun_data = fx.get_sun_data()
        self.sun_data['mass'] *= self.solar_mass
        self.sun_data['radius'] *= 1000.
        self.G = 6.673e-11
        self.escape_velocity = self.get_escape_velocity()
        self.radial_gravitational_acceleration = self.get_radial_gravitational_acceleration()
        self.mu_hat = (self.mass*self.sun_data['mass'])/(self.mass+self.sun_data['mass'])

    def get_escape_velocity(self):
        return np.sqrt((2.*self.G*self.mass)/self.radius)

    def get_radial_gravitational_acceleration(self, h = 0.):
        return -(self.G*self.mass)/((self.radius + h)**2.)

    def get_data_from_angle(self, f):
        p = self.a*(1-self.e**2)
        k = self.G*self.sun_data['mass']
        r = np.abs(p/(1.+(self.e*np.cos(f))))
        v = np.sqrt(k*((2./r)-(1./self.a)))
        a = (self.G*self.sun_data['mass'])/(r**2.)
        u = np.array([np.cos(f), np.sin(f)])
        uv = np.arctan2(self.e*np.sin(f), 1.+self.e*np.cos(f))
        uv = np.array([np.cos(uv),np.sin(uv)])
        return {'position':r*u, 'velocity':v*uv, 'acceleration':a*u}

    def integrate_data_by_time(self, t, dt, f0 = 0):
        data = self.get_data_from_angle(f = f0)
        t = np.linspace(0.,t,t/dt)
        x = np.zeros((len(t),2))
        v = x.copy()
        a = x.copy()
        x[0] = data['position']
        v[0] = data['velocity']
        a[0] = data['acceleration']
        for n,i in enumerate(t[:-1]):
            x_new = x[n] + v[n]*dt + 0.5*a[n]*(dt**2.)
            f = np.arctan2(x_new[1],x_new[0])
            data_new = self.get_data_from_angle(f)
            x[n+1] = data_new['position']
            v[n+1] = data_new['velocity']
            a[n+1] = data_new['acceleration']
        return t, x, v, a

    def plot_orbit(self, steps = 1000):
        plt.rcParams['axes.facecolor'] = 'black'
        plt.axes().set_aspect(1)
        f = np.linspace(0,2*np.pi,steps)
        r = self.get_data_from_angle(f)['position']
        axes = [1.2*np.min(r[0]),1.2*np.max(r[0]),1.2*np.min(r[1]),1.2*np.max(r[1])]
        plt.axis(axes)
        plt.plot(r[0],r[1], '-w')
        plt.plot(0,0,'oy',ms=15)
        plt.title("Orbit of Planet %s"%(self.planet_name))
        plt.show()

class Planet(object):

    def __init__(self, name, dt = 6e-7, frames_max = 10000, seed = 45355):

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

    def get_numerical_orbit(self):
        #USES LEAPFROG INTEGRATION
        t = np.linspace(0., self.T, self.frames)
        x = np.zeros((self.frames, 2))
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
        t_now = self.dt
        x_now = x[0].copy()
        v_now = v[0].copy() + 0.5*get_acceleration(x_now)*self.dt
        a_now = a[0].copy()
        save_index = 1
        i = 0

        while t_now <= self.T + self.dt:
            a_now = get_acceleration(x_now)
            v_now += a_now*self.dt
            x_now += v_now*self.dt

            if t_now >= t[save_index]:
                x[save_index] = x_now
                v[save_index] = v_now
                a[save_index] = a_now
                save_index += 1
            t_now += self.dt
            i += 1

        self.numerical = {'t': t, 'x': x, 'v': v, 'a': a}
        self.integration_complete = True

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
        string += '\n\tStarting Coordinates (x,y): (%g, %g) AU'%(self.x0, self.y0)
        string += '\n\tStarting Velocity (vx,vy): (%g, %g) AU/Yr'%(self.vx0, self.vy0)
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

class Orbits(object):

    def __init__(self, seed = 45355, dt = 6e-7, T = 3, frames = 10000):

        if frames > T/dt:
            fx.error(ValueError, 'Cannot save more frames than number of steps')

        self.dt = dt
        self.T = T
        self.frames = frames

        self.final_dt = self.T/float(self.frames)

        self.G = 4*np.pi**2.

        self.planet_data, self.planet_names = fx.get_planet_data(seed = seed, return_order = True)
        self.sun_data = fx.get_sun_data(seed = seed)
        if seed == None:
            self.seed = 45355
        else:
            self.seed = seed
        self.earth_mass = 1.989e30/5.972e24
        self.system = A2000(self.seed)
        self.numerical_orbit_data = {}
        t0 = time.time()
        self.analytical_orbit_data = self.get_analytical_orbits()
        for n, (name, planet) in enumerate(self.planet_data.iteritems()):
            t, x, v, a = self.leapfrog(planet)
            self.numerical_orbit_data[name] = {}
            self.numerical_orbit_data[name]['t'] = t
            self.numerical_orbit_data[name]['x'] = x
            self.numerical_orbit_data[name]['v'] = v
            self.numerical_orbit_data[name]['a'] = a
            print 'Planet %d of %d calculated'%(n,len(self.planet_data.keys()))
        print "\nTime Elapsed: %gs"%(time.time()-t0)

    def extract_data(self, data):
        op = {}
        for n,p in self.planet_data.iteritems():
            op[n] = p[data]
        return op

    def plot_compare(self, planet):
        legend = ['Max','Sun', 'Planet %s, Numerical'%(planet), 'Planet %s, Analytical'%(planet)]
        ncl = self.numerical_orbit_data[planet.lower()]
        acl = self.analytical_orbit_data[planet.lower()]
        psi = self.planet_data[planet]['psi']
        plt.axes().set_aspect(1)
        x = np.linspace(0,2.*np.pi, 1000)
        y = 0.002835*np.array([np.cos(x), np.sin(x)])
        plt.plot([-np.cos(psi),np.cos(psi)], [-np.sin(psi),np.sin(psi)], '-k')
        plt.plot([-np.cos(psi-np.pi/2.),np.cos(psi-np.pi/2.)], [-np.sin(psi-np.pi/2.),np.sin(psi-np.pi/2.)], '-k')
        plt.plot(y[0],y[1],'-y')
        plt.plot(ncl['x'][:,0], ncl['x'][:,1], '-r')
        plt.plot(acl[0], acl[1], '-b')
        plt.plot([-1,1],[0,0])
        plt.legend(legend)
        plt.show()

    def get_analytical_orbits(self, steps = 1e3):
        orbits = {}
        for n, d in self.planet_data.iteritems():
            d = self.planet_data[n]
            theta = np.linspace(0, 2*np.pi, steps)
            orbits[n] = self.get_analytical_position(theta, n)
        return orbits

    def get_analytical_position(self, theta, name):
        d = self.planet_data[name]
        p = d['a']*(1. - d['e']**2.)
        r = p/(1. + d['e']*np.cos(theta-d['omega']))
        ur = np.array([-np.cos(theta-(d['omega']-d['psi'])), -np.sin(theta-(d['omega']-d['psi']))])
        return np.abs(r)*ur

    def get_area(self, t0, dt, name):
        d = self.planet_data[name]
        con1 = t0 - 0.5*self.final_dt < self.numerical_orbit_data[name]['t']
        con2 = t0 + 0.5*self.final_dt > self.numerical_orbit_data[name]['t']
        t0_index = np.where(con1 & con2)

        con3 = t0+dt - 0.5*self.final_dt < self.numerical_orbit_data[name]['t']
        con4 = t0+dt + 0.5*self.final_dt > self.numerical_orbit_data[name]['t']
        t1_index = np.where(con3 & con4)

        #x_slice = self.numerical_orbit_data[name]['x'][t0_index[0][0]:t1_index[0][0]]
        #r = LA.norm(x_slice, axis = 1)

        return t0_index,t1_index[0]

    def get_max(self, data):
        op = self.extract_data(data)
        highest_value = max(op.values())
        all_max = {}
        for n,d in op.iteritems():
            if d == highest_value:
                all_max[n] = d
        return all_max

    def orbital_period(self):
        op = {}
        for n,p in self.planet_data.iteritems():
            op[n] = 2*np.pi*np.sqrt((p['a']**3.)/(self.G*self.sun_data['mass']))
        return op

    def euler_cromer(self, planet):
        t = np.linspace(0., self.T, self.frames)
        x = np.zeros((self.frames, 2))
        v = x.copy()
        a = x.copy()

        m = planet['mass']
        x[0,0], x[0,1] = planet['x0'], planet['y0']
        v[0,0], v[0,1] = planet['vx0'], planet['vy0']
        a[0] = self.get_acceleration(x[0])
        t_now = self.dt
        x_now = x[0].copy()
        v_now = v[0].copy()
        a_now = a[0].copy()
        save_index = 1
        i = 0
        while t_now <= self.T + self.dt:
            a_now = self.get_acceleration(x_now)
            v_now += a_now*self.dt
            x_now += v_now*self.dt

            if t_now >= t[save_index]:
                x[save_index] = x_now
                v[save_index] = v_now
                a[save_index] = a_now
                save_index += 1
            t_now += self.dt
            i += 1

        return t, x, v, a

    def leapfrog(self, planet):
        t = np.linspace(0., self.T, self.frames)
        x = np.zeros((self.frames, 2))
        v = x.copy()
        a = x.copy()

        x[0,0], x[0,1] = planet['x0'], planet['y0']
        v[0,0], v[0,1] = planet['vx0'], planet['vy0']
        a[0] = self.get_acceleration(x[0])
        t_now = self.dt
        x_now = x[0].copy()
        v_now = v[0].copy() + 0.5*self.get_acceleration(x_now)*self.dt
        a_now = a[0].copy()
        save_index = 1
        i = 0
        while t_now <= self.T + self.dt:
            a_now = self.get_acceleration(x_now)
            v_now += a_now*self.dt
            x_now += v_now*self.dt

            if t_now >= t[save_index]:
                x[save_index] = x_now
                v[save_index] = v_now
                a[save_index] = a_now
                save_index += 1
            t_now += self.dt
            i += 1

        return t, x, v, a

    def get_acceleration(self, r):
        r_magnitude = LA.norm(r)
        ur = np.divide(r, r_magnitude)

        a = -(self.G*self.sun_data['mass'])/(r_magnitude**2.)
        return a*ur

    def plot(self):
        legend = ['Sun']
        plt.axes().set_aspect(1)
        plt.plot(0,0,'oy',ms=10)
        for name, p in self.numerical_orbit_data.iteritems():
            legend.append(name)
            plt.plot(p['x'][:,0], p['x'][:,1])
        plt.legend(legend)
        plt.show()

    def save_XML(self, save = True):
        x = np.zeros([2, len(self.numerical_orbit_data.keys()), self.frames])

        for n, name in enumerate(self.planet_names):
            t = self.numerical_orbit_data[name]['t']
            x[0,n] = self.numerical_orbit_data[name]['x'][:,0]
            x[1,n] = self.numerical_orbit_data[name]['x'][:,1]
        if save == True:
            self.system.orbit_xml(x,t)
        return x, t

    def check_planet_positions(self):
        x, t = self.save_XML(save = True)
        print x.shape
        self.system.check_planet_positions(x, self.T, self.frames/self.T)

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
    a = Planet(name = 'Bertela', dt = 1e-4)
    print a
    print a.get_area(t0 = 0.4,dt = 0.3)-a.get_area(dt = 0.3)
