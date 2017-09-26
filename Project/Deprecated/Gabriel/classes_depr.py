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

class Planet(object):

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
