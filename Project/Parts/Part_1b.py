import numpy as np
import os, sys, time
from classes import Planet
from numpy import linalg as LA

class Rocket(object):

    def __init__(self, box_length = 1e-6, number_of_particles = 1e5, temperature = 1e4,
    particle_mass = 3.3474472e-27, box_step = 1e-12, box_time = 1e-9, payload_mass = 1000.,
    burn_time = 1200., steps_final = 1e10, seed = None, orbital_angle = 0.,
    debug = False, final_run = False):
        if seed == None:
            self.planet_name = "Sarplo"
        else:
            self.planet_name = "A"
        np.random.seed(seed = seed)
        self.L = float(box_length)
        self.N = int(number_of_particles)
        self.T = float(temperature)
        self.m_particle = float(particle_mass)
        self.m_payload = float(payload_mass)
        self.k = 1.38064852e-23
        self.p_analytical = self.N*self.k*self.T/(self.L**3.)
        self.box_step = box_step
        self.box_time = box_time
        self.burn_time = burn_time
        self.steps_final = steps_final
        self.planet = Planet(name = self.planet_name, seed = seed)
        self.orb_angle = orbital_angle
        self.sun_mass = self.planet.sun_data['mass']
        self.main()

    def main(self):
        t0 = time.time()
        sys.stdout.write("[1] Integrating Gas Box")
        sys.stdout.flush()
        self.exiting_particles, self.exiting_velocities = self.integrate_particles()
        t1 = time.time()
        print " - %.2fs"%(t1-t0)
        sys.stdout.write("[2] Analyzing Data")
        sys.stdout.flush()
        self.prep_rocket()
        t2 = time.time()
        print " - %.2fs"%(t2-t1)
        sys.stdout.write("[3] Launch Rocket")
        sys.stdout.flush()
        self.x_end, self.v_end, self.mass_end =\
        self.launch_rocket(chambers = self.chambers_needed, steps = 1e6)
        t3 = time.time()
        print " - %.2fs"%(t3-t2)
        sys.stdout.write("[4] Finding Realistic Values")
        self.find_max_chambers()
        t4 = time.time()
        print " - %.2fs"%(t4-t3)
        sys.stdout.write("[5] Relaunching Rocket With New Values")
        self.x_final, self.v_final, self.mass_final =\
        self.launch_rocket(chambers = self.max_chambers, steps = 1e6)
        t5 = time.time()
        print " - %.2fs"%(t5-t4)
        sys.stdout.write("[6] Final Vectorized Valued Launch")
        self.x_vec, self.v_vec =\
        self.launch_rocket_final(chambers = self.max_chambers, steps = 1e6)
        t6 = time.time()
        print " - %.2fs"%(t6-t5)
        print "Done\n"
        print self.print_data()

    def print_data(self):
        string = "Planet Name: %s"%(self.planet_name)
        string += "\n\nSingle Chamber Data:"
        string += "\n\tNumber of Escaping Particles Per Simulation: %g"%(self.exiting_particles)
        string += "\n\tTotal Momentum of Escaping Particles: %gkgm/s"%(self.delta_p)
        string += "\n\nMulti-Chamber Data:"
        string += "\n\tEstimated Chambers Required: %g"%(self.chambers_needed)
        string += "\n\tEstimated Fuel Mass Required: %gkg"%(self.mass_needed)
        string += "\n\tEstimated Total Rocket Force: %gN"%(self.numerical_force)
        string += "\n\tEstimated Number of Escaping Particles: %g"\
        %(self.exiting_particles*self.chambers_needed*self.sim_time)
        string += "\n\tEstimated Total Rocket Momentum Change: %gkgm/s"%\
        (self.delta_p*self.chambers_needed*self.sim_time)
        string += "\n\nFlight Details:"
        string += "\n\tEscape Velocity: %gm/s"%(self.escape_velocity)
        string += "\n\tFinal Altitude: %gm"%(self.x_end)
        string += "\n\tFinal Velocity: %gm/s"%(self.v_end)
        #string += "\n\tFinal Mass: %gkg"%(self.mass_end)
        string += "\n\nMore Accurate Results:"
        string += "\n\tMaximum Number of Chambers Required: %g"%(self.max_chambers)
        string += "\n\tMaximum Amount of Fuel Required: %gkg"\
        %(self.exiting_particles*self.m_particle*self.sim_time*self.max_chambers)
        string += "\n\nFinal Estimation Attempt:"
        string += "\n\tFinal Altitude: %gm"%(self.x_final)
        string += "\n\tFinal Velocity: %gm/s"%(self.v_final)
        string += "\n\nValues with Respect to Sun:"
        string += "\n\tStart Position: %sm"%(np.array([self.planet.x0, self.planet.y0]))
        string += "\n\tFinal Position: %sm"%(self.x_vec)
        string += "\n\tStart Velocity: %sm"%(np.array([self.planet.vx0, self.planet.vy0]))
        string += "\n\tFinal Velocity: %sm/s"%(self.v_vec)
        dX_tot = self.x_vec - np.array([self.planet.x0, self.planet.y0])
        string += "\n\tTotal Displacement: %sm"\
        %(dX_tot)
        dV_tot = self.v_vec - np.array([self.planet.vx0, self.planet.vy0])
        string += "\n\tTotal Delta V: %sm/s"\
        %(dV_tot)
        comp_disp = np.sqrt(dX_tot[0]**2. + dX_tot[1]**2.)\
        - np.sqrt(self.planet.x0**2. + self.planet.y0**2.)
        string += "\n\tComparative Displacement: %gm"%(comp_disp)
        return string

    def prep_rocket(self):
        self.delta_p = self.exiting_velocities*self.m_particle
        self.chambers_needed, self.particles_needed, self.volume_needed_liters,\
        self.mass_needed, self.escape_velocity = self.escape_atmosphere()
        self.numerical_force = self.get_numerical_force(chambers = self.chambers_needed)
        self.sim_time = self.burn_time/self.box_time

    def prep_rocket_w_chambers(self, chambers):
        self.delta_p = self.exiting_velocities*self.m_particle
        self.chambers_needed, self.particles_needed, self.volume_needed_liters,\
        self.mass_needed, self.escape_velocity = self.escape_atmosphere()
        multiplier = chambers/self.chambers_needed
        self.chambers_needed *= multiplier
        self.particles_needed *= multiplier
        self.volume_needed_liters *= multiplier
        self.mass_needed *= multiplier
        self.numerical_force = self.get_numerical_force(chambers = self.chambers_needed)
        self.sim_time = self.burn_time/self.box_time
        self.gravity_losses = self.planet.get_radial_gravitational_acceleration()

    def random_positions(self):
        return np.random.uniform(low = 0., high = self.L, size = (self.N, 3))

    def random_velocities(self):
        sigma = np.sqrt(self.k*self.T/self.m_particle)
        return np.random.normal(loc = 0.0, scale = sigma, size = (self.N, 3))

    def integrate_particles(self):
        x, v = self.random_positions(), self.random_velocities()
        exiting = 0.
        exiting_velocities = 0.
        low_bound = 0.25*self.L
        high_bound = 0.75*self.L
        for n in range(int(self.box_time/self.box_step)):
            x += v*self.box_step
            v_exiting = np.abs(v[:,2])
            collision_points = np.logical_or(np.less_equal(x, 0), np.greater_equal(x, self.L))
            x_exit_points = np.logical_and(np.greater_equal(x[:,0], low_bound),
            np.less_equal(x[:,0], high_bound))
            y_exit_points = np.logical_and(np.greater_equal(x[:,1], low_bound),
            np.less_equal(x[:,1], high_bound))

            exit_points = np.logical_and(x_exit_points, y_exit_points)
            exit_points = np.logical_and(np.less_equal(x[:,2], 0), exit_points)
            exit_indices = np.where(exit_points == True)
            not_exit_indices = np.where(exit_points == False)
            v_exiting[not_exit_indices] = 0.
            exiting_velocities += np.sum(v_exiting)

            collisions_indices = np.where(collision_points == True)
            exiting += len(exit_indices[0])
            sign_matrix = np.ones_like(x)
            sign_matrix[collisions_indices] = -1.
            sign_matrix[:,2][exit_indices] = 1.
            refill_matrix = np.zeros_like(x)
            x[:,2][exit_indices] += 0.99*self.L
            v = np.multiply(v,sign_matrix)
        return exiting, exiting_velocities

    def integrate_particles_light(self, T):
        sigma = np.sqrt(self.k*T/self.m_particle)
        x = np.random.uniform(low = 0., high = self.L, size = (1000, 3))
        v = np.random.normal(loc = 0.0, scale = sigma, size = (1000, 3))
        exiting = 0.
        exiting_velocities = 0.
        low_bound = 0.25*self.L
        high_bound = 0.75*self.L
        for n in range(int(self.box_time/self.box_step)):
            x += v*self.box_step
            v_exiting = np.abs(v[:,2])
            collision_points = np.logical_or(np.less_equal(x, 0), np.greater_equal(x, self.L))
            x_exit_points = np.logical_and(np.greater_equal(x[:,0], low_bound),
            np.less_equal(x[:,0], high_bound))
            y_exit_points = np.logical_and(np.greater_equal(x[:,1], low_bound),
            np.less_equal(x[:,1], high_bound))

            exit_points = np.logical_and(x_exit_points, y_exit_points)
            exit_points = np.logical_and(np.less_equal(x[:,2], 0), exit_points)
            exit_indices = np.where(exit_points == True)
            not_exit_indices = np.where(exit_points == False)
            v_exiting[not_exit_indices] = 0.
            exiting_velocities += np.sum(v_exiting)

            collisions_indices = np.where(collision_points == True)
            exiting += len(exit_indices[0])
            sign_matrix = np.ones_like(x)
            sign_matrix[collisions_indices] = -1.
            sign_matrix[:,2][exit_indices] = 1.
            refill_matrix = np.zeros_like(x)
            x[:,2][exit_indices] += 0.99*self.L
            v = np.multiply(v,sign_matrix)
        return exiting, exiting_velocities

    def launch_rocket(self, chambers, steps = 1e6):
        fuel_mass = self.exiting_particles*chambers*self.m_particle/self.sim_time
        mass_per_iteration = fuel_mass/steps
        dt = self.burn_time/steps
        mass = fuel_mass + self.m_payload
        x = 0.
        v = 0.
        n = 0
        while n < self.burn_time:
            force = (chambers*self.delta_p/self.box_time)
            a = force/mass + self.planet.radial_gravitational_acceleration
            v += a*dt
            x += v*dt
            mass -= mass_per_iteration
            n += dt
        return x, v, mass

    def launch_rocket_final(self, chambers, steps = 1e6):
        fuel_mass = self.exiting_particles*chambers*self.m_particle/self.sim_time
        mass_per_iteration = fuel_mass/steps
        dt = self.burn_time/steps
        mass = fuel_mass + self.m_payload
        x_p = np.array([self.planet.x0, self.planet.y0])
        x = np.array([self.planet.x0 + self.planet.radius, self.planet.y0])
        v = np.array([self.planet.vx0, self.planet.vy0 + self.planet.v_surface])
        n = 0
        while n < self.burn_time:
            force = (chambers*self.delta_p/self.box_time)
            a = np.array([0.,force/mass])\
            + self.get_gravity_acceleration(x, self.sun_mass)\
            + self.get_gravity_acceleration(x - x_p, self.planet.mass)
            v += a*dt
            x += v*dt
            mass -= mass_per_iteration
            n += dt
        return x, v

    def get_gravity_acceleration(self, r, m):
        r_magnitude = np.sqrt(r[0]**2. + r[1]**2.)
        if abs(r_magnitude) <= 1e-10:
            return np.zeros(2)
        ur = np.divide(r, r_magnitude)

        a = -self.planet.G*m
        a /= r_magnitude**2.
        return a*ur

    def wall_momentum(self, v_collisions):
        return 2*self.m_particle*v_collisions

    def get_delta_v(self, chambers = 1.):
        return (chambers*self.delta_p)/(self.m_payload)

    def escape_atmosphere(self):
        escape_velocity = self.planet.escape_velocity
        chambers_needed = (escape_velocity/(self.get_delta_v()*(self.burn_time/self.box_time)))
        particles_needed = chambers_needed*self.exiting_particles*(self.burn_time/self.box_time)
        volume_needed_liters = (self.m_particle*particles_needed)/.0708
        mass_needed = particles_needed*self.m_particle
        return chambers_needed, particles_needed, volume_needed_liters, mass_needed, escape_velocity

    def get_numerical_force(self, chambers = 1.):
        return self.get_delta_v(chambers = chambers)*(self.burn_time/self.box_time)

    def find_max_chambers(self):
        i = 1.
        step = 0.5
        v = self.v_end
        while abs(self.escape_velocity - v) > 1e-2:
            v = 0
            while v < self.escape_velocity:
                v_new = self.launch_rocket(chambers = self.chambers_needed*i, steps = 1e3)[1]
                if v_new > self.escape_velocity:
                    break
                else:
                    v = v_new
                i += step
            i -= step
            step /= 5.
        self.max_chambers = self.chambers_needed*i

if __name__ == '__main__':
    a = Rocket(seed = 28631, number_of_particles = 1e5)
