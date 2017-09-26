import numpy as np
import os, sys, time
from classes import Planet

class Gas_Box(object):

    def __init__(self, box_length = 1e-6, number_of_particles = 1e5, temperature = 1e4,
    particle_mass = 3.3474472e-27, box_step = 1e-12, box_time = 1e-9, payload_mass = 1000.,
    burn_time = 1200., steps_final = 1e10, seed = 314159265, debug = False):
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
        self.planet = Planet(name = "Sarplo")
        np.random.seed(seed = seed)
        self.x0 = self.random_positions()
        self.v0, self.sigma = self.random_velocities()
        self.v0_abs = np.sqrt(np.sum(np.power(self.v0, 2.), axis = 1))
        if debug == True:
            self._debug()
        else:
            t0 = time.time()
            print "[1] Integrating Gas Box"
            self.exiting_particles, self.exiting_velocities = self.integrate_particles_fast()
            t1 = time.time()
            print " - %.2fs"%(t1-t0)
            print "[2] Analyzing Data"
            self.prep_rocket()
            t2 = time.time()
            print " - %.2fs"%(t2-t1)
            print "[3] Launch Rocket"
            self.x_end, self.v_end, self.mass_end =\
            self.launch_rocket(steps = 1e6)
            t3 = time.time()
            print " - %.2fs"%(t3-t2)
            print "Done\n"
            print self.print_data()

    def print_data(self):
        string = "Single Chamber Data:"
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
        return string

    def prep_rocket(self):
        self.delta_p = self.exiting_velocities*self.m_particle
        self.chambers_needed, self.particles_needed, self.volume_needed_liters,\
        self.mass_needed, self.escape_velocity = self.escape_atmosphere()
        self.numerical_force = self.get_numerical_force(chambers = self.chambers_needed)
        self.sim_time = self.burn_time/self.box_time
        self.gravity_losses = self.planet.get_radial_gravitational_acceleration()

    def _debug(self):
        t0 = time.time()
        print 'Running Integration Loops...\n[1] Closed Box'
        self.run_closed_simulation(t_end = self.box_time, dt = self.box_step)
        t1 = time.time()
        print '[2] Open Rocket'
        t2, t3 = self.run_rocket_simulation_debug(t_end = self.box_time, dt = self.box_step)
        print 'Done, data calculated:\n'
        string = 'Calculation times: %.2fs for closed box, %.2fs for rocket chamber, %.2fs for launch'\
        %(t1-t0, t2-t1, t3-t2)
        string += '\n\nWall Hits: %g for closed box, %g for rocket'\
        %(self.closed_collisions, self.rocket_collisions)
        string += '\nTotal Velocity of Particles Hitting Wall: %gm/s for closed box, %gm/s for rocket chamber'\
        %(self.closed_v_collisions, self.rocket_v_collisions)
        string += '\nTotal Particle Momentum Per Wall: %gkgm/s for closed box, %gkgm/s for rocket chamber'\
        %(self.closed_p_wall, self.rocket_p_wall)
        string += '\n\nTotal Pressure in Chamber: %gpa for closed box, %gpa for rocket chamber, %gpa analytically'\
        %(self.closed_P, self.rocket_P, self.p_analytical)
        string += '\n\nTotal Number of Particles Exiting One Chamber: %g'%(self.exiting_particles)
        string += '\nTotal Velocity of Particles Exiting Chamber: %gm/s'%(self.v_exiting_particles)
        string += '\nTotal Momentum of Particles Exiting One Chamber: %gkgm/s'%(self.delta_p)
        string += '\n\nSingle Chamber Delta-V during 1e-9s w/ 1000kg: %g'\
        %(self.get_delta_v(chambers = 1.))
        string += '\nEscape Velocity to Escape Sarplo: %fm/s'%(self.escape_velocity)
        string += '\nRequired Number of Chambers to Escape Sarplo: %g'%(self.chambers_needed)
        string += '\nRequired Number of Particles to Escape Sarplo: %g'%(self.escape_required_particles)
        string += '\nFuel Mass Required: %gkg'%(self.mass_H2)
        string += '\nIf Particles == H2, Required Volume of Fuel: %gl'%(self.required_volume_H2)
        string += '\n\nGravity Losses: %gm/s'%(self.gravity_losses)
        string += '\n\nSingle Thruster Force: Numerical %gN, Analytical %gN'\
        %(self.numerical_force/self.chambers_needed, self.analytical_force/self.chambers_needed)
        string += '\nTotal Thruster Force: Numerical %gN, Analytical %gN'\
        %(self.numerical_force, self.analytical_force)
        string += '\n\nFinal Velocity of Rocket: %gm/s'%(self.true_final_velocity)
        string += '\n\nFinal Velocity: %gm/s, Final Position: %gm, Final Mass: %gkg'\
        %(self.v, self.x, self.final_mass)
        print string

    def run_closed_simulation(self, t_end = 1e-9, dt = 1e-12):
        self.closed_collisions, self.closed_v_collisions = \
        self.integrate_particles_debug(closed_box = True, t_end = t_end, dt = dt)

        self.closed_p_wall = self.wall_momentum(self.closed_v_collisions)
        self.closed_P = self.pressure(self.closed_p_wall)

    def run_rocket_simulation_debug(self, burn_time = 1200., t_end = 1e-9, dt = 1e-12, steps_final = 1e4):
        self.planet = Planet(name = 'Sarplo')
        self.exiting_particles, self.v_exiting_particles, self.rocket_collisions, self.rocket_v_collisions = \
        self.integrate_particles_debug(closed_box = False, t_end = t_end, dt = dt)

        t2 = time.time()
        self.rocket_p_wall = self.wall_momentum(self.rocket_v_collisions)
        self.rocket_P = self.pressure(self.rocket_p_wall)
        self.delta_p = self.m_particle*self.v_exiting_particles
        self.chambers_needed, self.escape_required_particles, self.required_volume_H2, \
        self.mass_H2, self.escape_velocity = self.escape_atmosphere()

        self.chambers_needed = int(self.chambers_needed)+1
        self.gravity_losses = self.planet.radial_gravitational_acceleration
        self.true_final_velocity = self.get_delta_v(chambers =\
        self.chambers_needed) + self.gravity_losses

        self.analytical_force = self.get_analytical_force()
        self.numerical_force = self.get_numerical_force()
        dt_multiplier = max((burn_time/self.steps_final)/dt, 1.)
        print '[3] Launching Rocket'
        self.x, self.v, self.final_mass = self.launch_rocket(dt_multiplier = dt_multiplier)
        t3 = time.time()
        return t2, t3

    def random_positions(self):
        return np.random.uniform(low = 0., high = self.L, size = (self.N, 3))

    def random_velocities(self):
        sigma = np.sqrt(self.k*self.T/self.m_particle)
        return np.random.normal(loc = 0.0, scale = sigma, size = (self.N, 3)), sigma

    def integrate_particles_debug(self, closed_box = True, t_end = 1e-9, dt = 1e-12):
        x, v = self.x0, self.v0
        if closed_box == True:
            closed_collisions = 0.
            closed_collisions_velocity = 0.
            v_collisions = np.zeros_like(v[:,2])
            for n in range(int(t_end/dt)):
                x += v*dt
                v_collisions = np.abs(v)
                collision_points = np.logical_or(np.less_equal(x, 0.), np.greater_equal(x, self.L))
                collisions_indices = np.where(collision_points == True)
                not_collisions_indices = np.where(collision_points == False)
                closed_collisions += len(collisions_indices[0])
                v_collisions[not_collisions_indices] = 0.
                closed_collisions_velocity += np.sum(v_collisions)
                sign_matrix = np.ones_like(x)
                sign_matrix[collisions_indices] = -1.
                v = np.multiply(v,sign_matrix)
            return closed_collisions/6., closed_collisions_velocity
        elif closed_box == False:
            rocket_collisions = 0.
            rocket_collisions_velocity = 0.
            exiting = 0.
            exiting_velocities = 0.
            for n in range(int(t_end/dt)):
                x += v*dt
                v_collisions = np.abs(v)
                v_exiting = np.abs(v[:,2])
                collision_points = np.logical_or(np.less_equal(x, 0), np.greater_equal(x, self.L))
                x_exit_points = np.logical_and(np.greater_equal(x[:,0], 0.25*self.L),
                np.less_equal(x[:,0], 0.75*self.L))

                y_exit_points = np.logical_and(np.greater_equal(x[:,1], 0.25*self.L),
                np.less_equal(x[:,1], 0.75*self.L))

                exit_points = np.logical_and(x_exit_points, y_exit_points)
                exit_points = np.logical_and(np.less_equal(x[:,2], 0), exit_points)
                exit_indices = np.where(exit_points == True)
                not_exit_indices = np.where(exit_points == False)
                v_exiting[not_exit_indices] = 0.
                v_exiting_sum = np.sum(v_exiting)
                exiting_velocities += v_exiting_sum

                collisions_indices = np.where(collision_points == True)
                not_collisions_indices = np.where(collision_points == False)
                rocket_collisions += len(collisions_indices[0])
                v_collisions[not_collisions_indices] = 0.
                rocket_collisions_velocity += np.sum(v_collisions) - v_exiting_sum
                exiting += len(exit_indices[0])
                sign_matrix = np.ones_like(x)
                sign_matrix[collisions_indices] = -1.
                sign_matrix[:,2][exit_indices] = 1.
                refill_matrix = np.zeros_like(x)
                x[:,2][exit_indices] += 0.99*self.L
                v = np.multiply(v,sign_matrix)
            return exiting, exiting_velocities, rocket_collisions/6., rocket_collisions_velocity

    def integrate_particles_fast(self):
        x, v = self.x0, self.v0
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

    def launch_rocket(self, steps = 1e6):
        mass_per_iteration = self.exiting_particles*self.chambers_needed*self.m_particle
        dt = self.burn_time/steps
        mass = mass_per_iteration*(dt/self.box_time) + self.m_payload
        v = 0.
        x = 0.
        n = 0
        while n < self.burn_time/dt:
            a = self.numerical_force/mass + self.gravity_losses
            v += a*dt
            x += v*dt
            mass -= mass_per_iteration
            n += 1
        return x, v, mass

    def wall_momentum(self, v_collisions):
        return 2*self.m_particle*v_collisions

    def pressure(self, p_wall):
        f = p_wall/(self.box_time*6.)
        return f/(self.L**2.)

    def get_delta_v(self, chambers = 1.):
        return (chambers*self.delta_p)/self.m_payload

    def escape_atmosphere(self):
        escape_velocity = self.planet.escape_velocity
        chambers_needed = (escape_velocity/(self.get_delta_v()*(self.burn_time/self.box_time)))
        particles_needed = chambers_needed*self.exiting_particles*(self.burn_time/self.box_time)
        volume_needed_liters = (self.m_particle*particles_needed)/.0708
        mass_needed = particles_needed*self.m_particle
        return chambers_needed, particles_needed, volume_needed_liters, mass_needed, escape_velocity

    def get_analytical_force(self):
        return self.chambers_needed*self.rocket_P*(self.L/2.)**2.

    def get_numerical_force(self, chambers = 1.):
        return self.get_delta_v(chambers = chambers)*(self.burn_time/self.box_time)

    def tests(self):
        def mean_kinetic_energy(self):
            K = 0.5*self.m_particle*np.power(self.v0_abs, 2.)
            mean_goal = (3./2.)*self.k*self.T
            true_mean = np.sum(K)/self.N
            return mean_goal, true_mean, mean_goal-true_mean

        def mean_absolute_velocity_integral(self, sigma_factor = 5):
            infty = self.sigma*sigma_factor
            return self.integrate_function_vectorized(f = self.velocity_probability_integrand,
            a = 0., b = infty)

        def mean_absolute_velocity(self):
            return np.sum(self.v0_abs)/self.N

        def integrate_function_vectorized(self, f, a, b, dt = 2e-3):
            x = np.linspace(float(a), float(b), (b-a)/dt)
            x_step_low = dt*f(x[1:])
            x_step_high = dt*f(x[:-1])
            return np.sum((x_step_low + x_step_high)/2.)

        def velocity_probability_integrand(self, v):
            a = np.power((self.m_particle/(2.*np.pi*self.k*self.T)), (3./2.))
            b = np.exp(-0.5*(self.m_particle*np.power(v,2.)/(self.k*self.T)))
            c = 4*np.pi*np.power(v,2.)
            return v*a*b*c

if __name__ == '__main__':
    a = Gas_Box(number_of_particles = 100000, steps_final = 1e9)
