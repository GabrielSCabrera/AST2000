import sympy.geometry as GEO
import numpy as np

class Satellite(object):

    def __init__(self, mass = 1100., cs_area = 15., seed = 45355):
        self.mass = mass
        self.cs_area = cs_area
        self.solar_system = Solar_System()
        self.seed = seed

    def take_pictures(self, a_phi = 70, a_theta = 70, theta0 = (np.pi/2)):
        a_phi = np.deg2rad(a_phi)
        a_theta = np.deg2rad(a_theta)
        inFile = open('himmelkule.npy', 'rb')
        himmelkulen = np.load(inFile)
        inFile.close()
        x_max = (2*np.sin(a_phi/2))/(1+np.cos(a_phi/2))
        x_min = -(2*np.sin(a_phi/2))/(1+np.cos(a_phi/2))
        y_max = (2*np.sin(a_theta/2))/(1+np.cos(a_theta/2))
        y_min = -(2*np.sin(a_theta/2))/(1+np.cos(a_theta/2))
        x = np.linspace(x_min,x_max,640)
        y = np.linspace(y_max,y_min,480)
        X, Y = np.meshgrid(x,y)
        XY = np.zeros((480,640,2))
        XY[:,:,0] = X; XY[:,:,1] = Y
        projections = np.zeros((360,480,640,3),dtype = np.uint8)
        for j in range(359):
            phi0 = np.deg2rad(j)
            rho = np.sqrt(X**2 + Y**2)
            c = 2*np.arctan(rho/2)
            theta = np.pi/2 - np.arcsin(np.cos(c)*np.cos(theta0) + Y*np.sin(c)*np.sin(theta0)/rho)
            phi = phi0 + np.arctan(X*np.sin(c)/(rho*np.sin(theta0)*np.cos(c) - Y*np.cos(theta0)*np.sin(c)))
            for n,(i,v) in enumerate(zip(theta, phi)):
                for m,(k,w) in enumerate(zip(i,v)):
                    pixnum = A2000.ang2pix(k,w)
                    temp = himmelkulen[pixnum]
                    projections[j][n][m] = (temp[2], temp[3], temp[4])
        return projections

    def get_orientation_phi(self, picture):
        projections = take_pictures()
        fit = np.zeros(360)
        for i in range(359):
            fit[i] = np.sum((projections[i] - picture)**2)
        phi = np.where(fit==min(fit))
        return phi

    def get_velocity(self, l = 656.3):
        v_refstar1, phi_1, v_refstar2, phi_2 = fx.get_vel_from_ref_stars(l, seed = None)
        rm = np.matrix([[np.sin(phi_2), -np.sin(phi_1)], [-np.cos(phi_2), np.cos(phi_1)]])
        vm = np.matrix([[v_refstar1], [v_refstar2]])
        vxy = (1./np.sin(phi_2 - phi_1))*rm*vm
        return vxy

    def get_position_from_dist(self, dist_list,time):
        number_of_planets = self.solar_system.number_of_planets
        if self.seed == 45355:
            planets = ['sarplo', 'jevelan', 'calimno', 'sesena', 'corvee', 'bertela',
            'poppengo', 'trento']
        else:
            planets = list(string.ascii_lowercase[:number_of_planets])
        if % 2 == 0:
            continue
        else:
            del planets[-1]
        circles = []
        for i,n in enumerate(planets):
            circles.append(GEO.Circle(GEO.Point(self.solar_system.orbits[n](time)),dist_list[i]))
        intersections = np.zeros((len(circles),2))
        for i in xrange(0,len(circles),2):
            inter = np.array(GEO.intersection(circles[i],circles[i+1]))
            for k in xrange(2):
                intersections[i+k] = inter[k]
        fit = []
        for i in xrange(len(intersections)):
            for k in xrange(len(intersections)):
                if k <= i:
                    continue
                fit.append(np.sum((intersections[i] - intersections[k])**2))
        return intersections[np.where(fit==min(fit))]
