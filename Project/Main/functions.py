'''[1] MODULES'''
from ast2000solarsystem_27_v4 import AST2000SolarSystem
from PIL import Image
import sympy.geometry as GEO
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
import matplotlib.patches as Patches
import matplotlib.axes as Axes
import math, itertools, string
import numpy.linalg as LA
import ui_tools as ui
import sys, os, time
import subprocess
import ref_stars as rs
import classes as cl
try:
    from numba import jit
    import numba as nb
except ImportError:
    string = "User must install module <numba> to use module <classes.py>"
    fx.error(ImportError, string)

kwargs = sys.argv
ip = ui.get_terminal_kwargs(kwargs)
if 'get_ref_stars' in ip:
    seed = ip['get_ref_stars']
    scripts = rs.Scripts()
    scripts.get_lambda_deg_from_ref_stars()

'''[2] USEFUL CONSTANTS'''

types = {'numbers':(int, float, long),
         'text':(str,)}
inf_str = ('inf','infty','infinity')

'''[3] USEFUL FUNCTIONS'''

def error(errortype = None, msg = None):
    if errortype == None and msg == None:
        raise Exception("An unknown error occurred")
    elif isinstance(errortype,type) == True and msg == None:
        raise errortype("An unknown error occurred")
    elif isinstance(errortype,type) == False and msg == None:
        raise Exception(str(errortype))
    else:
        raise errortype(str(msg))
    sys.exit(1)

'''[4] MATHEMATICAL FUNCTIONS'''

def unit_vector(v):
    a = LA.norm(v)
    if a == 0:
        return np.zeros_like(v)
    else:
        return v/float(a)

def rotate_vector(v, theta):
    M = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return np.array([np.sum(M[0]*v), np.sum(M[1]*v)])

def binomial_coefficient(n, r):
    return math.factorial(n)/(math.factorial(r)*math.factorial(n - r))

def standard_deviation(array):
    mean = np.mean(array)
    sum_array = np.sum((array - mean)**2.)/len(array)
    return np.sqrt(sum_array)

def solve_cubic_polynomial(a, b, c, d):
    p = -b/(3.*a)
    q = p**3. + (b*c-(3.*a*d))/(6.*(a**2.))
    r = c/(3.*a)

    complex_check = (q**2.) + ((r-p**2.)**3.)

    eq_1 = (q + complex_check_1**0.5)**(1./3.)
    eq_2 = (q - complex_check_2**0.5)**(1./3.)
    eq_3 = p

    x0 = (q + complex_check_1**0.5)**(1./3.) + (q - complex_check_2**0.5)**(1./3.)

def minimize_function(f, steps, **kwargs):
    '''<f> should be a function with N arguments, <kwargs> should be N arguments
    each with a min, max value as tuple (like: a = (amin,amax), b = (bmin, bmax)
    , ...) to check between, with each argument being one of the arguments in
    v_real. <steps> should be the number of values to check for between ARGmin,
    and ARGmax.  If several values give a lower combo, this function will only
    return the first combo it finds that matches this condition.'''

    dict_order = []
    iter_list = []
    for key, val in kwargs.iteritems():
        dict_order.append(key)
        iter_list.append(list(np.linspace(val[0], val[1], steps)))
    lowest_value = None
    best_combo = None
    for i in itertools.product(*iter_list):
        test_args = dict(itertools.izip(dict_order, i))
        new_value = f(**test_args)
        if lowest_value == None or new_value < lowest_value:
            lowest_value = new_value
            best_combo = test_args
    return lowest_value, best_combo

def chi_squared(y, y_real, x):
    '''Argument <y> should be a 1D array of length N. <y_real> should be a 1D
    array of length N that contains all the precalculated y_real(x) values'''

    sigma_squared = np.power(y - y_real, 2.)

    if len(sigma_squared) != len(y) or len(y_real) != len(y):
        error(IndexError, "All arrays passed to function <chi_squared> must be \
        of equal length")

    return np.sum(np.divide(np.power(y - y_real, 2.), sigma_squared))

def minimize_chi_squared(y, y_real, x, steps, **kwargs):
    '''Argument <y> should be a 1D array of length M with noisy values. <y_real>
    should be a function with N arguments, <kwargs> should be N arguments each
    with a min, max value as tuple (like: a = (amin,amax), b = (bmin, bmax), ...)
    to check between, with each argument being one of the arguments in <y_real>.
    <steps> should be the number of values to check for between ARGmin, and
    ARGmax.  If several values give a lower combo, this function will only return
    the first combo it finds that matches this condition. <x> should be an array
    of length M that represents all the x-values.'''

    dict_order = []
    iter_list = []
    for key, val in kwargs.iteritems():
        dict_order.append(key)
        iter_list.append(list(np.linspace(val[0], val[1], steps)))
    lowest_value = None
    best_combo = None
    for i in itertools.product(*iter_list):
        test_args = dict(itertools.izip(dict_order, i))
        new_value = chi_squared(y = y, y_real = y_real(x = x, **test_args))
        if lowest_value == None or new_value < lowest_value:
            lowest_value = new_value
            best_combo = test_args
    return lowest_value, best_combo

'''[5] INTEGRATION FUNCTIONS'''

def integrate_function_vectorized(f, a, b, dt = 2e-3):
    x = np.linspace(float(a), float(b), (b-a)/dt)
    x_step_low = dt*f(x[1:])
    x_step_high = dt*f(x[:-1])
    return np.sum((x_step_low + x_step_high)/2.)

'''[6] PLOTTING FUNCTIONS'''

def sort_histogram_data(data, n = 20):
    bins = np.zeros(n+1)
    bins[:-1] = np.linspace(np.min(data), np.max(data), n)
    bins[-1] = np.max(data) + 1.
    data = np.sort(data)
    bin_index = 0
    sorted_data = np.zeros(n)
    for n,i in enumerate(data):
        while True:
            if i >= bins[bin_index] and i <= bins[bin_index + 1]:
                sorted_data[bin_index] += 1
                break
            else:
                bin_index += 1
    return bins, sorted_data

def histogram(data, title = 'Histogram', xlabel = 'x', ylabel = 'y', last_width = None):
    x,y = data
    widths = np.diff(a = x)
    fig1 = plt2.figure()
    fig1.canvas.set_window_title(title)
    ax1 = fig1.add_subplot(111, aspect = 'auto')
    ax1.axis([x[0],x[-1],0.,1.1*np.max(y)])
    for n,(i,j,k) in enumerate(zip(x,y,widths)):
        ax1.add_patch(Patches.Rectangle((i,0.), k, j))
    plt2.xlabel(xlabel)
    plt2.ylabel(ylabel)
    plt2.title(title)
    plt2.xticks(x[:-1])
    plt2.show()

'''[7] PROJECT-SPECIFIC FUNCTIONS'''

'''
Steinn 52772
Simen 47566
Lars 78826
Ulrik 82275
Gabriel 45355'''

def get_planet_data(seed = 45355, planet_names = None, return_order = False):
    if seed == 45355:
        planet_names = ['sarplo', 'jevelan', 'calimno', 'sesena', 'corvee', 'bertela',
        'poppengo', 'trento']
        myStarSystem = AST2000SolarSystem(seed)
    elif seed == 82275:
        planet_names = ['kraid', 'brinstar', 'norfair', 'ridley', 'chozo', 'phazon',
        'serris', 'phantoon']
        myStarSystem = AST2000SolarSystem(seed)
    else:
        myStarSystem = AST2000SolarSystem(seed)
        number_of_planets = myStarSystem.number_of_planets
        planet_names = list(string.ascii_lowercase[:number_of_planets])
    properties = ['a', 'e', 'radius', 'omega', 'psi', 'mass', 'period', 'x0', 'y0',
    'vx0', 'vy0', 'rho0']
    data_dict = {}
    for n,i in enumerate(planet_names):
        data_dict[i] = {}
        for p in properties:
            exec('data_dict[i][p] = myStarSystem.%s[%d]'%(p,n))
    if return_order == True:
        return data_dict, planet_names
    else:
        return data_dict

def get_sun_data(seed = None):
    if seed == None:
        seed = 45355
    myStarSystem = AST2000SolarSystem(seed)
    data_dict = {'mass':myStarSystem.star_mass,
    'radius':myStarSystem.star_radius, 'temperature':myStarSystem.temperature}
    return data_dict

def get_vel_from_ref_stars(dl3, dl4):
    c = 3e8
    l= 656.3

    cmd = ['python', 'functions.py','get_ref_stars=%d'%(seed)]
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE ).communicate()[0]
    values = []
    for i in output.split():
        try:
            float(i)
            values.append(i)
        except:
            continue
    phi_1 = np.deg2rad(float(values[0]))
    dl1 = float(values[1])
    phi_2 = np.deg2rad(float(values[2]))
    dl2 = float(values[3])

    v_refstar1 = dl1*c/l
    v_refstar2 = dl2*c/l
    v_sat_1= dl3*c/l
    v_sat_2= dl4*c/l

    v_rel_1= v_refstar1 - v_sat_1
    v_rel_2= v_refstar2 - v_sat_2

    rm= np.matrix([[np.sin(phi_2), -np.sin(phi_1)], [-np.cos(phi_2), np.cos(phi_1)]])
    vm= np.matrix([[v_rel_1], [v_rel_2]])
    vxy= ((1./np.sin(phi_2 - phi_1))*rm*vm)*0.000210945021

    print "Calculated x-component of velocity: %g" %(vxy[0])
    print "Calculated y-component of velocity: %g" %(vxy[1])


def get_orientation_phi():
    while not os.path.isfile('find_orient.png'):
        time.sleep(0.1)
    else:
        time.sleep(5)
        image= Image.open('find_orient.png')
        picture= np.array(image)
        projections = np.load('projections.npy')
        fit = np.zeros(360)
        for i in range(359):
            fit[i] = np.sum((projections[i] - picture)**2)
        phi = fit.argmin()
        print "Calculated phi = %g. Enter this value into main terminal" %(phi)
        os.remove('find_orient.png')

def get_position_from_dist():
    while not os.path.isfile('pos.npy'):
        time.sleep(0.1)
    else:
        time.sleep(5)
        planets= []
        with open('planets.txt', 'r') as infile:
            planets= infile.read().split(' ')
        coords= np.load('coords.npy')
        dist_list= np.load('pos.npy')
        circles = []
        for i,n in enumerate(planets):
            circles.append(GEO.Circle(GEO.Point(coords[i]),dist_list[i]))
        intersections = np.zeros((len(circles),2))

        for i in xrange(0,len(circles)-3,2):
            inter = np.array(GEO.intersection(circles[i],circles[i+1]))
            for k in xrange(2):
                intersections[i+k] = inter[k]

        pt1 = 0; pt2 = 0
        for i in xrange(2,len(intersections)):
            pt1 += (np.sum((intersections[0] - intersections[i])**2))
            pt2 += (np.sum((intersections[1] - intersections[i])**2))
        if pt1 < pt2:
            print "Calculated x-coordinate: %g" %(intersections[0,0])
            print "Calculated y-coordinate: %g" %(intersections[0,1])
            print "Enter these values into main terminal"
        else:
            print "Calculated x-coordinate: %g" %(intersections[1,0])
            print "Calculated y-coordinate: %g" %(intersections[1,1])
            print "Enter these values into main terminal"

        os.remove('coords.npy')
        os.remove('pos.npy')
        os.remove('planets.txt')
def get_gas_data():
    """
    Creates dictionary which contains mass of molecules and index in spectrum array for the spectral lines of each molecule
    """

    spectrum= np.load('spectrum.npy')

    pm= 1.67e-27
    m_O= 16*pm
    m_H= pm
    m_C= 12*pm

    m_O2= 2*m_O
    m_H2O= 2*m_H + m_O
    m_CO2= m_C + 2*m_O
    m_CH4= m_C + 4*m_H
    m_CO= m_O + m_C
    m_N2O= 14*pm + m_O

    O2_1= (np.abs(spectrum[:,0] - 630)).argmin()
    O2_2= (np.abs(spectrum[:,0] - 690)).argmin()
    O2_3= (np.abs(spectrum[:,0] - 760)).argmin()
    O2= [m_O2, O2_1, O2_2, O2_3]

    H2O_1= (np.abs(spectrum[:,0] - 720)).argmin()
    H2O_2= (np.abs(spectrum[:,0] - 820)).argmin()
    H2O_3= (np.abs(spectrum[:,0] - 940)).argmin()
    H2O= [m_H2O, H2O_1, H2O_2, H2O_3]

    CO2_1= (np.abs(spectrum[:,0] - 1400)).argmin()
    CO2_2= (np.abs(spectrum[:,0] - 1600)).argmin()
    CO2= [m_CO2, CO2_1, CO2_2,0]

    CH4_1 = (np.abs(spectrum[:,0] - 1660)).argmin()
    CH4_2 = (np.abs(spectrum[:,0] - 2200)).argmin()
    CH4= [m_CH4, CH4_1, CH4_2, 0]

    CO= (np.abs(spectrum[:,0] - 2340)).argmin()
    CO= [m_CO, CO,0,0]

    N2O= (np.abs(spectrum[:,0] - 2870)).argmin()
    N2O= [m_N2O, N2O,0,0]

    mol= [O2,H2O,CO2,CH4,CO,N2O]

    molecules= ['O2', 'H2O', 'CO2', 'CH4', 'CO', 'N2O']
    props= ['mass', 'lambda_1', 'lambda_2', 'lambda_3']

    gases_dict= {}
    for n,i in enumerate(molecules):
        gases_dict[i]= {}
        for m,p in enumerate(props):
            gases_dict[i][p] = mol[n][m]
            if mol[n][m] == 0:
                del gases_dict[i][p]
    return gases_dict


if 'get_orient' in kwargs:
    print 'This is a terminal for calculating orientation, velocity and position of satelitte'
    print 'Satelitte is taking a picture, please wait...'
    get_orientation_phi()
    print '\nCalculate velocity:'
    print 'Please enter lambda shift from first reference planet, read from the main terminal:'
    l1= float(raw_input())
    print 'Please enter lambda shift from second reference planet, read from the main terminal:'
    l2= float(raw_input())
    get_vel_from_ref_stars(l1, l2)
    print '\nCalculating position, please wait...'
    get_position_from_dist()
    print 'Press any key to terminate program'
    raw_input()
    sys.exit(0)
