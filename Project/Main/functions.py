'''[1] MODULES'''
from ast2000solarsystem_27_v4 import AST2000SolarSystem
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
import matplotlib.patches as Patches
import matplotlib.axes as Axes
import math, itertools, string
import numpy.linalg as LA
try:
    from numba import jit
    import numba as nb
except ImportError:
    string = "User must install module <numba> to use module <classes.py>"
    fx.error(ImportError, string)

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

if __name__ == '__main__':
    pass
