import functions as fx
import ui_tools as ui
import classes as cs
import numpy as np
import types, sys

if len(sys.argv) > 1 and sys.argv[1] == 'dev':
    devmode= True
else:
    devmode= False

'''VITAL ONE TIME FUNCTIONS'''

def title():
    string1 =  '\n'
    string1 += '          =====       ====      ====    ==    ===   =======  ======== \n'
    string1 += '          ==   ==    ==  ==    ==  ==   ==   ==     ==          ==    \n'
    string1 += '          ==   ==   ==    ==  ==    ==  == ==       ==          ==    \n'
    string1 += '          =====     ==    ==  ==        ====        =======     ==    \n'
    string1 += '          == ==     ==    ==  ==        == ==       ==          ==    \n'
    string1 += '          ==  ==    ==    ==  ==    ==  ==  ==      ==          ==    \n'
    string1 += '          ==   ==    ==  ==    ==  ==   ==   ==     ==          ==    \n'
    string1 += '          ==    ==    ====      ====    ==    ===   =======    ====   \n'
    string1 += '\n'

    string2 =  '           ~~~~:::::::<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>:::::::~~~~ \n\n'

    string3 =  '          ==          ====    ==    ==  ===    ==    ====    ==    == \n'
    string3 += '          ==         ==  ==   ==    ==  ===    ==   ==  ==   ==    == \n'
    string3 += '          ==        ==    ==  ==    ==  ====   ==  ==    ==  ==    == \n'
    string3 += '          ==        ========  ==    ==  == ==  ==  ==        ======== \n'
    string3 += '          ==        ==    ==  ==    ==  ==  == ==  ==        ==    == \n'
    string3 += '          ==        ==    ==  ==    ==  ==   ====  ==    ==  ==    == \n'
    string3 += '          ==        ==    ==   ==  ==   ==    ===   ==  ==   ==    == \n'
    string3 += '          ========  ==    ==    ====    ==     ==    ====    ==    == \n'

    string4 =  '\n\n                 CREATED BY ANDERS JULTON AND GABRIEL CABRERA\n'
    ui.clear()
    ui.write_delay(string1, 0.00004)
    ui.write_delay(string2, 0.005)
    ui.write_delay(string3, 0.00004)
    ui.write_delay(string4, 0.005)
    ui.get_key_press()
    ui.clear()

def prep_seeds():
    globals()['seed_menu'] = {}
    globals()['seed_menu']['C'] = 'Custom'
    for i, k in enumerate(globals()['seeds'].iterkeys()):
        globals()['seed_menu'][str(i+1)] = k[0].upper() + k[1:].lower()

def main():
    if devmode == False:
        title()
    prep_seeds()
    if devmode == True:
        run_whole_sim()
    run_menu(menu = main_menu, links = main_links, title = 'Main Menu',
    back = False)

'''ON DEMAND FUNCTIONS'''

def get_planets_dict():
    planets_list = globals()['rocket'].solar_system.planet_order
    planets_dict = {}
    for n,p in enumerate(planets_list):
        planets_dict[str(n+1)] = ui.titleize(p)
    return planets_dict

def run_menu(menu, links, title = 'Menu', quit = True, back = True):
    while True:
        sel = ui.select_from_menu(options = menu, title = title, quit = quit,
        back = back)
        if sel == 'return':
            break
        else:
            option = links[sel]
            if isinstance(option, types.FunctionType):
                option()
            elif isinstance(option, str):
                new_menu = option + '_menu'
                new_links = option + '_links'
                new_title = option[0].upper() + option[1:].lower()
                print new_menu, new_links, new_title
                cmd = 'run_menu(menu = %s, links = %s, title = "%s")'\
                %(new_menu, new_links, new_title)
                #try:
                exec(cmd)
                #except NameError:
                    #ui.popup(msg = 'Missing links for <%s>'%(option))

def choose_seed():
    warning_msg =  'Warning, overwriting the previous seed will clear all data;\n'
    warning_msg += 'are you sure you want to change seed? (Y/N)'

    def choose_new_planets(seed):
        globals()['seed'] = seed
        globals()['rocket'] = cs.Rocket(seed = seed)
        globals()['planet'] = globals()['rocket'].planet.name
        globals()['target'] = globals()['rocket'].target.name
        reset_all()
        if ui.confirm(msg = 'Use default planets? (Y/N)') == False:
            select_trajectory()
        reset_all()

    while True:
        sel = ui.select_from_menu(options = seed_menu, title = "Seed Menu")
        if sel == 'return':
            break
        elif sel in seeds:
            sel2 = ui.confirm(msg = warning_msg)
            if sel2 == True:
                choose_new_planets(seeds[sel])
                break
        elif sel == 'custom':
            ui.clear()
            new_seed = ui.get_input(msg = 'Please enter a new seed: ', types = [int])
            if new_seed is not None:
                sel2 = ui.confirm(msg = warning_msg)
                if sel2 == True:
                    choose_new_planets(new_seed)
                    break
                else:
                    continue
            else:
                ui.popup(msg = 'Invalid seed, must be of type <int>')

def check_current_parameters():
    '''Displays the current simulation parameters, such as seed number etc...'''
    ui.clear()
    data = 'Seed: %d, '%(seed)
    data += 'Start Planet: %s, '%(globals()['rocket'].planet.name)
    data += 'Target Planet: %s\n'%(globals()['rocket'].target.name)
    ui.key_to_continue(data + '\n\nPress any key to continue')

def reset_all():
    '''Resets all currently loaded objects such that a new parameter can be used'''
    globals()['rocket'] = cs.Rocket(seed = globals()['seed'], planet = globals()['planet'],
    target = globals()['target'])

def select_trajectory():
    trajectories = get_planets_dict()
    sel = ui.select_from_menu(options = trajectories, title = 'Select a Starting Planet')
    while True:
        sel2 = ui.select_from_menu(options = trajectories, title = 'Select a Target Planet')
        if sel2 == sel:
            ui.popup('Target planet cannot be same as starting planet')
        else:
            break
    warning =  'Implement changes? - all previous data will be erased (Y/N)'
    warning += '\n\nPrevious Planet: %s, Previous Target: %s'\
    %(globals()['rocket'].planet.name, globals()['rocket'].target.name)
    warning += '\nNew Planet: %s, New Target: %s'\
    %(ui.titleize(sel), ui.titleize(sel2))
    if ui.confirm(warning) == True:
        globals()['planet'] = sel
        globals()['target'] = sel2
        reset_all()

def plot_planet_orbit():
    planets = get_planets_dict()
    while True:
        sel = ui.select_from_menu(planets)
        if sel == 'return':
            break
        else:
            globals()['rocket'].solar_system.planets[sel].plot()

def plot_solar_system_orbits():
    globals()['rocket'].solar_system.plot()

def plot_rocket_launch():
    if not globals()['rocket'].liftoff_calculated:
        ui.clear()
    globals()['rocket'].plot_liftoff()

def plot_intercept_launch_window():
    if not globals()['rocket'].transfer_calculated:
        ui.clear()
    globals()['rocket'].plot_intercept(numerical = False)

def plot_intercept_trajectory():
    if not globals()['rocket'].circularization_calculated:
        ui.clear()
    globals()['rocket'].plot_intercept()

def data_printouts():
    options = {'1':'Solar System', '2':'Sun', '3':'Planet', '4':'Data Charts', '5':'Rocket'}

    chart_options = {'1':'Semi-Major Axis', '2':'Eccentricity', '3':'Radius',
    '4':'Angle of Semi-Major Axis', '5':'Mass', '6':'Day Length',
    '7':'Surface Atmosphere Density', '8':'Semi-Minor Axis',
    '9':'Orbital Period', '10':'Surface Temperature', '11':'Apoapsis',
    '12':'Periapsis'}

    while True:
        sel = ui.select_from_menu(options = options, title = 'Available Data Printouts:')
        if sel == 'return':
            break
        elif sel == 'solar system':
            ui.clear()
            print globals()['rocket'].solar_system
            ui.key_to_continue('\nPress any key to return to menu')
        elif sel == 'sun':
            ui.clear()
            print globals()['rocket'].solar_system.sun
            ui.key_to_continue('\nPress any key to return to menu')
        elif sel == 'planet':
            planets = get_planets_dict()
            sel2 = ui.select_from_menu(options = planets, title = 'Select a Planet:')
            if sel2 == 'return':
                break
            else:
                ui.clear()
                print globals()['rocket'].solar_system.planets[sel2]
                ui.key_to_continue('\nPress any key to return to menu')
        elif sel == 'data charts':
            while True:
                sel3 = ui.select_from_menu(options = chart_options,
                title = 'Available Data Charts')
                if sel3 == 'return':
                    break
                else:
                    ui.clear()
                    get_chart(sel3)
                    ui.key_to_continue('\nPress any key to return to menu')
        elif sel == 'rocket':
            ui.clear()
            globals()['rocket'].print_str()
            ui.key_to_continue('\nPress any key to return to menu')

def get_chart(option):
    variables = {'semi-major axis':'a', 'eccentricity':'e', 'radius':'radius',
    'angle of semi-major axis':'psi', 'mass':'mass', 'day length':'period',
    'surface atmosphere density':'rho0', 'semi-minor axis':'b',
    'orbital period':'T', 'surface temperature':'temperature',
    'apoapsis':'apoapsis', 'periapsis':'periapsis'}

    units = {'semi-major axis':'AU', 'eccentricity':'', 'radius':'km',
    'angle of semi-major axis':'rads', 'mass':'Solar Masses',
    'day length':'Earth Days', 'surface atmosphere density':'kg/m^3',
    'semi-minor axis':'AU', 'orbital period':'yrs', 'surface temperature':'K',
    'apoapsis':'AU', 'periapsis':'AU'}

    print 'Data Chart - %s of Each Planet in Solar System %d\n'\
    %(ui.titleize(option), globals()['seed'])

    longest_p = 0
    longest_s = 0
    for p,v in globals()['rocket'].solar_system.__call__(variables[option]).iteritems():
        if len(p) > longest_p:
            longest_p = len(p)
        s = '%g %s'%(v, units[option])
        if len(s)+1 > longest_s:
            longest_s = len(s)+1

    for p,v in globals()['rocket'].solar_system.__call__(variables[option]).iteritems():
        if option == 'angle of semi-major axis':
            print '%*s |'%(longest_p + 2, ui.titleize(p)),\
            '%*g'%(longest_s-5, v/np.pi)+ u'\u03C0', ' %s'%(units[option])
        else:
            s = '%g %s'%(v, units[option])
            print '%*s |%*s'%(longest_p + 2, ui.titleize(p), longest_s + 1, str(s))

def run_whole_sim():
    ui.clear()
    print('Running Whole Simulation')
    globals()['rocket'].run()
    ui.key_to_continue('\nPress any key to return to menu')

'''MENU OPTIONS'''
main_menu = {'1':'Plots', '2':'Data Printouts', '3':'Run Whole Sim', 'O':'Options'}
options_menu = {'1':'Choose Seed', '2':'Select Trajectory', '3':'Check Current Parameters'}
plots_menu = {'1':'Solar System Orbits', '2':'Planet Orbit', '3':'Rocket Launch',
'4':'Intercept Launch Window', '5':'Intercept Trajectory'}

'''MENU LINKS'''
main_links = {'plots':'plots', 'data printouts':data_printouts, 'run whole sim':run_whole_sim,
'options':'options'}
options_links = {'choose seed':choose_seed, 'select trajectory':select_trajectory,
'check current parameters':check_current_parameters}
plots_links = {'solar system orbits':plot_solar_system_orbits,
'planet orbit':plot_planet_orbit, 'rocket launch':plot_rocket_launch,
'intercept launch window':plot_intercept_launch_window,
'intercept trajectory':plot_intercept_trajectory}

'''DATA DICTS'''
seeds = {'gabriel':45355, 'anders':28631, 'ulrik':82275}

'''DEFAULT CONDITIONS'''
seed = seeds['gabriel']
planet = 'sarplo'
target = 'jevelan'

'''DEPENDENT OBJECTS'''
rocket = cs.Rocket(seed = seed, planet = planet, target = target)

main()
