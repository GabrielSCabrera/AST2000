import functions as fx
import ui_tools as ui
import classes as cs
import types

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

    string4 =  '\n\n                CREATED BY ANDERS JULTON AND GABRIEL CABRERA\n'
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
    title()
    prep_seeds()
    run_menu(menu = main_menu, links = main_links, title = 'Main Menu',
    back = False)

'''ON DEMAND FUNCTIONS'''

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
                try:
                    exec(cmd)
                except NameError:
                    ui.popup(msg = 'Missing links for <%s>'%(option))

def choose_seed():
    warning_msg =  'Warning, overwriting the previous seed will clear all data;\n'
    warning_msg += 'are you sure you want to change seed? (Y/N)'

    def choose_new_planets(sel):
        if sel == True:
            if globals()['seed'] == globals()['seeds']['gabriel']:
                globals()['planet'] = 'sarplo'
                globals()['target'] = 'jevelan'
            elif globals()['seed'] == globals()['seeds']['ulrik']:
                globals()['planet'] = 'kraid'
                globals()['target'] = 'brinstar'
            else:
                globals()['planet'] = 'a'
                globals()['target'] = 'b'
        else:
            select_trajectory()

    while True:
        sel = ui.select_from_menu(options = seed_menu, title = "Seed Menu")
        if sel == 'return':
            break
        elif sel in seeds:
            sel2 = ui.confirm(msg = warning_msg)
            if sel2 == True:
                globals()['seed'] = seeds[sel]
                sel3 = ui.confirm(msg = 'Use default planets? (Y/N)')
                choose_new_planets(sel3)
                reset_all()
                break
        elif sel == 'custom':
            new_seed = ui.get_input(msg = 'Please enter a new seed: ', types = [int])
            if new_seed is not None:
                sel2 = ui.confirm(msg = warning_msg)
                if sel2 == True:
                    globals()['seed'] = new_seed
                    sel3 = ui.confirm(msg = 'Use default planets? (Y/N)')
                    choose_new_planets(sel3)
                    reset_all()
                    break
                else:
                    continue
            else:
                ui.popup(msg = 'Invalid seed, must be of type <int>')

def check_current_parameters():
    '''Displays the current simulation parameters, such as seed number etc...'''
    ui.clear()
    data = 'Seed: %d, '%(seed)
    data += 'Start Planet: %s, '%(globals()['planet_object'].name)
    data += 'Target Planet: %s\n'%(globals()['target_object'].name)
    ui.key_to_continue(data + '\n\nPress any key to continue')

def reset_all():
    '''Resets all currently loaded objects such that a new parameter can be used'''
    globals()['solar_system'] = cs.Solar_System(seed = globals()['seed'])
    globals()['planet_object'] = globals()['solar_system'].planets[globals()['planet']]
    globals()['target_object'] = globals()['solar_system'].planets[globals()['target']]

'''UNFINISHED FUNCTIONS'''

def select_trajectory():
    ui.error(Exception, '<select_trajectory> not yet implemented.  Fix this soon!')

'''MENU OPTIONS'''
main_menu = {'1':'Plots', '2':'Planets', '3':'Launch Rocket', 'O':'Options'}
options_menu = {'1':'Choose Seed', '2':'Select Trajectory', '3':'Check Current Parameters'}

'''MENU LINKS'''
main_links = {'plots':'plots', 'planets':'planets', 'launch rocket':'launch_rocket',
'options':'options'}
options_links = {'choose seed':choose_seed, 'select trajectory':select_trajectory,
'check current parameters':check_current_parameters}

'''DATA DICTS'''
seeds = {'gabriel':45355, 'anders':28631, 'ulrik':82275}

'''DEFAULT CONDITIONS'''
seed = seeds['gabriel']
planet = 'sarplo'
target = 'jevelan'

'''DEPENDENT OBJECTS'''
solar_system = cs.Solar_System(seed = seed)
planet_object = solar_system.planets[planet]
target_object = solar_system.planets[target]

main()
