import os, sys, datetime, string, time

os_nt = (os.name == 'nt')

'''TEXT EDITING IN TERMINAL'''

def clear(errorMsg = False):
    if os_nt == True:
        os.system('cls')
    else:
        try:
            term_command(cmd = 'clear', exit = False)
        except:
            print('\n'*30)
            if errorMsg == True:
                print('Memory Error, must press <enter> after entering selections\n\n')

def delete_chars(n = 1):
    type_error(n, int)
    if n < 1:
        error(ValueError, 'Cannot delete %d characters (minimum 1)'%(n))
    sys.stdout.write('\b'*n)

def delete_lines(n = 1):
    type_error(n, int)
    if n < 1:
        error(ValueError, 'Cannot delete %d characters (minimum 1)'%(n))
    sys.stdout.write('\x1b[2K' + '\x1b[1A\x1b[2K'*n + '\r')

def write(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def write_delay(string, delay):
    for c in string:
        write(c)
        pause(delay)

'''INPUT'''

def get_key_press(allowed = 'all', caseSensitive = False):
    os.system("stty raw -echo")
    if isinstance(allowed, (list, tuple)):
        allowed = [str(char).lower() for char in list(allowed)]
    elif isinstance(allowed, str):
        allowed = allowed.lower()
    if allowed == 'all':
        key = sys.stdin.read(1)
        if caseSensitive == False:
            key = key.lower()
        os.system("stty -raw echo")
        return key
    elif allowed == 'letters':
        allowed = list(string.ascii_lowercase)
    elif allowed == 'numbers':
        allowed = [0,1,2,3,4,5,6,7,8,9]
    elif allowed == 'wasd':
        allowed = ['w','a','s','d']
    elif allowed == 'yn':
        allowed = ['y','n']
    elif not isinstance(allowed, (list, tuple)):
        msg = 'Incorrect argument <key>, must enter a list or choose one of the following:\n'
        msg += '"all", "letters", "numbers", "wasd", "yn"'
        error(SyntaxError, msg)
    while True:
        key = sys.stdin.read(1)
        if caseSensitive == False:
            key = key.lower()
        if isinstance(allowed, (list, tuple)):
            condition = (key in allowed)
        elif isinstance(allowed, str):
            condition = (key == allowed)
        if condition == True:
            break
    os.system("stty -raw echo")
    return key

def key_to_continue(msg = 'Press Any Key to Continue', clearScreen = True):
    print msg
    get_key_press()
    if clearScreen == True:
        clear()

def select_from_menu(options, title = 'Menu', clearScreen = True, quit = True,
back = True):
    if clearScreen == True:
        clear(errorMsg = True)
    print(title)
    print("Select one of the following options:")
    if isinstance(options, dict):
        copy = options.copy()
        copy2 = options.copy()
        for i in range(len(options)):
            key = str(i+1)
            val = copy.pop(key, None)
            if val == None:
                break
            print('%s) %s'%(key, val))
        for k, v in copy.iteritems():
            copy2[k.lower()] = v
            copy2.pop(k)
            print('%s) %s'%(k, v))
        if back == True:
            print('R) Return')
            copy2['r'] = 'return'
        if quit == True:
            print('Q) Quit')
            copy2['q'] = 'quit'
        key_press = str(get_key_press(allowed = list(copy2.keys())))
        if key_press.lower() == 'q' and quit == True:
            exit(clearScreen = True)
        return copy2[key_press].lower()
    elif isinstance(options, (tuple, list)):
        more = []
        for n,i in enumerate(options):
            print('%s) %s'%(n+1, i))
        if quit == True:
            print('Q) Quit')
            more.append('q')
        if back == True:
            print('R) Return')
            more.append('r')
        key_press = get_key_press(allowed = list(range(1,len(options)+1)) + more)
        if key_press.lower() == 'q' and quit == True:
            exit(clearScreen = True)
        elif key_press.lower() == 'r' and back == True:
            return 'return'
        return int(key_press)-1

def get_input(msg = 'Input: ', types = None, minimum = None, maximum = None):
    if types == None:
        return raw_input(msg)
    if not isinstance(types, (tuple, list, dict)):
        types = [types]
    elif isinstance(types, (tuple, list)):
        types = list(types)
    else:
        error(TypeError, '<types> cannot be of type: <%s>'%(type(types)))
    string = raw_input(msg)
    try:
        if float in types:
            string = float(eval(string))
            if minimum != None and string < minimum:
                return None
            if maximum != None and string > maximum:
                return None
            return string
        elif int in types and int(eval(string)) - float(eval(string)) == 0:
            string = int(eval(string))
            if minimum is not None and string < minimum:
                return None
            if maximum is not None and string > maximum:
                return None
            return string
        else:
            return None
    except:
        if bool in types and string in ['True', 'False']:
            try:
                string = eval(string)
                return string
            except:
                if str not in types and type(string) != bool:
                    error(TypeError, 'Invalid input of type: <%s>'%(type(string)))
                    return None
                else:
                    return string
    return None

def confirm(msg = 'Proceed? (Y/N)', clearScreen = True, acceptNumber = False):
    if clearScreen == True:
        clear(errorMsg = True)
    print(msg)
    if acceptNumber == False:
        result = get_key_press(allowed = 'yn')
    else:
        result = get_key_press(allowed = ['y','n',0,1,2,3,4,5,6,7,8,9])
    if result == 'y':
        return True
    elif result == 'n':
        return False
    else:
        if acceptNumber == True:
            try:
                return int(result)
            except:
                return None

def input_number(msg = 'Enter a number: ', error = True):
    num = raw_input(msg)
    try:
        num = float(eval(num))
    except:
        if error == True:
            error(TypeError, 'User must enter a number\n"%s" cannot be evaluated as a number'%(num))
        elif error == False:
            return None
        else:
            msg = 'Argument: <error> only takes boolean values\nUser entered type: <%s>'%(type(num))
            error(SyntaxError, msg)
    else:
        if float(num) == None:
            error()
        return float(num)

def input_numbers(values, alias = None):
    final = {}
    for key in values:
        if key in alias:
            msg = '%s = %s\n'%(key, alias[key])
            msg += 'Keep default? (Y/N)'
            condition = confirm(msg = msg, acceptNumber = False)
            if condition == True:
                final[values[key]] = alias[key]
                continue
        msg = 'Assign a value:\n%s = '%(key)
        try:
            if not isinstance(condition, bool):
                condition = int(condition)
                msg += '%d'%(condition)
                number = True
            else:
                number = False
        except:
            number = False
        clear(errorMsg = True)
        while True:
            newkey = input_number(msg = msg, error = False)
            if newkey == None and number == False:
                msg = 'Assign a value:\n%s = '%(key)
                number = False
            elif newkey == None and number == True:
                newkey = ''
                break
            elif np.isfinite(newkey) == False:
                msg = 'Infinite values are not useable\nAssign a value:\n%s = '%(key)
                number = False
            else:
                break
        if number == True:
            final[values[key]] = eval(str(condition)[0] + str(newkey))
        elif number == False:
            final[values[key]] = newkey
    return final

'''PROGRAM FLOW'''

def pause(t):
    time.sleep(t)

def exit(msg = '', clearScreen = False):
    if clearScreen == True:
        clear()
    if len(msg) > 0:
        print(msg)
    sys.exit(1)

'''EXCEPTIONS'''

def type_error(var, types, msg = None):
    if not isinstance(types, (list, tuple)) and isinstance(types, type):
        types = [types]
    types = list(types)
    for n,i in enumerate(types):
        if type(var) == i:
            return var
        types[n] = str(i)
    if msg == None:
        msg = 'Argument <var> is of incorrect type: %s\nValid Types: '%(type(var))
        print(type(types))
        msg +=  ','.join(types)
    error(errortype = TypeError, msg = msg)

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

def popup(msg = 'Invalid', time = None):
    clear()
    if time == None:
        time = len(msg)*0.08
    print(msg)
    pause(time)

'''TERMINAL TOOLS'''

def term_command(cmd, exit = True):
    newpid = os.fork()
    if newpid == 0:
        os.execvp(cmd, [cmd,'-1'])
        if exit == True:
            os._exit(1)
    os.wait()

def get_terminal_kwargs(argv, allArgs = False):
    kwargs = {}
    others = []
    for arg in argv:
        try:
            splitArg = arg.split('=')
            if len(splitArg) == 2:
                kwargs[(splitArg[0].strip(' ')).lower()] = (splitArg[1].strip(' '))
            else:
                others.append(arg.lower())
        except:
            error(SyntaxError, 'Could not parse the given terminal arguments')
    if allArgs == False:
        return kwargs
    else:
        return kwargs, others

'''LOOPS AND LOADING BARS'''

def loop_except(func, error = Exception, msg = 'Attempt Failed, Try Again? (Y/N)',
clearScreen = True, endProgram = True):
    while True:
        try:
            return func
        except error:
            if confirm(msg = msg, clearScreen = clearScreen):
                continue
            elif endProgram == True:
                exit()
            else:
                break

def loading_bar(p1, p2 = None, msg = 'Loading...', length = 30, autoClear = True,
clearErrorMsg = False):
    string = '%s\n['%(msg)
    string = string + '*'*int(p1*length)
    string = string + ' '*(length-int(p1*length))
    if p2 != None:
        string = string + ']\n['
        string = string + "*"*int(p2*length)
        string = string + ' '*(length-int(p2*length))
    string = string + ']\n'
    string = string + ' '*(length/2 - 1)
    string = string + '%2d%%'%(int(p1*100))
    if autoClear == False:
        return string
    else:
        clear(errorMsg = clearErrorMsg)
        print(string)
