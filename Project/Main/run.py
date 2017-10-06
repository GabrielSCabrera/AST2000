import ui_tools as ui

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
    ui.write_delay(string1, 0.00008)
    ui.write_delay(string2, 0.008)
    ui.write_delay(string3, 0.00008)
    ui.write_delay(string4, 0.005)
    ui.get_key_press()
    ui.clear()

def main():
    title()
    #while True:


main()
