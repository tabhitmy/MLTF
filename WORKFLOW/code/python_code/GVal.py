# globalValue_NFDA_J.py
import os

# GVal is a great tool when developer need to maintain a 'hyper-global variable'  which is able to convey among files,
# Correspondingly, global variable can only convey among functions or classes in same file.
# Author: Josep Gao
# Date: July 1st, 2017
# Copyleft.

PARA = {}
loopPARA_cache = {}

# Specific for loop parameter setting in controlPanel()


def initLoopPARA(name_PARA, value_PARA):
    global loopPARA_cache
    loopPARA_cache[name_PARA] = [value_PARA, len(value_PARA)]
    return 0

# Specific for loop parameter setting in controlPanel()


def getLoopPARA_cache():
    global loopPARA_cache
    return loopPARA_cache


def setPARA(name_PARA, value_PARA):
    # Setting the hyper-global parameter with name and value.
    # If name is existed in PARA, overwriting it, otherwise creating.
    global PARA
    PARA[name_PARA] = value_PARA
    return 0


def getPARA(name_PARA):
    # Getting the hyper-global parameter according to the input name.
    # If name is not existed in PARA, error occurs.
    global PARA
    return PARA[name_PARA]


def initPARA(name_PARA, value_PARA):
    # Not applied, Later will be deprecated.
    global PARA
    PARA[name_PARA] = value_PARA
    return 0


def appendPARA(name_PARA, value_PARA):
    # Not applied, Later will be deprecated
    global PARA
    PARA[name_PARA].append(value_PARA)
    return 0


def show(*args):
    # A tool to show certain parameters' names and theirs values in the command line.
    # Input *args takes zero, one or more input variables, in string format, defined as parameter names in PARA.
    # When input is void, show all the parameters in the PARA at the calling time of GVal.show()
    print('#' * 70)
    print('### [ PARAMETER NAME ] ' + '.' * 31 + ' || Variables')
    print('#' * 70)

    if args:
        for args_key in args:
            print('[ ' + args_key + ' ] ' + '.' * (50 - len(args_key)) + ' | | ' + str(PARA[args_key]) + '\n')
    else:
        for key in PARA.keys():
            print('[ ' + key + ' ] ' + '.' * (50 - len(key)) + ' || ' + str(PARA[key]) + '\n')

    input('Showing the WorkSpace and Pause! Press [Enter] to continue... ')
    return 0
