# globalValue_NFDA_J.py
import os
import numpy as np
# GVal is a great tool when developer need to maintain a 'hyper-global variable'  which is able to convey among files,
# Correspondingly, global variable can only convey among functions or classes in same file.
# Author: Josep Gao
# Date: July 1st, 2017
# Copyleft.

PARA = {}
DVPARA = {}
loopPARA_cache = {}
dV_PARA_cache = []


def setDVPARA(name_PARA, value_PARA):
    global DVPARA
    DVPARA[name_PARA] = value_PARA
    return 0


def getDVPARA(name_PARA):
    global DVPARA
    return DVPARA[name_PARA]


def initLoopPARA(name_PARA, value_PARA):
    global loopPARA_cache

    if type(name_PARA) == int:
        if not str(name_PARA)[0:2] == str(getPARA('classifier_list_PARA')[0]):
            print(['Warning! The input loop parameter: ' + str(name_PARA) + ' is not the current set classifier: ' + str(getPARA('classifier_list_PARA')) + '. Invalid input will not be processed.'])
            return 0

    loopPARA_cache[name_PARA] = [np.round(value_PARA, decimals=5), max([len(value_PARA), 1])]

    if name_PARA == 'cleanup':
        loopPARA_cache = {}
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
    setPARA('DVPARA', DVPARA)
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
            print('[ ' + str(key) + ' ] ' + '.' * (50 - len(str(key))) + ' || ' + str(PARA[key]) + '\n')

    input('Showing the WorkSpace and Pause! Press [Enter] to continue... ')
    return 0


# def initLoopPARA(name_PARA, value_PARA):
#     global loopPARA_cache
#     print(loopPARA_cache)
#     print('loc-1')
#     print(str(name_PARA)[0:2])
#     print(str(getPARA('classifier_list_PARA')[0]))
#     input('pause...press enter')
#     if type(name_PARA) == int:
#         if not str(name_PARA)[0:2] == str(getPARA('classifier_list_PARA')[0]):
#             print(['Warning! The input loop parameter: ' + str(name_PARA) + ' is not the current set classifier: ' + str(getPARA('classifier_list_PARA')) + '. Invalid input will not be processed.'])
#             print(loopPARA_cache)
#             print('loc-2')
#             input('pause...press enter')
#             return 0

#     loopPARA_cache[name_PARA] = [np.round(value_PARA, decimals=2), max([len(value_PARA), 1])]
#     print(name_PARA)
#     print(type(name_PARA))
#     if name_PARA == 'cleanup':
#         loopPARA_cache = {}
#     print(loopPARA_cache)
#     print('loc-3')
#     input('pause...press enter')
#     return 0
