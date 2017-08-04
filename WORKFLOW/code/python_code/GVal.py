# globalValue_NFDA_J.py
import os

PARA = {}


def setPARA(name_PARA, value_PARA):
    global PARA
    PARA[name_PARA] = value_PARA
    return 0


def getPARA(name_PARA):
    global PARA
    return PARA[name_PARA]


def initPARA(name_PARA, value_PARA):
    global PARA
    PARA[name_PARA] = value_PARA
    return 0


def appendPARA(name_PARA, value_PARA):
    global PARA
    PARA[name_PARA].append(value_PARA)
    return 0


def show(*args):
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
