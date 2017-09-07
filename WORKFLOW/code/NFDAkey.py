# NFDAkey.py
keycontent = '234'
import os
import sys

locipe = os.environ['SSH_CONNECTION'][10:12]

if locipe == '21':
    username = os.environ["USERNAME"]
    if username == 'GaoMY':
        print(' ')
        print(' ')
        print('    $$$ WELCOME to MLTF(v0.1) ! $$$')
        print('    <Author: Josep GAO> ')
        print('    <Copyright Reserved.>')
        print(' ')
        print(' ')
        pscode = input('    USERNAME is [GaoMY], PLEASE INPUT PASSCODE and PRESS ENTER...')
        if pscode == keycontent:
            print(' ')
            print(' ')
            print('    ' + '$' * 20)
            print('    $ Wlkm! Commander! $')
            print('    ' + '$' * 20)
            sys.exit(1)
        else:
            print(' ')
            print(' ')
            print('    You enter a wrong passcode, EXIT! Please CHANGE YOUR USERNAME in run_NFDA.sh and re-run it! ')
            print(' ')
            sys.exit(0)
    else:
        print(' ')
        print(' ')
        print('    $$$ WELCOME to MLTF(v0.1) ! $$$')
        print('    <Author: Josep GAO> ')
        print('    <Copyright Reserved.>')
        print(' ')
        print(' ')
        input('    USERNAME is [' + username + '], PRESS ENTER to Confirm and start! ')
        sys.exit(1)
