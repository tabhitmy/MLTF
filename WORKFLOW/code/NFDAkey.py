# NFDAkey.py

import os
import sys
import time
from time import clock

locipe = os.environ['SSH_CONNECTION'][10:12]
keycontent_raw = '1324'
if locipe == '21':
    username = os.environ["USERNAME"]
    username_prefix = username[:5]
    if username_prefix == 'GaoMY':
        print(' ')
        print(' ')
        print('    $$$ WELCOME to MLTF(v0.1) ! $$$')
        print('    <Author: Josep GAO> ')
        print('    <Copyright Reserved.>')
        timenow = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print('    <' + timenow + '>')
        keycontent = (timenow[14:16] + timenow[11:13])
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
        elif pscode == keycontent_raw:
            print(' ')
            print(' ')
            print('    ' + '$' * 20)
            print('    $ Wlkm! Commander! $')
            print('    $ Try to use realtime password next time! $')
            print('    ' + '$' * 20)
            sys.exit(1)
        else:
            print(' ')
            print(' ')
            print('    You enter a wrong passcode, EXIT! Please CHANGE YOUR USERNAME in run_NFDA.sh and re-run it! ')
            print('    Or maybe you remember the default password?... ')
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
