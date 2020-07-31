#! /bin/env python3

import os
import os.path
import sys
import subprocess

class RSync(object):

    def __init__(self, timeout = 300):
        runpar = "rsync -vazu --progress -e ssh".split()
        if timeout is not None:
            runpar.append(f'--timeout={timeout:d}')
        self.runpar = runpar

    def run(self, argv):
        if "DELETE" in argv:
            delete = True
            argv.remove("DELETE")
        else:
            delete = False

        args = self.runpar.copy()
        if delete:
            args.append('--delete')
        if len(argv) == 1:
            argv.append(os.curdir)

        assert len(argv) == 2
        for i,arg in enumerate(argv):
            if arg.count(':') == 0:
                args.append(arg)
                continue
            assert arg.count(':') == 1
            host, target = arg.split(':')
            if target == '':
                target = os.curdir
            target0 = target
            target =  os.path.relpath(target, os.path.expanduser('~'))

            if (i == 0 and target0 == '.') or target0.endswith(os.sep):
                target += os.sep

            args.append(':'.join((host, target)))
        print('[Debug] calling: ' + ' '.join(args))
        while True:
            try:
                subprocess.run(args, check = True)
            except CalledProcessError as e:
                print('[Debug] Error: ', e)
                continue
            else:
                break

if __name__ == '__main__':
    argv = sys.argv[1:]
    RSync().run(argv)
