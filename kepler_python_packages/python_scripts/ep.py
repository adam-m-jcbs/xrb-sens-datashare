#! /bin/env python3

"""
you can use, e.g.,

  ep prog3d

to find and load ./projects/prog3d.py

TODO - update and use git.py library
TODO - integrate staring of FORTRAN code (KEPLER)
"""

import socket
import os
import os.path
import sys
import psutil
import subprocess
import shutil
import time
import re
import contextlib

from utils import iterable, environ

def set_defaults(mode):

    global cwd, hostname, username, layout, source, remote, git

    hostname = socket.gethostname()
    username = os.getlogin()
    cwd = os.getcwd()

    home = os.path.expanduser(os.path.expandvars('~/'))

    if 'mode' == 'python':
        remote = os.path.join(home, 'python', 'remote')
        source = os.path.join(home, 'python', 'source')
    else:
        remote = os.path.join(home, 'kepler', 'remote')
        source = os.path.join(home, 'kepler', 'source')

    if os.path.exists(remote) and os.path.samefile(remote, cwd):
        source = remote

    if os.path.isdir(os.path.join(cwd, '.git')):
        source = cwd

    print(f'[E] INFO: Unsing directory "{source}" in mode {mode}.')

    if mode == 'python':
        geometry = "84x74-116+0"
        if hostname == 'zinc.maths.monash.edu':
            geometry = "84x57-100+0"
        if hostname == 'yttrium.physics.monash.edu':
            geometry = "84x56-100+0"
        #font = "-adobe-courier-bold-r-normal--25-*-*-*-*-*-*-*"
        color = "#f0fff8"
        colors = ['-bg', color, '-fg', 'black', '-cr', 'cyan4', '-ms', 'cyan4']
    else:
        geometry="74x86-100+0"
        if hostname == 'zinc.maths.monash.edu':
            geometry = "74x44-100+0"
        font = "-adobe-courier-medium-r-normal--18-*-*-*-*-*-*-*"
        color = "#e8ffff"
        colors = ['-bg', color, '-fg', 'black', '-cr', 'cyan4', '-ms', 'cyan4']


    geometries = ['-g', geometry]
    try:
        fonts = ['-font', font ]
    except:
        fonts = []

    layout = geometries + fonts + colors

    git = shutil.which('git')

@contextlib.contextmanager
def sourcedir(path):
    cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(cwd)

def up(window = True):
    with sourcedir(source):
        args = [git, 'pull']
        try:
            subprocess.check_call(args, timeout = 30)
            status()
            return
        except subprocess.TimeoutExpired:
            print('git pull timed out ... ')
            pass
        except subprocess.CalledProcessError:
            print('error synchronizing (likely no ssh tunnel) ... ')
            pass
        except:
            raise

    status()

    if window is True:
        print(' ... starting emacs w/o up-to-date files.')

        print(' ... continuing to try in background')
        pid = os.fork()
        if pid > 0:
            return
        with sourcedir(source):
            try:
                subprocess.check_call(args)
            except:
                pass
            sys.exit()

def status():
    print('-'*32)
    with sourcedir(source):
        subprocess.check_call([git, 'status', '-s'])
    print('-'*32)

def ci(window = True, local = False):
    args = [git, 'commit', '-a', '-v']
    status()

    env = dict()
    editor = 'git_message_editor'
    if window is not True:
        editor += ' -nw'
    env['GIT_EDITOR'] = editor
    with environ(env), sourcedir(source):
        print('Starting git commit')
        with subprocess.Popen(
                args,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT) as proc:
            message, errmsg = proc.communicate()
            code = proc.returncode
    message = message.decode()
    for m in message.splitlines():
        print(' [git] {}'.format(m))
    if code == 0:
        print('git commit finished.')
    else:
        if message.find('nothing to commit, working directory clean') > 0:
            print('Done.')
        else:
            print('git commit failed.')
        return

    if local:
        return
    with sourcedir(source):
        args = [git, 'push']
        try:
            print('Starting git push')
            subprocess.check_call(args)
            print('git push finished.')
        except:
            print('#'*24)
            print('### git push failed! ###')
            print('#'*24)
    # try to continue push in background?


def find_local(filename, ignore = ('.git',), extensions = ('.py', '.f', '.f90', '.inc', '')):
    for dirpath, dirs, files in os.walk(source):
        for f in files:
            for e in extensions:
                if e != '' and not filename.endswith(e):
                    fn = filename + e
                else:
                    fn = filename
                if f == fn:
                    return os.path.join(dirpath, f)
        for d in iterable(ignore):
            try:
                dirs.remove(d)
            except:
                pass


def emacs(argv, window = True):
    args = [shutil.which('emacs')]
    if window:
        args += layout
    else:
        args += ['-nw']
    if len(argv) > 0:
        for a in argv:
            f = find_local(a)
            if f is not None:
                args += [f]
            elif a.startswith('-'):
                args += [a]
    else:
        args += ['.']
    with sourcedir(source):
        subprocess.check_call(args)

def get_message_inline():
    import readline
    lines = []
    print('Enter commit message below.')
    print('Press ^D on empty line to finish')
    done = False
    while not done:
        try:
            lines += [input('>').strip()]
        except EOFError:
            done = True
    print('')
    message = ('\n'.join(lines)).strip()
    return message


def ping():
    hostname = "www.monash.edu"
    try:
        subprocess.check_output(["/bin/ping", "-c1", hostname], universal_newlines = True)
    except subprocess.CalledProcessError:
        return False
    return True

def ssh_connect(window = False):
    # experimental
    with sourcedir(source):
        try:
            origin_url = subprocess.check_output(
                ['git', 'config', '--get', 'remote.origin.url'],
                universal_newlines = True)
            print('[ep] Using repository', origin_url)
            hostname = re.findall('^(\S+@\S+):', origin_url)[0]
        except subprocess.CalledProcessError:
            # don't know what is going on
            return True
    try:
        subprocess.check_output(
            ["/bin/ssh", hostname],
            universal_newlines = True,
            timeout = 10)
    except:
        connector = shutil.which('M')
        if connector is not None:
            args = [connector]
            if window:
                args = ['xterm'] + args
            try:
                subprocess.check_call(args)
            except subprocess.CalledProcessError:
                print('Could not connect to remote.')
                return False
    return True


if __name__ == '__main__':
    argv = sys.argv

    program = os.path.basename(argv[0])

    # test whether we do KEPLER or Python
    if program in ('ef', 'eff', 'nf', 'cif', 'upf',):
        mode = 'kepler'
    else:
        mode = 'python'
    set_defaults(mode)

    # test whether to run in window mode
    window = program in ('ep',)
    display = os.environ.get('DISPLAY', '')
    if display.startswith(':'):
        window = True
    try:
        # same host
        ssh_connection = os.environ.get('SSH_CONNECTION', '').split(' ')
        if ssh_connection[0] == ssh_connection[2]:
            window = True
    except:
        pass

    try:
        # same site ('/16')
        ssh_connection = os.environ.get('SSH_CONNECTION', '').split(' ')
        if ('.'.join(ssh_connection[0].split('.')[:2]) ==
            '.'.join(ssh_connection[2].split('.')[:2])):
            window = True
    except:
        pass

    # test whether to run locally.
    # test existance of internet connection doing a ping to google
    if '-l' in argv:
        argv.remove('-l')
        local = True
    else:
        local = not ping()

    if not local:
        # we no longer need this as we now use ProxyJump
        # local = not ssh_connect()
        pass

    if program in ('ci', 'cif',):
        ci(window, local)
    elif program in ('up', 'upf',):
        if not local:
            up(window)
        else:
            print('Nothing to do.')
    else:
        if window is True:
            # do the rest as background process
            pid = os.fork()
            if pid > 0:
                print('Running in backgorund - PID: {:d}'.format(pid))
                sys.exit()

        if not local:
            up(window)
        emacs(argv[1:], window)
        if not local:
            # this is needed becasue after the fork we no longer have
            # terminal access.
            # we no longer need this as we now use ProxyJump
            # local = not ssh_connect(window)
            pass
        ci(window, local)
