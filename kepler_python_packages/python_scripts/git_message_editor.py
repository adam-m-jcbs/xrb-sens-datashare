#! /bin/env python3

import sys, shutil, subprocess

if __name__ == '__main__':
    args = sys.argv
    prog = shutil.which('emacs')
    args[0] = prog
    args += '-f diff-mode -f flyspell-mode -f whitespace-mode'.split(' ')

    with subprocess.Popen(args) as proc:
        code = proc.returncode
    sys.exit(code)
