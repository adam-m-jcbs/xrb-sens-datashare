#! /bin/env python3

"""
remove broken ssh keys from ~/.ssh/known_hosts

usage:
  fix_ssh_key <hostname>
"""

import sys
import os.path
from subprocess import run, PIPE, DEVNULL
import re
import hashlib


def fix_key(key):
    """
    remove broken ssh_key from ~/.ssh/known_hosts

    parameters:
       key: <hostname>
    """
    try:
        result = run(['ssh', key],
                     stderr = PIPE,
                     stdout = PIPE,
                     stdin = DEVNULL,
                     timeout = 10)
    except:
        print('failed')
        return
    err = result.stderr.decode()
    x = re.findall('WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!', err)
    if len(x) == 0:
        print('No wrong key found.')
        return
    path = os.path.expanduser('~/.ssh/known_hosts')
    p = f'(?m)Offending (?:(?:RSA|DSA|ECDSA) )?key (?:for IP )?in {path}:(\d+)\s*$'
    numbers = [int(n) for n in re.findall(p, err)]
    numbers = sorted(numbers)[::-1]
    with open(path, 'rt') as f:
        lines = f.readlines()
    new = []
    for n in numbers:
        print(f'[SSH_FIX] removig line {n}: {lines[n-1].strip()}')
        del lines[n-1]
    with open(path, 'wt') as f:
        f.writelines(lines)

if __name__ == "__main__":
    fix_key(sys.argv[1])
