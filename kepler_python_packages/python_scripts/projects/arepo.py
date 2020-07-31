#!/bin/env python3

"""
Make Arepo restart file with accreted mass in it

Make script executable.
Pass 'base name' as first paramenter, e.g.,

arepo.py P-z40-3D
"""

import re
import glob
import os.path
import sys

def update_script(mass):
    """
    update Arepo restart script
    """
    with open('parm.txt', 'rt') as f:
        t = f.read()
    s = 'CentralMass {:17.5e}'.format(mass)
    t = re.sub(r'^(CentralMass\s+[-+\.0-9e]+\s*)$', s, t, flags = re.MULTILINE)
    with open('parm.txt', 'wt') as f:
        f.write(t)

def find_file(basename):
    """
    find last file name
    """
    filenames = glob.glob(basename + '.o*')
    return sorted(filenames)[-1]

def read_mass(filename):
    """
    read acretion mass as last entry of file
    """
    with open(filename, 'rt') as f:
        t = f.read()
    mass = re.findall(r'^CENTRALPOTENTIAL: Accreted [-+\.0-9e]+ from \d+ cells onto central mass potential, total = ([-+\.0-9e]+)\s*$',
                      t, flags = re.MULTILINE)[-1]
    return float(mass)

def replace(basename):
    filename = find_file(basename)
    mass = read_mass(filename)
    update_script(mass)

if __name__ == "__main__":
    prm = sys.argv[1:]
    if len(prm) > 0:
        basename = prm[0]
    else:
        basename = 'P-z40-3D'
    replace(basename)
