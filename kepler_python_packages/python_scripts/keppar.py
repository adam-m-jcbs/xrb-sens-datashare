"""
Provide "p" and "q" parameter names from KEPLER.

This routine should automatically update 'keppar_data.py' from 'kepdat.f'.
"""

import os.path
from make_keppar import make_keppar
from imp import reload

def update_keppar():
    pyfile = os.path.join(os.path.dirname(__file__),'keppar_data.py')
    kefile = os.path.expandvars(
        os.path.expanduser(
            os.path.join('~',
                         'kepler',
                         'source',
                         'kepdat.f')))
    if os.path.isfile(kefile):
        if os.path.isfile(pyfile):
            if os.path.getmtime(kefile) > os.path.getmtime(pyfile):
                make_keppar()
        else:
            make_keppar()

update_keppar()
import keppar_data
reload(keppar_data)
from keppar_data import p, q
