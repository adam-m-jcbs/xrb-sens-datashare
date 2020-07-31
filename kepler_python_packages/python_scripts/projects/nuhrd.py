#! /bin/env python3
"""
routines for neutrion HRD project with Qian
"""

import convdata
import os.path
import physconst
import numpy as np
from isotope import KepIon

path = '/home/alex/kepler/sollo09'
masses = [12, 15, 20, 25]
models = ['s{:d}'.format(m) for m in masses]
files = [os.path.join(path, m, m+'.cnv') for m in models]

outpath = '/m/web/Download/Qian'
outfiles = [os.path.join(outpath, m+'.txt') for m in models]

def evo_data():
    for o,f in zip(outfiles,files):
        c = convdata.loadconv(f)
        x = c.net
        data = np.array([
            c.time,
            c.xlumn,
            c.tc,
            c.dc,
            x.abu('h1', missing=0.),
            x.abu('he4', missing=0.),
            x.abu('c12', missing=0.),
            x.abu('o16', missing=0.),
            x.abu('ne20', missing=0.),
            x.abu('si28', missing=0.),
            x.abu("'fe'", missing=0.),
            ]).transpose()
        with open(o, 'w') as out:
            for y in ['time', 'L_nu', 'T_c', 'rho_c', 'h1', 'he4', 'c12', 'o16', 'ne20', 'si28', "'fe'" ]:
                out.write('{:>25s}'.format(y))
            out.write('\n')            
            for y in ['(sec)', '(erg/sec)', '(K)', '(g/cm**3)'] + ['(mass fraction)']*7:
                out.write('{:>25s}'.format(y))
            out.write('\n')            
            for x in data:
                for y in x:
                    out.write('{:25.17e}'.format(y))
                out.write('\n')            
    
if __name__ == "__main__":
    evo_data()
