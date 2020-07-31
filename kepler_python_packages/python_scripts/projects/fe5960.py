"""
Fe59/Fe60 T-dep rate test
"""

from physconst import EV, KB, SEC

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, os.path

import bdat

Fe60 = np.array([
    1.000000E+00,
    8.389142E-15,
    5.000000E+00,
    3.114082E-05,
    -9.553623E+00,
    1.000000E+00,
    3.981072E-05,
    -2.288696E+01,
    9.000000E+00,
    3.743950E-03,
    -2.451014E+01,
    ])

Fe59 = np.array([
    4.000000E+00, 1.802817E-07, 2.000000E+00, 8.912509E-06,-3.327536E+00, 6.000000E+00, 1.230269E-03,
    -5.484058E+00, 4.000000E+00, 2.951209E-05,-6.620290E+00, 4.000000E+00, 5.623413E-05,-8.417391E+00,
    8.000000E+00, 4.265795E-04,-1.186087E+01, 4.000000E+00, 1.778279E-04,-1.253333E+01, 4.000000E+00,
    2.238721E-04,-1.347246E+01, 2.000000E+00, 3.715352E-04,-1.404058E+01, 1.000000E+01, 6.456542E-05,
    -1.758841E+01, 6.000000E+00, 2.570396E-03,-1.820290E+01, 8.623837E-01,
    ])



def rate28(t, c):
    nrate = (len(c)-2) // 3
    k0 = 0
    k1 = k0 + 1
    k2 = k0 + 2
    zsum = c[k0]
    frate = c[k0] * c[k1]
    lk0 = k2
    t9m1 = 1.e9 / t
    for jk in range(nrate):
        lk1 = lk0+1
        lk2 = lk0+2
        t9y = c[lk0] * np.exp(c[lk2]*t9m1)
        zsum = zsum + t9y
        frate = frate + c[lk1]*t9y
        lk0 = lk0 + 3
    return frate / zsum

def plot():
    f = plt.figure()
    ax = f.add_subplot(111)

    # b = bdat.BDat('/home/alex/kepler/test/Fe60/rath00_10.1.bdat_weakT_fe60')
    b = bdat.BDat('/home/alex/kepler/test/Fe60/rath00_10.1.bdat_weakT_fe60_LMP-old')
    for d in b.data:
        if d.ion == 'Fe59':
            assert d.ic[2,0] == 28
            Fe59 = d.c[2][0:d.ic[2,1]]
        if d.ion == 'Fe60':
            assert d.ic[2,0] == 28
            Fe60 = d.c[2][0:d.ic[2,1]]

    t = np.logspace(7,10,100)
    r60 = rate28(t, Fe60)
    ax.plot(t,r60, label = r'$^{60}$Fe($\beta^-$)$^{60}$Co')
    r59 = rate28(t, Fe59)
    ax.plot(t,r59, label = r'$^{59}$Fe($\beta^-$)$^{59}$Co')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('T / K')
    ax.set_ylabel('rate / Hz')

    path=os.path.expanduser('~/kepler/test/Fe60')
    Fe59w=read_kepler_wrate_debug(os.path.join(path,'Fe59w.txt'))
    Fe60w=read_kepler_wrate_debug(os.path.join(path,'Fe60w.txt'))

    ax.plot(Fe60w[:,0]*1.e9, Fe60w[:,2], '+', label = r'$^{60}$Fe LMP($T,\rho$)')
    ax.plot(Fe59w[:,0]*1.e9, Fe59w[:,2], 'x', label = r'$^{59}$Fe LMP($T,\rho$)')


    ax.legend(loc='best')
    plt.draw()

    # print('{:15s} {:15s} {:15s}'.format('T', 'Fe59', 'Fe60'))
    # for T, R59, R60 in zip(t, r59, r60):
    #     print('{:15.10e} {:15.10e} {:15.10e}'.format(T, R59, R60))

def read_kepler_wrate_debug(filename, A = None, Z = None):
    # [DEBUG] (Z,A),t9,rho,rate=(          26 ,          59 )   1.4778503000731893        287968.52279572235        5.8227720038196049E-006

    data = []
    with open(filename, 'rt') as f:
        for line in f:
            if not line.startswith(' [DEBUG] (Z,A),t9,rho,rate=('):
                continue
            if Z is not None and Z != int(line[30:40]):
                continue
            if A is not None and A != int(line[44:54]):
                continue
            data += [[float(line[58+26*i:84+26*i]) for i in range(3)]]
    return np.array(data)


def decay_plot():
    t = np.array([1.49e6, 2.62e6])  # in Myr
    l = np.log(2) / (t * SEC)

    x = 1E6 * np.linspace(0, 20, 1000) * SEC

    f = plt.figure()
    ax = f.add_subplot(111)

    for lx,tx in zip(l,t):
        ax.plot(x * 1E-6 / SEC, lx * np.exp(-x*lx), label = r'$\tau_{{1/2}} = {:4.2f}\,\mathrm{{Myr}}$'.format(tx * 1e-6))

    ax.legend(loc='best')

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel('time / Myr')
    ax.set_ylabel('rate / Hz / atom')

    plt.draw()
