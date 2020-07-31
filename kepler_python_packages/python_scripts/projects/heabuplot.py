import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import convdata
import physconst
import color

def plot(cnv = None):
    filename = '/home/alex/kepler/test/s15.cnv.xz'
    if cnv is None:
        cnv = convdata.load(filename)
    x = cnv.net
    t = cnv.time / physconst.SEC * 1e-3
    he = x['He4']
    c = x['c12']
    o = x['o16']

    X = (he, c, o)
    L = (r'$^{4}\mathrm{He}$',
         r'$^{12}\mathrm{C}$',
         r'$^{16}\mathrm{O}$',
         )

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlabel(r'time / kyr')
    ax.set_ylabel(r'mass fraction')
    ax.set_xlim(11290,13245)
    ax.set_ylim(-0.01,1.01)

    C = color.isocolors(3)
    for x,l,c in zip(X,L,C):
        ax.plot(t, x, color = c, label = l)

    ax.legend(loc = 'best', ncol=2, fontsize=15)
    f.tight_layout()
    plt.draw()

    return cnv
