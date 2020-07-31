"""
routines for dark photons from pair SN studies
"""

import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl

import convdata
from physconst import NA

class TcNec(object):
    def __init__(self,
                 filename = '/home/alex/kepler/psn/hex100.cnv'):
        self.cnv = convdata.load(filename)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        c = self.cnv
        ii = np.argmax(c.tc)
        time = c.time - c.time[ii]
        ax.plot(time, c.dc*c.ye, label = r'$n_\mathrm{e,c}$')
        ax.plot(time, c.tc, label = r'$T_\mathrm{c}$')
        ax.set_yscale('log')
        ax.set_xscale('symlog')
        ax.set_xlabel(r'time relative to maximum $T_\mathrm{c}$ (s)')
        ax.set_ylabel(r'$T_\mathrm{c}$ (K), $n_\mathrm{e,c}=\rho_\mathrm{c}\,Y_\mathrm{e,c}$ ($N_\mathrm{A}\,\mathrm{cm}^{-3}$) ')
        ax.legend(loc = 'best')
        ax.set_ylim((1,1e10))
        ax.set_xlim((-1e7,1e5))
        fig.tight_layout()

        self.ax = ax
        self.figure = fig

    def table(self, filename = 'TcNec.txt'):
        ncol = 4
        c = self.cnv
        ii = np.argmax(c.tc)
        time = c.time - c.time[ii]
        with open(filename, 'wt') as f:
            f.write(('{:>13s}' * ncol + '\n').format(
                'time (s)',
                'T (K)',
                'rho (gcc)',
                'ne (1/cc)'))
            for x in zip(time, c.tc, c.dc, c.dc * c.dc * NA):
                f.write(('{:12.5e} ' * ncol).format(*x) + '\n')
