import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import os.path
from color import IsoColorRainbow

import kepdump
from physconst import XMSUN, RSUN

base_path = '~/Projjwal'

dump_lib = {
    'z25a' : 'z25/rot50_mlossWR_magnet/z25#presn',
    'z25b' : 'z25/rot50_mloss_modWRZm3_magnet/z25#presn',
    'z25c' : 'z25/rot50/z25#presn',
    'z25d' : 'z25/rot70/z25#presn',
    'z25e' : 'z25/rot90/z25#presn',
    'v25a' : 'v25/rot50_mlossWR_magnet/v25#presn',
    'v25b' : 'v25/rot30_mlossWR_magnet/v25#presn',
    'v25c' : 'v25/rot40_mlossWR_magnet/v25#presn',
    'v25d' : 'v25/reduced_mloss/rot30_mlossWR_magnet/v25#presn',
    'v25e' : 'v25/reduced_mloss/rot40_mlossWR_magnet/v25#presn',
    'v25f' : 'v25/reduced_mloss/rot50_mlossWR_magnet/v25#presn',
    'v25g' : 'v25/reduced_mloss/rot60_mlossWR_magnet/v25#presn',
    'v25k' : 'v25/rot60_mlossWR_magnet/v25#presn',
    'v15a' : 'v15/rot50_mlossWR_magnet/v15#presn',
    'v15b' : 'v15/reduced_mloss/rot50_mlossWR_magnet/v15#presn',
    'v15c' : 'v15/reduced_mloss/rot40_mlossWR_magnet/v15#presn',
    'v15d' : 'v15/reduced_mloss/rot30_mlossWR_magnet/v15#presn',
}

class Plot(object):
    def __init__(self, *args, **kwargs):
        self.dumps = dict()
        for key,filename in dump_lib.items():
            filename = os.path.join(base_path, filename)
            dump = kepdump.load(filename)
            self.dumps[key] = dump
        try:
            self.plot(*args, **kwargs)
        except:
            print('error plotting')

    def plot(self, zoom = None, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        col = IsoColorRainbow(len(self.dumps))
        for c,(key,dump) in zip(col, self.dumps.items()):
            ii = slice(1,-1)
            m = dump.zmm_sun[ii]
            j = dump.angjn[ii]
            ax.plot(m, j, label = key, color = c)
        ax.set_yscale('log')

        if zoom == True:
            ax.set_xlim(0, 6.1)
            ax.set_ylim(5e13, 2e17)
        else:
            ax.set_xlim(0, None)
            ax.set_ylim(5e13, 2e19)

        ax.set_xlabel('enclosed mass / solar masses')
        ax.set_ylabel(r'specific angular momentum / $\mathrm{erg}\,\mathrm{s}\,\mathrm{g}^{-1}$')
        ax.legend(loc = 'best', ncol=2)
        fig.tight_layout()
        self.fig = fig
        self.ax = ax

        self.savefig(**kwargs)

    def savefig(self, savefig = None, **kwargs):
        if savefig is not None:
            kw = self.__dict__.copy()
            kw.update(kwargs)
            savefig = savefig.format(**kw)
            savefig = os.path.expanduser(savefig)
            savefig = os.path.expandvars(savefig)
            self.fig.savefig(savefig)


class JPlot(object):
    """
    classical j plot for KEPLER / GRB
    """
    def __init__(self, dump, *args, **kwargs):
        if isinstance (dump, str):
            if dump in dump_lib:
                filename = dump_lib[dump]
                filename = os.path.join(base_path, filename)
            elif os.path.isfile(dump):
                filename = dump
            dump = kepdump.load(filename)
        self.dump = dump
        try:
            self.plot(*args, **kwargs)
        except:
            print('error plotting')


    def plot(self, zoom = False, **kwargs):
        dump = self.dump
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ii = slice(1,-1)
        m = dump.zmm_sun[ii]
        jeq = dump.angjeqn[ii]
        ax.plot(m, jeq, label = r'$j_\mathrm{equator}$', color = 'k', lw=3)
        jkerr = dump.anglstk[ii]
        ax.plot(m, jkerr, label = r'$j_\mathrm{LSO,Kerr}$', color = 'g', lw=1)
        jsch = dump.anglsts[ii]
        ax.plot(m, jsch, label = r'$j_\mathrm{LSO,Schwarzschild}$', color = 'b', lw=1)
        jlst = dump.anglstn[ii]
        ax.plot(m, jlst, label = r'$j_\mathrm{LSO}(J)$', color = 'r', lw=1)


        ax.set_yscale('log')

        if zoom == True:
            ax.set_xlim(0, 6.1)
            ax.set_ylim(2e14, 2e17)
        else:
            ax.set_xlim(0, None)
            ax.set_ylim(5e13, 2e19)

        ax.set_xlabel('enclosed mass / solar masses')
        ax.set_ylabel(r'specific angular momentum / $\mathrm{erg}\,\mathrm{s}\,\mathrm{g}^{-1}$')
        legend = ax.legend(loc = 'lower right')
        legend.set_draggable(True)
        fig.tight_layout()
        self.fig = fig
        self.ax = ax

        self.savefig(**kwargs)


    def savefig(self, savefig = None, **kwargs):
        if savefig is not None:
            kw = self.__dict__.copy()
            kw.update(kwargs)
            savefig = savefig.format(**kw)
            savefig = os.path.expanduser(savefig)
            savefig = os.path.expandvars(savefig)
            self.fig.savefig(savefig)

from kepdata import kepdata

class Dumps():
    def __init__(self,
                 target = '/m/web/Download/james',
                 ):
        for id, filename in dump_lib.items():
            infile = os.path.join(base_path, filename)
            outfile = os.path.join(target, id + '@presn')
            kepdata(
                filename=infile,
                outfile=outfile,
                )
