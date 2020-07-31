import numpy as np
import matplotlib.pylab as plt

import os.path

from numpy.linalg import norm

import physconst

from rotdata import RotData, RotRecord
from dsiplot import fw_min

from movie.mplbase import MPLBase
from movie.writers import make

from human import time2human

class RotPlot(object):
    def __init__(self, rot = None, model = None):
        if isinstance(rot, str):
            rot = RotData(rot)
        self.rot = rot

        if model is not None:
            self.get_rec(model)

    def get_rec(self, model = 0):
        self.index = model
        self.rec = self.rot.data[self.index]

    def plot_conv_layer(self, ax, index, label, color, all = False):
        zm_sun = self.rec.zm_sun

        # convection plotting with KEPLER centering
        iconv = self.rec.iconv == index
        ii = np.where(iconv[:-1] != iconv[1:])[0]
        for i in range(0,len(ii),2):
            z0 = 0.5 * (zm_sun[ii[i  ]] + zm_sun[ii[i  ]+1])
            z1 = 0.5 * (zm_sun[ii[i+1]] + zm_sun[ii[i+1]+1])
            if i == 0:
                l = label
            else:
                l = None
            ax.axvspan(z0, z1, color=color, label = l)
        if len(ii) == 0 and all:
            ax.axvspan(np.nan, np.nan, color=color, label = label)

        # datlabel= '0=rad 1=neut 2=osht 3=semi 4=conv 5=thal'

    def plot_conv(self, ax, all = True):
        conv = ( # old
            (5, 'neutral',    '#BFFFFF'),
            (3, 'overshoot',  '#FFBFFF'),
            (2, 'semiconv',   '#FFBFBF'),
            (1, 'convection', '#BFFFBF'),
            (4, 'thermohal',  '#FFFFBF'),
            )
        # conv = ( # new
        #      (1, 'neutral',    '#BFFFFF'),
        #      (2, 'overshoot',  '#FFBFFF'),
        #      (3, 'semiconv',   '#FFBFBF'),
        #      (4, 'convection', '#BFFFBF'),
        #      (5, 'thermohal',  '#FFFFBF'),
        #      )

        for args in conv:
            self.plot_conv_layer(ax, *args, all = all)

    def get_label(self):
        timecc = time2human(self.rot.data[-1].time - self.rec.time + 0.25)
        timect = time2human(self.rec.time)
        label = f'time since birth = {timect}, time to collapse = {timecc}, model = {self.rec.ncyc:05d}'
        return label

    def plot_n2x(self, model = None, fig = None, ax = None):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)

        if model is not None:
            self.get_rec(model)

        n2x = self.rec.n2x
        zm_sun = self.rec.zm_sun
        n2dyn = self.rec.n2dyn

        ax.plot(zm_sun[1:-2], +n2x[1:-2], label=r'$>0$')
        ax.plot(zm_sun[1:-2], -n2x[1:-2], label=r'$<0$')
        ax.set_yscale('log')

        ax.plot(zm_sun[1:-2], n2dyn[1:-2], label='dyn')

        self.plot_conv(ax)

        ax.set_xlabel('enclosed mass / solar masses')
        ax.set_ylabel(r'$N^2_{\mathrm{X}}$')
        ax.set_ylim(1e-15, 1e3)
        ax.text(0.99,0.99, self.get_label(), transform=ax.transAxes, ha='right',va='top')
        ax.legend(loc='best')

        fig.tight_layout()

        self.fig = fig
        self.ax = ax

        return self


    def plot_n2(self, model = None, fig = None, ax = None):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        if model is not None:
            self.get_rec(model)

        n2 = self.rec.n2
        zm_sun = self.rec.zm_sun
        n2dyn = self.rec.n2dyn

        ax.plot(zm_sun[1:-2], +n2[1:-2], label=r'$>0$')
        ax.plot(zm_sun[1:-2], -n2[1:-2], label=r'$<0$')
        ax.set_yscale('log')

        ax.plot(zm_sun[1:-2], n2dyn[1:-2], label='dyn')

        self.plot_conv(ax)

        ax.set_xlabel('enclosed mass / solar masses')
        ax.set_ylabel(r'$N^2$')
        ax.set_ylim(1e-15, 1e3)
        ax.text(0.99,0.99, self.get_label(), transform=ax.transAxes, ha='right',va='top')
        ax.legend(loc='best')

        fig.tight_layout()

        self.fig = fig
        self.ax = ax

        return self


    def plot_w(self, model = None, fig = None, ax = None):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        if isinstance(model, RotRecord):
            self.rec = model
        if model is not None:
            self.get_rec(model)

        angw = self.rec.angw
        wa = self.rec.angwa
        zmm_sun = self.rec.zmm_sun

        c = ('#FF7F00', '#FF007F', '#FF0000',)
        l = ('-', ':',)
        lb = ('x', 'y', 'z',)
        sym = r'\omega'

        ii = slice(1,-1) # zone centres

        ax.plot(zmm_sun[ii], wa[ii], label=fr'${sym}$', color='#DfDfDf', lw=5)
        for i in range(3):
            ax.plot(zmm_sun[ii], +angw[ii, i], label=fr'${sym}_{lb[i]}$', ls = l[0], color = c[i])
            ax.plot(zmm_sun[ii], -angw[ii, i], ls = l[1], color = c[i])

        self.plot_conv(ax)

        ax.set_xlabel('enclosed mass / solar masses')
        ax.set_ylabel(r'angular velocity / rad s$^{-1}$')
        ax.legend(loc='best')
        ax.set_yscale('log')
        ax.set_ylim(1.e-10, 10)
        ax.text(0.99,0.99, self.get_label(), transform=ax.transAxes, ha='right',va='top')
        ax.legend(loc='best')

        fig.tight_layout()

        self.fig = fig
        self.ax = ax

        return self


    def plot_pt(self, model = None, fig = None, ax = None, axb = None):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        if axb is None:
            axb = ax.twinx()
        if isinstance(model, RotRecord):
            self.rec = model
        if model is not None:
            self.get_rec(model)

        pw = self.rec.angpw
        tw  = self.rec.angtwdw * 180 / np.pi
        zmm_sun = self.rec.zmm_sun
        wn = self.rec.angnw

        ii = slice(1,-2) # internal zone interfaces

        ax.plot(zmm_sun[ii], np.log10(pw[ii]), label=r'$\log\,t$', color='red', lw=1)
        ax.plot(zmm_sun[ii], np.log10(wn[ii]), label=r'$\log\,\left(N^2/\Omega^2\right)$', color='green', lw=1)
        ax.plot(zmm_sun[ii], np.log10(-wn[ii]), color='green', lw=1, ls=':')

        axb.plot(zmm_sun[ii], tw[ii], color='blue', lw=1)
        ax.plot([np.nan], label=r'$\breve\theta$', color='blue', lw=1)


        self.plot_conv(ax)

        ax.set_xlabel('enclosed mass / solar masses')
        ax.set_ylabel(r'$\log\,t$, $\log\,\left(N^2/\Omega^2\right)$')
        axb.set_ylabel(r'$\breve\theta$')
        ax.legend(loc='best')
        #ax.set_yscale('log')
        ax.set_ylim(-6,12)
        axb.set_ylim(-5, 200)
        ax.text(0.99,0.99, self.get_label(), transform=ax.transAxes, ha='right',va='top')
        legend = ax.legend(loc='best')
        legend.draggable = True

        fig.tight_layout()

        self.fig = fig
        self.ax = ax

        return self


    def plot_d(self, model = None, fig = None, ax = None):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        if isinstance(model, RotRecord):
            self.rec = model
        if model is not None:
            self.get_rec(model)

        fjc = 1

        difi = self.rec.difi
        angd = self.rec.angd
        zmm_sun = self.rec.zmm_sun

        jmd = angd.shape[1]

        c = ('#FF0000', '#FFCF00', '#00DFFF', '#00BFFF', '#007FFF',)
        l = ('-',)
        lb = ('1', '2', '3', '4', '5',)
        sym = r'D'

        ii = slice(1,-1) # zone centres

        ax.plot(zmm_sun[ii], difi[ii], label=fr'${sym}$', color='#DfDfDf', lw=5)
        for i in range(jmd):
            ax.plot(zmm_sun[ii], +angd[ii, i], label=fr'${sym}_{lb[i]}$', ls = l[0], color = c[i])

        self.plot_conv(ax)

        ax.set_xlabel('enclosed mass / solar masses')
        ax.set_ylabel(r'$D\,\left(\mathrm{cm}^2\,s^{-1}\right)$')
        ax.legend(loc='best')
        ax.set_yscale('log')
        #ax.set_ylim(-6, 12)
        ax.text(0.99, 0.99, self.get_label(), transform=ax.transAxes, ha = 'right',va = 'top')
        legend = ax.legend(loc = 'best')
        legend.draggable = True

        fig.tight_layout()

        self.fig = fig
        self.ax = ax

        return self


    def movie(self,
            mode = 'w',
            moviedir = '~/Downloads',
            movieext = 'mp4',
            filename = None,
            **kwargs,
            ):
        kwargs.setdefault('size', (800, 600))
        kwargs.setdefault('dpi', 100)
        kwargs.setdefault('framerate', 60)
        if filename is None:
            filename = os.path.splitext(os.path.basename(self.rot.filename))[0]
            filename = os.path.join(moviedir, filename + '.' + movieext)
        make(
            True,
            filename,
            canvas = MPLBase,
            func = getattr(self, f'plot_{mode}'),
            enum = 'model',
            values = self.rot.nmodels,
            **kwargs,
            )
