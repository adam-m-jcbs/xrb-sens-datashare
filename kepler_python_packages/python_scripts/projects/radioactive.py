"""
extract Al26 and Fe60 data

examples:

from projects.radioactive import PlotYield as Y
y = Y('no-cutoff')
y.plot_m('~/Downloads/FeAl_m_{name}.pdf')

"""




import os.path

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

from matplotlib.ticker import FixedLocator

import kepdump
import stardb

from color import IsoColorRainbow
from physconst import XMSUN, RSUN, NA

from stardb import StarDB

db_lib = {
    'ertl' : '/home/alex/NuGrid/GCE/GG-ertl-radioactive.stardb.xz',
    'no-cutoff' : '/home/alex/NuGrid/GCE/GG-no_cutoff-radioactive.stardb.xz',
    'xi45' : '/home/alex/NuGrid/GCE/GG-xi45-radioactive.stardb.xz',
    'xi25' : '/home/alex/NuGrid/GCE/GG-xi25-radioactive.stardb.xz',
    'mu16' : '/home/alex/NuGrid/GCE/GG-mu16-radioactive.stardb.xz',
    'mc16' : '/home/alex/NuGrid/GCE/GG-mc16-radioactive.stardb.xz',
    }

class PlotYield(object):
    def __init__(self, db = 'ertl', name = None):
        if isinstance(db, str):
            if name is None:
                name = db
            db = db_lib.get(db, db)
            if os.path.isfile(db):
                db = stardb.load(db)
        if not isinstance(db, StarDB):
            raise Exception(f'Unknown DB {db!r}')
        if name is None:
            name = db.name
        db.setup_logger(silent=False)
        self.name = name
        self.db = db
        self.figures = []

    @staticmethod
    def plotall(base = '~/Downloads', closefigs = True, selection = None):
        for db in db_lib:
            y = PlotYield(db)
            if selection is None or 'z' in selection:
                y.plot_z(os.path.join(base, 'West_FeAl_z_{name}.pdf'))
            if selection is None or 'm' in selection:
                y.plot_m(os.path.join(base, 'West_FeAl_m_{name}.pdf'))
            if selection is None or 'i' in selection:
                y.plot_imf(os.path.join(base, 'West_FeAl_imf_{name}.pdf'))
            if closefigs:
                y.close_figures()

    def savefig(self, savefig = None, **kwargs):
        if savefig is not None:
            kw = self.__dict__.copy()
            kw.update(kwargs)
            savefig = savefig.format(**kw)
            savefig = os.path.expanduser(savefig)
            savefig = os.path.expandvars(savefig)
            self.fig.savefig(savefig)

    def close_figures(self):
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()

    def plot_m(self, savefig = None):
        db = self.db
        ife = np.where(db.ions == 'fe60')[0][0]
        ial = np.where(db.ions == 'al26')[0][0]

        iiz = np.where(db.fieldnames == 'metallicity')[0][0]
        iim = np.where(db.fieldnames == 'mass')[0][0]

        metallicities = db.values[iiz,:db.nvalues[iiz]]
        masses = db.values[iim,:db.nvalues[iim]]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        col = IsoColorRainbow(len(metallicities))
        for z,c in zip(metallicities, col):
            iz = db.get_star_slice(metallicity=z)
            ej =  (db.fielddata[iz]['mass'] - db.fielddata[iz]['remnant']) * XMSUN
            m = db.fielddata[iz]['mass']
            fe = db.data[iz,ife] * ej
            al = db.data[iz,ial] * ej
            with np.errstate(divide = 'ignore'):
                ax.plot(m, fe, color = c, ls = '-', label = fr'$[Z]={np.log10(z/0.0153):+3.1f}$')
            ax.plot(m, al, color = c, ls = ':')

        ax.set_yscale('log')
        ax.set_ylim(2.e24, 3.e28)
        ax.set_ylabel('ejecta including wind / mol')

        ax.set_xlabel('initial mass / solar masses')
        ax.xaxis.set_major_locator(FixedLocator(masses))

        leg1 = ax.legend(loc='best')

        lines = [mpl.lines.Line2D([0], [0], color='k', ls='-'),
                 mpl.lines.Line2D([0], [0], color='k', ls=':')]
        leg2 = ax.legend(lines,['$^{60}$Fe', '$^{26}$Al'], loc='upper left')
        ax.add_artist(leg1)

        ax.text(0.5, 0.99, self.name, ha='center', va='top', transform=ax.transAxes)

        fig.tight_layout()
        self.fig = fig
        self.ax = ax
        self.figures.append(self.fig)
        self.savefig(savefig)

    def plot_z(self, savefig = None):
        db = self.db
        ife = np.where(db.ions == 'fe60')[0][0]
        ial = np.where(db.ions == 'al26')[0][0]

        iim = np.where(db.fieldnames == 'mass')[0][0]
        masses = db.values[iim,:db.nvalues[iim]]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        col = IsoColorRainbow(len(masses))
        for m,c in zip(masses, col):
            im = db.get_star_slice(mass=m)
            ej =  (db.fielddata[im]['mass'] - db.fielddata[im]['remnant']) * XMSUN
            z = db.fielddata[im]['metallicity']
            lzsun = np.log10(np.maximum(1.e-12, z / 0.0153))
            fe = db.data[im,ife] * ej
            al = db.data[im,ial] * ej
            ax.plot(lzsun, fe, color = c, ls = '-', label = fr'$M={m:2.0f}\,\mathrm{{M}}_\odot$')
            ax.plot(lzsun, al, color = c, ls = ':')

        ax.set_yscale('log')
        ax.set_ylim(2.e24, 3.e28)
        ax.set_xlim(-6, 1)
        ax.set_ylabel('ejecta including wind / mol')
        ax.set_xlabel('[Z] ')

        leg1 = ax.legend(loc='best')

        lines = [mpl.lines.Line2D([0], [0], color='k', ls='-'),
                 mpl.lines.Line2D([0], [0], color='k', ls=':')]
        leg2 = ax.legend(lines,['$^{60}$Fe', '$^{26}$Al'], loc='upper left')
        ax.add_artist(leg1)

        ax.text(0.5, 0.99, self.name, ha='center', va='top', transform=ax.transAxes)

        fig.tight_layout()
        self.fig = fig
        self.ax = ax
        self.figures.append(self.fig)
        self.savefig(savefig)

    def plot_imf(self, savefig = None, return_data = False, fig = None, ax = None):
        db = self.db
        ife = np.where(db.ions == 'fe60')[0][0]
        ial = np.where(db.ions == 'al26')[0][0]

        iiz = np.where(db.fieldnames == 'metallicity')[0][0]
        metallicities = db.values[iiz,:db.nvalues[iiz]]

        mlo = 10
        mhi = 120
        gamma = -1.35

        jfe = 0
        jal = 1
        norm = mlo**(gamma + 1) - mhi**(gamma + 1)
        yields = np.ndarray(metallicities.shape + (2,))
        for i, z in enumerate(metallicities):
            iz = db.get_star_slice(metallicity=z)
            m = db.fielddata[iz]['mass']
            r = db.fielddata[iz]['remnant']
            ej =  (m - r) / m * XMSUN
            fe = db.data[iz,ife] * ej
            al = db.data[iz,ial] * ej
            # may need sorting ...
            bounds = np.ndarray(m.shape[0] + 1)
            bounds[0] = mlo
            bounds[1:-1] =  0.5 * (m[1:] + m[:-1])
            bounds[-1] = mhi
            weights = bounds[:-1]**(gamma + 1) - bounds[1:]**(gamma + 1)
            yields[i, jfe] = np.sum(fe * weights)
            yields[i, jal] = np.sum(al * weights)
        yields /= norm

        if return_data:
            return metallicities, yields

        # make the plot
        lzsun = np.log10(np.maximum(1.e-12, metallicities / 0.0153))
        c = 'k'
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        ax.plot(lzsun, yields[:,jfe], color = c, ls = '-', label = '$^{60}$Fe')
        ax.plot(lzsun, yields[:,jal], color = c, ls = ':', label = '$^{26}$Al')

        ax.set_xlabel('[Z] ')
        ax.set_xlim(-4.5, 0.5)

        ax.set_yscale('log')
        #ax.set_ylim(2.e24, 3.e28)
        ax.set_ylabel('ejecta including wind / mol per inital solar mass')

        ax.legend(loc='best')
        ax.text(0.5, 0.99, self.name, ha='center', va='top', transform=ax.transAxes)

        fig.tight_layout()
        self.fig = fig
        self.ax = ax
        self.figures.append(self.fig)
        self.savefig(savefig)

    @classmethod
    def plot_imf_db(cls, savefig = None, sel = None, zoom = False, ratio = False):
        if sel is None:
            # sel = db_lib.keys()
            sel = ['no-cutoff', 'xi45', 'xi25', 'ertl', 'mu16', 'mc16']
        sel = list(sel)

        col = IsoColorRainbow(len(sel))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, key in enumerate(sel):
            try:
                sel[i] = cls(key)
            except:
                pass

        jfe = 0
        jal = 1

        for c, db in zip(col, sel):
            z, y = db.plot_imf(return_data = True)
            lzsun = np.log10(np.maximum(1.e-12, z / 0.0153))
            if ratio:
                ax.plot(lzsun, y[:,jfe]/y[:,jal], color = c, ls = '-', label = db.name)
            else:
                ax.plot(lzsun, y[:,jfe], color = c, ls = '-', label = db.name)
                ax.plot(lzsun, y[:,jal], color = c, ls = ':')

        ax.set_xlabel('[Z] ')
        if ratio:
            #ax.set_xlim(-0.85, 0.35)
            #ax.set_ylim(0, 2.1)
            ax.set_xlim(-4.5, 0.5)
            ax.set_yscale('log')

            ax.set_ylabel('$^{60}$Fe / $^{26}$Al number ratio of ejecta including wind')
        else:
            ax.set_xlim(-4.5, 0.5)
            ax.set_yscale('log')

            ax.set_ylabel('ejecta including wind / mol per inital solar mass')

        loc2 = 'center left'

        if zoom:
            if ratio:
                ax.set_ylim(0.02, 3.5)
                ax.set_yscale('log')
            else:
                ax.set_xlim(-0.85, 0.35)
                ax.set_ylim(3.e24, 4.99e26)

            loc2 = 'lower center'

        leg1 = ax.legend(loc='best')

        if not ratio:
            lines = [mpl.lines.Line2D([0], [0], color='k', ls='-'),
                     mpl.lines.Line2D([0], [0], color='k', ls=':')]
            leg2 = ax.legend(lines,['$^{60}$Fe', '$^{26}$Al'], loc=loc2)
            ax.add_artist(leg1)

        fig.tight_layout()

        db.fig = fig
        db.ax = ax
        db.figures.append(fig)
        db.savefig(savefig)
