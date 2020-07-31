from math import floor, ceil, log10
from matplotlib.ticker import MaxNLocator

from . import gridscale

import numpy as np

from .basekep import BasePlotKep, BasePlotKepAxes

from physconst import Kepler as const

import isotope
import ionmap
from abusets import SolAbu

from utils import index1d, project

class Plot1(BasePlotKepAxes):

    colors = dict(
        dn = 2,
        un = 26,
        vn = 20,
        tn = 4,
        rn = 22,
        pn = 7,
        snu = 11,
        snn = 18,
        snl = 10,
        ni = 16,
        sag = 15,
        sal = 13,
        svg = 28,
        svl = 27,
        snw = 16,
        wsi = 23,
        wsd = 25,
        )

    def __init__(self, *args, **kwargs):

        kwargs['ylabel'] = r'$\rho$ (g/ccm), $P$ (erg/cc), $S$ (erg/g/s)'
        kwargs['yscale'] = 'log'

        kwargs['ylabel2'] = 'T (K), R (cm), v (cm/s)'
        kwargs['yscale2'] = 'log'

        super().__init__(*args, **kwargs)


    def draw(self):
        self.setup_logger(silent = False)

        self.clear()
        self.setx()

        xvar = self.xvar
        data = self.data

        iib = xvar.iib
        iim = xvar.iim
        x = xvar.x[iib]
        xm = xvar.xm[iim]

        dn = data.dn[iim]
        dn0 = floor(log10(np.min(dn)))
        dn1 = ceil(log10(np.max(dn)))
        self.ax.plot(xm, dn, color = self.colors['dn'], label = r'$\rho$')

        # compute pressure scale
        pn = data.pn[iim]
        pn0 = floor(log10(np.min(pn)))
        pn1 = ceil(log10(np.max(pn)))

        pscale = round(0.5 * ((pn1 + pn0) - (dn1 + dn0)))
        self.ax.plot(xm, pn * 0.1**pscale, color = self.colors['pn'], label = 'P')

        ylim = np.array([min(dn0, pn0 - pscale), max(dn1, pn1 - pscale)])

        # add energy generation
        sn = data.sn[iim]
        snn = data.snn[iim]

        # nu loss
        snu = np.zeros_like(sn)
        ii = sn < snn
        snu[ii] = snn[ii]-sn[ii]

        # Ni decay
        sni = np.zeros_like(sn)
        ii = np.logical_not(ii)
        sni[ii] = sn[ii] - snn[ii]

        # advection
        sadv = data.sadv[iim]

        # shear
        sv = data.sv[iim]

        sn1 = max(
            np.max(np.abs(snn)),
            np.max(snu),
            np.max(sni),
            np.max(np.abs(sadv)),
            np.max(np.abs(sv)),
            )

        # WIMP
        if data.wimp > 0:
            snw = data.snw[iib]
            snwcrsi = data.snwcrsi[iib]
            snwcrsd = data.snwcrsd[iib]
            sn1 = max(
                sn1,
                np.max(snw),
                np.max(snwcrsi),
                np.max(snwcrsd),
                )

        if sn1 > 1.e-99:
            sn1 = ceil(log10(sn1))
            sscale = sn1 - ylim[1]
        else:
            sn1 = 1
            sscale = 0

        sfac = 0.1**sscale
        scutoff = 10.**(ylim[0]+sscale)
        if np.max(snn) > scutoff:
            self.ax.plot(xm, snn * sfac, color = self.colors['snn'], label = 'S')
        if np.max(-snn) > scutoff:
            self.ax.plot(xm, -snn * sfac, color = self.colors['snl'], label = 'L')
        if np.max(snu) > scutoff:
            self.ax.plot(xm, snu * sfac, color = self.colors['snu'], label = r'$\nu$')
        if np.max(sni) > scutoff:
            self.ax.plot(xm, sni * sfac, color = self.colors['ni'], label = 'D')

        # advection
        if np.max(sadv) > scutoff:
            self.ax.plot(xm, sadv * sfac, color = self.colors['sag'], label = r'A$_\mathrm{d}$')
        if np.max(-sadv) > scutoff:
            self.ax.plot(xm, -sadv * sfac, color = self.colors['sal'], label = r'A$_\mathrm{l}$')

        # shear (rotation) [this should never be negative]
        if np.max(sv) > scutoff:
            self.ax.plot(xm, sv * sfac, color = self.colors['svg'], label = r'V$_\mathrm{d}$')
        if np.max(-sv) > scutoff:
            self.ax.plot(xm, -sv * sfac, color = self.colors['svl'], label = r'V$_\mathrm{l}$')

        # WIMP
        if data.wimp > 0:
            if np.max(snw) > scutoff:
                self.ax.plot(xm, snw * sfac, color = self.colors['snw'], label = r'S$_\mathrm{w}$')
            if np.max(snwcrsi) > scutoff:
                self.ax.plot(xm, snwcrsi * sfac, color = self.colors['wsi'], label = r'C$_\mathrm{i}$')
            if np.max(snwcrsd) > scutoff:
                self.ax.plot(xm, snwcrsd * sfac, color = self.colors['wsd'], label = r'C$_\mathrm{d}$')

        # finally, set scales and labels
        ylim = 10.**ylim
        self.ax.set_ylim(*ylim)

        ylabel = r'$\rho$ (g/cc), $P$ ($10^{{{}}}$ erg/cc), $S$ ($10^{{{}}}$ erg/g/s)'.format(pscale, sscale)
        self.ax.set_ylabel(ylabel)

        # temperature and its axis
        tn = data.tn[iim]
        self.ax2.plot(xm, tn, color = self.colors['tn'], label = 'T')

        tn0 = floor(log10(np.min(tn)))
        tn1 = ceil(log10(np.max(tn)))
        ylim2 = np.array([tn0, tn1])

        # add radius
        rn = data.rn[iib]
        rn1 = ceil(log10(np.max(rn)))
        rscale = rn1 - ylim2[1]
        self.ax2.plot(x, rn * 0.1**rscale, color = self.colors['rn'], label = 'R')

        # velocity - refine scaling algorithm
        voffset = 1 # show lowest voffset dex in radius plot
        un = data.un[iib]
        un1 = max(np.max(un), np.max(-un))
        if un1 > 1.e-99:
            un1 = ceil(log10(un1))
            vscale = un1 - ylim2[0]
        else:
            un1 = 1
            vscale = 0
        vscale -= voffset
        vfac = 0.1**vscale
        vcutoff = 10.**(vscale + ylim2[0])
        if np.max(un) > vcutoff:
            self.ax2.plot(x, un * vfac, color = self.colors['vn'], label = 'v')
        if np.max(-un) > vcutoff:
            self.ax2.plot(x, -un * vfac, color = self.colors['un'], label = 'u')

        # yaxis2 labels
        ylabel2 = r'$T$ (K), $R$ ($10^{{{}}}$ cm), $v$ ($10^{{{}}}$ cm/s)'.format(rscale, vscale)
        self.ax2.set_ylabel(ylabel2)

        ylim2 = 10.**ylim2
        self.ax2.set_ylim(*ylim2)

        self.decorations()

        self.close_logger(timing = 'Plot finished in')


class Plot2(BasePlotKepAxes):

    colors = dict(
        un = 7,
        )

    def __init__(self, *args, **kwargs):

        kwargs.setdefault('annotate', None)
        kwargs.setdefault('showconv', None)

        kwargs['ylabel'] = r'velocity (cm/s)'
        kwargs['yscale'] = 'linear'

        super().__init__(*args, **kwargs)

    def draw(self):
        self.setup_logger(silent = False)

        # clear old stuff
        self.clear()

        self.setx()

        xvar = self.xvar
        data = self.data

        # use parm 191 for ylim
        if data.vlimset > 0:
            self.ax.set_yscale('linear')
            self.ax.set_ylim(np.array([-1,1]) * data.vlimset)
        elif data.vlimset < 0:
            self.ax.set_yscale('linear')
            self.ax.autoscale(axis='y')
        else:
            self.ax.set_yscale('symlog')
            self.ax.autoscale(axis='y')

        iib = xvar.iib
        x = xvar.x[iib]
        un = data.un[iib]

        self.ax.plot(x, un, color = self.colors['un'], label = r'u')

        self.decorations()
        self.close_logger(timing = 'Plot finished in')

class Plot5(BasePlotKepAxes):

    colors = dict(
        tn = 22,
        dn = 4,
        )

    def __init__(self, *args, **kwargs):

        kwargs.setdefault('showconv', None)

        kwargs['ylabel'] = r'density (g/cm$^3$)'
        kwargs['yscale'] = 'log'
        kwargs['ylabel2'] = r'temperature (K)'
        kwargs['yscale2'] = 'log'

        super().__init__(*args, **kwargs)

    def draw(self):
        self.setup_logger(silent = False)

        # clear old stuff
        self.clear()
        self.setx()

        xvar = self.xvar
        data = self.data

        iim = xvar.iim
        xm = xvar.xm[iim]
        dn = data.dn[iim]
        tn = data.tn[iim]

        self.ax.plot(xm, dn, color = self.colors['dn'], label = r'$\rho$')
        self.ax2.plot(xm, tn, color = self.colors['tn'], label = r'T')

        self.decorations()
        self.close_logger(timing = 'Plot finished in')

class Plot78Base(BasePlotKepAxes):

    colors = dict(
        angwn = 24,
        angjn = 4,
        angj = (22,22,26,26,24,24),
        angw = (10,10,5,5,4,4),
        visc = 1,
        br = 13,
        bt = 3,
        difi = 16,
        angd1 = 12,
        angd2 = 10,
        angd3 = 9,
        angd4 = 20,
        angd5 = 7,
        bfvisc = 5,
        )

    linestyles = dict(
        angw = ('-',':','-',':','-',':'),
        angj = ('-',':','-',':','-',':'),
        )

    labels = dict(
        vec = ('x','y','z')
        )

    def __init__(self, *args, **kwargs):

        kwargs['ylabel'] = r'$j$ (cm$^2$/s)   $\omega$ (1/s)'
        kwargs['yscale'] = 'log'
        self.diffvar = kwargs.setdefault('diffvar', 'nu')
        self.diffvarname = kwargs.setdefault('diffvarname', r'$\nu$')
        kwargs.setdefault('ylabel2', r'{varname} (M$_{{\odot}}^2$/s)   $B_\mathrm{{r}}$, $B_{{\phi}}$ (Gauss)'). format(
            varname = self.diffvarname)
        kwargs['yscale2'] = 'log'

        super().__init__(*args, **kwargs)

    def draw(self):
        self.setup_logger(silent = False)

        # clear old stuff
        self.clear()
        self.setx()

        xvar = self.xvar
        data = self.data

        iim = xvar.iim
        iib = xvar.iib
        x = xvar.x[iib]
        xm = xvar.xm[iim]

        difim = data.parm.difim

        try:
            nang3d = data.parm.nang3d
        except:
            nang3d = 0

        # angular momentum and angular velocity
        angjn = data.angjn[iim]
        angwn = data.angwn[iim]
        ii = np.where(angjn > 0)[0]
        if nang3d:
            angj = data.angj[iim, :]
            angw = data.angw[iim, :]

        if len(ii) > 0:
            angjn0 = floor(log10(np.min(angjn[ii])))
            angjn1 = ceil(log10(np.max(angjn[ii])))
            if nang3d != 0:
                angjn0 -= 1
        else:
            angjn0 = -1
            angjn1 = 1
        if nang3d == 0:
            self.ax.plot(xm, angjn, color = self.colors['angjn'], label = r'$j$')
        else:
            for i in range(6):
                self.ax.plot(
                    xm, angj[:, i // 2] * (1 - 2 * (i % 2)),
                    color = self.colors['angj'][i],
                    linestyle = self.linestyles['angj'][i],
                    label = f'$j_\mathrm{{{self.labels["vec"][i // 2]}}}$')

        if len(ii) > 0:
            angwn0 = floor(log10(np.min(angwn[ii])))
            angwn1 = ceil(log10(np.max(angwn[ii])))
            if nang3d != 0:
                angwn0 -= 1
        else:
            angwn0 = -1
            angwn1 = 1
        wscale = round(0.5 * ((angwn1 + angwn0) - (angjn1 + angjn0)))
        if nang3d == 0:
            self.ax.plot(xm, angwn * 0.1**wscale, color = self.colors['angwn'], label = r'$\omega$')
        else:
            for i in range(6):
                self.ax.plot(
                    xm, angw[:, i // 2] * (1 - 2 * (i % 2)) * 0.1**wscale,
                    color = self.colors['angw'][i],
                    linestyle = self.linestyles['angw'][i],
                    label = f'$\omega_\mathrm{{{self.labels["vec"][i // 2]}}}$')

        # finally, set scales and labels
        ylim = np.array([min(angjn0, angwn0 - wscale), max(angjn1, angwn1 - wscale)])
        ylim = 10.**ylim
        self.ax.set_ylim(*ylim)

        ylabel = r'$j$ (cm$^2$/s)   $\omega$ ($10^{{{}}}$ 1/s)'.format(wscale)
        self.ax.set_ylabel(ylabel)

        # diffusion coefficients
        rn = data.rn[iib]
        dnf = data.dnf[iib]
        if xvar.name in ('log rn', ):
            vunit = rn**(-2)
            vuname = '1/s'
            vsubs = '_r'
        elif xvar.name in ('q', ):
            vuname = r'M$_{{\star}}^2$/s'
            vunit = (4 * np.pi * rn**2 * dnf / data.xmtot)**2
            vsubs = '_q'
        elif xvar.name in ('zm_sun', ):
            vuname = r'M$_{{\odot}}^2$/s'
            vunit = (4 * np.pi * rn**2 * dnf / const.solmass)**2
            vsubs = '_m'
        elif xvar.name in ('log zm_sun', ): #this may not be correct
            vuname = r'M$_{{\odot}}^2$/s'
            vunit = (2.4 * np.pi * rn**4 * dnf / (const.solmass * const.solrad**2))**2
            vsubs = '_m'
        else:
            vuname = r'cm^2/s'
            vunit = 1
            vsubs = ''

        if self.diffvar == 'nu':
            visc = difim * (data.angfjc * data.difi[iib] + data.angdg[iib] + data.bfvisc[iib]) * vunit
            difi = data.difi[iib] * data.angfjc * vunit
            bfvisc = data.bfvisc[iib] * vunit
        else:
            visc = difim * (data.difi[iib] + data.angfc * data.angdg[iib] + data.bfdiff[iib]) * vunit
            difi = data.difi[iib] * vunit
            bfvisc = data.bfdiff[iib] * vunit
        ii = np.where(visc > 0)[0]
        if len(ii) > 0:
            visc1 = ceil(log10(np.max(visc[ii])))
        else:
            visc1 = 0
        visc0 = visc1 - 15
        self.ax2.plot(x, visc, color = self.colors['visc'], label = self.diffvarname)

        if np.any(difi > 0):
            self.ax2.plot(x, np.maximum(difi, 1.e-99),
                          color = self.colors['difi'],
                          label = r'$\quad_\mathrm{{c}}$')

        for i in range(5):
            if self.diffvar == 'nu':
                diff = data.angd[iib,i] * vunit
            else:
                diff = data.angd[iib,i] * data.angfc * vunit
            if np.any(diff > 0):
                self.ax2.plot(x, np.maximum(diff, 1.e-99),
                              color = self.colors['angd{:d}'.format(i+1)],
                              label = r'$\quad_\mathrm{{{:d}}}$'.format(i+1))

        if np.any(bfvisc > 0):
            self.ax2.plot(x, np.maximum(bfvisc, 1.e-99),
                          color = self.colors['bfvisc'],
                          label = r'$\quad_\mathrm{{m}}$')

        # magnetic fields
        br = data.bfbr[iib]
        bt = data.bfbt[iib]
        iir = np.where(br > 0)[0]
        iit = np.where(bt > 0)[0]
        if len(iir) > 0 or len (iir) > 0:
            b1 = ceil(np.log10(max(np.max(br[iir]),np.max(bt[iit]))))
        else:
            b1 = 0
        bscale = b1 - visc1 + 1
        self.ax2.plot(x, br * 0.1**bscale, color = self.colors['br'], label = r'$B_r$')
        self.ax2.plot(x, bt * 0.1**bscale, color = self.colors['bt'], label = r'$B_\phi$')

        # finally, set scales and labels
        ylim = np.array([visc0, visc1])
        ylim = 10.**ylim
        self.ax2.set_ylim(*ylim)

        ylabel2 = r'{varname} ({vunit})   $B_\mathrm{{r}}$, $B_{{\phi}}$ ($10^{{{bscale}}}$ Gauss)'.format(
            bscale = bscale,
            vunit = vuname,
            varname = self.diffvarname
            )
        self.ax2.set_ylabel(ylabel2)

        self.decorations()
        self.close_logger(timing = 'Plot finished in')

class Plot7(Plot78Base):
    def __init__(self, *agrs, **kwargs):
        super().__init__(*agrs, **kwargs)

class Plot8(Plot78Base):
    def __init__(self, *agrs, **kwargs):
        kwargs['diffvar'] = 'D'
        kwargs['diffvarname'] = '$D$'
        super().__init__(*agrs, **kwargs)

class Plot4(BasePlotKepAxes):

    colors = dict(
        sig  = 18,
        sigi = 4,
        sige = 22,
        sigp = 20,
        sigr = 11,
        sigion = 7,
        )

    def __init__(self, *args, **kwargs):

        kwargs.setdefault('showconv', None)

        kwargs['ylabel'] = r'entropy ($k_\mathrm{{B}}$ / baryon)'
        kwargs['yscale'] = 'log'

        super().__init__(*args, **kwargs)

    def draw(self):
        self.setup_logger(silent = False)

        # clear old stuff
        self.clear()
        self.setx()

        xvar = self.xvar
        data = self.data

        iim = xvar.iim
        xm = xvar.xm[iim]
        s = data.entropies[iim,:]

        s1 = ceil(log10(np.max(s[:,0])))
        s0 = floor(log10(np.min(s[:,0])))
        s0 = max(s0, -1)

        # set scales and labels
        ylim = np.array([s0, s1])
        ylim = 10.**ylim
        self.ax.set_ylim(*ylim)

        for i, (c, l) in enumerate(
            zip(('sig', 'sigi', 'sige', 'sigp', 'sigr', 'sigion'),
                (''   , 'I'   , 'E'   , 'P'   , 'R'   , 'Z'     ))):
            sx = s[:,i].copy()
            ii = sx > 0
            if np.any(ii):
                sx[np.logical_not(ii)] = np.nan
                self.ax.plot(xm, sx, color = self.colors[c], label = l)

        self.decorations()
        self.close_logger(timing = 'Plot finished in')

class Plot6(BasePlotKepAxes):
    """
    make the custon-select ion plot

    Ions are set using setiso and addiso and store in isosym.  The
    number of ison is stored in numiso.
    """

    # using Ion class is more general but a bit slower.
    _use_ions = False

    isocolors = (
        22, 4, 11, 16, 7, 10, 20, 26, 1, 18, 2, 24, 9, 14, 13, 5, 19,
        25, 6, 21, 3, 8, 15, 17, 23, 22, 4, 11, 16, 7, 10, 20, 26, 1,
        18, 2, 24, 9, 14, 13, 5, 19, 25, 6, 21, 3, 8, 15, 17, 23,
        )

    colors = {i:c for i,c in enumerate(isocolors)}

    def __init__(self, *args, **kwargs):

        kwargs['ylabel'] = r'isotope mass fraction'
        kwargs['yscale'] = 'log'

        super().__init__(*args, **kwargs)

    def draw(self):
        self.setup_logger(silent = False)

        # clear old stuff
        self.clear()
        self.setx()

        xvar = self.xvar
        data = self.data
        niso = data.numiso
        isosym = data.isosym
        ppnb = data.ppnb
        ionnb = data.ionnb
        numib = data.numib
        ionsb = data.ionsb

        iim = xvar.iim
        xm = xvar.xm[iim]
        jm = xvar.jm

        if self._use_ions:
            # ions should come from data object where they may be cached
            pltions = isotope.ionarr([
                isosym[i][3:].strip()
                for i in range(data.numiso)
                ])
            inetb = data.netnumb[1]
            netions = isotope.ionarr([
                ionsb[ionnb[i,inetb - 1] - 1].strip()
                for i in range(numib[inetb - 1])
                ])
        else:
            pltions = np.array([
                isosym[i][3:].strip()
                for i in range(data.numiso)
                ])
            inetb = data.netnumb[1]
            netions = np.array([
                ionsb[ionnb[i,inetb - 1] - 1].strip()
                for i in range(numib[inetb - 1])
                ])
        ii = np.where(np.in1d(netions, pltions))[0]

        x0 = data.abunminb
        x1 = data.abunmaxb
        self.ax.set_ylim(x0, x1)

        for i in ii:
            ion = netions[i]
            iso = isotope.ion(ion)
            xiso = ppnb[i,iim] * iso.A
            ll = xiso > 0
            ic = np.where(ion == pltions)[0][0]
            if np.any(ll):
                xiso[np.logical_not(ll)] = np.nan
                self.ax.plot(xm, xiso, color = self.colors[ic], label = iso.LaTeX())

        self.decorations()
        self.close_logger(timing = 'Plot finished in')

class Plot3(BasePlotKepAxes):
    """
    make default isotope network plot

    parameters
    ----------

    - iplotb (default iplotb, see KEPLER doc):
      0: APPROX/QSE/NSE
      1: use BURN for APPROX
      2: use BURN above bmasslo
      3: use BURN everywhere
     1x: plot BURN summed up by element
     2x: plot BURN summed up by mass number

    - abunlim (default abunlim)
      default lower plot limit.  If set to <= 0, use 1.e-3 but plot linear

    - plotlimb (default: abunlim)
      do not plot BURN species less abundant than plotlimb everywhere
      (these could otherwise become visible when zooming, but would
      eat lots of time in standard plots)
    """

    # using Ion class is more general but a bit slower.

    ioncolors = (
          5, 22, 19, 6, 4, 11, 24, 16, 7, 10,
          20, 26, 14, 13, 21, 2, 18, 25, 9, 1,
          )

    idtion = {i:c for i,c in enumerate((
            'n', 'H', 'p', r'$^3$He', 'He', 'C', 'N', 'O', 'Ne', 'Mg',
            'Si', 'S', 'Ar', 'Ca', 'Ti', 'Cr', r'$^{52}$Fe', 'Fe', 'Ni'))}
    idtion[33] = r'$^{56}$Fe'
    idtion[34] = "'Fe'"

    colors = {i:c for i,c in enumerate(ioncolors)}

    nc_sym = '/'
    nc_color = 19

    ipacol = (
        1,   2,  4,  5,  6,  7,  8,  9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 9)
    ipaidx = np.array((
        1,      1001,  2003,  2004,  6012,  7014,  8016, 10020, 12024, 14028,
        16032, 18036, 20040, 22044, 24048, 26052, 26054, 28056, 26056))
    ipadic = {i:c for i,c in zip(ipaidx, ipacol)}
    ielidx = {i//1000:c for i,c in zip(ipaidx, ipacol)}
    iaidx = {(i%1000):c for i,c in zip(ipaidx, ipacol)}

    ielemcol  = (
        22,  4, 19, 16, 7, 10, 20, 26, 18, 12,
        24,  9, 14, 13, 5, 11, 25,  6, 21,  3,
         8, 15, 17, 23)

    _show_nc = True

    def __init__(self, *args, **kwargs):

        self.iplotb = kwargs.pop('iplotb', None)
        self.abunlim = kwargs.pop('abunlim', None)
        self.plotlimb = kwargs.pop('plotlimb', None)

        kwargs['ylabel'] = r'isotope mass fraction'

        super().__init__(*args, **kwargs)

        if self.data.abunlim > 0:
            yscale = 'log'
        else:
            yscale = 'linear'
        yscale = kwargs.get('yscale', yscale)
        self.ax.set_yscale(yscale)

    def draw(self):
        self.setup_logger(silent = False)

        # clear old stuff
        self.clear()
        self.setx()

        xvar = self.xvar
        data = self.data

        iim = xvar.iim
        xm = xvar.xm[iim]
        jm = xvar.jm

        if self.iplotb is None:
            iplotb = data.iplotb
        else:
            iplotb = self.iplotb

        iplotb1 = iplotb % 10
        mapmode = iplotb // 10

        show_nc = self._show_nc

        ppn = data.ppn
        ionn = data.ionn
        numi = data.numi
        netnum = data.netnum
        aion = data.aion

        if show_nc:
            xnc = np.ones_like(xm)

        if self.abunlim is None:
            abunlim = data.abunlim
        else:
            abunlim = self.abunlim
        if abunlim <= 0:
            abunlim = 1.e-3
        self.ax.set_ylim(abunlim, 1.)

        if self.plotlimb is None:
            plotlimb = abunlim
        else:
            plotlimb = self.plotlimb

        if iplotb1 in (1, 2, 3) and data.inburn == 1:
            ppnb = data.ppnb
            ionnb = data.ionnb
            numib = data.numib
            ionsb = data.ionsb
            aionb = data.aionb
            zionb = data.zionb
            netnumb = data.netnumb

            if iplotb1 == 1:
                llb = netnum[iim] == 1
            elif iplotb == 2:
                # cell *lower* boundary has to be above bmasslow
                llb = data.zm[0:jm] > data.bmasslow
            else:
                llb = np.tile(True, xm.shape)

            for inb in range(len(numib)):
                llx = netnumb[iim] == inb + 1
                ll = np.logical_and(llx, llb)
                if np.any(ll):
                    lli = np.logical_not(ll)
                    iion = ionnb[:numib[inb],inb] - 1
                    iz = np.round(zionb[iion]).astype(int)
                    ia = np.round(aionb[iion]).astype(int)
                    mfrac = ppnb[iion,iim] * aionb[iion, np.newaxis]
                    if mapmode == 1:
                        mfrac, ival = project(mfrac, iz, return_values = True)
                    elif mapmode == 2:
                        mfrac, ival = project(mfrac, ia, return_values = True)
                    else:
                        ival = np.arange(numib[inb])

                    for j,iv in enumerate(ival):
                        xiso = mfrac[j]
                        if show_nc:
                            xnc[ll] -= xiso[ll]
                        if not np.any(xiso > plotlimb):
                            continue
                        xiso[lli] = np.nan

                        if mapmode == 1:
                            ionsym = isotope.ion(Z=iv).LaTeX()
                            ic = self.ielidx.get(iv, 0)
                            if ic > 0:
                                color = self.colors[ic - 1]
                        elif mapmode == 2:
                            ionsym = str(iv)
                            ic = self.iaidx.get(iv, 0)
                            if ic > 0:
                                color = self.colors[ic - 1]
                        else:
                            ion = iion[j]
                            idx = 1000 * iz[iv] + ia[iv]
                            icol = (idx % 25) + 2
                            color = self.kepler_colors[icol]
                            ionsym = isotope.ion(ionsb[ion]).LaTeX()
                            ic = self.ipadic.get(idx, 0)
                            if ic > 0:
                                color = self.colors[ic - 1]

                        if ionsym is not None:
                            self.ax.plot(
                                xm, xiso,
                                color = color,
                                label = ionsym)

            lla = np.logical_not(llb)
        else:
            lla = np.tile(True, xm.shape)

        for ina in range(len(numi)):
            llx = netnum[iim] == ina + 1
            ll = np.logical_and(llx, lla)
            lli = np.logical_not(ll)
            if np.any(ll):
                for j in range(numi[ina]):
                    iion = ionn[j,ina] - 1
                    xiso = ppn[j,iim] * aion[iion]
                    if show_nc:
                        xnc[ll] -= xiso[ll]
                    xiso[lli] = np.nan
                    ic = j
                    ionsym = self.idtion.get(iion, None)
                    if ionsym is not None:
                        self.ax.plot(
                            xm, xiso,
                            color = self.colors[ic],
                            label = ionsym)
        if show_nc:
            ic = self.nc_color
            ionsym = self.nc_sym
            xiso = np.abs(xnc)
            self.ax.plot(
                xm, xiso,
                color = self.colors[ic],
                label = ionsym)

        self.decorations()
        self.close_logger(timing = 'Plot finished in')


class Plot9(BasePlotKep):


    ielemcol  = (
        22,  4, 19, 16, 7, 10, 20, 26, 18, 12,
        24,  9, 14, 13, 5, 11, 25,  6, 21,  3,
         8, 15, 17, 23)
    colors = {i:c for i,c in enumerate(ielemcol)}

    def __init__(self, *args, **kwargs):

        kwargs['ylabel'] = r'mass fraction'
        kwargs['yscale'] = 'log'

        kwargs['xlabel'] = r'mass number'
        kwargs['xscale'] = 'linear'

        self.kwargs = kwargs

        super().__init__(*args, **kwargs)
        self.update()


    def draw(self):
        self.setup_logger(silent = False)

        # clear old stuff
        self.clear()

        data = self.data
        jm = data.jm
        xm = data.xm

        if data.inburn == 0:
            self.logger.error('[ERROR] [Plot 9] no BURN data.')
            return False
        iblow = np.searchsorted(data.zm[0:jm], data.bmasslow) + 1
        if iblow >= jm:
            self.logger.error('[ERROR] [Plot 9] no BURN data.')
            return False

        ipromin = self.kwargs.get('ipromin', dataipromin)
        ipromax = self.kwargs.get('ipromax', dataipromax)

        iprownd = self.kwargs.get('iprownd', dataiprownd)

        iproyld = self.kwargs.get('iproyld', dataiproyld)

        minapro = self.kwargs.get('minapro', dataminapro)
        maxapro = self.kwargs.get('maxapro', datamaxapro)

        proymax = self.kwargs.get('proymax', dataproymax)
        proymin = self.kwargs.get('proymin', dataproymin)

        proamax = self.kwargs.get('proamax', dataproamax)
        proamin = self.kwargs.get('proamin', dataproamin)

        profmax = self.kwargs.get('profmax', dataprofmax)
        profmin = self.kwargs.get('profmin', dataprofmin)

        if iproyld in (101, 102, 103):
            # make network trace plots
            ionbmax = data.ionbmax
            nbmax = data.nbmax
            nabmax = data.nabmax
            ii = slice(0, nbmax)
            ions = isotope.ion(ionbmax[ii])
            if iproyld == 101:
                abu = data.burnamax[ii] * data.nabmax[ii]
            elif iproyld == 102:
                abu = data.burnmmax[ii] * const.solmassi
            elif iproyld == 103:
                abu = np.array(data.ibcmax[ii], dtype = np.float)
            else:
                raise Exception('Case not found')
        else:
            netnumb = data.netnumb
            ionnb = data.ionnb
            ppnb = data.ppnb
            aionb = data.aionb
            ionsb = data.ionsb

            numib = data.numib

            start = max(iblow, ipromin)
            stop = min(jm, ipromax)+1
            if stop <= start:
                if iprownd > 0:
                    start = stop = jm
                else:
                    self.logger.error('[ERROR] No valid zones.')
                    self.close_logger(timing = 'Done in')
                    return
            iisel = slice(start, stop)

            ibmax = ppnb.shape[1]
            iibmax = slice(None, ibmax)
            llb = np.tile(False, ibmax)
            llb[iisel] = True

            ions = np.array([], dtype = np.object)
            abu = np.array([], dtype = np.float)
            for inb in range(len(numib)):
                llx = netnumb[iibmax] == inb + 1
                ll = np.logical_and(llx, llb)
                if np.any(ll):
                    iion = ionnb[:numib[inb],inb] - 1
                    mfrac = ppnb[iion,:][:,ll] * aionb[iion, np.newaxis]
                    mfrac = np.sum(mfrac * xm[iibmax][np.newaxis, ll], axis=1)
                    xions = isotope.ion(ionsb[iion])
                    llold = np.in1d(xions, ions)
                    oldions = xions[llold]
                    oldfrac = mfrac[llold]
                    ii = index1d(oldions, ions)
                    abu[ii] += oldfrac[:]

                    llnew = np.logical_not(llold)
                    if np.any(llnew):
                        ions = np.append(ions, xions[llnew])
                        abu = np.append(abu, mfrac[llnew])

            # Deal with wind.
            #
            # This is very prelimiary as there is no
            # provisions for networks in the code at this time.
            if iprownd == 1:
                inwb = len(numib) - 1
                iion = ionnb[:numib[inwb], inwb] - 1
                mfrac = data.windb[iion] * aionb[iion]
                xions = isotope.ion(ionsb[iion])

                llold = np.in1d(xions, ions)
                oldions = xions[llold]
                oldfrac = mfrac[llold]
                ii = index1d(oldions, ions)
                abu[ii] += oldfrac[:]

                llnew = np.logical_not(llold)
                if np.any(llnew):
                    ions = np.append(ions, xions[llnew])
                    abu = np.append(abu, mfrac[llnew])

        # ensure list is sorted
        ii = np.argsort(ions)
        ions = ions[ii]
        abu = abu[ii]

        # pf - 0
        # YD - 1
        # yd - 2
        # Y  - 3
        # y - 4

        # pfe - 5
        # YDE - 6
        # yde - 7
        # YE - 8
        # Ye - 9

        # pfa - 10
        # YDA - 11
        # yda - 12
        # YA - 13
        # ya - 14

        # burnaplt - 101
        # burnmplt - 102
        # burncplt - 103

        if iproyld in (5, 6, 7, 8, 9,):
            xmap = isotope.ufunc_Z
            xkw = dict(elements = True)
        elif iproyld in (10, 11, 12, 13, 14):
            xmap = isotope.ufunc_A
            xkw = dict(isobars = True)
        elif iproyld in (0, 1, 2, 3, 4):
            xmap = lambda x: x
            xkw = dict()
        if iproyld in (0, 5, 10):
            imap = ionmap.Decay(
                ions,
                molfrac_in = False,
                # solprod = True,
                molfrac_out = False,
                stable = True,
                ** xkw
                )
            ival = xmap(imap.decions)
            mfrac = imap(abu)
            # should introduce and use datafile method
            # store/cache solabu
            solabu = SolAbu('solabu.dat')
            mfrac /= solabu[imap.decions] * np.sum(mfrac)
        elif iproyld in (1, 2, 6, 7, 11, 12,):
            imap = ionmap.Decay(
                ions,
                molfrac_in = False,
                molfrac_out = False,
                stable = True,
                **xkw
                )
            mfrac = imap(abu)
            ival = xmap(imap.decions)
        elif iproyld in (8, 9, 13, 14):
            ia = xmap(ions)
            mfrac, ival = project(abu, ia, return_values = True)
        elif iproyld in (3, 4, 101, 102, 103, ):
            ival = ions
            mfrac = abu

        yscale = 'log'
        if iproyld in range(15):
            if iproyld in (0, 5, 10,):
                yunit = 'production factor (solar)'
                ymin = profmin
                ymax = profmax
            elif iproyld in (1, 3, 6, 8, 11, 13,):
                mfrac /= const.solmass
                yunit = 'yield (solar masses)'
                ymin = proymin
                ymax = proymax
            elif iproyld in (2, 4, 7, 9, 12, 14):
                mfrac /= np.sum(mfrac)
                yunit = 'yield (mass fraction)'
                ymin = proamin
                ymax = proamax
            if iproyld % 5 in (0, 1, 2):
                ydecay = 'decayed'
            else:
                ydecay = 'undecayed'
            if iproyld // 5 == 0:
                ykind = ' isotope'
            elif iproyld // 5 == 1:
                ykind = ' element'
            else:
                ykind = ' isobar'
            ylabel = '{} {} {}'.format(ydecay, ykind, yunit)
        elif iproyld in (101, 102, 103):
            ylabel = ('maximum BURN network mass fraction',
                      'BURN network maxima mass coordinates',
                      'BURN network maxima cycles',
                )[iproyld - 101]
            if iproyld == 101:
                ymax = 1.
                ymin = 1.e-35
                mfrac /= np.sum(mfrac)
            elif iproyld == 102:
                ymag = np.max(mfrac)
                ymax = +1.05 * ymag
                ymin = -0.05 * ymag
                yscale = 'linear'
            elif iproyld == 103:
                ymag = data.ncyc
                ymax = +1.05 * ymag
                ymin = -0.05 * ymag
                yscale = 'linear'

        if iproyld in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 101, 102, 103):
            xlabel = 'mass number'
        elif iproyld in (5,6,7,8,9,):
            xlabel = 'charge number'

        if iproyld in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14):
            self.ax.plot(ival, mfrac,
                         'o-',
                         color = self.colors[0],
                         lw = 1,
                         ms = 4)
        elif iproyld in (0, 1, 2, 3, 4, 101, 102, 103):
            iz = isotope.ufunc_Z(ival)
            ia = isotope.ufunc_A(ival)
            jj = np.where(iz[1:] != iz[:-1])[0] + 1
            jj = [0] + jj.tolist() + [len(iz)]
            for j in range(len(jj)-1):
                j0 = jj[j]
                j1 = jj[j+1]
                jz = iz[j0]
                col = self.colors[(jz + 23) % 24]
                ii = slice(j0,j1)
                x = ia[ii]
                y = mfrac[ii]
                if iproyld in (0,1,2):
                    self.ax.plot(x, y,
                                 'o-',
                                 color = col,
                                 lw = 1,
                                 ms = 4)
                    ll = np.array([True]*(j1-j0))
                else:
                    self.ax.plot(x, y,
                                 '-',
                                 color = col,
                                 lw = 1,
                                 ms = 4)
                    # todo - cache solabu
                    try:
                        solabu
                    except:
                        solabu = SolAbu('solabu.dat')
                    ll = solabu.contains(ival[ii])
                    self.ax.plot(x[ll], y[ll],
                                 'o',
                                 color = col,
                                 lw = 1,
                                 ms = 4,
                                 )
                    lli = np.logical_not(ll)
                    self.ax.plot(x[lli], y[lli],
                                 'o',
                                 color = col,
                                 lw = 1,
                                 ms = 4,
                                 markerfacecolor = 'none',
                                 )
                if not np.any(ll):
                    ll = lli
                ja = ia[ii][ll]
                if len(ll) == 1:
                    va = 'bottom'
                    xl = ja[0]
                    yl = y[ll][0]
                elif len(ja) == 2 and ja[1] == ja[0] + 1:
                    xl = 0.5 * (ja[0] + ja[1])
                    if yscale == 'log':
                        yl = np.sqrt(y[ll][0] * y[ll][1])
                    else:
                        yl = 0.5 * (y[ll][0] + y[ll][1])
                    va = 'center'
                else:
                    i = np.arange(len(ll))[ll][(len(ja)-1) // 2]
                    if yscale == 'log':
                        jy = np.log10(np.maximum(y, 1.e-99))
                    else:
                        jy = y
                    dys = 0
                    if i > 0:
                        dys = jy[i-1] - jy[i]
                    if i < len(y) - 1:
                        dys += jy[i+1] - jy[i]
                    va = ('bottom', 'top')[dys > 0]
                    xl = x[i]
                    yl = y[i]
                ha = 'center'
                label = isotope.Element(jz).LaTeX()
                # TODO: cache labels and use paths
                self.ax.text(xl, yl, label,
                             horizontalalignment = ha,
                             verticalalignment = va,
                             transform = self.ax.transData,
                             clip_on = True,
                             fontsize = self._default_label_size,
                             )


        # add element labels
        if iproyld in (5, 6, 7, 8, 9):
            if yscale == 'log':
                y = np.log10(np.maximum(mfrac, 1.e-99))
            else:
                y = mfrac
            dy = y[1:] - y[:-1]
            dys = np.zeros_like(y)
            dys[:-1] += dy[:]
            dys[1:] -= dy[:]
            ll = dys > 0
            for i in range(len(mfrac)):
                label = isotope.Element(ival[i]).LaTeX()
                xl = ival[i]
                yl = mfrac[i]
                ha = 'center'
                va = ('bottom', 'top')[ll[i]]
                # TODO: cache labels and use paths
                self.ax.text(xl, yl, label,
                             horizontalalignment = ha,
                             verticalalignment = va,
                             transform = self.ax.transData,
                             clip_on = True,
                             fontsize = self._default_label_size,
                             )

        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_yscale(yscale)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        scale = min(ymax, 10**np.ceil(np.log10(np.maximum(np.max(mfrac), 1e-99))))
        ymin = min(ymin, scale * 1e-5)
        self.ax.set_ylim(ymin, scale)
        self.close_logger(timing = 'Plot finished in')


kepplots = {
    1 : Plot1,
    2 : Plot2,
    3 : Plot3,
    4 : Plot4,
    5 : Plot5,
    6 : Plot6,
    7 : Plot7,
    8 : Plot8,
    9 : Plot9,
    }
