"""
Utilities for XRB
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import kepdump
import ionmap
import isotope
import color
import bdat
import datetime
import subprocess
import re
import color

import utils

class FigTemplate(object):
    def get_fig(self, fig = None, ax = None, figsize = (6.4, 4.8), **kwargs):
        if ax is None:
            if fig is None:
                fig = plt.figure(figsize=figsize)

                if mpl.get_backend() == 'TkAgg':
                    try:
                        dpi0 = fig.get_dpi()
                        output = subprocess.run(['xrandr'], stdout=subprocess.PIPE).stdout.decode()
                        pixels = np.array(re.findall('(\d+)x(\d+) ', output)[0], dtype = np.int)
                        mm = np.array(re.findall('(\d+)mm x (\d+)mm', output)[0], dtype = np.int)
                        dpi = 25.4 * np.sqrt(0.5 * np.sum(pixels**2) / np.sum(mm**2))
                        print(f' [Plot] old dpi was {dpi0}')
                        print(f' [Plot] setting dpi to {dpi}')
                        fig.set_dpi(dpi)
                    except:
                        raise
                        print(f' [Plot] could not use xrandr to determine dpi')

            ax = fig.add_subplot(111)
        return fig, ax

    def yscale(self, dump,
               ybase = None,
               ytop = None,
               r = 'y',
               scaley = True,
               center = 'f',
               **kwargs):

        ym = np.append(np.cumsum(dump.xm[-2::-1])[::-1], 0.)

        jbase = np.searchsorted(dump.zm, dump.bmasslow) + 1
        if ybase is None:
            ybase = ym[jbase]
        else:
            jbase = np.searchsorted(ym, ybase) + 1

        if r == 'radius':
            y = dump.rn[-2] - dump.rn
            y[-1] = y[-2]
            ybase = y[jbase-1]
            ylabel = r'Depth / {}cm'
        elif r == 'logy':
            y = np.log10(np.maximum(dump.y, 1e-99))
            ytop = y[-3] - np.log10(2)
            ybase = y[jbase-1]
            ylabel = r'$\log($ Column depth / $\mathrm{{g}}\,\mathrm{{cm}}^{{-2}}$ $)$'
            scaley = False
        elif r == 'y':
            y = dump.y
            ybase = y[jbase-1]
            ylabel = r'Column depth / {}g$\,$cm$^{{-2}}$'
        elif r == 'm':
            y = dump.zm_
            y -= y[jbase-1]
            ybase = 0.
            ytop = y[-2]
            ylabel = r'Mass above substrate  / {}g'
        else:
            y = ym
            ylabel = 'Exterior mass coordinate / {}g'

        if ytop is None:
            ytop = 0

        if scaley:
            # master code for scale formatting
            yl = int(np.floor(np.log10(max(abs(ybase), abs(ytop)))))
            scale = np.power(10., -yl)
            ybase *= scale
            ytop *= scale
            y = y * scale
            if yl == 0:
                ylabel = ylabel.format('')
            else:
                ylabel = ylabel.format(fr'$10^{{{yl:d}}}\,$')

        if center == 'm':
            y = dump.face2center(y)

        return y, ybase, ytop, ylabel, jbase


class IonMapPlot(FigTemplate):
    def __init__(self,
                 dump,
                 abunlim=1.e-10,
                 cmap = None,
                 vmax = None,
                 mode = 'massfrac',
                 invalid = None,
                 **kwargs,
                 ):
        if isinstance(dump, str):
            dump = kepdump.load(dump)
        self.dump = dump

        if cmap is None:
            cmap = color.ColorBlendBspline(('white',)+tuple([color.ColorScale(color.colormap('viridis_r'), lambda x: (x-0.2)/0.7)]*2) + ('black',), frac=(0,.2,.9,1),k=1)

        b = dump.abub.copy()
        if mode == 'massfrac':
            a = ionmap.decay(b, decay = False, isobars = True, molfrac_out = False)
            data = np.log10(np.maximum(a.data, abunlim))
            vmax = 0
            vmin = np.log10(abunlim)
            label = 'log( mass fraction )'
            unit = None

        elif mode in ('ME', 'MElog', 'MEkeV'):
            B = bdat.BDat()
            Ab = isotope.ufunc_A(b.ions)
            me = B.mass_excess(b.ions) - Ab * B.mass_excess('fe56') / 56
            label = 'Mass excess relative to ${{}}^{{56}}$Fe'
            unit = 'MeV/nucleon'
            b.data = b.data * me
            a = ionmap.decay(b, decay = False, isobars = True, molfrac_out = True)
            if mode == 'ME':
                data = a.data
                vmin = 0
                cmap = color.ColorScaleGamma(cmap, -3)
            elif mode == 'MEkeV':
                data = a.data * 1000
                vmin = 0
                if vmax is not None:
                    vmax *= 1000
                unit = 'keV/nucleon'
                cmap = color.ColorScaleGamma(cmap, -3)
            else:
                data = np.log10(np.maximum(a.data, abunlim))
                vmin = np.min(data)
                label = f'log( {label} )'
            if vmax is None:
                vmax = np.max(data)
            # cmap = color.colormap('viridis_r')
            # cmap = color.ColorScaleGamma(cmap, 1/3)

        if invalid is None:
            invalid = vmin

        if unit is not None:
            label = f'{label} ({unit})'

        A = isotope.ufunc_A(a.ions)

        amax = np.max(A)
        missing, = np.where(~np.in1d(np.arange(amax)+1, A))

        x = np.arange(amax + 1) + 0.5
        y, ybase, ytop, ylabel, jbase = self.yscale(dump, **kwargs)
        data = np.insert(np.transpose(data), missing, np.tile(invalid, data.shape[0]), axis = 0)

        fig, ax = self.get_fig(**kwargs)

        m = ax.pcolormesh(y, x, data[:,1:], cmap = cmap, vmin=vmin, vmax = vmax)
        cb = fig.colorbar(m, label = label)
        ax.set_xlim(ybase, ytop)
        ax.set_ylim(0, max(x))
        ax.set_ylabel('Mass number')
        ax.set_xlabel(ylabel)

        fig.tight_layout()

        self.abu = a
        self.fig = fig
        self.ax = ax

class IonCompPlot(FigTemplate):
    """
    compare compositions
    """
    def __init__(self,
                 dump1,
                 dump2,
                 abunlim=1.e-10,
                 cmap = None,
                 vmax = None,
                 mode = 'ratio',
                 invalid = None,
                 magmax = 3,
                 **kwargs,
                 ):
        if isinstance(dump1, str):
            dump1 = kepdump.load(dump1)
        self.dump1 = dump1
        if isinstance(dump2, str):
            dump2 = kepdump.load(dump2)
        self.dump2 = dump2

        b1 = dump1.abub.copy()
        a1 = ionmap.decay(b1, decay = False, isobars = True, molfrac_out = False)

        b2 = dump2.abub.copy()
        a2 = ionmap.decay(b2, decay = False, isobars = True, molfrac_out = False)

        A1 = isotope.ufunc_A(a1.ions)
        A2 = isotope.ufunc_A(a2.ions)
        amax = max(np.max(A1), np.max(A2))

        iiA = np.arange(amax+1)
        llA1 = np.in1d(iiA, A1)
        llA2 = np.in1d(iiA, A2)

        jbase1 = np.searchsorted(dump1.zm, dump1.bmasslow) + 1
        jbase2 = np.searchsorted(dump2.zm, dump2.bmasslow) + 1

        y1 = dump1.y[jbase1 - 1] - dump1.y[:-1]
        y2 = dump2.y[jbase2 - 1] - dump2.y[:-1]

        y = np.hstack([y1, y2])
        y, ind = np.unique(y, return_inverse = True)

        zmax = len(y) - 1

        data1 = np.tile(np.nan, (zmax, amax + 1))
        data2 = data1.copy()

        ind1 = ind[:len(y1)]
        ind2 = ind[len(y1):]

        iiz = np.arange(zmax)
        llz1 = (iiz >= ind1[0]) & (iiz < ind1[-1])
        llz2 = (iiz >= ind2[0]) & (iiz < ind2[-1])

        iiz1 = list()
        for i, (i0, i1) in enumerate(zip(ind1[:-1], ind1[1:])):
            iiz1 += [i+1] * (i1 - i0)
        iiz2 = list()
        for i, (i0, i1) in enumerate(zip(ind2[:-1], ind2[1:])):
            iiz2 += [i+1] * (i1 - i0)

        data1[llz1[:, None] & llA1[None,:]] = a1.data[iiz1, :].flatten()
        data2[llz2[:, None] & llA2[None,:]] = a2.data[iiz2, :].flatten()

        if mode == 'a1':
            data = np.log10(np.maximum(data1, abunlim))
            vmax = 0
            vmin = np.log10(abunlim)
            label = 'log( mass fraction )'
            if cmap is None:
                cmap = color.ColorBlendBspline(('white',)+tuple([color.ColorScale(color.colormap('viridis_r'), lambda x: (x-0.2)/0.7)]*2) + ('black',), frac=(0,.2,.9,1),k=1)
            if invalid is None:
                invalid = vmin
        elif mode == 'ratio':
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                data = np.log10(data2 / data1)
            label = 'log( mass fraction ratio )'
            if invalid is None:
                invalid = 0
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            vext = min(max(abs(vmin), abs(vmax)), magmax)
            vmin, vmax = np.array([-1, 1]) * vext
            cmap = color.ColorMap.from_Colormap('coolwarm')
            #cmap = color.ColorBlendBspline(('white',)+tuple([color.ColorScale(color.colormap('viridis_r'), lambda x: (x-0.2)/0.7)]*2) + ('black',), frac=(0,.2,.9,1),k=1)
        else:
            raise NotImplementedError('unkown mode')

        x = np.arange(amax + 1) + 0.5

        missing = np.isnan(data)
        data[missing] = invalid

        data[data > vmax] = vmax
        data[data < vmin] = vmin

        ybase = 0
        ytop = np.max(y)
        ylabel = r'Column above substrate / $\mathrm{{g}}\,\mathrm{{cm}}^{{-2}}$'

        fig, ax = self.get_fig(**kwargs)
        # m = ax.pcolormesh(y, x, data[:, 1:].transpose(), cmap = cmap, vmin=vmin, vmax = vmax)

        data = np.minimum(1,np.maximum(0, np.dstack([data]*4)))
        data = data[:, 1:].transpose((1,0,2))
        m = ax.pcolorfast(y, x, data)

        # cb = fig.colorbar(m, label = label)
        ax.set_xlim(ybase, ytop)
        ax.set_ylim(0.5, max(x))
        ax.set_ylabel('Mass number')
        ax.set_xlabel(ylabel)

        fig.tight_layout()

        self.fig = fig
        self.ax = ax

class ImpurityPlot(FigTemplate):
    def __init__(self, dump, **kwargs):
        if isinstance(dump, str):
            dump = kepdump.load(dump)
        self.dump = dump
        fig, ax = self.get_fig(**kwargs)
        label = 'Impurity parameter'
        y, ybase, ytop, ylabel, jbase = self.yscale(dump, **kwargs)
        ii = slice(jbase, -1)
        b = dump.abub.copy()
        ysum = np.sum(b.data[ii, :], axis=1)
        z = np.array([i.Z for i in b.ions])
        zbar = np.sum(z[np.newaxis, :] * b.data[ii, :], axis=1) / ysum
        imp = np.sum(b.data[ii, :] * (z[np.newaxis,:] - zbar[:, np.newaxis])**2, axis=1) / ysum
        ax.plot(y[ii], imp)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(label)


class MEPlot(object):
    def __init__(self,
                 dumps = (2840, 3760),
                 base = 'run22#{}',
                 labels = None,
                 fig = None,
                 ax = None,
                 mscale = 1e21,
                 norm = 'c12',
                 accmass = None,
                 boundaries = True,
                 ):
        """
        TODO - not just plot zone centres but zones as steps
        """

        if not isinstance(dumps, (tuple, list, np.ndarray)):
            dumps = [dumps]

        if ax is None:
            if fig is None:
                fig = plt.figure()
            ax = fig.add_subplot(111)

        B = bdat.BDat()
        norm = isotope.ion(norm)
        menorm = B.mass_excess(norm) / norm.A
        mefe56 = B.mass_excess('fe56')/56 - menorm
        print(f' [MEPlot] Normaslising to {norm} (ME = {menorm} MeV/nucleon)')
        zm0 = None
        ep = None
        for i, dump in enumerate(dumps):
            if isinstance(dump, kepdump.KepDump):
                d = dump
                dump = d.filename.rsplit('/',1)[-1].rstrip('z') + f'#{d.ncyc}'
            else:
                dump = base.format(dump)
                print(f' [MEPlot] processing {dump}')
                d = kepdump.load(dump)
            b = d.abub
            me = B.mass_excess(b.ions)
            mex = np.sum(b.data * me, axis=1) - menorm
            jblo = np.searchsorted(d.zm, d.bmasslow) + 1
            jj = slice(jblo, -1)
            zsub = d.zm_[jblo-1]
            m = (d.zmm_[jj] - zsub) / mscale
            if labels is not None and len(labels) > i:
                label = labels[i]
            else:
                label = f'{dump} ({str(datetime.timedelta(seconds=int(d.time)))})'
            ex = mex[jj]

            # add zone boundaries
            if boundaries:
                m = np.insert(m, 0, 0)
                m = np.append(m, (d.zm_[-2] - zsub) / mscale)
                ex = np.insert(ex, 0, ex[0])
                ex = np.append(ex, ex[-1])
            ax.plot(m, ex, label = label)

            if accmass is not None:
                if zm0 is None:
                    zm1 = d.zm_[-2] - zsub
                    zm0 = zm1 - accmass
                jhi = np.searchsorted(d.zm_ - zsub, zm1)
                ii = slice(jblo, jhi+1)
                e = np.sum(d.xm[ii] * mex[ii])
                if ep is not None:
                    de = ep - e
                    print(f' [MEPlot] energy difference is {de/accmass} MeV/nucleon')
                ep = e

                jlo = np.searchsorted(d.zm_ - zsub, zm0)
                ii = slice(jlo, jhi+1)
                e = np.sum(d.xm[ii] * mex[ii])
                print(f' [MEPlot] total is {e/accmass + menorm - mefe56} MeV/nucleon.')


        ax.axhline(mefe56, color = 'gray', ls = '--', label = r'${}^{56}\mathrm{Fe}$')

        surf = d.compsurfb
        mesurf = np.sum(surf * me / isotope.ufunc_A(b.ions)) - menorm
        ax.axhline(mesurf, color = 'gray', ls = ':', label = r'accretion')

        m_surf = np.max(m)
        xlim = (-0.02 * m_surf, 1.02 * m_surf)
        ax.set_xlim(xlim)

        ax.axvspan(xlim[0], 0, color = '#cccccc', label = 'substrate')

        if accmass is not None:
            ax.axvspan(*((np.array([zm0, zm1])) / mscale),
                       color = '#ffffbb',
                       zorder = -2,
                       ls = '-',
                       lw = 0.5,
                       label = 'Accretion column')

        if mscale is None or mscale == 1:
            xlscale = ''
        else:
            xlscale = int(np.log10(mscale))
            xlscale = fr'$10^{{{xlscale:d}}}\,$'

        ax.set_xlabel(f'Mass above substrate / {xlscale}g')
        if abs(menorm) > 1e-5:
            offset = f'relative to {isotope.ion(norm).mpl()} '
        else:
            offset = ''
        ax.set_ylabel(f'Mass excess per nucleon {offset}/ MeV')

        ax.legend(loc = 'best')
        fig.tight_layout()
        fig.show()

        self.fig = fig
        self.ax = ax


import abuset
import mass_table

class ME(object):
    def __init__(self, mode = 'bdat', silent = False, norm = 'c12'):
        if mode == 'bdat':
            M = bdat.BDat()
            ions = M.ext_ions
            me = M.mass_excess(ions)
        elif mode in ('Ame12', 'Ame03', 'Ame16', ):
            M = mass_table.MassTable(version = mode)
            ions = M.ions
            me = M.mass_excess()

        norm = isotope.ion(norm)

        print(f' [ME] Computing ... ', end = '', flush = True)
        EF = np.linspace(0, 25, 25001)
        A = ions.A
        Z = ions.Z
        ionsidx = ions.idx
        mena = me[np.argwhere(norm == ions.ions())[0][0]] / norm.A
        x = (me[:,np.newaxis] + Z[:,np.newaxis] * EF[np.newaxis,:]) / A[:,np.newaxis] - mena - norm.Z / norm.A * EF[np.newaxis,:]
        ix = np.array([ionsidx[i] for i in np.argmin(x, axis=0)])
        ii = np.where(ix[1:] != ix[:-1])[0] + 1
        ii = np.insert(ii, 0, 0)
        ij = utils.index1d(ix[ii], ionsidx)
        imax = [(ions[xij], EF[xii], x[xij, xii]) for xij, xii in zip(ij, ii)]
        print(' [ME] done.')

        if not silent:
            print(f'{"Ion":5s}: {"EF/MeV":<6s} {"ME/A":>8s}')
            for i in imax:
                print(f'{i[0]!s:5s}: {float(i[1]):6.3f} {float(i[2]):8.5f}')

        self.imax = imax
        self.M = M
        self.ions = ions
        self.me = me
        self.x = x
        self.ij = ij
        self.ii = ii
        self.EF = EF
        self.mode = mode
        self.norm = norm
        self.mena = mena


    def plot(self, *, ax = None, fig = None):
        if ax is None:
            if fig is None:
                fig = plt.figure(figsize=(6.4,4.8+0.5))
            ax = fig.add_subplot(111)

        ii = np.append(self.ii, len(self.EF)-1)
        for ij in self.ij:
            ax.plot(self.EF, self.x[ij,:], alpha = 0.25)
        ax.set_prop_cycle(None)
        for i,ij in enumerate(self.ij):
            jj = slice(ii[i], ii[i+1])
            ax.plot(self.EF[jj], self.x[ij,jj], lw = 3, label = self.ions[ij].mpl())
        vals = [i[2] for i in self.imax]
        vals.append(self.x[self.ij[-1],-1])
        ymax = np.max(vals)
        ymin = np.min(vals)
        d = ymax - ymin
        ax.set_ylim(ymin - 0.1 * d, ymax + 0.1 * d )
        ax.legend(loc = 'best', title = self.mode)
        ax.set_xlabel('Electron Fermi energy / MeV')
        ax.set_ylabel(f'Mass excess per nucleon relative to {self.norm.LaTeX()} / MeV')
        fig.tight_layout()
        fig.show()
        self.fig = fig
        self.ax = ax

from abuset import IonList

# =======================================================================
# all the plots below need to be checked for propoper zone and mass coordinate usage
# =======================================================================

class Heat(FigTemplate):
    def __init__(self, b = None, **kwargs):
        if not isinstance(b, bdat.BDat):
            b = bdat.BDat()
        ions = IonList(sorted(b.ext_ions, key = lambda x: 1024 * x.A - x.Z))
        A = ions.A
        ai = np.argwhere(A[1:] > A[:-1])[:, 0] + 1
        ai = np.insert(ai, (0, len(ai)), (0, len(A)))
        amax = np.max(ions.A) + 1
        nz = ai[1:] - ai[:-1]
        zmax = np.max(nz)
        me = b.mass_excess(ions)
        a = np.tile(None, (amax, zmax))
        e = np.tile(np.nan, (amax, zmax))
        de = np.tile(0., (amax, zmax))
        zi = np.tile(0, amax)
        zm = np.tile(0, amax)
        for iz, j0, j1 in zip(nz, ai[:-1], ai[1:]):
            Ai = A[j0]
            a[Ai, :iz] = ions[j0:j1]
            e[Ai, :iz] = me[j0:j1]
            zi[Ai] = iz
            de[Ai, :iz-1] = e[Ai, 1:iz] - e[Ai, :iz-1]

            # find first minimum
            ii = np.argwhere(de[Ai,:iz-1] > 0)
            if len(ii) > 0:
                im = ii[0,0]
            else:
                im = 0
            zm[Ai] = im

        mode = kwargs.get('mode', 'single')
        nuloss = kwargs.get('nuloss', 0.35)

        self.A = A
        self.a = a
        self.amax = amax
        self.de = de

        # compute heating for all isotopes
        ecmax = np.max(de)
        heat = np.tile(np.nan, (len(ions), zmax))
        ec = np.tile(np.nan, (len(ions), zmax))
        iion = 0
        for ia in range(amax):
            for iz in range(zi[ia]-1, -1, -1):
                i = iz
                i1 = i + 1
                if de[ia, i] < 0:
                    heat[iion + i, 0] = -de[ia, i]
                    ec[iion + i, 0] = 0.
                    if i1 <= zi[ia]-1:
                        heat[iion + i, 1:] = heat[iion + i1, :-1]
                        ec[iion + i, 1:] = ec[iion + i1, :-1]
                    continue
                if i < zi[ia]-2:
                    dq = 0.
                    if de[ia, i1] > de[ia, i]:
                        eci = de[ia, i]
                    elif mode == 'single':
                        eci = de[ia, i]
                        while i1 < zi[ia]-1 and eci > de[ia, i1]:
                            dq += eci - de[ia, i1]
                            i1 += 1
                    elif mode == 'double':
                        eci = 0.5 * (de[ia, i] + de[ia, i1])
                        i1 += 1
                        while i1 < zi[ia]-1:
                            if de[ia, i1] < eci:
                                dq += eci - de[ia, i1]
                                i1 += 1
                            else:
                                break
                    else:
                        raise Exception('uknown mode')
                    heat[iion + i, 0] = dq
                    ec[iion + i, 0] = eci
                    if i1 <= zi[ia]-1:
                        heat[iion + i, 1:] = heat[iion + i1, :-1]
                        ec[iion + i, 1:] = ec[iion + i1, :-1]
                    continue
                if i == zi[ia]-2:
                    heat[iion + i, 0] = 0.
                    ec[iion + i, 0] = de[ia, i]
                    heat[iion + i, 1:] = heat[iion + i1, :-1]
                    ec[iion + i, 1:] = ec[iion + i1, :-1]
                    continue
                heat[iion + i, 0] = 0.
                ec[iion + i, 0] = ecmax
            iion = iion + zi[ia]

        heat[np.isnan(heat)] = -1e10
        jj = heat > 0
        ii = np.where(jj)
        vals, kk = np.unique(ec[jj], return_inverse = True)
        hmap = np.zeros((len(ions), len(vals)))
        hmap[ii[0], kk] = heat[ii]
        data = (np.cumsum(hmap, axis = -1) / ions.A[:,np.newaxis]) * (1 - nuloss)

        self.heat = heat
        self.ec = ec
        self.ions = ions
        self.data = data
        self.vals = vals

        # compute stable subset projected to A
        staba = [j0 + zm[A[j0]] for j0 in ai[:-1]]
        dataa = data[staba,:]
        missing, = np.where(~np.in1d(np.arange(amax), A[staba]))
        dataa = np.insert(dataa, missing, np.zeros(len(vals)), axis = 0)
        dataa = dataa - dataa[:,0][:,np.newaxis]
        valsa, ii = np.unique(vals, return_index = True)
        dataa = dataa[:, ii]

        self.dataa = dataa
        self.valsa = valsa

    def plot_map(self, **kwargs):
        vals = self.valsa
        data = self.dataa
        amax = self.amax

        x = np.append(vals, vals[-1] * 1.05)
        y = np.arange(amax) + 0.5
        label = 'Cummulative heating in MeV / nucleon'
        xlabel = 'Fermi Energy / MeV'
        ylabel = 'Mass number'
        vmax = kwargs.get('vmax', 0.3)

        cmap = color.ColorBlendBspline(('white',)+tuple([color.ColorScale(color.colormap('viridis_r'), lambda x: (x-0.2)/0.7)]*2) + ('black',), frac=(0,.2,.9,1),k=1)

        fig, ax = self.get_fig(**kwargs)
        m = ax.pcolormesh(x, y, data[1:,:], cmap = cmap, vmax = vmax)
        cb = fig.colorbar(m, label = label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.show()

        self.fig_map = fig
        self.ax_map = ax
        self.cb_map = cb

    def abuheatmap(self, abu, **kwargs):
        vals = self.vals

        try:
            if self.abu is abu:
                ii = self.ii_abu
            else:
                raise
        except:
            ii = utils.index1d(abu.idx(), self.ions.idx)
        heat = self.data[ii, :] * abu.X()[:,np.newaxis]
        heat, A = utils.project(heat, abu.A(), return_values = True)
        amax = np.max(A)
        missing, = np.where(~np.in1d(np.arange(amax-1)+1, A))
        heat = np.insert(heat, missing, np.zeros(len(vals)), axis = 0)

        x = np.append(vals, vals[-1] * 1.05)
        y = np.arange(amax) + 0.5

        cmap = color.colormap('viridis_wrb')

        vmax = None
        label = 'cummulative heating in MeV / nucleon'
        xlabel = 'Fermi Energy / MeV'
        ylabel = 'Mass number'

        fig, ax = self.get_fig(**kwargs)
        m = ax.pcolormesh(x, y, heat[1:,:], cmap = cmap, vmax = vmax)
        cb = fig.colorbar(m, label = label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()

        self.abu = abu
        self.ii_abu = ii

        self.fig = fig
        self.ax = ax

    def abuheat(self, abu, **kwargs):
        vals = self.vals

        ii = utils.index1d(abu.idx(), self.ions.idx)
        heat = self.data[ii, :] * abu.X()[:,np.newaxis]
        heat = np.sum(heat, axis = 0)

        xlabel = 'Fermi Energy / MeV'
        ylabel = 'cummulative heating in MeV / nucleon'

        fig, ax = self.get_fig(**kwargs)
        ax.plot(vals, heat)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()

        self.fig = fig
        self.ax = ax

    def dumpheat(self, dump, **kwargs):
        if isinstance(dump, str):
            dump = kepdump.load(dump)

        x, xbase, xtop, xlabel, jbase = self.yscale(dump, **kwargs)

        iimap = slice(jbase, dump.jm+1)
        b = dump.ppnb[:,iimap]
        ionsb = abuset.IonList(dump.ionsb)
        try:
            if self.dump == dump:
                ii = self.ii_dump
            else:
                raise
        except:
            ii = utils.index1d(ionsb.idx, self.ions.idx)
        A = ionsb.A
        heat = np.sum((self.data[ii, -1] * A)[:,np.newaxis] * b[:,:], 0)

        ylabel = 'heating in MeV / nucleon'

        fig, ax = self.get_fig(**kwargs)

        ax.plot(x[iimap], heat)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xbase, xtop)

        fig.tight_layout()
        fig.show()

        self.ii_dump = ii
        self.dump = dump

        self.fig = fig
        self.ax = ax

    def dumpheatmap(self, dump, **kwargs):
        if isinstance(dump, str):
            dump = kepdump.load(dump)

        x, xbase, xtop, xlabel, jbase = self.yscale(dump, **kwargs)

        iimap = slice(jbase, dump.jm+1)
        b = dump.ppnb[:,iimap]
        ionsb = abuset.IonList(dump.ionsb)
        try:
            if self.dump == dump:
                ii = self.ii_dump
        except:
            ii = utils.index1d(ionsb.idx, self.ions.idx)
        A = ionsb.A

        mode = kwargs.get('mode', 'E')
        if mode == 'E':
            heat = np.sum((self.data[ii, :] * A[:,np.newaxis])[:,np.newaxis,:] * b[:,:,np.newaxis], 0)
            heat = np.transpose(heat[:,:])
            ylabel = 'Electron Fermi energy / MeV'
            vals = self.vals
            y = np.append(vals, vals[-1] * 1.05)
        elif mode == 'A':
            heat = (self.data[ii, -1] * A)[:, np.newaxis] * b[:,:]
            heat, A = utils.project(heat, A, return_values = True)
            amax = np.max(A)
            missing, = np.where(~np.in1d(np.arange(amax-1)+1, A))
            heat = np.insert(heat, missing, 0., axis = 0)
            y = np.arange(amax+1) + 0.5
            ylabel = 'Mass number'
        else:
            raise Exception('unkown mode {mode}')

        label = 'Cummulative heating in MeV / nucleon'
        cmap = color.colormap('viridis_wrb')
        fig, ax = self.get_fig(**kwargs)
        m = ax.pcolormesh(x[jbase:], y, heat, cmap = cmap)
        cb = fig.colorbar(m, label = label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xbase, xtop)

        fig.tight_layout()
        fig.show()

        self.ii_dump = ii
        self.dump = dump

        self.fig = fig
        self.ax = ax

import winddata
import scales
from kepion import KepIon

class XrbWind(object):
    def __init__(self, filename):
        w = winddata.load(filename)
        ionsel = np.array([i not in (0,2) for i in range(19)])
        c = color.IsoColorBlind(np.count_nonzero(ionsel))
        fig = plt.figure(figsize = (8,6))
        ax = fig.add_subplot(111)
        for i,x in enumerate(np.array(KepIon.approx_ion_names)[ionsel]):
            ax.plot(
                w.time,
                w.dmsion(x),
                label = KepIon(x).LaTeX(),
                color = c[i],
                )
        ax.set_xscale('timescale')
        ax.set_yscale('log')
        ax.set_ylabel('mass loss rate (g/s)')
        ymax = np.max(w.dms)
        ax.set_ylim(np.array([1.e-10, 1]) * ymax)
        fig.tight_layout()
        ax.legend(loc = 'best')
        plt.show()
        self.fig = fig
        self.ax = ax

from utils import is_iterable

class MEX(object):
    def __init__(self, norm = 'c12', mode = 'bdat'):
        if mode == 'bdat':
            M = bdat.BDat()
            ions = M.ext_ions
            me = M.mass_excess(ions)
        elif mode in ('Ame12', 'Ame03', 'Ame16', ):
            M = mass_table.MassTable(version = mode)
            ions = M.ions
            me = M.mass_excess()

        self.me = dict()
        for i,m in zip(ions,me):
            self.me[i] = m

        self.norm = isotope.ion(norm)
        self.menorm = self.get_me(self.norm) / self.norm.A

    def get_me(self, ions):
        if not is_iterable(ions):
            shape = ()
            ions = ions,
        else:
            shape = np.shape(ions)
        if not hasattr(self, '_mass_excess_data'):
            self._mass_excess_data = dict()
        me = []
        for i in ions:
            try:
                me.append(self.me[i])
            except KeyError:
                print(f' [WARNING] NOT FOUND: {i} (returning 0)')
                me.append(0.)
        if shape == ():
            return me[0]
        return np.array(me)


    def sum(self, abu):
        me = self.get_me(abu.ions())
        mex = np.sum(abu.Y() * me) - self.menorm
        return mex


#=======================================================================
# Zac's grid
#=======================================================================

from pathlib import Path
from collections import OrderedDict
from itertools import product

class Grid5(object):
    _default_path = Path('/home/alex/Zac/grid5')
    _default_pattern = r'grid5_{batch}/xrb{run}/xrb{run}z'
    _default_table = r'model_table.txt'

    def __init__(self, base = None, pattern = None, table = None):
        if base is None:
            base = self._default_path
        self.base = base
        if pattern is None:
            pattern = self._default_pattern
        self.pattern = pattern
        if table is None:
            table = self._default_table
        self.table = table
        self.model_table = np.genfromtxt(
            base / table,
            dtype = None,
            names = True,
            )
        self.make_grid()
        self.make_index()
        self.load_models()

    def make_grid(self):
        # could also be generated from uinque entries in table, but
        # would need to know which fields to read.
        filename = self.base / 'README.txt'
        p = re.compile(r'(?m)^(\w+)\s+\[([^\]]+)\]\s*$')
        with open(filename, 'rt') as f:
            text = f.read()
        finds = p.findall(text)
        grid = OrderedDict()
        for seq in finds:
            grid[seq[0]] = np.fromstring(seq[1], sep=',')
        self.grid = grid

    def make_index(self):
        self.index = np.argsort(self.model_table, order=self.keys)
        self.model_table = self.model_table[self.index]
        self.combinations = list(product(*list(self.grid.values())))

    def get_model(self, key):
        ii = np.tile(np.array(True, dtype = bool), self.len)
        for k,v in key.items():
            ii &= self.model_table[k] == v
        return np.where(ii.flat)[0][0]

    def load_models(self):
        self.runs = list()
        for m in self.model_table:
            filename = self.base / self.pattern.format(
                batch = m['batch'],
                run = m['run'],
                )
            self.runs.append(kepdump.load(str(filename)))
            print('.', end = '', flush = True)
        self.runs = np.array(self.runs, dtype = np.object)

    def slice(self, key, return_coordinates = False):
        if isinstance(key, tuple):
            key = {k:v for k,v in zip(self.keys, key) if v is not None}

        ii = np.tile(np.array(True, dtype = bool), self.len)
        values = []
        for k in self.keys:
            v = key.get(k, None)
            if v is None:
                values.append(self.kvalues(k))
            else:
                ii &= self.model_table[k] == v
        dims = [v.shape[0] for v in values]
        result = [np.reshape(np.where(ii)[0], dims)]
        if return_coordinates:
            result.append(values)
        if len(result) == 1:
            return result[0]
        return result

    @property
    def keys(self):
        return list(self.grid.keys())

    @property
    def values(self):
        return list(self.grid.values())

    def kvalues(self, key):
        if isinstance(key, (int, np.int)):
            key = self.keys[key]
        return self.grid[key]

    def kdim(self, key):
        if isinstance(key, (int, np.int)):
            key = self.keys[key]
        return self.grid[key].shape[0]

    @property
    def dims(self):
        return list(len(v) for v in self.grid.values())

    @property
    def len(self):
        return len(self.model_table)

class Grid5Plots(object):
    def __init__(self, grid = None, key = dict(accrate=.1, x=.7, z=None, qb=None, mass=1.4)):
        if grid is None:
            grid = Grid5()
        self.grid = grid
        g = grid
        ME = MEX('Fe56')
        ii = g.slice(key)
        f = lambda x: ME.sum(g.runs[x].abub.project(zones=60))
        me = np.vectorize(f)(ii)
        print(me)
