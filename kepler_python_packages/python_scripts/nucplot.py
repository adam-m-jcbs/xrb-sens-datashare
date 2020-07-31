"""
Python module to plot chart of nuclei.

(under construction)
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle, Patch
from matplotlib.text import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
from matplotlib.ticker import MaxNLocator, FuncFormatter

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm, colorConverter

from itertools import chain

from isotope import ion, Ion
from abuset import IonSet, AbuSet
from abusets import SolAbu
from logged import Logged
from color import colormap


def split_path(px, cx):
    ii = np.where(np.in1d(cx, [Path.CLOSEPOLY, Path.STOP]))[0]
    ii = np.hstack(([-1], ii))
    if ii[-1] < len(cx) - 1:
        ii = np.hstack((ii, [len(cx)-1]))
    return tuple((px[i:j,:], cx[i:j]) for i,j in zip(ii[:-1]+1, ii[1:]+1))

def reverse_path(px, cx):
    c = np.ndarray((0))
    p = np.ndarray((0,2))
    for py, cy in split_path(px, cx):
        assert cy[0] == Path.MOVETO
        # it may be possible to fix the above case, but ...
        if cy[-1] in (Path.CLOSEPOLY, Path.STOP):
            pi = np.vstack((py[-2::-1,:],
                            py[-1:,:]))
            ci = np.hstack((cy[0],
                            cy[-2:0:-1],
                            cy[-1]))
        else:
            pi = py[::-1,:]
            ci = np.hstack((cy[0], cy[-1:0:-1]))

        p = np.vstack((p, pi))
        c = np.hstack((c, ci))
    return p, c


class Iso(object):
    """
    Plot one Isotope.

    TODO: (speed)
    *) transform manually
    *) return paths (instead) so they can be appended mannually
       and only one patch is generated from combined path
    [generation of paths seems to take exessive time]
    [for movies: that could be reused!]
    """
    text_cache = dict()
    ref_cache = [None]*6
    rec = np.array([[0,0],[0,1],[1,1],[1,0],[0,0]])
    def __init__(self,
                 iso = ion('C12'),
                 A = None,
                 Z = None,
                 iso_ref = ion('Sm222'),
                 border = 0.15,
                 frame = 0.1,
                 color = 'black',
                 fp = None,
                 scale = None,
                 axes = None,
                 N0 = 0,
                 Z0 = 0,
                 overlap = True,
                 stable = None,
                 space = 0.05,
                 clip_on = True,
                 alpha = 1,
                 zorder = 1,
                 draw = True,
                 showrect = True,
                 textangle = 0,
                 solid = False,
                 showtext = True,
                 verbose = False,
                 ):

        self.verbose = verbose

        if axes is None and draw is True:
            figure = plt.figure()
            axes = fig.add_subplot(111)
            axes.set_aspect('equal')
            show = True
        else:
            show = False
        self.axes = axes

        if not isinstance(iso, Ion):
            iso = ion(iso)
        if Z is None:
            Z = iso.Z
        if A is None:
            A = iso.A
        N = A - Z
        s = iso.mpl()

        if not isinstance(iso_ref, Ion):
            iso_ref = ion(iso_ref)
        s_ref = iso_ref.mpl()

        self.A = A
        self.Z = Z
        self.N = N
        self.ion = ion(iso)

        # create vertices
        Pm = Path.MOVETO
        Pl = Path.LINETO
        Pc = Path.CLOSEPOLY

        c = np.ndarray((0))
        p = np.ndarray((0,2))

        if showrect:
            if overlap:
                pr = -0.5*frame + (1 + frame)*self.rec
                if not solid:
                    pr = np.vstack((pr,
                        0.5*frame + (1 - frame)*self.rec[::-1,:]))
            else:
                pr = +0.5*space + (1 - space)*self.rec
                if not solid:
                    pr = np.vstack((pr,
                        0.5*space + frame + (1 - 2*(frame) - space)*self.rec[::-1,:]))
            pr += np.array([[N + N0, Z + Z0]]) - 0.5
            cr = np.array([Pm] + [Pl]*3 + [Pc])
            if not solid:
                cr = np.hstack((cr, cr))

            p = np.vstack((p, pr))
            c = np.hstack((c, cr))

        corner = np.array([[0,0],[1,1]])
        if overlap:
            box = -0.5*frame + (1 - frame)*corner
        else:
            box = +0.5*space + frame + (1 - 2*(frame) - space)*corner
        self.box = box + np.array([[N + N0, Z + Z0]]) - 0.5

        # TEXT

        # get ref scale
        # buffer last request
        if scale is None and showtext:
            if (s_ref     == self.ref_cache[0] and
                fp        == self.ref_cache[1] and
                border    == self.ref_cache[2] and
                space     == self.ref_cache[3] and
                textangle == self.ref_cache[4]):
                self.scale = self.ref_cache[5]
            else:
                text = self.get_text(s_ref, fp)
                v = self.rotvert(text.vertices, textangle)
                fxmin, fymin = v.min(axis=0)
                fxmax, fymax = v.max(axis=0)
                fwidth = fxmax - fxmin
                fheight = fymax - fymin
                if overlap:
                    self.scale = (1 - 2 * border)/fwidth
                else:
                    self.scale = (1 - 2 * border - space)/fwidth
                self.ref_cache[:] = [s_ref, fp, border, space, textangle, self.scale]

        if showtext:
            self.text = self.get_text(s, fp)

            v = self.rotvert(self.text.vertices, textangle)

            fxmin, fymin = v.min(axis=0)
            fxmax, fymax = v.max(axis=0)
            fwidth = fxmax - fxmin
            fheight = fymax - fymin
            y_offset = -fymin + 0.5 * -fheight
            x_offset = -fxmin + 0.5 * -fwidth

            x = N0 + N
            y = Z0 + Z

            pt = (v  + [[x_offset, y_offset]]) * self.scale + [[x, y]]
            ct = self.text.codes

            if solid:
                pt, ct = reverse_path(pt, ct)

            p = np.vstack((p, pt))
            c = np.hstack((c, ct))


        self.p = p
        self.c = c

        if draw is False:
            return

        self.path = Path(
            self.p,
            codes = self.c)
        self.patch = PathPatch(
            self.path,
            clip_on = clip_on,
            facecolor = color,
            edgecolor = 'none',
            lw = 0,
            alpha = alpha,
            zorder = zorder)
        if axes is not None:
            axes.add_patch(self.patch)
        if show:
            plt.draw()

    @staticmethod
    def rotvert(v, angle):
        ca = np.cos(angle / 180. * np.pi)
        sa = np.sin(angle / 180. * np.pi)
        mat = np.array([[ca,-sa],
                        [sa, ca]])
        v = np.inner(v,mat)
        return v

    def get_text(
        self,
        s,
        fp = None):
        """
        get text vertices, buffered.
        """
        # these need to be stored in a cache file instead?
        try:
            text = self.text_cache[s]
        except:
            if self.verbose:
                print('creating {:s}'.format(s))
            text = self.text_cache.setdefault(
                s,
                TextPath(
                    xy=(0,0),
                    s = s,
                    fontproperties = fp)
                )
        return text

    def fill(self,
             color,
             axes = None,
             alpha = None,
             zorder = 2,
             clip_on  = True):
        p = np.array([self.box[self.rec[:,0],0], self.box[self.rec[:,1],1]]).transpose()
        Pm = Path.MOVETO
        Pl = Path.LINETO
        Pc = Path.CLOSEPOLY
        c = np.array([Pm] + [Pl]*3 + [Pc])
        path = Path(
            p,
            codes = c)
        rgba = colorConverter.to_rgba(color, alpha)
        patch = PathPatch(
            path,
            clip_on = clip_on,
            facecolor = rgba,
            edgecolor='none',
            lw = 0,
            zorder = zorder)
        if axes is not None:
            axes.add_patch(patch)
        else:
            return patch

    @staticmethod
    def frac_coord(fraction,
                   angle = 45.):
        """
        Return fraction coordinates ((x0,y0),(x1,y1)) and begining and
        end sides (s0,s1) as np.ndarray.

        fraction is 0..1
        angle is deg if abs(angle) > 2*pi
        sides are 0,1,2,3
        """
        if abs(angle) > 2*np.pi:
            angle = angle * np.pi / 180.
        angle = (angle + np.pi*2) % (2. * np.pi)

        # deterime quadrant and rotate into first quadrant
        quadrant = int(angle / (np.pi/2))
        angle = angle - np.pi/2 * quadrant

        # now we should be in first quadrant (quadtant number 0)
        assert 0 <= angle <= np.pi/2

        # symmetry around diagonal in first octant
        mode = 0
        if angle > np.pi/4:
            mode = 1
            angle = np.pi/2 - angle

        # deal only with first octant
        assert angle <= np.pi/4
        A0 = 0.5 * np.tan(angle)
        if fraction < A0:
            f = np.sqrt(fraction / A0)
            x0 = 1. - f
            y0 = 0.
            x1 = 1.
            y1 = f * 2 * A0
            s0 = 0
            s1 = 1
        elif  fraction > 1. - A0:
            f = np.sqrt((1. - fraction) / A0)
            x0 = 0.
            y0 = 1. - f * 2 * A0
            x1 = f
            y1 = 1.
            s0 = 3
            s1 = 2
        else:
            x0 = 0.
            x1 = 1.
            y0 = fraction - A0
            y1 = 2 * A0 + y0
            s0 = 3
            s1 = 1

        # map first octant into first quadrant
        if mode % 2 == 1:
            ss = [1,0,3,2]
            x0,y0,x1,y1 = 1-y1, 1-x1, 1-y0, 1-x0
            s0, s1 = ss[s1], ss[s0]

        # map odd quadrants
        if quadrant % 2 == 1:
            ss = [1,2,3,0]
            x0,y0,x1,y1 = 1-y0, x0, 1-y1, x1
            s0, s1 = ss[s0], ss[s1]

        # map lower plane (Australia)
        if quadrant >= 2:
            ss = [2,3,0,1]
            x0,y0,x1,y1 = 1-x0, 1-y0, 1-x1, 1-y1
            s0, s1 = ss[s0], ss[s1]

        return np.array([[x0,y0],[x1,y1]]), np.array((s0, s1))


    @staticmethod
    def frac_corner(sides):
        """
        Return propoer corner coordinate if both sides are not the
        same or None otherwise.
        """
        s0 = sides.min()
        s1 = sides.max()
        if s0 == s1:
            return None
        assert s1 - s0 in (1,3)
        if s0 == 0 and s1 == 1:
            return np.array([[1,0.]])
        elif s0 == 1:
            return np.array([[1,1.]])
        elif s0 == 2:
            return np.array([[0,1.]])
        return np.array([[0,0.]])

    def frac_fill_one(self,
                      fractions,
                      color,
                      angle = 45.,
                      alpha = None,
                      zorder = -1,
                      axes = None,
                      show = True,
                      clip_on  = True):
        """
        fill one fraction section

        needs one color and 2 fraction values
        """
        fractions = np.array(fractions)
        assert len(fractions) == 2
        assert fractions[0] < fractions[1]

        coord0, sides0 = self.frac_coord(fractions[0], angle)
        coord1, sides1 = self.frac_coord(fractions[1], angle)
        corner0 = self.frac_corner(np.array([sides0[0],sides1[0]]))
        corner1 = self.frac_corner(np.array([sides0[1],sides1[1]]))
        p = coord0.copy()
        if corner1 is not None:
            p = np.vstack((p ,corner1))
        p = np.vstack((p , coord1[::-1]))
        if corner0 is not None:
            p = np.vstack((p ,corner0))
        p = np.vstack((p, coord0[np.array([0])]))

        # transform
        p = (self.box[0,:][np.newaxis,:] +
             (self.box[1,:] - self.box[0,:])[np.newaxis,:] * p)

        Pm = Path.MOVETO
        Pl = Path.LINETO
        Pc = Path.CLOSEPOLY
        c = np.array([Pm] + [Pl]*(p.shape[0]-2) + [Pc])
        path = Path(
            p,
            codes = c)
        rgba = colorConverter.to_rgba(color, alpha)
        patch = PathPatch(
            path,
            clip_on = clip_on,
            facecolor = rgba,
            edgecolor='none',
            lw = 0,
            zorder = zorder)
        if axes is None:
            axes = self.axes
        if axes is None:
            show = False
        if show:
            axes.add_patch(patch)
        else:
            return patch

    def frac_fill(self,
                  fractions,
                  colors,
                  angle = 45.,
                  alpha = None,
                  zorder = -1,
                  axes = None,
                  show = True,
                  cummulative = True,
                  normalize = True,
                  clip_on  = True):
        """
        fill set of fractions with corresponding colors.

        assume fractions start at 0 unless there is one more fractions
        than colors.

        'cummulative' assume fractions are relative. (default)

        'normalize' normailzes fractions to 1.0 (default)
        """
        fractions = np.array(fractions)
        colors = np.array(colors)
        if cummulative:
            frac = fractions.cumsum()
        else:
            frac = fractions
        if fractions.size == colors.size:
            frac = np.insert(frac, 0, 0)
        if normalize:
            frac /= frac[-1]
        assert frac.size == colors.size + 1
        if axes is None:
            axes = self.axes
        if axes is None:
            show = False
        patches = []
        for f0,f1,c in zip(frac[:-1],frac[1:], colors):
            p = self.frac_fill_one((f0,f1), c,
                                   angle = angle,
                                   alpha = alpha,
                                   zorder = zorder,
                                   axes = axes,
                                   show = show,
                                   clip_on = clip_on)
            if p is not None:
                patches += [p]
        if show:
            return patches

class IsoPlot(Logged):
    magic = np.array([2,8,20,28,50,82,126])

    def __init__(self,
                 ions,
                 abu = None,
                 figure = None,
                 axes = None,
                 onepath = True,
                 showmagic = True,
                 magicmode = 'line', # 'line' | 'span'
                 magicspanwidth = 0.01,
                 magiclinewidth = 0.5,
                 yelement = False,
                 cm = None,
                 label = 'ion', # ion | EA | EN | None
                 alpha = 1,
                 log_abu_min = -9,
                 log_abu_max = 0,
                 showrect = True, # True | False | 'solar'
                 solabu = None,
                 rescale = 1,
                 figsize = None,
                 pixelsize = None,
                 dpi = None,
                 **kwargs):

        def format_el(z, pos=None):
            """Element tick label formatter."""
            if z < 0:
                return ''
            try:
                iso = ion(Z=int(round(z)))
            except IndexError:
                iso = ion(Ion.VOID)
            try:
                s = iso.mpl()
            except IndexError:
                s = ''
            return s

        if isinstance(ions, AbuSet):
            abu = ions.X()
            ions = ions.ions()
        if not isinstance(ions, IonSet):
            ions = IonSet(ions)
        kwargs.setdefault('overlap', False)
        kwargs.setdefault('draw', not onepath)
        if axes is None:
            if figure is None:
                if figsize is None:
                    figsize = np.array([6.4, 4.8])
                figure = plt.figure(figsize = figsize)
                if dpi is None:
                    dpi = figure.get_dpi()
                else:
                    figure.set_dpi(dpi)
                if pixelsize is not None:
                    figsize = np.array(pixelsize) / dpi
                    figure.set_size_inches(figsize)
                if rescale != 1:
                    dpi *= rescale
                    figsize /= rescale
                    figure.set_dpi(dpi)
                    figure.set_size_inches(figsize)
            axes = figure.add_subplot(111)
            axes.set_aspect('equal')
        kwargs['axes'] = axes
        kwargs.setdefault('showtext', label == 'ion')
        kwargs.setdefault('showrect', showrect)

        if showrect == 'solar':
            if solabu is None:
                solabu = SolAbu(silent = True)

        color = kwargs.setdefault('color', 'black')
        solid = kwargs.setdefault('solid', False),
        clip_on = kwargs.setdefault('clip_on',  True)

        iso = []
        for i,ix in enumerate(ions):
            kw = kwargs.copy()
            for k,v in kw.items():
                if np.isscalar(v) or np.iterable(v) == 0:
                    continue
                if len(v) == len(ions):
                    kw[k] = v[i]
            if showrect == 'solar':
                kw['showrect'] = ix in solabu
            iso.append(Iso(ix, **kw))
        self.iso = np.array(iso)

        axes.set_xlabel('neutron number')
        if yelement:
            axes.set_ylabel('element')
            axes.yaxis.set_major_formatter(FuncFormatter(format_el))
        else:
            axes.set_ylabel('charge number')
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes = axes

        A, Z, N = np.array([[i.A, i.Z, i.N] for i in self.iso]).transpose()
        Nmax = N.max()
        Zmax = Z.max()
        Amax = A.max()
        Nmin = N.min()
        Zmin = Z.min()
        Amin = A.min()

        border = kwargs.get('border', 1.5)
        axes.set_xlim(kwargs.get('xlim', [Nmin-border, Nmax+border]))
        axes.set_ylim(kwargs.get('ylim', [Zmin-border, Zmax+border]))

        if showmagic:
            parms = dict(
                color = 'k',
                aa = True,
                )
            if magicmode == 'span':
                for m in self.magic:
                    for f in (axes.axhspan, axes.axvspan):
                        f(m - 0.5 - 0.5 * magicspanwidth, m - 0.5 + 0.5 * magicspanwidth, **parms)
                        f(m + 0.5 - 0.5 * magicspanwidth, m + 0.5 + 0.5 * magicspanwidth, **parms)
            else:
                for m in self.magic:
                    for f in (axes.axhline, axes.axvline):
                        f(m - 0.5, lw = magiclinewidth, **parms)
                        f(m + 0.5, lw = magiclinewidth, **parms)
        z_label = ()
        n_label = ()
        a_label = ()
        if label in ('EN', 'EA', ):
            # find min/max A/Z/N
            Np = Nmax + 1
            Zp = Zmax + 1
            Ap = Amax + 1

            ZN = np.ndarray((Zp))
            ZN[:] = Np
            for n,z in zip(N,Z):
                ZN[z] = min(ZN[z],n)

            NZ = np.ndarray((Np))
            NZ[:] = Zp
            for n,z in zip(N,Z):
                NZ[n] = min(NZ[n],z)

            AZ = np.ndarray((Ap))
            AZ[:] = Zp
            for a,z in zip(A,Z):
                AZ[a] = min(AZ[a],z)

            kwargs['showtext'] = True
            kwargs['showrect'] = False
            z_label =  np.array([Iso(iso = ion(Z=z),
                                     iso_ref = ion('Xe'),
                                     Z = z,
                                     A = z+zn-1,
                                     **kwargs)
                                 for z,zn in enumerate(ZN) if zn < Np])
            if label == 'EN':
                n_label = np.array([Iso(iso = ion(N=n),
                                        iso_ref = ion(N=128),
                                        Z = nz-1,
                                        A = n+nz-1,
                                        **kwargs)
                                    for n,nz in enumerate(NZ) if nz < Zp])
            else:
                a_label = np.array([Iso(iso = ion(A=a),
                                        iso_ref = ion(A=222),
                                        Z = az-1,
                                        A = a,
                                        textangle = -45,
                                        **kwargs)
                                    for a,az in enumerate(AZ) if az < Zp])

        if onepath:
            p = np.ndarray((0,2))
            c = np.ndarray((0))
            for iso in chain(self.iso, z_label, n_label, a_label):
                p = np.vstack((p, iso.p))
                c = np.hstack((c, iso.c))

            path = Path(
                p,
                codes = c)
            patch = PathPatch(
                path,
                clip_on = clip_on,
                facecolor = color,
                edgecolor = 'none',
                lw = 0,
                alpha = alpha)
            axes.add_patch(patch)

        layout_fixes = []

        if abu is not None:
            if cm is None:
                cm = 'GalMap3'
            cm = colormap(cm)
            for i,a in zip(self.iso, abu):
                a = (np.log10(np.maximum(a, 1.e-99)) - log_abu_min + log_abu_max) / (log_abu_max - log_abu_min)
                if a > 0:
                    colors = cm(a)
                    i.fill(colors, axes)
            norm = Normalize(vmin = log_abu_min, vmax = log_abu_max, clip = True)
            sm = ScalarMappable(norm = norm, cmap = cm)
            sm.set_array(np.array([log_abu_min, log_abu_max]))
            cbar = figure.colorbar(sm, ax = axes)
            #cbar.ax.yaxis.label.set_ha('center')
            #cbar.ax.yaxis.label.set_va('top')
            #cbar.set_label('log(mass fraction)')
            cbar.ax.text(0.5, 0.5, 'log(mass fraction)',
                         color = 'white',
                         rotation = 'vertical',
                         ha = 'center',
                         va = 'center',
                         weight = 'bold',
                         transform = cbar.ax.transAxes)
            layout_fixes.append('cbar')
            self.cbar = cbar
            self.sm = sm
            self.cm = cm

        self.figure = figure
        self.axes = axes

        figure.tight_layout()

        for f in layout_fixes:
            if f == 'cbar':
                cbar.ax.set_position(cbar.ax.get_position().translated(+.05, 0))

        figure.show()

from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d

class IsoPlot3D(Logged):
    def __init__(self, data, plot = True, **kwargs):
        assert isinstance(data, AbuSet)
        self.data = data
        if plot:
            self.plot(**kwargs)

    def plot(self, zrange = [0,999], nrange = [0, 999]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d',aspect='equal')
        fp = FontProperties()
        data = self.data
        p = np.ndarray((0,2))
        c = np.ndarray((0))
        n = len(data)

        X = []
        Z = []
        N = []
        for i,xk in data:
            if not ((nrange[0] <= i.N <= nrange[1]) and
                    (zrange[0] <= i.Z <= zrange[1])):
                continue
            ii = Iso(
                i,
                overlap = False,
                draw = False,
                fp = fp)
            p = np.vstack((p, ii.p))
            c = np.hstack((c, ii.c))
            X.append(xk)
            N.append(i.N)
            Z.append(i.Z)

        X = np.array(X)
        N = np.array(N)
        Z = np.array(Z)

        path = Path(
            p,
            codes = c)

        patch = PathPatch(
            path,
            clip_on = True,
            facecolor='black',
            edgecolor='none',
            lw = 0,
            alpha=1)

        ax.add_patch(patch)

        pathpatch_2d_to_3d(
            patch,
            z=0,
            zdir='z')

        dx = dy = dz = X**(1/3)

        xpos = N - 0.5*dx
        ypos = Z - 0.5*dy
        zpos = dz * 0.

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
                 color=[0,.8,.2],
                 edgecolor=[0,.8,.2],
                 lw = 0.1,
                 zsort='average',
                 alpha=1.)

        lim = max(np.max(N),np.max(Z))

        ax.set_xlabel('N')
        ax.set_ylabel('Z')
        ax.set_zlabel('data')

        ax.set_xlim3d(-1, lim + 1)
        ax.set_ylim3d(-1, lim + 1)
        ax.set_zlim3d( 0, lim + 2)

        fig.show()


def test():
    data = SolAbu()
    i = IsoPlot(data, label='EN')
    return i

def testf():
    ions = np.array([ion('h1')])
    ions = IonSet(ions)
    p = IsoPlot(ions, label='EA')
    p.iso[0].frac_fill([.2,.5,.3],('#ff0000','#00FF00','#0000ff'), alpha = .25, angle = 225.)
    p.iso[0].frac_fill([.2,.5,.3],('#ff0000','#00FF00','#0000ff'), alpha = .25, angle = 135.)

def test3d():
    data = SolAbu()
    IsoPlot3D(data, plot = True, zrange = [0, 10])

def testnuc():
    ions = IonSet(['c12','c13','n14','n15','o16'])
    ions = IonSet(['o16','f17','ne18','f18','ne19','f19'])
    p = IsoPlot(ions, color='blue')

def testnuc1():
    ions = IonSet(['sm221','sm222','sm223'])
    for i in ions.copy():
        ions.add(i + 'h2')
        ions.add(i + 'he4')
    p = IsoPlot(ions, solid = True)

def testnuc2():
    ions = IonSet(['ca40','ca41'])
    p = IsoPlot(ions, solid = [True, False])

import kepdump

def testb(dump = '/home/alex/kepler/test/adelle/run32/run32#5000', **kwargs):
    d = kepdump.load(dump)
    b = d.abub
    a = b.project(zones = kwargs.pop('zones', 51))
    kwargs.setdefault('label', None)
    kwargs.setdefault('showrect', 'solar')
    IsoPlot(a, **kwargs)


if __name__ == "__main__":
    pass
