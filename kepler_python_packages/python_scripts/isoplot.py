"""
Python module to plot isotopic abundances data.
This is to replace IDL/yieldplot[mult]

(under construction)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from matplotlib.patches import PathPatch, Rectangle, Patch
from matplotlib.text import Text, TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
#from matplotlib.cbook import is_numlike
from numbers import Number
from matplotlib.ticker import AutoMinorLocator

from isotope import Ion,  Elements
from abusets import SolAbu
from color import isocolors

from mpl_fixes import MyText as Text

class IsoPlot(object):
    """
    plot isotopes.
    """

    text_cache = dict()

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    colors = isocolors(12)[(np.arange(12)*17) % 12]

    def __init__(self,
                 abu = None,
                 stable = True,
                 xlim = None,
                 ylim = None,
                 y_min = 1e-15,
                 truncate = True,
                 truncate_limit = 1e-23,
                 ax = None,
                 colors = None,
                 linewidth = 1.,
                 markersize = 8.,
                 markerfill = True,
                 markerthick = 1.,
                 fp = None,
                 fontsize = None,
                 pathfont = False,
                 xborder = 0.025,
                 yborder = 0.1,
                 title = None,
                 xtitle = 'mass number',
                 ytitle = 'mass fraction',
                 logy = True,
                 showline = True,
                 showmarker = True,
                 pathmarker = False,
                 align = 'center', # center | first | last
                 showtext = True,
                 stabletext = True,
                 dist = 0.25,
                 norm = None,
                 normtype = 'span',
                 show = None,
                 normrange = 2.):
        """
        Make isotopic abuncance plot.

        abu:
            AbuSet instance

        TODO
        - assert ions sorted?
        - Some of the parameters should be renamed for consistency.
        (under development)
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if show is None:
                show = True
        else:
            if show is None:
                show = False

        if colors is None:
            colors=self.colors
        ncolors = len(colors)

        z = abu.Z()
        a = abu.A()
        x = abu.X()

        if stable is not None:
            if stable is True:
                sol = SolAbu()
                stable = sol.contains(abu)
            elif stable is not False:
                assert len(a) == len(stable)
            else:
                stable = None

        if x.min() <= 1.e-15 and logy:
            y_min = 1.e-15
        else:
            y_min = x.min()
        if logy:
            x = np.log10(np.maximum(x, 1e-99))
            y_min = np.log10(y_min)
            truncate_limit = np.log10(truncate_limit)

        step, =  np.where(z[1:] != z[:-1])
        nz = len(step) + 1
        seq = np.zeros(nz + 1, dtype = np.int64)
        seq[1:-1] = step + 1
        seq[-1] = len(z)

        # we do need limit being set for alignemt use, font scale, etc.
        xlim = self.get_xlim(xlim, xborder, a)

        if ylim is None:
            ylim = np.array([y_min,x.max()])
            ylim += np.array([-1,1]) * yborder * (ylim[1]-ylim[0])
        elif len(ylim) == 1:
            ylim = np.array([ylim, x.max()])
            ylim += np.array([0,1]) * yborder * (ylim[1]-ylim[0])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_autoscalex_on(False)

        ii = -1
        for iz in range(nz):
            na = seq[iz + 1] - seq[iz]
            az = np.ndarray(na)
            xz = np.ndarray(na)
            fz = np.ndarray(na, dtype = np.bool)
            for ia in range(na):
                ii += 1
                az[ia] = a[ii]
                xz[ia] = x[ii]
                fz[ia] = markerfill if stable is None else stable[ii]
            if truncate:
                mi, = np.where(xz > truncate_limit)
                if len(mi) == 0:
                    continue
                az, xz, fz = az[mi], xz[mi], fz[mi]
            color = colors[np.mod(iz,ncolors)]
            if showline:
                line = plt.Line2D(
                    az, xz,
                    color = color,
                    linewidth = linewidth)
                ax.add_line(line)
            if showmarker:
                if pathmarker:
                    Pm = Path.MOVETO
                    Pl = Path.LINETO
                    Pc = Path.CLOSEPOLY
                    for mx,my,mf in zip(az, xz, fz):
                        mpos = ax.transData.transform([[mx,my]])
                        nvert = 64
                        mvert = np.linspace(0,2*np.pi,nvert)
                        p = (np.array((np.sin(mvert),
                                       np.cos(mvert))).transpose()
                             * 0.5*markersize)
                        p += mpos
                        p = ax.transData.inverted().transform(p)
                        c = [Pm] + (nvert-2)*[Pl] + [Pc]

                        path = Path(p, codes = c)
                        if mf:
                            patch = PathPatch(
                                path,
                                clip_on = True,
                                facecolor = color,
                                edgecolor = 'none',
                                linewidth = 0,
                                alpha = 1)
                        else:
                            patch = PathPatch(
                                path,
                                clip_on = True,
                                facecolor = 'none',
                                edgecolor = color,
                                linewidth = markerthick,
                                alpha = 1)
                        ax.add_patch(patch)
                else:
                    for mf in [True, False]:
                        ma = az[fz == mf]
                        mx = xz[fz == mf]
                        if len(ma) == 0:
                            continue
                        if mf:
                            markeredgecolor = 'none'
                            markeredgewidth = 0.
                            markerfacecolor = color
                        else:
                            markeredgecolor = color
                            markeredgewidth = markerthick
                            markerfacecolor = 'none'
                        line = plt.Line2D(
                                ma, mx,
                                marker = 'o',
                                markeredgecolor = markeredgecolor,
                                markeredgewidth = markeredgewidth,
                                linewidth = 0.,
                                markersize = markersize,
                                markerfacecolor = markerfacecolor)
                        ax.add_line(line)

            s = Elements[z[seq[iz]]]

            if fp is None:
                fp = FontProperties(
                    size = fontsize)

            if showtext:
                if stabletext and np.count_nonzero(fz) > 0:
                    mi, = np.nonzero(fz)
                    ms = slice(mi[0], mi[-1]+1)
                    ma, mx = az[ms], xz[ms]
                else:
                    ma, mx = az, xz
                dd,ha,va = self.align(
                    ma, mx, markersize, ax, align, dist,
                    pathfont, pathmarker)
                if pathfont:
                    # These would be useful to paint fonts directly as PathPatch
                    text = self.get_text(s, fp)
                    fxmin, fymin = text.vertices.min(axis=0)
                    fxmax, fymax = text.vertices.max(axis=0)
                    fwidth = fxmax - fxmin
                    fheight = fymax - fymin

                    # dependency on alignment
                    #if not is_numlike(va):
                    if not isinstance(va, Number):
                        if va == 'center':
                            va = 0.5
                        elif va == 'top':
                            va = 1.
                        else:
                            va = 0.
                    #if not is_numlike(ha):
                    if not isinstance(ha, Number):
                        if ha == 'center':
                            ha = 0.5
                        elif ha == 'left':
                            ha = 0.
                        else:
                            ha = 1.
                    y_offset = -fymin - fheight * va
                    x_offset = -fxmin - fwidth  * ha

                    # set target position
                    fx, fy = dd

                    # now we need to find size (assume centered)
                    fext = np.array([[fwidth, fheight]])
                    fpos = ax.transData.transform([[fx,fy]])
                    frange = ax.transData.inverted().transform(
                        np.vstack((fpos - 0.5*fext, fpos + 0.5*fext)))
                    fscale = np.abs(frange[1]-frange[0])/fext.reshape(-1)

                    p = (text.vertices  + [[x_offset, y_offset]]) * [fscale] + [[fx, fy]]
                    c = text.codes

                    path = Path(p, codes = c)
                    patch = PathPatch(
                        path,
                        clip_on = True,
                        facecolor = 'k',
                        edgecolor = 'none',
                        lw = 0,
                        alpha = 1,
                        zorder = 3)
                    ax.add_patch(patch)
                else:
                    text = Text(dd[0],dd[1],s,
                                va = va,
                                ha = ha,
                                clip_on = True,
                                fontproperties = fp,
                                zorder = 3)
                    ax.add_artist(text)


        if ytitle is not None:
            if logy:
                ytitle = 'log( ' + ytitle + ' )'
            ax.set_ylabel(ytitle)
        if xtitle is not None:
            ax.set_xlabel(xtitle)
        if title is not None:
            ax.set_title(title)

        ax.set_xscale('linear')
        ax.set_yscale('linear')

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        if norm is not None:
            #if not is_numlike(norm):
            if not isinstance(norm, Number):
                if isinstance(norm, str):
                    norm = Ion(norm)
                if isinstance(norm, Ion):
                    norm = abu[norm]
            if logy:
                norm = np.log10(norm)

            if normtype == 'line':
                lines = ['--',':']
                colors = ['k','k']
            else:
                lines = ['-','-']
                colors = ['#404040','#E0E0E0']
                if normrange is not None:
                    colors[0] = 'w'

            ax.axhline(norm,
                       linestyle = lines[0],
                       color = colors[0],
                       zorder = -1)
            if normrange is not None:
                if np.size(normrange) == 1:
                    if logy:
                        normrange = norm + np.log10(np.array([1./normrange,normrange]))
                    else:
                        normrange = norm + np.array([-normrange,+normrange])
                    if normtype == 'line':
                        ax.axhline(normrange[0],
                                   linestyle = lines[1],
                                   color = colors[1],
                                   zorder = -1)
                        ax.axhline(normrange[1],
                                   linestyle = lines[1],
                                   color = colors[1],
                                   zorder = -1)
                    else:
                        ax.axhspan(*normrange,
                                    color = colors[1],
                                    zorder = -2)

            #  ax.set_position((0.078,0.07,0.921,0.929))

        self.ax = ax
        self.figure = ax.figure

        if show:
            plt.draw()

    @staticmethod
    def get_xlim(xlim, xborder, a):
        if xlim is None:
            xlim = np.array([a.min(), a.max()], dtype=np.float64)
            if xborder > 0.5:
                xlim += np.array([-1,1]) * xborder
            else:
                xlim += np.array([-1,1]) * xborder * (xlim[1]-xlim[0])
        return xlim


    def align(self,
              x, y, symsize, ax,
              align = 'first',
              dist = 0.25,
              pathfont = None,
              pathmarker = None):

        vax=['bottom','top']

        if pathmarker == True and pathfont == True:
            dy = symsize
            vax = np.array([1.,0.])
        if pathmarker == True and pathfont == False:
            dy = symsize * 0.5
            vax = np.array([1.,0.]) + np.array([1.,-1.])*dist
        if pathmarker == False and pathfont == True:
            dy = symsize * 0.5
            vax = np.array([1.,0.]) + np.array([1.,-1.])*dist
        if pathmarker == False and pathfont == False:
            dist = 0.5
            vax = np.array([1.,0.]) + np.array([1.,-1.])*dist
            dy = 0.

        if align == 'first':
            if len(x) > 1:
                top = y[0] > y[1]
            else:
                yy = ax.get_ylim()
                top = y[0] > 0.5 * (yy[0] + yy[1])
            top = 1 if top else 0
            va = vax[top]
            ha = 'center'
            p = np.array([[x[0],y[0]]])
            d = np.array([[0, dy *[-1,+1][top]]])
        if align == 'last':
            if len(x) > 1:
                top = y[-1] > y[-2]
            else:
                yy = ax.get_ylim()
                top = y[-1] > 0.5 * (yy[0] + yy[1])
            top = 1 if top else 0
            va = vax[top]
            ha = 'center'
            p = np.array([[x[-1],y[-1]]])
            d = np.array([[0, dy * [-1,+1][top]]])
        if align == 'center':
            va = None
            if len(x) > 1:
                xm = 0.5 * (x[-1] + x[0])
                i = 0
                while x[i] < xm - 0.1:
                    i += 1
                if np.abs(x[i]-xm) < 0.1:
                    top = (( y[i-1]*(x[i+1]-x[i])
                             + y[i+1]*(x[i]-x[i-1]))
                           /(x[i+1]-x[i-1])
                           < y[i] )
                else:
                    d = np.array([[0., 0.]])
                    p = 0.5 * np.array([[x[i-1]+x[i],y[i-1]+y[i]]])
                    va = 'center'
            else:
                yy = ax.get_ylim()
                top = y[0] > 0.5 * (yy[0] + yy[1])
                i = 0
            ha = 'center'
            if va is None:
                top = 1 if top else 0
                va = vax[top]
                p = np.array([[x[i],y[i]]])
                d = np.array([[0., dy * [-1,+1][top]]])

        p = ax.transData.transform(p)
        p += d
        p = ax.transData.inverted().transform(p)
        return p[0], ha, va

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
            # print('creating {:s}'.format(s))
            text = self.text_cache.setdefault(
                s,
                TextPath(
                    xy=(0,0),
                    s = s,
                    fontproperties = fp)
                )
        return text

class IsoPlotMult(object):
    def __init__(
        self,
        *args,
        **kwargs):
        """
        pass most parameters to IsoPlot.
        """

        kw = dict(kwargs)

        xborder = kw.get('xborder', 1)
        xlim    = kw.get('xlim', None)
        abu     = kw.get('abu', None)
        ax      = kw.get('ax', None)
        logy    = kw.setdefault('logy', True)
        ytitle  = kw.setdefault('ytitle', 'mass fraction')
        overlap = kw.pop('overlap', 5)
        sections= kw.pop('sections', 3)
        if abu is None and len(args) > 0:
            abu = args[0]
            kw['abu'] = abu
        isoplot = kw.pop('isoplot', IsoPlot)

        assert ax is None or len(ax) == sections, "cannot use single axis"
        assert sections >= 1, "need at least one section"
        assert sections == round(sections), "sections needs to be a whole"

        a = abu.A()
        xxlim = np.ndarray((sections, 2),
                           dtype=np.float64)
        if xlim is None:
            xlim = np.array([a.min(), a.max()])
            if xborder <= 0.5:
                xborder *= (xlim[1]-xlim[0])
            xlim += np.array([-1,1]) * xborder
        xwidth = xlim[1] - xlim[0]
        if overlap <= 0.5:
            overlap *= xwidth
        ob = overlap + 2 * xborder
        totlen = xwidth + (sections-1)*ob
        secwidth = totlen / sections

        xxlim = np.array([[i*(secwidth - ob),
                           (i+1)*secwidth - i*ob] for i in range(sections)])
        xxlim += xlim[0]
        figure = kwargs.get('figure', None)
        if figure is None:
            figure = plt.figure()
        if ax is None:
            axs = np.ndarray(sections, dtype = np.object)
            axs[sections-1] = figure.add_subplot(sections, 1, sections)
            for i in range(sections-2,-1,-1):
                axs[i] = figure.add_subplot(sections, 1, i+1,
                                         sharey = axs[sections-1])
        else:
            axs = ax
        ytitle_pos = np.round((sections-1.5)/2)
        plots = []
        for i in range(sections):
            kwx = dict(kw)
            kwx['ax'] = axs[i]
            kwx['xlim'] = xxlim[i]
            kwx['xborder'] = 0
            if i != sections-1:
                kwx['xtitle'] = None
            if i != ytitle_pos:
                kwx['ytitle'] = None
            else:
                if sections % 2 == 0:
                    l = axs[i].yaxis.get_label()
                    p = l.get_position()
                    p = (p[0],-0.5 * figure.subplotpars.wspace)
                    l.set_position(p)
            plots += [isoplot(**kwx)]

        self.plots = plots
        self.axs = axs
        self.figure = figure

def test():
    from kepdump import loaddump
    a = loaddump('/home/alex/kepler/test/s25#presn').AbuSet(300)
    IsoPlotMult(abu = a, sections = 4)
    plt.draw()


def test2():
    s1 = SolAbu('Lo09')
    s2 = SolAbu('As12')
    r = s1 / s2
    IsoPlotMult(
        r,
        ytitle = 'Lodders 2009 / Aspund 2012',
        overlap = 2,
        # sections = 6,
        sections = 1,
        logy = False)

def test3():
    s1 = SolAbu('Lo09')
    s2 = SolAbu('Lo03')
    r = s1 / s2
    IsoPlotMult(
        r,
        ytitle = 'Lodders 2009 / Lodders 2003',
        overlap = 2,
        # sections = 6,
        sections = 1,
        logy = False)

def test4():
    s1 = SolAbu('As12')
    s2 = SolAbu('As09')
    r = s1 / s2
    IsoPlotMult(
        r,
        ytitle = 'Asplund 2012 / Asplund 2009',
        overlap = 2,
        # sections = 6,
        sections = 1,
        logy = False)

if __name__ == "__main__":
    pass
