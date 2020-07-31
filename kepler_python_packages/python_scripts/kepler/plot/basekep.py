"""
Basic plot functionallyity for KEPLER plots.
"""

# TODO: retain axes if xzoomed
# t = p.fig.canvas.toolbar._nav_stack
# t._elements
# list(e.items())
# contains axes p.ax etc as keys
# t._pos is current position (0-based), -1 if no zoom yet
# need to retain history and reset after update

import numpy as np

from matplotlib.pylab import rcParams
from matplotlib.ticker import AutoMinorLocator
from matplotlib.transforms import Transform, Affine2D, \
     blended_transform_factory, Bbox, TransformedBbox
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.text import TextPath, Text
from matplotlib.path import Path

from math import sqrt

from logged import Logged

from .dataplot import DataPlot
from .xvar import XVar
from .grid import Grid

class DataTransform(Transform):
    input_dims = 2
    output_dims = 2
    is_separable = False
    has_inverse = False
    def __init__(self, xdata, ydata, ax, undef = np.nan):
        self.xdata = xdata
        self.ydata = ydata
        self.yscale = 'linear'
        if ax.get_yscale() == 'log':
            with np.errstate(all='ignore'):
                self.ydata = np.log(self.ydata)
            self.yscale = 'log'
        self.ax = ax
        self._parents = dict()
        self.undef = undef
    def transform_non_affine(self, values):
        trans = self.ax.transAxes + self.ax.transData.inverted()
        res = values.copy()
        xpos = trans.transform(values)[:,0]
        ii = np.argsort(self.xdata)
        yval = np.interp(xpos, self.xdata[ii], self.ydata[ii], left=self.undef, right=self.undef)
        if self.yscale == 'log':
            yval = np.exp(yval)
        res[:,1] = yval
        res[:,0] = xpos
        return self.ax.transData.transform(res)

class DataPointTransform(DataTransform):
    def __init__(self, xdata, ydata, ax, x, **kwargs):
        super().__init__(xdata, ydata, ax, **kwargs)
        self.x = x
    def transform_non_affine(self, values):
        f0 = super().transform_non_affine(np.array([self.x, 1.]).reshape(-1, values.shape[1]))[0]
        unit = self.ax.figure.get_dpi() / 72
        f = Affine2D().scale(unit).translate(f0[0], f0[1]).transform(values)
        return f

class TextDataTransform(Transform):
    input_dims = 2
    output_dims = 2
    is_separable = False
    has_inverse = False
    def __init__(self, ax, xpos, ypos, scale = 1):
        self.ax = ax
        self.scale = scale
        self.xpos = xpos
        self.ypos = ypos
        self._parents = dict()
    def transform_non_affine(self, values):
        f0 = self.ax.transData.transform([self.xpos, self.ypos])
        unit = self.scale * self.ax.figure.get_dpi() / 72
        f = Affine2D().scale(unit).translate(f0[0], f0[1]).transform(values)
        return f

class BandTransform(Transform):
    input_dims = 2
    output_dims = 2
    is_separable = False
    has_inverse = False
    def __init__(self, ax, size = 10, ypos = 0.5, align = 0):
        self.ax = ax
        self.size = size
        self.ypos = ypos
        self.align = align
        self._parents = dict()
    def transform_non_affine(self, values):
        unit = self.size * self.ax.figure.get_dpi() / 72
        y0 = self.ax.transAxes.transform([[0, self.ypos]])[0,1]
        data = blended_transform_factory(
            self.ax.transData,
            Affine2D().scale(unit).translate(0, y0 - unit * self.align)
            ).transform(values)
        return data

class TextLegendHandler(object):
    def __init__(self, text = None, **kwargs):
        self.text = text
        self.kwargs = kwargs

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        assert isinstance(orig_handle, self.__class__)
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        x = x0 + 0.5 * width
        y = y0 + 0.2 * height

        text = orig_handle.text
        if text is None:
            text = self.text or '---'
        kwargs = dict(
            x = x,
            y = y,
            text = text,
            ha = 'center',
            va = 'baseline',
            fontsize = fontsize,
            )
        kwargs.update(self.kwargs)
        kwargs.update(orig_handle.kwargs)
        t = Text(**kwargs)
        handlebox.add_artist(t)
        return t


class PathLegendHandler(object):
    def __init__(self, path = None, scale = None, **kwargs):
        self.path = path
        self.scale = scale
        self.kwargs = kwargs

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        assert isinstance(orig_handle, self.__class__)
        assert self.path is None
        scale = 1
        if orig_handle.scale is not None:
            scale = orig_handle.scale
        if self.scale is not None:
            scale *= self.scale
        kwargs = dict(
            clip_on = False,
            )
        kwargs.update(self.kwargs)
        kwargs.update(orig_handle.kwargs)
        xmode = kwargs.pop('xmode', 'center')
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        # THIS IS A BUG IN LEGEND
        # trans = Affine2D().translate(x0 + 0.5 * width, y0).scale(scale)
        # kwargs['transform'] = trans
        # path = orig_handle.path
        # patch = PathPatch(path, **kwargs)
        verts = orig_handle.path.vertices.copy()
        codes = orig_handle.path.codes.copy()
        if xmode == 'fill':
            verts[:,0] = x0 + verts[:,0] * width
        else:
            verts[:,0] += x0 + 0.5 * width
        verts[:,1] = y0 + verts[:,1] * scale
        path = Path(verts, codes = codes)
        patch = PathPatch(path, **kwargs)
        handlebox.add_artist(patch)
        return patch

class VertsLegendHandler(object):
    def __init__(self, verts = None, codes = None, **kwargs):
        self.verts = verts
        self.codes = codes
        self.kwargs = kwargs

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        assert isinstance(orig_handle, self.__class__)
        assert self.codes is None
        assert self.verts is None
        kwargs = dict(
            clip_on = False,
            )
        kwargs.update(self.kwargs)
        kwargs.update(orig_handle.kwargs)
        xmode = kwargs.pop('xmode', 'center')
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        verts = orig_handle.verts.copy()
        if xmode == 'fill':
            verts[:,0] = x0 + verts[:,0] * width
        else:
            verts[:,0] += x0 + 0.5 * width
        verts[:,1] = y0 + verts[:,1] * height
        codes = orig_handle.codes
        path = Path(verts, codes = codes)
        patch = PathPatch(path, **kwargs)
        handlebox.add_artist(patch)
        return patch


class BasePlotKep(DataPlot, Logged):
    # TODO: allow to overwrite from resource file
    kepler_colors = [
        "#000000",  #  0 black
        "#ffffff",  #  1 white
        "#7f7f7f",  #  2 grey
        "#7f0000",  #  3 dark red
        "#ff0000",  #  4 red
        "#ff007f",  #  5 orange red
        "#7f007f",  #  6 dark purple
        "#ff00ff",  #  7 magenta
        "#ff7fff",  #  8 light magenta
        "#ff7f7f",  #  9 peach
        "#ff7f00",  # 10 coral
        "#ffff00",  # 11 yellow
        "#ffff7f",  # 12 light yellow
        "#7f7f00",  # 13 olive yellow
        "#7fff00",  # 14 medium spring green
        "#007f00",  # 15 dark green
        "#00ff00",  # 16 green
        "#7fff7f",  # 17 grey green
        "#00ff7f",  # 18 spring green
        "#007f7f",  # 19 dark cyan
        "#00ffff",  # 20 cyan
        "#7fffff",  # 21 bright blue violet
        "#007fff",  # 22 slate blue
        "#00007f",  # 23 dark blue
        "#0000ff",  # 24 blue
        "#7f7fff",  # 25 blue violet
        "#7f00ff",  # 26 medium slate blue
        # non-traditionla values
        "#b1b100",  # 27 lighter olive yellow
        "#00b100",  # 28 medium dark green
        ]

    colors = dict()

    _default_label_size = 9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k,c in self.colors.items():
            if isinstance(c, int):
                self.colors[k] = self.kepler_colors[c]
            if isinstance(c, tuple):
                self.colors[k] =  tuple(self.kepler_colors[x] if isinstance(x, int) else x for x in c)

        yscale = kwargs.pop('yscale', 'log')
        ylabel = kwargs.pop('ylabel', '')

        self.ax.set_yscale(yscale)
        self.ax.set_ylabel(ylabel)

        ylabel2 = kwargs.pop('ylabel2', None)
        yscale2 = kwargs.pop('yscale2', None)

        if ylabel2 is not None:
            self.ax2 = self.ax.twinx()
            self.ax2.set_ylabel(ylabel2)
            self.ax2.set_yscale(yscale2)
        else:
            self.ax2 = None

    def make_default_frame(self, *args, **kwargs):
        from .framekep import FrameKep
        return FrameKep(self, *args, **kwargs)

class BasePlotKepAxes(BasePlotKep, Logged):

    cnvsym = ['.', ',', '!', 'i', '|', ':']
    cnames = ['rad', 'neut', 'osht', 'semi', 'conv', 'thal']

    def __init__(self, *args, **kwargs):
        """
        set up basic KEPLER plot (Plots 1-8)
        """

        super().__init__(*args, **kwargs)

        self.showconv = kwargs.get('showconv', 'solid')

        # self.showzones = kwargs.pop('showzones', 'grid')
        self.showzones = kwargs.pop('showzones', 'axis')

        self.annotate = kwargs.get('annotate', 'line')


        if self.showzones in ('axis', 'axis_label'):
            self.ax3 = self.ax.twiny()
            self.ax3.set_xscale(
                'gridscale',
                refaxis = self.ax,
                )
            if self.showzones == 'axis_label':
                self.ax3.set_xlabel('zone interface number')
        else:
            self.ax3 = None

        self.xvar = XVar(self.data)

        self.update()

    def clear(self):
        """
        clear old stuff before new plot
        """
        super().clear()
        try:
            self.ax3.legend_.remove()
        except:
            pass

    def setx(self):
        """
        add axis scale variants (based on parm 132 or specified)
        """
        self.ax.set_xscale(self.xvar.scale)
        self.ax.set_xlabel(self.xvar.label)

    def adjust_axes(self):
        x = self.xvar.x[self.xvar.iib]
        xlim = x[[0, -1]]
        border = 0.01 * (xlim[1] - xlim[0]) * np.array([-1, 1])
        xlim += border

        self.ax.set_xlim(*xlim)
        if self.ax.get_xscale() == 'linear':
            self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        if self.ax.get_yscale() == 'linear':
            self.ax.yaxis.set_minor_locator(AutoMinorLocator())

    line_annotations = {
        'line_text' : 'annotate_line_text',
        'line_text_fixed' : 'annotate_line_text_fixed',
        'line_fixed' : 'annotate_line_fixed',
        'line_float' : 'annotate_line_float',
        'line' : 'annotate_line',
        }

    def annotation(self):
        """
        Line annotation or legend.
        """
        lines, labels = self.ax.get_legend_handles_labels()
        if self.ax2 is not None:
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
            leg_ax = self.ax2
            if self.ax2.get_yscale() == 'linear':
                self.ax2.yaxis.set_minor_locator(AutoMinorLocator())
        else:
            leg_ax = self.ax

        if self.annotate == 'legend':
            legend = leg_ax.legend(
                lines,
                labels,
                loc = 'best',
                fontsize = 'small',
                frameon = True,
                ncol = 1,
                facecolor = [1,1,1],
                edgecolor = [0,0,0],
                framealpha = 0.2,
                )
            legend.set_draggable(True)
        elif self.annotate in self.line_annotations:
            annotate_method = self.__getattribute__(self.line_annotations[self.annotate])
            for l,t in zip(lines, labels):
                annotate_method(l, t)
        elif self.annotate is None:
            pass
        else:
            raise AttributeError('invalid value for annotate: {}'.format(self.annotate))

    def show_grid(self):
        """
        Show grid.
        """
        x = self.xvar.x[self.xvar.iib]
        if self.showzones == 'grid':
            self.grid = Grid(x, self.ax)
        elif self.showzones in ('axis', 'axis_label'):
            self.ax3.xaxis._scale.update_refvalues(x)

    def decorations(self):
        """
        Add convection, annotation, legend.
        """

        self.convection()
        self.adjust_axes()
        self.annotation()
        self.show_grid()


    def update(self, *args, **kwargs):
        """
        Update plot.

        Parameters
        ----------
        irtype : int
            Variable usef or x axis.

            See kepler manual at
            https://2sn.org/kepler/doc/parameters/function.html#parm-132

        TODO
        ----
            Allow strings for variable names in 'xvar'.
        """
        irtype = kwargs.pop('irtype', None)
        self.xvar.update(irtype = irtype)
        super().update(*args, **kwargs)


    def convection(self, *args, **kwargs):
        """
        Show convection.

        Parameters
        ----------
        showconv : str
            One of ``['char', 'path', 'cband', 'band', 'solid']``.

            Method to indicate convective region.

            Default method is ``'solid'``.

        Style Notes
        -----------
            For zoneal displayes, immermost and outermost zones are drawn
            to edge of the grid.
        """
        showconv = kwargs.get('showconv', self.showconv)
        if showconv is None:
            return

        x = self.xvar.x
        xm = self.xvar.xm
        jm = self.xvar.jm
        iib = self.xvar.iib

        iconv = self.data.iconv
        legtype = 'conv'
        # TODO - using text is very slow and needs to be replaced by a path-patch
        if showconv == 'char':
            for m,i in zip(x[iib], iconv[iib]):
                self.ax.text(m, 1., self.cnvsym[i],
                             horizontalalignment = 'center',
                             verticalalignment = 'center',
                             transform = self.ax.transData,
                             clip_on = True,
                             )
            conv = set(iconv[iib])
            convpatches = []
            convlabels = []
            for i,p in enumerate(self.cnvsym):
                if i in conv:
                    convpatches += [TextLegendHandler(p)]
                    convlabels += [self.cnames[i]]
            if not self.check_legend(legtype):
                conv_legend = self.fig.legend(
                    convpatches,
                    convlabels,
                    loc = 'lower right',
                    framealpha = 1.,
                    fontsize = 'small',
                    handler_map={TextLegendHandler: TextLegendHandler()},
                    )
                conv_legend.set_draggable(True)
                self.add_legend(legtype, conv_legend)
        elif showconv == 'text':
            for m,i in zip(x[iib], iconv[iib]):
                path = self.get_text_path(self.cnvsym[i])
                trans = TextDataTransform(self.ax, m, 1.)
                patch = PathPatch(
                    path,
                    transform = trans,
                    clip_on = True,
                    fc = rcParams['text.color'],
                    ec = 'none',
                    zorder = 10)
                self.ax.add_patch(patch)
            conv = set(iconv[iib])
            convpatches = []
            convlabels = []
            for i,p in enumerate(self.cnvsym):
                if i in conv:
                    path = self.get_text_path(self.cnvsym[i])
                    convpatches += [PathLegendHandler(
                        path,
                        fc = rcParams['text.color'],
                        ec = 'none',
                        )]
                    convlabels += [self.cnames[i]]
            if not self.check_legend(legtype):
                conv_legend = self.fig.legend(
                    convpatches,
                    convlabels,
                    loc = 'lower right',
                    framealpha = 1.,
                    fontsize = 'small',
                    handler_map={PathLegendHandler: PathLegendHandler()},
                    )
                conv_legend.set_draggable(True)
                self.add_legend(legtype, conv_legend)
        elif showconv == 'path':
            Pm = Path.MOVETO
            Pl = Path.LINETO
            seg = np.array([Pm,Pl])
            nmax = (jm + 1) * 4
            paths = np.ndarray((nmax, 2))
            npattern = [1,1,1,2,1,2]
            patterns = np.array([
                [0.0, 0.1, 0.0, 0.0],
                [0.0, 0.3, 0.0, 0.0],
                [0.7, 1.0, 0.0, 0.0],
                [0.0, 0.3, 0.4, 0.5],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.2, 0.4, 0.5],
                ])
            height = 10
            yloc = 1.
            org = self.ax.transData.transform([x[0], yloc])
            target = self.ax.transAxes.inverted().transform([org, org + np.array([0, height])])
            x0 = target[0,1]
            dx = target[1,1] - target[0,1]
            trans = self.ax.get_xaxis_transform()
            n = 0
            for m,i in zip(x[1:], iconv[:jm]):
                ni = npattern[i]
                paths[n:n+2*ni, 1] = patterns[i,:2*ni]
                paths[n:n+2*ni, 0] = m
                n += 2 * ni
            paths = paths[:n,:]
            codes = np.tile(seg, n // len(seg))
            paths[:,1] = x0 + paths[:,1]*dx
            path = Path(paths,
                        codes = codes)
            patch = PathPatch(path,
                              linewidth = 1,
                              color = rcParams['text.color'],
                              transform = trans,
                              zorder = 10)
            self.ax.add_patch(patch)
            conv = set(iconv[iib])
            convpatches = []
            convlabels = []
            for i,p in enumerate(self.cnvsym):
                if i in conv:
                    n = npattern[i]
                    verts = np.zeros((n*2, 2))
                    verts[:,1] = patterns[i, :n*2]
                    codes = np.tile(seg, n)
                    convpatches += [
                        VertsLegendHandler(
                            verts = verts,
                            codes = codes,
                            color = rcParams['text.color'],
                            linewidth = 1,
                            )
                        ]
                    convlabels += [self.cnames[i]]
            if not self.check_legend(legtype):
                conv_legend = self.fig.legend(
                    convpatches,
                    convlabels,
                    loc = 'lower right',
                    framealpha = 1.,
                    fontsize = 'small',
                    handler_map={VertsLegendHandler: VertsLegendHandler()},
                    )
                conv_legend.set_draggable(True)
                self.add_legend(legtype, conv_legend)
        elif showconv == 'cband':
            alpha = 2/3
            height = 5
            cbox = TransformedBbox(
                Bbox([[0,0],[1,1]]),
                blended_transform_factory(self.ax.transAxes, self.ax.figure.transFigure))
            trans = BandTransform(self.ax, size = height, ypos = 1)
            col=[self.fig.get_facecolor(),
                 (0.75,1.,1.),
                 (1.,0.5,0.),
                 (1.,0.,0.),
                 (0.,1.,0.),
                 (1.,1.,0.),
                 ]
            ii = np.where(iconv[1:jm-1] != iconv[2:jm])[0] + 2
            if len(ii) == 0:
                return
            jconv = iconv[ii-1].tolist()+[iconv[ii[-1]]]
            xcoord = np.ndarray(len(ii)+2)
            xcoord[0] = x[0]
            xcoord[-1] = x[jm]
            xcoord[1:-1] = xm[ii]
            for c0,c1,jc in zip(xcoord[0:], xcoord[1:], jconv):
                xy = (c0, 0)
                rect = Rectangle(xy, c1-c0, 1,
                                 ec = 'none',
                                 fc = col[jc],
                                 fill = True,
                                 alpha = alpha,
                                 transform = trans,
                                 zorder = -10,
                                 )
                self.ax.add_patch(rect)
                rect.set_clip_box(cbox)

            # figure legend
            conv = set(iconv[iib])
            convpatches = []
            convlabels = []
            for i,p in enumerate(col):
                if i in conv:
                    convpatches += [Patch(fc = col[i],
                                          alpha = alpha,
                                          ec = 'none',
                                          )]
                    convlabels += [self.cnames[i]]
            if not self.check_legend(legtype):
                conv_legend = self.fig.legend(
                    convpatches,
                    convlabels,
                    loc = 'lower right',
                    framealpha = 1.,
                    fontsize = 'small',
                    )
                conv_legend.set_draggable(True)
                self.add_legend(legtype, conv_legend)

        elif showconv == 'band':
            Pm = Path.MOVETO
            Pl = Path.LINETO
            Pc = Path.CLOSEPOLY
            seg = np.array([Pm,Pl,Pl,Pl])

            npattern = [1,1,1,2,1,2]
            patterns = np.array([
                [0.0, 0.1, 0.0, 0.0],
                [0.0, 0.3, 0.0, 0.0],
                [0.7, 1.0, 0.0, 0.0],
                [0.0, 0.3, 0.4, 0.5],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.2, 0.4, 0.5],
                ])
            height = 10
            yloc = 1.
            org = self.ax.transData.transform([x[0], yloc])
            target = self.ax.transAxes.inverted().transform([org, org + np.array([0, height])])
            x0 = target[0,1]
            dx = target[1,1] - target[0,1]
            trans = self.ax.get_xaxis_transform()

            ii = np.where(iconv[1:jm-1] != iconv[2:jm])[0] + 2
            if len(ii) == 0:
                return
            jconv = iconv[ii-1].tolist()+[iconv[ii[-1]]]
            xcoord = np.ndarray(len(ii)+2)
            xcoord[0] = x[0]
            xcoord[-1] = x[jm]
            xcoord[1:-1] = xm[ii]

            nmax = len(jconv) * 8
            paths = np.ndarray((nmax, 2))

            n = 0
            for c0,c1,jc in zip(xcoord[0:], xcoord[1:], jconv):
                ni = npattern[jc]
                for i in range(ni):
                    paths[n  :n+2, 1] = patterns[jc,2*i:2*(i+1)]
                    paths[n+2:n+4, 1] = patterns[jc,2*i:2*(i+1)][::-1]
                    paths[n  :n+4, 0] = [c0,c0,c1,c1]
                    n += 4
            paths = paths[:n,:]
            codes = np.tile(seg, n // len(seg))

            paths[:,1] = x0 + paths[:,1]*dx
            path = Path(paths,
                        codes = codes)
            patch = PathPatch(path,
                              linewidth = 1,
                              fc = rcParams['text.color'],
                              ec = 'none',
                              transform = trans,
                              zorder = 10)
            self.ax.add_patch(patch)

            conv = set(iconv[iib])
            convpatches = []
            convlabels = []
            for j,p in enumerate(self.cnvsym):
                if j in conv:
                    nj = npattern[j]
                    verts = np.zeros((nj*4, 2))
                    n = 0
                    c0 = 0
                    c1 = 1
                    for i in range(nj):
                        verts[n  :n+2, 1] = patterns[j, 2*i:2*(i+1)]
                        verts[n+2:n+4, 1] = patterns[j, 2*i:2*(i+1)][::-1]
                        verts[n  :n+4, 0] = [c0,c0,c1,c1]
                        n += 4
                    codes = np.tile(seg, nj)
                    convpatches += [
                        VertsLegendHandler(
                            verts = verts,
                            codes = codes,
                            ec = 'none',
                            fc = rcParams['text.color']
                            )
                        ]
                    convlabels += [self.cnames[j]]
            if not self.check_legend(legtype):
                conv_legend = self.fig.legend(
                    convpatches,
                    convlabels,
                    loc = 'lower right',
                    framealpha = 1.,
                    fontsize = 'small',
                    handler_map={VertsLegendHandler: VertsLegendHandler(xmode = 'fill')},
                    )
                conv_legend.set_draggable(True)
                self.add_legend(legtype, conv_legend)

        elif showconv == 'solid':
            alpha = 1/4
            col=[self.fig.get_facecolor(),
                 (0.75,1.,1.),
                 (1.,0.5,0.),
                 (1.,0.,0.),
                 (0.,1.,0.),
                 (1.,1.,0.),
                 ]
            ii = np.where(iconv[1:jm-1] != iconv[2:jm])[0] + 2
            if len(ii) == 0:
                return
            jconv = iconv[ii-1].tolist()+[iconv[ii[-1]]]
            xcoord = np.ndarray(len(ii)+2)
            xcoord[0] = x[0]
            xcoord[-1] = x[jm]
            xcoord[1:-1] = xm[ii]
            for c0,c1,jc in zip(xcoord[0:], xcoord[1:], jconv):
                self.ax.axvspan(c0,c1,
                                ec = 'none',
                                fc = col[jc],
                                fill = True,
                                alpha = alpha,
                                zorder = -10)

            # figure legend
            conv = set(iconv[iib])
            convpatches = []
            convlabels = []
            for i,p in enumerate(col):
                if i in conv:
                    convpatches += [Patch(facecolor = col[i],
                                          alpha = alpha,
                                          edgecolor = 'none')]
                    convlabels += [self.cnames[i]]
            if not self.check_legend(legtype):
                conv_legend = self.fig.legend(
                    convpatches,
                    convlabels,
                    loc = 'lower right',
                    framealpha = 1.,
                    fontsize = 'small',
                    )
                conv_legend.set_draggable(True)
                self.add_legend(legtype, conv_legend)
        else:
            raise AttributeError('Invalid value for "showconv"')


    def _binned_lable_locations(self, line, nanno = 8, xlim = None):
        xline,yline = line.get_xydata().transpose()
        if xlim is None:
            xlim = np.array([np.min(xline), np.max(xline)])
            xlim = line.axes.get_xlim()
        xlabel = (np.arange(nanno+1)+0.5)[:-1]*(xlim[1] - xlim[0])/nanno + xlim[0]
        ylim = line.axes.get_ylim()
        UNDEF = 2 * ylim[0] - ylim[1]
        ylabel = np.interp(xlabel, xline, yline,left=UNDEF, right=UNDEF)
        return xlabel, ylabel

    def annotate_line_text(self, line, label, nanno = 8, xlim = None, size = None):
        xdata = (np.arange(nanno+1)+0.5)[:-1]/nanno
        if size is None:
            size = self._default_label_size
        xline,yline = line.get_xydata().transpose()
        trans = DataTransform(xline, yline, line.axes)
        for x in xdata:
            line.axes.text(
                x, 1., label,
                horizontalalignment = 'center',
                verticalalignment = 'center',
                size = size,
                clip_on = True,
                transform = trans
                )

    def annotate_line_text_fixed(self, line, label, nanno = 8, xlim = None, size = None):
        xlabel, ylabel = self._binned_lable_locations(line, nanno = nanno, xlim = xlim)
        if size is None:
            size = self._default_label_size
        for x, y in zip(xlabel, ylabel):
            line.axes.text(
                x, y, label,
                horizontalalignment = 'center',
                verticalalignment = 'center',
                size = size,
                clip_on = True,
                )

    def annotate_line_fixed(self, line, label, nanno = 8, size = None, format = None, xlim = None):
        xlabel, ylabel = self._binned_lable_locations(line, nanno = nanno, xlim = xlim)
        label, fsize = self.get_label_marker(label, size = size, format = format)
        line.axes.plot(xlabel, ylabel, ls = 'none',
                       marker = label,
                       markeredgecolor = 'none',
                       markeredgewidth = 0,
                       markerfacecolor = rcParams['text.color'],
                       markersize = fsize,
                       )

    def get_text_path(self, text, size = None, fp = None):
        if fp is None:
            fp = FontProperties()
        if size is not None:
            fp.set_size(size)
        if not hasattr(self, '_text_path_buf'):
            self._text_path_buf = dict()
        try:
            return self._text_path_buf[label, size]
        except:
            pass
        path = TextPath(
            xy=(0,0),
            s = text,
            fontproperties = fp)
        self._text_path_buf[text, size] = path
        return path

    def get_label_path(self, label, size = None, fp = None):
        if fp is None:
            fp = FontProperties()
        if size is not None:
            fp.set_size(size)
        if not hasattr(self, '_label_text_path_buf'):
            self._label_text_path_buf = dict()
        try:
            return self._label_text_path_buf[label, fp]
        except KeyError:
            pass
        text = TextPath(xy=(0,0),
                        s = label,
                        fontproperties = fp)
        fxmin, fymin, fxmax, fymax = text.get_extents().bounds
        fwidth = fxmax - fxmin
        fheight = fymax - fymin
        text = text.transformed(
            Affine2D().translate(
                -fxmin + 0.5 * -fwidth,
                -fymin + 0.5 * -fheight,
                )
            )
        fsize = 2 * max(
            np.max(np.abs(text.vertices[:, 0])),
            np.max(np.abs(text.vertices[:, 1])))
        self._label_text_path_buf[label, fp] = text, fsize
        return text, fsize

    def get_label_marker(self, label, size = None, format = None):
        if size is None:
            size = self._default_label_size
        if format is None:
            format = 'path'
        if format == 'text':
            if not '$' in label:
                label = r'${}$' + label
            fsize = size * sqrt(0.5)
        elif format == 'path':
            label, fsize = self.get_label_path(label, size)
        else:
            raise AttributeError('Invalid format {}'.format(format))
        return label, fsize

    def annotate_line_float(self, line, label, format = None, nanno = 8, size = None, **kwargs):
        xline,yline = line.get_xydata().transpose()
        label, fsize = self.get_label_marker(label, size = size, format = format)
        mark = tuple(((np.array([0.5,1.0])/(nanno+1)).tolist()))
        line.axes.plot(xline, yline, ls = 'none',
                       marker = label,
                       markeredgecolor = 'none',
                       markeredgewidth = 0,
                       markerfacecolor = rcParams['text.color'],
                       markersize = fsize,
                       markevery = mark,
                       )

    def annotate_line(self, line, label, nanno = 8, size = None, **kwargs):
        xline,yline = line.get_xydata().transpose()
        if size is None:
            size = self._default_label_size
        path, fsize = self.get_label_path(label, size = size)
        xdata = (np.arange(nanno+1)+0.5)[:-1]/nanno
        for x in xdata:
            trans = DataPointTransform(xline, yline, line.axes, x)
            patch = PathPatch(
                path,
                edgecolor = 'none',
                facecolor = rcParams['text.color'],
                fill = True,
                clip_on = True,
                transform = trans,
                zorder = 3,
                )
            line.axes.add_patch(patch)
