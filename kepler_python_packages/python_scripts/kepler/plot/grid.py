import numpy as np
from matplotlib.transforms import Transform, Affine2D, \
     blended_transform_factory, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Patch
from matplotlib.text import Text

from logged import Logged

class GridTransform(Transform):
    input_dims = 2
    output_dims = 2
    is_separable = False
    has_inverse = False
    def __init__(self, ax, size = 10, mode = 'top'):
        self.ax = ax
        self.size = size
        self.mode = mode
        self._parents = dict()
    def transform_non_affine(self, values):
        unit = self.size * self.ax.figure.get_dpi() / 72
        if self.mode == 'full':
            val = values
            ii = values[:,1] > 0
            val[ii,1] = 1
            data = blended_transform_factory(
                self.ax.transData,
                self.ax.transAxes
                ).transform(values)
        elif self.mode == 'top':
            y0 = self.ax.transAxes.transform([[0,1]])[0,1]
            data = blended_transform_factory(
                self.ax.transData,
                Affine2D().scale(unit).translate(0, y0 + 0.5 * unit)
                ).transform(values)
        else:
            raise AttributeError(' [Grid] Unknown mode {}'.format(self.mode))
        return data

class Grid(Logged):
    def clear(self):
        """
        clear model patches and texts
        """
        if '_elements' not in self.__dict__:
            return
        ax = self.ax
        for item in self._elements:
            if isinstance(item, Patch):
                ax.patches.remove(item)
            elif isinstance(item, Text):
                ax.artists.remove(item)
            else:
                raise Exception('Unknown Grid Element')
        del self._elements

    def show(self, state = None):
        if state is None:
            show = not self.show()
        else:
            show = state
        for item in self._elements:
            item.set_visible(show)

    def __init__(self,
                 xdata, ax, size = 10,
                 silent = False):
        self.setup_logger(silent)
        self.logger.info('Plotting grid')

        self._elements = list()
        self.ax = ax

        colors = ['white','red','green','blue']

        ndata = len(xdata)
        maxlev = int(np.log10(ndata)*2)+1
        nlines = np.zeros(maxlev, dtype = np.int)
        xval = np.ndarray((maxlev, ndata))

        Pm = Path.MOVETO
        Pl = Path.LINETO
        seg = np.array([Pm,Pl])

        trans = GridTransform(ax, size = 10)

        cbox = TransformedBbox(
            Bbox([[0,0],[1,1]]),
            blended_transform_factory(ax.transAxes,ax.figure.transFigure))

        for i, x in enumerate(xdata):
            mag = 0
            if i == 0:
                mag = maxlev//2-1
            else:
                while (i % 10) == 0:
                    mag += 1
                    i //= 10
            half = int((i % 5) == 0)
            lev = 2 * mag + half
            xval[lev, nlines[lev]] = x
            nlines[lev] += 1
        codes = np.ndarray((maxlev,ndata*2))
        paths = np.ndarray((maxlev,ndata*2,2))
        for lev in range(maxlev):
            if nlines[lev] > 0:
                codes = np.tile(seg,nlines[lev])
                paths = np.ndarray((nlines[lev]*2,2))
                paths[0::2,0] = xval[lev,0:nlines[lev]]
                paths[1::2,0] = xval[lev,0:nlines[lev]]
                paths[0::2,1] = 0
                mag  = (lev // 2)
                half = (lev % 2)
                paths[1::2,1] = (2 + 1.5*mag + half) * 0.15
                path = Path(paths,
                            codes = codes)
                patch = PathPatch(path,
                                  linewidth = 0.1 * (1 + mag + half),
                                  color = colors[mag % len(colors)],
                                  transform = trans)
                ax.add_patch(patch)
                patch.set_clip_box(cbox)
                self._elements.append(patch)

        self.close_logger(timing = 'Plotting grid finished in')
