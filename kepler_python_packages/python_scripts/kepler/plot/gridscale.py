import numpy as np

from matplotlib.transforms import Transform
from matplotlib.ticker import Locator, Formatter, NullFormatter
from matplotlib.scale import register_scale, LinearScale


__all__ = ['gridscale']

class GridScaleTransformer(object):
    # TODO - allow x or y scale

    # allow for general mapping on second scale (will require
    # different/generic ticker and label)

    # do actual work

    _default_y_value = 0

    def __init__(self,
                 **kwargs
                 ):


        rvalues = kwargs.get('refvalues', np.array([0,1]))
        raxis = kwargs.get('refaxis')

        self.raxis = raxis
        self.update_refvalues(rvalues)

    def update_refvalues(self, rvalues):
        ii = np.argsort(rvalues)
        self.values = rvalues[ii]
        self.zones = ii
        self.rzones = np.arange(len(rvalues))
        self.rvalues = rvalues

    def data2axis(self, d):
        s = d.shape
        v = np.interp(d, self.rzones, self.rvalues,
                      left = 2 * self.rvalues[0] - self.rvalues[-1],
                      right = 2 * self.rvalues[-1] - self.rvalues[0],
                      )
        ii = np.where(d < self.rzones[0])
        v[ii] = self.rvalues[0] + (
            (d[ii] - self.rzones[0]) *
            (self.rvalues[-1] - self.rvalues[0]) /
            (self.rzones[-1] - self.rzones[0])
            )
        ii = np.where(d > self.rzones[-1])
        v[ii] = self.rvalues[-1] + (
            (d[ii] - self.rzones[-1]) *
            (self.rvalues[0] - self.rvalues[-1]) /
            (self.rzones[0] - self.rzones[-1])
            )
        a = np.array([v.flatten(), [self._default_y_value] * len(v)]).transpose()
        a = (self.raxis.get_xaxis_transform() + self.raxis.transAxes.inverted()).transform(a)
        f = a[:,0].reshape(s)
        if np.any(np.isnan(f)):
            print('[d2f] ', d, '-->', f)
        return f

    def axis2data(self, f):
        s = f.shape
        a =  np.array([f.flatten(), [self._default_y_value] * len(f)]).transpose()
        a = (self.raxis.transAxes + self.raxis.get_xaxis_transform().inverted()).transform(a)
        v = a[:,0].reshape(s)
        d = np.interp(v, self.values, self.zones,
                      left = 2 * self.zones[0] - self.zones[-1],
                      right = 2 * self.zones[-1] - self.zones[0],
                      )
        ii = np.where(v < self.values[0])
        d[ii] = self.zones[0] + (
            (v[ii] - self.values[0]) *
            (self.zones[-1] - self.zones[0]) /
            (self.values[-1] - self.values[0])
            )
        ii = np.where(v > self.values[-1])
        d[ii] = self.zones[-1] + (
            (v[ii] - self.values[-1]) *
            (self.zones[0] - self.zones[-1]) /
            (self.values[0] - self.values[-1])
            )
        if np.any(np.isnan(d)):
            print('[f2d] ', f, '-->', d)
        return d

class GridScaleTransformBase(Transform):
    input_dims = 1
    output_dims = 1
    has_inverse = True
    is_separable = True
    def __init__(self, transform):
        super().__init__()
        self._transform = transform

class GridScaleTransform(GridScaleTransformBase):
    def transform_non_affine(self, a):
        return self._transform.data2axis(a)
    def inverted(self):
        return InvertedGridScaleTransform(self._transform)

class InvertedGridScaleTransform(GridScaleTransformBase):
    def transform_non_affine(self, a):
        return self._transform.axis2data(a)
    def inverted(self):
        return GridScaleTransform(self._transform)

class GridLocator(Locator):
    divs = np.array([   1,   2,   5,
                        10,  20,  50])
    def __init__(self, transformer, n = 6):
        self._n = n
        self._transformer = transformer

    def tick_values(self, vmin, vmax):
        "Return locations of ticks as function of vmin and vmax."
        vmin, vmax = sorted((vmin, vmax))
        zmin, zmax = sorted(self._transformer.zones[[0, -1]])
        vmin = max(vmin, zmin)
        vmax = min(vmax, zmax)
        span = np.abs(vmax-vmin)
        scale = 10**max(0, np.floor(np.log10(span))-1)
        ntick = span/scale
        m = 1
        for x in self.divs:
            if ntick / x < self._n:
                m = x
                break
        scale = scale * m
        tick0 = np.ceil(vmin / scale) * scale
        tick1 = np.floor(vmax / scale) * scale
        ticks = np.arange(tick0, tick1 + 1, scale)
        return ticks

    def __call__(self):
        "Return the locations of the ticks"
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

class GridMinorLocator(Locator):
    divs = np.array([   1, 2, 5, 10, 20, 50, 100])
    def __init__(self, transformer, n = 11):
        self._n = n
        self._transformer = transformer

    def tick_values(self, vmin, vmax):
        majorlocs = self.axis.get_majorticklocs()
        if len(majorlocs) > 1:
            span = majorlocs[1] - majorlocs[0]
        else:
            span = 1
        vmin, vmax = sorted((vmin, vmax))
        zmin, zmax = sorted(self._transformer.zones[[0, -1]])
        vmin = max(vmin, zmin)
        vmax = min(vmax, zmax)
        scale = 10**max(0, np.floor(np.log10(span))-1)
        ntick = span/scale
        m = 1
        for x in self.divs:
            if ntick / x < self._n:
                m = x
                break
        scale = scale * m
        tick0 = np.ceil(vmin / scale) * scale
        tick1 = np.floor(vmax / scale) * scale
        ticks = np.arange(tick0, tick1 + 1, scale)
        ticks = np.array([x for x in ticks if not x in majorlocs])
        return ticks

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

class GridMajorFormatter(Formatter):
    def __call__(self, x, pos = None):
        majorlocs = self.axis.get_majorticklocs()
        s = '{:d}'.format(int(x))
        if pos is not None and pos > 0:
            a = np.array([majorlocs.flatten(), [1] * len(majorlocs)]).transpose()
            f = self.axis.axes.transData.transform(a)[:,0]
            l = np.ceil(np.log10(np.maximum(majorlocs, 1))) + 1
            p0 = 0
            for p in range(1, pos+1):
                if (f[p] - f[p0]) >= 0.5 * (l[p0] + l[p] + 1) * 8:
                    p0 = p
            if p0 < p:
                s = ''
        return s

class GridScale(LinearScale):
    """
    Grid Scale, extende linaer outside, need references radial scale
    """
    name = 'gridscale'

    def __init__(self, axis, **kwargs):
        """
        need to supply reference scale
        """
        self.axis = axis
        self._kwargs = kwargs.copy()
        self._transformer = GridScaleTransformer(
            **self._kwargs)
        self._min = 0
        self._max = 1
        if 'refvalues' in kwargs:
            self.update_limits()
        self.axis.axes.callbacks.connect('xlim_changed', self._adjust_grid_scale)

    def get_transform(self):
        return GridScaleTransform(self._transformer)

    def set_default_locators_and_formatters(self, axis):
        super().set_default_locators_and_formatters(axis)
        axis.set_minor_locator(GridMinorLocator(self._transformer))
        axis.set_major_locator(GridLocator(self._transformer))
        axis.set_major_formatter(GridMajorFormatter())
        axis.set_minor_formatter(NullFormatter())

    #def limit_range_for_scale(self, vmin, vmax, minpos = None):
    #    return self._transformer.rzones[[0,-1]]

    def _adjust_grid_scale(self, event):
        xmin, xmax = self.axis.axes.get_xlim()
        if xmin != self._min or xmax != self._max:
            self._min = xmin
            self._max = xmax
            self.update_limits()

    def update_limits(self):
        self.axis.axes.set_xlim(self._transformer.axis2data(np.array([0,1])))

    def update_refvalues(self, refvalues):
        self._transformer.update_refvalues(refvalues)
        self.update_limits()

register_scale(GridScale)
