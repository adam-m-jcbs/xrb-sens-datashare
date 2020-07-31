import numpy as np
import types
from functools import partial

from collections import Iterable, Callable

from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from matplotlib.colors import rgb2hex, colorConverter

from scipy.special import erf

# ALEX PRIVATE PACKAGES
from ._utils import Slice, iterable, is_iterable

# package imports

from .utils import color, rgb, is_string_like
from .models import color_model, _color_models
from .ofilter import get_output_filter, set_output_filter

#######################################################################
#######################################################################
# Define Color[Function] as a replacement for Colormap

_colors = dict()
def register_color(name, color, *args, **kwargs):
    assert isinstance(name, str)
    assert name not in _colors
    if isinstance(color, Color):
        assert len(args) == len(kwargs) == 0
    else:
        assert issubclass(color, Color)
        if len(kwargs) == 0 and len(args) == 1 and isinstance(args[0], dict):
            kwargs = args[0]
            args = ()
        color = (color, args, kwargs)
    _colors[name] = color

def get_cfunc(name):
    """
    return color_function object of given name
    """
    return _colors.get(name, None)

def colormap(name, *args, **kwargs):
    if isinstance(name, Color):
        return name
    if isinstance(name, type) and issubclass(name, Color):
        return name(*args, **kwargs)
    c = get_cfunc(name)
    if isinstance(c, Color):
        return c
    if isinstance(c, type) and issubclass(c, Color):
        return c(*args, **kwargs)
    if isinstance(c, tuple):
        assert len(c) == 3
        _args = list(c[1])
        _args[0:] = list(args)
        _kwargs = dict(c[2])
        _kwargs.update(kwargs)
        return c[0](*_args, **_kwargs)
    try:
        return ColorMap.from_Colormap(name, *args, **kwargs)
    except:
        pass
    return ColorSolid(name, *args, **kwargs)

class Color(Colormap):
    """
    Base class to provide colormap functionallity but on 'color
    function' basis.

    Users may overwrite the __init__ routing to set up coloring functions.
    This should usually call the base method provided here.

    The main coloring is done in the function _function.  The values,
    normalized to 0..1, will be passed as a 1D array for simplicity.
    If the class attribute _alpha is False (default) it is assumed
    _function will return a numpy array of RGB values (:,3) and an
    array of RGBA values (:.4) otherwise, also normalized to the range
    0...1.

    Much of the remaining functionallity should be very similar to
    colormaps.
    """
    bytes = False # default for bytes
    _alpha = False # tell whether function computes alpha
    _missing = None
    _over = None
    _under = None
    _bad = None
    _filter = None
    _xfilter = None
    normalize = False # None = normalize by default if any values outside range

    def __init__(self, *args, **kwargs):
        self.bytes = kwargs.get('bytes', False)
        self.name  = kwargs.get('name', self.__class__.__name__)
        self.alpha = kwargs.get('alpha', 1.)

        if self._missing is not None:
            missing = rgb(self._missing, alpha = self.alpha)
            self._rgba_bad = missing
            self._rgba_under = missing
            self._rgba_over = missing
        else:
            self._rgba_bad = np.array([0.0, 0.0, 0.0, 0.0])  # If bad, don't paint anything.
            self._rgba_under = None
            self._rgba_over = None
        if self._bad is not None:
            self._rgba_bad = rgb(self._bad)
        if self._over is not None:
            self._rgba_over = rgb(self._over)
        if self._under is not None:
            self._rgba_under = rgb(self._under)

    def __call__(self, data, *args, **kwargs):
        """
        Process data and return RGBA value.
        """
        alpha     = kwargs.setdefault('alpha', None)
        normalize = kwargs.setdefault('normalize', self.normalize)
        bytes     = kwargs.setdefault('bytes', self.bytes)
        return_data = kwargs.setdefault('return_data', False)
        data, shape = self._input_to_data(data, normalize)
        if alpha is not None:
            alpha = np.clip(alpha, 0, 1)
        mask_bad   = np.logical_not(np.isfinite(data))
        mask_under = ~mask_bad
        mask_under[~mask_bad] = data[~mask_bad] < 0
        mask_over = ~mask_bad
        mask_over[~mask_bad]  = data[~mask_bad] > 1
        mask = np.logical_not(
            np.logical_or(
                np.logical_or(
                    mask_bad,
                    mask_under),
                mask_over))
        out = self._get_out(data, mask, *args, **kwargs)

        # SIMPLE TREATMENT FOR INVALID/LOW/HIGH DATA
        if self._rgba_under is None:
            self._rgba_under = self._default_color(0., *args, **kwargs)
        if self._rgba_over is None:
            self._rgba_over = self._default_color(1., *args, **kwargs)
        out[mask_under,:] = self._rgba_under
        out[mask_over ,:] = self._rgba_over
        if alpha is not None:
            out[:,-1] = alpha
        out[mask_bad  ,:] = self._rgba_bad

        retval = self._return(out, shape, data, **kwargs)
        if return_data:
            retval = (retval, data)
        return retval

    def _default_color(self, x, alpha, *arg, **kwargs):
        return self._get_out(np.array([x]), alpha, *arg, **kwargs)[0]

    def _get_out(self, data, mask = None, *args, **kwargs):
        out = np.ndarray((data.size, 4))
        if mask is None:
            mask = np.tile(True, data.size)
        if self._alpha:
            out[mask,:] = self._function(data[mask], *args, **kwargs)
        else:
            out[mask,:3] = self._function(data[mask], *args, **kwargs)
            out[:   , 3] = self.alpha
        return out

    def _inherit_color(self, color):
        """
        copy default values and save color

        How will these be filtered?
        """
        self._color = colormap(color)
        assert isinstance(color, Color)
        self.bytes = self._color.bytes
        self._missing = self._color._missing
        self._under = self._color._under
        self._over = self._color._over
        self._bad = self._color._bad

    def _function(self, data, *args, **kwargs):
        """
        prototype conversion

        # if self._alpha:
        #     out = np.tile(data, 4).reshape(-1,4)
        # else:
        #     out = np.tile(data, 3).reshape(-1,3)
        """
        raise NotImplementedError()

    def _update_alpha(self, alpha):
        """
        compute output array
        """

        if not self._alpha:
            if alpha is None:
                alpha = 1
            else:
                out[:, 3] = alpha
        return alpha
    @staticmethod
    def _input_to_data(data, normalize):
        """
        Normalize and get shape
        """
        # May need to allow other formats as well.
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if normalize in (None, True):
            M = data.max()
            m = data.min()
            if normalize is None:
                normalize = m < 0 or M > 1
        if normalize:
            d = M - m
            if d == 0:
                d = 1
            data = (data - m) / d
        shape = data.shape
        data  = data.reshape(-1)
        return data, shape

    def set_filter(self, flt):
        """
        set new filter function and return old one
        """
        self._filter, f = flt, self._filter
        return f

    def add_filter(self, flt, append = True):
        """
        add new filter function (chain) and return previous one
        """
        f = self._filter
        if f is None:
            self._filter = flt
        elif append is True:
            self._filter = lambda rgba: flt(f(rgba))
        else:
            self._filter = lambda rgba: f(flt(rgba))
        return f

    def get_filter(self):
        """
        return current _filter function
        """
        return self._filter

    def set_xfilter(self, flt):
        """
        set new xfilter function and return old one
        """
        self._xfilter, f = flt, self._filter
        return f

    def add_xfilter(self, flt, append = True):
        """
        add new xfilter function (chain) and return previous _xfilter
        """
        f = self._xfilter
        if f is None:
            self._xfilter = flt
        elif append is True:
            self._xfilter = lambda rgba, x: flt(f(rgba, x), x)
        else:
            self._filter = lambda rgba: f(flt(rgba, x), x)
        return f

    def get_xfilter(self):
        """
        return current _xfilter function
        """
        return self._xfilter

    def _return(self, out, shape, data,
                bytes = None,
                ignorefilter = False,
                ignorexfilter = False,
                **kwargs
                ):
        """
        output conversion and filtering
        """
        # this part won't work if there was a sequence with alterating
        # _filter(s) and _xfilter(s)
        if self._xfilter is not None and not ignorexfilter:
            out = self._xfilter(out, data.flatten())
        if self._filter is not None and not ignorefilter:
            out = self._filter(out)

        # filter output for color correction
        filter_output= kwargs.get('filter_output', True)
        if filter_output:
            output_filter = kwargs.get('output_filter', get_output_filter())
            if output_filter is not None:
                out = output_filter._filter(out)

        out = out.reshape(shape + (4,))
        if bytes is None:
            bytes = self.bytes
        if bytes:
            out = np.array(np.minimum(out*256, 255), dtype = np.ubyte)
        return out

    def Colormap(self, N = 256, name = None):
        """
        Return matplotlib.colors.Colormap object with N levels.
        """
        x = np.linspace(0, 1, N)
        if name is None:
            name = self.name
        cm = ListedColormap(
            self.__call__(x),
            name = name,
            N = N)
        cm.set_bad(self._rgba_bad)
        cm.set_under(self._rgba_under)
        cm.set_over(self._rgba_over)
        return cm

    def set_bad(self, color='k', alpha=None):
        '''
        Set color to be used for masked values.
        '''
        if alpha is None:
            if not isinstance(color, str) and len(color) == 4:
                alpha = color[3]
            else:
                alpha = self.alpha
        self._rgba_bad = colorConverter.to_rgba(color, alpha)


    def set_under(self, color='k', alpha=None):
        '''
        Set color to be used for low out-of-range values.
        Requires norm.clip = False
        '''
        if alpha is None:
            if not isinstance(color, str) and len(color) == 4:
                alpha = color[3]
            else:
                alpha = self.alpha
        self._rgba_under = colorConverter.to_rgba(color, alpha)

    def set_over(self, color='k', alpha=None):
        '''
        Set color to be used for high out-of-range values.
        Requires norm.clip = False
        '''
        if alpha is None:
            if not isinstance(color, str) and len(color) == 4:
                alpha = color[3]
            else:
                alpha = self.alpha
        self._rgba_over = colorConverter.to_rgba(color, alpha)

    def _set_extremes(self):
        pass

    def _init(self):
        raise NotImplementedError("Color Function")

    def is_gray(self):
        """
        Return whether color is gray.
        """
        # Subclasses may overwrite this.
        return False

    _N = 1024
    def get_N(self):
        return self._N
    N = property(get_N)

class ColorMap(Color):
    _alpha = True
    def __init__(self,
                 cmap = None,
                 layout = None,
                 model = None,
                 color = None,
                 models = None,
                 normalize = None,
                 gamma = 1.,
                 gamma_func = None,
                 **kwargs):
        """
        Set up ColorMap.

        This is based on Color[functions].  The power is that the
        methed alloes arbitraty smooth/fine interpolation.

        model: color model to use
        models: list of models for numeric values
        color: use this color if not given in layout
        alpha: alpha value
        layout: [X][C|CCC][A][G|GG|GGG|GGGG][M][N]
             X: coordinate, non-zero values normalized to [0,1]
             XX: coordinate for color and alpha
             XXX: coordinate for each colors but not alpha
             XXXX: coordinate for each color and alpha
             C: grayscale
             CCC: three color values
             A: alpha
             G: gamma, same for all
             GG: gamma for color and alpha
             GGG: gamma for three colors but not alpha
             GGGG: gamma for each color and alpha
             M: model
             N: normalize (see below)
        cmap: use <0 for invalid data?
        bytes: - set default
        normalize: normalize valuse to valid range
             None | -1: auto-determine
             True | +1: yes
             False|  0: no
           Normalization is based on [0,1] range is given, translate
           to valid range for parameters.

        NOTES:

        X coordinates < 0:
          [bad [, lower, upper]]

        The interval generally is scaled lineraly so that X[-1]
        becomes 1.

        In each interval each component is interpolated from the
        beginning to end value according to the a function normalized to
        0...1 for the interval.  The scaling itself is detemined by
        the 'gamma' parameter at the end of the interval, so the first
        values is ignored, and so are the gamma for negiative indices
        (see above).  Negative values of gamma scale (1 - x) with
        **(-gamma).

        gamma can be a scalar, then it is interpreted as a power for
        interpolation in an interval.

        """
        alpha = kwargs.get('alpha', None)
        super().__init__(**kwargs)
        self._gamma = kwargs.get('gamma', 1.)

        assert layout is not None
        layout = layout.upper()

        ncoord = layout.count('X')
        assert ncoord in (0,1,2,3,4), "can be 0 - 4 coordinates"
        ipos = layout.find('X')
        if ncoord == 0 :
            if cmap.ndim == 1:
                n = cmap.size
            else:
                n = cmap.shape[0]
            coord = np.arange(n).reshape((n, 1))
            ncoord = 1
        else:
            n = cmap.shape[0]
            if ncoord in (2,3):
                xncoord = 4
            else:
                xncoord = ncoord
            coord = np.ndarray((n, xncoord))
            for i in range(ncoord):
                coord[:,i] = cmap[:,ipos]
                ipos = layout.find('X', ipos + 1)
            if ncoord == 2:
                coord[:,3] = coord[:,1]
                coord[:,1] = coord[:,0]
                coord[:,2] = coord[:,0]
            if ncoord == 3:
                coord[:,3] = np.arange(n).reshape((n, 1))
            ncoord = xncoord
        if coord.dtype is not np.float64:
            coord = np.array(coord, dtype = np.float64)
        for j in range(ncoord):
            ii = coord[:,j] >= 0
            i, = np.where(ii)
            coord[ii,j] -= coord[i[0], j]
            coord[ii,j] /= coord[ii, j].max()

        assert layout.count('A') < 2, "can have only one alpha value"
        ipos = layout.find('A')
        if ipos >= 0:
            alpha = cmap[:,ipos]
        else:
            if alpha is None:
                alpha = 1.
            alpha = np.tile(alpha, n)

        assert layout.count('N') < 2, "can have only one normalization value"
        ipos = layout.find('N')
        if ipos >= 0:
            normal = cmap[:,ipos]
        else:
            if normalize == -1:
                normalize = None
            normal = np.tile(normalize, n)
        norm = np.empty_like(normal, dtype = np.object)
        for i,x in enumerate(normal):
            if x == 1:
                x == True
            elif x == 0:
                x = False
            else:
                x = None
            norm[i] = x
        normal = norm

        assert layout.count('M') < 2, "can have only one model value"
        ipos = layout.find('M')
        if ipos >= 0:
            model = cmap[:,ipos]
        else:
            if model is None:
                model = 0
            model = np.tile(model, n)

        # models is converted to array of ColorModel objects
        if models is None:
            models = ['RGB', 'HSV', 'HSL', 'HSI'] # add more; add doc; add querry routine
        models = np.array(models).reshape(-1)
        m = np.empty_like(model, dtype = np.object)
        for i,x in enumerate(model):
            if not isinstance(x, str):
                x = models[x]
            m[i] = _color_models[x.upper()]
        model = m

        nc = layout.count('C')
        assert nc in (1, 3), "Color has to be C or CCC"
        if nc == 0:
            if color is None:
                color = 0.
            if len(color) == 1:
                color = np.array([mx.gray(color) for mx in model])
            elif len(color) == 3:
                color = np.tile(color, (3, 1))
            else:
                raise AttributeError("Wrong format in 'color' keyword.")
        if nc == 1:
            ipos = layout.find('C')
            c = cmap[:,ipos]
            color = np.array([mx.gray(cx) for (mx, cx) in zip(model, c)])
        else:
            color = np.ndarray((n,3))
            ipos = -1
            for i in range(3):
                ipos = layout.find('C', ipos + 1)
                color[:, i] = cmap[:, ipos]
        # normalize
        # 1) auto-detect
        d = dict()
        for i in range(n):
            if normal[i] is None:
                m = model[i]
                c = color[i]
                try:
                    limits = d[m]
                    limits[:, 0] = np.minimum(limit[:, 0], c)
                    limits[:, 1] = np.maximum(limit[:, 1], c)
                except:
                    limits = np.tile(c, (2, 1)).transpose()
                d[m] = limits
        for m, l in d.items():
            d[m] = not m.is_normal(l)
        # 2) do normalization
        for i in range(n):
            m = model[i]
            if normal[i] is None:
                normal[i] = d[m]
            if normal[i]:
                color[i, :] = m.normalize(color[i, :])

        # combine color and alpha
        color = np.hstack((color, alpha.reshape(-1, 1)))

        # ADD/TREAT 'invalid' colors [bad [, lower, upper]]
        ii = coord[:, 0] < 0
        im,= np.where(ii)
        assert im.size in (0, 1, 3), "Only [bad [, lower, upper]] allowed for 'invalid' colors"
        if im.size > 0:
            i = im[0]
            self._rgba_bad = np.hstack((model[i](color[i, 0:3]),color[i, 3]))
            if im.size > 1:
                i = im[1]
                self._rgba_lower = np.hstack((model[i](color[i, 0:3]),color[i, 3]))
                i = im[2]
                self._rgba_upper = np.hstack((model[i](color[i, 0:3]),color[i, 3]))
            jj = np.logical_not(ii)
            cmap  = cmap [:, jj]
            color = color[:, jj]
            model = model[:, jj]
            coord = coord[:, jj]

        # convert to N x 4 array for gamma
        ng = layout.count('G')
        assert nc in (1, 3), "Gamma has to be G, GG, GGG, or GGGG"
        if ng == 0:
            gamma = np.tile(1., (n, 4))
        if ng == 1:
            ipos = layout.find('G')
            g = cmap[:, ipos]
            gamma = np.tile(g, (4, 1)).transpose()
        if ng == 2:
            ipos = layout.find('G')
            g = cmap[:, ipos]
            gamma = np.tile(g,(4, 1)).transpose()
            ipos = layout.find('G', ipos+1)
            g = cmap[:, ipos]
            gamma[:, 3] = g
        if ng == 3:
            gamma = np.ndarray((n,4), dtype = cmap.dtype)
            ipos = -1
            for i in range(3):
                ipos = layout.find('G', ipos + 1)
                gamma[:, i] = cmap[:, ipos]
            gamma[:, 3] = np.tile(1., n)
        if ng == 4:
            gamma = np.ndarray((n, 4), dtype = cmap.dtype)
            ipos = -1
            for i in range(4):
                ipos = layout.find('G', ipos + 1)
                gamma[:, i] = cmap[:, ipos]

        # translate to functions
        if gamma_func is None:
            gamma_func = lambda x, gamma: (
                np.power(x, gamma) if gamma >= 0 else
                1 - np.power(1 - x, -gamma))
        assert isinstance(gamma_func, types.FunctionType), (
            "gamma_func needs to be a function")
        if gamma.dtype == np.object:
            g = gamma
        else:
            g = np.empty_like(gamma, dtype = object)
        identity = lambda x: x
        # TODO: paralellize
        gamma_hash = {None: identity}
        for i,f in enumerate(gamma.flat):
            if not isinstance(f, types.FunctionType):
                g.flat[i] = gamma_hash.setdefault(f, partial(gamma_func, gamma = f))
        gamma = g

        # save setup
        self.n = n
        self.gamma = gamma
        self.color = color
        self.model = model
        self.coord = coord
        self.ncoord = ncoord

    def _function(self, data, *args, **kwargs):
        out = np.ndarray((data.size, 4))
        coord = self.coord ** (1 / self._gamma) # gamma from LinearSegmentedColormap
        assert self.ncoord in (1, 4), "require consisient set of gamma"
        if self.ncoord == 1:
            # use np.piecewise instead?
            color0 = self.color[0, :]
            coord0 = coord[0, 0]
            for i in range(1, self.n):
                if self.model[i-1] != self.model[i]:
                    color0[0:3] = self.model[i].inverse()(self.model[i-1](color0[0:3]))
                color1 = self.color[i, :]
                coord1 = coord[i,0]
                if coord0 < coord1: # allow discontinuous colormaps
                    ind = np.logical_and(data >= coord[i-1], data <= coord[i])
                    if np.count_nonzero(ind):
                        dcolor = color1 - color0
                        dcoord = coord1 - coord0
                        colcoord = (data[ind] - coord0) / dcoord
                        for j in range(4):
                            out[ind,j] = color0[j] + self.gamma[i,j](colcoord)*dcolor[j]
                        if self.model[i] != _color_models['RGB']:
                            out[ind, 0:3] = self.model[i](out[ind, 0:3])
                color0 = color1
                coord0 = coord1
        else:
            assert np.all(self.model[0] == self.model[:]),'All color models need to be equal if using independent coordinates'
            for j in range(4):
                coord0 = coord[0, j]
                color0 = self.color[0, j]
                for i in range(1, self.n):
                    color1 = self.color[i, j]
                    coord1 = coord[i, j]
                    if coord0 < coord1: # allow discontinuous colormaps
                        ind = np.logical_and(data >= coord0, data <= coord1)
                        if np.count_nonzero(ind):
                            dcolor = color1 - color0
                            dcoord = coord1 - coord0
                            colcoord = (data[ind] - coord0) / dcoord
                            out[ind, j] = color0 + self.gamma[i, j](colcoord)*dcolor
                    color0 = color1
                    coord0 = coord1
            if self.model[0] != _color_models['RGB']:
                 # transform only valid data
                 ind = np.logical_and(data >= 0, data <= 1)
                 out[ind, 0:3] = self.model[i](out[ind, 0:3])
        return out

    @staticmethod
    def from_Colormap_spec(colors, **kwargs):
        if not isinstance(colors, dict):
            return None
        if not ('red' in colors):
            assert isinstance(colors, Iterable)
            if (isinstance(colors[0], Iterable) and
                len(colors[0]) == 2 and
                not is_string_like(colors[0])):
                # List of value, color pairs
                vals, colors = zip(*colors) # 2to3 - list(zip(*colors))
            else:
                vals = np.linspace(0., 1., len(colors))
            cmap = np.ndarray([[val] + list(colorConverter.to_rgba(color)) for val, color in zip(vals, colors)])
            return ColorMap(cmap, layout = 'XCCCA', **kwargs)
        if isinstance(colors['red'], Callable):
            cmap = np.array([np.zeros(9), np.ones(9)], dtype = object)
            cmap[1,5] = lambda x: np.clip(colors['red'](x),0,1)
            cmap[1,6] = lambda x: np.clip(colors['green'](x),0,1)
            cmap[1,7] = lambda x: np.clip(colors['blue'](x),0,1)
            if 'alpha' in colors:
                cmap[1,8] = lambda x: np.clip(colors['alpha'](x),0,1)
            else:
                cmap[0,4] = 1.
            return ColorMap(cmap, layout = 'XCCCAGGGG', **kwargs)
        xmap = []
        for c in ('red', 'green', 'blue', 'alpha'):
            color = colors.get(c, None)
            if color is None:
                if c == 'alpha':
                    color = ((0,0,1.), (1,1,1.))
                else:
                    color = ((0,0,0.), (1,1,0.))
            color = np.array(color)
            shape = color.shape
            assert len(shape) == 2
            assert shape[1] == 3
            # copied from matplotlib.color.py
            x  = color[:, 0]
            y0 = color[:, 1]
            y1 = color[:, 2]
            if x[0] != 0 or x[-1] != 1:
                raise ValueError("Data mapping points must start with x = 0. and end with x=1")
            if np.sometrue(np.sort(x) - x):
                raise ValueError("Data mapping points must have x in increasing order")
            # end copy
            xc = [[x[0], y1[0]]]
            for i in range(1,shape[0]-1):
                xc += [[x[i], y0[i]]]
                if y0[i] != y1[i]:
                    xc += [[x[i], y1[i]]]
            i = shape[0]-1
            xc += [[x[i], y0[i]]]
            xmap += [np.array(xc)]
        nn = np.array([len(xc) for xc in xmap])
        n = np.max(nn)
        cmap = np.ones((n,8))
        for i,xc in enumerate(xmap):
            cmap[0:nn[i],i::4] = xc
        if np.all(cmap[:,0:4] == cmap[:,0][:,np.newaxis]):
            cmap = cmap[:,3:]
            layout = 'XCCCA'
        else:
            layout = 'XXXXCCCA'
        return ColorMap(cmap, layout = layout, **kwargs)

    @staticmethod
    def from_Colormap(cmap, name = None, gamma = None, **kwargs):
        if isinstance(cmap, ColorMap):
            return cmap
        if isinstance(cmap, str):
            name = cmap
            cmap = get_cmap(name)
        if isinstance(cmap, (LinearSegmentedColormap, ListedColormap)):
            rgba_bad = cmap._rgba_bad
            rgba_under = cmap._rgba_under
            rgba_over = cmap._rgba_over
            if name is None:
                name = cmap.name
            if isinstance(cmap, LinearSegmentedColormap):
                if gamma is None:
                    gamma = cmap._gamma
                segmentdata = cmap._segmentdata
                f = ColorMap.from_Colormap_spec(
                    segmentdata,
                    name = name,
                    gamma = gamma,
                    **kwargs)
            else:
                colors = np.array(cmap.colors)
                f = ColorMap(
                    cmap = colors,
                    layout = 'CCC',
                    name = name,
                    gamma = gamma,
                    **kwargs)
            if rgba_under is not None:
                f.set_under(rgba_under)
            if rgba_over is not None:
                f.set_over(rgba_over)
            if rgba_bad is not None:
                f.set_bad(rgba_bad)
        else:
            if gamma is None:
                gamma = 1.0
            f = ColorMap.from_Colormap_spec(
                cmap,
                name = name,
                gamma = gamma,
                **kwargs)
        return f

#######################################################################
# Some specific color maps & examples

class ColorMapGal(ColorMap):
    maps = {
        0: np.array(
            [[0,0,0,1,0],
             [5,0,0,0,2],
             [7,1,0,0,0.5]]),
        1: np.array(
            [[0,1,1,1,0],
             [2,0,1,0,2],
             [3,0,0.75,0.75,1],
             [4,0,0,1,1],
             [5,0,0,0,1],
             [6,1,0,0,1],
             [7,1,1,0,0.75]]),
        2: np.array(
            [[0,1,1,1,0],
             [2,0,1,0,2],
             [3,0,0.75,0.75,1],
             [4,0,0,1,1],
             [6,1,0,0,1],
             [7,1,1,0,0.75]]),
        3: np.array(
            [[0,1,1,1,0],
             [2,0,1,0,2],
             [3,0,1,1,1],
             [4,0,0,1,1],
             [5,1,0,1,1],
             [6,1,0,0,1],
             [6.25,1,.75,0,2]]),
        4: np.array(
            [[0,1,1,1,0],
             [1,.75,.75,.75,2],
             [2,0,1,0,2],
             [3,0,1,1,1],
             [4,0,0,1,1],
             [5,1,0,1,1],
             [6,1,0,0,1],
             [6.25,1,.75,0,2]]),
        5: np.array(
            [[0,1,1,1,0],
             [1,1,1,.5,2],
             [2,0,1,0,2],
             [3,0,0.75,0.75,1],
             [4,0,0,1,1],
             [5,0,0,0,1],
             [6,1,0,0,1]]),
        6: np.array(
            [[0,1,1,1,0],
             [1,1,1,.5,2],
             [2,0,1,0,2],
             [3,0,0.75,0.75,1],
             [4,0,0,1,1],
             [5,1,0,0,1],
             [6,0,0,0,2]]),
        }
    _len =  len(maps)
    def __init__(self, mode = 1):
        try:
            cmap = self.maps[mode]
        except:
            raise AttributeError('Invalid mode')
        super().__init__(
            cmap = cmap,
            layout = 'XCCCG')

for i in range(ColorMapGal._len):
    register_color('GalMap{:d}'.format(i), ColorMapGal(i))


class ColorMapGray(ColorMap):
    maps = {
        0:  np.array(
            [[0,0,0],
             [1,1,1]]),
        1:  np.array(
            [[0,0,0],
             [1,1,lambda x: 0.5*(1-np.cos(x*np.pi))]]),
        }
    _len =  len(maps)
    def __init__(self, mode = 0):
        try:
            cmap = self.maps[mode]
        except:
            raise AttributeError('Invalid mode.')
        super().__init__(
            cmap = cmap,
            layout = 'XCG')
    @staticmethod
    def is_gray():
        return True

for i in range(ColorMapGray._len):
    register_color('GrayMap{:d}'.format(i), ColorMapGray(i))


class ColorRGBWaves(Color):
    """
    red-green-blue+waves

    parameter:
       nwaves = 200
    """
    _alpha = False
    def __init__(self, nwaves = 200, **kwargs):
        super().__init__(**kwargs)
        self.waves = nwaves*np.pi
        self._N = np.max((np.round(12 * nwaves), 1024))
    def _function(self, x, *args, **kwargs):
        return _color_models['HSV'](np.array([
                    300 * x,
                    x**0.5,
                    1 - 0.25 * (np.sin(x * self.waves)**2)
                    ]).transpose())

register_color('RGBWaves', ColorRGBWaves)

class ColorRKB(Color):
    """
    red-black-blue
    """
    _alpha = False
    def _function(self,x, *args, **kwargs):
        def theta(x):
            y = np.zeros_like(x)
            y[x<0] = -1
            y[x>0] = +1
            return y
        return _color_models['HSV'](np.array([
                    30 + 180 * x + 30 * theta(x - 0.5),
                    np.ones_like(x),
                    np.minimum(5*np.abs(x - 0.5),1)
                    ]).transpose())

register_color('RKB', ColorRKB)

class ColorBWR(ColorMap):
    """
    blue white red with adjustable white at paramter value

    parameters:
      white = 0.5 - white point x_0
      gamma = 2.0 - gamma for smoothness, abs(x-x_0)**gamma
    """
    def __init__(self,
                 white = 0.5,
                 gamma = 2.0,
                 **kwargs):
        assert 0 <= white <= 1
        assert gamma > 0
        cmap = np.array(
            [[0    ,0,0,1,0.0],
             [white,1,1,1,gamma],
             [1    ,1,0,0,1./gamma]])
        super().__init__(
            cmap = cmap,
            layout = 'XCCCG',
            **kwargs)

register_color('BWR', ColorBWR)

class ColorBWGRY(ColorMap):
    """
    blue white green-red-yellow with adjustable white at paramter value

    parameter:
       p = 0.5 - adjustable value of white
    """
    def __init__(self,
                 p = 0.5,
                 **kwargs):
        assert 0 <= p <= 1
        p13 = (1-p)/3
        cmap = np.array(
            [[0      ,0,0,1,0.00],
             [p      ,1,1,1,2.00],
             [1-2*p13,0,1,0,1.00],
             [1-1*p13,1,0,0,0.85],
             [1      ,1,1,0,1]])
        super().__init__(
            cmap = cmap,
            layout = 'XCCCG',
            **kwargs)

register_color('BWGRY', ColorBWGRY)

class ColorKRGB(Color):
    """
    red-green-blue+waves
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def _function(self,x, *args, **kwargs):
        return _color_models['HSV'](np.array([
                    x*270,
                    np.ones_like(x),
                    np.minimum(10*x,1)
                    ]).transpose())

register_color('KRGB', ColorKRGB)

class ColorBWC(ColorMap):
    """
    grey-white-color
    """
    def __init__(self,
                 p = 0.5,
                 mode = 0,
                 **kwargs):
        assert 0 <= p <= 1
        p2 = (1-p)/3 + p
        if mode == 0:
            cmap = np.array(
                [[0 ,   0,0,0,0.],
                 [p , 120,0,1,1.],
                 [p2, 120,1,1,2.],
                 [1 ,-120,1,1,1.]])
        elif mode == 1:
            cmap = np.array(
                [[0 ,  0,0,0,0.],
                 [p ,120,0,1,1.],
                 [p2,120,1,1,2.],
                 [1 ,420,1,1,1.]])
        elif mode == 2:
            f = lambda x: np.sin(0.5*x*np.pi)
            cmap = np.array(
                [[0 ,  0,0,0,0.],
                 [p ,240,0,1,1.],
                 [p2,240,1,1,f],
                 [1 ,-30,1,1,1.]])
        else:
            cmap = np.array(
                [[0 ,  0,0,0,0.],
                 [p ,240,0,1,1.],
                 [p2,240,1,1,1.],
                 [1 ,-30,1,1,1.]])
        super().__init__(
            cmap = cmap,
            layout = 'XCCCG',
            model = 'HSV',
            normalize = False,
            **kwargs)

register_color('BWC', ColorBWC)

class ColorMapFunction(ColorMap):
    """
    generate color function from colormap by linear interpolation
    """
    def __new__(cls, name, *args, **kwargs):
        return cls.from_Colormap(name, *args, **kwargs)

# -----------------------------------------------------------------------

class ColorDiverging(Color):
    """
    from K Moreland
    """
    def __init__(self, colors, **kwargs):
        kw = dict(kwargs)
        model = ColorMshInverse()
        colors = iterable(colors)
        colors = [rgb(c) for c in colors]
        if len(colors) == 1:
            if kw.pop('diverging', 'True'):
                c = ColorHSVInverse()(colors[0])
                c[0] = (c[0] + 180) % 360
                c = ColorHSV()(c)
                colors += [c]
            else:
                colors = (colors[0], rgb(1.))
        colors = np.array([
            model(c) for c in colors
            ])
        if len(colors) == 2:
            if (colors[0][1] > 3 and
                colors[1][1] > 3 and
                (colors[0][2] - colors[1][2]) % 360 > 60):
                Mmid = max(colors[0][0], colors[1][0]) # , 88)
                col = np.array([Mmid, 0., 0.])
                colors = np.insert(colors, 1, col, axis = 0)
        values = kw.pop('values', None)
        if values is None:
            values = list(np.linspace(0, 1, len(colors), endpoint = True))
        else:
            values = list(iterable(values))
        if values[0] != 0:
            values = [0.] + values
        if values[-1] != 1:
            values = values +  [1.]
        values = np.array(values)
        assert len(values) == len(colors)
        assert np.alltrue(values[:-1] < values[1:])

        # add double nodes for adjusted hue values for unsaturated colors
        for i in range(len(colors)-2, 0, -1):
            if colors[i][1] < 3:
                col = colors[i].copy()
                col[2] = self._adjust_hue(colors[i+1][0],
                                          colors[i+1][1],
                                          colors[i+1][2],
                                          colors[i][2])
                colors = np.insert(colors, i+1, col, axis = 0)
                values = np.insert(values, i+1, values[i])

                col = colors[i].copy()
                col[2] = self._adjust_hue(colors[i-1][0],
                                          colors[i-1][1],
                                          colors[i-1][2],
                                          colors[i][2])
                colors[i, :] = col

        # add cap point
        values = np.append(values, values[-1] + 0.1)
        colors = np.append(colors, (colors[-1]).reshape(1,-1), axis = 0)
        self._colors = colors
        self._values = values

        super().__init__(**kw)

    # @staticmethod
    # def _adjust_hue_vec(M_sat, s_sat, h_sat, M_unsat):
    #     """
    #     rotate h to retain const dE
    #     """
    #     h = h_sat.copy()
    #     ii = M_sat < M_unsat
    #     h_spin = s_sat * np.sqrt((M_unsat[ii]/ M_sat[ii])**2 - 1) / np.sin(s_sat[ii]  * np.pi / 180)
    #     # spin away from purple
    #     s = (h_sat[ii] + 300) % 360 > 0
    #     h[ii] += h_spin * (2 * s.astype(int) - 1)
    #     return h

    @staticmethod
    def _adjust_hue(M_sat, s_sat, h_sat, M_unsat):
        """
        rotate h to retain const dE
        """
        if M_sat >= M_unsat:
            return h_sat
        h_spin = s_sat * np.sqrt((M_unsat / M_sat)**2 - 1) / np.sin(s_sat * np.pi / 180)
        # spin away from purple
        s = (h_sat + 300) % 360 > 0
        h = h_sat + h_spin * (2 * s.astype(int) - 1)
        return h

    def _function(self, data, *args, **kwargs):
        out = np.zeros(data.shape + (3,))
        # --- begn --- this part should be in __init__
        d = self._values[1:] - self._values[:-1]
        d[d==0] = 1
        dc = self._colors[1:] - self._colors[:-1]
        ii = dc[:, 2] > 180
        dc[ii, 2] = dc[ii, 2] - 360
        ii = dc[:, 2] < -180
        dc[ii, 2] = dc[ii, 2] + 360
        # --- end ---
        for i in range(len(d)):
            ii = np.logical_and(data >= self._values[i], data < self._values[i+1])
            out[ii, :] = (self._colors[np.newaxis, i] +
                          dc[np.newaxis, i] *
                          (data[ii, np.newaxis] - self._values[i]) / d[i])
        return ColorMsh()(out)


########################################################################

class ColorSolid(Color):
    _alpha = True
    def __init__(self, color, alpha = 1., **kwargs):
        assert alpha is not None
        self._color = rgb(color, alpha = alpha)
        super().__init__(**kwargs)
    def _function(self, data, *args, **kwargs):
        out = np.ndarray(data.shape + (4,))
        out[:,:] = self._color[np.newaxis,:]
        return out

########################################################################


class ColorMapList(ColorMap):
    """
    create maps based on list of color names
    """

    def __init__(self, colors, gamma = 1, **kwargs):
        colors = [rgb(c) for c in colors]
        n = len(colors)
        if n == 1:
            colors = colors + colors
            n = 2
        if gamma == 1:
            cmap = np.ndarray((n, 5))
            cmap[:, 0] = np.linspace(0, 1, n, endpoint = True)
            cmap[:, 1:4] = np.array(colors)
            cmap[:, 4] = gamma
        else:
            nx = 2 * n - 1
            cmap = np.ndarray((nx, 5))
            cmap[:     , 0] = np.linspace(0, 1, nx, endpoint = True)
            cmap[0:nx:2, 1:4] = np.array(colors)
            cmap[1:nx:2, 1:4] = 0.5 * (cmap[0:nx-2:2, 1:4] + cmap[2:nx:2, 1:4])
            cmap[0:nx:2, 4] = 1 / gamma
            cmap[1:nx:2, 4] = gamma
        super().__init__(
            cmap = cmap,
            layout = 'XCCCG',
            model = 'RGB',
            normalize = False,
            **kwargs)

########################################################################

# Some color maps for color blind
#
#    from Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl)
#    http://www.sron.nl/~pault/

class ColorBlindBWR(Color):
    """
    Blue-White-Red colormap.  Good for color-blind

    from Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl)
    http://www.sron.nl/~pault/

    Keyword Arguments:
    center:
      specify location of center. 0 .. 1, default: 0.5
    gamma:
      specify gamma on x  0 < gamma < \infty, default: 1
    """
    def __init__(self, *args, **kwargs):
        xargs = dict(kwargs)
        self._gamma = xargs.pop('gamma', 1)
        self._center = xargs.pop('center', 0.5)
        super().__init__(*args, **xargs)
    def _function(self, x, *args, **kwargs):
        if self._center != 0.5 or self._gamma != 1:
            ii = x < self._center
            x = x.copy()
            x[ii] = (1 - (1 - (x[ii] / self._center)) ** self._gamma) * 0.5
            ii = np.logical_not(ii)
            x[ii] = 0.5 * (1 + ((x[ii] - self._center) / (1 - self._center)) ** self._gamma)
        r = 0.237 + x * (-2.13 + x * (26.92 + x * (-65.5 + x * (63.5 - x * 22.36))))
        g = ((0.572 + x * (1.524 - x * 1.811))/(1 + x * (-0.291 + x * 0.1574)))**2
        b = 1/(1.579 + x * (-4.03 + x * (12.92 + x * (-31.4 + x * (48.6 - x * 23.36)))))
        return _color_models['RGB'](np.array([r, g, b]).transpose())

register_color('BlindBWR', ColorBlindBWR)

class ColorBlindWR(Color):
    """
    White-Red colormap.  Good for color-blind

    from Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl)
    http://www.sron.nl/~pault/

    Keyword Arguments:
    gamma:
      specify gamma on x  0 < gamma < \infty, default: 1
      if gamma < 0, apply -gamma to 1-x
    """
    def __init__(self, *args, **kwargs):
        xargs = dict(kwargs)
        self._gamma = xargs.pop('gamma', 1)
        super().__init__(*args, **xargs)
    def _function(self, x, *args, **kwargs):
        if self._gamma < 0:
            x = 1 - (1-x)**(-self._gamma)
        elif self._gamma != 1:
            x = x**self._gamma
        r = (1 - 0.392 * (1 + erf((x - 0.869) / 0.255)))
        g = (1.021 - 0.456 * (1 + erf((x - 0.527) / 0.376)))
        b = (1 - 0.493 * (1 + erf((x - 0.272) / 0.309)))
        return _color_models['RGB'](np.array([r, g, b]).transpose())

register_color('BlindWR', ColorBlindWR)

class ColorBlindRainbow(Color):
    """
    Rainbow colormap.  Good for color-blind

    from Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl)
    http://www.sron.nl/~pault/

    Keyword Arguments:
    gamma:
      specify gamma on x  0 < gamma < \infty, default: 1
      if gamma < 0, apply -gamma to 1-x
    """
    def __init__(self, gamma = 1, **kwargs):
        self._gamma = gamma
        super().__init__(**kwargs)
    def _function(self, x, *args, **kwargs):
        if self._gamma < 0:
            x = 1 - (1-x)**(-self._gamma)
        elif self._gamma != 1:
            x = x**self._gamma
        r = (0.472 + x * (-0.567 + x * 4.05)) / (1 + x * (8.72 + x * (-19.17 + x * 14.1)))
        g = 0.108932 + x * (-1.22635 + x * (27.284 + x * (-98.577 + x * (163.3 + x * (-131.395 + x * 40.634)))))
        b = 1 / (1.97 + x * (3.54 + x * (-68.5 + x * (243 + x * (-297 + x * 125)))))
        return _color_models['RGB'](np.array([r, g, b]).transpose())

register_color('BlindRainbow', ColorBlindRainbow)

class ColorCircle(Color):
    def __init__(self, start = 0, stop = 360, **kwargs):
        self._start = start
        self._stop  = stop
        super().__init__(**kwargs)
    def _function(self, x, **kwargs):
        y = np.empty_like(x)
        y[:] = 1
        x = self._start + (self._stop - self._start) * x
        return _color_models['HSV'](np.array([x, y, y]).transpose())

register_color('Circle', ColorCircle)

class ColorCircle(Color):
    def __init__(self, model = 'HSV', start = 0, stop = 360, **kwargs):
        kw = dict(kwargs)
        model = color_model(model)
        index = kw.pop('index', None)
        if index is None:
            ii = np.where(model.limits == 360)[0]
            if len(ii) != 1:
                raise ValueError('require color model with one angle')
            index = ii[0]
        self._index = index
        self._model = model
        self._start = start
        self._stop = stop
        self._func = kw.pop('func', lambda x: x)
        self._init = kw.pop('init', None)
        if self._init is not None:
            assert len(self._init) == 3, 'need to provide init 3-array'
        super().__init__(**kwargs)

    _alpha = False
    def _function(self, data, **kwargs):
        if self._init is not None:
            x = np.ndarray(data.shape + (3,))
            x[:, :] = np.array(self._init)[np.newaxis, :]
        else:
            x = np.ones(data.shape + (3,))
        x[:, self._index] = (self._start + self._func(data) * (self._stop - self._start))
        rgb = self._model(x)
        return rgb

register_color('ColorCircle', ColorCircle)

class ColorGray(Color):
    """
    Create gray shade colormap, black to white.

    Allows sprcification of gamma corrections,
    also for each component separately.
    """
    def __init__(self, gamma = 1, reverse = False, **kwargs):
        if is_iterable(gamma):
            if len(gamma) != 3:
                raise AttributeError('Wrong dimension for gamma: {}'.format(gamma))
            gamma = np.array(gamma)
        self._gamma = gamma
        self._reverse = reverse
        super().__init__(**kwargs)

    def _function(self, x, **kwargs):
        if self._reverse:
            x = 1 - x
        if is_iterable(self._gamma):
            rgb = np.ndarray(x.shape + (3,))
            ii = self._gamma >= 0
            rgb[:, ii] = x[:, np.newaxis] ** self._gamma[np.newaxis, ii]
            ii = np.logical_not(ii)
            rgb[:, ii] = 1 - (1 - x[:, np.newaxis]) ** (-self._gamma[np.newaxis, ii])
        else:
            if self._gamma < 0:
                x = 1 - (1 - x)**(-self._gamma)
            else:
                x = x**self._gamma
            rgb = np.tile(x.reshape((-1, 1)), 3)
        return _color_models['RGB'](rgb)

register_color('Gray', ColorGray)

class ShadeColor(Color):
    def __init__(self, color = 'Blue', **kwargs):
        kw = dict(kwargs)
        self._color = rgb(color)[:3]
        self._gamma = kw.pop('gamma', 1.)
        self._model = color_model(kw.pop('model', 'HSV'))
        self._index = kw.pop('index', 1)
        self._method = kw.pop('method', 0)
        self._func = kw.pop('func', lambda x: x)
        super().__init__(**kwargs)

    def _function(self, data, **kwargs):
        rgb = np.tile(self._color, data.shape + (1,))
        ccc = self._model.inverse(rgb)
        x = self._func(data)**self._gamma
        if self._method == 0:
            x = 1 - x
        ccc[:, self._index] = x
        return self._model(ccc)

########################################################################
########################################################################
########################################################################

class ColorBlend(Color):
    """
    blend N = 2+ color functions together

    provide N-1 blend functions "func" (the last weight is the remainder).

    values outside 0 <= data <= 1 are take from first function

    provide N functions to preprocess values that go into call to
    color function (optional, identity by default).  Can be used to
    scale input range.
    """

    def __init__(self, colors, *args, **kwargs):
        kw = dict(kwargs)
        func = kw.pop('func', lambda x: x)
        ifunc = kw.pop('ifunc', lambda x: x)
        colors = [colormap(c) for c in iterable(colors)]
        assert len(colors) > 0
        func = tuple(iterable(func))
        assert len(colors) - len(func) in (0, 1)
        ifunc = tuple(iterable(ifunc))
        if len(ifunc) == 1:
            ifunc = ifunc * len(colors)
        assert len(colors) == len(ifunc)

        self._func = func
        self._ifunc = ifunc
        self._colors = colors
        self.bytes = self._colors[0].bytes
        super().__init__(*args, **kw)

    # _function returns rgba array
    _alpha = True

    def _function(self, data, *args, **kwargs):
        kw = dict(kwargs)
        kw['bytes'] = False
        kw['return_data'] = False
        kw['normalize'] = False

        xt = np.zeros_like(data)
        rgba = np.zeros(data.shape + (4,))

        # fill regular values
        n = len(self._colors)
        nf = len(self._func)
        for i in range(n):
            if i == n - 1 and nf == n - 1:
                x = 1 - xt
                if not np.alltrue(np.logical_and(x >= 0, x <= 1)):
                    if np.min(x) > -1.e-8:
                        x = np.maximum(x, 0)
                    if np.max(x) < 1 + 1.e-8:
                        x = np.minimum(x, 1)
                    if not np.alltrue(np.logical_and(x >= 0, x <= 1)):
                        raise AssertionError(
                            'function values sum needs to be 0 <= \sum_i f[x] <= 1\n'
                            + 'min={}, max={}'.format(np.min(x), np.max(x)))
            else:
                x = self._func[i](data)
                xt += x
                if i == n - 1:
                    assert np.alltrue(np.isclose(xt, 1)),'weights  don\'t add up'
            v = self._ifunc[i](data)
            rgba += x[:, np.newaxis] * self._colors[i](v, *args, **kw)
        ii = np.isclose(rgba[:,3], 1)
        rgba[ii, 3] = 1
        return rgba

class ColorBlendBsplineInterpolate(ColorBlend):
    def __init__(self, colors, *args, **kwargs):
        from bspline import BsplineInterpolate
        kw = dict(kwargs)
        k = kw.pop('k', 3)

        n = len(colors)
        assert n > 1

        t = kw.pop('frac', None)
        if t is None:
            t = np.linspace(0, 1, n, endpoint = True)
        if t[0] != 0:
            t = [0] + list(t)
        if t[-1] != 1:
            t = list(t) + [1]
        t = np.array(t)
        assert len(t) == n

        Bi = []
        for i in range(n-1):
            y = np.zeros_like(t)
            y[i] = 1
            Bi += [BsplineInterpolate(k, t, y)]

        super().__init__(colors, func = Bi , **kw)

class ColorBlendBspline(ColorBlend):
    def __init__(self, colors, *args, **kwargs):
        from bspline import Bspline
        kw = dict(kwargs)

        n = len(colors)
        assert n > 1

        t = kw.pop('frac', None)
        if t is not None:
            if t[0] != 0:
                t = [0] + list(t)
            if t[-1] != 1:
                t = list(t) + [1]
            t = np.array(t)
            nt = len(t)
            k = kw.pop('k', n-nt+1)
        else:
            k = kw.pop('k', None)
            if k is None:
                k = min(3, n-1)
            nt = n - k + 1
            assert nt >= 2
            t = np.linspace(0, 1, nt, endpoint = True)
        Bi = []
        for i in range(n-1):
            Bx = Bspline(k, t)
            Bi += [partial(Bx, i)]

        super().__init__(colors, func = Bi , **kw)

class ColorJoin(ColorBlend):
    """
    Join (stack) different colors

    TODO: make work for level data???
    """
    class IFunc(object):
        def __init__(self, x0, dx):
            self.x0 = x0
            self.dx = dx
        def __call__(self, x):
            return (x - self.x0) / self.dx

    class IntervalFunc(object):
        def __init__(self, x0, x1):
            self.x0 = x0
            self.x1 = x1
        def __call__(self, x):
            ii = np.logical_and(x >= self.x0, x < self.x1)
            out = np.zeros_like(x)
            out[ii] = 1
            return out

    class BlendFunc(object):
        def __init__(self, x0, x1, d0, d1):
            self.x0 = x0
            self.x1 = x1
            self.d0 = d0
            self.d1 = d1
        def _func(self, x):
            """
            function should be symmetric about x = 0.5
            """
            raise NotImplementedError()
        def __call__(self, x):
            ii = np.logical_and(x >= self.x0, x < self.x1)
            out = np.zeros_like(x)
            out[ii] = 1
            if self.d0 > 0:
                jj = np.logical_and(x >= self.x0, x < self.x0 + self.d0)
                out[jj] = self._func((x[jj] - self.x0) / self.d0)
            if self.d1 > 0:
                jj = np.logical_and(x <= self.x1, x > self.x1 - self.d1)
                out[jj] = self._func((self.x1 - x[jj]) / self.d1)
            return out

    class LinFunc(BlendFunc):
        def _func(self, x):
            return x

    class CubeFunc(BlendFunc):
        def _func(self, x):
            return 0.5+4*(x-0.5)**3

    class SqrFunc(BlendFunc):
        def _func(self, x):
            return 0.5+2*(x-0.5)**2*np.sign(x-0.5)

    class RootFunc(BlendFunc):
        def _func(self, x):
            return 0.5+0.5*np.abs(x-0.5)**(1/2)*2**(1/2)*np.sign(x-0.5)

    class CubeRootFunc(BlendFunc):
        def _func(self, x):
            return 0.5+0.5*np.abs(x-0.5)**(1/3)*2**(1/3)*np.sign(x-0.5)

    class CosFunc(BlendFunc):
        def _func(self, x):
            return 0.5 * (1 - np.cos(x * np.pi))

    class CosFunc3(BlendFunc):
        def _func(self, x):
            return 0.5 * (1 - np.cos(x * np.pi)**3)

    class GenericFunc(BlendFunc):
        def __init__(self, *args):
            self._func = args[0]
            super().__init__(*(args[1:]))

    methods = {
        'linear' : LinFunc,
        'cubic' : CubeFunc,
        'square' : SqrFunc,
        'root' : RootFunc,
        'cuberoot' :CubeRootFunc,
        'cos' : CosFunc,
        'cos3' : CosFunc3,
        }

    def __init__(self, colors,
                 values = None,
                 blend = None,
                 method = 'cos',
                 **kwargs):
        colors = [colormap(c) for c in iterable(colors)]

        kw = dict(kwargs)

        #values = kw.pop('values', None)
        if values is None:
            values = list(np.linspace(0, 1, len(colors)+1, endpoint=True))
        else:
            values = list(iterable(values))
        if values[0] != 0:
            values = [0.] + values
        if values[-1] != 1:
            values = values +  [1.]
        values = np.array(values)
        assert len(values) == len(colors) + 1
        assert np.alltrue(values[:-1] < values[1:])

        #blend = kw.pop('blend', None)
        if blend is not None:
            if not is_iterable(blend):
                blend = np.tile(blend, len(values) - 2)
            if not isinstance(blend, np.ndarray):
                blend = np.array(blend)
            assert len(blend) == len(values) - 2
            assert np.alltrue(blend > 0)
            assert np.alltrue(blend <= 0.5)

        n = len(colors)

        x0 = values[ :-1].copy()
        x1 = values[1:  ].copy()
        d0 = values[1:] - values[:-1]
        delta = np.minimum(d0[1:], d0[:-1])
        if blend is not None:
            dx0 = delta * blend * 2
            x0[1:  ] -= 0.5 * dx0
            x1[ :-1] += 0.5 * dx0
            dx = x1[:-1] - x0[1:]
            d0 = x1 - x0
        x1[-1] = 1.1

        ifunc = [self.IFunc(x0[i], d0[i]) for i in range(n)]

        if blend is not None:
            dx = np.concatenate((np.tile(-1,1), dx, np.tile(-1,1)))
            if isinstance(method, str):
                m = self.methods[method]
            else:
                m = partial(self.GenericFunc, method)
            func = [m(x0[i], x1[i], dx[i], dx[i+1]) for i in range(n)]
        else:
            func = [self.IntervalFunc(x0[i], x1[i]) for i in range(n)]

        kw['func'] = func
        kw['ifunc'] = ifunc
        super().__init__(colors, **kw)

class ColorBlendWave(ColorBlend):
    def __init__(self, color0, color1, *args, **kwargs):
        kw = dict(kwargs)
        nwaves = kw.pop('nwaves', 20)
        phase = kw.pop('phase', 0)
        amplitude = kw.pop('amplitude', 0.5)
        center = kw.pop('center', None)
        power = kw.pop('power', 1)
        if center is None:
            center = 1 - amplitude
        if power > 0:
            func = lambda x : center + amplitude * np.sin(2 * np.pi * (x + phase) * nwaves)**power
        elif power == 0:
            func = lambda x : center + amplitude * np.sign(np.sin(2 * np.pi * (x + phase) * nwaves))
        else:
            raise AttributeError('power: negative values not allowed.')
        colors = (color0, color1)
        super().__init__(colors, func = func , **kw)

class ColorBlendGamma(ColorBlend):
    def __init__(self, color0, color1, *args, **kwargs):
        kw = dict(kwargs)
        gamma = kw.pop('gamma', 1)
        if gamma > 0:
            func = lambda x: x ** gamma
        else:
            func = lambda x: 1 - (1 - x) ** (-gamma)
        colors = (color0, color1)
        super().__init__(colors, func = func , **kw)
