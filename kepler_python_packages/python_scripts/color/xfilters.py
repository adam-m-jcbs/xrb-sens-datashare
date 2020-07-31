import numpy as np

from .filters import ColorFilter
from .models import color_model
from .utils import color, rgb

class XFilter(ColorFilter):
    """
    Class to modify data color array before returned.

    The xfilter takes as call arguments RGBA and data arrays
    normalized to 0...1.

    When used a _xfilter function parameters must be passes by
    keyword.
    """

    _filter = None

    def _xfilter(self, rgba, data):
        """
        take rgba and data input, filter, replace data in place, return modified array
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        # use as _xfilter
        if (self._color is None and
            len(args) == 2 and
            len(kwargs) == 0 and
            isinstance(args[0], np.ndarray) and
            isinstance(args[1], np.ndarray) and
            len(args[0].shape) > 1 and
            args[0].shape[-1] == 4 and
            np.allclose(args[0].shape[:-1], args[1].shape)):
            return self._xfilter(*args)

        return super().__call__(*args, **kwargs)


class ComponentFilter(XFilter):
    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        self._amplitude = kw.pop('amplitude', 1)
        self._model = kw.pop('model', 'HCL')
        self._property = kw.pop('property', 'L')
        self._index = kw.pop('index', None)
        self._method = kw.pop('method', None)
        self._anglefac = kw.pop('anglefac', 1)
        func = kw.pop('func', None)
        if np.isscalar(func):
            self._func = lambda x : np.tile(func, x.shape)
        else:
            self._func = func
        super().__init__(*args, **kw)

    _models = {
        'RGB' : dict(R = (0,0), G = (1,0), B = (2,0)),
        'CMY' : dict(C = (0,0), M = (1,0), Y = (2,0)),
        'HSL' : dict(H = (0,2), S = (1,1), L = (2,0)),
        'HSV' : dict(H = (0,2), S = (1,1), V = (2,1)),
        'HSI' : dict(H = (0,2), S = (1,1), I = (2,0)),
        'HCL' : dict(H = (0,2), C = (1,1), L = (2,0)),
        'HCL2': dict(H = (0,2), C = (1,1), L = (2,0)),
        'YIQ' : dict(Y = (0,0), I = (1,1), Q = (2,0)),
        'LChuv':dict(L = (0,0), C = (1,1), huv=(2,2)),
               }

    def _xfilter(self, rgba, data):
        ii = np.logical_and(0 <= data, data <= 1)
        model = color_model(self._model)
        amplitude = self._amplitude
        ccca = model.inverse(rgba[ii, :])
        f = self._func(data[ii])
        try:
            i, method = self._models[self._model][self._property]
        except:
            i = method = None
        if self._method is not None:
            method = self._method
        if self._index is not None:
            i = self._index
        if method == 0:
            l1 = model.limits[i,1]
            if np.isinf(l1):
                l1 = np.max(ccca[:, i])
            ccca[:, i] = l1 - (l1 - ccca[:, i]) * (1 - amplitude * f)
            ccca[:, i] = np.clip(ccca[:, i], model.limits[i,0], model.limits[i,1])
        elif method == 1:
            l0 = model.limits[i,0]
            if np.isneginf(l0):
                l0 = np.min(ccca[:, i])
            ccca[:, i] = l0 + (ccca[:, i] - l0) * (1 - amplitude * f)
            ccca[:, i] = np.clip(ccca[:, i], model.limits[i,0], model.limits[i,1])
        elif method == 2:
            if amplitude > 0:
                amplitude *= (model.limits[i,1] - model.limits[i,0]) * self._anglefac
            else:
                amplitude = -amplitude
            ccca[:, i] += amplitude * f
            ccca[:, i] = np.mod(ccca[:, i], model.limits[i, 1])
        else:
            raise ValueError('Invalid Method: {}'.format(method))
        rgba[ii, :] = model(ccca)
        return rgba

class WaveFilter(ComponentFilter):
    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        nwaves = kw.pop('nwaves', 200)
        phase = kw.pop('phase', 0)
        func = lambda x : 0.5 * (1 + np.sin(2 * np.pi * (x + phase) * nwaves))
        kw.setdefault('func', func)
        kw.setdefault('amplitude', 0.5)
        kw.setdefault('model', 'HCL')
        kw.setdefault('property', 'L')
        kw.setdefault('anglefac', 1 / 6)
        super().__init__(*args, **kw)

class BlendInFilter(XFilter):
    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        self._frac     = kw.pop('frac', 0.25)
        self._location = kw.pop('location', None)
        self._model    = kw.pop('model', 'HSL')
        self._index    = kw.pop('index', 2)
        self._reverse  = kw.pop('reverse', True)
        # TODO - add args
        super().__init__(*args, **kw)

    def _xfilter(self, rgba, data):
        ii = np.logical_and(0 <= data, data <= 1)
        model = color_model(self._model)
        i = self._index
        reverse = self._reverse
        ccca = model.inverse(rgba[ii])
        if self._location is not None and self._frac > 0:
            f = 0.5 * (1 - np.cos(np.pi * np.clip((data[ii]-self._location) / self._frac, -1, 1)))
        elif self._location is not None and self._frac < 0:
            f = 0.5 * (1 + np.cos(np.pi * np.clip((data[ii]-self._location) / self._frac, -1, 1)))
        elif self._frac > 0:
            f = 0.5 * (1 - np.cos(np.pi * np.minimum(data[ii] / self._frac, 1)))
        else:
            f = 0.5 * (1 - np.cos(np.pi * np.minimum((data[ii] - 1) / self._frac, 1)))
        if reverse:
            ccca[:, i] = model.limits[i,1] - (model.limits[i,1] - ccca[:, i]) * f
        else:
            ccca[:, i] = (ccca[:, i] - model.limits[i,0]) * f + model.limits[i,0]
        rgba[ii, :] = model(ccca)
        return rgba

class AlphaFilter(XFilter):
    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        self._func = kw.pop('func', lambda x: x)
        super().__init__(*args, **kw)

    # unfortunately pcolorfast alpha is broken (mpl 1.5 beta)
    # seems to be working now (mpl 3)
    def _xfilter(self, rgba, data):
        ii = np.logical_and(0 <= data, data <= 1)
        rgba[ii, 3] = np.clip(self._func(data[ii]), 0, 1)
        return rgba

class FuncBackgroundFilter(AlphaFilter):
    """
    blend in to background based on function
    """
    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        background = kw.pop('background', '#ffffff')
        assert isinstance(background, str)
        col = color(background)
        if col is None:
            col = background
        background = rgb(col, no_alpha = True)
        assert background.shape == (3,)
        self._background = background
        super().__init__(*args, **kw)

    def _xfilter(self, rgba, data):
        ii = np.logical_and(0 <= data, data <= 1)
        x = np.clip(self._func(data[ii]), 0, 1)
        rgba[ii, 0:3] = (1 - x)[:,np.newaxis]*self._background[np.newaxis,:] + x[:,np.newaxis] * rgba[ii, 0:3]
        return rgba

class SetBackgroundFilter(XFilter):
    """
    replacment routine to simulate alpha on background (default: White)

    use alpah channle value
    """
    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        background = kw.pop('background', '#ffffff')
        assert isinstance(background, str)
        col = color(background)
        if col is None:
            col = background
        background = rgb(col, no_alpha = True)
        assert background.shape == (3,)
        self._background = background
        super().__init__(*args, **kw)

    def _xfilter(self, rgba, data):
        ii = np.logical_and(0 <= data, data <= 1)
        x = rgba[ii, 3]
        rgba[ii, 0:3] = (1 - x)[:,np.newaxis]*self._background[np.newaxis,:] + x[:,np.newaxis] * rgba[ii, 0:3]
        rgba[ii, 3] = 1
        return rgba

class HueRotateFilter(XFilter):
    """
    Rotate Hue value based on data.

    See Notes in FilterColorHue Rotate.
    Could be implemented using ComponentFilter.
    """

    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        self._angle = kw.pop('angle', 60)
        model = kw.pop('model', 'HCL')
        model = color_model(model)
        index = kw.pop('index', None)
        if index is None:
            ii = np.where(model.limits == 360)[0]
            if len(ii) != 1:
                raise ValueError('require color model with one angle')
            index = ii[0]
        self._index = index
        self._model = model
        self._func = kw.pop('func', lambda x: x)
        super().__init__(*args, **kw)

    def _xfilter(self, rgba, data):
        ii = np.logical_and(0 <= data, data <= 1)
        ccca = self._model.inverse()(rgba[ii])
        ccca[:, self._index] += self._angle * self._func(data[ii])
        rgba[ii] = self._model(ccca)
        return rgba

    @staticmethod
    def valid_models():
        return FilterColorHueRotate.valid_models


#-----------------------------------------------------------------------

class ColorXFilterColor(XFilter):
    """
    Class to filter by color (RGB).

    TODO: add other color models
    """

    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)

        xcolor = kw.pop('color', '#ffffff')
        assert isinstance(xcolor, str)
        col = color(xcolor)
        if col is None:
            col = xcolor
        try:
            xcolor = rgb(col, no_alpha = True)
        except:
              pass
        assert xcolor.shape == (3,)
        reverse = kw.pop('reverse', False)
        if reverse:
            xcolor = 1 - xcolor
        self._xcolor = xcolor
        self._method = kw.pop('method', 'min')

        self._func = kw.pop('func', lambda x: x)
        self._clip = kw.pop('clip', True)

        super().__init__(*args, **kw)

    def _xfilter(self, rgba, data):
        """
        take rgba input, filter, replace data in place, return modified array
        """
        ii = np.logical_and(0 <= data, data <= 1)
        jj = self._xcolor > 0
        c = self._xcolor[np.newaxis, jj]
        jj = np.append(jj, np.tile(False, 4 - len(jj)))
        kk = np.ix_(ii, jj)
        if self._method == 'vec':
            f = np.sqrt(np.sum((rgba[kk] * c)**2, axis = 1) / np.sum(c**2))
        elif self._method == 'min':
            f = np.min(rgba[kk] / c, axis = 1) * np.max(c)
        f *= self._func(data[ii])
        rgba[kk] -= f[:, np.newaxis] * c
        if self._clip:
            rgba[kk] = np.clip(rgba[kk], 0, 1)
        return rgba
