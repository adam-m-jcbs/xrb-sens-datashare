import numpy as np

from .levels import LevelData
from .models import _color_models, color_model
from .functions import Color, colormap
from ._utils import is_iterable, iterable
from .utils import color, rgb

# for blind filter
from .models import ColorsRGB, ColorxyYInverse, ColorXYZ, ColorsRGBInverse

class ColorFilter(LevelData):
    """
    Class to filter color array before it is returned.

    The filter takes the call argument in RGBA space 0...1.

    When using as a _filter function, parameters must be passes by keyword.
    """

    def _filter(self, rgba):
        """
        take rgba input, filter, replace data in place, return modified array
        """
        raise NotImplementedError()

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            color = args[0]
            args = args[1:]
            if 'color' in kwargs:
                raise AttributeError('duplicate specification of `color`')
        else:
            color = kwargs.pop('color', None)
        if not isinstance(color, Color):
            color = colormap(color)
        if color is None and len(args) > 0:
            raise Exception('require valid color function')
        super().__init__(*args, **kwargs)

        self._color = color
        self.bytes = color.bytes
        try:
            self._n =  self._color._n
        except:
            pass

    def __call__(self, *args, **kwargs):
        # use as _filter
        if (self._color is None and
                len(args) == 1 and
                len(kwargs) == 0 and
                isinstance(args[0], np.ndarray) and
                args[0].shape[-1] == 4 and
                len(args[0].shape) > 1):
            return self._filter(*args)

        # normal call
        if len(args) > 0 and isinstance(args[0], np.ndarray):
            kw = dict(kwargs)
            kw['bytes'] = False
            kw['return_data'] = True
            kw['filter_output'] = False
            rgba, data = self._color.__call__(*args, **kw)
            retval = self._return(
                rgba.reshape((-1,4)), args[0].shape, data,
                **kwargs)
            if kwargs.get('return_data', False):
                retval = (retval, data)
            return retval

        # call for set of colors
        return super().__call__(*args, **kwargs)

    def __getitem__(self, index):
        return self._index_filter(self._color.__getitem__(index), index)

#-----------------------------------------------------------------------

class ColorFilterColor(ColorFilter):
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

        background = kw.pop('background', '#FFFFFF')
        assert isinstance(background, str)
        col = color(background)
        if col is None:
            col = background
        try:
            background = rgb(col, no_alpha = True)
        except:
            pass
        assert background.shape == (3,)
        self._background = background

        self._method = kw.pop('method', 'vec')
        self._clip = kw.pop('clip', True)

        super().__init__(*args, **kw)

    def _filter(self, rgba):
        """
        take rgba input, filter, replace data in place, return modified array
        """
        if self._method == 'vec':
            fc = np.sqrt(np.sum((rgba[:, :3] * self._xcolor[np.newaxis, :])**2,
                               axis = 1) /
                        np.sum(self._xcolor**2))
        elif self._method == 'min':
            jj = self._xcolor > 0
            c = self._xcolor[np.newaxis, jj]
            jj = np.append(jj, np.tile(False, 4 - len(jj)))
            fc = np.min(rgba[:, jj] / c, axis = 1) * np.max(c)
        else:
            raise ValueError('Method "{}" not supported'.format(self._method))
        fb = 1 - fc
        rgba[:, :3] = (fc[:, np.newaxis] * self._xcolor[np.newaxis, :] +
                       fb[:, np.newaxis] * self._background[np.newaxis, :])
        if self._clip:
            rgba[:, :3] = np.clip(rgba[:, :3], 0, 1)
        return rgba


#-----------------------------------------------------------------------
#    Color blind filters
#
#    see also
#    http://vision.psychol.cam.ac.uk/jdmollon/papers/colourmaps.pdf

class FilterColorBlindGreen(ColorFilter):
    """
    Simulate Green Blindness.  Done in sRGB space with gamma = 2.2

    http://www.sron.nl/~pault/colourschemes.pdf
    """
    def _filter(self, rgba):
        gamma = 2.2
        r = rgba[:, 0]**gamma
        g = rgba[:, 1]**gamma
        b = rgba[:, 2]**gamma
        RG = (0.02138 + 0.677 * g + 0.2802 * r)**(1 / gamma)
        B = (0.02138 + + 0.95724 * b + 0.02138 * g - 0.012138 * r)**(1 / gamma)
        rgb = np.array([RG, RG, B]).transpose()
        np.clip(rgb, 0, 1, out = rgb)
        rgba[:,  :3] = rgb
        return rgba

class FilterColorBlindRed(ColorFilter):
    """
    Simulate Red Blindness.  Done in sRGB space with gamma = 2.2.

    http://www.sron.nl/~pault/colourschemes.pdf
    """
    def _filter(self, rgba):
        gamma = 2.2
        r = rgba[:, 0]**gamma
        g = rgba[:, 1]**gamma
        b = rgba[:, 2]**gamma
        RG = (0.003974 + 0.8806 * g + 0.1115 * r)**(1 / gamma)
        B = (0.003974 + 0.992052 * b - 0.003974 * g + 0.003974 * r)**(1 / gamma)
        rgb = np.array([RG, RG, B]).transpose()
        np.clip(rgb, 0, 1, out = rgb)
        rgba[:,  :3] = rgb
        return rgba

class FilterColorGraySimple(ColorFilter):
    """
    gray color filter
    """
    def _filter(self, rgba):
        weights = np.array([0.2126, 0.7152, 0.0722])
        weights = np.array([1, 1, 1]) / 3
        Y = np.dot(rgba[:, :3], weights)
        rgb = np.tile(Y, (3,1)).transpose()
        np.clip(rgb, 0, 1, out = rgb)
        rgba[:,  :3] = rgb
        return rgba

class FilterColorGray(ColorFilter):
    """
    gray color filter
    """
    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        self._gamma = kw.pop('gamma', 563/256) # approx 2.2
        super().__init__(*args, **kw)
    def _filter(self, rgba):
        weights = np.array([0.2126, 0.7152, 0.0722])
        Y = np.dot(rgba[:, :3]**self._gamma, weights)**(1 / self._gamma)
        np.clip(Y, 0, 1, out = Y)
        rgba[:,  :3] = Y[:, np.newaxis]
        return rgba

class FilterVisCheck(ColorFilter):
    """
    Color filter from vision check java class, Vischeck.class

    Supported modes are:
        'deuteranope'
        'protanope'
        'tritanope'

    vischeck.com
    http://www.vischeck.com/downloads/vischeckJ/VischeckJ1.zip
    """
    def __init__(self,
                 color,
                 mode = 'deuteranope',
                 gamma = None,
                 **kwargs):
        kw = dict(kwargs)
        assert mode in self._projections
        self._mode = mode
        if gamma is None:
            gamma = 2.2
        if not is_iterable(gamma):
            gamma = np.tile(gamma, 3)
        if not isinstance(gamma, np.ndarray):
            gamma = np.array(gamma)
        assert len(gamma)  == 3
        self._gamma = gamma
        super().__init__(color, **kw)

    _matrix = np.array([
        [0.05059983, 0.08585369, 0.00952420],
        [0.01893033, 0.08925308, 0.01370054],
        [0.00292202, 0.00975732, 0.07145979]])
    _matrixI = np.linalg.inv(_matrix)

    _I = np.sum(np.array(_matrix), axis = 1)

    _v0 = np.array([0.9856 , 0.7325  , 0.001079])
    _v1 = np.array([0.08008, 0.15790 , 0.5897  ])
    _v2 = np.array([0.0914 , 0.007009, 0.0     ])
    _v3 = np.array([0.1284 , 0.2237  , 0.3636  ])

    _v0 = np.cross(_I, _v0)
    _v1 = np.cross(_I, _v1)
    _v2 = np.cross(_I, _v2)
    _v3 = np.cross(_I, _v3)

    _v0d = -_v0/_v0[1]
    _v0d[1] = 0
    _v1d = -_v1/_v1[1]
    _v1d[1] = 0

    _v0p = -_v0/_v0[0]
    _v0p[0] = 0
    _v1p = -_v1/_v1[0]
    _v1p[0] = 0

    _v2t = -_v2/_v2[2]
    _v2t[2] = 0
    _v3t = -_v3/_v3[2]
    _v3t[2] = 0

    _dd = (1, _I[2] / _I[0], _v0d, _v1d)
    _dp = (0, _I[2] / _I[1], _v0p, _v1p)
    _dt = (2, _I[1] / _I[0], _v2t, _v3t)

    _projections = {
        'deuteranope': _dd,
        'protanope'  : _dp,
        'tritanope'  : _dt,
        }

    def _project(self, rgb, i, lim, v0, v1):
        i1, i2 = ((i +  np.array([1,2])) % 3).tolist()
        ii = rgb[:, i2] < lim * rgb[:, i1]
        rgb[ii, i] = np.inner(rgb[ii, :], v0)
        ii = np.logical_not(ii)
        rgb[ii, i] = np.inner(rgb[ii, :], v1)
        return rgb

    def _filter(self, rgba, **kwargs):
        rgb = rgba[:, :3] ** self._gamma[np.newaxis, :]
        rgb = np.inner(rgb, self._matrix)
        self._project(rgb, *self._projections[self._mode])
        rgb = np.inner(rgb, self._matrixI)
        np.clip(rgb, 0, 1, out = rgb)
        rgba[:, :3] =  rgb ** (1 / self._gamma[np.newaxis, :])
        return rgba


class FilterColorBlindness(ColorFilter):
    """
    Color Blindness Filter

    http://mudcu.be/sphere/js/Color.Blind.js
    https://github.com/skratchdot/color-blind/tree/master/lib
    """

    def __init__(self, color, mode = 'protan', profile = None, anomalize = False, **kwargs):
        self._mode = mode
        self._profile = profile
        self._anomalize = anomalize
        super().__init__(color, **kwargs)

    _blinder = dict(
        protan = dict(
            x  =  0.7465,
            y  =  0.2535,
            m  =  1.273463,
            yi = -0.073894,
            ),
        deutan = dict(
            x  =  1.4,
            y  = -0.4,
            m  =  0.968437,
            yi =  0.003331,
            ),
        tritan = dict(
            x  = 0.1748,
            y  = 0,
            m  = 0.062921,
            yi = 0.292119,
            ),
        custom = dict(
            x  = 0.735,
            y  = 0.265,
            m  = -1.059259,
            yi = 1.026914
            ),
        )

    _achorma_weights = np.array([0.212656, 0.715158, 0.072186])
    _achorma_v = 1.75
    _achorma_n = _achorma_v + 1
    _gamma = 563 / 256

    def _filter(self, rgba, **kwargs):
        # transform to linear RGB
        rgb = rgba[:, :3]
        if self._profile == 'sRGB':
            rgb = ColorsRGB._s(rgb)
        elif self._profile == 'gamma':
            rgb = rgb ** self._gamma
        # apply blinding transfrmations
        if self._mode == 'achroma':
            # D65 in sRGB
            z = np.inner(rgba[:, :3], self._achorma_weights)
            rgb = np.tile(z, (3,1)).transpose()
        else:
            xyY = ColorxyYInverse()(rgb)
            lx,ly,lm,lyi = [self._blinder.get(self._mode, 'custom')[i] for i in ('x','y','m','yi')]
            # The confusion line is between the source color and the confusion point
            slope = (xyY[:, 1] - ly) / (xyY[:, 0] - lx)
            # slope, and y-intercept (at x=0)
            yi = xyY[:, 1] - xyY[:, 0] * slope
            # Find the change in the x and y dimensions (no Y change)
            dx = (lyi - yi) / (slope - lm)
            dy = (slope * dx) + yi
            # Find the simulated colors XYZ coords
            zXYZ = np.empty_like(rgb)
            zXYZ[:, 0] = dx * xyY[:, 2] / dy
            zXYZ[:, 1] = xyY[:, 2]
            zXYZ[:, 2] = (1 - (dx + dy)) * xyY[:, 2] / dy
            # Calculate difference between sim color and neutral color
            # find neutral grey using D65 white-point
            ngx = 0.312713 * xyY[:, 2] / 0.329016
            ngz = 0.358271 * xyY[:, 2] / 0.329016
            dXYZ = np.zeros_like(rgb)
            dXYZ[:, 0] = ngx - zXYZ[:, 0]
            dXYZ[:, 2] = ngz - zXYZ[:, 2]
            # find out how much to shift sim color toward neutral to fit in RGB space
            # convert d to linear RGB
            dRGB = ColorXYZ()(dXYZ, clip = False)
            dRGB[np.argwhere(dRGB == 0)] = 1.e-10
            rgb = ColorXYZ()(zXYZ, clip = False)
            _rgb = (np.choose(rgb < 0, (1, 0)) - rgb) / dRGB
            _rgb = np.choose((_rgb > 1) | (_rgb < 0), (_rgb, 0))
            adjust = np.amax(_rgb, axis=1)
            rgb += adjust[:, np.newaxis] * dRGB
        # anomalize
        if self._anomalize:
            rgb[:,:] = (self._achorma_v * rgb + rgba[:, :3]) / self._achorma_n
        # transform back to compressed/non-linerar space
        if self._profile == 'sRGB':
            rgb = ColorsRGBInverse._s(rgb[:,:3])
        elif self._profile == 'gamma':
            rgb = rgba[:,:3] ** (1 / self._gamma)

        rgba[:, :3] = rgb
        return rgba

#######################################################################

class FilterColorInvert(ColorFilter):
    """
    invert colors
    """
    def _filter(self, rgba):
        rgba[:, 0:3] = 1 - rgba[:,  :3]
        return rgba

class FilterColorHueRotate(ColorFilter):
    """
    rotate hue by angle in counterclockwise direction

    provide color model and angle, requires keyword arguments

    Can provide index for non-cmmpatible color model, but this is not
    recommended.  The routine determines correct index from the
    'limits' field of the color model (np.ndarray((3,2))) as the index
    wuth an upper limit of 360.  Use static method valid_models() to
    get a list of all supported color models.  Default model is HCL;
    LChuv also seems to give good results.
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
        super().__init__(*args, **kw)

    def _filter(self, rgba):
        ccca = self._model.inverse()(rgba)
        ccca[:, self._index] += self._angle
        return self._model(ccca)

    @staticmethod
    def valid_models():
        """
        return list of valid color models for rotation
        """
        return [n for n,m in _color_models.items()
                if len(np.where(m.limits == 360)[0]) == 1]

class FilterColorGraysRGB0(ColorFilter):
    """
    Filter to gray using sRGB space, return linear gray values.
    """
    def _filter(self, rgba):
        weights = np.array([0.2126, 0.7152, 0.0722])
        gamma = 2.4
        rgb = rgba[:, 0:3]
        ii = rgb <= 0.04045
        rgb[ii] /= 12.92
        ii = np.logical_not(ii)
        rgb[ii] = ((rgb[ii] + 0.055) / 1.055)**gamma
        Y = np.dot(rgb, weights)
        ii = Y <= 0.0031308
        Y[ii] *= 12.92
        ii = np.logical_not(ii)
        Y[ii] = 1.055 * Y[ii]**(1 / gamma) - 0.055
        rgb = np.tile(Y, (3,1)).transpose()
        np.clip(rgb, 0, 1, out = rgb)
        rgba[:, 0:3] = rgb
        return rgba

class FilterColorGraysRGB(ColorFilter):
    """
    Filter to gray using sRGB space, return linear gray values.
    """
    _weights = np.array([0.2126, 0.7152, 0.0722])
    def _filter(self, rgba):
        rgb = ColorsRGBInverse._s(rgba[:, :3])
        Y = np.dot(rgb, self._weights)
        Y = ColorsRGB._s(Y)
        np.clip(Y, 0, 1, out = Y)
        rgba[:, 0:3] = Y[:, np.newaxis]
        return rgba

class FilterColorGamma(ColorFilter):
    """
    gamma color filter
    """
    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        gamma = kw.pop('gamma', 1.0)
        if not is_iterable(gamma):
            gamma = np.tile(gamma, 3)
        else:
            gamma = np.array(gamma)
        assert len(gamma.shape) == 1
        assert gamma.shape[0] in (3, 4)
        self._gamma = gamma
        super().__init__(*args, **kw)

    def _filter(self, rgba):
        gamma = self._gamma.copy()
        ii = np.where(self._gamma < 0)
        rgba[:, ii] = 1 - rgba[:, ii]
        gamma[ii] = - self._gamma[ii]
        i = slice(0, len(gamma))
        rgba[:, i] **= gamma[np.newaxis, :]
        rgba[:, ii] = 1 - rgba[:, ii]
        return rgba

class FilterColorModel(ColorFilter):
    """
    transform output to new Color Model
    """
    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        model = kw.pop('model', 'RGB')
        if isinstance(model, str):
            model = _color_models[model]
        assert isinstance(model, ColorModel)
        self._color_model = model
        super().__init__(*args, **kw)

    def _filter(self, rgba):
        return self._color_model(rgba)

class FilterColorTransparent(ColorFilter):
    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        transparent = kw.pop('transparent', '#ffffff')
        transparent = iterable(transparent)
        assert len(transparent) in (1,2)
        if len(transparent) == 1:
            transparent = np.tile(transparent, 2)
        else:
            transparent = np.array(transparent)
        tt = []
        for t in transparent.reshape(-1):
            assert isinstance(t, str)
            col = color(t)
            if col is None:
                col = t
            try:
                t = rgb(col, no_alpha = True)
            except:
                pass
            assert t.shape == (3,)
            tt += [t]
        self._transparent = tt
        super().__init__(*args, **kw)

    # unfortunately pcolorfast alpha is broken (mpl 1.5 beta)
    def _filter(self, rgba):
        ii = np.alltrue(np.logical_and(
            rgba[:, :3] >= self._transparent[0][np.newaxis, :],
            rgba[:, :3] <= self._transparent[1][np.newaxis, :]),
            axis = 1)
        rgba[ii,3] = 0
        return rgba
