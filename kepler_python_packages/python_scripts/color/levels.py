import numpy as np

from .functions import Color, register_color, colormap
from .models import _color_models
from .utils import rgb2hex, rgb
from ._utils import Slice


class LevelData(Color):
    """
    base class to provide set of colors for line plots or level-based
    color functions

    TODO - allow output a as float tuples, maybe make default?

    TODO - integrte color filtering for final output.
    """
    _hexcols = np.array([])
    _xarr = None
    _missing = None
    _col_func = None
    def __init__(self, n = None, **kwargs):
        if n is None:
            if self._xarr is not None:
                assert len(self._xarr) <= len(self._hexcols)
                n = len(self._xarr)
            else:
                n = len(self._hexcols)
        self._n = n
        if self._xarr is not None:
            nx = len(self._xarr)
        else:
            nx = len(self._hexcols)
        if n > nx:
            if self._col_func is not None:
                self._xarr = None
                c = IsoColors(self._col_func, n, **kwargs)
                self._hexcols = c._hexcols.copy()
            else:
                if self._xarr is not None:
                    self._hexcols = self._hexcols[self._xarr[nx - 1]]
                    self._xarr = None
                self._hexcols = self._hexcols[np.arange(n) % nx]
        super().__init__(**kwargs)
    def _function(self, data, *args, **kwargs):
        values = self.__call__()
        values = rgb(values)
        index = np.minimum((data * len(values)).astype(int), len(values) - 1)
        out = np.choose(index.reshape(-1,1), values.reshape(-1, 3))
        return _color_models['RGB'](out)
    def __call__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], np.ndarray):
            return super().__call__(*args, **kwargs)
        # TODO - deal with len(args) > 1 and iterable args[0]
        none = object()
        n = kwargs.get('n', none)
        if n is none:
            if len(args) > 0:
                n = args[0]
                assert isinstance(n, int), 'require integer argument'
            else:
                n = None
        if n is None:
            n = self._n
        assert 0 < n <= self._n, (
            'no more than {} colors'.format(self._n))
        return self[:n]
    def _get(self, index):
        if self._xarr is None:
            colors = self._hexcols[index]
        else:
            colors = self._hexcols[self._xarr[self._n - 1][index]]
        return self._index_filter(colors, index)
    def __getitem__(self, index):
        # TODO - add iterable index lists, e.g., multi-D np arrays?
        if isinstance(index, slice):
            return [self._get(j) for j in Slice(index, size = self._n)]
        else:
            return self._get(index)
    def __len__(self):
        return self._n

    def _index_filter(self, colors, index):
        if self._n > 1:
            scale = 1 / (self._n - 1)
        else:
            scale = 1
        if isinstance(colors, str):
            return self._hexfilter(colors, index * scale)
        return np.array([
            self._hexfilter(c, i * scale)
            for c, i in zip(colors, index)])
    def _hexfilter(self, h, x):
        if self._filter is None and self._xfilter is None:
            return h
        assert isinstance(h, str)
        rgba = rgb(h, alpha = 1.).reshape((1,4))

        if self._xfilter is not None:
            x = np.array(x).reshape((1,))
            rgba = self._xfilter(rgba, x)
        if self._filter is not None:
            rgba = self._filter(rgba).reshape(4)
        h = rgb2hex(rgba.reshape(4))
        return h

class Reverse(LevelData):
    def __init__(self, *args, **kwargs):
        self._color = args[0]
        super().__init__(*(args[1:]), **kwargs)
    def __call__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], np.ndarray):
            x = args[0]
            ii = np.logical_and(0 <= x, x <= 1)
            x[ii] = 1 - x[ii]
            args = (x,) + args[1:]
            return self._color(*args, **kwargs)
        return self._color(*args, **kwargs)[::-1]
    def __getitem__(self, *args):
        x = self._color()[::-1]
        return x.__getitem__(*args)
    def __len__(self):
        return self._color.len()

class IsoColorBlind(LevelData):
    """
    Colour-blind proof distinct colours module, based on work by Paul Tol
    Pieter van der Meer, 2011
    SRON - Netherlands Institute for Space Research
    """
    _hexcols = np.array([
        '#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
        '#DDCC77', '#CC6677', '#882255', '#AA4499', '#661100',
        '#6699CC', '#AA4466', '#4477AA',
        ])
    _xarr = [
        [12],
        [12, 6],
        [12, 6, 5],
        [12, 6, 5, 3],
        [0, 1, 3, 5, 6],
        [0, 1, 3, 5, 6, 8],
        [0, 1, 2, 3, 5, 6, 8],
        [0, 1, 2, 3, 4, 5, 6, 8],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 1, 2, 3, 4, 5, 9, 6, 7, 8],
        [0, 10, 1, 2, 3, 4, 5, 9, 6, 7, 8],
        [0, 10, 1, 2, 3, 4, 5, 9, 6, 11, 7, 8],
        ]

register_color('IsoBlind', IsoColorBlind)

class IsoColorBlind4(LevelData):
    """
    Colour-blind proof distinct colours module,
    based on Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl)
    """
    _hexcols = np.array([
        '#332288', '#88CCEE', '#999933', '#AA4499', '#4477AA',
        ])

register_color('IsoBlind4', IsoColorBlind4)

class IsoColorBlind8(LevelData):
    """
    Colour-blind proof distinct colours module,
    based on Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl)
    """
    _hexcols = np.array([
        '#332288', '#88CCEE', '#117733', '#DDCC77', '#CC6677',
        '#AA4499', '#44AA99', '#882255',
        ])

register_color('IsoBlind8', IsoColorBlind8)

class IsoGraySave(LevelData):
    """
    Colour-blind proof and gray-save distinct colors module,
       based on work by Paul Tol
    Pieter van der Meer, 2011
    SRON - Netherlands Institute for Space Research
    """
    _hexcols = np.array([
        '#809BC8', '#FF6666', '#FFCC66', '#64C204',
        ])
    _xarr = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 1, 2, 3],
        ]
register_color('IsoGraySave', IsoGraySave)

class IsoColorAlt(LevelData):
    """
    Colour-blind proof colors scheme,
       based on work by Paul Tol
    SRON - Netherlands Institute for Space Research
    """
    _missing = '#DDDDDD'
    _hexcols = np.array([
        '#3366AA', '#11AA99', '#66AA55', '#CCCC55', '#777777',
        '#FFEE33', '#EE7722', '#EE3333', '#992288',
        ])

register_color('IsoAlt', IsoColorAlt)

class IsoColorRainbow(LevelData):
    """
    Colour-blind proof colors scheme,
       based on work by Paul Tol
    SRON - Netherlands Institute for Space Research
    """
    _missing = '#EEEEEE'
    _hexcols = np.array([
        '#781C81', '#3F4EA1', '#4683C1', '#57A3AD', '#6DB388',
        '#B1BE4E', '#DFA53A', '#E7742F', '#D92120',
        ])

register_color('IsoRainbow', IsoColorRainbow)

class IsoColorRainbow2(LevelData):
    """
    Colour-blind proof colors scheme,
       based on work by Paul Tol
    SRON - Netherlands Institute for Space Research
    """
    _missing = '#777777'
    _hexcols = np.array([
        '#BB2E72', '#B178A6', '#D6C1DE', '#1965B0', '#5289C7',
        '#78AFDE', '#4EB265', '#90C987', '#CAEDA8', '#F7EE55',
        '#F6C141', '#F1932D', '#E8601C', '#DC050C',
        ])

register_color('IsoRainbow2', IsoColorRainbow2)

class IsoColorDivergeBR(LevelData):
    """
    Colour-blind proof colors scheme,
       based on work by Paul Tol
    SRON - Netherlands Institute for Space Research
    """
    _missing = '#FFEE99'
    _hexcols = np.array([
        '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
        '#FDDBC7', '#F4A582', '#D6604D', '#B2182B',
        ])

register_color('IsoDivergeBR', IsoColorDivergeBR)

class IsoColorDivergePG(LevelData):
    """
    Colour-blind proof colors scheme,
       based on work by Paul Tol
    SRON - Netherlands Institute for Space Research
    """
    _missing = '#FFEE99'
    _hexcols = np.array([
        '#762A83', '#9970AB', '#C2A5CF', '#E7D4EB', '#F7F7F7',
        '#D9FDD3', '#ACD39E', '#5AAE61', '#1B7837',
        ])

register_color('IsoDivergePG', IsoColorDivergePG)

class IsoColorLight(LevelData):
    """
    Light colors for marking text.

    Colour-blind proof colors scheme,
       based on work by Paul Tol
    SRON - Netherlands Institute for Space Research
    """
    _hexcols = np.array([
        '#BBCCEE', '#CCEEFF', '#CCDDAA', '#EEEEBB', '#FFCCCC'
        ])

register_color('IsoLight', IsoColorLight)

class IsoColorDark(LevelData):
    """
    Dark colors for text.

    Colour-blind proof colors scheme,
       based on work by Paul Tol
    SRON - Netherlands Institute for Space Research
    """
    _hexcols = np.array([
        '#222255', '#225555', '#225522', '#666633', '#663333'
        ])

    # TODO: add _xarr

register_color('IsoDark', IsoColorDark)

class IsoColorRainbowLight(LevelData):
    """
    Light rainbow colors

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """

    _hexcols = np.array([
        '#77AADD', '#77CCCC', '#88CCAA', '#DDDD77', '#DDAA77',
        '#DD7788', '#CC99BB',
        ])

register_color('IsoRainbowLight', IsoColorRainbowLight)

class IsoColorRainbowMedium(LevelData):
    """
    Medium rainbow colors

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """

    _hexcols = np.array([
        '#4477AA', '#44AAAA', '#44AA77', '#AAAA44', '#AA7744',
        '#AA4455', '#AA4488',
        ])

register_color('IsoRainbowMedium', IsoColorRainbowMedium)

class IsoColorRainbowDark(LevelData):
    """
    Dark rainbow colors

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """
    _hexcols = np.array([
        '#114477','#117777', '#117744', '#777711', '#774411',
        '#771122', '#771155',
        ])

register_color('IsoRainbowDark', IsoColorRainbowDark)

class IsoColorPPT(LevelData):
    """
    Powerpoint Colors for dark background

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """
    _hexcols = np.array([
        '#FFFFFF', '#FFFFCC', '#FFCC66', '#809BC8', '#64C204',
        '#FF6666',
        ])
    _missing = '#424242'

register_color('IsoPPT', IsoColorPPT)


class IsoColorSequenceEarth(LevelData):
    """
    Sequence Data Earth Colors

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """
    _hexcols = np.array([
        '#FFFFE5', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29',
        '#EC7014', '#CC4C02', '#993404', '#662506', '#8C2D04',
        '#FFFBD5', '#D95F0E', '#FED98E',
        ])
    _xarr = [
        [ 4],
        [ 2, 6],
        [ 1, 3,11],
        [10,12, 4, 6],
        [10,12, 4,11, 7],
        [10, 2, 3, 4,11, 7],
        [10, 2, 3, 4, 5, 6, 9],
        [ 0, 1, 2, 3, 4, 5, 6, 9],
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8],
        ]
    _col_func = 'BlindWR'

register_color('IsoSequenceEarth', IsoColorSequenceEarth)


class IsoColorDivergeBWR(LevelData):
    """
    Diverging Data Blue-White-Red

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """
    _hexcols = np.array([
        '#3D52A1', '#3A89C9', '#77B7E5', '#B4DDF7', '#E6F5FE',
        '#FFFAD2', '#FFE3AA', '#F9BD7E', '#ED875E', '#D24D3E',
        '#AE1C3E', '#99C7EC', '#F5A275', '#008BCE', '#D03232',
        ])
    _xarr = [
        [ 5],
        [11,12],
        [11, 5,12],
        [13, 3, 7, 14],
        [13, 3, 5, 7,14],
        [ 1,11, 4, 6,12, 9],
        [ 1,11, 4, 5, 6,12, 9],
        [ 1, 2, 3, 4, 6, 7, 8, 9],
        [ 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [ 0, 1, 2, 3, 4, 6, 7, 8, 9,10],
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],
        ]
    _col_func = 'BlindBWR'

register_color('IsoDivergeBWR', IsoColorDivergeBWR)

class IsoColorRainbow(LevelData):
    """
    Rainbow data

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """
    _hexcols = np.array([
        '#781C81', '#413B93', '#4065B1', '#488BC2', '#55A1B1',
        '#63AD99', '#7FB972', '#B5BD4C', '#D9AD3C', '#E68E34',
        '#E6642C', '#D92120', '#404096', '#416CB7', '#4D95BE',
        '#5BA7A7', '#6EB387', '#A1BE56', '#D3B33F', '#E59435',
        '#E6682D', '#3F479B', '#4277BD', '#529DB7', '#62AC9B',
        '#86BB6A', '#C7B944', '#E39C37', '#E76D2E', '#3F4EA1',
        '#4683C1', '#57A3AD', '#6DB388', '#B1BE4E', '#DFA53A',
        '#E7742F', '#3F56A7', '#4B91C0', '#5FAA9F', '#91BD61',
        '#D8AF3D', '#E77C30', '#3F60AE', '#539EB6', '#CAB843',
        '#E78532', '#404096', '#498CC2', '#BEBC48', '#E68B33',
        '#7DB874', '#57A3AD', '#DEA73A',
        ])
    _xarr = [
        [11],
        [46,11],
        [46,50,11],
        [46,51,52,11],
        [46,23,50,27,11],
        [46,47, 5,48,49,11],
        [ 0,42,43,32,44,45,11],
        [ 0,36,37,38,39,40,41,11],
        [ 0,29,30,31,32,33,34,35,11],
        [ 0,21,22,23,24,25,26,27,28,11],
        [ 0,12,13,14,15,16,17,18,19,20,11],
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11],
        ]
    _col_func = 'BlindRainbow'

# register_color('IsoDivergeBWR', IsoColorDivergeBWR)

class IsoColorRainbow14(LevelData):
    """
    Banded Rainbow Scheme

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """
    _hexcols = np.array([
        '#882E72', '#B178A6', '#D6C1DE', '#1965B0', '#5289C7',
        '#7BAFDE', '#4EB265', '#90C987', '#CAE0AB', '#F7EE55',
        '#F6C141', '#F1932D', '#E8601C', '#DC050C',
        ])

register_color('IsoRainbow14', IsoColorRainbow14)

class IsoColorRainbow15(LevelData):
    """
    Banded Rainbow Scheme

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """
    _hexcols = np.array([
        '#114477', '#4477AA', '#77AADD', '#117755', '#44AA88',
        '#99CCBB', '#777711', '#AAAA44', '#DDDD77', '#771111',
        '#AA4444', '#DD7777', '#771144', '#AA4477', '#DD77AA',
        ])

register_color('IsoRainbow15', IsoColorRainbow15)

class IsoColorRainbow18(LevelData):
    """
    Banded Rainbow Scheme

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """
    _hexcols = np.array([
        '#771155', '#AA4488', '#CC99BB', '#114477', '#4477AA',
        '#77AADD', '#117777', '#44AAAA', '#77CCCC', '#777711',
        '#AAAA44', '#DDDD77', '#774411', '#AA7744', '#DDAA77',
        '#771122', '#AA4455', '#DD7788',
        ])

register_color('IsoRainbow18', IsoColorRainbow18)

class IsoColorRainbow21(LevelData):
    """
    Banded Rainbow Scheme

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """
    _hexcols = np.array([
        '#771155', '#AA4488', '#CC99BB', '#114477', '#4477AA',
        '#77AADD', '#117777', '#44AAAA', '#77CCCC', '#117744',
        '#44AA77', '#88CCAA', '#777711', '#AAAA44', '#DDDD77',
        '#774411', '#AA7744', '#DDAA77', '#771122', '#AA4455',
        '#DD7788',
        ])

register_color('IsoRainbow21', IsoColorRainbow21)

#-----------------------------------------------------------------------

class IsoColorRainbowMulti(LevelData):
    """
    multiple shades of rainbow colors

    http://www.sron.nl/~pault/colourschemes.pdf
    Doc. no. : SRON/EPS/TN/09-002
    Issue : 2.2
    Date : 29 December 2012
    """
    colors = (IsoColorRainbowLight, IsoColorRainbowMedium, IsoColorRainbowDark)
    _hexcols = np.concatenate([c._hexcols for c in colors])

########################################################################

class IsoColors(LevelData):
    """
    create line colors or level data based on color function

    color can be name or class
    """
    def __init__(self, color = None, n = 7, endpoint = None, **kwargs):
        # TODO: add *args
        if isinstance(color, str):
            color = colormap(color, **kwargs)
        elif isinstance(color, type):
            color = color(**kwargs)
        assert isinstance(color, Color)
        if endpoint is None:
            c0 = color(0)
            c1 = color(1)
            dc = np.sum((c1-c0)**2)
            endpoint = dc > 2 / n
        self._hexcols = np.array([rgb2hex(color(np.array(x))[0:3])
                                  for x in np.linspace(0, 1, n, endpoint = endpoint)])
        super().__init__(**kwargs)

class IsoColorLights(LevelData):
    """
    create line colors or level data based on color function
    For each color, add different levels of lightness.

    color can be name or class
    """
    def __init__(self,
                 color = None,
                 n = 7,
                 lights = 3,
                 maxlight = 0.8,
                 minlight = 0.5,
                 collate = False,
                 endpoint = None,
                 **kwargs):
        # TODO: add *args
        if isinstance(color, str):
            color = colormap(color, **kwargs)
        elif isinstance(color, type):
            color = color(**kwargs)
        assert isinstance(color, Color)
        if endpoint is None:
            c0 = color(0)
            c1 = color(1)
            dc = np.sum((c1-c0)**2)
            endpoint = dc > 2 / n
        cols = np.array([color(np.array(x))
                         for x in np.linspace(0, 1, n, endpoint = endpoint)])
        h,s,l = _color_models['HSL'].inverse(cols[:,0], cols[:,1], cols[:,2])
        a = cols[:,3]
        if isinstance(lights, (int, float)):
            lights = np.linspace(minlight, maxlight / max(l), lights, endpoint = True)
        assert isinstance(lights, np.ndarray)
        h = np.repeat(h, len(lights))
        s = np.repeat(s, len(lights))
        a = np.repeat(a, len(lights))
        l = np.repeat(l, len(lights)) * np.tile(lights, len(l))
        r,g,b = _color_models['HSL'](h,s,l)
        cols = np.array([r,g,b,a]).transpose()
        cols = np.array([rgb2hex(c[0:3]) for c in cols])
        if not collate:
            cols = cols.reshape((-1, len(lights))).transpose().reshape(-1)
        self._hexcols = cols
        super().__init__(**kwargs)
