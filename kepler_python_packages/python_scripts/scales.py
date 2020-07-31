"""
Define some scales for stellar evolution application
"""

import math
import numpy as np

from matplotlib import rcParams

from matplotlib.scale import register_scale, LinearScale
from matplotlib.transforms import Transform

from matplotlib.ticker import AutoLocator, ScalarFormatter, Formatter, Locator
from matplotlib.ticker import NullLocator, NullFormatter, FixedLocator, FixedFormatter

import human.time as th
from human import time2human

def locfunction(vmin, vmax, scale, offset):
    """
    Locator function for range with scale and offset.

    Parameters
    ----------
    vmin : scalar
        Minimum value of range.
    vmax : scalar
        Maximum value of range.
    scale : scalar
        Scale factor.
    offset : scalar
        Range offset factor.

    Returns
    -------
    locs : ndarray
        Locations of ticks.

    Notes
    -----
    `scale` is applied after subtraction of `offset`.

    Examples
    --------
    >>> import scales
    >>> scales.locfunction(1e3, 1.02e3, 10, 1.e3)
    array([ 1000.,  1010.,  1020.])
    """
    return (np.arange(np.ceil((vmin - offset) / scale),
                      np.floor((vmax - offset) / scale) + 1)
            * scale + offset)

class TimeLocator(Locator):
    divs = np.array([   1,   2,   5,
                        10,  20,  50,
                        100, 200, 500,
                        1000])
    def __init__(self,
                 n = 6,
                 same_units = False,
                 align = 'left',
                 comma = True,
                 **kwargs):
        # in principle, kwargs should be empty ...
        super().__init__()
        self._n = n
        self._offset = 0
        self._unit = None
        self._offset_str = ''
        self._same_units = same_units
        self._align = align
        self._comma = True

    def tick_values(self, vmin, vmax):
        """
        Return the values of the located ticks given **vmin** and **vmax**.

        .. note::
            To get tick locations with the vmin and vmax values defined
            automatically for the associated :attr:`axis` simply call
            the Locator instance::

                >>> print((type(loc)))
                <type 'Locator'>
                >>> print((loc()))
                [1, 2, 3, 4]

        """
        vmin, vmax = sorted((vmin, vmax))
        d = vmax - vmin

        # compute scale
        scale = time2human(d, dec_lim = 0.1 * self._n)
        v, unit = th.split_unit(scale, num_val = True)
        u = th.unit2scale(unit)
        f = 1
        while round(v) != v or v < 10:
            v *= 10
            f /= 10
        m = 1
        for x in self.divs:
            if v / x < self._n:
                m = x
                break
        scalev = m * f
        scale = scalev * u
        oos = np.floor(np.log10(abs(scalev)))
        oov = np.floor(np.log10(max(np.abs([vmin, vmax])) / u))

        # compute offset
        if self._same_units in (True, 'strict'):
            # case 1
            # will use same base unit for offset as for scale

            oo = 10**min(3, np.ceil(np.log10(abs(scalev))) + 1)
            if self._align == 'left':
                o = np.floor(vmax/(u * oo)) * oo * u
            elif self._align == 'right':
                o = np.ceil(vmin/(u * oo)) * oo * u
            elif self._align == 'center':
                o = np.round((vmin + vmax)/(2 * u * oo)) * oo * u
            else:
                raise ValueError('Invalid alignment: "{}"\nChoose one of left|center|right'.format(self._align))

            digits = 1
            while True:
                s = time2human(o, digits,
                               unit = unit,
                               rounding = True,
                               comma = self._comma,
                               unit_upgrade = self._same_units != 'strict')
                offset = th.human2time(s)
                if offset != o and abs(offset - o) > scale:
                    digits += 1
                else:
                    break
        else:
            # case 2
            # use most appropriate unit for scale
            if self._align == 'left':
                o = vmin
            elif self._align == 'right':
                o = vmax
            elif self._align == 'center':
                o = 0.5 * (vmin + vmax)
            else:
                raise ValueError('Invalid alignment: "{}"\nChoose one of left|center|right'.format(self._align))

            digits = 1
            while True:
                s = time2human(o, digits, rounding = True, comma = self._comma)
                offset = th.human2time(s)
                oot = np.floor(np.log10(max(np.abs(np.array([vmin, vmax]) - offset)) / u))
                if offset != o and (oot >= 3 or oot > oos + 1):
                    digits += 1
                else:
                    break

        if ((vmin * vmax <= 0) or ((oos + 1 >= oov) and (oov < 3))):
            offset = 0
            s = None

        locs = locfunction(vmin, vmax, scale, offset)

        self._offset = offset
        self._offset_str = s
        self._unit = unit
        self._u    = u
        self._locs = locs
        self._scale = scale
        self._oos = oos # order of mag of scale in units
        self._oov = oov # order of mag of absolute value in units

        return np.array(locs)

    def _get_offset(self):
        return self._offset, self._offset_str, self._unit

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)


class TimeFormatter(Formatter):
    # TODO - scientific format for large offsets
    _styles = {
        'div': {
            0 : '{var}',
            1 : '{var} {sign} {offs} {offu}',
            2 : '{var} / {unit}',
            3 : '( {var} {sign} {offs} {offu} ) / {unit}',
            4 : '{var} / {unit} {sign} {offs}',
            },
        'bra' : {
            0 : '{var}',
            1 : '{var} {sign} {offs} {offu}',
            2 : '{var} ({unit})',
            3 : '{var} {sign} {offs} {offu} ({unit})',
            4 : '{var} ({unit}) {sign} {offs}',
            },
        'ket' : {
            0 : '{var}',
            1 : '{var} {sign} {offs} {offu}',
            2 : '{var} ({unit})',
            3 : '({var} {sign} {offs} {offu}) ({unit})',
            4 : '{var} ({unit}) {sign} {offs}',
            },
        'sbra' : {
            0 : '{var}',
            1 : '{var} {sign} {offs} {offu}',
            2 : '{var} [{unit}]',
            3 : '{var} {sign} {offs} {offu} [{unit}]',
            4 : '{var} [{unit}] {sign} {offs}',
            },
        'sket' : {
            0 : '{var}',
            1 : '{var} {sign} {offs} {offu}',
            2 : '{var} [{unit}]',
            3 : '({var} {sign} {offs} {offu}) [{unit}]',
            4 : '{var} [{unit}] {sign} {offs}',
            },
        'in': {
            0 : '{var}',
            1 : '{var} {sign} {offs} {offu}',
            2 : '{var} in {unit}',
            3 : '( {var} {sign} {offs} {offu} ) in {unit}',
            4 : '{var} in {unit} {sign} {offs}',
            },
        }

    def __init__(self,
                 var = None,
                 tick_units = False,
                 same_units = False,
                 formats = None,
                 style = 'bra',
                 **kwargs):
        """
        var - name of variable
            (default: 'Time')
        tick_units - show units with tick labels rather than in label
            values: True | False
            default: False
        same_units - enforce same base unit for label and offset.
            values: True | False | 'strict'
            default: False
            if set to 'strict' then also no magnitide modifier is
            allowed (e.g., k, M, n).  This gives the mode pretty
            labels, but they may not be the most legible for all
            applications.

        formats (dict with keys 0-4) will be called with dict/keywords
         - var:  variable name
         - unit: variable unit
         - sign: sign of offset (negated as offset is subtracted)
         - offs: variable offset
         - offu: unit of offset (if independent from variable)

        You may also just overwrite some of the keys, the rest will
        then be picked from the style default

        There are five different kinds of formats, not all will be
        relevant depending on parameter settings.
         0: units on tick labels, no offset
         1: units on tick labels, with offset
         2: no unit on tick label, no offset
         3: no unit on tick label, with offset of different scale
         4: no unit on tick label, with offset of same scale

        style - select pre-defined style
         - div: use division for unit removal
         - bra: bracket for units (default)
         - ket: same as bra, but extra braket around values
        """
        if var is None:
            var = 'Time'
        self._var = var
        self._offset = 0
        self._unit = None
        self._tick_units = tick_units
        self._same_units = same_units

        self._formats = self._styles.get(style, dict()).copy()
        if formats is not None:
            self._formats.update(formats)

    def __call__(self, x, pos=None):
        """
        Return the format for tick value `x` at position pos.
        ``pos=None`` indicates an unspecified location.
        """
        t = x - self._offset
        if abs(t) < 1e-15 * abs(self._offset):
            t = 0
        if self._tick_units:
            return time2human(t)
        return th.split_unit(time2human(t, unit = self._unit, unit_upgrade = False))[0]

    def format_data(self, value):
        """
        Returns the full string representation of the value with the
        position unspecified.
        """
        return self.__call__(value)

    def format_data_short(self, value):
        """
        Return a short string version of the tick value.

        Defaults to the position-independent long value.
        """
        return self.format_data(value)

    def get_offset(self):
        """
        no offsets at awkward places for us any more
        """
        return ''

    def set_locs(self, locs):
        """
        set location and update axis label
        """
        self.locs = locs
        if isinstance(self.axis.major.locator, TimeLocator):
            o, s, u = self.axis.major.locator._get_offset()

        values = dict(
            var = self._var,
            unit = u,
            )

        if o != 0:
            ov, ou = th.split_unit(s)
            if o < 0:
                so = '+'
                ov = ov[1:]
            else:
                so = '-'
            values['offs'] = ov
            values['sign'] = so
            values['offu'] = ou

        if self._tick_units:
            if o == 0:
                case = 0
            else:
                case = 1
        else:
            if o == 0:
                case = 2
            else:
                if ou != u:
                    case = 3
                else: # self._same_units == 'strict'
                    case = 4

        label = self._formats[case].format(**values).strip()

        self._offset = o
        self._unit = u
        self.axis.set_label_text(label)

class TimeMinorLocator(Locator):
    """
    This minor locator relies (depends) on the presence of a TimeLocator as major locator
    """
    def __init__(self, **kwargs):
        """
        set parameters for minor locator

        minor_ticks: dictonary of scale values 1, 2, 5 to number of
        minor intervals if value provided, other wise value itself
        """

        # set number of ticks based on value
        self._ticks = kwargs.get('minor_ticks', {1: 10})

    def __call__(self):
        """
        return minor tick locations

        requires TimeLocator as axis major locator
        """
        locator = self.axis.major.locator
        assert isinstance(locator, TimeLocator)
        n0 = round(locator._scale / (locator._u * 10**locator._oos))
        # determin number of divisions based on value, which is in {1,2,5}
        n = self._ticks.get(n0, n0)
        scale = locator._scale / n
        offset = locator._offset
        vmin, vmax = sorted(self.axis.get_view_interval())

        locs = locfunction(vmin, vmax, scale, offset)

        # filter out major locs
        locs = [l for l in locs if np.min(np.abs(locator._locs - l)) > 0.1 * scale]

        return locs

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a {} type.'.format(type(self)))

class TimeScale(LinearScale):
    """
    Linear Time Scale
    """
    name = 'timescale'

    def __init__(self, axis, **kwargs):
        """
        """
        self.axis = axis
        self._kwargs = kwargs.copy()

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        symmetrical log scaling.
        """
        axis.set_major_locator(TimeLocator(**self._kwargs))
        axis.set_major_formatter(TimeFormatter(**self._kwargs))
        if self._kwargs.get('minor', True):
            minor_locator = TimeMinorLocator(**self._kwargs)
        else:
            minor_locator = NullLocator()
        axis.set_minor_locator(minor_locator)
        axis.set_minor_formatter(NullFormatter())

register_scale(TimeScale)

#=======================================================================

class MultiScaleTransformer(object):
    # do actual work
    def __init__(self,
                 data = None,
                 frac = None,
                 **kwargs
                 ):

        assert len(data) == len(frac)
        assert frac[0] == 0 and frac[-1] == 1
        # TODO - check both increase monotonically
        # TODO - add extra initialization methods

        self.frac = np.array(frac)
        self.data = np.array(data)
        self.scale = (
            (self.frac[1:] - self.frac[:-1]) /
            (self.data[1:] - self.data[:-1])
            )
        self.iscal = 1 / self.scale

    def data2axis(self, d):
        f = d.copy()
        ii = np.where(d < self.data[1])
        f[ii] = (d[ii] - self.data[0]) * self.scale[0] + self.frac[0]
        for i in range(len(self.scale[1:-1])):
            ii = np.where((d >= self.data[i+1]) & (d < self.data[i+2]))
            f[ii] = (d[ii] - self.data[i+1]) * self.scale[i+1] + self.frac[i+1]
        ii = np.where(d >= self.data[-2])
        f[ii] = (d[ii] - self.data[-2]) * self.scale[-1] + self.frac[-2]
        return f

    def axis2data(self, f):
        d = f.copy()
        ii = np.where(f < self.frac[1])
        d[ii] = (f[ii] - self.frac[0]) * self.iscal[0] + self.data[0]
        for i in range(len(self.iscal[1:-1])):
            ii = np.where((f >= self.frac[i+1]) & (f < self.frac[i+2]))
            d[ii] = (f[ii] - self.frac[i+1]) * self.iscal[i+1] + self.data[i+1]
        ii = np.where(f > self.frac[-2])
        d[ii] = (f[ii] - self.frac[-2]) * self.iscal[-1] + self.data[-2]
        return d

class MultiScaleTransformBase(Transform):
    input_dims = 1
    output_dims = 1
    has_inverse = True
    is_separable = True
    def __init__(self, transform):
        super().__init__()
        self._transform = transform

class MultiScaleTransform(MultiScaleTransformBase):
    def transform_non_affine(self, a):
        return self._transform.data2axis(a)
    def inverted(self):
        return InvertedMultiScaleTransform(self._transform)

class InvertedMultiScaleTransform(MultiScaleTransformBase):
    def transform_non_affine(self, a):
        return self._transform.axis2data(a)
    def inverted(self):
        return MultiScaleTransform(self._transform)

class MultiScaleBase(LinearScale):
    """
    Multiple broken scales
    """
    name = 'multiscalebase'

    def __init__(self, axis, **kwargs):
        """
        need to supply list of
        - data start values
        - axis fractions
        """
        self.axis = axis
        self._kwargs = kwargs.copy()
        self._transformer = MultiScaleTransformer(**self._kwargs)

    def get_transform(self):
        return MultiScaleTransform(self._transformer)

register_scale(MultiScaleBase)

class MultiScale(MultiScaleBase):
    """
    Multiple broken scales
    """
    name = 'multiscale'

    def __init__(self, axis, **kwargs):
        super().__init__(axis, **kwargs)
    __init__.__doc__ = MultiScaleBase.__init__.__doc__


    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        multi-scaling.
        """
        axis.set_major_locator(FixedLocator(locs = self._transformer.data))
        axis.set_major_formatter(NullFormatter())

        # simple first step: minor locations in middle of range, later
        # add locators that dynamically changes ticks if range is
        # truncated.
        data = self._transformer.data
        minor_locs = 0.5 * (data[1:] + data[:-1])
        axis.set_minor_locator(FixedLocator(locs = minor_locs))
        # axis.set_minor_formatter(FixedFormatter(seq = self._labels))
        axis.set_minor_formatter(LabelMultiScaleFormatter(seq = self._labels))
        axis.set_tick_params(which = 'minor', length = 0)
        # TODO - replace major_formatter by intelligent formater that
        # returns value when not used to format axis.
        if axis.axis_name == 'y':
            axis.axes.fmt_ydata = lambda x: '{:g}'.format(x)
        elif axis.axis_name == 'x':
            axis.axes.fmt_xdata = lambda x: '{:g}'.format(x)
        else:
            raise NotImplementedError()

register_scale(MultiScale)

class LabelMultiScaleFormatter(FixedFormatter):

    def set_locs(self, locs):
        if self.axis.axis_name == 'y':
            for tick in self.axis.get_minor_ticks():
                # this seems like a bug to me but seems to work
                label = tick.label
                label.set_rotation(90)
                label.set_verticalalignment('center')
                label.set_horizontalalignment('right')
            # else:
            #     print('xxx')
            #     self.axis.set_tick_params(which = 'minor', pad = label.get_fontsize() / 2)

        super().set_locs(locs)

class LabeledMultiScale(MultiScaleBase):
    """
    Multi-scale with labels

    use major ticks as section separators
    use minor labels for section labels
    (hide minor labels)
    """
    name = 'labeledmultiscale'
    def __init__(self, axis, **kwargs):
        """
        need to supply list of labels for each section
        """
        super().__init__(axis, **kwargs)
        self._labels = self._kwargs.pop('labels')


    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        labels.
        """
        axis.set_major_locator(FixedLocator(locs = self._transformer.data))
        axis.set_major_formatter(NullFormatter())

        # simple first step: minor locations in middle of range, later
        # add locators that dynamically changes ticks if range is
        # truncated.
        data = self._transformer.data
        minor_locs = 0.5 * (data[1:] + data[:-1])
        axis.set_minor_locator(FixedLocator(locs = minor_locs))
        # axis.set_minor_formatter(FixedFormatter(seq = self._labels))
        axis.set_minor_formatter(LabelMultiScaleFormatter(seq = self._labels))
        axis.set_tick_params(which = 'minor', length = 0)
        # TODO - replace major_formatter by intelligent formater that
        # returns value when not used to format axis.
        if axis.axis_name == 'y':
            axis.axes.fmt_ydata = lambda x: '{:g}'.format(x)
        elif axis.axis_name == 'x':
            axis.axes.fmt_xdata = lambda x: '{:g}'.format(x)
        else:
            raise NotImplementedError()

register_scale(LabeledMultiScale)

#=======================================================================

class UnitFormatter(ScalarFormatter):
    def __init__(self, **kwargs):
        kw = kwargs.copy()
        self._unit = kw.pop('unit', None)
        self._var = kw.pop('var', 'value')
        self._sigfig = kw.pop('sigfig', 2)
        kw.setdefault('useMathText', True)
        kw.setdefault('useLocale', False)
        super().__init__(**kw)
        super().set_powerlimits((-2,3))

    def fix_minus(self, s):
        '''
        disable fixing by default
        '''
        return s

    def get_offset(self):
        '''
        disable weired offset
        '''
        return ''

    def set_locs(self, locs):
        # offset has wrong sign
        super().set_locs(locs)
        if self._unit is not None:
            label = '{} ({})'.format(self._var, self._unit)
        else:
            label = self._var
        # this require proper dissection or own get_offset equivaltent
        o = super().get_offset()
        if o is not '':
            if o.startswith('$'):
                pass
            else:
                if o.startswith(('+', '-')):
                    pass
                else:
                    o = 'x ' + o
            label = label +  ' ' + o
        self.axis.set_label_text(label)

    def _set_offset(self, range):
        # *** modified from MatPlotLib original to include self._sigfig
        # offset of 20,001 is 20,000, for example
        locs = self.locs

        if locs is None or not len(locs) or range == 0:
            self.offset = 0
            return
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(locs)
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        ave_loc = np.mean(locs)
        if len(locs) and ave_loc:  # dont want to take log10(0)
            ave_oom = math.floor(math.log10(np.mean(np.absolute(locs))))
            range_oom = math.floor(math.log10(range))

            if np.absolute(ave_oom - range_oom) >= self._sigfig - 1:  # modified form MPL
                p10 = 10 ** range_oom
                if ave_loc < 0:
                    self.offset = (math.ceil(np.max(locs) / p10) * p10)
                else:
                    self.offset = (math.floor(np.min(locs) / p10) * p10)
            else:
                self.offset = 0

class UnitScale(LinearScale):
    """
    Scale with unit
    """
    name = 'unitscale'

    def __init__(self, axis, **kwargs):
        """
        """
        self.axis = axis
        self.kw = kwargs.copy()

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        symmetrical log scaling.
        """
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(UnitFormatter(**self.kw))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

register_scale(UnitScale)

#=======================================================================


def test():
    from matplotlib import rc
    rc('mathtext',**{'default':'sf'})

    import matplotlib.pyplot as plt
    import physconst

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_xscale('timescale', same_units = 'strict', align = 'left')
    #ax.set_xscale('timescale', same_units = True)
    #ax.set_xscale('timescale', same_units = False)
    #ax.set_xscale('timescale', tick_units = True)
    #ax.set_xscale('timescale', tick_units = True, same_units = True)
    #ax.set_xscale('timescale', tick_units = True, same_units = 'strict')
    #ax.set_yscale('unitscale', var = 'mass', unit = r'$\mathsf{M}_\odot$')
    ax.set_yscale('labeledmultiscale',
                  frac = [0,.5,.7,1],
                  data = [0,30,30.1,32],
                  labels = [r'10 Myr',
                            r'100 yr',
                            r'40 kyr'])
    x = np.arange(0, 1000, 1)
    y = x**0.5
    ax.plot(x,y)

    ax.set_xscale('multiscale',
                  frac = [0,.5,.7,1],
                  data = [0,800,820,1000])

    plt.draw()
    # ax.set_xlim((np.array([0,1]) + 1.e9)*physconst.SEC)
    # ax.set_ylim((np.array([0,1])*1e-9 + 2))
    #return fig, ax
