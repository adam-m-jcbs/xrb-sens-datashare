from .util import _units, _Units, _div_lim


class Unit(object):
    def __init__(self, name, value,
                 div_lim = None,
                 ):
        self.name = name
        self.value = value
        self.div_lim = div_lim or self.value

class Units(object):
    def __init__(self, *args):
        self.data = sorted(args, key = lambda x: x.value)
        self.lookup = {x.name: i for i,x in enumerate(self.data)}
        for i, x in enumerate(self.data):
            if x.value == 1:
                self.base = x
                self.base_index = i
                break
        else:
            raise Exception('[UNITS] No base unit found.')

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[self.lookup[index]]
        return self.data[index]
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        for x in self.data:
            yield x

import datetime
SEC = 31556952
def time_convert(val):
    if isinstance(val, datetime.timedelta):
        val = val.total_seconds()
    return val

time_config = {
    'convert' : time_convert,
    'units' : Units(
        Unit('s', 1),
        Unit('min', 60, div_lim = 100),
        Unit('h', 3600, div_lim = 100),
        Unit('d', 86400),
        Unit('yr', 31556952),
        )
    }

byte_config = {
    'integer' : True,
    'units' : Units(
        Unit('B', 1),
        ),
    'non_si_scaling_default' : True,
    }

class Human(object):
    """
    basis class for general scaling

    config has the following fields:

    convert:
      conversion function to basic unit type
    units:
        Units object

    """
    def __init__(self, config):
        self.convert = config.pop('convert', None)
        self.integer = config.pop('integer', False)
        self.units = config.pop('units')
        self.non_si_power = config.pop('non_si_power', 1024)
        self.non_si_prefix = config.pop('non_si_prefix', 'i')
        self.non_si_scaling_default = config.pop('non_si_scaling_default', False)


        # TODO - exclude some units if requested on unit object as option
        self._unit2scale = dict()
        for unit in self.units:
            for i,u in enumerate(_units):
                self._unit2scale[u + unit.name] = unit.value * 10**(-3 * i)
            for i,u in enumerate(_Units[1:]):
                self._unit2scale[u + unit.name] = unit.value * 10**(3 *(i + 1))

        # compute divlim scales.
        # todo - allow overwrite of div limits by Unit property
        self.scales = [None]
        self.divmax = []
        for i in range(self.units.base_index + 1, len(self.units)):
            scale = self.units[i].value / self.units[i-1].value
            if abs(scale - int(scale)) < 1.e-10:
                self.scales += [scale]
            else:
                self.scales += [None]

        print(self.scales)


    def __call__(self, value,
                 digits = 2
                 ):

        if self.convert is not None:
            val = self.convert(val)

        aval = abs(time)
        if self.integer:
            oval = round(aval)
            assert abs((oval + 1.e-16)/(aval + 1.e-16) - 1) < 1.e-14
            xval = oval
        else:
            xval = aval

        su = self.units.base.name

        length = digits + 1

        div_lim1 = _div_lim(1, 3)
        div_lims = []
        for sc in self.scales:
            pass




    def split_unit(self, s,
                   num_val = False,
                   num_unit = False):
        if s.count(' ') == 1:
            v,u = s.split()
        else:
            j = -1
            for i,c in enumerate(s):
                if c in '1234567890.':
                    j = i + 1
            if j == -1:
                v = 1
                u = s.strip()
            elif j == len(s):
                v = s.strip()
                u = 's'
            else:
                v = s[:j].strip()
                u = s[j:].strip()
        if num_val:
            try:
                v = int(v)
            except:
                v = float(v)
            iv = int(v)
            if iv == v:
                v = iv
        if num_unit:
            u = self.unit2scale(u)
        return v,u

    def unit2scale(self, unit):
        scale = self._unit2scale.get(unit, 0)
        if scale != 0:
            return scale

    def max_unit(self, units):
        return sorted(units, key = self.unit2scale)[-1]

    def human2val(self, s):
        s = s.replace(',', '')
        v,u = self.split_unit(s)
        try:
            return int(v) * self.unit2scale(u)
        except:
            return float(v) * self.unit2scale(u)


h2t = Human(time_config)
b2t = Human(byte_config)
