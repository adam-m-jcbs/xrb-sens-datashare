import numpy as np
import six

from .colors import _color

from matplotlib.colors import rgb2hex

def hex2val(h):
    if isinstance(h, np.ndarray):
        assert h.size == 1
        if np.issubdtype(h.dtype, int):
            val = int(h) / 255
        else:
            val = float(h)
    elif isinstance(h, int):
        val = h / 255
    elif isinstance(h, str):
        if h[0] == '#':
            i0 = 1
        else:
            i0 = 0
        if i0 == 1:
            val = int(h[i0:], base = 16) / (16**len(h[i0:])-1)
        elif h.count('.') == 0:
            val = int(h[i0:], base = 0) / 255
        else:
            val = float(h)
    else:
        val = float(h)
    assert 0 <= val <= 1, 'invalid color component value: "{}"'.format(h)
    return val

def hex2rgb(h):
    """
    Given a string, return rgb(a) ndarray of floats 0...1

    hex string formats:
    #rrggbb, #rrggbbaa, #rgb, #rgba, #kk, #k
    where 'k' stangs for 'black' and is converted to gray value
    (same values for r, g, and b)

    interger string formats
    n       - interpret as 256-based rgb value r*65536 + g*256 + b
              Note that plain 0x, 0o, 0b, ... are interpreted this way as well
    r,g,b
    r,g,b,a
    k,a
    k,      - 256-based values for a, r, g, a, and k

    float-based string formats
    r,g,b
    r,g,b,a
    k,a
    k,
    k       - 1-based values for a, r, g, a, and k

    MIXED FLOAT / INTEGER SPECIFICATIONS ARE INTERPRETED IN TYPE BY COMPONENT
    """
    if not np.isscalar(h) and len(h) == 2:
        val = np.array([hex2val(v) for v in h])
        val = np.append(np.tile(val[0], 3), val[1])
    elif not np.isscalar(h) and len(h) in (3, 4):
        val = np.array([hex2val(v) for v in h])
    elif not isinstance(h, str) and np.issubdtype(np.array(h).dtype, int):
        x = int(h)
        val = np.ndarray(3)
        for i in range(3):
            val[2-i] = (x % 256) / 255
            x = x // 256
        if x > 0:
            raise AttributeError('integer out of bounds: "{}"'.format(h))
    elif not isinstance(h, str):
        val = hex2val(h)
        val = np.tile(val, 3)
    elif h[0] != '#' or h.count(',') > 0:
        if h.count(',') in (1,2,3):
            if h.count(',') == 1 and len(h.split(',')[1]) == 0:
                h = h[:-1]
            val = [hex2val(s) for s in h.split(',')]
            if len(val) == 2:
                val = np.append(np.tile(val[0], 3), val[1])
            elif len(val) ==1:
                val = np.tile(val[0], 3)
            else:
                val = np.array(val)
        elif h.count('.') == 1:
            val = float(h)
            assert 0 <= val <= 1, 'Invalid float color value: "{}"'.format(h)
            val = np.tile(val, 3)
        else:
            # interpret as number, RGB only, no alpha
            x = int(h, base = 0)
            val = np.ndarray(3)
            for i in range(3):
                val[2-i] = (x % 256) / 255
                x = x // 256
            if x > 0:
                raise AttributeError('integer out of bounds: "{}"'.format(h))

    else:
        i0 = 1
        if len(h[i0:]) in (6,8):
            val = np.array([int(h[i:i+2], base = 16) / 255
                            for i in range(i0, len(h), 2)])
        elif len(h[i0:]) in (3,4):
            val = np.array([int(h[i:i+1], base = 16) / 15
                            for i in range(i0, len(h), 1)])
        elif len(h[i0:]) in (9,12):
            val = np.array([int(h[i:i+3], base = 16) / 4095
                            for i in range(i0, len(h), 1)])
        elif len(h[i0:]) in (1,2):
            val = np.tile(int(h[i0:], base = 16) / (16**len(h[i0:])-1), 3)
        else:
            raise AttributeError('Invalid hexadecimal color specification string: "{}"'.format(h))
    return val

class ColorGetter(object):
    """
    TODO - add filter to transform color to correct for output device,
    e.g., gamma = 1/2.2 for screen, as per module default setting.

    Likely there should be a version for end user (does correction) and
    one for internal use?
    """
    def __init__(self, rgb = False):
        self.rgb = rgb
    def __call__(self, arg):
        if isinstance(arg, str):
            val = _color.get(arg, None)
            if val is None:
                try:
                    rgba = hex2rgb(arg)
                except:
                    return None
                if self.rgb:
                    return rgb(arg)
                return arg
            if self.rgb:
                return rgb(val)
            return val
        raise ValueError('require string argument')
    def __getitem__(self, arg):
        return self.__call__(arg)
    def __getattr__(self, arg):
        try:
            c = self.__call__(arg)
        except:
            raise AttributeError(arg)
        else:
            if c is None:
                raise AttributeError(arg)
        return c

color = ColorGetter()
colrgb = ColorGetter(rgb = True)

def rgb(arg, **kwargs):
    """
    convert string to rgb[a] values.

    use color names if matching value is found
    """
    if arg is None:
        return None
    if not np.isscalar(arg):
        if not isinstance(arg, np.ndarray):
            x = np.array(arg)
        else:
            x = arg
        if x.dtype.kind == 'U':
            return np.array([rgb(v, **kwargs) for v in x])
    if not np.isscalar(arg) and len(arg) in (3, 4):
        if not isinstance(arg, np.ndarray):
            val = np.array(arg)
        else:
            val = arg
    else:
        try:
            try:
                val = _color.get(arg, None)
            except:
                val = None
            if val is not None:
                arg = val
            val = hex2rgb(arg)
        except Exception as e:
            raise AttributeError('invalid color: "{}"'.format(arg)) from e
    alpha = kwargs.get('alpha', None)
    force_alpha = kwargs.get('force_alpha', None)
    if force_alpha is not None:
        force_alpha = hex2val(force_alpha)
        if len(val) == 3:
            val = np.append(val, force_alpha)
        else:
            val[3] = force_alpha
    elif len(val) == 3 and alpha is not None:
        alpha = hex2val(alpha)
        val = np.append(val, alpha)
    no_alpha = kwargs.get('no_alpha', False)
    if no_alpha:
        val = val[:3]
    return val

def is_string_like(obj):
    """
    Return True if *obj* is of string type

    used to avoid iterating over members
    """
    # from depricated matplotlib
    return isinstance(obj, (six.string_types, np.str_, np.unicode_))
