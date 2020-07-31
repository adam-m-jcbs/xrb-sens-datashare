# Define color models

import numpy as np

from collections import Iterable

try:
    from ._utils import MetaSingletonHash as Meta
    colorclassparm = dict(metaclass = Meta)
except:
    colorclassparm = dict()

class ColorModel(object, **colorclassparm):
    """
    Color Model base class.
    Note that this is generated as "singleton" - only one object of each class.
    """
    limits =  np.tile(np.array([0., 1.]), (3, 1))
    _range = limits.copy()
    @classmethod
    def _inverse(self):
        raise NotImplementedError()
    @classmethod
    def inverse(cls, *args, **kwargs):
        """
        Return inverse color transform.

        Subclasses to define method _inverse to return instance of
        inverse object.
        """
        if len(args) > 0 or len(kwargs) > 0:
            return cls._inverse()(*args, **kwargs)
        return cls._inverse()
    def __call__(self, *agrs, **kwargs):
        """
        Accepts and return [x,3|4] array.
        Optionally deal with 3|4 vectors, 3|4 scalars.
        Treatment of (3|4)x(3|4) is ambiguous and will be interpreted as [x,3|4].
        """
        raise NotImplementedError()

    # a set of conversion routines to be used by derived classes
    @staticmethod
    def _args_to_vectors(args):
        """
        TODO - need to add auto-convert to gray
        """
        assert len(args) in (1, 3, 4)
        if len(args) in (3, 4):
            if not isinstance(args[0], np.ndarray):
                mode = 0
                p0 = np.array([args[0]])
                p1 = np.array([args[1]])
                p2 = np.array([args[2]])
                if len(args) == 4:
                    p3 = np.array([args[2]])
                else:
                    p3 = None
            else:
                mode = 1
                if len(args) == 3:
                    (p0, p1, p2), p3 = args, None
                else:
                    p0, p1, p2, p3 = args
        else:
            arg = args[0]
            if isinstance(arg, Iterable) and not isinstance(arg, np.ndarray):
                arg = np.array(arg, dtype = np.float64)
            assert isinstance(arg, np.ndarray)
            if len(arg.shape) == 2:
                if arg.shape[1] in (3, 4):
                    mode = 2
                    if arg.shape[1] == 3:
                        (p0, p1, p2), p3 = arg.transpose(), None
                    else:
                        p0, p1, p2, p3 = arg.transpose()
                else:
                    mode = 3
                    if arg.shape[0] == 3:
                        (p0, p1, p2), p3 = arg, None
                    else:
                        p0, p1, p2, p3 = arg
            elif len(arg.shape) == 1 and arg.shape[0] in (3, 4):
                mode = 4
                if arg.shape[0] == 3:
                    (p0, p1, p2), p3 = arg[:, np.newaxis], None
                else:
                    p0, p1, p2, p3 = arg[:, np.newaxis]
            else:
                raise ValueError('Invalid Data / Shape')
        if p3 is not None:
            mode += 8
        else:
            # set p3 to dummy array of 1.
            p3 = np.tile(1., p2.shape)
        return p0, p1, p2, p3, mode

    # deal with gray values
    @staticmethod
    def _gray_args_to_vectors(args):
        """
        Convert input to g,a vectors; if 2 values a rgivem assume second is a
        """
        assert len(args) in (1, 2)

        if len(args) == 2:
            if not isinstance(args[0], np.ndarray):
                mode = 0
                p0 = np.array([args[0]])
                p1 = np.array([args[1]])
            else:
                mode = 1
                p0, p1 = args
        else:
            arg = args[0]
            if isinstance(arg, Iterable) and not isinstance(arg, np.ndarray):
                arg = np.array(arg, dtype = np.float64)
            if not isinstance(arg, np.ndarray):
                mode = 0
                p0, p1 = np.array(args), None
            elif len(arg.shape) == 0:
                p0, p1 = arg[np.newaxis], None
                mode = 4
            elif len(arg.shape) == 1:
                if arg.shape[0] == 2:
                    mode = 4
                    p0, p1 = arg[:,np.newaxis]
                else:
                    mode = 2
                    p0, p1 = arg, None
            elif len(arg.shape) == 2:
                if arg.shape[1] in (1, 2):
                    mode = 2
                    if arg.shape[1] == 1:
                        p0, p1 = arg.transpose(), None
                    else:
                        p0, p1 = arg.transpose()
                else:
                    mode = 3
                    if arg.shape[0] == 1:
                        p0, p1 = arg, None
                    else:
                        p0, p1 = arg
            else:
                raise ValueError('Invalid Data / Shape')
        if p1 is not None:
            mode += 8
        else:
            # set p1 to dummy array of 1.
            p1 = np.tile(1., p0.shape)
        return p0, p1, mode

    @staticmethod
    def _args_to_array(args, splitalpha = True):
        """
        TODO - need to add auto-convert to gray
        """
        assert len(args) in (1, 3, 4)
        if len(args) == 3:
            if not isinstance(args[0], np.ndarray):
                mode = 0
                ccc = np.array([args])
            else:
                mode = 1
                ccc = np.array(args).transpose()
        elif len(args) == 4:
            if not isinstance(args[0], np.ndarray):
                mode = 8
                ccca = np.array([args])
            else:
                mode = 9
                ccca = np.array(args).transpose()
        else:
            arg = args[0]
            if isinstance(arg, Iterable) and not isinstance(arg, np.ndarray):
                arg = np.array(arg, dtype = np.float64)
            assert isinstance(arg, np.ndarray)
            if len(arg.shape) == 2:
                if arg.shape[1] == 4:
                    mode = 10
                    ccca = np.array(arg)
                elif arg.shape[1] == 3:
                    mode = 2
                    ccc = np.array(arg)
                elif arg.shape[0] == 4:
                    mode = 11
                    ccca = np.array(arg).transpose()
                else:
                    assert arg.shape[0] == 3
                    mode = 3
                    ccc = np.array(arg).transpose()
            else:
                assert len(arg.shape) == 1
                if arg.shape[0] == 3:
                    mode = 4
                    ccc = np.array(args)[:,np.newaxis]
                elif arg.shape[0] == 4:
                    mode = 12
                    ccca = np.array(args)[:,np.newaxis]
                else:
                    raise ValueError('Invalid Data / Shape')

        if mode < 8:
            a = np.tile(1., ccc.shape[1:])
            if not splitalpha:
                ccca = np.hstack((ccc, a[..., np.newaxis]))
        elif mode >= 8 and splitalpha:
            ccc, a = np.split(ccca, [3], axis = 1)
            a = a[..., 0]
        if splitalpha:
            return ccc, a, mode
        return ccca, mode

    @staticmethod
    def _vectors_to_return(p0, p1, p2, p3, mode):
        if mode == 0:
            return p0[0], p1[0], p2[0]
        if mode == 1:
            return p0, p1, p2
        if mode == 2:
            return np.vstack((p0, p1, p2)).transpose()
        if mode == 3:
            return np.vstack((p0, p1, p2))
        if mode == 4:
            return np.hstack((p0, p1, p2))

        if mode == 8:
            return p0[0], p1[0], p2[0], p3[0]
        if mode == 9:
            return p0, p1, p2, p3
        if mode == 10:
            return np.vstack((p0, p1, p2, p3)).transpose()
        if mode == 11:
            return np.vstack((p0, p1, p2, p3))
        if mode == 12:
            return np.hstack((p0, p1, p2, p3))

        raise ValueError('Invalid mode')

    @staticmethod
    def _array_to_return(*args, splitalpha = True):
        if splitalpha:
            ccc, a, mode = args
            if mode >= 8:
                ccca = np.hstack((ccc, a[..., np.newaxis]))
        else:
            ccca, mode = args
            if mode < 8:
                ccc, a = np.split(y,[3])
        if mode == 0:
            return ccc[0][0], ccc[0][1], ccc[0][2]
        if mode == 1:
            return ccc[:,0], ccc[:,1], ccc[:,2]
        if mode == 2:
            return ccc
        if mode == 3:
            return ccc.transpose()
        if mode == 4:
            return ccc[0]

        if mode == 8:
            return ccca[0][0], ccca[0][1], ccca[0][2], ccca[0][3]
        if mode == 9:
            return ccca[:,0], ccca[:,1], ccca[:,2], ccca[:,3]
        if mode == 10:
            return ccca
        if mode == 11:
            return ccca.transpose()
        if mode == 12:
            return ccca[0]

        raise ValueError('Invalid mode')

    # _gray_func = lambda x: x

    @classmethod
    def gray(cls, *args):
        """
        Return gray value for given scalar in the source color space.

        Return should be:
        scalar --> tuple
        np_array --> np_array:
            [1|2,x] --> [3|4,x] for x > 1
            else:     [x,3|4]

        Here we provide as default the method for RGB as this is used
        in all of the 'inverse' transforms.

        If a class provides _gray_func this is applied to the gray
        value first.

        If a class provides arrays _gray_index and _gray_value
        then additionally we set in, e.g., [x, 3]
        [x, _gray_index] = [_gray_value]
        Typical use is a 2-vector, e.g.,
            _gray_index = [1,2]
            _gray_value = [0,1]
        for use in color circle values like HSV.
        """
        g, a, mode = cls._gray_args_to_vectors(args)
        try:
            g = cls._gray_func(g)
        except:
            pass
        if len(g.shape) == 1:
            ccc = np.tile(g, (3,1)).transpose()
        else:
            ccc = g
        try:
            ccc[:,np.array(cls._gray_index)] = np.array(cls._gray_value)[np.newaxis,:]
        except:
            pass
        return cls._array_to_return(ccc, a, mode)

    @classmethod
    def is_normal(cls, limits):
        """
        Check whether range is valid or should be normalized.

        This just covers a set of default checks from my old IDL
        routines.
        """
        for i in range(limits.shape[0]):
            if cls.limits[i,1] > 1 and limits[i,1] <= 1:
                return False
            if cls.limits[i,1] <= 1 and limits[i,1] > cls.limits[i,1]:
                return False
            if cls.limits[i,0] > 1 and limits[i,0] < cls.limits[i,0]:
                return False
        return True

    @classmethod
    def normalize(cls, *args):
        """
        By default we just scale 0...1 range to limits no matter what
        the values.
        """
        ccc, a, mode = cls._args_to_array(args)
        m = cls.limits[:,0][np.newaxis,:]
        M = cls.limits[:,1][np.newaxis,:]
        ccc = m + ccc*(M - m)
        print("why do we normalize?")
        return cls._array_to_return(ccc, a, mode)

def _make_transform(rxy, gxy, bxy, wxy):
    """
    Make transform matrix and its inverse given xy coordinates of rgb
    vectors and white point.

    http://www.ryanjuckett.com/programming/rgb-color-space-conversion/
    """
    xy = np.array([rxy, gxy, bxy]).transpose()
    xyz = np.append(xy, (1 - xy[0] - xy[1])[np.newaxis, :], axis = 0)
    wxyz = np.append(wxy, 1 - wxy[0] - wxy[1])
    wXYZ = wxyz / wxyz[1]
    I = np.linalg.inv(xyz)
    scale = np.inner(I, wXYZ)
    M = xyz * scale[np.newaxis, :]
    I *= (1 / scale)[:, np.newaxis]
    return M, I

class ColorModelMatrix(ColorModel):
    """
    Prototype for matrix color classes.

    provides __call__ method
    requires _matrix class attribute
    """
    _offset0 = np.zeros((3,))
    _offset1 = np.zeros((3,))
    _matrix = np.identity(3)
    @classmethod
    def _transform(cls, a):
        return (np.inner(a + cls._offset0[np.newaxis, :],
                         cls._matrix) +
                + cls._offset1[np.newaxis, :])
    @classmethod
    def __call__(cls, *args, clip = True):
        __doc__ = ColorModel.__call__.__doc__
        ccc, a, mode = cls._args_to_array(args)
        if clip:
            np.clip(ccc,
                    (cls.limits[:,0])[np.newaxis,:],
                    (cls.limits[:,1])[np.newaxis,:],
                    out = ccc)
        ccc = cls._transform(ccc)
        if clip:
            np.clip(ccc,
                    (cls._range[:,0])[np.newaxis,:],
                    (cls._range[:,1])[np.newaxis,:],
                    out = ccc)
        return cls._array_to_return(ccc, a, mode)

class ColorModelDynamicMatrix(ColorModelMatrix, metaclass = type):
    """
    Prototype for matrix color classes.

    provides __call__ method
    requires _matrix class attribute
    """
    def _transform(self, a):
        return (np.inner(a + self._offset0[np.newaxis, :],
                         self._matrix) +
                + self._offset1[np.newaxis, :])

    def __call__(self, *args):
        __doc__ = ColorModelMatrix.__call__.__doc__
        ccc, a, mode = self._args_to_array(args)
        np.clip(ccc,
                (self.limits[:,0])[np.newaxis,:],
                (self.limits[:,1])[np.newaxis,:],
                out = ccc)
        ccc = self._transform(ccc)
        np.clip(ccc,
                (self._range[:,0])[np.newaxis,:],
                (self._range[:,1])[np.newaxis,:],
                out = ccc)
        return self._array_to_return(ccc, a, mode)

    def __init__(self, matrix = None, **kwargs):
        """
        allow to set up transform by providing color matrix
        """
        if matrix is not None:
            self._matrix = matrix
            self._matrixI = np.linalg.inv(matrix)
        super().__init__()

#=======================================================================
# White points from
# http://en.wikipedia.org/wiki/Standard_illuminant

white_points_CIE1931_2 = dict(
    A =   [0.44757, 0.40745],
    B =   [0.34842, 0.35161],
    C =   [0.31006, 0.31616],
    D50 = [0.34567, 0.35850],
    D55 = [0.33242, 0.34743],
    D65 = [0.31271, 0.32902],
    D75 = [0.29902, 0.31485],
    E =   [    1/3,     1/3],
    F1 =  [0.31310, 0.33727],
    F2 =  [0.37208, 0.37529],
    F3 =  [0.40910, 0.39430],
    F4 =  [0.44018, 0.40329],
    F5 =  [0.31379, 0.34531],
    F6 =  [0.37790, 0.38835],
    F7 =  [0.31292, 0.32933],
    F8 =  [0.34588, 0.35875],
    F9 =  [0.37417, 0.37281],
    F10 = [0.34609, 0.35986],
    F11 = [0.38052, 0.37713],
    F12 = [0.43695, 0.40441],
    )
white_points_CIE1964_10 = dict(
    A =   [0.45117, 0.40594],
    B =   [0.34980, 0.35270],
    C =   [0.31039, 0.31905],
    D50 = [0.34773, 0.35952],
    D55 = [0.33411, 0.34877],
    D65 = [0.31382, 0.33100],
    D75 = [0.29968, 0.31740],
    E =   [    1/3,     1/3],
    F1 =  [0.31811, 0.33559],
    F2 =  [0.37925, 0.36733],
    F3 =  [0.41761, 0.38324],
    F4 =  [0.44920, 0.39074],
    F5 =  [0.31975, 0.34246],
    F6 =  [0.38660, 0.37847],
    F7 =  [0.31569, 0.32960],
    F8 =  [0.34902, 0.35939],
    F9 =  [0.37829, 0.37045],
    F10 = [0.35090, 0.35444],
    F11 = [0.38541, 0.37123],
    F12 = [0.44256, 0.39717],
    )

white_points_temperature = dict(
    A =   2856,
    B =   4874,
    C =   6774,
    D50 = 5003,
    D55 = 5503,
    D65 = 6504,
    D75 = 7504,
    E =   5454,
    F1 =  6430,
    F2 =  4230,
    F3 =  3450,
    F4 =  2940,
    F5 =  6350,
    F6 =  4150,
    F7 =  6500,
    F8 =  5000,
    F9 =  4150,
    F10 = 5000,
    F11 = 4000,
    F12 = 3000,
    )

white_points_info = dict(
    A =   "Incandescent / Tungsten",
    B =   "{obsolete} Direct sunlight at noon",
    C =   "{obsolete} Average / North sky Daylight",
    D50 = "Horizon Light. ICC profile PCS",
    D55 = "Mid-morning / Mid-afternoon Daylight",
    D65 = "Noon Daylight: Television, sRGB color space",
    D75 = "North sky Daylight",
    E =   "Equal energy",
    F1 =  "Daylight Fluorescent",
    F2 =  "Cool White Fluorescent",
    F3 =  "White Fluorescent",
    F4 =  "Warm White Fluorescent",
    F5 =  "Daylight Fluorescent",
    F6 =  "Lite White Fluorescent",
    F7 =  "D65 simulator, Daylight simulator",
    F8 =  "D50 simulator, Sylvania F40 Design 50",
    F9 =  "Cool White Deluxe Fluorescent",
    F10 = "Philips TL85, Ultralume 50",
    F11 = "Philips TL84, Ultralume 40",
    F12 = "Philips TL83, Ultralume 30",
    )

# http://en.wikipedia.org/wiki/RGB_color_space
primaries = dict(
    scRGB     = [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], # D65
    sRGB      = [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], # D65
    AdobeRGB  = [[0.64, 0.33], [0.21, 0.71], [0.15, 0.06]], # D65
    PAL       = [[0.64, 0.33], [0.29, 0.60], [0.15, 0.06]], # D65
    US_NTSC   = [[0.63, 0.34], [0.31, 0.595], [0.155, 0.07]], # D65
    Jp_NTSC   = [[0.63, 0.34], [0.31, 0.595], [0.155, 0.07]], # D93
    Apple     = [[0.625, 0.340], [0.280, 0.595], [0.155, 0.070]], # D65
    NTSC      = [[0.67, 0.33], [0.21, 0.71], [0.14, 0.08]], # C
    HDTV      = [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], # D65
    UHDTV     = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], # D65
    AdobeWide = [[0.735, 0.265], [0.115, 0.826], [0.157, 0.018]], # D50
    Wide      = [[0.7347, 0.2653], [0.1152, 0.8264], [0.1566, 0.0177]], # D50
    ProPhoto  = [[0.7347, 0.2653], [0.1596, 0.8404], [0.0366, 0.0001]], # D50
    CIE       = [[0.73467, 0.26533], [0.27376, 0.71741], [0.16658, 0.00886]], # E
    CIE2      = [[0.7350, 0.2650], [0.2740, 0.7170], [0.1670, 0.0090]], # E
    SGI       = [[0.625, 0.340], [0.280, 0.595], [0.155, 0.070]], # D65, same as Apple
    ColorMatch= [[0.630, 0.340], [0.295, 0.605], [0.150, 0.075]], # D65
    )

# XYZ to LMS transform matrices
_m_Bradford = np.array([ # Bradford transformation
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296],
    ])
_m_VonKries_D65 = np.array([# RLAB
    [ 0.40024,  0.70760, -0.08081],
    [-0.22630,  1.16532,  0.04570],
    [ 0.00000,  0.00000,  0.91822],
    ])
_m_VonKries_E = np.array([ # RLAB, CIE equal energy illuminant
    [ 0.38971, 0.68898, -0.07868],
    [-0.22981, 1.18340,  0.04641],
    [ 0.00000, 0.00000,  1.00000],
    ])
_m_CAT02 = np.array( # http://en.wikipedia.org/wiki/CIECAM02
    [[ 0.7328, 0.4296, -0.1624],
     [-0.7036, 1.6975,  0.0061],
     [ 0.0030, 0.0136,  0.9834],
     ])
_m_CAT97s = np.array( # spectrally-sharpened Bradford chromatic adaptation matrix
    [[ 0.8562,  0.3372, -0.1934],
     [-0.8360,  1.8327,  0.0033],
     [ 0.0357, -0.0469,  1.0112],
     ])

def _xyY2XYZ(x, y, Y):
    """
    Convert xyY to XYZ.

    TODO - make full ColorModel interface
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    ii = y > 0
    Yy = np.ones_like(y)
    Yy[ii] = Y[ii] / y[ii]
    X = x * Yy
    Z = (1 - x - y) * Yy
    return X, Y, Z

def _XYZ2xyY(X, Y, Z):
    """Convert XYZ to xyY."""
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    if not isinstance(Z, np.ndarray):
        Z = np.array(Z)
    XYZ = X + Y + Z
    ii = XYZ > 0
    XYZ = 1 / XYZ[ii]
    x = np.tile(1 / 3, X.shape)
    y = x.copy()
    x[ii] = X[ii] * XYZ
    y[ii] = Y[ii] * XYZ
    return x, y, Y

def _BradfordMatrix(sw, dw, transform = 'Bradford'):
    """
    assume transform is matrix from XYZ to RGB
    assume sw and dw are xy coordinates of white point
    (could provide XYZ triples instead)
    sw = source white
    dw = destination white

    http://www.babelcolor.com/download/A%20review%20of%20RGB%20color%20spaces.pdf
    """
    # cone resonse matrix
    if isinstance(transform, str):
        if transform == 'Bradford':
            transform = _m_Bradford
        elif transform == 'Von Kries':
            transform = _m_VonKries_D65

    XYZsw = np.array(_xyY2XYZ(sw[0], sw[1], 1.))
    RGBsw = np.inner(transform, XYZsw)
    XYZdw = np.array(_xyY2XYZ(dw[0], dw[1], 1.))
    RGBdw = np.inner(transform, XYZdw)
    M = np.dot(np.dot(np.linalg.inv(transform),
                          np.diag(RGBdw / RGBsw)),
                 transform)
    return M

#######################################################################
# define specific color models

class ColorRGB(ColorModel):
    """
    RGB is essentially just identity.
    """
    @classmethod
    def _inverse(cls):
        return cls()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        return cls._array_to_return(*cls._args_to_array(args))

#-----------------------------------------------------------------------

class ColorCMY(ColorRGB):
    """
    Convert CMY to RGB or inverse.
    """
    _something = 0
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        cmy, a, mode = cls._args_to_array(args)
        rgb = 1 - cmy
        return cls._array_to_return(rgb, a, mode)

#-----------------------------------------------------------------------

class ColorHSV(ColorModel):
    """
    HSV color model.

    hue = [0, 360]
    saturation = [0, 1]
    value =[0, 1]
    """
    limits = np.array([[0., 360.], [0., 1.], [0., 1.]])
    _perm = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1], [2, 1, 0], [1, 2, 0], [0, 2, 1]])
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        h, s, v, a, mode = cls._args_to_vectors(args)
        np.mod(h, 360., out = h)
        np.clip(s, 0, 1, out = s)
        np.clip(v, 0, 1, out = v)
        c = v * s
        p = h / 60
        x = c * (1 - np.abs(np.mod(p, 2.) - 1.))
        m = v - c
        z = np.zeros_like(x)
        col = np.vstack((c,x,z)).transpose()
        ip = np.int64(p)
        rgb = col[np.tile(np.arange(len(x)),(3,1)).transpose(), cls._perm[ip]]
        rgb += m[:,np.newaxis]
        np.clip(rgb, 0, 1, out = rgb)
        return cls._array_to_return(rgb, a, mode)
    @classmethod
    def _inverse(cls, *args, **kwargs):
        return ColorHSVInverse(*args, **kwargs)
    _gray_index = [0,1]
    _gray_value = [90,0]

class ColorHSVInverse(ColorModel):
    """
    Convert RGB to HSV.
    """
    _range = ColorHSV.limits
    @classmethod
    def __call__(cls, *args):
        """
        Convert colors.

        Return:
          hue = [0, 360]
          saturation = [0, 1]
          value =[0, 1]
        """
        r, g, b, a, mode = cls._args_to_vectors(args)
        M = np.maximum(r,np.maximum(g,b))
        m = np.minimum(r,np.minimum(g,b))
        C = M - m
        h = np.zeros_like(C)
        i = np.logical_and(M == r, C != 0)
        h[i] = np.mod((g[i] - b[i]) / C[i], 6)
        i = np.logical_and(M == g, C != 0)
        h[i] = (b[i] - r[i]) / C[i] + 2
        i = np.logical_and(M == b, C != 0)
        h[i] = (r[i] - g[i]) / C[i] + 4
        H = h * 60
        V = M
        S = np.zeros_like(C)
        i = V != 0
        S[i] = C[i] / V[i]
        return cls._vectors_to_return(H, S, V, a, mode)
    @staticmethod
    def _inverse():
        return ColorHSV()

#-----------------------------------------------------------------------

class ColorHSL(ColorModel):
    """
    HSL color model.

    hue = [0, 360]
    saturation = [0, 1]
    lightness =[0, 1]
    """
    limits = np.array([[0., 360.],[0., 1.],[0., 1.]])
    _perm = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1], [2, 1, 0], [1, 2, 0], [0, 2, 1]])
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        h, s, l, a, mode = cls._args_to_vectors(args)
        np.mod(h, 360, out = h)
        np.clip(s,0,1, out = s)
        np.clip(l,0,1, out = l)
        c = (1 - np.abs(2 * l - 1)) * s
        p = h / 60
        x = c * (1 - np.abs(np.mod(p, 2) - 1))
        m = l - 0.5 * c
        z = np.zeros_like(x)
        col = np.vstack((c,x,z)).transpose()
        ip = np.clip(np.int64(p), 0, 5)
        rgb = col[np.tile(np.arange(len(x)),(3,1)).transpose(),cls._perm[ip]]
        rgb += m[:,np.newaxis]
        np.clip(rgb, 0, 1, out = rgb)
        return cls._array_to_return(rgb, a, mode)
    @staticmethod
    def _inverse():
        return ColorHSLInverse()
    _gray_index = [ 0, 1]
    _gray_value = [90, 0]

class ColorHSLInverse(ColorModel):
    """
    Convert RGB to HSL.
    """
    _range = ColorHSL.limits
    @classmethod
    def __call__(cls, *args):
        """
        Convert colors.

        Return:
          hue = [0, 360]
          lightness = [0, 1]
          saturation = [0, 1]
        """
        r, g, b, a, mode = cls._args_to_vectors(args)
        M = np.maximum(r, np.maximum(g, b))
        m = np.minimum(r, np.minimum(g, b))
        C = M - m
        h = np.zeros_like(C)
        i = np.logical_and(M == r, C != 0)
        h[i] = np.mod((g[i]-b[i]) / C[i], 6)
        i = np.logical_and(M == g, C != 0)
        h[i] = (b[i]-r[i]) / C[i] + 2
        i = np.logical_and(M == b, C != 0)
        h[i] = (r[i]-g[i]) / C[i] + 4
        H = h * 60
        L = 0.5 * (M + m)
        S = np.zeros_like(C)
        d = 1 - np.abs(2 * L - 1)
        i = d != 0
        S[i] = C[i] / d[i]
        return cls._vectors_to_return(H, S, L, a, mode)
    @staticmethod
    def _inverse():
        return ColorHSL()

#-----------------------------------------------------------------------

class ColorHSI(ColorModel):
    """
    HSI color model.

    hue = [0, 360]
    saturation = [0, 1]
    intensity =[0, 1]
    """
    limits = np.array([[0., 360.],[0., 1.],[0., 1.]])
    _perm = np.array([[0, 1, 2],[2, 0, 1],[1, 2, 0]])
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        h, s, i, a, mode = cls._args_to_vectors(args)
        np.mod(h, 360, out = h)
        np.clip(i, 0, 1, out = i)
        np.clip(s, 0, 1, out = s)
        p = h / 60
        f = 0.5 * np.mod(p, 2)
        c = s * i * 3
        x = c * (1 - f)
        y = c * f
        z = np.zeros_like(x)
        m = i - i * s
        col = np.vstack((x,y,z)).transpose()
        ip = np.int64(p / 2)
        rgb = col[np.tile(np.arange(len(x)),(3,1)).transpose(),cls._perm[ip]]
        rgb += m[:,np.newaxis]
        np.clip(rgb, 0, 1, out = rgb)
        return cls._array_to_return(rgb, a, mode)
    @staticmethod
    def _inverse():
        return ColorHSIInverse()
    _gray_index = [ 0, 1]
    _gray_value = [90, 0]

class ColorHSIInverse(ColorModel):
    """
    Convert RGB to HSI.
    """
    _range = ColorHSI.limits
    _rad2deg = 180 / np.pi

    @classmethod
    def __call__(cls, *args):
        """
        Convert colors.

        Return:
          hue = [0, 360]
          lightness = [0, 1]
          saturation = [0, 1]
        """
        r, g, b, a, mode = cls._args_to_vectors(args)
        np.clip(r, 0, 1, out = r)
        np.clip(g, 0, 1, out = g)
        np.clip(b, 0, 1, out = b)
        m = np.minimum(r, np.minimum(g, b))

        if True:
            M = np.maximum(r, np.maximum(g, b))
            C = M - m
            h = np.zeros_like(C)
            i = np.logical_and(M == r, C != 0)
            h[i] = np.mod((g[i] - b[i]) / C[i], 6)
            i  = np.logical_and(M == g, C != 0)
            h[i] = (b[i] - r[i]) / C[i] + 2
            i = np.logical_and(M == b, C != 0)
            h[i] = (r[i] - g[i]) / C[i] + 4
            H = h * 60
        else:
            rg = r - g
            rb = r - b
            gb = g - b
            x = rg**2 + rb * gb
            i = x > 0
            H = np.zeros_like(r)
            H[i] = np.arccos(0.5 * (rg[i] + rb[i]) / np.sqrt(x[i])) * cls._rad2deg

        I = (r + g + b) / 3
        S = np.zeros_like(r)
        i = I != 0
        S[i] = 1 - m[i] / I[i]
        return cls._vectors_to_return(H, S, I, a, mode)
    @staticmethod
    def _inverse():
        return ColorHSI()

#-----------------------------------------------------------------------

class ColorHCL(ColorModel):
    """
    HCL color model 'luma/chroma/hue' (renamed for consitency)

    hue = [0, 360]
    chroma = [0, 1]
    luma =[0, 1]

    Use Y'_601 = 0.30*R + 0.59*G + 0.11*B

    http://en.wikipedia.org/wiki/HSL_and_HSV#Color-making_attributes
    """
    limits = np.array([[0., 360.], [0., 1.], [0., 1.]])
    _perm = np.array([[0, 1, 2],[1, 0, 2], [2, 0, 1], [2, 1, 0], [1, 2, 0],[0, 2, 1]])
    _luma_vec = np.array([0.30, 0.59, 0.11])
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        h, c, y, a, mode = cls._args_to_vectors(args)
        h = np.mod(h, 360)
        c = np.clip(c, 0, 1, out = c)
        y = np.clip(y, 0, 1, out = y)
        p = h / 60
        x = c * (1 - np.abs(np.mod(p, 2.) - 1.))
        z = np.zeros_like(x)
        ip = np.int64(p)
        col = np.vstack((c,x,z)).transpose()
        rgb = col[np.tile(np.arange(len(x)),(3,1)).transpose(),cls._perm[ip]]
        m = y - np.dot(rgb, cls._luma_vec)
        rgb += m[:,np.newaxis]
        rgb = np.clip(rgb,0,1, out = rgb)
        return cls._array_to_return(rgb, a, mode)
    @staticmethod
    def _inverse():
        return ColorHCLInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))

class ColorHCLInverse(ColorModel):
    """
    Convert RGB to HCL.

    Return:
      hue = [0, 360]
      chroma = [0, 1]
      luma =[0, 1]

    http://en.wikipedia.org/wiki/HSL_and_HSV#Color-making_attributes
    """
    _range = ColorHCL.limits
    _luma_vec = ColorHCL._luma_vec
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, a, mode = cls._args_to_vectors(args)
        np.clip(r, 0, 1, out = r)
        np.clip(g, 0, 1, out = g)
        np.clip(b, 0, 1, out = b)
        M = np.maximum(r, np.maximum(g, b))
        m = np.minimum(r, np.minimum(g, b))
        C = M - m
        h = np.zeros_like(C)
        i = np.logical_and(M == r, C > 0)
        h[i] = np.mod((g[i] - b[i]) / C[i], 6)
        i  = np.logical_and(M == g, C > 0)
        h[i] = (b[i] - r[i]) / C[i] + 2
        i = np.logical_and(M == b, C > 0)
        h[i] = (r[i] - g[i]) / C[i] + 4
        H = h * 60
        y = np.dot(np.array([r, g, b]).transpose(), cls._luma_vec)
        return cls._vectors_to_return(H, C, y, a, mode)
    @staticmethod
    def _inverse():
        return ColorHCL()

#-----------------------------------------------------------------------

class ColorHCL2(ColorHCL):
    """
    HCL color model 'luma/chroma/hue' (renamed for consitency)

    Input:
      hue = [0, 360]
      chroma = [0, 1]
      luma =[0, 1]

    Use Y'709 = 0.21*R + 0.72*G + 0.07*B

    http://en.wikipedia.org/wiki/HSL_and_HSV#Color-making_attributes
    """
    limits = np.array([[0., 360.],[0., 1.],[0., 1.]])
    _perm = np.array([[0,1,2],[1,0,2],[2,0,1],[2,1,0],[1,2,0],[0,2,1]])
    _luma_vec = np.array([0.21, 0.72, 0.07])
    @staticmethod
    def _inverse():
        return ColorHCL2Inverse()

class ColorHCL2Inverse(ColorHCLInverse):
    """
    Convert RGB to HCL.

    Return:
      hue = [0, 360]
      chroma = [0, 1]
      luma =[0, 1]

    Use Y'709 = 0.21*R + 0.72*G + 0.07*B

    http://en.wikipedia.org/wiki/HSL_and_HSV#Color-making_attributes
    """
    _range = ColorHCL.limits
    _luma_vec = ColorHCL2._luma_vec
    @staticmethod
    def _inverse():
        return ColorHCL2()

#-----------------------------------------------------------------------

class ColorYIQ(ColorModelMatrix):
    """
    YIQ color model.

    y = [0, 1]
    |i| <= 0.596
    |q| <= 0.523

    'gray' value:  I = Q = 0
    """
    limits = np.array([[0., 1.],[-0.596, +0.596],[-0.523, +0.523]])
    _matrixI = np.array(
        [[0.299,  0.587,  0.114],
         [0.596, -0.275, -0.321],
         [0.212, -0.523,  0.311]])
    _matrix = np.linalg.inv(_matrixI)
    _gray_index = [1,2]
    _gray_value = [0,0]
    @staticmethod
    def _inverse():
        return ColorYIQInverse()

class ColorYIQInverse(ColorModelMatrix):
    """
    Convert RGB to YIQ.

    Return:
      y = [0, 1]
      |i| <= 0.596
      |q| <= 0.523
    """
    _range = ColorYIQ.limits
    _matrix = ColorYIQ._matrixI
    @staticmethod
    def _inverse():
        return ColorYIQ()

#-----------------------------------------------------------------------

class ColorYUV(ColorModelMatrix):
    """
    YUV color model.

    Input:
      y = [0, 1]
      |u| <= 0.436
      |v| <= 0.615

    Rec. 601

    http://en.wikipedia.org/wiki/YUV
    """
    limits = np.array([[0., 1.],[-0.436, +0.436],[-0.615, +0.615]])
    _matrix = np.array(
        [[ 1,  0      ,  1.13983],
         [ 1, -0.39465, -0.58060],
         [ 1,  2.03211,  0      ]])
    _gray_index = [1, 2]
    _gray_value = [0, 0]
    @staticmethod
    def _inverse():
        return ColorYUVInverse()

class ColorYUVInverse(ColorModelMatrix):
    """
    Convert RGB to YUV.

    Return:
      y = [0, 1]
      |u| <= 0.436
      |v| <= 0.615

    Rec. 601

    http://en.wikipedia.org/wiki/YUV
    """
    _range = ColorYUV.limits
    _matrix = np.array(
        [[+0.299  , +0.587   , +0.114  ],
         [-0.14713, -0.28886 , +0.463  ],
         [+0.615  , -0.551499, -0.10001]])
    @staticmethod
    def _inverse():
        return ColorYUV()

#-----------------------------------------------------------------------

class ColorYUV2(ColorModelMatrix):
    """
    Y'UV color model.

    Input:
      y = [0, 1]
      |u| <= 0.436 (?)
      |v| <= 0.615 (?)

    Rec. 709

    http://en.wikipedia.org/wiki/YUV
    """
    limits = np.array([[0., 1.],[-0.436, +0.436],[-0.615, +0.615]])
    _matrix = np.array(
        [[ 1,  0      ,  1.28033],
         [ 1, -0.21482, -0.38059],
         [ 1,  2.12798,  0      ]])
    _gray_index = [1, 2]
    _gray_value = [0, 0]
    @staticmethod
    def _inverse():
        return ColorYUV2Inverse()

class ColorYUV2Inverse(ColorModelMatrix):
    """
    Convert RGB to Y'UV.

    Return:
      |u| <= 0.436 (?)
      |v| <= 0.615 (?)

    Rec. 709

    http://en.wikipedia.org/wiki/YUV
    """
    _range = ColorYUV.limits
    _matrix = np.array(
        [[ 0.2126 ,  0.7152 ,  0.0722 ],
         [-0.09991, -0.33609,  0.436  ],
         [ 0.615  , -0.55861, -0.05639]])
    @staticmethod
    def _inverse():
        return ColorYUV2()

#-----------------------------------------------------------------------

class ColorYCgCo(ColorModelMatrix):
    """
    YCgCo color model.

    Input:
      y = [0, 1]
      |Cg| <= 0.5 (?)
      |Co| <= 0.5 (?)

    http://en.wikipedia.org/wiki/YCgCo
    """
    limits = np.array([[0., 1.], [-0.5, +0.5], [-0.5, +0.5]])
    _matrix = np.array(
        [[ 1, -1,  1],
         [ 1,  1,  0],
         [ 1, -1, -1]])
    _gray_index = [1, 2]
    _gray_value = [0, 0]
    @staticmethod
    def _inverse():
        return ColorYCgCoInverse()

class ColorYCgCoInverse(ColorModelMatrix):
    """
    Convert RGB to YCgCo.

    Return:

      |Cg| <= 0.5 (?)
      |Co| <= 0.5 (?)

      http://en.wikipedia.org/wiki/YCgCo
      """

    _range = ColorYCgCo.limits
    _matrix = np.array(
        [[ 0.25,  0.5,  0.25],
         [-0.25,  0.5, -0.25],
         [ 0.50,    0, -0.50]])
    @staticmethod
    def _inverse():
        return ColorYCgCo()

#-----------------------------------------------------------------------

class ColorYCbCr(ColorModelMatrix):
    """
    YCrCb color model.

    Input:
      y = floating-point value between 16 and 235
      Cb, Cr: floating-point values between 16 and 240
    """
    limits = np.array([[16., 235.],[16., 240.],[16., 240.]])
    _matrix = np.array(
        [[ 1,  0    ,  1.402  ],
         [ 1, -0.344, -0.714  ],
         [ 1, +1.772,  0      ]])
    @staticmethod
    def _inverse():
        return ColorYCbCrInverse()
    @classmethod
    def _transform(cls, a):
        return (np.inner(a, cls._matrix) - np.array([0., 128., 128.])[np.newaxis,:]) / 256.
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))

class ColorYCbCrInverse(ColorModelMatrix):
    """
    Convert RGB to YCbCr.

    Return:
      y = floating-point value between 16 and 235
      Cb, Cr: floating-point values between 16 and 240
    """
    _range = ColorYCbCr.limits
    _matrix = np.array(
        [[ 0.299,  0.587,  0.114  ],
         [-0.169, -0.331,  0.499  ],
         [ 0.499, -0.418, -0.0813 ]])
    @classmethod
    def _transform(cls, a):
        return np.inner(a * 256 + np.array([0., 128., 128.])[np.newaxis,:], cls._matrix)
    @staticmethod
    def _inverse():
        return ColorYCrCb()

#-----------------------------------------------------------------------

class ColorYDbDr(ColorModelMatrix):
    """
    YDrDb color model.

    Input:
      y = [0, 1]
      Db, Dr: [-1.333, +1.333]

    http://en.wikipedia.org/wiki/YDbDr
    """
    limits = np.array([[0., 1.],[-1.333,1.333],[-1.333,1.333]])
    _matrixI = np.array(
        [[ 0.299,  0.587,  0.114  ],
         [-0.450, -0.883, +1.333  ],
         [-1.333,  1.116,  0.217 ]])
    _matrix = np.linalg.inv(_matrixI)
    @staticmethod
    def _inverse():
        return ColorYDbDrInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))

class ColorYDbDrInverse(ColorModelMatrix):
    """
    Convert RGB to YDbDr.

    Return:
      y = [0, 1]
      Db, Dr: [-1.333, +1.333]

    http://en.wikipedia.org/wiki/YDbDr
    """
    _range = ColorYDbDr.limits
    _matrix = ColorYDbDr._matrixI
    @staticmethod
    def _inverse():
        return ColorYDrDb()

#-----------------------------------------------------------------------

class ColorYPbPr(ColorModelMatrix):
    """
    YPbPr color model.

    Input:
      y = [0, 1]
      Pb,Pr = [-0.5, 0.5]
    """
    limits = np.array([[0., 1.],[-0.5, 0.5],[-0.5, 0.5]])
    _R = 0.2126
    _G = 0.7152
    _B = 0.0722
    _matrixI = np.array(
        [[    _R,   _G,     _B],
         [  - _R, - _G, 1 - _B],
         [1 - _R, - _G,   - _B]])
    _matrix = np.linalg.inv(_matrixI)
    _gray_index = [1, 2]
    _gray_value = [0, 0]
    @staticmethod
    def _inverse():
        return ColorYPbPrInverse()

class ColorYPbPrInverse(ColorModelMatrix):
    """
    Convert RGB to YPbPr.

    Return:
      y = [0, 1]
      Pb,Pr = [-0.5, 0.5]
    """
    _range = ColorYPbPr.limits
    _matrix = ColorYPbPr._matrixI
    @staticmethod
    def _inverse():
        return ColorYPbPr()

#-----------------------------------------------------------------------

class ColorXYZ(ColorModelMatrix):
    """
    CIE XYZ color model.

    Input:
       X, Y, Z

       http://www.ryanjuckett.com/programming/rgb-color-space-conversion/

    White point E (x,y = 1/3, 1/3)
    """
    _scale = 1 # / 0.17697
    limits = np.array([[0., 1.],[0., 1.],[0., 1.]]) * _scale
    # _matrixI = np.array( # CIE 1931 white point E
    #     [[0.49   , 0.31   , 0.20   ],
    #      [0.17697, 0.81240, 0.01063],
    #      [0.00   , 0.01   , 0.99   ]]) * _scale
    # _matrix = np.linalg.inv(_matrixI)

    # _matrixI, _matrix = _make_transform(
    #     [0.73467, 0.26533], # CIE RGB primaries
    #     [0.27376, 0.71741],
    #     [0.16658, 0.00886],
    #     [1/3, 1/3], # CIE 1931 white point E
    #     )

    # _matrixI, _matrix = _make_transform(
    #     [0.73467, 0.26533], # CIE RGB primaries
    #     [0.27376, 0.71741],
    #     [0.16658, 0.00886],
    #     [0.3127,  0.3290], # D65
    #     )

    _matrixI, _matrix = _make_transform(
        [0.64, 0.33], # sRGB primaries
        [0.30, 0.60],
        [0.15, 0.06],
        [0.312713,  0.329016], # D65
        )

    @staticmethod
    def _inverse():
        return ColorXYZInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))

class ColorXYZInverse(ColorModelMatrix):
    """
    Convert RGB to XYZ.

    Return:
      X, Y, Z
    """
    _range = ColorXYZ.limits
    _matrix = ColorXYZ._matrixI
    @staticmethod
    def _inverse():
        return ColorXYZ()

#-----------------------------------------------------------------------

class ColorLMS(ColorModelMatrix):
    """
    CIE CAT02 LMS color model.

    Input:
       L, M, S

    TODO - add other normalizations
    http://en.wikipedia.org/wiki/LMS_color_space
    """
    _M = _m_CAT02

    _matrixI = np.array(np.inner(_M, ColorXYZ._matrixI.transpose()))
    limits = np.inner(_M,  ColorXYZ.limits.transpose())
    _matrix = np.linalg.inv(_matrixI)
    @staticmethod
    def _inverse():
        return ColorLMSInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))

class ColorLMSInverse(ColorModelMatrix):
    """
    Convert RGB to LMS.

    CIE CAT02 LMS color model.

    Return:
      L, M, S
    """
    _range = ColorLMS.limits
    _matrix = ColorLMS._matrixI
    @staticmethod
    def _inverse():
        return ColorLMS()

#-----------------------------------------------------------------------

class ColorxyY(ColorModel):
    """
    CIE xyY color model. (1931)

    Input:
       x, y, Y

       http://en.wikipedia.org/wiki/Chromaticity_coordinate
    """
    limits = np.array([[0., 1.],[0., 1.],ColorXYZ.limits[1]])
    @staticmethod
    def _inverse():
        return ColorxyYInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        x, y, Y, a, mode = cls._args_to_vectors(args)
        Yy = np.ones_like(y)
        ind = y != 0
        Yy[ind] = Y[ind] / y[ind]
        X = Yy * x
        Z = Yy * (1 - x - y)
        rgb = ColorXYZ()(np.vstack((X, Y, Z)).transpose())
        return cls._array_to_return(rgb, a, mode)

class ColorxyYInverse(ColorModel):
    """
    Convert RGB to xyY.

    Return:
      x, y, Y
    """
    _range = ColorxyY.limits
    @staticmethod
    def _inverse():
        return ColorxyY()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, a, mode = cls._args_to_vectors(args)
        X, Y, Z = ColorXYZInverse()(r, g, b)
        s = X + Y + Z
        ii = s != 0
        si = 1 / s[ii]
        x = np.tile(1 / 3, X.shape)
        y = x.copy()
        x[ii] = X[ii] * si
        y[ii] = Y[ii] * si
        return cls._vectors_to_return(x, y, Y, a, mode)

#-----------------------------------------------------------------------

class ColorLab(ColorModel):
    """
    CIE L*a*b* color model

    use D65 (6504 K)
    X=95.047, Y=100.00, Z=108.883
    http://en.wikipedia.org/wiki/CIE_Standard_Illuminant_D65
    http://en.wikipedia.org/wiki/Lab_color_space
    """
    _Xn =  95.047 / 100
    _Yn = 100.000 / 100
    _Zn = 108.883 / 100
    limits = np.array([[-1, 1], [-1, 1], [-1, 1.]])*np.inf
    limits[0, :] = [0, 100]
    @staticmethod
    def _inverse():
        return ColorLabInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))
    def _fn(x):
            ind = x > 6 / 29
            y = x.copy()
            y[ind] = x[ind]**3
            ind = np.logical_not(ind)
            y[ind] = 3 * (6 / 29)**2 * (x[ind] - 4 / 29)
            return y
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        L, a, b, alpha, mode = cls._args_to_vectors(args)
        Y = cls._Yn * cls._fn((L + 16) / 116)
        X = cls._Xn * cls._fn((L + 16) / 116 + a / 500)
        Z = cls._Zn * cls._fn((L + 16) / 116 - b / 200)
        rgb = ColorXYZ()(np.vstack((X, Y, Z)).transpose())
        return cls._array_to_return(rgb, alpha, mode)
    _gray_index = [1, 2]
    _gray_value = [0, 0]

class ColorLabInverse(ColorModel):
    """
    CIE L*a*b* color model - inverse

    use D65 (6504 K)
    X=95.047, Y=100.00, Z=108.883
    http://en.wikipedia.org/wiki/CIE_Standard_Illuminant_D65
    http://en.wikipedia.org/wiki/Lab_color_space
    """
    _range = ColorLab.limits
    _Xn = ColorLab._Xn
    _Yn = ColorLab._Yn
    _Zn = ColorLab._Zn
    @staticmethod
    def _inverse():
        return ColorLab()
    @staticmethod
    def _f(x):
        y = x.copy()
        ind = x > (6 / 29)**3
        y[ind] = x[ind]**(1 / 3)
        ind = np.logical_not(ind)
        y[ind] = 1 / 3 * (29 / 6)**2 * x[ind] + 4 / 29
        return y
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, alpha, mode = cls._args_to_vectors(args)
        X, Y, Z = ColorXYZInverse()(r, g, b)
        L = 116 *  cls._f(Y / cls._Yn) - 16
        a = 500 * (cls._f(X / cls._Xn) - cls._f(Y / cls._Yn))
        b = 200 * (cls._f(Y / cls._Yn) - cls._f(Z / cls._Zn))
        return cls._vectors_to_return(L, a, b, alpha, mode)

#-----------------------------------------------------------------------

class ColorMsh(ColorModel):
    """
    Color Model from Kenneth Moreland for diverging color maps.
    """
    limits = np.array([[0., np.inf], [0., 90.], [0, 360.]])
    _deg2rad = np.pi / 180
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        M, s, h, alpha, mode = cls._args_to_vectors(args)
        s = s * cls._deg2rad
        h = h * cls._deg2rad
        cs = np.cos(s)
        ss = np.sin(s)
        ch = np.cos(h)
        sh = np.sin(h)
        L = M * cs
        a = M * ss * ch
        b = M * ss * sh
        r, g, b = ColorLab()(L, a, b)
        return cls._vectors_to_return(r, g, b, alpha, mode)
    @staticmethod
    def _inverse():
        return ColorMshInverse()
    _gray_index = [1, 2]
    _gray_value = [0, 0]

class ColorMshInverse(ColorModel):
    """
    Color Model from Kenneth Moreland for diverging color maps.
    """
    _range = ColorMsh.limits
    _rad2deg = 1 / ColorMsh._deg2rad
    @staticmethod
    def _inverse():
        return ColorMsh()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, alpha, mode = cls._args_to_vectors(args)
        L, a, b = ColorLabInverse()(r, g, b)
        M = np.sqrt(L**2 + a**2 + b**2)
        ii = M > 0
        s = np.zeros_like(r)
        s[ii] = np.arccos(L[ii] / M[ii]) * cls._rad2deg
        ii = np.logical_or(a != 0, b != 0)
        h = np.zeros_like(r)
        h[ii] = np.mod(np.arctan2(b[ii], a[ii]) * cls._rad2deg, 360)
        return cls._vectors_to_return(M, s, h, alpha, mode)

#-----------------------------------------------------------------------

class ColorMsh2(ColorModel):
    """
    4-pi polar color model based on Lab, Alexander Heger

    centered at L, a, b = (50, 0, 0)

    Does not seem very useful.
    """
    limits = np.array([[0., np.inf], [0., 180.], [0., 360.]])
    _deg2rad = np.pi / 180
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        M, s, h, alpha, mode = cls._args_to_vectors(args)
        s = s * cls._deg2rad
        h = h * cls._deg2rad
        cs = np.cos(s)
        ss = np.sin(s)
        ch = np.cos(h)
        sh = np.sin(h)
        L = 50 - M * cs
        a = M * ss * ch
        b = M * ss * sh
        r, g, b = ColorLab()(L, a, b)
        return cls._vectors_to_return(r, g, b, alpha, mode)
    @staticmethod
    def _inverse():
        return ColorMsh2Inverse()
    @classmethod
    def _gray_func(cls, g):
        ccc = np.zeros(g.shape + (3,))
        L = 100 * g - 50
        M = np.abs(L)
        ccc[:, 0] = M
        ccc[L > 0, 1] = 180
        return ccc

class ColorMsh2Inverse(ColorModel):
    """
    4-pi polar color model based on Lab, Alexander Heger

    centered at L, a, b = (50, 0, 0)

    Does not seem very useful.
    """
    _range = ColorMsh2.limits
    _rad2deg = 1 / ColorMsh2._deg2rad
    @staticmethod
    def _inverse():
        return ColorMsh2()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, alpha, mode = cls._args_to_vectors(args)
        L, a, b = ColorLabInverse()(r, g, b)
        L -= 50
        M = np.sqrt(L**2 + a**2 + b**2)
        ii = M > 0
        s = np.zeros_like(r)
        s[ii] = np.arccos(- L[ii] / M[ii]) * cls._rad2deg
        ii = np.logical_or(a != 0, b != 0)
        h = np.zeros_like(r)
        h[ii] = np.mod(np.arctan2(b[ii], a[ii]) * cls._rad2deg, 360)
        return cls._vectors_to_return(M, s, h, alpha, mode)

#-----------------------------------------------------------------------

class ColorLab2(ColorModel):
    """
    Hunter/Adams Lab color model

    use D65 (6504 K)
    X=95.047, Y=100.00, Z=108.883
    http://en.wikipedia.org/wiki/CIE_Standard_Illuminant_D65
    http://en.wikipedia.org/wiki/Lab_color_space
    """
    _Xn =  95.047 / 100
    _Yn = 100.000 / 100
    _Zn = 108.883 / 100

    _Ka = 175 / 198.04 * (_Xn + _Yn)
    _Kb =  70 / 218.11 * (_Yn + _Zn)
    _K  = _Ka # / 100
    _ke = _Kb / _Ka

    limits = np.array([[-1,1],[-1,1],[-1,1.]])*np.inf
    @staticmethod
    def _inverse():
        return ColorLab2Inverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        L, a, b, alpha, mode = cls._args_to_vectors(args)
        ind = L != 0
        ca = np.ones_like(L)
        cb = np.ones_like(L)
        Y = (L / 100)**2 * cls._Yn
        ca[ind] = a[ind] / (cls._K * L[ind])
        cb[ind] = b[ind] / (cls._K * L[ind])
        X = (ca + 1) * (Y / cls._Yn) * cls._Xn
        Z = (1 - (cb / cls._ke))  * (Y / cls._Yn) * cls._Zn
        rgb = ColorXYZ()(np.vstack((X, Y, Z)).transpose())
        return cls._array_to_return(rgb, alpha, mode)
    _gray_index = [1, 2]
    _gray_value = [0, 0]

class ColorLab2Inverse(ColorModel):
    """
    Hunter/Adams Lab color model

    use D65 (6504 K)
    X=95.047, Y=100.00, Z=108.883
    http://en.wikipedia.org/wiki/CIE_Standard_Illuminant_D65
    http://en.wikipedia.org/wiki/Lab_color_space
    """
    _range = ColorLab.limits
    _Xn = ColorLab2._Xn
    _Yn = ColorLab2._Yn
    _Zn = ColorLab2._Zn

    _Ka = ColorLab2._Ka
    _Kb = ColorLab2._Kb
    _K =  ColorLab2._K
    _ke = ColorLab2._ke

    @staticmethod
    def _inverse():
        return ColorLab2()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, alpha, mode = cls._args_to_vectors(args)
        X, Y, Z = ColorXYZInverse()(r, g, b)
        ind = Y != 0
        ca = np.ones_like(Y)
        cb = np.ones_like(Y)
        L = 100 * np.sqrt(Y / cls._Yn)
        ca[ind] =      (X[ind] / cls._Xn) / (Y[ind] / cls._Yn) - 1
        cb[ind] = cls._ke * (1 - (Z[ind] / cls._Zn) / (Y[ind] / cls._Yn))
        a = cls._K * L * ca
        b = cls._K * L * cb
        return cls._vectors_to_return(L, a, b, alpha, mode)

#-----------------------------------------------------------------------

class ColorLuv(ColorModel):
    """
    CIE LUV color model (L*, u*, v*)

    Use D65
    http://en.wikipedia.org/wiki/Standard_illuminant

    0 <= L <= 100
    U, V typically \pm 100

    TODO - XYZ needs to use same primary and illuminant
         ... actually, just do its own transform to XYZ
    """
    # C (6774 K)
    # For 2^\deg observer and standard illuminant C, u'n = 0.2009, v'n = 0.4610.
    # upn = 0.2009, vpn = 0.4610,  Yn = 0.54
    # http://en.wikipedia.org/wiki/CIELUV
    # _upn = 0.2009
    # _vpn = 0.4610
    # _Yn  = 0.54

    # D65
    _Yn = 1
    _Xn = 0.95047
    _Zn = 1.08883
    _upn = 4 * _Xn / (_Xn + 15 * _Yn + 3 * _Zn)
    _vpn = 9 * _Yn / (_Xn + 15 * _Yn + 3 * _Zn)

    limits = np.array([[0, 100], [-100, 100], [-100, 100.]])
    limits[1:3,:] *= np.inf
    limits[0,1] /= _Yn
    @staticmethod
    def _inverse():
        return ColorLuvInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        L, u, v, alpha, mode = cls._args_to_vectors(args)
        ii = L != 0
        up = np.zeros_like(L)
        vp = np.zeros_like(L)
        up[ii] = u[ii] / (13 * L[ii]) + cls._upn
        vp[ii] = v[ii] / (13 * L[ii]) + cls._vpn
        X = np.zeros_like(L)
        Y = np.zeros_like(L)
        Z = np.zeros_like(L)
        ii = L <= 8
        Y[ii] = cls._Yn * L[ii] * (3/29)**3
        ii = np.logical_not(ii)
        Y[ii] = cls._Yn * ((L[ii] + 16) / 116)**3
        ii = vp != 0
        X[ii] = Y[ii] * 9 * up[ii] / (4 * vp[ii])
        Z[ii] = Y[ii] * (12 - 3 * up[ii] - 20 * vp[ii]) / (4 * vp[ii])
        rgb = ColorXYZ()(np.vstack((X, Y, Z)).transpose())
        return cls._array_to_return(rgb, alpha, mode)

class ColorLuvInverse(ColorModel):
    __doc__ = "Inverse color model\n" + ColorLuv.__doc__

    _range = ColorLab.limits
    _Yn = ColorLuv._Yn
    _upn = ColorLuv._upn
    _vpn = ColorLuv._vpn

    @staticmethod
    def _inverse():
        return ColorLuv()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, alpha, mode = cls._args_to_vectors(args)
        X, Y, Z = ColorXYZInverse()(r, g, b)
        L = np.zeros_like(r)
        ii = Y <= (6 / 29) **3 * cls._Yn
        L[ii] = (29 / 3)**3 * Y[ii] / cls._Yn
        ii = np.logical_not(ii)
        L[ii] = 116 * (Y[ii] / cls._Yn)**(1/3) - 16
        xyz = X + 15 * Y + 3 * Z
        ii = xyz != 0
        up = np.zeros_like(r)
        vp = np.zeros_like(r)
        up[ii] = 4 * X[ii] / xyz[ii]
        vp[ii] = 9 * Y[ii] / xyz[ii]
        u = 13 * L * (up - cls._upn)
        v = 13 * L * (vp - cls._vpn)
        return cls._vectors_to_return(L, u, v, alpha, mode)

#-----------------------------------------------------------------------

class ColorUVW(ColorModel):
    """
    CIE UVW color model (1960)

    http://en.wikipedia.org/wiki/CIE_1960_color_space
    """
    limits = ColorXYZ.limits * np.array([2/3, 1, 2])[:, np.newaxis]
    @staticmethod
    def _inverse():
        return ColorUVWInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        U, V, W, alpha, mode = cls._args_to_vectors(args)
        X = 1.5 * U
        Y = V
        Z = 1.5 * U - 3 * V + 2 * W
        rgb = ColorXYZ()(np.vstack((X, Y, Z)).transpose())
        return cls._array_to_return(rgb, alpha, mode)

class ColorUVWInverse(ColorModel):
    __doc__ = "Inverse color model\n" + ColorUVW.__doc__

    _range = ColorUVW.limits

    @staticmethod
    def _inverse():
        return ColorUVW()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, alpha, mode = cls._args_to_vectors(args)
        X, Y, Z = ColorXYZInverse()(r, g, b)
        U = 2 / 3 * X
        V = Y
        W = 0.5 * (3 * Y - X + Z)
        return cls._vectors_to_return(U, V, W, alpha, mode)

#-----------------------------------------------------------------------

class ColorYuv0(ColorModel):
    """
    CIE Yuv color model (1960)

    http://en.wikipedia.org/wiki/CIE_1960_color_space
    http://www.poynton.com/PDFs/coloureq.pdf, section 5.2
    """
    limits = np.array([ColorxyY.limits[0], [0, 1], [0, 1]])
    @staticmethod
    def _inverse():
        return ColorYuv0Inverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        Y, u, v, alpha, mode = cls._args_to_vectors(args)
        f = 2 * u - 8 * v + 4
        ii = f != 0
        x = np.tile(1/3, Y.shape)
        y = x.copy()
        x[ii] = 3 * u / f
        y[ii] = 2 * v / f
        rgb = ColorxyY()(np.vstack((x, y, Y)).transpose())
        return cls._array_to_return(rgb, alpha, mode)

class ColorYuv0Inverse(ColorModel):
    __doc__ = "Inverse color model\n" + ColorYuv0.__doc__

    _range = ColorYuv0.limits

    @staticmethod
    def _inverse():
        return ColorYuv0()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, alpha, mode = cls._args_to_vectors(args)
        x, y, Y = ColorxyYInverse()(r, g, b)
        f = 12 * y - 2 * x + 3
        ii = f != 0
        u = np.zeros_like(r)
        v = np.zeros_like(r)
        u[ii] = 4 * x / f
        v[ii] = 6 * y / f
        return cls._vectors_to_return(Y, u, v, alpha, mode)

#-----------------------------------------------------------------------

class ColorYuv(ColorModel):
    """
    CIE Yu'v' color model (1976)

    based on Yuv:
      u' = u
      v' = 1.5 * v

    http://www.poynton.com/PDFs/coloureq.pdf, section 5.3
    """
    limits = np.array([ColorxyY.limits[0], [0, 1], [0, 1]])
    @staticmethod
    def _inverse():
        return ColorYuvInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        Y, u, v, alpha, mode = cls._args_to_vectors(args)
        f = 6 * u - 16 * v + 12
        ii = f != 0
        x = np.tile(1/3, Y.shape)
        y = x.copy()
        x[ii] = 9 * u / f
        y[ii] = 4 * v / f
        rgb = ColorxyY()(np.vstack((x, y, Y)).transpose())
        return cls._array_to_return(rgb, alpha, mode)

class ColorYuvInverse(ColorModel):
    __doc__ = "Inverse color model\n" + ColorYuv.__doc__

    _range = ColorYuv.limits

    @staticmethod
    def _inverse():
        return ColorYuv()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, alpha, mode = cls._args_to_vectors(args)
        x, y, Y = ColorxyYInverse()(r, g, b)
        f = 12 * y - 2 * x + 3
        ii = f != 0
        u = np.zeros_like(r)
        v = np.zeros_like(r)
        u[ii] = 4 * x / f
        v[ii] = 9 * y / f
        return cls._vectors_to_return(Y, u, v, alpha, mode)

#-----------------------------------------------------------------------

class ColorUVW2(ColorModel):
    """
    CIE U*V*W* color model (1964)

    based on Yuv.  Obsolote, except used for CRI calculation

    For 2^\deg observer and standard illuminant C, un = 0.2009, vn = 0.3073.

    use C (6774 K)
    http://en.wikipedia.org/wiki/Standard_illuminant

    http://en.wikipedia.org/wiki/CIE_1964_color_space
    """
    limits = np.array([[-100,100], [-100, 100], [-17, 100.]])
    _un = 0.2009
    _vn = 0.3073
    @staticmethod
    def _inverse():
        return ColorUVW2Inverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        U, V, W, alpha, mode = cls._args_to_vectors(args)
        Y = ((W + 17) / 25) **3
        ii = W != 0
        u = np.tile(cls._un, U.shape)
        v = np.tile(cls._vn, U.shape)
        u[ii] = U[ii] / (13 * W[ii]) + cls._un
        v[ii] = V[ii] / (13 * W[ii]) + cls._vn
        rgb = ColorYuv0()(np.vstack((Y, u, v)).transpose())
        return cls._array_to_return(rgb, alpha, mode)

class ColorUVW2Inverse(ColorModel):
    __doc__ = "Inverse color model\n" + ColorUVW2.__doc__

    _range = ColorUVW2.limits
    _un = ColorUVW2._un
    _vn = ColorUVW2._vn

    @staticmethod
    def _inverse():
        return ColorUVW2()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, alpha, mode = cls._args_to_vectors(args)
        Y, u, v = ColorYuv0Inverse()(r, g, b)
        W = 25 * Y**(1/3) - 17
        V = 13 * W * (v - cls._vn)
        U = 13 * W * (u - cls._un)
        return cls._vectors_to_return(U, V, W, alpha, mode)

#-----------------------------------------------------------------------

class ColorLChuv(ColorModel):
    """
    LCh_uv color model.

    Cylindrical representation of CIE LUV color model (L*, u*, v*)
    """
    limits = ColorLuv.limits.copy()
    limits[2,:] = [0, 360.]
    limits[1,0] = 0
    _deg2rad = np.pi / 180
    @staticmethod
    def _inverse():
        return ColorLChuvInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        L, C, huv, alpha, mode = cls._args_to_vectors(args)
        phi = huv * cls._deg2rad
        U = C * np.cos(phi)
        V = C * np.sin(phi)
        rgb = ColorLuv()(np.vstack((L, U, V)).transpose())
        return cls._array_to_return(rgb, alpha, mode)

class ColorLChuvInverse(ColorModel):
    __doc__ = "Inverse color model\n" + ColorLChuv.__doc__

    _range = ColorLChuv.limits
    _rad2deg = 180 / np.pi

    @staticmethod
    def _inverse():
        return ColorLChuv()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, alpha, mode = cls._args_to_vectors(args)
        L, U, V = ColorLuvInverse()(r, g, b)
        C = np.sqrt(U**2 + V**2)
        ii = C > 0
        huv = np.zeros_like(L)
        huv[ii] = np.mod(np.arctan2(V[ii], U[ii]) * cls._rad2deg, 360)
        return cls._vectors_to_return(L, C, huv, alpha, mode)

#-----------------------------------------------------------------------

class ColorLCH(ColorModel):
    """
    L*CH color model.

    Cylindrical representation of CIE L*a*b* color model
    """
    limits = ColorLab.limits.copy()
    limits[2,:] = [0, 360.]
    limits[1,0] = 0
    _deg2rad = np.pi / 180
    @staticmethod
    def _inverse():
        return ColorLCHInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        L, C, H, alpha, mode = cls._args_to_vectors(args)
        phi = H * cls._deg2rad
        a = C * np.cos(phi)
        b = C * np.sin(phi)
        rgb = ColorLab()(np.vstack((L, a, b)).transpose())
        return cls._array_to_return(rgb, alpha, mode)

class ColorLCHInverse(ColorModel):
    __doc__ = "Inverse color model\n" + ColorLCH.__doc__

    _range = ColorLCH.limits
    _rad2deg = 180 / np.pi

    @staticmethod
    def _inverse():
        return ColorLCH()
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, alpha, mode = cls._args_to_vectors(args)
        L, a, b = ColorLabInverse()(r, g, b)
        C = np.sqrt(a**2 + b**2)
        ii = C > 0
        H = np.zeros_like(L)
        H[ii] = np.mod(np.arctan2(b[ii], a[ii]) * cls._rad2deg, 360)
        return cls._vectors_to_return(L, C, H, alpha, mode)

#-----------------------------------------------------------------------
# there seems to be an issue that LMS can return negative values.

class ColorCAM(ColorModel):
    """
    CIECAM02

    http://en.wikipedia.org/wiki/CIECAM02

    """
    limits = np.array([[-1,1],[-1,1],[-1,1.]])*np.inf
    _LW = 100 # cd/m^2
    _Yb =  20 # luminace of background
    _Yw = 100 # luminace of reference white
    _LA = _LW * _Yb / _Yw # suppsed to be LW/5 for 'gray'

    _F = 1 # factor determining degree of adaptation

    _D = _F * (1 - 1 / 3.6 * np.exp((-_LA + 42) / 92)) # The degree of adaptation
    # _D = 0 # no adaptation

    # reference white
    _Lwr = _Mwr = _Swr = _Ywr = 100
    # illuminant white
    _Lw = _Mw = _Sw = _Yw = 100

    _fL = (1 + (_Yw * _Lwr / (_Ywr * _Lw) - 1)* _D)
    _fM = (1 + (_Yw * _Mwr / (_Ywr * _Mw) - 1)* _D)
    _fS = (1 + (_Yw * _Swr / (_Ywr * _Sw) - 1)* _D)

    _M = _m_CAT02

    _MH = np.array(
        [[ 0.38971, 0.68898, -0.07868],
         [-0.22981, 1.18340,  0.04641],
         [ 0.00000, 0.00000,  1.00000]])

    _ML = np.inner(_M, np.linalg.inv(_MH).transpose())

    _k = 1 / (5 * _LA + 1)
    _FL = 1 / 5 * _k**4*(5 * _LA) + 1 / 10 * (1 - _k**4)**2 * (5 * _LA)**(1 / 3)

    @staticmethod
    def _inverse():
        return ColorCAMInverse()
    @classmethod
    def gray(cls, *args):
        __doc__ = ColorModel.gray.__doc__
        return cls.inverse(ColorRGB().gray(*args))
    @classmethod
    def _fn(cls, y):
        y1 = y - 0.1
        xp = y1 * 27.13 / (400 - y1)
        x = xp **(1 / 0.42) * 100 / cls._FL
        return x
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        Lap,Map,Sap, a, mode = cls._args_to_vectors(args)
        Lp = cls._fn(Lap)
        Mp = cls._fn(Map)
        Sp = cls._fn(Sap)
        Lc, Mc, Sc = np.inner(cls._ML, np.array([Lp, Mp, Sp]).transpose())
        L = Lc / cls._fL
        M = Mc / cls._fM
        S = Sc / cls._fS
        r,g,b = ColorLMS()(L, M, S)
        return cls._vectors_to_return(r, g, b, a, mode)

class ColorCAMInverse(ColorModel):
    """
    CIE CAM02

    http://en.wikipedia.org/wiki/CIECAM02

    """
    _range = ColorCAM.limits

    _LW = 100 # cd/m^2
    _Yb =  20 # luminace of background
    _Yw = 100 # luminace of reference white
    _LA = _LW * _Yb / _Yw # suppsed to be LW/5 for 'gray'

    _F = 1. # factor determining degree of adaptation

    _D = _F * (1 - 1 / 3.6 * np.exp((-_LA + 42) / 92)) # The degree of adaptation
    # _D = 0 # no adaptation

    # reference white
    _Lwr = _Mwr = _Swr = _Ywr = 100
    # illuminant white
    _Lw = _Mw = _Sw = _Yw = 100

    _fL = (1 + (_Yw * _Lwr / (_Ywr * _Lw) - 1)* _D)
    _fM = (1 + (_Yw * _Mwr / (_Ywr * _Mw) - 1)* _D)
    _fS = (1 + (_Yw * _Swr / (_Ywr * _Sw) - 1)* _D)

    _MH = np.array(
        [[ 0.38971, 0.68898, -0.07868],
         [-0.22981, 1.18340,  0.04641],
         [ 0.00000, 0.00000,  1.00000]])

    _M = _m_CAT02

    _ML =  np.inner(_MH, np.linalg.inv(_M).transpose())

    _k = 1 / (5 * _LA + 1)
    _FL = 1 / 5 * _k**4*(5 * _LA) + 1/10 * (1 - _k**4)**2 * (5 * _LA)**(1 / 3)

    @staticmethod
    def _inverse():
        return ColorCAM()
    @classmethod
    def _f(cls, x):
        xp = (cls._FL * x / 100)**0.42
        y = 400 * xp / (27.13 + xp) + 0.1
        return y
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        r, g, b, a, mode = cls._args_to_vectors(args)
        L, M, S = ColorLMSInverse()(r,g,b)

        Lc = cls._fL * L
        Mc = cls._fM * M
        Sc = cls._fS * S

        Lp, Mp, Sp = np.clip(np.inner(cls._ML, np.array([Lc, Mc, Sc]).transpose()), 0, np.inf)

        Lap = cls._f(Lp)
        Map = cls._f(Mp)
        Sap = cls._f(Sp)

        return cls._vectors_to_return(Lap, Map, Sap, a, mode)

#-----------------------------------------------------------------------

class ColorsRGB(ColorModel):
    """
    Convert sRGB to RGB

    sRGB color model ICE 61966-2-1

    http://en.wikipedia.org/wiki/SRGB

    see also http://www.labri.fr/perso/granier/Cours/IOGS/color/ciexyz29082000.pdf
    """
    _matrix, _matrixI = _make_transform(
        [0.64, 0.33], # sRGB
        [0.30, 0.60],
        [0.15, 0.06],
        [0.31271, 0.32902], # D65 white point
        )

    _gamma = 2.4
    _a = 0.055
    _K0 = 0.4045 #  IEC 61966-2-1
    _phi = 12.92
    # more accurate
    _K0 = 0.040482
    # smooth transition:
    _K0  =  0.0392857
    _phi = 12.9232102
    @classmethod
    def _s(cls, x):
        mask = x > cls._K0
        x[mask] = ((x[mask] + cls._a) / (1 + cls._a))**cls._gamma
        mask = np.logical_not(mask)
        x[mask] /= cls._phi
        return np.clip(x, 0, 1, out = x)
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        sRGB, a, mode = cls._args_to_array(args)
        sRGB_linear = cls._s(sRGB)
        XYZ = np.inner(sRGB_linear, cls._matrix)
        rgb = ColorXYZ()(XYZ)
        return cls._array_to_return(rgb, a, mode)
    @staticmethod
    def _inverse():
        return ColorsRGBInverse()

class ColorsRGBInverse(ColorModel):
    """
    Convert RGB to sRGB.

    Inverse of ...
    """ + ColorsRGB.__doc__

    _matrix = ColorsRGB._matrixI
    _gamma  = ColorsRGB._gamma
    _a      = ColorsRGB._a
    _K0     = ColorsRGB._K0
    _phi    = ColorsRGB._phi

    @classmethod
    def _s(cls, x):
        mask = x > cls._K0 / cls._phi
        x[mask] = x[mask]**(1 / cls._gamma) * (1 + cls._a) - cls._a
        mask = np.logical_not(mask)
        x[mask] = x[mask] * cls._phi
        return np.clip(x, 0, 1, out = x)
    @classmethod
    def __call__(cls, *args):
        __doc__ = ColorModel.__call__.__doc__
        rgb, a, mode = cls._args_to_array(args)
        XYZ = ColorXYZInverse()(rgb)
        sRGB_linear = np.inner(XYZ, cls._matrix)
        sRGB = cls._s(sRGB_linear)
        return cls._array_to_return(sRGB, a, mode)
    @staticmethod
    def _inverse():
        return ColorsRGB()

#-----------------------------------------------------------------------

class ColorAdobeRGB(ColorsRGB):
    """
    Adobe RGB color model.  Color Scaling only.

    TODO XYZ transform

    http://en.wikipedia.org/wiki/Adobe_RGB_color_space
    http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf
    """
    _gamma = 563/256

    _matrix = np.array([
        [ 2.04159, -0.56501, -0.34473],
        [-0.96924,  1.87597,  0.04156],
        [ 0.01344, -0.11836,  1.01517],
        ])

    @classmethod
    def _s(cls, x):
        return x**cls._gamma
    @staticmethod
    def _inverse():
        return ColorAdobeRGBInverse()

class ColorAdobeRGBInverse(ColorsRGBInverse):
    """
    Convert RGB to Adobe RGB.

    """
    _gamma = 1 / ColorAdobeRGB._gamma

    _matrix = np.array([
        [0.57667, 0.18556, 0.18823],
        [0.29734, 0.62736, 0.07529],
        [0.02703, 0.07069, 0.99134],
        ])

    @classmethod
    def _s(cls, x):
        return x**cls._gamma
    @staticmethod
    def _inverse():
        return ColorAdobeRGB()

#-----------------------------------------------------------------------

class ColorI1I2I3(ColorModelMatrix):
    """
    I_1 I_2 I_3 Color Space

    http://faculty.kfupm.edu.sa/ICS/lahouari/Teaching/colorspacetransform-1.0.pdf

    Y. I. Ohta, T. Kanade, and T. Sakai, "Color information for region
    segmentation,"  Computer Graphics and Image Processing, vol. 13,
    pp. 222-241, 1980.
    """

    _matrixI = np.array([
        [ 1/3, 1/3, 1/3],
        [ 1/2,  0, -1/2],
        [-1/4, 1/2,-1/4],
        ])

    _matrix = np.linalg.inv(_matrixI)

    _range = np.array([[0,1], [0,1], [0, 1.]])
    limits = np.array([[0,1], [-0.5, 0.5], [-0.5, 0.5]])

    @staticmethod
    def _inverse():
        return ColorI1I2I3Inverse()

class ColorI1I2I3Inverse(ColorModelMatrix):
    """
    I_1 I_2 I_3 Color Space

    http://faculty.kfupm.edu.sa/ICS/lahouari/Teaching/colorspacetransform-1.0.pdf

    Y. I. Ohta, T. Kanade, and T. Sakai, "Color information for region
    segmentation," Computer Graphics and Image Processing, vol. 13,
    pp. 222-241, 1980.  """

    _matrix = ColorI1I2I3._matrixI

    limits = ColorI1I2I3._range
    _range = ColorI1I2I3.limits

    @staticmethod
    def _inverse():
        return ColorI1I2I3()

#-----------------------------------------------------------------------

class ColorLSLM(ColorModelMatrix):
    """
    Convert LSLM to RGB.

    Linear transformation of RGB based on the opponent signals
    of the cones: black-white, red-green, and yellow-blue.

    http://faculty.kfupm.edu.sa/ICS/lahouari/Teaching/colorspacetransform-1.0.pdf
    """
    _matrixI = np.array([
        [0.209,  0.715,  0.076],
        [0.209,  0.715, -0.924],
        [3.148, -2.799, -0.349]
        ])

    _matrix = np.linalg.inv(_matrixI)

    _offset1 = np.array([0.5]*3)

    limits = np.array([[-0.5, 0.5], [-.924, +0.924], [ -3.148, +3.148]])
    _range = np.array([[0,1], [0,1], [0,1]])

    @staticmethod
    def _inverse():
        return ColorLSLMInverse()
    _gray_index = [  1,   2]
    _gray_value = [0.5, 0.5]


class ColorLSLMInverse(ColorModelMatrix):
    """
    Convert RGB to LSLM.
    """
    _matrix = ColorLSLM._matrixI
    _offset0 = -ColorLSLM._offset1
    _range = ColorLSLM.limits
    limits = ColorLSLM._range
    @staticmethod
    def _inverse():
        return ColorLSLM()

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

# register color models

_color_models = dict()
def register_color_model(name, model):
    assert isinstance(name, str)
    assert issubclass(type(model), ColorModel)
    _color_models[name] = model
def color_models():
    return _color_models
def color_model(model = 'RGB'):
    if isinstance(model, ColorModel):
        return model
    if isinstance(model, type) and issubclass(model, ColorModel):
        return model()
    return _color_models[model]

register_color_model('RGB', ColorRGB())
register_color_model('CMY', ColorCMY())
register_color_model('HSV', ColorHSV())
register_color_model('HSL', ColorHSL())
register_color_model('HSI', ColorHSI())
register_color_model('HCL', ColorHCL()) # Rev 601, seems also go as 'yCH'
register_color_model('HCL2', ColorHCL()) # Rev 709, seems also go as 'yCH'
register_color_model('YIQ', ColorYIQ())
register_color_model('YUV', ColorYUV()) # Rev 601
register_color_model('YUV2', ColorYUV2()) # Rev 709
register_color_model('YCbCr', ColorYCbCr())
register_color_model('YCgCo', ColorYCgCo())
register_color_model('YDbDr', ColorYDbDr())
register_color_model('YPbPr', ColorYPbPr())
register_color_model('I1I2I3', ColorI1I2I3())
register_color_model('LSLM', ColorLSLM())
register_color_model('XYZ', ColorXYZ()) # CIE XYZ
register_color_model('LMS', ColorLMS()) # CIE CAM 02 LMS
register_color_model('xyY', ColorxyY()) # CIE xyY
register_color_model('Yuv0', ColorYuv0()) # CIE Yuv (1960), obsolete
register_color_model('Yuv', ColorYuv()) # CIE Yu'v' (1976)
register_color_model('UVW', ColorUVW()) # CIE UVW
register_color_model('UVW2', ColorUVW2()) # CIE U*V*W* (1964), obsolete (except used for CRI calculation)
register_color_model('Lab', ColorLab()) # CIE L*a*b*, 6504 K
register_color_model('LCH', ColorLCH()) # CIE L*CH -  cylindrical representation of L*a*b*, 6504 K
register_color_model('Lab2', ColorLab2()) # Hunter Lab, 6504 K
register_color_model('Msh', ColorMsh()) # Polar representation of CIE L*a*b* (K Moreland, 2 pi)
register_color_model('Msh2', ColorMsh2()) # Polar representation of CIE L*a*b* (4 pi)
register_color_model('Luv', ColorLuv()) # CIE LUV, 6774 K
register_color_model('LChuv', ColorLChuv()) # cylindrical representation of CIE LUV, 6774 K
register_color_model('CAM', ColorCAM()) # CIE CAM 02
register_color_model('sRGB', ColorsRGB()) # IEC 61966-2-1
register_color_model('AdobeRGB', ColorAdobeRGB()) # Adobe RGB
