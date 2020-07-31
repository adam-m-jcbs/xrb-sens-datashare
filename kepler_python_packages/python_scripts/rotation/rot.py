"""
Provides a rotation class and utils.

Implements different algorithms and optimisations.

Currently, QERRotator seems best, closely followed by ERRotator.

QuatRotator seems to come in next, though it is a bit clunky code
due to limited support for `.imag` on numpy ndarrays and functions to
create quaternions with just imaginary part with a direct helper
function.  Should write my own.
"""

import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm
import quaternion as quat
import math
try:
    import numba
except:
    print('[rot] WARNING: Numba not found.')
    class numba:
        @staticmethod
        def jit(*args, **kwargs):
            def no_jit(f):
                print(f'[rot] WARNING: not optimising "{f.__name__}".')
                return f
            return no_jit

e0 = np.asarray([0., 0., 0.])
ex = np.asarray([1., 0., 0.])
ey = np.asarray([0., 1., 0.])
ez = np.asarray([0., 0., 1.])
basis = np.asarray([ex, ey, ez])
rad2deg = 180/np.pi
deg2rad = np.pi/180

class NoRotator(object):
    def __init__(self, *args, **kwargs):
        pass
    def rotate(self, v, *args, **kwargs):
        return v
    def __call__(self, *args, **kwargs):
        return self.rotate(*args, **kwargs)
    def update(self, *args, **kwargs):
        pass

class BaseRotator(object):
    """
    Basic Rotator functionallities.

    Allow np.ndarrays for time, rotation vectors, and vectors.  For
    the latter twon it is assume the vector 3-dimension is tha last
    one of the array.

    The return eattay will first have time dimensions, then rotator
    dimensions, and last vector dimension, terminated by the final
    3-dimension.

    The `align` parmeter allwos vectorisation of of the first `align`
    dimensions of the vector array with the last `align` non-vector
    dimensions of the omega array.

    The `phase` parameter allows to add a phase offset to the rotation
    matices.  It needs to be broadcastbale to the non-vector dimensions
    of the omega array.

    TODO: Use tuples to allow alignment with arbitraty axes, and with
    time axes.
    """

    void = object()

    def __init__(self, omega, time = None, phase = None):
        self.omega = np.asarray(omega, dtype = np.float64)
        self.time = None
        self.phase = None
        self.init()
        self.settimeandphase(time, phase)

    def init(self):
        pass

    def settime(self, time):
        return self.settimeandphase(time = time, phase = None)

    def setphase(self, phase):
        return self.settimeandphase(time = None, phase = phase)

    def settimeandphase(self, time = None, phase = None):
        if time is not None:
            self.time = np.asarray(time)
        if phase is not None:
            phase = np.asarray(phase)
            self.phase = phase / LA.norm(self.omega, axis = -1)
        if ((time is not None or phase is not None) and
            (time is not None or self.time is not None)):
            self.update()
        return self

    def update(self):
        raise NotImplementedError()

    def rotate(self, v):
        raise NotImplementedError()

    def __call__(self, v, time = None, phase = None, align = 0):
        if self.time is None and time is None:
            time = 1
        if time is not None or phase is not None:
            self.settimeandphase(time, phase)
        return self.rotate(v, align)

class QuatRotator2(BaseRotator):
    """This uses the rotation functions from the quaternion module,
    but this is rather slow.
    """
    neutral = quat.one

    def init(self):
        self.rotation = np.tile(self.neutral, self.omega.shape[:-1])

    def update(self):
        wt = np.multiply.outer(self.time, self.omega)
        if self.phase is not None:
            wt += self.omega * self.phase[..., np.newaxis]
        self.rotation = quat.from_rotation_vector(wt)

    def rotate(self, v, align = 0, phase = None):
        if align == 0:
            return quat.rotate_vectors(self.rotation, v, axis=-1)

        assert self.rotation[0].shape[-align:] == v.shape[:align]
        # need to be iterated manually.
        dr = self.rotation.shape
        dv = v.shape
        nr = len(dr)
        nv = len(dv)

        ia = np.product(dv[:align], dtype=np.int)
        sr = [np.product(dr[:nr-align], dtype=np.int), ia]
        sv = [ia, np.product(dv[align:-1], dtype=np.int), 3]
        so = [sr[0]] + sv

        o = np.empty(so)
        vl = v.reshape(sv)
        rl = self.rotation.reshape(sr)

        # this may be a case for numba?
        for i in range(ia):
            o[:,i] = quat.rotate_vectors(rl[:,i], vl[i], axis=-1)

        do = dr + dv[align:]
        return o.reshape(do)

class QuatRotator(BaseRotator):
    """Uses quaternions for rotation.

    Generaotor, g = [rotation verctor] / 2
    Rotator q = exp(g * t)

    Rotated vecor is v' = q * v * q.conjugate()

    where both rotation vector and v components go into the imaginary part.
    """
    neutral = (quat.one,) * 2

    def init(self):
        self.rotation = self.neutral
        q = np.zeros(self.omega.shape[:-1] + (4,))
        q[...,1:] = self.omega[...,:] * 0.5
        self.generator = quat.from_float_array(q)

    def update(self):
        q = np.multiply.outer(self.time, self.generator)
        if self.phase is not None:
            q += self.generator * self.phase
        q = np.exp(q)
        self.rotation = (q, q.conj())

    def rotate(self, v, align = 0, phase = None):
        if align > 0:
            assert self.rotation[0].shape[-align:] == v.shape[:align]

        v = np.asarray(v)
        q = np.zeros(v.shape[:-1] + (4,))
        q[...,1:] = v[...,:]
        q = quat.from_float_array(q)

        dr = self.rotation[0].shape
        dq = q.shape
        nr = len(dr)
        nq = len(dq)
        sr = dr + (1,) * (nq - align)
        sq = (1,) * nr + dq

        return quat.as_float_array(
            self.rotation[0].reshape(sr)
            * q.reshape(sq)
            * self.rotation[1].reshape(sr)
            )[...,1:]

class BaseMatrixRotator(BaseRotator):
    """
    standard matrix multiplication version
    """
    neutral = np.eye(3)

    def init(self):
        self.rotation = np.tile(self.neutral, self.omega.shape[:-1])

    def rotate(self, v, align = 0):
        v = np.asarray(v)
        if align > 0:
            assert self.rotation.shape[-align-2:-2] == v.shape[:align]
        nr = len(self.rotation.shape) - 2
        nv = len(v.shape) - 1
        ir = list(range(2, 2 + nr))
        iv = list(range(2 + nr - align, 2 + nr - align + nv))
        iv[0:align] = ir[len(ir)-align:]
        io = ir + iv[align:] + [0]
        ir += [0, 1]
        iv += [1]
        return np.einsum(self.rotation, ir, v, iv, io)

# numba does not do anything here ...
# @numba.jit(parallel=True)
def _update_MR(a, b, n):
    for i in range(n):
        b[i] = expm(a[i])

class MatrixRotator(BaseMatrixRotator):
    """
    use generator and matrix exponnent (expm)
    """

    _idxg1 = np.array([2,1,0,2,1,0])
    _idxg2 = np.array([1,2,2,0,0,1])
    _idxo1 = np.array([0,0,1,1,2,2])
    _s1 = np.array([1,-1,1,-1,1,-1])

    def init(self):
        super().init()
        omega = self.omega
        g = np.zeros(omega.shape[:-1] + (3,3))
        # g[...,2,1] = +omega[...,0]
        # g[...,1,2] = -omega[...,0]
        # g[...,0,2] = +omega[...,1]
        # g[...,2,0] = -omega[...,1]
        # g[...,1,0] = +omega[...,2]
        # g[...,0,1] = -omega[...,2]
        g[..., self._idxg1, self._idxg2] = self._s1 * omega[..., self._idxo1]
        self.generator = g

    def update(self):
        # if len(np.shape(self.time)) == 0 and len(np.shape(self.omega)) == 1:
        #     a = self.generator * self.time
        #     self.rotation = expm(a)
        #     return
        a = np.multiply.outer(self.time, self.generator)
        if self.phase is not None:
            a += self.generator * self.phase[..., np.newaxis, np.newaxis]
        d = (-1,3,3)
        b = np.empty_like(a).reshape(d)
        _update_MR(a.reshape(d), b, np.product(a.shape[:-2], dtype = np.int))
        self.rotation =  b.reshape(a.shape)

        # # alternative naive code is about 1% slower for big arrays
        # self.rotation =  np.asarray(
        #     [expm(x) for x in np.reshape(a, (-1,3,3))]).reshape(a.shape)


@numba.jit(nopython=True)
def _update_ER_scalar(omega, time):
    w = time * omega
    x = LA.norm(w)
    a = np.cos(x * 0.5)
    if x < 1.e-99:
        x = 1.e-99
    y = 1 / x
    b, c, d = w * (np.sin(x * 0.5) * y)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [[aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)],
         [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)],
         [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc]])

# numba makes things slower for large arrays
# @numba.jit(parallel=True)
def _update_ER(omega, time, phase):
    w = np.multiply.outer(time, omega)
    if phase is not None:
        w += omega * phase[..., np.newaxis]
    w = np.moveaxis(w, -1, 0)
    x = LA.norm(w, axis = 0)
    x = np.maximum(x, 1.e-99)
    y = 1 / x
    b, c, d = w * (np.sin(x * 0.5) * y)
    a = np.cos(x * 0.5)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [[aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)],
         [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)],
         [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc]])

class ERRotator(BaseMatrixRotator):
    """Direct computation using Euler-Rodrigues formula"""

    def update(self):
        if len(np.shape(self.time)) == 0 and len(np.shape(self.omega)) == 1:
            if self.phase is None:
                self.rotation = _update_ER_scalar(self.omega, self.time)
                return
            self.rotation = _update_ER_scalar(self.omega, self.time + self.phase)
            return
        m = _update_ER(self.omega, self.time, self.phase)
        self.rotation = np.moveaxis(m, (0, 1), (-2,-1))

# numba makes things slower for large arrays
# @numba.jit(parallel=True)
def _update_ERQ(q):
    a, b, c, d = np.moveaxis(quat.as_float_array(q), -1, 0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [[aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)],
         [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)],
         [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc]])

class QERRotator(BaseMatrixRotator):
    """
    Use quaternions to compuse rotation, then apply rotation using matrix.
    """
    neutral = np.eye(3)

    def init(self):
        self.rotation = self.neutral
        q = np.zeros(self.omega.shape[:-1] + (4,))
        q[...,1:] = self.omega[...,:] * 0.5
        self.generator = quat.from_float_array(q)

    def update(self):
        q = np.multiply.outer(self.time, self.generator)
        if self.phase is not None:
            q += self.generator * self.phase
        m = _update_ERQ(np.exp(q))
        self.rotation = np.moveaxis(m, (0, 1), (-2,-1))


# set default rotator
Rotator = QERRotator

def rotate(v, w, time = None, phase = None):
    """
    just do rotation of v about w
    """
    if phase is not None and time is None:
        time = 0
    return Rotator(w, time = time, phase = phase)(v)


def w2f_(w):
    """
    convert rotation vector to quaternion in array shape + (,4)
    """
    w = np.asarray(w)
    f = np.empty(w.shape[:-1] + (4,))
    r = np.maximum(LA.norm(w, axis=-1), 1.e-99)
    y = np.sin(r * 0.5) / r
    f[..., 0] = np.cos(r * 0.5)
    f[..., 1:] = w[..., :] * y[..., np.newaxis]
    return f

def w2q_(w):
    """
    convert rotation vector to quaternion
    """
    return quat.from_float_array(w2f_(w))

def w2q(w):
    """
    convert rotation vector to quaternion
    """
    w = np.asarray(w)
    f = np.zeros(w.shape[:-1] + (4,))
    f[..., 1:] = w[..., :] * 0.5
    return np.exp(quat.from_float_array(f))

def w2f(w):
    """
    convert rotation vector to quaternion in array shape + (,4)
    """
    return quat.as_float_array(w2q(w))


def f2w_(f):
    """
    convert quaternion as shape + (4,) to rotation vector
    """
    f = np.asarray(f)
    w = np.empty(f.shape[:-1] + (3,))
    r = np.arccos(f[..., 0]) * 2.
    sir = np.full_like(r, 2.)
    if np.shape(r) == ():
        if np.abs(r) > 1.e-99:
            sir[()] = r[()] / np.sin(r[()] * 0.5)
    else:
        ii = np.abs(r) > 1.e-99
        sir[ii] = r[ii] / np.sin(r[ii] * 0.5)
    w[..., :] = f[..., 1:] * sir[..., np.newaxis]
    return w

def q2w_(q):
    """
    convert quaternion to to rotation vector
    """
    return f2w_(quat.as_float_array(np.asarray(q)))

def f2w__(f):
    """
    convert quaternion as shape + (4,) to rotation vector

    This version seems slower but I should assume it be more precise.
    """
    f = np.asarray(f)
    n = LA.norm(f[..., 1:], axis = -1)
    if np.shape(n) == ():
        if np.abs(n) > 1.e-99:
            x = 2. * np.arctan2(n, f[0]) / n
        else:
            x = 2. / f[0]
    else:
        ii = np.abs(n) > 1.e-99
        x = np.empty_like(n)
        x[ii] = 2. * np.arctan2(n[ii], f[..., 0][ii]) / n
        ii = ~ii
        x[ii] = 2. / (f[..., 0][ii])
    return f[..., 1:] * x[..., np.newaxis]

def q2w__(q):
    """
    convert quaternion to to rotation vector
    """
    return f2w__(quat.as_float_array(np.asarray(q)))


def q2w(q):
    """
    convert quaternion to to rotation vector
    """
    return quat.as_float_array(np.log(np.asarray(q)) * 2.)[..., 1:]


def f2w(f):
    """
    convert quaternion as shape + (4,) to rotation vector
    """
    return q2w(quat.from_float_array(np.asarray(f)))


# Pauli matices
sigma = np.asarray([
    np.array([[1,  0 ],[ 0 , 1 ]], dtype = np.complex),
    np.array([[0,  1 ],[-1 , 0 ]], dtype = np.complex),
    np.array([[0,  1j],[ 1j, 0 ]], dtype = np.complex),
    np.array([[1j, 0 ],[ 0 ,-1j]], dtype = np.complex),
           ])

def q2u(q):
    """
    transform quaternion to SU(2)
    """
    return np.tensordot(quat.as_float_array(np.asarray(q)), sigma, axes = (-1,-3))

def w2u(w):
    """
    transform omega vector to SU(2)
    """
    return q2u(w2q(w))

def u2q(u):
    """
    convert SU(2) to quaternion
    """
    u = np.asarray(u)
    f = np.empty(u.shape[:-2] + (4,))
    f[..., 0] = ((u[..., 0, 0] + u[..., 1, 1]) *   0.5  ).real
    f[..., 3] = ((u[..., 0, 0] - u[..., 1, 1]) * (-0.5j)).real
    f[..., 2] = ((u[..., 0, 1] + u[..., 1, 0]) * (-0.5j)).real
    f[..., 1] = ((u[..., 0, 1] - u[..., 1, 0]) *   0.5  ).real
    return quat.from_float_array(f)

def u2w(u):
    """
    convert SU(2) to rotation vector
    """
    return q2w(u2q(u))

def w2p(w):
    """
    convert rotation vector in xyz to polar (r,theta,phi)
    """
    w = np.asarray(w)
    p = np.empty(w.shape)
    p[..., 0] = LA.norm(w, axis = -1)
    p[..., 1] = np.arctan2(LA.norm(w[..., :2], axis = -1), w[..., 2])
    p[..., 2] = np.arctan2(w[..., 1], w[..., 0])
    return p

def p2w(p):
    """
    convert rotation vector in polar (r,theta,phi) to xyz
    """
    p = np.asarray(p)
    w = np.empty(p.shape)
    xy = np.sin(p[..., 1]) * p[..., 0]
    w[..., 0] = xy * np.cos(p[..., 2])
    w[..., 1] = xy * np.sin(p[..., 2])
    w[..., 2] = np.cos(p[..., 1]) * p[..., 0]
    return w

def l2p(latitude, logitude, radius = 1.):
    """
    convert logitude, lattitude[, radius = 1] to polar
    """
    p = np.array([radius, (90 - latitude) * deg2rad, logitude * deg2rad])
    return p

def p2l(p, return_radius = False):
    """
    convert polar to logitude, latitude
    """
    radius = p[0]
    latitude = 90 - p[1] * rad2deg
    logitude = p[2] * rad2deg
    if return_radius:
        return latitude, logitude, radius
    return latitude, logitude

def w2c(w):
    """
    convert rotation vector in xyz to cylindrical (rho,phi,z)
    """
    w = np.asarray(w)
    c = np.empty(w.shape)
    c[..., 0] = LA.norm(w[..., :2], axis = -1)
    c[..., 1] = np.arctan2(w[..., 1], w[..., 0])
    c[..., 2] = w[..., 2]
    return c

def c2w(c):
    """
    convert rotation vector in cylindrical (rho,phi,z) to xyz
    """
    c = np.asarray(c)
    w = np.empty(c.shape)
    w[..., 0] = c[..., 0] * np.cos(c[..., 1])
    w[..., 1] = c[..., 0] * np.sin(c[..., 1])
    w[..., 2] = c[..., 2]
    return w


def rotate2(w, w0 = None, rotate = None, rotate0 = None, align = None, lim = 1.e-99):
    """
    rotation vector to rotate from w0 to w.

    Default for w0 = unit vector in x-direction

    For anti-aligned vectors (sin(x) < 1e-99) the rotation axis is chosen
    sort of by chance.

    We could also just use arccos of dot product, may be faster though
    less accurate.

    Need vectorised version, but cases are a pain.
    """
    w = np.asarray(w)
    l = LA.norm(w, axis = -1)
    if l == 0:
        return np.zeros_like(w)
    w = w / l
    if w0 is None:
        w0 = np.tile(ex, w.shape[:-1])
    else:
        w0 = np.asarray(w0)
        w0 = w0 / LA.norm(w0, axis = -1)

    x = np.cross(w0, w)
    s = LA.norm(x)
    c = np.tensordot(w0, w, axes = (-1,-1))

    if s < lim:
        if align is not None:
            v = align - np.tensordot(align, w, axes = (-1, -1)) * w
            n = LA.norm(v)
        # find a normal vector by cross with unit vectors
        else:
            v = np.array([w[1], -w[0], 0])
            n = LA.norm(v)
        if n < lim:
            v = np.array([-w[2], 0, w[0]])
            n = LA.norm(v)
        if n < lim:
            # this part should never be needed
            v = np.array([0, w[2], w[1]])
            n = LA.norm(v)
        if n > lim:
            return (0.5 * np.pi * (c - 1) / n) * v

    s = np.maximum(s, lim)
    p = np.arctan2(s, c)
    if align is not None and np.tensordot(x, align, axes=(-1,-1)) < 0:
        p -= 2 * np.pi
    w1 = x * (p / s)
    if rotate0 is not None:
        r = w0 * rotate0
        w1 = wchain(r, w1)
    if rotate is not None:
        r = w * rotate
        w1 = wchain(w1, r)
    return w1


def rotate3(w, w1, w2):
    """
    get rotator for rotation about w from w1 to w2
    """
    w = np.asarray(w)
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)

    # compute normal componnents
    p1 = np.cross(w, w1, axis = -1)
    p2 = np.cross(w, w2, axis = -1)

    wx = rotate2(p2, p1)

    align = np.tensordot(w, wx, axes = (-1, -1))
    if align < 0:
        wx *= 1 - np.pi * 2 / LA.norm(wx)
    return wx


def rotstep(w, v, nstep = 0.5):
    """
    rotate v about w in steps
    """
    if np.shape(steps) != ():
        f2 = np.asarray(steps)
    elif steps <= 1:
        f2 = np.asarray(steps)
    else:
        f2 = np.arange(steps) / (steps - 1)
    return  Rotator(w, time = f2)(v)


def rotstep3(w, w1, w2, nstep = 0.5):
    """
    rotate about w in steps from w1 to w2

    sort of like precession
    """
    w = np.asarray(w)
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)

    wx = Rotate3(w, w1, w2)

    n1 = LA.norm(w1, axis = -1)
    n2 = LA.norm(w2, axis = -1)

    if np.shape(steps) != ():
        f2 = np.asarray(steps)
    elif steps <= 1:
        f2 = np.asarray(steps)
    else:
        f2 = np.arange(steps) / (steps - 1)
    f1 = np.asarray(1 - f2)

    return  Rotator(wx, time = wx)(w1 / n1) * \
        np.asarray(f1 * n1 + f2 * n2)[..., np.newaxis]

def wv2tr(w, v):
    """
    Decompose rotation of v about w into twist and rotation
    components, in that order.

    When recompositing, twist needs to be applied first.

    This routine seems highly inefficient.
    """
    w = np.asarray(w)
    v = np.asarray(v)
    v1 = Rotator(w)(v)
    r = rotate2(v1, v)
    t = wchain(w, -r)
    return t, r

def wv2rt(w, v):
    """
    Decompose rotation of v about w into rotation and twist
    components, in that order.

    When recompositing, rotation needs to be applied first.

    This routine seems highly inefficient.
    """
    w = np.asarray(w)
    v = np.asarray(v)
    v1 = Rotator(w)(v)
    r = rotate2(v1, v)
    t = wchain(-r, w)
    return r, t

# TODO - add routine(s) that computes projected twist (pendulum,
# coriolis) for rotation of v about w.  See sphere.items.Arrow
# Could also compute combined rotation vector, or variants using
# rotate2 (geodesic) decompositon.  Needs tests.

def wchain(w1, w2):
    """
    Rotation vector resulting from rotation w1 than w2.

    Not commutative.  Done by multiplying associated quaternions
    """
    return q2w(w2q(w2) * w2q(w1))

# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
def wchain_(w1, w2):
    """
    Rotation composition using Rodrigues Formula.

    For testing only, do not use that.
    """
    a = np.asarray(w1)
    b = np.asarray(w2)
    an = LA.norm(a)
    bn = LA.norm(b)
    ae = a / an
    be = b / bn
    a2 = an * 0.5
    b2 = bn * 0.5
    ab = np.tensordot(ae, be, axes = (-1,-1))
    bxa = np.cross(be, ae)
    g2 = np.arccos(np.cos(a2)*np.cos(b2) - np.sin(a2)*np.sin(b2) * ab)
    ta2 = np.tan(a2)
    tb2 = np.tan(b2)
    tg2 = np.tan(g2)
    return (
        ae * ta2[..., np.newaxis] +
        be * tb2[..., np.newaxis] +
        bxa * (ta2 * tb2)[..., np.newaxis]) * (
            2. * g2 / (
            (1 - ta2 * tb2 * ab) * tg2))[..., np.newaxis]

def rotscale(w1, w2, steps = 0.5, rotation = None, align = None):
    """
    Smoothly rotate and scale w1 to w2

    If `steps` <= 1 or left out, computer middle vector.

    If  `steps` is an array return vactors for these fractions.

    if `rotation is not None, perform gradual rotation according to
    step, gradually scaling totation angle from w1 to w2.  This
    internal rotation is applied after the rotscae operation about the
    final rotation axis.

    TODO - milti-D and contractions w/r w1, w2, steps (einsum)
    """
    if np.shape(steps) != ():
        f2 = np.asarray(steps)
    elif steps <= 1:
        f2 = np.asarray(steps)
    else:
        f2 = np.arange(steps) / (steps - 1)
    f1 = np.asarray(1 - f2)
    n1 = LA.norm(w1, axis = -1)
    n2 = LA.norm(w2, axis = -1)
    w =  Rotator(np.multiply.outer(f2, rotate2(w2, w1, align=align)))(w1 / n1) * \
        np.asarray(f1 * n1 + f2 * n2)[..., np.newaxis]
    if rotation is not None:
        r = w  * (rotation * f2 / LA.norm(w, axis = -1))[..., np.newaxis]
        w = wchain(w, r)
    return w

def w2axyz(w, deg = False, lim = 1.e-12):
    """
    get rotation angle and xyz vector of axis
    """
    w = np.asarray(w, dtype = np.float)
    a = LA.norm(w, axis = -1)
    ii = a < lim
    if np.shape(ii) == ():
        if ii:
            w = np.array([1., 0., 0.,])
            a = 0
        else:
            w = w / a
    else:
        jj = ~ii
        w = w.copy()
        w[ii, :] = np.array([1., 0., 0.])
        a[ii] = 0
        w[jj, :] /= a[jj, np.newaxis]

    if deg:
        a *= 180 / np.pi
    axyz = np.empty(np.shape(a) + (4,))
    axyz[..., 0] = a
    axyz[..., 1:] = w
    return axyz

def q2A(q):
    f = quat.as_float_array(np.asarray(q))
    a, b, c, d = np.moveaxis(f, -1, 0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    A = np.array(
        [[aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)],
         [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)],
         [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc]])
    ii = np.arange(len(f.shape) + 1)
    jj = np.roll(ii, 2)
    return np.moveaxis(A, ii, jj)

def w2A(w):
    return q2A(w2q(w))

# TODO - add direct A2w, A2f, A2q, and f2A routines


#=======================================================================
# Action on Euler Angles
#=======================================================================
# this will require detailed and clear explanantion
# in particular relaitive to Wikipedai
# 1) clear statement on order of matrcies vs order of transforms
#    Ax * Ay * Az * v  is applying rotation Az *first*
# 2) passive vs active rotations
# 3) add extensive tests for order, etc.
#=======================================================================

# https://stackoverflow.com/questions/30279065/how-to-get-the-euler-angles-from-the-rotation-vector-sensor-type-rotation-vecto
# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
def w2xyz(w, deg = False):
    """
    Convert omega to Euler angles XYZ order
    """
    q0, q1, q2, q3 = np.moveaxis(w2f(w), -1, 0)

    a = np.empty_like(w)
    a[...,0] = np.arctan2(2*(q0*q1 + q2*q3), q0**2 + q3**2 - q2**2 - q1**2)
    a[...,1] = np.arcsin(np.maximum(np.minimum(2*(q0*q2 - q1*q3), 1), -1))
    a[...,2] = np.arctan2(2*(q0*q3 + q1*q2), q0**2 + q1**2 - q2**2 - q3**2)

    if deg:
        return a * (180. / np.pi)
    return a

def w2zyx(w, deg = False):
    """
    Convert omega to Euler angles ZYX order
    """
    q0, q1, q2, q3 = np.moveaxis(w2f(w), -1, 0)

    a = np.empty_like(w)
    a[...,0] = np.arctan2(2*(q0*q3 - q1*q2), q0**2 + q1**2 - q2**2 - q3**2)
    a[...,1] = np.arcsin(np.maximum(np.minimum(2*(q0*q2 + q1*q3), 1), -1))
    a[...,2] = np.arctan2(2*(q0*q1 - q2*q3), q0**2 - q1**2 - q2**2 + q3**2)

    if deg:
        return a * (180. / np.pi)
    return a

def zyx2f(e):
    """
    convert Euler angles ZYX to unit quaternion

    Is this correct?  TEST!
    """

    e = np.asarray(e) * 0.5

    f = np.empty(e.shape[:-1] + (4,))

    cz = np.cos(e[...,0])
    sz = np.sin(e[...,0])
    cy = np.cos(e[...,1])
    sy = np.sin(e[...,1])
    cx = np.cos(e[...,2])
    sx = np.sin(e[...,2])

    f[...,0] = cz * cy * cx - sz * sy * sx
    f[...,1] = sz * sy * cx + cz * cy * sx
    f[...,2] = cz * sy * cx - sz * cy * sx
    f[...,3] = sz * cy * cx + cz * sy * sx

    return f

def zyx2q(e):
    """
    convert Euler angles ZYX to unit quaternion
    """
    return quat.from_float_array(zyx2f(e))


def xyz2f(e):
    """
    convert Euler angles XYZ to unit quaternion in float as shape + (4,)
    """

    e = np.asarray(e) * 0.5

    f = np.empty(e.shape[:-1] + (4,))

    cx = np.cos(e[...,0])
    sx = np.sin(e[...,0])
    cy = np.cos(e[...,1])
    sy = np.sin(e[...,1])
    cz = np.cos(e[...,2])
    sz = np.sin(e[...,2])

    f[...,0] = cz * cy * cx + sz * sy * sx
    f[...,1] = cz * cy * sx - sz * sy * cx
    f[...,2] = sz * cy * sx + cz * sy * cx
    f[...,3] = sz * cy * cx - cz * sy * sx

    return f

def xyz2q(e):
    """
    convert Euler angles XYZ to unit quaternion
    """
    return quat.from_float_array(xyz2f(e))

def zyx2w(e):
    return f2w(zyx2f(e))

def xyz2w(e):
    return f2w(xyz2f(e))

# https://en.wikipedia.org/wiki/Euler_angles
# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
def A2zyx(A):
    """
    from matrix return vector with rotation angles about z, y, x axes
    """
    A = np.asarray(A)
    e = np.ndarray(A.shape[:-2] + (3,))
    e[..., 0] = np.arctan2(-A[..., 0, 1], A[..., 0, 0])
    e[..., 1] = np.arcsin(np.maximum(np.minimum(A[...,0, 2], 1), -1))
    e[..., 2] = np.arctan2(-A[..., 1, 2], A[..., 2, 2])
    return e

def A2xyz(A):
    """
    from matrix return vector with rotation angles about x, y, z axes
    """
    A = np.asarray(A)
    e = np.ndarray(A.shape[:-2] + (3,))
    e[..., 0] = np.arctan2(A[..., 2, 1], A[..., 2, 2])
    e[..., 1] = np.arcsin(np.maximum(np.minimum(-A[...,2, 0], 1), -1))
    e[..., 2] = np.arctan2(A[..., 1, 0], A[..., 0, 0])
    return e
