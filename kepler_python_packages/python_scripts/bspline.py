"""
Bspline routines

(C) Alexander Heger 2016
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval, polyder

class BsplineScalarRecursive(object):
    """
    Demonstation of algorithm.
    """
    def __init__(self, k, t):
        assert k >= 0
        # assert len(t) >= k
        self._n = len(t)
        self._t = np.concatenate((np.tile(t[0], k), t, np.tile(t[-1], k)))
        self._k = k

    def _b(self, i, k, x):
        """
        k is degree is 0+
        i is 0 <= i < len(t)
        t[0] <= x <= t[-1]
        i < len(t)
        """
        t = self._t
        if k == 0:
            if t[i] <= x < t[i+1]:
                return 1
            if i >= self._n + self._k - 2 and x == t[-1]:
                return 1
            return 0
        w = 0
        if t[i+k] != t[i]:
            w += (x - t[i]) / (t[i+k] - t[i]) * self._b(i, k-1, x)
        if t[i+k+1] != t[i+1]:
            w += (t[i+k+1] - x) / (t[i+k+1] - t[i+1]) * self._b(i+1, k-1, x)
        return w

    def __call__(self, i, x, p = 0):
        if p == 0:
            return self._b(i,self._k, x)
        else:
            return self._prime(i, x, p = p, k = None)

    def _prime(self, i, x, k = None, p = 1):
        if k is None:
            k = self._k
        if p == 0:
            return self._b(i, k, x)
        w = 0
        v = self._t[i + k + 1] - self._t[i + 1]
        if v != 0:
            w -= self._prime(i+1, x, k - 1, p - 1) / v
        v = self._t[i + k    ] - self._t[i    ]
        if v != 0:
            w += self._prime(i  , x, k - 1, p - 1) / v
        return k * w

class BsplineBase(object):
    """
    Setup for B-Spline routines
    """
    def __init__(self, k, t, *args, derivatives = False, periodic = False):
        assert k >= 0
        self._n = len(t)
        if periodic:
            _t = np.array(t)
            _t = np.concatenate((
                _t[0] + _t[-k-1:-1] - _t[-1],
                _t,
                _t[-1] + _t[1:k+1] - _t[0]))
        else:
            _t = np.concatenate((np.tile(t[0], k), t, np.tile(t[-1], k)))
        self._t = _t
        self._k = k
        self._derivatives = derivatives

        _n = len(t)
        _k = self._k
        _t = self._t
        a = np.zeros((_n+2*_k,_n+_k,_k+1,_k+1))
        ii = np.arange(_n+_k)
        a[ii, ii, 0, 0] = 1
        f = np.ndarray((4,))
        for k in range(1, _k+1):
            for i in range(_n + 2 * _k - k - 1):
                for j in range(i, min(i + k + 1, _n+_k)):

                    v = np.tile(_t[i+k:i+k+2] - _t[i:i+2], 2)
                    f[:] = [-_t[i], _t[i+k+1], 1, -1]
                    ii = v != 0
                    f[ii] /= v[ii]

                    # # setting to zero is not needed as the corresponding
                    # # matrix elements are zero anyway
                    # ii = np.logical_not(ii)
                    # f[ii] = 0

                    # # non-vectorized algorithm
                    # v0 = _t[i+k] - _t[i]
                    # if v0 != 0:
                    #     f[0] = - _t[i]
                    #     f[2] = 1
                    #     f[0::2] /= v0
                    # # else:
                    # #     f[0::2] = 0
                    # v1 = _t[i+k+1] - _t[i+1]
                    # if v1 != 0:
                    #     f[1] = _t[i+k+1]
                    #     f[3] = - 1
                    #     f[1::2] /= v1
                    # # else:
                    # #     f[1::2] = 0

                    a[i,j,k, :k  ]  = np.dot(f[ :2], a[i:i+2, j, k-1, :k])
                    a[i,j,k,1:k+1] += np.dot(f[2: ], a[i:i+2, j, k-1, :k])

                    # # non-vectorized algorithm
                    # for n in range(k):
                    #     a[i,j,k,n]  = (f[0] * a[i  , j, k-1, n] +
                    #                    f[1] * a[i+1, j, k-1, n])
                    # for n in range(1, k+1):
                    #     a[i,j,k,n] += (f[2] * a[i  , j, k-1, n-1] +
                    #                    f[3] * a[i+1, j, k-1, n-1])

        # let's compute derivatives here:
        if derivatives:
            m = np.arange(1, _k + 2)
            for p in range(_k - 1, -1, -1):
                # compute all derivatives, including those not actually needed
                # (skipping zeroing unused elements)
                a[:, :, p, :p+1] = m[:p+1] * a[:, :, p+1, 1:p+2]

                # # we really only need those coefficients,
                # # but loops may be slower ...
                # for i in range(_n + 2 * _k - k - 1):
                #     for j in range(i, min(i + k + 1, _n+_k)):
                #         a[i,j,p,:p+1] = m[:p+1] * a[i,j,p+1,1:p+2]
        self._a = a

    def __call__(*args, **kwargs):
        raise NotImplementedError()

# these routines are not actually slower than the library functions

def polyval(x, a):
    k = len(a) - 1
    w = a[k]
    for n in range(k - 1, -1, -1):
        w = w * x + a[n]
    return w

def polyder(a, p = 1):
    k = len(a) - 1
    m = np.arange(1, k + 2)
    a = a.copy()
    for n in range(k - 1, k - p - 1, -1):
        a[:n+1] = m[:n+1] * a[1:n+2]
    return a[:k-p+1]

class BsplineScalar(BsplineBase):
    def __call__(self, i, x, p = 0):
        """
        k is degree is 0+
        0 <= i < len(t)
        t[0] <= x <= t[-1]
        p < k
        """
        t = self._t
        jj = np.where(np.logical_and(x >= t[:-1], x < t[1:]))[0]
        if len(jj) == 0:
            if x == t[-1]:
                j = self._n + self._k - 2
            else:
                return 0
        else:
            j = jj[0]

        if self._derivatives:
            k = self._k - p
            a = self._a[i, j, k, :k+1]
        else:
            a = self._a[i, j, -1]
            if p > 0:
                a = polyder(a, p)

        return polyval(x, a)

class Bspline(BsplineBase):
    def __init__(self, *args, **kwargs):
        kw = dict(kwargs)
        kw['derivatives'] = True
        super().__init__(*args, **kw)

    def __call__(self, i, x, p = 0):
        """
        k is degree is 0+
        0 <= i < len(t)
        t[0] <= x <= t[-1]
        p < k

        NumPy vectorized version.
        """
        _x = x
        if np.isscalar(x):
            x = (x,)
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        w = np.zeros_like(x)
        t = self._t
        for j in range(self._k, self._n + self._k - 1):
            if j == self._n + self._k - 2:
                jj = np.where(np.logical_and(x >= t[j], x <= t[j + 1]))[0]
            else:
                jj = np.where(np.logical_and(x >= t[j], x < t[j + 1]))[0]

            if self._derivatives:
                k = self._k - p
                a = self._a[i, j, k]
            else:
                a = self._a[i, j, -1]
                if p > 0:
                    a = polyder(a, p)

            w[jj] = polyval(x[jj], a)

        if np.isscalar(_x):
            w = w[0]
        return w

class BsplineInterpolateBase(object):
    def _init_xy(self, k, x, y, *args, periodic = False, **kwargs):
        if periodic:
            _y = np.array(y)
            _k = k // 2
            _y = np.concatenate(
                (_y[0] + _y[-_k-1:-1] - _y[-1],
                 _y,
                 _y[-1] + _y[1:_k+1] - _y[0]))
        else:
            _y = np.concatenate(
                (np.tile(y[0], k // 2),
                 y,
                 np.tile(y[-1], k // 2)))
        if len(x) == len(y) + 1 and k % 2 == 0 and not periodic:
            n0 = len(x)
            x0 = np.linspace(0, 1, n0, endpoint = True)
            n1 = len(y) + ((k + 1) % 2)
            x1 = np.linspace(0, 1, n1, endpoint = True)
            x = np.interp(x1, x0, x)
        elif len(x) == len(y) and k % 2 == 0:
            _y = 0.5 * (_y[:-1] + _y[1:])
        self._y = _y
        return x

class BsplineInterpolateScalar(BsplineScalar, BsplineInterpolateBase):
    def __init__(self, k, x, y, **kwargs):
        x = self._init_xy(k, x, y, **kwargs)
        super().__init__(k, x, **kwargs)

    def __call__(self, x, p = 0):
        w = 0
        for i in range(self._n + self._k - 1):
            w += super().__call__(i, x, p = p) * self._y[i]
        return w

class BsplineInterpolate(Bspline, BsplineInterpolateBase):
    def __init__(self, k, x, y, **kwargs):
        x = self._init_xy(k, x, y, **kwargs)
        super().__init__(k, x, **kwargs)

    def __call__(self, x, p = 0):
        _x = x
        if np.isscalar(x):
            x = (x,)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        w = np.zeros_like(x)

        for i in range(self._n + self._k - 1):
            w += super().__call__(i, x, p = p) * self._y[i]

        if np.isscalar(_x):
            w = w[0]
        return w


def bsplinetest(k = 5, n = 11):
    import matplotlib.pyplot as plt
    xi = np.arange(n)
    x = np.linspace(min(xi),max(xi),1000, endpoint=True)
    fig = plt.figure()

    # vector test
    ax = fig.add_subplot(2,1,1)
    b = Bspline(k, xi, periodic = False)
    for i in range(n + k - 1):
        ax.plot(x, b(i,x))
    plt.show()
    plt.pause(0.01)

    # scalar test
    ax = fig.add_subplot(2,1,2)
    b = BsplineScalar(k, xi, periodic = False)
    for i in range(n + k - 1):
        ax.plot(x,[b(i,t) for t in x])
    plt.show()

    return b

def testb(k=3, n=11, test = 0, derivatives = True, **kwargs):
    periodic = kwargs.get('periodic', False)
    if test == 1:
        xi = np.arange(n) * 2 * np.pi / (n-1) + .5
        yi = np.cos(xi)
        periodic = kwargs.get('periodic', True)
    if test == 0:
        xi = np.arange(n)
        yi = xi + 1
    elif test == 3:
        xi = (-2, -1, 0, 1, 2)
        yi = (0, 0, 6, 0, 0)
        k = 3
    elif test == 2:
        xi = (0, 1, 2, 3)
        yi = (0, 1, 0)
        k = 2
    elif test == 4:
        xi =  (0, 1, 2, 3, 4, 5,)
        yi =  (0, 0, 1, 0, 0, )
        k = 4

    x = np.linspace(min(xi), max(xi), 1000, endpoint = True)

    fig = plt.figure()

    # vector test
    ax = fig.add_subplot(2,1,1)
    bv = BsplineInterpolate(k, xi, yi, periodic = periodic, derivatives = derivatives)
    for i in range(k):
        ax.plot(x,bv(x, p=i))
    if len(xi) != len(yi):
        xd = np.linspace(min(xi), max(xi), len(yi))
    else:
        xd = xi
    ax.plot(xd, yi, 'ko')
    ax.set_xlim((min(xi), max(xi)))
    plt.show()

    plt.pause(0.01)

    # scalar test
    ax = fig.add_subplot(2,1,2)
    bs = BsplineInterpolateScalar(k, xi, yi, periodic = periodic, derivatives = derivatives)
    for i in range(k):
        ax.plot(x,[bs(t, p=i) for t in x])
    if len(xi) != len(yi):
        xd = np.linspace(min(xi), max(xi), len(yi))
    else:
        xd = xi
    ax.plot(xd, yi, 'ko')
    ax.set_xlim((min(xi), max(xi)))
    plt.show()


def testb2(k=3, n=3):
    #xi = 1/(np.arange(n)+1)[::-1]
    xi = np.arange(n)

    x = np.linspace(min(xi), max(xi), 1000, endpoint = True)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    s = np.zeros_like(x)
    for i in range(n):
        yi = np.zeros(n)
        yi[i] = 1
        b = BsplineInterpolate(k, xi, yi)
        y = b(x)
        ax.plot(x, y)
        s += y

    print(np, min(s), np.max(s))

    ax.set_xlim((min(xi),max(xi)))
    plt.show()


def testb3():
    n = 2
    k = 1
    x = 0.5

    yi = np.zeros(n)
    yi[1] = 1

    xi = np.arange(n)

    b = BsplineInterpolate(k, xi, yi)
    print(b(x))
    return b
