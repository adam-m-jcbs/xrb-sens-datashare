import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms
import matplotlib.collections

from logged import Logged

from utils import CachedAttribute

import reduce

import kepdump

def tdma(a, b, c, r):
    """
    Solve tridiagonal system of equations.

    inspired by Numerical Recipes.
     _                      _
    | b_0  c_0               |
    | a_1  b_1 c_1           |
    |                        | * x = r

    |_              a_n b_n _|

    """
    n = r.shape[0]
    u = np.ndarray(n)
    g = np.ndarray(n)
    if b[0] == 0:
        raise Exception(__name__ +': rewrite equations')
    beta = 1. / b[0]
    u[0] = r[0] * beta
    for j in range(1, n):
        g[j] = c[j-1] * beta
        beta = 1. / (b[j] - a[j] * g[j])
        if beta == np.nan:
            raise Exception('tridag failed')
        u[j] = (r[j] - a[j] * u[j-1]) * beta
    for j in range(n-1,0,-1):
        u[j-1] -= g[j] * u[j]
    return u

def test_tdma():
    a = np.array([np.nan,3,3])
    c = np.array([2,1,np.nan])
    b = np.array([6,5,8])
    r = np.array([10,16,30])
    x = np.array([1,2,3])
    assert np.allclose(tdma(a,b,c,r), x)
    print('Test OK')

class ASpline(Logged):
    """
    Area-preserving spline

    (C)
    20140524 Alexander Heger, original concept
    20150115 Conrad Chan, add boundary conditions
    """

    def __init__(self, x, y,
                 magic = True,
                 silent = True,
                 **kwargs):
        """
        Initialize the interpolating spline.

        Input:
            x[0...n]        bin interface coordinates
            y[0...n-1]      bin average values

                |------|-------|------//--|--------|
                0      1       2         n-1       n    x[0...n]
                    0      1       2          n-1       y[0...n-1]

            magic           perform smearing operations
            silent          print log

        Example:
            Give the values of the zone interfaces (x) and the zone values (y):
            In [1]: x = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
            In [2]: y = np.array([5.,3.,4.,7.,8.,10.,15.,18.,23.])
            In [3]: s = ASpline(x, y)

            The spline objet can be called to return the value:
            In [4]: s(5)
            Out[4]: 8.3973774194162303

            As well as the first and second derivative:
            In [5]: s.prime(5)
            Out[5]: 1.3451668770736693
            In [6]: s.curve(5)
            Out[6]: 0.68865578471238809

            The spline can be integrated between 2 bounds:
            In [7]: s.integrate(1, 3)
            Out[7]: 7.0

        Boundary conditions:
            Boundary conditions can be specified at the two boundaries as
            f, f', or f''.
            f0, fn: f
            p0, pn: f'
            c0, cn: f''

            For example, to specify f = 1 on the left and f' = 1 on the right,
            initialize the spline as:
            In [7]: s = ASpline(x, y, f0 = 1, pn = 0)

        Magic:
            This attempts to eliminate artifical extrema by inserting new zones
            (smearing) the data. The distribution of y vs x is changed, but
            the area below the spline is conserved.
        """
        self.setup_logger(silent)
        self.x, self.y = x, y
        self.signature = []
        d = np.empty_like(self.y)
        d[:] = x[1:] - x[:-1]
        self.d = d
        self.n = d.shape[0]
        self.f = self._solve_interface(d, y, **kwargs)
        if magic:
            self._magic(**kwargs)
        self._set_coefficients()

    @staticmethod
    def _solve_interface(d, y,
                         f0 = None,
                         fn = None,
                         p0 = None,
                         pn = None,
                         c0 = None,
                         cn = None):
        """
        Find interface values.
        """
        n = d.shape[0]

        tl = np.ndarray(n + 1)
        td = np.ndarray(n + 1)
        tu = np.ndarray(n + 1)
        tr = np.ndarray(n + 1)

        if (f0, p0, c0).count(None) == 3:
            p0 = 0
        if (fn, pn, cn).count(None) == 3:
            pn = 0

        assert (f0, p0, c0).count(None) == 2
        assert (fn, pn, cn).count(None) == 2

        if f0 is not None:
            td[0] = d[0] / 3
            tu[0] = d[0] / 6
            tr[0] = y[0] - f0
        elif p0 is not None:
            td[0] = 1
            tu[0] = 0
            tr[0] = p0
        elif c0 is not None:
            td[0] = -1 / d[0]
            tu[0] =  1 / d[0]
            tr[0] = c0

        if fn is not None:
            tl[-1] = d[-1] / 6
            td[-1] = d[-1] / 3
            tr[-1] = fn - y[-1]
        elif pn is not None:
            tl[-1] = 0
            td[-1] = 1
            tr[-1] = pn
        elif cn is not None:
            tl[-1] = -1 / d[-1]
            td[-1] =  1 / d[-1]
            tr[-1] = cn
            pass

        tl[1:-1] = d[0:-1] / 6
        td[1:-1] = (d[0:-1] + d[1:]) / 3
        tu[1:-1] = d[1:] / 6
        tr[1:-1] = y[1:] - y[0:-1]

        f1 = tdma(tl, td, tu, tr)

        p = np.ndarray(n)
        q = np.ndarray(n)

        p[:] = f1[:-1]
        q[:] = (f1[1:] - f1[:-1]) / (2 * d[:])

        f = np.ndarray(n + 1)

        f[0] = y[0] - (p[0] * d[0] / 2) - (q[0] * d[0]**2 / 3)
        f[1:] = y[:] + (p[:] * d[:] / 2) + (2 * q[:] * d[:]**2 / 3)

        return f

    @staticmethod
    def _coefficients(d, y, f, n):
        """
        compute polynomial interpolant
        with same integral as flat interpolant
        but using common face boundaries

        The spline is a + b * t + c * t**2

        The coefficient are for an interval from
        [-0.5 ... +0.5] * d

        t = value - x[lower bound] - d / 2
        """
        a = np.ndarray(n)
        b = np.ndarray(n)
        c = np.ndarray(n)

        f2 = f[:-1] + f[1:]
        df = f[1:] - f[:-1]
        dip1 = 1 / d[:]
        a[:] = 0.25 * (6 * y[:] - f2)
        b[:] = df * dip1
        c[:] = 3 * (f2 - 2 * y[:]) * dip1**2

        return a, b, c

    def _set_coefficients(self):
        self.a, self.b, self.c = self._coefficients(self.d, self.y, self.f, self.n)

    def _find_extrema(self, d, y, f, n,
                      method = 'face'):
        """
        methods are
           peak - distance of peak to average
           face - min distance of face to average [cheap]
           zone - zone size [does not work well]
           fmin - min distance face to peak [FAILs]
           fmax - max distance face to peak

        TODO - allow neighboring extrema

        Layout of arrays:

             |------|-------|------//--|--------|
             0      1       2         n-1       n
                 0      1       2          n-1
             *      *       *          *        *    x[0, ..., n  ]
                 *       *        *          *       d[0, ..., n-1], y, df, se, de
             x      *       *          *        x   dy[1, ..., n-1]

        """

        df = np.ndarray((n, 2))
        df[:,0] = f[ :-1] - y[:]
        df[:,1] = f[1:  ] - y[:]
        # df[0 ,:] = 0. # should be np.nan

        dy = np.ndarray(n + 1)
        dy[1:-1] = y[1:] - y[:-1]
        dy[[0, -1]] = 0. # should be np.nan

        # SPLINE:
        #   extrema for
        #      abs(df1) > 2 * abs(df0) or
        #      abs(df0) > 2 * abs(df1) or
        #      df0 * df1 > 0
        #   min for (df0 + df1) > 0
        #   max for (df0 + df1) < 0
        se = (((abs(df[:, 1]) > 2 * abs(df[:, 0])) +
               (abs(df[:, 0]) > 2 * abs(df[:, 1])) +
               (df[:, 1] * df[:, 0] > 0))).astype(int)

        se[(df[:,0] + df[:,1]) > 0] *= -1
        se[0] = 0

        # add the magnitude of extrema for assessent of violation

        if method in ('peak', 'fmin', 'fmax'):
            a, b, c = self._coefficients(d, y, f, n)
            ii = c != 0
            # ii[0] = False
            jj = np.logical_not(ii)

            fm = np.ndarray(n)
            fm[ii] = a[ii] - 0.25 * b[ii]**2 / c[ii]
            fm[jj] = df[jj, 0] # just pick one, could have used 1
        elif method in ('face'):
            ii = np.tile(True, n)
            jj = np.logical_not(ii)
        if method in ('peak', 'fmin', 'fmax', 'face'):
            sd = np.ndarray(n)

        if method == 'fmin':
            # min peak to face values
            # seems to not work
            sd[ii] = np.minimum(np.abs(df[ii,0] + y[ii] - fm[ii]), np.abs(df[ii, 1] + y[ii] - fm[ii]))
        elif method == 'fmax':
            # min peak to face values
            # seems to not work
            sd[ii] = np.maximum(np.abs(df[ii,0] + y[ii] - fm[ii]), np.abs(df[ii, 1] + y[ii] - fm[ii]))
        elif method == 'peak':
            # max to average
            # seems to produce good results and mames sense
            sd[ii] = np.abs(y[ii] - fm[ii])
        elif method == 'face':
            # largest face to average
            # seems to produce good results, cheap
            # interpretation not 100% clear, but similaar to above
            sd[ii] = np.maximum(np.abs(df[ii, 0]), np.abs(df[ii, 1]))

        if method in ('peak', 'fmin', 'fmax', 'face'):
            # for all ...
            sd[jj] = 0.

        # DATA:
        #   extrema if dy[-] * dy[+] < 0
        #   min for either test dy[-] < 0 or dy[+] > 0
        #   max for either test dy[-] > 0 or dy[+] < 0
        de = np.ndarray(n, dtype = np.int)
        de[1:-1] = (dy[2:-1] * dy[1:-2] < 0).astype(int)
        de[1:-1][dy[2:-1] > 0] *= -1
        de[0]   = 1 - 2 * (dy[1  ] > 0).astype(int)
        de[n-1] = 1 - 2 * (dy[n-1] < 0).astype(int)
        if dy[1] == 0.:
            de[0] = 0
        if dy[n-1] == 0.:
            de[n-1] = 0
        # de[0] = 0

        # the spline will have formal extrema at the boundary because
        # of the f' == 0 boundary condition; copy type from data
        se[[1, n-1]] = de[[1, n-1]]

        # OK, now see if we violate somethings.  Strong violations.
        v = de != se

        if method in ('peak', 'fmin', 'fmax', 'face'):
            # biggest extrema
            # i = np.argmax(v[:-3].astype(int) * sd[:-3]) #Exclude the last 3
            i = np.argmax(v[:].astype(int) * sd[:])
        elif method == 'zone':
            # biggest zone - moderately OK
            i = np.argmax(v.astype(int)[:] * d[:])
        else:
            raise Exception('unknown method')

        # biggest area ? - similar to above
        # i = np.argmax(v.astype(int)[1:] * d[1:] * sd[1:]) + 1

        if not v[i]:
        #    print('no extrema')
           i = 0
        # else:
        #    print('{} at {:d}'.format(
        #        ['minimum', 'maximum'][(1 + se[i])//2],
        #        i))

        return i


    def _magic(self, smear = True, maxreplace = 100, method = None, chunksig = [], **kwargs):
        """
        Smear (add) zones to retain monotonicity of behaviour of data.

        Note:
          use smear = False only with method = 'zone'
        """
        d = self.d
        y = self.y
        n = self.n
        f = self.f
        if smear is False and method is None:
            method = 'zone'
        elif method is None:
            method = 'face'

        if chunksig == []:
            #Find new locations for operations and apply, then add to the new chunk
            for cycle in np.arange(maxreplace):
                i = self._find_extrema(d,y,f,n, method = method) - 1 # change code to use i
                if i == -1:
                    break
                d, y, new = self._operate(i, d, y, smear, method, False)
                n += 1
                f = self._solve_interface(d, y, **kwargs)
                chunksig += [new]
        else:
            #Apply operations in the supplied chunk
            for op in chunksig:
                d, y, new = self._operate(op, d, y, smear, method, True)
                n += 1
                f = self._solve_interface(d, y, **kwargs)

        if len(chunksig) > 0:
            self.signature += [chunksig]
            x = np.ndarray((len(y) + 1,))
            # x = np.empty_like(y)
            x[0] = self.x[0]
            x[1:] = x[0] + np.cumsum(d[:])
            self.x = x
            self.f = f
            self.d = d
            self.y = y
            self.n = n

        # print('replaced {} zones'.format(len(self.signature)))
        self._check(d,y,f,n)

    @staticmethod
    def _operate(i, d, y, smear = True, method = None, direct = False):
        # i: the zone containing violation
        # ii: chooses which zone to fix
        # if we operate on zone m, then we always inherit values from m-1 and m
        if not direct:
            dy0 = (y[i - 1] - y[i - 2])
            dy1 = (y[i    ] - y[i - 1])
            dy2 = (y[i + 1] - y[i    ])
            dy3 = (y[i + 2] - y[i + 1])

            # find the side to fix
            if abs(dy1) > abs(dy2):
                ii = 0
            else:
                ii = 1
            ii += i
            # In the old version, ii was used to determine which side to inherit from
            # and always inserted at i+1. Now we insert at ii+1.
            new = ii + 1
        else:
            # enabling direct changes the variable i into the location specifier, rather
            # than the extrema specifier
            new = i

        # insert new value
        y = np.insert(y, new, y[new])
        d = np.insert(d, new, d[new])

        # insert new zone using the 1/3 cut
        d_new = min(d[new - 1], d[new + 1]) / 3
        d[new - 1] -= d_new
        d[new    ]  = d_new * 2
        d[new + 1] -= d_new

        # new zone value is the average of neighbors
        y[new] = 0.5 * (y[new - 1] + y[new + 1])

        return d, y, new

    def remagic(self, signature, find = 0):
        """
        Re-apply magic. Specify how many new locations to find.
        """

        assert len(self.signature) <= len(signature)
        #Update changes
        for i, chunk in enumerate(signature):
            if i < len(self.signature):
                assert chunk == self.signature[i]
            else:
                self._magic(chunksig = chunk, maxreplace = 0)
                self._set_coefficients()

        #Make new changes
        self._magic(maxreplace = find, chunksig = [])
        self._set_coefficients()


    def _magic_old(self, smear = False, maxreplace = 100):
        """
        Old version of 'magic'
        """
        d = self.d
        y = self.y
        n = self.n
        f = self.f
        nreplace = 0
        for cycle in range(maxreplace):
            self._find_extrema(d, y, f, n)

            # TODO - find biggest violations in entire domain and
            # start fixing those first
            count = 0
            for i in range(2, n - 2):
                # df1 is 'b' coefficient (times d**2)
                # df0 is 'c' coefficient (times d**2)
                #    == curvature in normalized interval
                # df is location of extrema (times 2)
                #    relative to zone center
                #    x_max = -b / (2 * c)
                df1 =     (f[i + 1] - f[i]) * d[i + 1]
                df0 = 3 * (f[i + 1] + f[i] - 2 * y[i + 1])
                if df0 != 0:
                    df = -df1 / df0
                else:
                    df = 1.e99

                dy0 = (y[i    ] - y[i - 1])
                dy1 = (y[i + 1] - y[i    ])
                dy2 = (y[i + 2] - y[i + 1])
                dy3 = (y[i + 3] - y[i + 2])

                # one needs to identify what all of these mean ...
                if ((abs(df) < d[i + 1]) and # spline extrema in interval
                    not (((dy0 * dy1 < 0) and (df0 * dy0 < 0)) or # data extrema LHS and correct curvature
                         ((dy1 * dy2 < 0) and (df0 * dy1 < 0)) or # data extrema here and correct curvature
                         ((dy2 * dy3 < 0) and (df0 * dy2 < 0))) and # data extrema RHS and correct curvature
                    ((abs(dy1) < 0.5 * abs(dy2)) or    # extrema in interval ... needs to be applied to different quantity
                     (abs(dy2) < 0.5 * abs(dy1))) and  # ... all of these seem misguided (including following)
                    (dy1 * dy2 >= 0) and # no data extrema here
                    ((dy1 * dy0 >= 0) or # no data xtrema LHS
                     (dy2 * dy3 >= 0))): # no data extrema RHS

                    # insert new value
                    y = np.insert(y, i + 1, y[i + 1])
                    d = np.insert(d, i + 1, d[i + 1])
                    n += 1

                    if abs(dy1) > abs(dy2):
                        ii = 0
                    else:
                        ii = 1

                    # insert new zone
                    if smear:
                        ii += i
                        d_new = min(d[ii], d[ii + 2]) / 3
                        d[ii    ] -= d_new
                        d[ii + 1]  = d_new * 2
                        d[ii + 2] -= d_new

                        y_new = 0.5 * (y[ii] + y[ii + 2])
                        y[ii+1] = y_new

                        print('Inserting intermediate zone {}.'.format(ii + 1))
                    else:
                        # maybe adjust frac to max fix better?
                        frac = 1 / 3.
                        frac = np.array([frac, 1.- frac])

                        j0 = i + 1
                        jj = np.array([j0, j0])
                        if ((ii == 1) and
                            (abs(dy3) > abs(dy2)) and
                            (d[i+3] * d[i+2] > 0)):
                            jj += [1, 2]
                            j0 = i + 3
                            frac = np.array([0.5, 0.5])
                            ymul = - np.array([1, -1]) * frac[::-1]
                            dy = min(abs(dy3), abs(dy2)) * np.sign(dy2)
                        elif (
                            (d[i] > d[i+1]) and
                            (dy0 * dy1 > 0.)):
                            jj += [0, -1]
                            j0 = i
                            frac = np.array([0.5, 0.5])
                            ymul = - np.array([1, -1]) * frac[::-1]
                            dy = -min(abs(dy0), abs(dy1)) * np.sign(dy1)
                        else:
                            jj += [ii, 1 - ii]
                            j0 = i + 1
                            ymul = np.array([1, -1]) * frac[::-1]
                            dy = -dy2 if ii == 0 else dy1

                        d[jj] = d[j0] * frac
                        y[jj] = y[j0] + ymul * dy

                        print('Inserting intercell zone {}.'.format(i + ii + 1))

                    nreplace += 1
                    count += 1
                    break
            if count == 0:
                break
            f = self._solve_interface(d, y)

        if nreplace > 0:
            x = np.empty_like(y)
            x[0] = self.x[0]
            x[1:] = x[0] + np.cumsum(d[1:])
            self.x = x
            self.f = f
            self.d = d
            self.y = y
            self.n = n

        print('replaced {} zones'.format(nreplace))
        self._check(d,y,f,n)

    @staticmethod
    def _check(d, y, f, n):
        """
        Check for monotonicity conservation (doesn't do anything else).
        """
        # TODO: verctorize using numpy
        for i in range(2, n-2):
            df1 = (f[i    ] - y[i + 1])
            df2 = (f[i + 1] - y[i + 1])
            # check for extrema (value change on one side is twice
            # that on the other side, or both face values have same
            # relative location)
            if ((abs(df2) > 2 * abs(df1)) or
                (abs(df1) > 2 * abs(df2)) or
                (df1 * df2 > 0)):
                # check for monotonicity
                # we could write this enture section as ...
                # dy = y[i-1:i+3] - y[i:i+4]
                # if np.alltrue(dy[1:] * dy[:-1] > 0.): print('...')
                dy0 = y[i - 2] - y[i - 1]
                dy1 = y[i - 1] - y[i    ]
                dy2 = y[i    ] - y[i + 1]
                dy3 = y[i + 1] - y[i + 2]
                # the first and third comparison allow extrema zones
                # neighboring the peak zone.  Do we really want to
                # allow that?  Can it be prevented?
                #
                # second, it does not check for type of extrema, so
                # oscillations and extrema of wrong kind are not found.

                # if ((dy0 * dy1 > 0.) and
                #     (dy1 * dy2 > 0.) and
                #     (dy2 * dy3 > 0.)):
                #     print('Monotonicity strongly violated in zone {} - value {}'.format(i, y[i]))
                # # SPLINE:
                # #   extrema for
                # #      abs(df2) > 2. abs(df1) or
                # #      abs(df1) > 2. abs(df2) or
                # #      df2 * df1 > 0
                # #   min for (df1 + df2) > 0
                # #   max for (df1 + df2) < 0
                # # DATA:
                # #   extrema if dy1*dy2 < 0
                # #   min for dy1 < 0 or dy2 > 0
                # #   max for dy1 > 0 or dy2 < 0
                # elif (((df1 + df2) * dy1 < 0) and
                #     (dy1 * dy2 > 0)):
                #     print('Extrema of wrong type in zone {} - value {}'.format(i, y[i]))
                # # weak violation means extrema is in wrong bin
                # elif (dy1 * dy2 > 0.):
                #     print('Monotonicity weakly violated in zone {} - value {}'.format(i, y[i]))

    def _interval(self, x):
        """
        Compute interval for values and interpolation paramter.
        """
        i = np.searchsorted(self.x, x) - 1
        if isinstance(x, np.ndarray):
            j = np.where(i == -1)
            if len(j[0]) > 0:
                i[j] = 0
                k = np.where(np.isclose(self.x[0], x[j]))
                if len(k[0]) > 0:
                    x = x.copy()
                    (x[j])[k] = self.x[0]
            j = np.where(i == self.n)
            if len(j[0]) > 0:
                i[j] = self.n - 1
                k = np.where(np.isclose(self.x[self.n], x[j]))
                if len(k[0]) > 0:
                    x = x.copy()
                    (x[j])[k] = self.x[self.n]
        else:
            if i == -1:
                i = 0
                if np.allclose(self.x[0], x):
                    x = self.x[0]
            if i == self.n:
                i = self.n - 1
                if np.allclose(self.x[self.n], x):
                    x = self.x[self.n]

        t = x - self.x[i] - 0.5 * self.d[i]
        return t, i

    def __call__(self, x):
        """
        Evaluate the spline.
        """
        t, i = self._interval(x)
        y = self.a[i] + (self.b[i] + self.c[i] * t) * t
        return y

    def integrate(self, x1, x2):
        """
        Integrate the spline from x1 to x2.
        """
        assert x2 > x1
        I = 0
        i3 = 1 / 3
        t1, i1 = self._interval(x1)
        t2, i2 = self._interval(x2)

        # Translated value at the zone interface right of the lower integration bound
        tr = 0.5 * self.d[i1]
        I -= (self.a[i1] + (0.5 * self.b[i1] + i3 * self.c[i1] * t1) * t1) * t1
        I += (self.a[i1] + (0.5 * self.b[i1] + i3 * self.c[i1] * tr) * tr) * tr

        # Translated value at the zone interface left of the upper integration bound
        tl = -0.5 * self.d[i2]
        I -= (self.a[i2] + (0.5 * self.b[i2] + i3 * self.c[i2] * tl) * tl) * tl
        I += (self.a[i2] + (0.5 * self.b[i2] + i3 * self.c[i2] * t2) * t2) * t2

        # In-between cells
        I -= self.y[i1] * self.d[i1]
        for i in range(i1, i2):
            I += self.y[i] * self.d[i]

        return I

    def prime(self, x):
        """
        Evaluate spline derivative.
        """
        t, i = self._interval(x)
        y = self.b[i] + 2 * self.c[i] * t
        return y

    def curve(self, x):
        """
        Evaluate spline curvature.
        """
        t, i = self._interval(x)
        y = 2 * self.c[i]
        return y

class sample_data(Logged):
    def __init__(self,
                 #filename = '/u/alex/x/ed250z0.presn.flash'):
                 filename = '../hydro/ed250z0.presn.flash'):
        self.data = np.loadtxt(filename)

    def _value(self, index, center = True):
        """Return arrays compatible with KepDump"""
        x = np.zeros(self.data.shape[0]+2)
        x[1:-1] = self.data[:, index]
        x[  -1] = np.nan
        if center:
            x[0] = np.nan
        return x

    @CachedAttribute
    def zm(self):
        '''interior mass coordinate (g)'''
        return self._value(0, center = False)

    @CachedAttribute
    def rn(self):
        '''radius coordinate (cm)'''
        return self._value(1, center = False)

    @CachedAttribute
    def un(self):
        '''interface velocity (cm/s)'''
        return self._value(2, center = False)

    @CachedAttribute
    def dn(self):
        '''cell density (g/cm**3)'''
        return self._value(3, center = True)

    @CachedAttribute
    def tn(self):
        '''cell temperature (K)'''
        return self._value(4, center = True)

    # m=[0,a[*,0]]
    # r=[0,a[*,1]]
    # v=[0,a[*,2]]
    # d=a[*,3]
    # t=a[*,4]
    # p=a[*,5]
    # e=a[*,6]
    # s=a[*,7]
    # w=a[*,8]
    # abar=a[*,9]
    # ye=a[*,10]
    # x=a[*,11:29]

def set_axvlines(x,
                 xmin = 0,
                 xmax = 1,
                 axes = None,
                 **kwargs):
    if axes is None:
        axes = plt.gca()
    trans = mpl.transforms.blended_transform_factory(
        axes.transData, axes.transAxes)
    segments = np.ndarray([x.shape[0], 2, 2])
    # todo: allow scalar for x
    segments[:,:,0] = np.transpose([x, x])
    # todo: allow arrays
    segments[:,:,1] = np.array([xmin, xmax])[np.newaxis,:]
    lines = mpl.collections.LineCollection(
            segments,
            transform = trans,
            **kwargs)
    axes.add_collection(lines)

def label_zones(axes, x, irange):
    mpl.rcParams['text.usetex'] = False
    if axes is None:
        axes = plt.gca()
    trans = mpl.transforms.blended_transform_factory(
        axes.transData, axes.transAxes)
    for i in irange:
        axes.text(0.5*(x[i]+x[i+1]), 0.98, str(i+1),
                  verticalalignment = 'top',
                  horizontalalignment='center',
                  size = 10,
                  color = '#cccc00',
                  zorder = -10,
                  transform = trans)

def test():
    data = sample_data()
    data = kepdump.load('/home/alex/kepler/test/u8.1#presn')
    x = data.rn[:-1]
    y = data.dn[:-1]
    f = plt.figure()
    ax = f.add_subplot(111)
    set_axvlines(x,
                 axes = ax,
                 color = '#cfcfcf',
                 linewidth = 3.,
                 )
    ax.hlines(y = y[1:],
              xmin = x[:-1],
              xmax = x[1:],
              color = '#ffcfcf',
              linewidth = 3.,
              )
    s = ASpline(x,y, magic = True, smear = True, maxreplace=999)
    set_axvlines(s.x,
                 axes = ax,
                 color = 'k',
                 linewidth = .5,
                 )
    ax.hlines(y = s.y[1:],
              xmin = s.x[:-1],
              xmax = s.x[1:],
              color = 'r',
              linewidth = .5,
              zorder = 10,
              )

    xr = np.array([1.535, 1.56]) * 1.e14
    yr = np.array([0., 1.]) * 2.e-8

    xr = np.array([1.465,1.55])*1.e14
    yr = np.array([0.75,1.45])*1.e-9

    # xr = np.array([0.9,1.1]) *  38795082629.909073
    # yr = np.array([0,10])

    ax.set_xlim(xr)
    ax.set_ylim(yr)

    n = 100
    xx = np.ndarray(n * (s.x.shape[0] - 1) + 1)
    for i in range(s.x.shape[0] - 1):
        xx[n*i:n*(i+1)] = s.x[i] + s.d[i+1] * np.linspace(0,1,n,False)
    xx[-1] = s.x[-1]
    yy = s(xx)
    # xx,yy = reduce.reduce(xx,yy, axes=ax)
    ax.plot(xx,yy,color='g',linewidth=0.5)

    # label_zones(ax, s.x, range(1305, min(1325, len(s.x)-1)))
    label_zones(ax, s.x, range(1005, min(10015, len(s.x)-1)))

    plt.draw()
    return s

def test2():
    filename = '/home/alex/x/SFHagefullFnx.dat'
    #filename = '../hydro/SFHagefullFnx.dat'
    data = np.loadtxt(filename)
    n = data.shape[0]
    y = np.ndarray(n)
    y[:] = data[:,2]
    x = np.zeros(n+1)
    x[0:-1] = data[:,0] - 0.5 * data[:,1]
    x[-1] = data[-1,0] + 0.5 * data[-1,1]
    s = ASpline(x, y, magic = True, c0 = 0, cn = 0)

    t = np.linspace(0.25, 14, 100000)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('age (Gyr)')
    ax.set_ylabel('SFR (whateverunit)')

    ax.errorbar(data[:,0], data[:,2], xerr=data[:,1]*0.5, yerr=data[:,3], fmt='o')
    ax.axhline(y=0, color = 'k', ls='--')
    ax.plot(t,s(t))
    ax.plot(t,s.prime(t))
    ax.plot(t,s.curve(t))
    ax.set_xlim(0, 14)
    plt.show()
