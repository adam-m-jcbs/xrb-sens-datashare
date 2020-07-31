"""
Reduce plot size.

(C) Alexander Heger, August 2000

IDEA:
remove datapoints that result into ploting below the
specified resolution.

HOW IT WORKS:
From a starting point determine opening angle under which an
ellipse with major axis of the given resolution appears.  Compute
Intersection of this opening angle and the previous.  Keep only
most remote point as long as there is an overlap of the opening
angles (i.e., remove intermediate points that lay on a streight
line within the given resolution).

if [x|y]log then res==d[x|y]/[x|y] ELSE res==d[x|y]


HISTORY:
20001230 - rewritten
20001231 - debugged (variable d2 introduced)
20010101 - parameter PLOT added
20010814 - support for XY data in one array
20030604 - TRUNCATE parameter added

20120812 - implementation in Python
"""

import numpy as np
import matplotlib.pyplot as plt

from logged import Logged

def reduce(*args, **kwargs):
    """
    Create and call Reduce object.
    """
    r = Reduce(*args, **kwargs)
    return r()

class Reduce(Logged):
    """
    Remove superfluous points from polygon based on resolution.

    PARAMETERS
    truncate:
        None   - Do nothing.
        'cut'  - Keep first point outside
                (default)
        'clip' - truncate coordinate to range
                 May change slope of lines
    method: (values can be added)
        1 - remove 'dense points'
            Point within one pixel
        2 - remove points on horz/vert line between points
            (within one pixel)
        4 - remove point on straight lines at arbitrary angle
        default: 7
    axes:
        axes object that can be used to extract range, log, res
    vx:
        (n), (2,n) or (n,2) array for reduction
        if (n) then vy must be present as well
        otherwise vy must not be present
    vy:
        2nd coordinate if vx is 1D array
        must have same dimension as vx
    log*:
        True  - Axis data is to be reduced logarithmically (base e)
        False - Axis data is to be reduced linearly
    range_*:
        range data [min, max]
    ranges:
        range data [[xmin, xmax],[ymin,ymanx]]
    res*:
        absolute resolution info
    relres*:
        relative resolution info
    silent:
        True/False (default)

    METHODS
    __call__:
        PARAMETERS
        array_mode:
            0    - return (x,y) tuple
            1    - return np.ndarray((2,n))
            2    - return np.ndarray((n,2))
            None - return same as input format
                   (default)
        RETURN VALUE
        reduced data in same layout as input:
        vx only: one (2,n) or (n,2) array
        vx, vy:  return tuple of 1D arrays
    """

    def __init__(self,
                 vx,
                 vy = None,

                 axes = None,

                 res = None,
                 res_x = None,
                 res_y = None,

                 relres = None,
                 relres_x = None,
                 relres_y = None,

                 ranges = None,
                 range_x = None,
                 range_y = None,

                 log = None,
                 log_x = None,
                 log_y = None,

                 truncate = 'cut',
                 method = 7,

                 silent = False):

        self.setup_logger(silent)

        if ranges is not None:
            assert np.shape(ranges) == (2,2)
            if range_x is None:
                range_x = ranges[0]
            if range_y is None:
                range_y = ranges[1]

        if log is not None:
            if log_x is None:
                log_x = log
            if log_y is None:
                log_y = log

        if res is not None:
            if res_x is None:
                res_x = res
            if res_y is None:
                res_y = res

        if relres is not None:
            if relres >= 1.:
                relres = 1./relres
            if relres_x is None:
                relres_x = relres
            if relres_y is None:
                relres_y = relres

        if relres_x is not None:
            if relres_x >= 1.:
                relres_x = 1./relres_x
        if relres_y is not None:
            if relres_y >= 1.:
                relres_y = 1./relres_y

        if axes is not None:
            if range_x is None:
                range_x = np.array(axes.get_xlim())
            if range_y is None:
                range_y = np.array(axes.get_ylim())
            if log_x is None:
                log_x = axes.get_xscale() == 'log'
            if log_y is None:
                log_y = axes.get_yscale() == 'log'
            if res_x is None and relres_x is None:
                relres_x = 0.5/axes.bbox.width
            if res_y is None and relres_y is None:
                relres_y = 0.5/axes.bbox.height

        if log_x is None:
            log_x = False
        if log_y is None:
            log_y = False

        if res_x is None and relres_x is None:
            relres_x = 1.e-3
        if res_y is None and relres_y is None:
            relres_y = 1.e-3

        nnx0 = len(vx)
        nny0 = len(vy) if vy is not None else (0,)

        array_mode = 0
        nndx = vx.ndim
        ndx = vx.shape
        if nndx == 2:
            assert vy is None, 'Accepting only one muli-D array.'
            if ndx[0] == 2:
                array_mode = 1
                vy = vx[1,:]
                vx = vx[0,:]
            elif ndx[1] == 2:
                array_mode = 2
                vy = vx[:,1]
                vx = vx[:,0]
        self.logger.info('array_mode: {:1d}'.format(array_mode))
        self.array_mode = array_mode

        n = vx.size
        assert vy.size == n, 'dimension error'

        if np.size(range_x) != 2:
            range_x = np.array([vx.min(), vx.max()])
        if np.size(range_y) != 2:
            range_y = np.array([vy.min(), vy.max()])

        if np.size(range_x) == 2:
            range_x = np.array([range_x.min(), range_x.max()])
            if truncate == 'clip':
                vx = np.maximum(np.minimum(vx, range_x[1]), range_x[0])
            if res_x is None:
                if log_x:
                    res_x = (np.log(range_x[1])-np.log(range_x[0])) * relres_x
                else:
                    res_x = (range_x[1] - range_x[0]) * relres_x

        if np.size(range_y) == 2:
            range_y = np.array([range_y.min(), range_y.max()])
            if truncate == 'clip':
                vy = np.maximum(np.minimum(vy, range_y[1]), range_y[0])
            if res_y is None:
                if log_y:
                    res_y = (np.log(range_y[1])-np.log(range_y[0])) * relres_y
                else:
                    res_y = (range_y[1] - range_y[0]) * relres_y

        if not isinstance(range_x, np.ndarray):
            assert len(range_x) == 2, 'invalid range x'
            range_x = np.ndarray(range_x)
        if not isinstance(range_y, np.ndarray):
            assert len(range_y) == 2, 'invalid range y'
            range_y = np.ndarray(range_y)

        nnx0 = len(vx)
        nny0 = len(vy)

        x = vx
        y = vy

        # delete data points outside of range
        # TODO - remove point at neighboring quadrants?
        if ( (np.size(range_x) == np.size(range_y) == 2) and
             (truncate == 'cut') ):
            timer = 'Truncate'
            self.add_timer(timer)
            quad = (
                ( np.array((x >= range_x[0]), dtype=np.int) +
                  np.array((x > range_x[1]), dtype=np.int))
                + 3 *
                ( np.array((y >= range_y[0]), dtype=np.int) +
                  np.array((y > range_y[1]), dtype=np.int))
                )

            map = np.ndarray(n, dtype=np.int)
            n1 = 0
            i = 0
            i1 = -1
            while i < n:
                add = quad[i] == 4
                if i > 0 and quad[i] != quad[i-1]:
                    if i1 != i-1:
                        map[n1] = i-1
                        n1 += 1
                    add = True
                if add:
                    map[n1] = i
                    n1 += 1
                    i1 = i
                i += 1

            # reduce vector length
            self.logger.info('{:s}: reduction {:d} to {:d} points ({:f}).'.format(
                    timer,
                    n,
                    n1,
                    n1/n))
            n = n1
            map = map[:n]
            x = x[map]
            y = y[map]
            self.logger_timing(timer = timer,
                               finish = True)
        else:
            map = np.arange(n, dtype=np.int)


        # some specials for the log treatment
        # use working array with log values and scaled by resolution
        # use map to select final choice of elements
        res_xi = 1. / res_x
        res_yi = 1. / res_y

        if log_x:
            x = np.log(x) * res_xi
        else:
            x = x * res_xi
        if log_y:
            y = np.log(y) * res_yi
        else:
            y = y * res_yi

        if method & 1:
            # --------------------
            # PART_I:
            # --------------------
            # the obvious: remove dense points
            # keep, however, first & last point
            # remove point when within a ellipse with major axis res_x and res_y
            timer = 'Method 1'
            self.add_timer(timer)
            i1 = 0
            n1 = 0
            xmap = np.ndarray(n, dtype=np.int64)
            xmap[n1] = i1
            for i in range(1, n - 1):
                dx = x[i] - x[i1]
                dy = y[i] - y[i1]
                if (dx**2 + dy**2) > 1.:
                    i1 = i
                    n1 = n1 + 1
                    xmap[n1] = i1
            i1 = n - 1
            n1 += 1
            xmap[n1] = i1

            # reduce vector length
            self.logger.info('{:s}: reduction {:d} to {:d} points ({:f}).'.format(
                    timer,
                    n,
                    n1 + 1,
                    (n1+1)/n))
            n = n1 + 1
            xmap = xmap[:n]
            map = map[xmap]
            x = x[xmap]
            y = y[xmap]
            self.logger_timing(timer = timer,
                               finish = True)

        if method & 2:
            # --------------------
            # PART_II:
            # --------------------
            # remove horz/vert lines intermediate points
            # but we need to assure that point at extrema are not lost
            timer = 'Method 2'
            self.add_timer(timer)
            i1 = 0
            n1 = 0
            px = mx = 0.
            py = my = 0.
            xmap = np.ndarray(n, dtype=np.int64)
            xmap[n1] = i1
            for i in range(2, n - 1):
                dx  = x[i] - x[i1]
                dy  = y[i] - y[i1]
                px = max(px, dx)
                mx = min(mx, dx)
                py = max(py, dy)
                my = min(my, dy)

                dx2 = dx**2
                dy2 = dy**2
                if ((dx2 > 1. and dy2 > 1.) or
                    (dx2 > 1. and mx < dx < px) or
                    (dy2 > 1. and my < dy < py)):
                    i1 = i - 1
                    n1 += 1
                    xmap[n1] = i1
                    px = mx = x[i] - x[i1]
                    py = my = y[i] - y[i1]
            i1 = n - 1
            n1 = n1 + 1
            xmap[n1] = i1

            # reduce vector length
            self.logger.info('{:s}: reduction {:d} to {:d} points ({:f}).'.format(
                    timer,
                    n,
                    n1 + 1,
                    (n1+1)/n))
            n = n1 + 1
            xmap = xmap[:n]
            map = map[xmap]
            x = x[xmap]
            y = y[xmap]
            self.logger_timing(timer = timer,
                               finish = True)

        if method & 4:
            # --------------------
            # PART_III:
            # --------------------
            # the difficult: non-horz./vert. lines
            # keep first and last point
            timer = 'Method 4'
            self.add_timer(timer)
            pi = np.pi
            pi2 = 2. * pi

            n1 = 0
            i0 = 0

            i1 = 1
            i2 = 1
            d1 = 0.
            d2 = 0.
            dchi = -1.

            i = 1
            xmap = np.ndarray(n, dtype=np.int64)
            xmap[n1] = i0
            while i < n:
                remote = False

                dx = x[i] - x[i0]
                dy = y[i] - y[i0]
                d = np.sqrt(dx**2 + dy**2)

                if (d <= 1.):
                    remote = True
                else:
                    # d > 1
                    psi = np.arcsin(1./d) * 0.5 # 0 < 2*psi < pi/2
                    phi = np.arctan2(dy, dx) # -pi < phi < pi
                    phi = np.mod(phi + psi, pi2) # make sure all angles are positive
                    dphi = 2. * psi # < pi
                    if (dchi < 0.):
                        # setup of first point (with d > 1)
                        chi = phi
                        dchi = dphi
                        d1 = d
                        remote = True
                    else:
                        # compute overlap
                        xi = np.mod(phi - chi, pi2)
                        if (xi > pi):
                            xi -= pi2

                if not remote:
                    if xi < 0.:
                        chi = phi
                        dchi = np.minimum(dchi+xi, dphi)
                    else:
                        dchi = np.minimum(dphi-xi, dchi)
                    if dchi < 0.:
                        # point is out of line
                        n1 += 1
                        xmap[n1] = i1
                        if d1 > (d2 + 1.):
#                        if (i2 > i1):
                            n1 += 1
                            xmap[n1] = i2
                        # start new segment
                        d1 = 0.
                        i0 = i - 1
                        i1 = i
                        i2 = i
                        d2 = d
                        dchi = -1.
                        # redo current point
                        continue

                # save current coordiantes
                i2 = i
                d2 = d
                # check if most remote point in that direction
                if d >= d1:
                    i1 = i
                    d1 = d

                # end of loop:
                i += 1

            n1 += 1
            xmap[n1] = i1
            if i2 > i1:
                n1 = n1 + 1
                xmap[n1] = i2

            # reduce vector length
            self.logger.info('{:s}: reduction {:d} to {:d} points ({:f}).'.format(
                    timer,
                    n,
                    n1 + 1,
                    (n1+1)/n))
            n = n1 + 1
            xmap = xmap[:n]
            map = map[xmap]
            x = x[xmap]
            y = y[xmap]
            self.logger_timing(timer = timer,
                               finish = True)

        # --------------------
        # FINAL:
        # --------------------
        x = 0
        y = 0

        # assign values
        self.vx = vx[map]
        self.vy = vy[map]
        self.map = map

        nnx1 = self.vx.size
        nny1 = self.vy.size
        self.logger.info('Reduction from {:d} to {:d} points ({:f})'.format(
                nnx0,
                nnx1,
                (nnx1 + nny1)/(nnx0 + nny0)))

        self.close_logger(timing = 'finished in')

    def __call__(self, array_mode = None):
        """
        Return reduced arrays in desired format.

        (input format by default)
        """

        if array_mode is None:
            array_mode = self.array_mode
        if self.array_mode == 1:
            v = np.array([self.vx,self.vy])
            return v
        elif self.array_mode == 2:
            v = np.array([self.vx,self.vy]).transpose()
            return v
        return self.vx, self.vy

def test_reduce():

    import matplotlib.pyplot as plt
    import matplotlib.patches as pat
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n = 100000
    t = np.linspace(0,1,num = n)

    # t=randomu(1,n+1)
    # t=t^100

    t = t * 2 * np.pi

    # x=exp(cos(3*t)*2)
    # y=sin(exp(2.5*t/MAX(t)))

    # y=cos(t)
    # x=sin(5*t)*exp(5*t/MAX(t))

    # y=cos(t*!DPi)
    # x=sin(5*t)*exp(5*t/MAX(t))

    y = np.cos(t)**5
    x = np.sin(t)**5

    # rotate
    phi = np.pi / 3
    c = np.cos(phi)
    s = np.sin(phi)
    x1 = c * x - s * y
    y = s * x + c * y
    x = x1

    # translate
    offset = [2.,0.]

    # range
    ranges = np.array([[-0.35, 0.35],[-0.75, 0.75]])

    x += offset[0]
    y += offset[1]
    ranges += np.array([offset,offset]).transpose()


    ax.set_xscale('log')

    ### plot(x,y,/NODATA,XMARGIN=[1,1],YMARGIN=[1,1]
    ax.plot(x,y,'k')
#    xx = np.array([x,y])
    xx = np.array([x,y]).transpose()

    # xx=DBLARR(2,n+1)
    # xx[0,*]=x
    # xx[1,*]=y

    r = pat.Rectangle((ranges[0,0],ranges[1,0]),
                      ranges[0,1]-ranges[0,0],
                      ranges[1,1]-ranges[1,0],
                      fill = True,
                      edgecolor = None,
                      facecolor = 'b',
                      alpha= 0.1)
    ax.add_patch(r)
    ax.axhline(ranges[1,0],
               color = 'b',
               alpha= 0.1)
    ax.axhline(ranges[1,1],
               color = 'b',
               alpha= 0.1)
    ax.axvline(ranges[0,0],
               color = 'b',
               alpha= 0.1)
    ax.axvline(ranges[0,1],
               color = 'b',
               alpha= 0.1)


    #    x,y = reduce(x, y,
    xx = reduce(xx,
                method = 7,
                relres = 1.e-3,
                axes = ax,
                range_x = np.array(ranges[0]),
                range_y = np.array(ranges[1]),
#                truncate = None,
#                truncate = 'cut',
#                truncate = 'clip',
                silent = False)
#                silent = True)

    # x = xx[0,:]
    # y = xx[1,:]
    x = xx[:,0]
    y = xx[:,1]

    ax.plot(x,y,'g')
    ax.plot(x,y,'rx')
    ax.plot(x[0],y[0],'ro')
    ax.plot(x[-1],y[-1],'r^')

    # help,x

    nn = x.size

    print(' Reduction from {:d} to {:d} (factor {:f})'.format(
          n,
          nn,
          n/nn))

    plt.show()
