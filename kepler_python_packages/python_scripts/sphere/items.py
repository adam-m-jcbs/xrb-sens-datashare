"""
library of scene itms
"""

import numpy as np
from numpy import linalg as LA

from rotation import rotate2, Rotator, rotscale, w2axyz, NoRotator, rotate, wchain
from rotation import ex, ey, ez, basis

# from ..rot import w2zyx, w2xyz

from .base import SceneItem
from .primitives import tube, cone, cylinder, sphere, arrow, rotation_ellipsoid

class Colors():
    red = np.asarray([1., 0., 0.])
    green = np.asarray([0., 1., 0.])
    blue = np.asarray([0., 0., 1.])
    yellow = np.asarray([1., 1., 0.])
    white = np.asarray([1., 1., 1.])
    black = np.asarray([0., 0., 0.])
    cyan = np.asarray([0., 1., 1.])
    magenta = np.asarray([1., 0., 1.])
    grey = np.asarray([.7, .7, .7])


def f1f2(nV):
    f2 = np.arange(nV) / (nV - 1)
    f1 = 1 - f2
    return f1, f2

def normalize(v):
    v = np.asarray(v)
    return v / LA.norm(v)

class Trace(SceneItem):
    def __init__(self, r1, r2, c1, c2, pos,
                 w1 = None,
                 w2 = None,
                 alpha = None,
                 alpha2 = None,
                 linear = False,
                 res = 10,
                 phase = None,
                 rad = 0.02,
                 nTv = 50,
                 fixedtime = None,
                 **kwargs):
        super().__init__()

        # if len(kwargs) > 0:
        #     print(f'[Trace] Extra keywords: {kwargs}')

        self.w1 = np.asarray(w1)
        self.w2 = np.asarray(w2)
        self.res = res

        pos = np.asarray(pos)
        pos = pos / LA.norm(pos)

        self.pos = pos
        self.r1 = r1
        self.r2 = r2
        self.linear = linear
        self.phase = phase

        self.c1 = np.asarray(c1)
        self.c2 = np.asarray(c2)

        if alpha is None:
            alpha = 1.
        if alpha2 is None:
            alpha2 = alpha
        if self.c1.shape == (3,):
            self.c1 = np.append(c1, alpha)
        else:
            self.c1[3] *= alpha
        if self.c2.shape == (3,):
            self.c2 = np.append(c2, alpha2)
        else:
            self.c2[3] *= alpha2

        self.rad = rad
        self.nTv = nTv
        self.fixedtime = fixedtime
        self.static = fixedtime is not None

    def draw(self, time = None):
        static = time is None
        if static != self.static:
            return
        if self.static:
            time = self.fixedtime
        nV = 10
        if self.w1 is not None:
            nV += int(LA.norm(self.w1-self.w2) * time * self.res)
        f1, f2 = f1f2(nV)
        if self.w1 is not None:
            if self.linear:
                w = np.outer(f1, self.w1) + np.outer(f2, self.w2)
            else:
                w = rotscale(self.w1, self.w2, nV)

            rot = Rotator(w, time, self.phase)
        else:
            rot = NoRotator()
        v = np.outer(f1 * self.r1 + f2 * self.r2, self.pos)
        v = rot(v, align = 1)
        c = np.outer(f1, self.c1) + np.outer(f2, self.c2)

        self.add_actor(
            tube(v, self.rad, c, self.nTv, capping = True),
            static)

    draw_static = draw


class Rot(SceneItem):
    def __init__(self, p, c, w,
                 theta = None,
                 dt = None,
                 c2 = None,
                 alpha = 1.,
                 alpha2 = None,
                 rad = 0.01,
                 phase = None,
                 hat = None,
                 hat_size = None,
                 hat_include = True,
                 arrow = False,
                 arrow_cone = 0.1,
                 arrow_shaft = 0.03,
                 arrow_tip = 0.35,
                 res = 50,
                 theta0 = 0,
                 t0 = 0,
                 fixedtime = None,
                 ):

        # TODO - track on cone or sphere from p1 to p2 with axis w
        #        likely separate class due to issue with hat

        # TODO - similar to arrow, lenfth should be tip to toe, not
        #        adding hat to length.
        super().__init__()
        self.fixedtime = fixedtime

        wn = LA.norm(w)
        self.void = wn < 1.e-10
        if self.void:
            return

        if theta is None:
            if dt is None:
                dt = 1
            theta = wn * dt
            theta0 = wn * t0
        nV = 10
        nV += int(LA.norm(p) * np.abs(theta) * res)
        p = np.asarray(p)
        c1 = np.asarray(c)
        if c2 is None:
            c2 = c
        else:
            c2 = np.asrray(c2)
        if alpha is None:
            alpha = 1.
        alpha1 = alpha
        if alpha2 is None:
            alpha2 = alpha
        f1, f2 = f1f2(nV)
        c1 = np.asarray([*c1, alpha1])
        c2 = np.asarray([*c2, alpha2])
        self.c = np.multiply.outer(f1, c1) + np.multiply.outer(f2, c2)
        if arrow:
            rad = theta * arrow_shaft * LA.norm(np.cross(p, w)) / wn
        if hat not in (None, False, ):
            if hat_size is None:
                hat_size = rad * arrow_cone / arrow_shaft
            radius = hat_size
            height = radius * arrow_tip / arrow_cone
            if hat_include:
                theta -= height / LA.norm(np.cross(p, w))
            pos = Rotator(w)(p, (theta + theta0) / wn)

        t = (f2 * theta + theta0) / wn
        self.rad = rad
        self.res = res
        self.rot = Rotator(w, phase = phase)
        v = Rotator(w)(p, t)
        self.v = v

        # --------
        # draw hat
        # --------
        self.hat = hat
        if hat in (None, False, ):
            return
        if hat in ('straight', 'circle', True,):
            resh = int(res * hat_size / rad)
            if hat in ('straight', ):
                u = np.cross(w, pos)
            else: # 'circle', True
                t = height / LA.norm(np.cross(pos, w))
                v = Rotator(w)(pos, t)
                u = v - pos
            self.pos = pos
            self.u = u
            self.resh = resh
        elif hat in ('curved', ):
            nV = int(res * hat_size / rad)
            g1, g2 = f1f2(nV)
            t = g2 * height / LA.norm(np.cross(pos, w))
            vc = Rotator(w)(pos, t)
            self.radc = radius * g1
            self.nV = nV
            self.vc = vc
        else:
            raise AttributeError('Invalid hat type.')
        self.hat_size = hat_size
        self.radius = radius
        self.height = height

    def draw_static(self):
        if self.fixedtime is not None:
            self.mydraw(self.fixedtime, static = True)

    def draw(self, time):
        if self.fixedtime is None:
            self.mydraw(time)

    def mydraw(self, time, static = False):
        if self.void:
            return
        rot = self.rot.settime(time)
        v = rot(self.v, time = time)
        self.add_actor(tube(v, self.rad, self.c, self.res, capping = True))

        # -----------------------------------------------------------------------
        # no arrow
        # -----------------------------------------------------------------------
        if self.hat in (None, False, ):
            return

        # -----------------------------------------------------------------------
        # draw straight arrow
        # -----------------------------------------------------------------------
        if self.hat in ('straight', 'circle', True,):
            pos, u = rot([self.pos, self.u])
            self.add_actor(
                cone(self.radius, self.height, 1., pos, u, self.c[-1],
                     res = self.resh,
                     )
               )
            return
        # -----------------------------------------------------------------------
        # draw *curved*  arrow hat
        # -----------------------------------------------------------------------
        vc = rot(self.vc)
        self.add_actor(tube(vc, self.radc, self.c[-1], self.nV,
                            capping = True),
                       static)


class Arc(SceneItem):
    def __init__(self, p1, p2, c,
                 c2 = None,
                 alpha = 1,
                 alpha2 = None,
                 rad = 0.01,
                 w = None,
                 time = None,
                 phase = None,
                 res = 50,
                 ):
        super().__init__()

        nV = 10
        nV += int(LA.norm(p2 - p1) * res)
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        c1 = c
        if c2 is None:
            c2 = c
        alpha1 = alpha
        if alpha2 is None:
            alpha2 = alpha
        c1 = np.asarray([*c1, alpha1])
        c2 = np.asarray([*c2, alpha2])
        f1, f2 = f1f2(nV)
        self.c = np.multiply.outer(f1, c1) + np.multiply.outer(f2, c2)
        self.rad = rad
        self.res = res
        if w is not None:
            self.rot = Rotator(w, phase = phase)
        else:
            self.rot = NoRotator()
        self.v = rotscale(p1, p2, nV)

    def draw(self, time):
        v = self.rot(self.v, time)
        self.add_actor(tube(v, self.rad, self.c, self.res, capping = True))


class Coord(SceneItem):
    def __init__(self, r, c,
                 w, pos = None,
                 alpha = 0.1,
                 rad = 0.01,
                 phase = None,
                 res = 50,
                 phi = None,
                 lat = False,
                 ):
        super().__init__()

        nV = 10
        nV += int(np.pi * 2 * res)

        c = np.asarray([*c, alpha])
        self.c = np.tile(c, (nV, 1))
        self.rad = rad
        self.res = res
        if pos is not None:
            pos = normalize(pos) * r
        else:
            # use phi / lat
            raise AttributeError('to be implemented')
        self.rot = Rotator(w, phase = phase)
        f1, f2 = f1f2(nV)
        phi = 2. * np.pi * f2
        if lat is False:
            w = np.cross(pos, w)
            w /= LA.norm(w)
        self.v = Rotator(w)(pos, phi / LA.norm(w))

    def draw(self, time):
        v = self.rot(self.v, time)
        self.add_actor(tube(v, self.rad, self.c, self.res, capping = False))


class CoordGrid(SceneItem):
    def __init__(self,
                 r, c, w, pos,
                 ntheta = None,
                 nphi = None,
                 **kwargs):
        """
        nphi - number of sections per 180 deg
        ntheta - number of sections per 90 deg
        """
        super().__init__()

        if ntheta is None and nphi is None:
            ntheta = 6
        if nphi is None:
            nphi = 2 * ntheta
        if ntheta is None:
            ntheta = (nphi + 1) // 2

        # find origin on same longitude as pos
        wn = LA.norm(w)
        pos = pos - np.dot(w, pos) / wn
        pos = r * pos / LA.norm(pos)
        n = np.round

        # longitudes
        t = np.pi / nphi * np.arange(nphi) / wn
        lopos = Rotator(w)(pos, t)
        for lp in lopos:
            self.add_item(Coord(r, c, w, pos = lp, lat = False, **kwargs))

        # latitudes
        t = (np.arange(2 * ntheta - 1) - (ntheta - 1)) / ntheta
        lapos = rotscale(pos, w, steps = t)
        for lp in lapos:
            self.add_item(Coord(r, c, w, pos = lp, lat = True, **kwargs))

class Trail(SceneItem):
    def __init__(self, r, c,
                 w1, w2, pos,
                 time = None,
                 length = 10 * np.pi,
                 phase = None,
                 exp = True,
                 rad = 0.02,
                 res = 50,
                 alpha = 1.,
                 ):
        super().__init__()
        if exp:
            length = length * 2

        nV = 10
        nV += int(LA.norm(w1 - w2) * length * res)

        self.pos = normalize(pos) * r
        self.rot1 = Rotator(w1, phase = phase)
        self.rot2 = Rotator(w2, phase = phase)

        f1, f2 = f1f2(nV)

        c1 = np.asarray([*c, alpha])
        c2 = np.asarray([*c, 0.])
        if exp:
            f = np.exp(-f2 * np.log(510))
        else:
            f = f1
        self.c = np.outer(f, c1) + np.outer(1-f, c2)

        self.dt = f2 * length
        self.rad = rad
        self.res = res

    def draw(self, time):
        # need to add "time align" to rotator
        v = np.asarray([
            self.rot1(
                self.rot2(
                    self.pos, time - dt),
                dt)
            for dt in self.dt])
        self.add_actor(tube(v, self.rad, self.c, self.res, capping = True))

class Track(SceneItem):
    # currently this is static (for BH scene)
    def __init__(self, r, c,
                 t = None,
                 exp = True,
                 rad = 0.02,
                 res = 50,
                 alpha = 1.,
                 alpha_final = 0.,
                 ):
        super().__init__()
        self.rad = rad
        self.res = res

        nV = len(r)

        f1, f2 = f1f2(nV)

        c1 = np.asarray([*c, alpha])
        c2 = np.asarray([*c, alpha_final])
        if exp:
            f = np.exp(-f2 * np.log(510))
        else:
            f = f1
        self.c = np.outer(f, c1) + np.outer(1-f, c2)

        self.v = r

    def draw_static(self):
        self.add_actor(
            tube(self.v, self.rad, self.c, self.res, capping = True),
            static = True)

class Pin(SceneItem):
    def __init__(
            self, r1, r2, vec,
            c = [0.7, 0.7, 0.7],
            size = 0.01,
            w = None,
            res = 20,
            phase = None,
            ):
        super().__init__()
        self.res = res
        self.height = (r2 - r1)
        self.c = np.asarray(c)
        self.radius = size

        r = (r2 + r1) * 0.5
        self.u = r * normalize(vec)
        if w is not None:
            self.rot = rot = Rotator(w, phase = phase)
        else:
            self.rot = NoRotator()

    def draw(self, time):
        u = self.rot(self.u, time = time)
        self.add_actor(cylinder(
            u, self.radius, self.height, u,
            self.c, res = self.res, capping = True))

class Marker(SceneItem):
    def __init__(
            self, r, vec,
            c = [1., 1., 0],
            size = 0.05,
            w = None,
            phase = None,
            res = None,
            alpha = None,
            fixedtime = None,
            ):
        super().__init__()
        self.res = res
        self.c = np.asarray(c)
        self.alpha = alpha
        if res is None:
            res = 20
        self.res = res
        self.radius = size
        if r is not None:
            self.pos = r * normalize(vec)
        else:
            self.pos = np.asarray(vec)

        self.fixedtime = fixedtime
        self.static = fixedtime is not None
        if w is not None:
            w = np.asarray(w)
            self.rot = Rotator(w)
        else:
            self.rot = NoRotator()
            self.static = True

    def draw(self, time = None):
        static = time is None
        if static != self.static:
            return
        if self.fixedtime is not None:
            time = self.fixedtime
        v = self.rot(self.pos, time)
        self.add_actor(
            sphere(
                v, self.radius, self.c,
                alpha = self.alpha,
                res = self.res),
            static)

    draw_static = draw


class ShearVector(SceneItem):
    def __init__(
            self, vec, r1, r2, w1, w2,
            c = [1., 1., 0.],
            cm = [1., 0.5, 0.2],
            size = 0.25,
            res = 20,
            mag = False,
            torque = False,
            alpha = None,
            phase = None,
            linear = False,
            align = None,
            fixedtime = None,
            arc = False,
            ):
        super().__init__()
        self.res = res
        self.alpha = alpha
        self.c = np.asarray(c)
        self.cm = np.asarray(cm)
        self.torque = torque
        self.mag = mag
        self.size = size
        self.arc = arc
        w1 = np.asarray(w1)
        w2 = np.asarray(w2)
        if linear:
            w = 0.5 * (w1 + w2)
        else:
            w = rotscale(w1, w2, 0.5, align = align)
        r = 0.5 * (r1 + r2)
        dr = r2 - r1
        self.dwdr = (w2 - w1) / dr
        self.rot = Rotator(w, phase = phase)
        self.v = normalize(vec) * r
        self.fixedtime = fixedtime

    def draw_static(self):
        if self.fixedtime is not None:
            self.mydraw(self.fixedtime, static = True)

    def draw(self, time):
        if self.fixedtime is None:
            self.mydraw(time)

    def mydraw(self, time, static = False):
        v = self.rot(self.v, time)
        if self.torque:
            vl2 = np.tensordot(v, v, axes = (-1,-1))
            sh = (np.tensordot(self.dwdr, v, axes = (-1,-1)) / vl2) * v
            sh *= self.size
        else:
            sh = np.cross(self.dwdr, v, axis = -1) * self.size
        sl = LA.norm(sh)
        c = self.c
        if self.mag:
            size = 0.1 * sl
            if self.torque and np.tensordot(v, sh, axes = (-1,-1)) < 0:
                c = self.cm
            # TODO - add color circle for shear direction
            a = sphere(
                v, size, c,
                alpha = self.alpha,
                res = self.res,
                )
        elif self.arc and not self.torque:
            # theta = sl / LA.norm(v)
            # w = normalize(np.cross(v, sh, axis = -1))
            w = self.dwdr
            theta = sl * LA.norm(w) / np.maximum(LA.norm(np.cross(v, w)), 1.e-10)
            theta0 = -0.5 * theta

            rot = Rot(v, c, w,
                      alpha = self.alpha,
                      res = self.res,
                      hat = 'curved', #'curved',
                      arrow = True,
                      theta0 = theta0,
                      theta = theta,
                      fixedtime = 0,
                      )
            rot.draw_static()
            self.replace_actors(rot.get_actors(), static)
            a = None
        else:
            v -= 0.5 * sh
            size = sl
            a = arrow(
                v, sh, size, c,
                alpha = self.alpha,
                res = self.res,
                )
        if a is not None:
            self.add_actor(a, static)


class Sphere(SceneItem):
    def __init__(self, r, c, alpha = None, pos = [0., 0., 0.], res = 100):
        """
        Create a Sphere
        """
        super().__init__()

        self.pos = np.asarray(pos)
        self.res = res
        self.radius = r
        self.alpha = alpha
        self.c = c

    def draw_static(self):
        self.add_actor(
            sphere(
                self.pos, self.radius, self.c,
                alpha = self.alpha,
                res = self.res,
                ),
            static = True)

class RotationEllipsoid(SceneItem):
    def __init__(self, axis, r, c, alpha = None, pos = [0., 0., 0.], res = 100):
        """
        Create a Sphere
        """
        super().__init__()

        self.axis = np.asarray(axis)
        self.pos = np.asarray(pos)
        self.res = res
        self.radius = r
        self.alpha = alpha
        self.c = c

    def draw_static(self):
        self.add_actor(
            rotation_ellipsoid(
                self.pos, self.axis, self.radius, self.c,
                alpha = self.alpha,
                res = self.res,
                ),
            static = True)


class RotCircArrow(SceneItem):
    """
    create a circular rotation arrow
    """
    def __init__(self, r, w, rad, pos, c,
                 size = 0.02,
                 **kwargs):
        super().__init__()
        w = np.asarray(w)
        pos = np.array(pos)

        wn = LA.norm(w)
        pn = LA.norm(pos)
        p0 = w * r / wn
        px = pos - w * np.tensordot(w, pos, axes = (-1,-1)) / wn**2
        p1 = px / LA.norm(px) * rad
        p = p0 + p1

        kwargs.setdefault('res', 100)
        kwargs.setdefault('theta', 2 * np.pi * 7/8)
        kwargs.setdefault('rad', size)

        self.add_item(Rot(p, c, w, **kwargs))


class Arrow(SceneItem):
    """
    draw an arrow
    """
    def __init__(self, r, pos, vec, c,
                 alpha = None,
                 size = None,
                 w = None,
                 res = None,
                 phase = None,
                 rot_mode = 'w',
                 fixedtime = None,
                 ):
        super().__init__()

        pos = np.asarray(pos)
        vec = np.asarray(vec)

        pos = r * normalize(pos)

        if res is None:
            res = 100
        self.res = res
        self.c = np.asarray(c)
        self.alpha = alpha

        if size is None:
            size = LA.norm(vec)
        self.size = size

        self.fixedtime = fixedtime
        self.static = fixedtime is not None
        if w is None:
            self.static = True
            self.rotu = NoRotator()
            self.rotv = NoRotator()
            self.rotp = NoRotator()
        else:
            self.rotv = Rotator(w, phase = phase)
            if rot_mode in ('w', 'project', None):
                self.rotu = Rotator(w, phase = phase)
            if rot_mode in ('project', ):
                pn = LA.norm(pos)
                wn = LA.norm(w)
                p = np.tensordot(pos, w, axes = (-1,-1)) * pos / pn**2
                self.rotp = Rotator(-p, phase = phase)
            else:
                self.rotp = NoRotator()
            if rot_mode in ('inertial', ):
                self.rotu = NoRotator()

        self.v = pos
        self.u = vec

    def draw(self, time = None):
        static = time is None
        if static != self.static:
            return
        if self.fixedtime is not None:
            time = self.fixedtime
        v = self.rotv(self.v, time)
        u = self.rotu(self.rotp(self.u, time), time)
        self.add_actor(arrow(
            v, u, self.size, self.c,
            alpha = self.alpha,
            res = self.res), static = static)

    draw_static = draw


class Axes(SceneItem):
    """
    draw XYZ axes
    """
    def __init__(self, r, pos,
                 rot = None,
                 c = None,
                 alpha = None,
                 w = None,
                 size = 1.,
                 res = None,
                 org = True,
                 ro = None,
                 co = None,
                 ao = None,
                 phase = None,
                 rot_mode = None,
                 fixedtime = None,
                 ):
        super().__init__()
        if c is None:
            c = np.asarray([
                Colors.red,
                Colors.green,
                Colors.blue,
                ])
        if alpha is None:
            alpha = 1.
        if np.shape(alpha) == ():
            alpha = np.tile(np.asarray(alpha, dtype = np.float), (3,))

        if r is not None:
            pos = r * normalize(pos)

        if rot is None:
            rot = rotate2(pos)
        xyz = basis.copy()
        xyz = rotate(xyz, rot)
        for vx, cx, ax in zip(xyz, c, alpha):
            self.add_item(Arrow(
                r, pos, vx, cx,
                alpha = ax,
                size = size,
                w = w,
                res = res,
                phase = phase,
                rot_mode = rot_mode,
                fixedtime = fixedtime,
                ))
        if org:
            if co is None:
                co = Colors.white
            if ao is None:
                ao = np.average(alpha)
            if ro is None:
                ro = 0.06 * size
            self.add_item(Marker(
                r, pos,
                c = co,
                size = ro,
                w = w,
                alpha = ao,
                phase = phase,
                res = None,
                fixedtime = fixedtime,
                ))


class RotArrow(SceneItem):
    """
    create a rotation arrow
    """
    def __init__(self, r, c, w,
                 alpha = 1.,
                 res = 100,
                 rot = False,
                 rotloc = 0.5,
                 rotsize = 0.01,
                 rotrad = 0.2,
                 rothat = 'curved',
                 rotpos = None,
                 pos = [0., 0., 0.],
                 eps = 1.e-99,
                 size = 1.,
                 ):
        super().__init__()
        w = np.asarray(w)
        wn = LA.norm(w)
        self.pos = np.asarray(pos) + w * r / np.maximum(wn, eps)
        self.size = wn * size
        self.res = res
        self.c = c
        self.alpha = alpha

        # TODO - add flat type
        if rot in (True, ):
            if rotpos is None:
                rotpos = np.asarray([0., 0., 1.])
                if LA.norm(np.cross(rotpos, w)) == 0:
                    rotpos = np.asarray([0., 1., 0.])
            self.add_item(
                RotCircArrow(
                    r + wn * rotloc,
                    w,
                    wn * rotrad,
                    rotpos,
                    c,
                    size = wn * rotsize,
                    hat = rothat,
                    )
                )

    # TODO - allow time-dependent vector

    def draw_static(self):
        self.add_actor(arrow(
            self.pos, self.pos, self.size, self.c,
            alpha = self.alpha,
            res = self.res), static = True)


def field(w, npoint = 3, symmetry = 4, order = None, **kwargs):
        """
        spacing in powers of 2

        total points on equator is 2**npoints * symmetry

        and polar circle 2 * int(poins on equator / 2)

        """
        tol = np.sqrt((1 + np.sqrt(5))/2)

        points = []
        angles = []
        phases = []

        phase = kwargs.get('phase', None)
        phasep = kwargs.get('phasep', phase)
        if phasep is None:
            phasep = phase
        if order is None:
            if phasep is not None:
                order = 0
            else:
                order = 1
        if phasep is None:
            phasep = 0
        phaset = kwargs.get('phaset', None)
        if phaset is None:
            phaset = 0

        assert order == round(order)
        mtrig = abs(order)
        mphase = abs(order - 1)

        phasel = kwargs.get('phasel', None)
        if phasel is None:
            phasel = 0
        phasem = kwargs.get('phasem', None)
        if phasem is None:
            phasem = np.pi

        n = int(2**npoint * symmetry * 0.5)
        t = np.pi * (np.arange(n + 1) / n - 0.5)
        c0 = np.cos(t)
        s0 = np.sin(t)
        for tx,cx,sx in zip(t, c0, s0):
            mpoint = int(np.log(cx * tol) / np.log(2) + npoint)
            m = np.maximum(int(2**mpoint * symmetry), 1)
            p = np.arange(m) * 2 * np.pi / m + tx * phasem / (2 * np.pi)
            pp = mtrig * p - phasep
            s1 = np.sin(pp)
            c1 = np.cos(pp)
            px = mphase * p + phaset + tx * phasel / (2 * np.pi)
            z = cx * c1
            y = cx * s1
            x = sx * np.tile(1, c1.shape)
            for x1,y1,z1,p1 in zip(x, y, z, p):
                points.append((x1, y1, z1))
                angles.append((tx, p1))
            phases.extend(px.tolist())
        points = Rotator(rotate2(w))(np.asarray(points))
        angles = np.asarray(angles)
        return points, angles, phases


class FieldBase(object):
    _npoint = 3
    _symmetry = 4
    _phase = None
    _phasep = None
    _phaset = None
    _phasel = None
    _phasem = None
    _order = None
    def __init__(self, w, **kwargs):
        kwargs.setdefault('order', self._order)
        kwargs.setdefault('phase', self._phase)
        kwargs.setdefault('phaset', self._phaset)
        kwargs.setdefault('phasep', self._phasep)
        kwargs.setdefault('phasel', self._phasel)
        kwargs.setdefault('phasem', self._phasem)
        kwargs.setdefault('npoint', self._npoint)
        kwargs.setdefault('symmetry', self._symmetry)
        self.points, self.angles, self.phases = field(w, **kwargs)


class SceneItemField(SceneItem, FieldBase):
    def __init__(self, *args, **kwargs):
        SceneItem.__init__(self)
        FieldBase.__init__(self, *args, **kwargs)


class MarkerField(SceneItemField):
    def __init__(self, r, w, **kwargs):
        super().__init__(w, **kwargs)
        self.add_items(tuple(
            Marker(r, vec, w = w, **kwargs)
            for vec in self.points))

class SceneItemDualField(SceneItemField):
    _align = None
    _linear = False

    def __init__(self, w1, w2, **kwargs):
        linear = kwargs.setdefault('linear', self._linear)
        align = kwargs.setdefault('align', self._align)
        w1 = np.asarray(w1)
        w2 = np.asarray(w2)
        w = kwargs.get('w', None)
        if w is None:
            if linear:
                w = 0.5 * (w1 + w2)
            else:
                w = rotscale(w1, w2, 0.5, align = align)
        super().__init__(w, **kwargs)

class TraceField(SceneItemDualField):
    # TODO - array should be created in Trace to use single rotation
    #        matrix operation
    def __init__(self, r1, r2, w1, w2, c1, c2, w = None, **kwargs):
        super().__init__(w1, w2, **kwargs)
        kwargs.pop('phase', None)
        self.add_items(tuple(
            Trace(r1, r2, c1, c2, vec,
                  w1 = w1,
                  w2 = w2,
                  phase = p,
                  **kwargs)
            for vec, p in zip(self.points, self.phases)))


class ShearVectorField(SceneItemDualField):
    # TODO - array should be created in Trace to use single rotation
    #        matrix operation
    _fixedtime = None

    def __init__(self, r1, r2, w1, w2, c, **kwargs):
        super().__init__(w1, w2, **kwargs)
        kwargs.setdefault('fixedtime', self._fixedtime)
        self.add_items(tuple(
            ShearVector(vec, r1, r2, w1, w2, c, **kwargs)
            for vec in self.points))
