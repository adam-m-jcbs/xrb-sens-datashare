"""
Definition of Scenes
"""

import os.path

import numpy as np

import scipy.spatial.transform
import scipy.interpolate

from .items import *
from .base import Scene
from rotation import e0, rotate, rotate2, rotate3

class ShellScene(Scene):
    _name = "Shells"
    _r1 = 0.9
    _r2 = 1.1
    _res = 120
    _c1 = np.asarray([0., 1., 0.5])
    _c2 = np.asarray([0., 0.5, 1.0])
    _w1 = np.asarray([-0.2, 1., 0.])
    _w2 = np.asarray([0.2, 1., 0.])*0.8
    _alpha_s1 = 0.1
    _alpha_s2 = 0.1
    _alpha_a1 = 1
    _alpha_a2 = 1
    _rot = True

    def setup_parameters(self, *args, **kwargs):
        super().setup_parameters(self, *args, **kwargs)
        self.resolve_defaults(kwargs)

    def build_model(self, time = None):
        super().build_model()
        self.build_shells(time)

    def build_shells(self, time):
        # spheres
        self.add(Sphere(self.r1, self.c1, self.alpha_s1,
                        res = self.res))
        self.add(Sphere(self.r2, self.c2, self.alpha_s2,
                        res = self.res))

        # arrows
        self.add(RotArrow(self.r1, self.c1, self.w1,
                          alpha = self.alpha_a1,
                          res = self.res,
                          rot = self.rot))
        self.add(RotArrow(self.r2, self.c2, self.w2,
                          alpha = self.alpha_a2,
                          res = self.res,
                          rot = self.rot))


class TrailScene(ShellScene):
    _vt = np.asarray([0., -0.3, 1.])
    _ct = np.asarray([1., 1., 0.])
    _cm = np.asarray([1., 0., 0.])
    _cp = np.asarray([1., 1., 1.])

    _w1 = np.asarray([-0.2, 1., 0.])
    _w2 = np.asarray([ 0.2, 1., 0.]) * 0.95

    def setup_parameters(self, *args, **kwargs):
        super().setup_parameters(*args, **kwargs)
        self.resolve_defaults(kwargs)

    def build_model(self, time = None):
        super().build_model()

        # pin with trail
        self.add(Trail (self.r1, self.ct, self.w1, self.w2, self.vt))
        self.add(Marker(self.r1, self.vt, w = self.w2, c = self.cm))
        self.add(Marker(self.r2, self.vt, w = self.w2, c = self.c2))
        self.add(Pin   (self.r1, self.r2, self.vt, w = self.w2, c = self.cp))

        # cross hair compose
        self.add(Coord(self.r2, self.c2, w = self.w2, pos = self.vt, lat = True , alpha=0.2, rad=0.015))
        self.add(Coord(self.r2, self.c2, w = self.w2, pos = self.vt, lat = False, alpha=0.2, rad=0.015))

        # coordinate grid
        self.add(CoordGrid(self.r1, self.c1, w = self.w1, pos = self.vt, nphi= 6))

class BasicScene(ShellScene):
    pass

class FieldLineScene(ShellScene):
    _vt = np.array([1., -0.3, 0.])

    _w1 = np.asarray([-0.2, 1., 0.])
    _w2 = np.asarray([0.2, 1., 0.])*0.8

    def setup_parameters(self, *args, **kwargs):
        super().setup_parameters(self, *args, **kwargs)
        self.resolve_defaults(kwargs)
        if not isinstance(self.vt, tuple):
            self.vt = self.vt,

    def build_model(self, time = None):
        super().build_model()

        for v in self.vt:
            self.add(Trace(
                self.r1, self.r2, self.c1, self.c2, v,
                w1 = self.w1,
                w2 = self.w2))
            self.add(Marker(self.r1, v, w = self.w1, c = self.c1))
            self.add(Marker(self.r2, v, w = self.w2, c = self.c2))


class MarkerFieldScene(ShellScene):
    def build_model(self, time = None):
        super().build_model()
        self.add(MarkerField(self.r1, self.w1, c = self.c1))
        self.add(MarkerField(self.r2, self.w2, c = self.c2))

class ShearFieldScene(ShellScene):
    _rot = False

    _w1 = np.asarray([-0.2, 1., 0.])
    _w2 = np.asarray([+0.2, 1., 0.])

    _c = np.asarray([1.,1.,0.])
    _cm = np.asarray([1., 0.5, 0.2])
    _mag = False
    _torque = False
    _size = 0.25
    _resv = 20
    _arc = False
    _alpha = None
    _fixedtime = None
    _align = None

    def setup_parameters(self, *args, **kwargs):
        super().setup_parameters(self, *args, **kwargs)
        self.resolve_defaults(kwargs)

    def build_model(self, time = None):
        super().build_model()
        self.build_field(time)

    def build_field(self, time = None):
        self.add(ShearVectorField(
            self.r1, self.r2, self.w1, self.w2, self.c,
            cm = self.cm,
            size = self.size,
            mag = self.mag,
            torque = self.torque,
            alpha = self.alpha,
            res = self.resv,
            align = self.align,
            fixedtime = self.fixedtime,
            arc = self.arc,
            ))

class ShearVectorScene(ShellScene):
    _rot = False

    _vt = np.array([0., 0., 1.])

    _c = np.asarray([ 1., 1., 0.])
    _cm = np.asarray([ 1., 0.5, 0.2])
    _mag = False
    _arc = False
    _torque = False
    _size = 0.25
    _resv = 20
    _alpha = None

    def setup_parameters(self, *args, **kwargs):
        super().setup_parameters(self, *args, **kwargs)
        self.resolve_defaults(kwargs)
        if not isinstance(self.vt, tuple):
            self.vt = self.vt,

    def build_model(self, time = None):
        super().build_model()

        for v in self.vt:
            self.add(ShearVector(
                v, self.r1, self.r2, self.w1, self.w2, self.c,
                cm = self.cm,
                size = self.size,
                mag = self.mag,
                torque = self.torque,
                alpha = self.alpha,
                res = self.resv,
                arc = self.arc,
                ))

class TraceFieldScene(ShellScene):
    _rot = False

    _w1 = np.asarray([-0.2, 1., 0.])
    _w2 = np.asarray([+0.2, 1., 0.])

    _phase = None
    _phaset = None
    _phasep = None
    _phasel = None
    _phasem = None
    _resv = 10
    _rad = 0.02
    _linear = False
    _order = None

    def setup_parameters(self, *args, **kwargs):
        super().setup_parameters(self, *args, **kwargs)
        self.resolve_defaults(kwargs)

    def build_model(self, time = None):
        super().build_model()
        self.add(TraceField(
            self.r1, self.r2, self.w1, self.w2,
            c1 = self.c1,
            c2 = self.c2,
            phase = self.phase,
            phaset = self.phaset,
            phasep = self.phasep,
            phasel = self.phasel,
            phasem = self.phasem,
            order = self.order,
            linear = self.linear,
            rad = self.rad,
            res = self.resv,
            ))


class DivergentShearFieldScene(ShearFieldScene):
    _rot = False

    _w10 = np.asarray([0., 1., 0.])
    _w20 = np.asarray([0., 1., 0.]) * 0.8
    _w =  np.asarray([0., 0., 1.]) * 0.9 # * 2? *0.5?

    _size = 0.05
    _align = _w
    _fixedtime = 0

    def setup_parameters(self, *args, **kwargs):
        super().setup_parameters(self, *args, **kwargs)
        self.resolve_defaults(kwargs)

        self.rotator1 = Rotator(+self.w)
        self.rotator2 = Rotator(-self.w)

    def update_items(self, time):
        self.clear_items()
        self.build_model(time)

    def build_model(self, time = 0):
        self.w1 = self.rotator1(self.w10, time)
        self.w2 = self.rotator2(self.w20, time)
        self.build_shells(time)
        self.build_field(time)


class OscillatingShearFieldScene(ShearFieldScene):
    _rot = False

    # w is used as token and for cam offset and relative period only
    _w = np.asarray([0., 1., 0.])

    _w10 = _w
    _w20 = _w
    _m10 = 0.75
    _m20 = _m10
    _a10 = 0.5
    _a20 = _a10
    _p10 = 0
    _p20 = np.pi
    _period = 2 * np.pi / 0.75

    _size = 0.1
    _fixedtime = None
    _alpha_a1 = 0.25
    _alpha_a2 = 0.25

    def setup_parameters(self, *args, **kwargs):
        super().setup_parameters(self, *args, **kwargs)
        self.resolve_defaults(kwargs)

        self.pbar = 2 * np.pi / self.period

    def update_items(self, time):
        self.clear_items()
        self.build_model(time)

    def build_model(self, time = 0):
        self.w1 = self.w10 * (self.m10 + self.a10 * np.sin(time * self.pbar + self.p10))
        self.w2 = self.w20 * (self.m20 + self.a20 * np.sin(time * self.pbar + self.p20))
        self.w = rotscale(self.w1, self.w2, 0.5)
        self.build_shells(time)
        self.build_field(time)


class RotationScene(Scene):
    _name = "Rotations"

    _r1 = 1
    _res = 120
    _c1 = np.asarray([0.5, 0.5, 0.5])
    _alpha_s1 = 0.2

    _size = 0.5

    _pos1 = np.asarray([1., 0., 0.])
    _alpha_a1 = 0.1
    _alpha_a2 = 1.0
    _alpha_te = 0.1
    _alpha_tl = 1.0
    _ct = np.asarray([1., 1., 0.])

    _w = np.asarray([0., 1., 0.])
    _cw = _ct
    _alpha_w = 1.
    _size_w = 0.5

    _pos2 = np.asarray([1., 1., 1.])

    _rot_mode = None
    _track_mode = 'trail'

    _graph_mode = 'revolve'

    _trail_length = 0.75

    def setup_parameters(self, *args, **kwargs):
        super().setup_parameters(*args, **kwargs)
        self.resolve_defaults(kwargs)

        if self.graph_mode == 'pos2':
            self.w = rotate2(self.pos2, self.pos1)
            if not hasattr(self, 'time'):
                self.time = 1
        elif self.graph_mode == 'revolve':
            if not hasattr(self, 'time'):
                self.time = (2 * np.pi) / LA.norm(self.w)
        else:
            raise Exception(f'Unknown Graph mode "{self.graph_mode}".')

    def build_model(self, time = None):
        super().build_model()

        self.add(Sphere(
            self.r1, self.c1, self.alpha_s1,
            res = self.res,
            ))
        self.add(Axes(
            self.r1, self.pos1,
            alpha = self.alpha_a1,
            size = self.size,
            ))
        if self.track_mode == 'trace':
            self.add(Trace(
                self.r1, self.r1, self.ct, self.ct, self.pos1,
                w1 = e0,
                w2 = self.w,
                linear = True,
                alpha = self.alpha_te,
                ))
        elif self.track_mode == 'trail':
            tl = (self.trail_length
                  * LA.norm(np.cross(self.pos1, self.w, axis = -1))
                  / (LA.norm(self.w)**2 * LA.norm(self.pos1))
                  * 2 * np.pi)
            self.add(Trail(
                self.r1, self.ct, e0, self.w, self.pos1,
                alpha = self.alpha_tl,
                length = tl,
                ))
        self.add(Axes(
            self.r1, self.pos1,
            w = self.w,
            alpha = self.alpha_a2,
            size = self.size,
            rot_mode = self.rot_mode,
            ))
        self.add(RotArrow(
            self.r1, self.cw, self.w,
            size = self.size_w,
            alpha = self.alpha_w,
            ))


class RotVecScene(Scene):
    """
    changed parameters
    """
    _name = "RotVec"
    _camfocus = np.asarray([0., 0., 0.])

    # sphere
    _r1 = 1
    _res = 120
    _c1 = np.asarray([0.5, 0.5, 0.5])
    _alpha_s1 = 0.2

    # axes
    _pos1 = np.asarray([ 1., 1., 1.])
    _pos2 = np.asarray([-1., 1., 1.])
    _alpha_a1 = 0.2
    _alpha_a2 = 1.0
    _size = 0.5

    #track
    _alpha_t = 0.2
    _ct = np.asarray([1., 1., 0.])

    # rot vector
    _cw = np.asarray([1., 1., 0.])
    _alpha_w = 1.
    _size_w = 0.5

    # shadow
    _alpha_ws = 0.2
    _alpha_ts = 0.1
    _alpha_a2s = 0.1

    # mode settings
    _time = 1.
    _circ_mode = 'rot'
    _rot_mode = None
    _vec_mode = 'limit'

    def setup_parameters(self, *args, **kwargs):
        super().setup_parameters(*args, **kwargs)
        self.resolve_defaults(kwargs)

        self.w0 = rotate2(self.pos2, self.pos1)
        self.wr = normalize(self.pos2 - self.pos1)

    def update_items(self, time):
        self.clear_items()
        self.build_model(time)

    def build_model(self, time = 0):
        super().build_model()

        phase = time * 2 * np.pi / self.time

        if self.circ_mode == 'twist':
            w1 = normalize(self.pos1) * phase
            w = wchain(w1, self.w0)
        elif self.circ_mode == 'rot_num':
            w = rotate(self.w0, self.wr, phase = phase)
            n0 = LA.norm(self.w0)
            n1 = np.pi * 2 - n0
            w /= n0

            # NR
            # good guess
            lx = n0 + (1 - np.cos(phase)) * (n1 - n0) / 2
            eps = 1.e-5
            fac = 1.
            while True:
                d = LA.norm(rotate2(rotate(self.pos1, w * lx),  self.pos2))
                if d < 1.e-5:
                    break
                d1 = LA.norm(rotate2(rotate(self.pos1, w * (lx + eps)),  self.pos2))
                lx -= d * eps / (d1 - d) * fac
            w *= lx
        elif self.circ_mode == 'rot':
            w = rotate(self.w0, self.wr, phase = phase)
            w = rotate3(w, self.pos1, self.pos2)
        else:
            raise Exception(f'Unsupported circ_mode "{self.circ_mode}".')

        if self.vec_mode in ('limit', 'shadow', 'dual',):
            wn = LA.norm(w)
            ws = w * (1 - 2 * np.pi / wn)
            if self.vec_mode in ('limit', 'shadow', ):
                if wn > np.pi:
                    w, ws = ws, w
        if self.vec_mode in ('project', ):
            ws = w
            rot_mode_s = 'project'
            assert self.rot_mode != 'project', 'would be just overplotting'
        else:
            rot_mode_s = self.rot_mode

        self.add(Sphere(
            self.r1, self.c1, self.alpha_s1,
            res = self.res,
            ))
        self.add(Axes(
            self.r1, self.pos1,
            alpha = self.alpha_a1,
            size = self.size,
            ))
        self.add(Trace(
            self.r1, self.r1, self.ct, self.ct, self.pos1,
            w1 = e0,
            w2 = w,
            linear = True,
            alpha = self.alpha_t,
            fixedtime = 1.,
            ))
        self.add(Axes(
            self.r1, self.pos1,
            w = w,
            alpha = self.alpha_a2,
            size = self.size,
            rot_mode = self.rot_mode,
            fixedtime = 1.,
            ))
        self.add(RotArrow(
            self.r1, self.cw, w,
            size = self.size_w,
            alpha = self.alpha_w,
            ))
        if self.vec_mode in ('shadow', 'dual', ):
            self.add(Trace(
                self.r1, self.r1, self.ct, self.ct, self.pos1,
                w1 = e0,
                w2 = ws,
                linear = True,
                alpha = self.alpha_ts,
                fixedtime = 1.,
                ))
            self.add(RotArrow(
                self.r1, self.cw, ws,
                size = self.size_w,
                alpha = self.alpha_ws,
                ))
        if self.vec_mode in ('shadow', 'dual', 'project', ):
            if rot_mode_s == 'project':
                self.add(Axes(
                    self.r1, self.pos1,
                    w = ws,
                    alpha = self.alpha_a2s,
                    size = self.size,
                    rot_mode = rot_mode_s,
                    fixedtime = 1.,
                    ))


class BlackHoleSpinScene(Scene):
    """
    Show time-dependent spin of black hole.
    """
    CLIGHT = 29979245800
    XMSUN = 1.9891e33
    GRAV = 6.67259e-8

    _name = "Black Hole Spin"
    _camfocus = np.asarray([0., 0., 0.])
    _campos = np.asarray([5., 0., 5.])

    _camalign = None # None, 'final' | 'initial'

    # data
    _path = os.path.split(__file__)[0]
    _filename = 'z40.txt'
    _file_format = 2
    _time_offset = 0.

    # BH sphere
    _cb = np.asarray([0.5, 0.5, 0.5])
    _ab = 1.
    _scale_r = -1.
    _res = 120

    # spin
    _ca = np.asarray([1., 1., 0.])
    _aa = 1.
    _size_a = 1.
    _scale_a = 1.
    # a_norm = cJ/GM^2
    # spin vec size = rg * a_norm

    # ergosphere
    _ce = np.asarray([0.5, 0.5, 0.75])
    _ae = 0.1
    _min_am = 0.0001

    # time scaling
    _time_mode = 'frame' # frame | time | logtime
    _delay = 1 / 60
    _interp_mode = 'rot' # rot | spline

    # moving BH
    _move = True
    _track = True
    _scale_p = 100 # scale of position in units of radius scale
    _at = 0.5
    _ct = np.asarray([1., 1., 0.])
    _rt = 0.02
    _et = True

    def setup_parameters(self, *args, **kwargs):
        super().setup_parameters(self, *args, **kwargs)
        self.resolve_defaults(kwargs)

        # load and store raw model data
        filename = self.filename
        if self.path is not None:
            filename = os.path.join(self.path, filename)
        filename = os.path.expandvars(filename)
        filename = os.path.expanduser(filename)
        data = np.loadtxt(filename, comments = ('#', ))
        print(f'[BH] Loaded {len(data)} time sclices.')
        self.bh_time = data[:, 0] + self.time_offset
        self.bh_mass = data[:, 1]
        self.bh_spin = data[:, 2:5]
        if self.file_format == 2 and self._move:
            self.bh_mom = data[:, 5:8]
        else:
            self.bh_mom = np.zeros_like(self.bh_spin)

        if self.scale_r == 0:
            self.scale_r = -self.bh_mass[-1]
        if self.scale_r < 0:
            self.scale_r = -self.scale_r * 2 * self.GRAV * self.XMSUN / self.CLIGHT**2

        self.scale_p = self.scale_p * self.scale_r

        self.bh_velo = self.bh_mom / self.bh_mass[:, np.newaxis]
        self.bh_pos = np.empty_like(self.bh_velo)
        t = self.bh_time
        # self.bh_poss = []
        for i, y in enumerate(self.bh_velo.transpose()):
            spl = scipy.interpolate.splrep(t, y)
            spla = scipy.interpolate.splantider(spl)
            # self.bh_poss.append(spla)
            self.bh_pos[:, i] = scipy.interpolate.splev(t, spla)

        if self.time_mode in ('time', 'logtime'):
            t = self.bh_time
            if self.time_mode == 'logtime':
                t = np.log(t)
            a_norm = self.bh_spin * self.CLIGHT / (self.GRAV * self.bh_mass**2)[:, np.newaxis]
            if self.interp_mode == 'rot':
                jrot = scipy.spatial.transform.Rotation.from_rotvec(a_norm)
                self.interpj = scipy.spatial.transform.RotationSpline(t, jrot)
            elif self.interp_mode == 'spline':
                self.interpj = scipy.interpolate.interp1d(t, a_norm.transpose(), kind='cubic')
            else:
                raise Exception(f'Unknown interp_mode "{interp_mode}".')
            self.interpm = scipy.interpolate.interp1d(t, self.bh_mass, kind='cubic')
            self.interpp = scipy.interpolate.interp1d(t, self.bh_pos.transpose(), kind='cubic')
        else:
            assert self.time_mode == 'frame'

        if self.camalign is not None:
            if self.camalign == 'final':
                bh_dir = self.bh_pos[-1]
            elif self.camalign == 'initial':
                bh_dir = self.bh_pos[1]
            bh_dir = bh_dir / LA.norm(bh_dir)
            self.campos = -bh_dir * LA.norm(self.campos)
            print('[BH] Cam pos set to ', self.campos)
            self.camthick = -LA.norm(self.bh_pos[-1]) / self.scale_p

    def update_items(self, time):
        self.clear_items()
        self.build_model(time)

    def interpolate_frame(self, time):
        """
        Just return nearest frame.
        """
        i = int(np.round(time))
        n = self.bh_time.shape[0]
        i = max(0, min(n-1, i))
        t = self.bh_time[i]
        m = self.bh_mass[i]
        j = self.bh_spin[i]
        p = self.bh_pos[i]
        return t, m, j, p

    def interpolate_rot_time(self, time):
        """
        interpolate time using scipy rotation interpolation.
        """
        t = min(self.bh_time[-1], max(self.bh_time[0], time))
        if self.time_mode == 'logtime':
            x = np.log(t)
        else:
            x = t
        if self.interp_mode == 'rot':
            a_norm = self.interpj(x).as_rotvec()
        elif self.interp_mode == 'spline':
            a_norm = self.interpj(x)
        m = self.interpm(x)
        p = self.interpp(x)
        j = a_norm * m**2 * self.GRAV / self.CLIGHT
        return t, m, j, p

    def bh_par(self, time):
        if self.time_mode == 'frame':
            t, m, j, p = self.interpolate_frame(time)
        elif self.time_mode in ('time', 'logtime', ):
            t, m, j, p = self.interpolate_rot_time(time)
        else:
            raise AttributeError(f'Unknown time_mode "{self.time_mode}".')

        j *= self.scale_a
        a = LA.norm(j) / m
        r = (
            (self.GRAV * m +
             np.sqrt((self.GRAV * m)**2
                     - (a * self.CLIGHT)**2))
            / self.CLIGHT**2
            )
        jmax = self.GRAV * m**2 / self.CLIGHT
        rg = self.GRAV * m / self.CLIGHT**2
        ra = rg * (j / jmax)
        # print(f't={t}, a={LA.norm(j/jmax)}')
        # print(t, r, 2*rg)
        # mirr = np.sqrt((np.sqrt(m**4 - (LA.norm(j)*self.CLIGHT/self.GRAV)**2) + m**2)*0.5)
        # print(f't={t}, m_irr={mirr}')
        return t, r, ra, rg, p

    def build_model(self, time = None):
        super().build_model()

        if time is None:
            time  = self.bh_time[0]

        t, r, ra, rg, pos = self.bh_par(time)

        r = r / self.scale_r
        ra = ra / self.scale_r
        pos = pos / self.scale_p

        # event horizon
        self.add(Sphere(
            r, self.cb, self.ab,
            res = self.res,
            pos = pos,
            ))
        # spin vector scaled to |a|=rg for a=M
        self.add(RotArrow(
            r, self.ca, ra,
            size = self.size_a,
            alpha = self.aa,
            pos = pos,
            ))
        # ergosphere
        ran = LA.norm(ra)
        if (ran > r * self.min_am) and (self.ae > 0):
            rs = 2 * rg / self.scale_r
            rx = r * ra / ran
            self.add(RotationEllipsoid(
                rx, rs, self.ce, self.ae,
                res = self.res,
                pos = pos,
                ))
        # track
        if self._move and self.track:
            r = np.asarray([[0.,0.,0.], pos])
            r = self.bh_pos / self.scale_p
            self.add(Track(
                r,
                self.ct,
                rad = self.rt,
                alpha = self.at,
                alpha_final = self.at,
                exp = self.et,
                ))

        # A(filename='z40p.txt', scale_p=1, campos=[0,6,4.2], camthick=1e4).run()
        # A(filename='z40.txt', scale_p=1, camalign='final').run()
        # A(filename='z40.txt', scale_p=1, campos=[ 4, -1.5,  6.2], camthick=31000).run()
