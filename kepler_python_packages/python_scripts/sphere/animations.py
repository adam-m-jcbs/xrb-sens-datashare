"""
Definition of Animations
"""

import numpy as np
from numpy import linalg as LA

import human.time

from rotation import Rotator
from rotation import rotscale

from .base import Scene
from .base import Animation

from .scenes import TrailScene
from .scenes import BasicScene
from .scenes import FieldLineScene
from .scenes import ShearFieldScene
from .scenes import ShearVectorScene
from .scenes import TraceFieldScene
from .scenes import DivergentShearFieldScene
from .scenes import OscillatingShearFieldScene
from .scenes import RotationScene
from .scenes import RotVecScene
from .scenes import BlackHoleSpinScene

class EmptyAnimation(Animation):
    _scene = Scene
    _nstep = 1

class ShellAnimation(Animation):
    _ref = 'w'
    _cycle = True
    _corot = False
    _camalign = False
    _camup = None
    _camoffset = "w/2"
    _phase0 = 0
    _phase1 = 1
    _framealign = 0
    _period = None

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        ref = kwargs.pop('ref', self._ref)
        cycle = kwargs.pop('cycle', self._cycle)
        framealign = kwargs.pop('framealign', self._framealign)
        corot = kwargs.pop('corot', self._corot)
        camalign = kwargs.pop('camalign', self._camalign)
        camup = kwargs.pop('camup', self._camup)
        phase0 = kwargs.pop('phase0', self._phase0)
        phase1 = kwargs.pop('phase1', self._phase1)
        camoffset = kwargs.pop('camoffset', self._camoffset)
        period = kwargs.pop('period', self._period)

        scene = self.scene
        camera = scene.camera

        if ref == 'w':
            if hasattr(self, 'w'):
                w = self.w
            elif hasattr(scene, 'w'):
                w = scene.w
            else:
                w = rotscale(scene.w1, scene.w2)
        elif ref == 'w1':
            w = scene.w1
        elif ref == 'w2':
            w = scene.w2
        else:
            raise ArttribueError(f'Unknown reference {ref}')

        if isinstance(camoffset, str) and camoffset == "w/2":
            camoffset = w / 2

        if period is None:
            if hasattr(scene, 'period'):
                period = scene.period
            else:
                period = (2 * np.pi) / LA.norm(w)

        if cycle:
            stepdiv = 1 / self.nstep
        else:
            stepdiv = 1 / (self.nstep - 1)

        times = (phase0 + (np.arange(self.nstep) + framealign) * (phase1 - phase0) * stepdiv) * period

        camup0 = np.asarray(camera.GetViewUp())
        campos0 = np.asarray(camera.GetPosition())
        campos1 = campos0

        if camup is not None:
            camup0 = camup

        if np.shape(camalign) == (3,):
            camup1 = camalign
        elif camalign:
            camup1 = w
        else:
            camup1 = camup0

        if corot:
            m = Rotator(w, time = times)
            campos = m(campos1) + camoffset[np.newaxis, :]
            camup = m(camup1)
        else:
            campos = np.tile(campos1, (self.nstep,1)) + camoffset[np.newaxis, :]
            camup = np.tile(camup1, (self.nstep, 1))

        self.camfocalpoint = camoffset
        self.camup = camup
        self.campos = campos
        self.times = times

    def setup(self):
         camera = self.scene.camera
         camera.SetFocalPoint(self.camfocalpoint)

    def draw(self, iframe):
        camera = self.scene.camera
        camera.SetPosition(self.campos[iframe])
        camera.SetViewUp(self.camup[iframe])
        self.scene.update(self.times[iframe])

class TrailAnimation(ShellAnimation):
    _nstep = 3600
    _phase0 = 0
    _phase1 = 39
    _corot = True
    _scene = TrailScene

class FlyAroundAnimation(Animation):
    _nstep = 300
    _w = np.asarray([-1., 0., 0.])
    _pos = np.asarray([0., 0., 8.])
    _center = np.asarray([0., 0., 0.])
    _phase0 = 0
    _phase1 = 1
    _cycle = True

    def init(self, *args, **kwargs):
        center = kwargs.pop('center', self._center)
        w = kwargs.pop('w', self._w)
        pos = kwargs.pop('r', self._pos)
        phase0 = kwargs.pop('phase0', self._phase0)
        phase1 = kwargs.pop('phase1', self._phase1)
        cycle = kwargs.pop('cycle', self._cycle)

        w = np.asarray(w)
        center = np.asarray(center)
        pos = np.asarray(pos)

        w = w * 2 * np.pi / LA.norm(w)
        if cycle:
            stepdiv = 1 / self.nstep
        else:
            stepdiv = 1 / (self.nstep - 1)
        t = (phase0 + (phase1 - phase0) * stepdiv) * np.arange(self.nstep)
        campos = Rotator(w, time = t)(pos - center)
        camup = np.cross(w, campos, axis = -1)
        campos = campos + center

        self.campos = campos
        self.camup = camup
        self.camfocalpoint = center

    def setup(self):
         camera = self.scene.camera
         camera.SetFocalPoint(self.camfocalpoint)

    def draw(self, iframe):
        camera = self.scene.camera
        camera.SetPosition(self.campos[iframe])
        camera.SetViewUp(self.camup[iframe])


class BasicAnimation(FlyAroundAnimation):
    _scene = BasicScene

class FieldLineAnimation(ShellAnimation):
    _nstep = 1200
    _phase0 = 0
    _phase1 = 20
    _corot = True
    _ref = 'w1'

    _vt = None

    _scene = FieldLineScene

    def preprocess(self, args, kwargs):
        vt = kwargs.get('vt', self._vt)
        if vt is not None:
            kwargs['vt'] = vt
        super().preprocess(args, kwargs)

class ShearFieldAnimation(ShellAnimation):
    _nstep = 600
    _phase0 = 0
    _phase1 = 1
    _corot = False
    _ref = 'w'

    _mag = False
    _torque = False
    _size = 0.25
    _arc = False

    _scene = ShearFieldScene

    def preprocess(self, args, kwargs):
        mag = kwargs.get('mag', self._mag)
        if mag is not None:
            kwargs['mag'] = mag
        torque = kwargs.get('torque', self._torque)
        if torque is not None:
            kwargs['torque'] = torque
        size = kwargs.get('size', self._size)
        if size is not None:
            kwargs['size'] = size
        arc = kwargs.get('arc', self._arc)
        if arc is not None:
            kwargs['arc'] = arc
        super().preprocess(args, kwargs)

class ShearVectorAnimation(ShellAnimation):
    _nstep = 600
    _phase0 = 0
    _phase1 = 1
    _corot = False
    _ref = 'w'

    _mag = False
    _torque = False
    _size = None
    _arc = False

    _scene = ShearVectorScene

    def preprocess(self, args, kwargs):
        mag = kwargs.get('mag', self._mag)
        if mag is not None:
            kwargs['mag'] = mag
        torque = kwargs.get('torque', self._torque)
        if torque is not None:
            kwargs['torque'] = torque
        size = kwargs.get('size', self._size)
        if size is not None:
            kwargs['size'] = size
        arc = kwargs.get('arc', self._arc)
        if arc is not None:
            kwargs['arc'] = arc
        super().preprocess(args, kwargs)


class TraceFieldAnimation(ShellAnimation):
    _nstep = 600
    _phase0 = 0
    _phase1 = 1
    _corot = False
    _ref = 'w'
    _scene = TraceFieldScene


class DivergentShearFieldAnimation(ShearFieldAnimation):
    _camoffset = np.asarray([0., 0.1, 0.])
    _framealign = 0.5
    _scene = DivergentShearFieldScene
    _fixedtime = 0
    _size = None

    def preprocess(self, args, kwargs):
        kwargs.setdefault('fixedtime', self._fixedtime)
        super().preprocess(args, kwargs)

class OscillatingShearFieldAnimation(ShearFieldAnimation):
    _scene = OscillatingShearFieldScene
    _fixedtime = None
    _size = None
    _camalign = np.asarray([0., 1., 0.])
    _camoffset = _camalign * 0.5

    def preprocess(self, args, kwargs):
        kwargs.setdefault('fixedtime', self._fixedtime)
        super().preprocess(args, kwargs)

class RotationAnimation(Animation):
    _scene = RotationScene

    _nstep = None
    _ncyc = 1
    _time = 5
    _cycle = True

    def preprocess(self, args, kwargs):
        self.resolve_defaults(kwargs, None)
        super().preprocess(args, kwargs)

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.time = self.time * self.ncyc
        self.nstep = int(self.time / self.delay)
        self.timescale = self.scene.time / self.time * self.ncyc
        if not self.cycle:
            self.timescale *= self.nstep / (self.nstep - 1)


class RotVecSetup(Animation):
    _scene = RotationScene
    _time = 5

    _pos1 = np.asarray([ 1., 1., 1.])
    _pos2 = np.asarray([-1., 1., 1.])

    _camfocus = np.asarray([0., 0., 0.])

    _alpha_t = 0.2
    _alpha_a1 = 0.2

    def preprocess(self, args, kwargs):
        self.resolve_defaults(kwargs, None)

        kwargs['pos1'] = self.pos1
        kwargs['pos2'] = self.pos2

        kwargs['graph_mode'] = 'pos2'
        kwargs['track_mode'] = 'trace'
        kwargs['alpha_te'] = self.alpha_t
        kwargs['alpha_a1'] = self.alpha_a1

        kwargs['camfocus'] = self.camfocus

        super().preprocess(args, kwargs)

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.nstep = int(self.time / self.delay)
        self.timescale = self.scene.time / self.time


class RotVecAnimation(Animation):
    _scene = RotVecScene

    _nstep = None
    _time = 5
    _cycle = True

    _phase0 = 0
    _phase1 = +1

    def preprocess(self, args, kwargs):
        self.resolve_defaults(kwargs, None)
        super().preprocess(args, kwargs)

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.nstep = int(self.time / self.delay)

        timeoffset = self.phase0
        self.timescale = self.scene.time * (self.phase1 - self.phase0) / self.time
        if not self.cycle:
            self.timescale *= self.nstep / (self.nstep - 1)

class BlackHoleSpinAnimation(Animation):
    _scene = BlackHoleSpinScene

    _delay = 1 / 60
    _cycle = False
    _timescale = int(1 / _delay)

    _time_mode = 'frame' # frame | time | logtime

    _timestamp = dict(
        text = 'timeformatter',
        pos = (4+0j, 4+0j),
        align = (0j, 0j),
        size = 16,
        color = '#C0C0C0',
        angle = 0,
        font = 'courbd.ttf',
        )

    def preprocess(self, args, kwargs):
        self.resolve_defaults(kwargs, None)
        kwargs['time_mode'] = self.time_mode
        super().preprocess(args, kwargs)

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)

        if self.time_mode == 'frame':
            self.nstep = self.scene.bh_time.shape[0]
            self.time = self.nstep * self.delay
        elif self.time_mode == 'time':
            self.timeoffset = self.scene.bh_time[0]
            duration = self.scene.bh_time[-1] - self.scene.bh_time[0]
            self.timescale = duration / ((self.nstep - 1) * self.delay)
        elif self.time_mode == 'logtime':
            rat = self.scene.bh_time[-1] / self.scene.bh_time[0]
            def timefunction(iframe):
                frac = iframe / (self.nstep - 1)
                time = self.scene.bh_time[0] * rat**frac
                return time
            self.timefunction = kwargs.get('timefunction', timefunction)
        else:
            raise Exception(f'Unknown time_mode "{self.time_mode}".')

    def timeformatter(self, sframe):
        time = self.timefunc(sframe - self.startframe)
        time = human.time.time2human(time, cut=False)
        time, unit = human.time.split_unit(time)
        return f'time = {time:>3} {unit}'
