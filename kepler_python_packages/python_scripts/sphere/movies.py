#! /bin/env python3

import socket
import os
import sys

import numpy as np

from rotation import rotate

from .animations import TrailAnimation
from .animations import BasicAnimation
from .animations import FieldLineAnimation
from .animations import ShearFieldAnimation
from .animations import TraceFieldAnimation
from .animations import DivergentShearFieldAnimation
from .animations import OscillatingShearFieldAnimation
from .animations import RotVecAnimation
from .animations import EmptyAnimation
from .animations import RotVecSetup
from .animations import BlackHoleSpinAnimation

from .stories import RotationStory

class MovieLib():
    def update_movies(self, movies):
        if self.store == False:
            return tuple()
        movies = tuple(os.path.join(self.base, m) for m in movies)
        return movies

    def update_base(self):
        if self.base is None:
            host = socket.gethostname()
            if host == 'phosphorus.physics.monash.edu':
                self.base = '/m/web/Download/rot'
            elif host == 'nitrogen.physics.monash.edu':
                self.base = '/m/web/Download/rot'
            elif host == 'w.2sn.net':
                self.base = '/home/alex/Downloads/'
            elif host == 'yttrium.physics.monash.edu':
                self.base = '/home/alex/Downloads/'
            else:
                self.base = os.getcwd()

    def __init__(self, *setups, **kwargs):
        self.base = kwargs.pop('base', None)
        self.store = kwargs.pop('store', True)
        self.format = kwargs.pop('format', 'webm')
        self.kwargs = kwargs
        self._start(setups, kwargs)

    def run(self, *setups, **kwargs):
        self.store = kwargs.pop('store', self.store)
        self.format = kwargs.pop('format', self.format)
        self.base = kwargs.pop('base', self.base)
        kw = self.kwargs.copy()
        kw.update(self.kwargs)
        self._start(setups, kw)

    def _start(self, setups, kwargs):
        if isinstance(self.format, str):
            self.format = self.format,
        self.update_base()
        if len(setups) == 1 and isinstance(setups[0], (list, tuple, np.ndarray)):
            setups = setups[0]
        for setup in setups:
            self.run_one(setup, **kwargs)

    def run_one(self, setup, **kwargs):
        setup = setup.strip()
        cfg = setup.split(':')
        if len(cfg) == 2:
            setup, filename = cfg
        else:
            filename = setup
        movies = self.update_movies(
            tuple(f'{filename}.{f}' for f in self.format))
        animation = None
        kw = dict()
        if setup == 'trackcc':
            animation = TrailAnimation
            kw['phase1'] = 39
            kw['nstep'] = 3600
        elif setup == 'fly':
            animation = BasicAnimation
            kw['nstep'] = 600
        elif setup == 'corot': # 20 sec
            animation = FieldLineAnimation
            kw['nstep'] = 1200
        elif setup == 'inertial': # 20 sec
            animation = FieldLineAnimation
            kw['corot'] = False
            kw['nstep'] = 1200
        elif setup == 'windc':
            animation = FieldLineAnimation
            kw['ref'] = 'w2'
            kw['phase1'] = 95
            kw['time'] = 60
            kw['w2'] = np.asarray([0.2, 1.0, 0.0]) * 0.95,
        elif setup == 'windd':
            animation = FieldLineAnimation
            kw['ref'] = 'w2'
            kw['phase1'] = 95
            kw['time'] = 60
            kw['w2'] = np.asarray([0.2, 1.0, 0.0]) * 0.95,
            kw['vt'] = np.asarray([0.0,-0.3, 1.0]),
        elif setup == 'winde':
            animation = FieldLineAnimation
            kw['ref'] = 'w'
            kw['phase1'] = 117
            kw['time'] = 72
            kw['w2'] = np.asarray([0.2, 1.0, 0.0]) * 0.95,
            kw['vt'] = np.asarray([0.0,-0.3, 1.0]),
        elif setup == 'sheari':
            animation = ShearFieldAnimation
            kw['phase1'] = 0.25
            kw['time'] = 2.5
        elif setup == 'shearia':
            animation = ShearFieldAnimation
            kw['phase1'] = 0.25
            kw['time'] = 2.5
            kw['arc'] = True
        elif setup == 'shearc':
            animation = ShearFieldAnimation
            kw['corot'] = True
        elif setup == 'shearca':
            animation = ShearFieldAnimation
            kw['corot'] = True
            kw['arc'] = True
        elif setup == 'torqi':
            animation = ShearFieldAnimation
            kw['torque'] = True
            kw['phase1'] = 0.25
            kw['time'] = 2.5
        elif setup == 'torqc':
            animation = ShearFieldAnimation
            kw['torque'] = True
            kw['corot'] = True
        elif setup == 'osc1i':
            animation = TraceFieldAnimation
            kw['order'] = 1
        elif setup == 'osc1c':
            animation = TraceFieldAnimation
            kw['order'] = 1
            kw['corot'] = True
        elif setup == 'osc0i,':
            animation = TraceFieldAnimation
            kw['order'] = 0
        elif setup == 'osc0c':
            animation = TraceFieldAnimation
            kw['order'] = 0
            kw['corot'] = True
        elif setup == 'oscm1i':
            animation = TraceFieldAnimation
            kw['order'] = -1
        elif setup == 'oscm1c':
            animation = TraceFieldAnimation
            kw['order'] = -1
            kw['corot'] = True
        elif setup == 'osc2i':
            animation = TraceFieldAnimation
            kw['order'] = 2
        elif setup == 'osc2c':
            animation = TraceFieldAnimation
            kw['order'] = 2
            kw['corot'] = True
        elif setup == 'osc3i':
            animation = TraceFieldAnimation
            kw['order'] = 3
        elif setup == 'osc3c':
            animation = TraceFieldAnimation
            kw['order'] = 3
            kw['corot'] = True
        elif setup == 'sheardi':
            animation = DivergentShearFieldAnimation
            kw['corot'] = False
            kw['fixedtime'] = None
        elif setup == 'sheardia':
            animation = DivergentShearFieldAnimation
            kw['arc'] = True
            kw['corot'] = False
            kw['fixedtime'] = None
        elif setup in ('sheardeia', 'sheardefa'):
            animation = DivergentShearFieldAnimation
            kw['w10'] = np.asarray([0.,1.,0.])
            kw['w20'] = kw['w10']
            kw['arc'] = True
            kw['corot'] = False
            if setup == 'sheardeia':
                kw['fixedtime'] = None
            else:
                kw['fixedtime'] = 0
        elif setup == 'torqdi':
            animation = DivergentShearFieldAnimation
            kw['torque'] = True
            kw['corot'] = False
            kw['fixedtime'] = None
        elif setup == 'sheardf':
            animation = DivergentShearFieldAnimation
            kw['corot'] = False
            kw['fixedtime'] = 0
        elif setup == 'sheardfa':
            animation = DivergentShearFieldAnimation
            kw['arc'] = True
            kw['corot'] = False
            kw['fixedtime'] = 0
        elif setup == 'torqdf':
            animation = DivergentShearFieldAnimation
            kw['torque'] = True
            kw['corot'] = False
            kw['fixedtime'] = 0
        elif setup == 'shearoi':
            animation = OscillatingShearFieldAnimation
            kw['torque'] = False
            kw['corot'] = False
            kw['fixedtime'] = None
        elif setup == 'shearoia':
            animation = OscillatingShearFieldAnimation
            kw['arc'] = True
            kw['torque'] = False
            kw['corot'] = False
            kw['fixedtime'] = None
        elif setup == 'torqoi':
            animation = OscillatingShearFieldAnimation
            kw['torque'] = True
            kw['corot'] = False
            kw['fixedtime'] = None
        elif setup in ('shearmi', 'shearmia'):
            animation = OscillatingShearFieldAnimation
            kw['torque'] = False
            kw['corot'] = False
            kw['fixedtime'] = None
            w = np.asarray([0., 1., 0.])
            kw['w'] = w
            kw['w10'] = rotate(w, [0.,0.,1.], phase = +0.25 * np.pi)
            kw['w20'] = rotate(w, [0.,0.,1.], phase = -0.25 * np.pi)
            kw['m10'] = 0.6
            kw['m20'] = 0.6
            kw['a10'] = 0.4
            kw['a20'] = 0.4
            kw['period'] = np.pi * 2 / 0.6
            if setup == 'shearmia':
                kw['arc'] = True
        elif setup == 'torqmi':
            animation = OscillatingShearFieldAnimation
            kw['torque'] = True
            kw['corot'] = False
            kw['fixedtime'] = None
            w = np.asarray([0., 1., 0.])
            kw['w'] = w
            kw['w10'] = rotate(w, [0.,0.,1.], phase = +0.25 * np.pi)
            kw['w20'] = rotate(w, [0.,0.,1.], phase = -0.25 * np.pi)
            kw['m10'] = 0.6
            kw['m20'] = 0.6
            kw['a10'] = 0.4
            kw['a20'] = 0.4
            kw['period'] = np.pi * 2 / 0.6
        elif setup == 'shearof':
            animation = OscillatingShearFieldAnimation
            kw['torque'] = False
            kw['corot'] = False
            kw['fixedtime'] = 0
        elif setup == 'shearofa':
            animation = OscillatingShearFieldAnimation
            kw['arc'] = True
            kw['torque'] = False
            kw['corot'] = False
            kw['fixedtime'] = 0
        elif setup == 'torqof':
            animation = OscillatingShearFieldAnimation
            kw['torque'] = True
            kw['corot'] = False
            kw['fixedtime'] = 0
        elif setup in ('shearmf', 'shearmfa'):
            animation = OscillatingShearFieldAnimation
            kw['torque'] = False
            kw['corot'] = False
            kw['fixedtime'] = 0
            w = np.asarray([0., 1., 0.])
            kw['w'] = w
            kw['w10'] = rotate(w, [0.,0.,1.], phase = +0.25 * np.pi)
            kw['w20'] = rotate(w, [0.,0.,1.], phase = -0.25 * np.pi)
            kw['m10'] = 0.6
            kw['m20'] = 0.6
            kw['a10'] = 0.4
            kw['a20'] = 0.4
            if setup == 'shearmfa':
                kw['arc'] = True
        elif setup == 'torqmf':
            animation = OscillatingShearFieldAnimation
            kw['torque'] = True
            kw['corot'] = False
            kw['fixedtime'] = 0
            w = np.asarray([0., 1., 0.])
            kw['w'] = w
            kw['w10'] = rotate(w, [0.,0.,1.], phase = +0.25 * np.pi)
            kw['w20'] = rotate(w, [0.,0.,1.], phase = -0.25 * np.pi)
            kw['m10'] = 0.6
            kw['m20'] = 0.6
            kw['a10'] = 0.4
            kw['a20'] = 0.4
        elif setup == 'axisrot':
            animation = RotationStory
        elif setup == 'rotvec_setup':
            animation = RotVecSetup
        elif setup == 'rotvec':
            animation = RotVecAnimation
            kw['phase0'] = -1
            kw['circ_mode'] = 'twist'
            kw['time'] = 10
        elif setup == 'rotcirc':
            animation = RotVecAnimation
            kw['time'] = 10
        elif setup == 'rotcircl':
            animation = RotVecAnimation
            kw['time'] = 5
            kw['phase1'] = 0.5
            kw['vec_mode'] = 'limit'
        elif setup == 'rotcircs':
            animation = RotVecAnimation
            kw['time'] = 5
            kw['phase1'] = 0.5
            kw['vec_mode'] = 'shadow'
            kw['alpha_t'] = 0.5
        elif setup == 'rotcircd':
            animation = RotVecAnimation
            kw['time'] = 10
            kw['vec_mode'] = 'dual'
            kw['alpha_t'] = 0.5
        elif setup == 'rotcircp':
            animation = RotVecAnimation
            kw['time'] = 10
            kw['rot_mode'] = 'project'
        elif setup == 'rotcircdp':
            animation = RotVecAnimation
            kw['time'] = 10
            kw['vec_mode'] = 'dual'
            kw['alpha_t'] = 0.5
            kw['rot_mode'] = 'project'
        elif setup == 'rotcircnp':
            animation = RotVecAnimation
            kw['offscreen'] = True
            kw['time'] = 10
            kw['vec_mode'] = 'project'
        elif setup == 'z40':
            animation = BlackHoleSpinAnimation
            kw['cc_note.text'] = "z40, J100; Chan+ (2018)"
            kw['filename'] = 'z40.txt'
            kw['size_a'] = 100.
            kw['scale_r'] = -10.
            kw['time_mode'] = 'logtime'
            kw['nstep'] = 600
        elif setup == 'z40p':
            animation = BlackHoleSpinAnimation
            kw['cc_note.text'] = "z40p; Chan+ (2019)"
            kw['filename'] = 'z40p.txt'
            kw['size_a'] = 1.
            kw['scale_r'] = -2.
            kw['time_mode'] = 'logtime'
            kw['nstep'] = 600
        elif setup == 'z40p_x4':
            animation = BlackHoleSpinAnimation
            kw['cc_note.text'] = "z40p, Jx4; Chan+ (2019)"
            kw['filename'] = 'z40p.txt'
            kw['scale_a'] = 4.
            kw['size_a'] = 1.
            kw['scale_r'] = -2.
            kw['time_mode'] = 'logtime'
            kw['nstep'] = 600
        elif setup == 'z40_vel':
            animation = BlackHoleSpinAnimation
            kw['cc_note.text'] = "z40; Chan+ (2018)"
            kw['filename'] = 'z40.txt'
            kw['scale_p'] = 1.
            kw['campos'] = [4., -1.5, 6.2]
            kw['time_mode'] = 'logtime'
            kw['interp_mode'] = 'spline'
            kw['camthick'] = 31000.
            kw['nstep'] = 600
        elif setup == 'z40p_vel':
            animation = BlackHoleSpinAnimation
            kw['cc_note.text'] = "z40p; Chan+ (2019)"
            kw['filename'] = 'z40p.txt'
            kw['scale_p'] = 1.
            kw['campos'] = campos=[0, 6, 4.2]
            kw['time_mode'] = 'logtime'
            kw['interp_mode'] = 'spline'
            kw['camthick'] = 1e4
            kw['nstep'] = 600
        elif setup == 'empty':
            animation = EmptyAnimation
            kw['time'] = 1
        else:
            raise AttributeError(f'Unknown movie setup "{setup}"')
        if animation is not None:
            kw.setdefault('movie', movies)
            kw.update(kwargs)
            animation(**kw).run()


if __name__ == "__main__":
    argv = sys.argv[1:]
    args = argv[1:]
    kwargs = dict()
    d = dict()
    for i,a in enumerate(args):
        try:
            args[i] = eval(a, d, d)
        except:
            pass
    for a in list(args):
        try:
            d = eval(f'dict({a})', d, d)
            kwargs.update(d)
            args.remove(a)
        except:
            pass
    MovieLib(*args, **kwargs)


# from rotation.sphere.movies import MovieLib as M
# M(antialias=-4, offscreen=True).run('shearoia', 'shearmia', 'shearofa', 'shearmfa', 'sheardfa', 'sheardia', 'shearca', 'shearia', 'rotcircnp', ' rotcircdp', 'rotcircd', 'rotcircs', 'rotcircl', 'rotcircp', 'rotcirc', 'rotvec', 'axisrot' )
