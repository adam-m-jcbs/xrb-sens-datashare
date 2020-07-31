"""
Kepler main routines

Example:

"""

# STDL import
import numpy as np
import os
import os.path
import time
import re

from sys import maxsize
from collections import OrderedDict
from math import floor, ceil, log10
from functools import partial
from types import MethodType

# alex lib imports (REPLACE)
import keppar
from logged import Logged

# kepler imports
from ..datainterface import DataInterface
from ..plot.framekep import FrameKep
from ..plot.manager import PlotManager
from ..plot.base import BasePlot
from ..exception import KeplerTerminated
from ..cmdloop import KeplerCmdLoop

# local dir imports
from . import kepbin
from .util import b2s, s2b, z2int, z2str, gets
from .interface import Item, CharVar, Parameter
from ..hooks.framework import CycleHooks

# define local variables
default_plot = []

_var_def = OrderedDict([
    ('jm', 0),
    ('ndump', 0),
    ('nsdump', 0),
    ('nedit', 0),
    ('gencom.mode', 4),
    ('vars.jdtc', 0),
    ])

class KeplerData(DataInterface):
    # TODO - cache values using cycle ID?
    def __init__(self, kepler):
        self._kepler = kepler._kepler
        self.kepler = kepler
    @property
    def filename(self):
        return b2s(self._kepler.namecom.nameprob)
    @property
    def nameprob(self):
        return b2s(self._kepler.namecom.nameprob)
    @property
    def runpath(self):
        return self.kepler.path
    @property
    def parm(self):
        return self.kepler.parm
    @property
    def qparm(self):
        return self.kepler.qparm
    @property
    def var(self):
        return self.kepler.var
    def __getattr__(self, var):
        o = object()
        if var.count('.') > 0:
            split = var.split('.', 1)
            return getattr(getattr(self, split[0]), split[1])
        val = getattr(self.var, var, o)
        if val is not o:
            return val
        val = getattr(self.parm, var, o)
        if val is not o:
            return val
        val = getattr(self.qparm, var, o)
        if val is not o:
            return val
        val = getattr(self._kepler.vars, var, o)
        if val is not o:
            val = val.copy()
            return val
        # try loadbuf
        jm = self.qparm.jm
        buf, label, ierr = self._kepler.loadbuf_(var, 1, jm)
        if ierr == 0:
            val = np.zeros(jm+2)
            val[1:-1] = buf[:jm]
            return val
        val = getattr(self._kepler.charsave, var, o)
        if val is not o:
            val = val.copy()
            return val
        raise AttributeError(var)
    @property
    def iconv(self):
        jm = self.qparm.jm
        icon, label, ierr = self._kepler.loadbuf_('convect', 1, jm)
        return np.int_(np.round(icon))
    @property
    def angit(self):
        jm = self.qparm.jm
        it = np.zeros(jm+2)
        it[1:-1] = np.cumsum(self.angi[1:jm+1]*self.xm[1:jm+1])
        return it
    @property
    def jdtc(self):
        return self.var.jdtc
    @property
    def datatime(self):
        return time.asctime()
    @property
    def idtcsym(self):
        sym = [x.tostring().decode().strip() for x in self._kepler.charsave.idtcsym]
        return sym
    @staticmethod # copy from kepdump --> should be in keputils
    def center2face(x, logarithmic=False):
        """
        create zone interface quantity by averaging zone values
        """
        y = x.copy()
        if logarithmic:
            y[:-1] *= x[1:]
            y[1:-2] = np.sqrt(y[1:-2])
        else:
            y[:-1] += x[1:]
            y      *= 0.5
        y[0]    = x[1]
        y[-2]   = x[-2]
        y[-1]   = np.nan
        return y
    @property
    def dnf(self):
        """
        Density at zone interface (flat extrapolation for boundaries) (g/cm**3).
        """
        return self.center2face(self.dn)
    @property
    def entropies(self):
        """
        return array of entropes from call to EOS

        This is done not just using loadbuf as this would reqire
        many more calls to the equation of state.
        """
        return self._kepler.getentropies_(1,self.qparm.jm)
    # BURN interfce
    @property
    def numib(self):
        return z2int(self._kepler.vars.znumib)
    @property
    def netnumb(self):
        """
        here we have to define a function to overwrite that a float
        value would be returned by loadbuf
        """
        return z2int(self._kepler.vars.znetnumb)
    @property
    def ionnb(self):
        return z2int(self._kepler.vars.zionnb)
    @property
    def ionsb(self):
        return np.array([i.decode('us_ascii') for i in self._kepler.charsave.ionsb])
    @property
    def ionbmax(self):
        return np.array([i.decode('us_ascii') for i in self._kepler.charsave.ionbmax])
    @property
    def isosym(self):
        return np.array([i.decode('us_ascii') for i in self._kepler.charsave.isosym])
    @property
    def ppnb(self):
        x = np.ndarray((self.qparm.jm, self.qparm.imaxb), buffer=self._kepler.vars.ppnb).transpose()
        x = np.insert(x, [0, self.qparm.jm],0,axis=1)
        return x
    # APPROX/QSE/NSE interface
    @property
    def ionn(self):
        return z2int(self._kepler.vars.zionn)
    @property
    def numi(self):
        return z2int(self._kepler.vars.znumi)
    @property
    def ppn(self):
        x = np.ndarray((self.qparm.jm, self.qparm.imax), buffer=self._kepler.vars.ppn).transpose()
        x = np.insert(x, [0,self.qparm.jm], 0, axis=1)
        return x

# the following now is a singleton metaclass
class MetaSingletonHash(type):
    """
    Singleton metaclass based on hash

    First creates object to be able to test hash.

    If same hash is found, return old object and discard new one,
    otherwise return old one.

    Usage:
       class X(Y, metaclass = MetaSingletonHash)

    class X needs to provide a __hash__ function
    """
    def __call__(*args, **kwargs):
        cls = args[0]
        try:
            cache = cls._cache
        except:
            cache = dict()
            cls._cache = cache
        obj = type.__call__(*args, **kwargs)
        key = cls._key_func(obj)
        return cache.setdefault(key, obj)

    def _check_cache(*args, **kwargs):
        cls = args[0]
        obj = args[1]
        key = cls._key_func(obj)
        if hasattr(cls, '_cache'):
            exists = key in cls._cache
            if exists:
                print(f' [{cls.__name__.upper()}] found original {key}')
            return exists
        return False

    def _key_func(cls, obj):
        return obj.__hash__()

def args2namegen(args, kwargs, return_reduced = True):
    '''
    KEPLER master disassembly code
    '''
    args_ = list(args)
    kwargs_ = dict(kwargs)
    if len(args_) == 0:
        name = kwargs_.pop('name')
    else:
        name = args_.pop(0)
    # is name restart or generator?
    match = re.findall('^(.+)(?:z\d*|g|#.*)$', name)
    if len(match) == 1:
        gen = name
        name = match[0]
        if 'gen' in kwargs_:
            raise AttributeError('Duplicate generator specification.')
    else:
        if len(args_) == 0:
            try:
                gen = kwargs_.pop('gen')
            except KeyError:
                raise AttributeError('Missing generator specification.')
        else:
            gen = args_.pop(0)
        match = re.findall('^(?:z\d*|g|#.*)$', gen)
        if len(match) == 1:
            gen = name + match[0]
    if return_reduced:
        return name, gen, args_, kwargs_
    return name, gen

class KepParm():
    """
    hold kepler code run parameters
    """
    _defaults = OrderedDict(
        name = 'xxx',
        gen = 'xxxz',
        path = '~/kepler/test',
        killburn = False,
        progname = 'python',
        parms = (),
        )

    def __init__(self,
                 *args, **kwargs):
        keppar = self._args2kep(args, kwargs, return_reduced = False)
        self.__dict__.update(keppar)

    def args(self):
        args = [
            self.progname,
            self.name,
            self.gen,
            ] + list(self.parms)
        if self.killburn:
            args += ['k']
        return args

    def kwargs(self):
        kw = {k : getattr(self, k) for k in self._defaults}
        return kw

    @classmethod
    def _args2kep(cls, args, kwargs, return_reduced = True):
        """
        return kepler args and remaining args and parms
        """
        kep = dict()
        debug = kwargs.get('debug', False)
        # this is for debugging ...
        set_defaults = kwargs.pop('set_defaults', debug)
        if set_defaults:
            args_ = list(args)
            kwargs_ = dict(kwargs)
            j = 0
            for i, (k, v) in enumerate(cls._defaults.items()):
                if k in kwargs:
                    kep[k] = kwargs_.pop(k)
                elif i == j and len(args) > i:
                    kep[k] = args_.pop(0)
                    j += 1
                else:
                    kep[k] = v
        else:
            name, gen, args_, kwargs_ = args2namegen(args, kwargs, return_reduced = True)
            kep['name'] = name
            kep['gen'] = gen
            kep['killburn'] = kwargs_.get('killburn', False)
            kep['progname'] = kwargs_.get('progname', 'python')
            kep['path'] = kwargs_.get('path', None)
            kep['parms'] = args_

        parms = kep['parms']
        if parms is None:
            parms = ()
        elif isinstance(parms, str):
            parms = tuple(parms.split())
        if 'k' in parms:
            kep['killburn'] = True
            parms = list(parms)
            parms.remove('k')
            parms = tuple(parms)
        kep['parms'] = parms

        path = kep.get('path', None)
        if path is None:
            path = os.getcwd()
        else:
            path = os.path.expanduser(os.path.expandvars(path))
        kep['path'] = path

        if return_reduced:
            return kep, args_, kwargs_
        return kep

    @classmethod
    def args2kep(cls, args, kwargs, return_reduced = True):
        """
        return kepler args and remaining args and parms
        """
        if return_reduced:
            kep, args_, kwargs_ = cls._args2kep(args, kwargs, return_reduced = True)
            return cls(**kep), args_, kwargs_
        return cls(*args, **kwargs)

    @property
    def portfile(self):
        return os.path.join(self.path, self.name + '.port')


class Kepler(CycleHooks, metaclass = MetaSingletonHash):
    # TODO metaclass to avoid producing duplicate objects that
    # overwrite callbacks, etc.

    _loadargs = ('NBURN', 'JMZ', 'NAME', 'FULDAT', 'profile', 'update')

    def __init__(self, *args, **kwargs):
        kwl = {k:kwargs.pop(k) for k in self._loadargs if k in kwargs}
        self._kepler, loaded = kepbin.load(return_loaded = True, **kwl)
        if self.__class__._check_cache(self):
            print(f' [KEPLER] Returning original object {self} instead of a new one.')
            print(f'          Use "start" method if you need to restart local session')
            print(f'          or use process interface for multiple sessions.')
            return
        self._server = kwargs.pop('server', None)
        self._debug = kwargs.pop('debug', False)
        self._remote = kwargs.pop('remote', False)
        if self._server:
            self._server._set_callbacks(self)
        else:
            self._set_callbacks()

        self.clear_hooks()

        start = kwargs.get('start', True)
        if start:
            if not loaded:
                self.start(*args, debug = self._debug, **kwargs)
            else:
                print(f' [KEPLER] Use "start" method if you need to restart.')

    def __hash__(self):
        if hasattr(self, '_kepler'):
            return self._kepler.__name__
        return None

    def start(self, *args, **kwargs):
        """
        initialise kepler binary module
        """
        self._debug = kwargs.get('debug', self._debug)

        kepparm = KepParm(*args, **kwargs)

        # set up current path
        self.cwd = os.getcwd()
        self.path = kepparm.path
        os.chdir(self.path)

        kepargs = kepparm.args()
        commands = np.zeros(21, dtype="S80")
        for i, cmd in enumerate(kepargs):
            commands[i] = f'{cmd:<80s}'

        S = np.ndarray(80 * 21, dtype='S1', buffer = commands.data)
        self._kepler.start_(len(kepargs)-1, S)
        self.status = 'started'

        self.kd = KeplerData(self)
        self.parm = Parameter(self._kepler, keppar.p, 'p')
        self.qparm = Parameter(self._kepler, keppar.q, 'q')
        self.var = Parameter(self._kepler, _var_def)

        if not self._server:
            kw = dict()
            plot_style = kwargs.pop('style', None)
            if plot_style is not None:
                kw['style'] = plot_style
            self.kp = PlotManager(self.kd, **kw)
        else:
            self.kp = self._server.kp
        plot = kwargs.pop('plot', False)
        if plot:
            self.plot()

    @property
    def jmz(self):
        return self._kepler.vars.xm.shape[0] - 1
    @property
    def nburn(self):
        return self._kepler.vars.windb.shape[0]

    def execute(self, cmdline):
        """
        Execute Kepler command as on the terminal.  Pass string.
        """
        self._kepler.execute_(cmdline + ' '*(132-len(cmdline)), 1)

    def _cycle(self):
        self._kepler.cycle_(False)

    def cycle(self, n = 1):
        """
        cycle for n steps.
        """
        if self.status != "started":
            raise KeplerTerminated()

        self._kepler.gencom.mode = 1
        try:
            for i in range(n):
                self._hook_cycle(cycler = self._cycle)
                if not self._remote:
                    x = gets()
                    if x == 's':
                        break
                    elif x != '':
                        x = str(x).strip()
                        self.execute(x)
                if i == n-1 or self._kepler.gencom.mode == 0:
                    break
            # this needs to be adjusted to trigger plots from python
            # not form KEPLER
            if self.parm.itvstart < 1:
                # copy from cycle.f
                iseconds = int(time.time())
                if (self.qparm.ilastpl > iseconds):
                    self.qparm.ilastpl = 0
                if (self.parm.npixedit != 0):
                    if ((((self.qparm.ncyc % self.parm.npixedit) == 0) and
                         (iseconds - self.qparm.ilastpl >= self.parm.ipdtmin)) or
                          (i == n - 1)):
                        self.kp.update(interactive = False)
                        self.qparm.ilastpl = iseconds
        except KeplerTerminated:
            self.status = 'terminated'
            self.kp.closewin()

        if not self._remote:
            ncyc = self.qparm.ncyc
            print(f' [CYCLE] Stop at cycle {ncyc:d}')

    #@property # broken autocompletion
    def s(self, n = 1):
        """
        Do a single step.  No need to call as function.
        """
        self.cycle(n)

    #@property # broken autocompletion
    def g(self):
        """
        Do run continuously ('go').
        """
        self.cycle(maxsize)

    def terminate(self, message = ''):
        self.closewin()
        os.chdir(self.cwd)
        self._kepler.terminate_(message)

    def plot(self, *args, **kwargs):
        self.kp.plot(*args, **kwargs)

    def update(self, *args, **kwargs):
        self.kp.update(*args, **kwargs)

    def closewin(self):
        self.kp.closewin()

    def pt(self, mode):
        p = FrameKep(self.kd, mode)

    #@property # broken autocompletion
    def run(self):
        """
        Interactive KEPLER mode

        TODO replace with something that allows plots to continue work.
        App?
        """

        KeplerCmdLoop(self).cmdloop()

    def __getattr__(self, key):
        if 'var' in self.__dict__:
            try:
                return self.var[key]
            except:
                pass
        if 'parm' in self.__dict__:
            try:
                return self.parm[key]
            except:
                pass
            if key[0] == 'p':
                try:
                    num = int(key[1:])
                    return self.parm[num]
                except:
                    pass
        if 'qparm' in self.__dict__:
            try:
                return self.qparm[key]
            except:
                pass
            if key[0] == 'q':
                try:
                    num = int(key[1:])
                    return self.qparm[num]
                except:
                    pass
        if '_kepler' in self.__dict__:
            o = object()
            val = getattr(self._kepler.vars, key, o)
            if val is not o:
                return val
        # try loadbuf
        if 'qparm' in self.__dict__ and '_kepler' in self.__dict__:
            jm = self.qparm.jm
            buf, label, ierr = self._kepler.loadbuf_(key, 1, jm)
            if ierr == 0:
                val = np.zeros(jm+2)
                val[1:-1] = buf[:jm]
                return val
        raise AttributeError()

    def __setattr__(self, key, value):
        if 'parm' in self.__dict__:
            if hasattr(self.parm, key):
                self.parm[key] = value
                return
            if key[0] == 'p':
                try:
                    num = int(key[1:])
                    if 0 < num <= len(self.parm._index):
                        self.parm[num] = value
                        return
                except:
                    pass
        if 'qparm' in self.__dict__:
            if hasattr(self.qparm, key):
                raise AttributeError("read only variable '{}'".format(key))
                self.qparm[key] = value
                return
        super().__setattr__(key, value)

    def __getitem__(self, key):
        index = None
        if not isinstance(key, tuple):
            attr = key
        else:
            attr = key[0]
            index = tuple(key[1:])
        if not isinstance(attr, str):
            raise TypeError('Require name of remote string.')
        attrs = attr.split('.')
        val = self
        for a in attrs:
            try:
                val = getattr(val, a)
            except AttributeError:
                raise AttributeError(attr)
        if isinstance(val, MethodType):
            raise AttriburError(attr)
        if index is not None:
            return val[index]
        return val

    def __setitem__(self, key, value):
        index = None
        if not isinstance(key, tuple):
            attr = key
        else:
            attr = key[0]
            index = tuple(key[1:])
        if not isinstance(attr, str):
            raise TypeError('Require name of remote string.')
        attrs = attr.split('.')
        obj = self
        for a in attrs[:-1]:
            try:
                obj = getattr(obj, a)
            except AttributeError:
                raise AttributeError(attr)
        if index is not None:
            try:
                var = getattr(obj, attrs[-1])
            except AttributeError:
                raise AttributeError(attr)
            try:
                var[index] = value
            except IndexError:
                raise IndexError(index, attr)
            except Exception as error:
                raise Exception(error)
            return None
        else:
            try:
                return setattr(obj, attrs[-1], value)
            except AttributeError:
                raise AttributeError(attr)
            except Exception as error:
                raise Exception(error)


        return val

    # ============================================================
    # define and set callbacks
    # ============================================================
    #
    # need to add interfaces for KEPLER to request data
    #
    def endkepler(self, code = None):
        """
        Callback to end KEPLER.
        """
        print(' [endkepler] has been called.')
        self.status = 'terminated'
        raise KeplerTerminated(code)

    def plotkepler(self):
        """
        Callback to make / update plots.
        """
        print(' [kepler] called python plot', self.parm.ipixtype)
        if self.parm.itvstart == 0:
            return
        if self.parm.itvstart == -1:
            self.parm.itvstart == 0
            self.closewin()
        if self.parm.ipixtype == 0 and len(self.kp) == 0:
            return
        p = self.parm.ipixtype
        if p > 0:
            self.kp.plot(p, interactive = False, duplicates = False)
        else:
            self.kp.update(interactive = False)

    def ttykepler(self, msg):
        """
        Callback to get input.

        Not sure this is needed or a good name.
        """
        # update to use different feature
        prompt = b2s(self._kepler.namecom.nameprob).strip('\x00').strip() + '> '
        s = input(prompt)
        msg[:] = 0
        for i, c in enumerate(s):
            msg[i] = bytes(c, encoding = 'us-ascii')[0]

    def _set_callbacks(self):
        self._kepler.endkepler = self.endkepler
        self._kepler.plotkepler = self.plotkepler
        self._kepler.ttykepler = self.ttykepler
