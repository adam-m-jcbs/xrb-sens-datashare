from re import compile as recompile
from operator import __add__
from functools import reduce
from os import getpid

from copy import copy

_re_index = recompile('\[[^\]\[]*\]')

_done_token = 'Done.'
_ioid_token = 'IOID='

def slice_index(arg):
    if arg.strip() == '':
        return ()
    try:
        return eval(arg)
    except:
        pass
    args = arg.split(':')
    args = tuple(int(x) if x != '' else None for x in args)
    return slice(*args)

def str2attr(a):
    # there is some danger dissolving tuples that may be array indices
    if isinstance(a, tuple):
        return reduce(__add__, (str2attr(i) if isinstance(i, (str,)) else (i,) for i in a), ())
    if not isinstance(a, str):
        return a
    a = a.split('.')
    res = ()
    # extract indices that were strings
    for x in a:
        ii = _re_index.split(x)
        if len(ii) == 1:
            res += (ii[0],)
        else:
            if len(ii[0]) == 0:
                raise ValueError()
            res += (ii[0],)
            for i in ii[1:]:
                if len(i.strip()) > 0:
                    raise ValueError()
            ii =  _re_index.findall(x)
            for i in ii:
                res += (slice_index(i[1:-1]),)
    # normalise by removing empty ()?
    res = tuple(i for i in res if i != ())
    return res


def joinattr(a, b):
    if isinstance(a, str):
        a = (a,)
    else:
        a = tuple(a)
    if isinstance(b, str):
        b = (b,)
    else:
        b = tuple(b)
    return a + b

class _Action:
    CALL = 'call'
    CLIENTCALL = 'ccall'
    START = 'start'
    ASYNC = 'async'
    GET = 'get'
    DATA = 'data'
    SET = 'set'
    TYPE = 'type'
    EXECUTE = 'execute'
    STATUS = 'status'
    REPLY = 'reply'
    CLIENTREPLY = 'creply'
    OUTPUT = 'output'
    STOP = 'stop'
    DONE = 'Done.'
    META = 'meta'
    GETSTARTUP = 'startup'
    CYCLE = 'cycle'
    PING = 'ping'
    PINGREPLY = 'pingreply'
    WAIT = 'wait'
    BUSY = 'busy'
    LOAD = 'load'
    START = 'start'

class State:
    RUNNING = 'running'
    STOPPED = 'stopped'
    REFUSED = 'refused'
    CONNECT = 'connected'

class Party:
    CLIENT = 'client'
    SERVER = 'server'

# currently only used for debug/tracking
_id = 0
def _id_factory():
    # add MAC or use UUID(non-debug)
    global _id
    _id += 1
    task_id = getpid() * 2**16 + _id
    return f'{task_id:08X}'

# we use all classes, we could get rid of 'action', at least make internal.
class Task(dict):
    _slots = ()
    _require = ()
    _defaults = {}
    _require_types = {}
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        if attr in self._defaults:
            return copy(self._defaults[attr])
        raise AttributeError(attr)
    def __setattr__(self, attr, value):
        if attr in self._slots:
            return super().__setattr__(attr, value)
        self[attr] = value
    def __init__(self, *args, **kwargs):
        assert isinstance(self._require, tuple), 'require tuple for "_require".'
        for r in self._require:
            if not r in kwargs:
                kwargs[r] = args[0]
                args = args[1:]
            else:
                break
        if len(args) > 0:
            raise TypeError('unexpected arguments')
        if hasattr(self, '_action'):
            assert 'action' not in kwargs
            kwargs['action'] = self._action
        if not 'action' in kwargs:
            kwargs['action'] = getattr(_Action, self.__class__.__name__.upper())
        for r in ('action',) + self._require:
            assert r in kwargs, f' [{self.__class__.__name__}] require "{r}"'
        super().__init__(**kwargs)
        for k,v in self._require_types.items():
            if k in self:
                assert isinstance(self[k], v)
        self.id = (_id_factory(),)
        self.validate()
    def validate(self):
        pass
    def __str__(self):
        s = ', '.join((f'{k} = {v!r}' for k,v in self.items() if k != 'action'))
        return f'{self.action.capitalize()}({s})'
    @property
    def noreply(self):
        return self.get('noreply', False)
    @property
    def doreply(self):
        return not self.noreply

class AttrTask(Task):
    _require = ('attr',)
    def validate(self):
        if not isinstance(self.attr, tuple):
            self.attr = str2attr(self.attr)

class BaseCall(AttrTask):
    _defaults = dict(
        args = (),
        kwargs = {},
        )
class Call(BaseCall):
    pass
class ClientCall(BaseCall):
     pass

class Get(AttrTask):
    pass

class Data(Get):
    pass

class Type(AttrTask):
    pass

class Set(AttrTask):
    _require = ('attr', 'value')

# class Reply(Task):
#     _require = ('data', )
#     _defaults = dict(data = None)

# maybe do this generally to copy only ID of tasks ...
class Reply(Task):
    _require = ('data', 'task')
    _defaults = dict(data = None)
    _require_types = dict(task = Task)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = self.id + self.task.id
        del self['task']

class Execute(Task):
    _require = ('cmd',)
    _require_types = dict(cmd = str)

class Cycle(Task):
    _require_types = dict(n = int)
    _defaults = dict(n = 1)

class Start(Task):
    _require = ('base', 'generator', 'parms')
    _require_types = dict(base = str, generator = str)
    _defaults = dict(parms = ())

class Load(Task):
    pass

class Status(Task):
    pass

class Stop(Task):
    pass

class NoReply(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noreply = True

class Output(NoReply):
    _require = ('data',)
    _require_types = dict(data = str)
    _defaults = dict(data = '')
    # have referebce ID?
class Done(Output):
    _require = ()
    _require_types = dict()
    _defaults = dict()
class Wait(Done):
    pass

class GetStartup(Task):
    pass

class Ping(Task):
    pass

class PingReply(Task):
    pass

# we may not want to implement/use the following

class Async(Call):
    def validate(self):
        super().validate()
        assert self.noreply == True
        self['noreply'] = True
class Meta(Task):
    _require = ('meta', 'task',)

# _done_tasks = (Cycle, Execute, Start, )

# def task_requires_done(task):
#     return isinstance(task, _done_tasks)
