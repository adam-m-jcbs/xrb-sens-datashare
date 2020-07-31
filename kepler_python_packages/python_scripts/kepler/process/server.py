from os import getpid
from sys import exc_info
from traceback import format_tb

from sys import __stderr__
from contextlib import redirect_stdout

from numpy import ndarray, shape

from types import MethodType, ModuleType

from ..code import Kepler

from .exception import *
from .api import joinattr, str2attr
from .api import Task, Get, Set, Call, Stop, Execute, Data, Status, Type, Reply, \
     Output, Cycle, ClientCall, Start, Load
from .api import State

from .connection import getconnection
from .connection import NoData, NoSpace, TooBig, PickleError

from .capture import StreamRedirectorContext
from contextlib import nullcontext

class PlotManagerProxy():
    _slots = (
        '_name',
        '_process',
        '_debug',
        '_kwargs',
        )
    def __init__(self, process, name = 'kp', debug = False, **kwargs):
        self._process = process
        self._name = 'kp'
        self._debug = debug
        self._kwargs = kwargs

    # TODO - introduce proxies as needed
    def closewin(self, *args, **kwargs):
        task = ClientCall(
            joinattr(self._name, 'closewin'),
            args = args,
            kwargs = kwargs,
            )
        self._process.clientpost(task)
    def plot(self, *args, **kwargs):
        kwargs.setdefault('interactive', False)
        task = ClientCall(
            joinattr(self._name, 'plot'),
            args = args,
            kwargs = kwargs,
            )
        self._process.clientpost(task)
    def update(self, *args, **kwargs):
        kwargs.setdefault('interactive', False)
        task = ClientCall(
            joinattr({self._name}, 'plot'),
            args = args,
            kwargs = kwargs,
            )
        self._process.clientpost(task)
    # the following is experimental
    def __getattr__(self, attr):
        task = Get(joinattr({self._name},attr))
        value = self._process.clientcall(task)
        if isinstance(value, exception):
            print(f'  [{self.__calss__.__name__}] GET exception {value}')
        return value
    def __setattr__(self, attr, value):
        if attr in self._slots:
            return super().__setattr__(attr, value)
        task = Set(
            joinattr(self._name, attr),
            value = value,
            )
        self._process.clientpost(task)

class KepProcess():
    def __init__(self, connection, params, redirect = None):
        """
        Start Kepler in process.

        connection is tuple with name followed by data
        """
        print(f'  [KepProcess] PID = {getpid()}')
        args = params.get('args', ())
        kwargs = params.get('kwargs', {})
        self._debug = kwargs.setdefault('debug', False)
        if isinstance(connection, tuple):
            connection = getconnection(connection, name = 'server', debug = self._debug)
        self._connection = connection
        self._cache_np = kwargs.pop('cache_np', False)
        self._cache_np0 = kwargs.pop('cache_np0', False)
        self._base = str2attr(kwargs.pop('base', 'kepler'))
        kwargs.setdefault('plot', False)
        kwargs['server'] = self
        self.kp = PlotManagerProxy(self, name = 'kp')

        self._noredirect = nullcontext()
        if isinstance(redirect, StreamRedirectorContext):
            self._redirect = redirect
        elif redirect == True:
            self._redirect = StreamRedirectorContext(self._connection, 'stdout', debug = self._debug)
        else:
            self._redirect = self._noredirect

        kwargs['remote'] = True
        self._kwargs = kwargs
        self._args = args

        self.stop = False
        self.run()

    def _dbg(self, *args, **kwargs):
        if self._debug:
            with redirect_stdout(__stderr__):
                print(*args, **kwargs)

    def load(self):
        self.kepler = Kepler(*self._args, **self._kwargs)
        self.FortranType = type(self.kepler._kepler.vars)

    def run(self):
        while True:
            task = self._connection.get()
            self._dbg(f'  [server.run] GET {task}')
            result = self.handle(task)
            if self.stop:
                break
            # this should never happen
            if result is not None:
                print(f'  [server.run] unexpected result {result} from task {task}.')
        if self._redirect != self._noredirect:
            self._dbg(f'  [server.run] need cleaning up redirect...')
            self._redirect.close()

    def handle(self, task):
        self._dbg(f'  [server.handle] GET {task}')
        if isinstance(task, Stop):
            self._connection.task_done()
            self.stop = True
            return
        if isinstance(task, Reply):
            print(f'  [server.handle] unexpected REPLY {task}')
            self._connection.task_done()
            return
        doreply = task.doreply
        try:
            result = self.process(task)
            self._dbg(f'  [server.handle] RESULT from process(ing): {type(result)}({result})')
            if doreply:
                reply = Reply(result, task = task)
                self._dbg(f'  [server.handle] sending reply {reply}.')
                try:
                    self._connection.put(reply)
                except (PickleError, ):
                    reply = Reply(RemoteComplexObjectError(repr(result)),
                                  task = task)
                    self._dbg(f'  [server.handle] pickle errror {result}.')
                    self._connection.put(reply)
        except Exception as error:
            if self._debug:
                self._dbg(f'  [server.handle] ERROR {error!r}')
                exc_type, exc_value, exc_traceback = exc_info()
                for lines in format_tb(exc_traceback):
                    for line in lines.splitlines():
                        self._dbg('  [server.handle] ' + line)
            if doreply:
                reply = Reply(RemoteException(error),
                              task = task)
                self._connection.put(reply)
        self._connection.task_done()

    def clientcall(self, task):
        assert isinstance(task, ClientCall)
        _async_tasks = []
        self._dbg(f'  [server.clientcall] PUT {task}')
        self._connection.put(task)
        while True:
            # TODO - assure there is no CYCLE calls
            atask = self._connection.get()
            self._dbg(f'  [server.clientcall] RECEIVE {atask}')
            if isinstance(atask, Stop):
                self._connection.task_done()
                self.stop = True
                return
            if isinstance(atask, Reply):
                self._connection.task_done()
                result = atask.data
                break
            if atask.noreply:
                _async_tasks.append(atask)
                self._dbg(f'  [server.clientcall] storing for later {atask}')
                continue
            self.handle(atask)
        for atask in _async_tasks:
            self._dbg(f'  [server.clientcall] handling async task {atask}')
            self.handle(atask)
        return result

    def clientpost(self, task):
        assert isinstance(task, ClientCall)
        task['noreply'] = True
        self._dbg(f'  [server.clientpost] PUT {task}')
        self._connection.put(task)

    # here we need extra classes,
    # - things that would be good for sync,
    #   such as plot, but do not need a reply *value*, hence can be
    #   abandonned by client (or remote connection infrastructure)
    #   * this may be subcase if clientcall, maybe just an extra flag
    #     in the ClientCall stating whether data is needed or just
    #     acknowledgement
    # - data being pushed that should be buffered by infrastructure?
    #   for example from local server hooks?

    def process(self, task):
        self._dbg(f'  [server.process] GET {task}')
        if isinstance(task, Load):
            with self._redirect:
                self.load()
            return None
        base = task.get('base', self._base)
        obj = self
        for a in base:
            obj = getattr(obj, a)
        if isinstance(task, Call):
            attr = task.get('attr', None)
            if attr is None:
                return RemoteMissingAttributeError()
            attrs = attr
            try:
                for a in attrs:
                    if isinstance(a, str):
                        obj = getattr(obj, a)
                    else:
                        obj = obj[a]
            except:
                return RemoteAttributeError(attr)
            args = task.args
            kwargs = task.kwargs
            try:
                if base[0] == 'kepler':
                    redirect = self._redirect
                else:
                    redirect = self._noredirect
                with redirect:
                    result = obj(*args, **kwargs)
            except Exception as error:
                return RemoteExecutionError(attr, args, kwargs, error)
            return result
        elif isinstance(task, (Get, Type, Data)):
            attr = task.get('attr', None)
            if attr is None:
                return RemoteMissingAttributeError()
            attrs = attr
            cache_np = (not isinstance(task, Data)) or self._cache_np
            cache_np = task.get('cache_np', cache_np)
            cache_np0 = (not isinstance(task, Data)) and self._cache_np0
            cache_np0 = task.get('cache_np0', cache_np0)
            try:
                for a in attrs:
                    if isinstance(a, str):
                        if a == '__':
                            cache_np = False
                            continue
                        if not a.startswith('__'):
                            if a.endswith('__'):
                                a = a[:-2]
                                cache_np = False
                                cache_np0 = False
                            elif a.endswith('_'):
                                a = a[:-1]
                                cache_np = True
                                cache_np0 = True
                        obj = getattr(obj, a)
                    else:
                        obj = obj[a]
            except:
                return RemoteAttributeError(attr)
            if isinstance(obj, MethodType):
                return RemoteMethodError(attr)
            index = task.get('index', None)
            if index is not None:
                try:
                    obj = obj[index]
                except:
                    return RemoteIndexError(index, attr)
                cache_np = False
            if isinstance(task, Type):
                return type(obj)
            elif cache_np and isinstance(obj, (ndarray, )):
                # caching creates proxy objects that can be used for assignment
                # to subscripts
                if shape(obj) == ():
                    if not cache_np0:
                        return obj[()]
                return RemoteNumpyArrayError(attr)
            else:
                return obj
        elif isinstance(task, Set):
            attr = task.get('attr', None)
            if attr is None:
                return RemoteMissingAttributeError()
            attrs = attr
            try:
                for a in attrs[:-1]:
                    if isinstance(a, str):
                        obj = getattr(obj, a)
                    else:
                        obj = obj[a]
            except:
                return RemoteAttributeError(attr)
            try:
                value = task.get('value')
            except:
                return RemoteMissingValueError()

            a = attrs[-1]
            if 'index' in task:
                index = task.get('index')
                try:
                    if isinstance(a, str):
                        obj = getattr(obj, a)
                    else:
                        obj = obj[a]
                except AttributeError:
                    return RemoteAttributeError(attr)
                try:
                    obj[index] = value
                except IndexError:
                    return RemoteIndexError(index, attr)
                except Exception as error:
                    return RemoteException(error)
                return None
            else:
                try:
                    if isinstance(a, str):
                        return setattr(obj, a, value)
                    else:
                        obj[a] = value
                        return
                except AttributeError:
                    return RemoteAttributeError(attr)
                except Exception as error:
                    return RemoteException(error)
        elif isinstance(task, Execute):
            cmd = task.get('cmd', None) # set default in API
            if cmd is None:
                raise RemoteMissingCommandError()
            with self._redirect:
                for c in cmd.splitlines(): # split by comma as well?
                    self.kepler.execute(c)
            return
        elif isinstance(task, Status):
            if self.stop:
                return State.STOPPED
            return State.RUNNING
        elif isinstance(task, Cycle):
            with self._redirect:
                self.kepler.cycle(task.n)
            return

        return RemoteActionError(task)

    # ============================================================
    # define and set callbacks
    # ============================================================
    #
    # need to add interfaces for KEPLER to request data
    #
    def endkepler(self, code = None):
        self._dbg('  [server.endkepler] remote end callback')
        task = ClientCall(
            'terminate',
            kwargs = dict(code = code),
            noreply = True,
            )
        self.clientpost(task)

    def plotkepler(self):
        self._dbg('  [server.plotkepler] remote plot callback')
        task = ClientCall('plot',
            # maybe these need to be adjusted?
            kwargs = dict(
                itvstart = self.kepler.kd.parm.itvstart,
                ipixtype = self.kepler.kd.parm.ipixtype,
                duplicates = False,
                interactive = False,
                ),
            )
        self.clientcall(task)

    def ttykepler(self, msg):
        self._dbg('  [server.ttykepler] remote input callback')
        task = ClientCall('ttygets')
        s = self.clientcall(task)
        self._dbg(f'  [server.ttykepler] result: {s!r}')

        msg[:] = 0
        for i, c in enumerate(s):
             msg[i] = bytes(c, encoding = 'us-ascii')[0]

    def _set_callbacks(self, kepler):
        kepler._kepler.ttykepler = self.ttykepler
        kepler._kepler.endkepler  = self.endkepler
        kepler._kepler.plotkepler = self.plotkepler
