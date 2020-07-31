from sys import maxsize, exc_info
from traceback import format_tb
from multiprocessing import get_context
from time import sleep

from contextlib import contextmanager

from numpy import ndarray

from ..plot import plot
from ..plot.manager import PlotManager

from ..datainterface import DataInterface
from ..cmdloop import KeplerCmdLoop

from .exception import *
from ..exception import *
from .server import KepProcess
from .api import joinattr, str2attr
from .api import State
from .api import Task, Get, Set, Call, Stop, Execute, Data, Status, Type, \
     Reply, Output, Done, Cycle, Wait, ClientCall, Start, Load

from .connection import getconnection, setupconnection
from .connection import NoData, NoSpace, TooBig, PickleError, NoConnection

from .capture import StreamRedirectorContext

from ..code.util import gets
from ..hooks.framework_client import ClientCycleHooks

import atexit

ipythonfuck = [
    '_repr_mimebundle_',
    '_ipython_canary_method_should_not_exist_',
    ]

class ProxyTemplate(ClientCycleHooks):
    _slots = (
        '_remote',
        '_connection',
        'kp',
        'kd',
        'status',
        '_startup',
        '_debug',
        '_cache',
        '_sentinel',
        '_done',
        '_reply',
        '_redirect',
        '_capture',
        '_daemon',
        '_ctx',
        '_port',
        '_rport',
        '_host',
        '_name',
        '_output_buf',
        '_tunnel',
        '_stack',
        '_pre_cycle_hooks',
        '_post_cycle_hooks',
            )
    _setup_parms = (
        'connection',
        'context',
        'name',
        'host',
        'port',
        'daemon',
        )
    def __init__(self, *args, **kwargs):
        self._debug = kwargs.setdefault('debug', False)
        self._cache = CacheProxy(debug = self._debug)
        self._redirect = kwargs.setdefault('redirect', True)
        self._name = kwargs.get('name', None)
        self._output_buf = None
        setup_parms = dict()
        for i in self._setup_parms:
            if i in kwargs:
                setup_parms[i] = kwargs.pop(i)
        params = dict(
            args = args,
            kwargs = kwargs,
            )
        self._sentinel = kwargs.pop('sentinel', '+')
        self._stack = []
        self._reply = None
        self._done = None
        self._startup = []
        setup_parms['params'] = params
        self.setup_process(**setup_parms)
        self.capture_startup()
        self.clear_hooks()
        self.kd = ProxyDataInterface(self, debug = self._debug)
        self.kp = PlotManager(self.kd, debug = self._debug)
        atexit.register(self.terminate)
        self.printstartup()
        self.status = self.servercall(Status())
        print(f' [PROXY] Status: {self.status}')

    def capture_startup(self):
        # read startup message
        if self._debug:
            print(' [Client] Capturing startup message ...')
        self._output_buf = []
        self.servercall(Load())
        self._startup = self._output_buf
        self._output_buf = None

    def startup(self):
        lines = []
        for task in self._startup:
            for line in task.data.splitlines():
                lines.append(line)
        return '\n'.join(lines)

    def printstartup(self):
        for line in self.startup().splitlines():
            self.oprint(line)

    def setup_process(self, *args, **kwargs):
        """
        Needs to set up:

          self._remote
          self._connection
        """
        raise NotImplementedError(f'Need to implement {__qualname__}')

    def __call__(self, attr, *args, **kwargs):
        return self.call(attr, *args, **kwargs)

    def oprint(self, line):
        if line.startswith(' '):
            line = line[1:]
        print(f'{self._sentinel}{line}')

    def process(self, task):
        """
        Process remote requests.

        To be expanded.
        """
        if self._debug:
            print(f' [CLIENT.PROCESS] {task}')
        if isinstance(task, ClientCall):
            attr   = task.attr
            args   = task.args
            kwargs = task.kwargs
            obj = self
            for a in attr:
                obj = getattr(obj, a)
            return obj.__call__(*args, **kwargs)
        elif isinstance(task, Wait):
            self._done = task.ioid
            if self._debug:
                print(f' [CLIENT.PROCESS] Wait for {task.ioid}')
            return
        elif isinstance(task, Done):
            self._done = None
            if self._debug:
                print(f' [CLIENT.PROCESS] Done for {task.ioid}')
            return
        elif isinstance(task, Output):
            if self._output_buf is not None:
                self._output_buf.append(task)
                return
            for line in task.data.splitlines():
                self.oprint(line)
            return
        raise NotImplementedError(task)

    def handle(self, task):
        if self._debug:
            print(f' [CLIENT.HANDLE] {task}')
        if isinstance(task, Reply):
            print(f' [CLIENT.HANDLE] unexpected REPLY {task}')
            return
        doreply = task.doreply
        try:
            result = self.process(task)
            if self._debug:
                print(f' [CLIENT.HANDLE] RESULT from process(ing): {type(result)}({result})')
            if doreply:
                reply = Reply(result, task = task)
                if self._debug:
                    print(f' [CLIENT.HANDLE] sending reply {reply}.')
                try:
                    self._connection.put(reply)
                except (PickleError, ):
                    if self._debug:
                        print(f' [CLIENT.HANDLE] Pickle Error {result}.')
                    reply = Reply(RemoteComplexObjectError(repr(result)),
                                  task = task)
                    self._connection.put(reply)
        except Exception as error:
            if self._debug:
                print(' [CLIENT.HANDLE] Error ', error)
                exc_type, exc_value, exc_traceback = exc_info()
                for lines in format_tb(exc_traceback):
                    for line in lines.splitlines():
                        print(' [CLIENT.HANDLE] ' + line)
            if doreply:
                reply = Reply(RemoteException(error),
                              task = task)
                self._connection.put(reply)

    @contextmanager
    def _callstack(self, task):
        self._stack.append((self._reply, self._done))
        self._done = None
        self._reply = task.id
        self._connection.put(task)
        yield
        self._reply, self._done = self._stack.pop()

    def servercall(self, task):
        _async_tasks = []
        if self._debug:
            print(f' [CLIENT.SERVERCALL] SEND {task}')
        with self._callstack(task):
            while True:
                reply = self._connection.get()
                if self._debug:
                    print(f' [CLIENT.SERVERCALL] received {reply}')
                if isinstance(reply, Reply):
                    result = reply['data']
                    if isinstance(result, RemoteException):
                        if self._debug:
                            print(f' [CLIENT.SERVERCALL] Error: {result}')
                    self._connection.task_done()
                    assert reply.id[-1] == task.id[0], reply
                    self._reply = None
                elif reply.noreply and not isinstance(reply, Output):
                    _async_tasks.append(reply)
                    if self._debug:
                        print(f' [CLIENT.SERVERCALL] Storing for later: {reply}')
                    continue
                else:
                    self.handle(reply)
                    self._connection.task_done()
                if self._done == None and self._reply == None:
                    break

        for reply in _async_tasks:
            if self._debug:
                print(f' [CLIENT.SERVERCALL] handling async task: {reply}')
            self.handle(reply)
            self._connection.task_done()
        return result

    def serverpost(self, task):
        assert task.noreply, task
        if self._debug:
            print(f' [CLIENT.SERVERPOST] PUT {task}')
        self._connection.put(task)

    post = serverpost

    def call(self, attr, *, args = (), kwargs = {}, **xkwargs):
        attr_ = str2attr(attr)
        task = Call(
            attr_,
            args = args,
            kwargs = kwargs,
            **xkwargs,
            )
        return self.servercall(task)

    def task(self, *args, **kwargs):
        task = Task(*args, **kwargs)
        return self.servercall(task)

    def terminate(self, code = None, timeout = None):
        # we may want to add debuggung capabilities - unlike fortran
        # executable, we may still examine stuff as the module data is
        # still available
        #
        # _remote here is the kepler process itself, need to adjust
        # for remote processes
        if not self._remote.is_alive():
            return
        # it is really the terminate function call that should be used!
        # result = self.servercall(Call('terminate'))
        self.serverpost(Stop(noreply=True))
        # print(f' [terminate] {result}')
        # this may be overkill
        self._connection.join(timeout)
        self._connection.close()
        self._remote.join(timeout)
        if self._remote.is_alive():
            self._remote.terminate()
        assert not self._remote.is_alive(), \
               f'Remote process {self._remote.pid} is still running.'
        print(f' [terminate] process {self._remote.name} with PID {self._remote.pid} has ended.')
        if self._capture is not None:
            self._capture.close()
        self.closewin()
        atexit.unregister(self.terminate)
        self.status = 'terminated'

    def __del__(self):
        try:
            self.terminate()
        except:
            pass

    def execute(self, cmd):
        task = Execute(cmd)
        self.servercall(task)

    def __getattr__(self, attr):
        if attr in ipythonfuck:
            raise AttributeError()
        attr_ = str2attr(attr)
        try:
            return self._cache[attr_]
        except (TypeError, KeyError):
            pass
        task = Get(attr_)
        value = self.servercall(task)
        if isinstance(value, (RemoteComplexObjectError, RemoteMethodError)):
            proxy = ProxyObject(attr_, self, debug = self._debug)
            self._cache[attr_] = proxy
            return proxy
        if isinstance(value, RemoteException):
            raise value
        return _adjust_ndarray(value)

    def __setattr__(self, attr, value):
        if attr in self._slots:
            return super().__setattr__(attr, value)
        attr_ = str2attr(attr)
        task = Set(attr_, value)
        value = self.servercall(task)
        if isinstance(value, RemoteException):
            raise value
        return value

    def remote_type(self, attr):
        attr_ = str2attr(attr)
        task = Type(attr_)
        value = self.servercall(task)
        if isinstance(value, RemoteException):
            raise value
        return value

    # ========================================
    # dictionary interface
    # ========================================
    def __delitem__(self, key):
        raise NotImplementedError('Deleting remote items not allowed.')

    def __missing__(self, key):
        raise NotImplementedError()

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)
    # ========================================


    def _cycle(self):
        task = Cycle()
        return self.servercall(task)

    def cycle(self, n = 1):
        """
        todo - asynchronous event loop
        """
        if self.status == 'terminated':
            raise KeplerNotRunning()
        for i in range(n):
            result = self._hook_cycle(cycler = self._cycle)
            if isinstance(result, RemoteException):
                raise result
            while True:
                x = gets()
                if x == 's':
                    break
                elif x != '':
                    x = str(x).strip()
                    self.execute(x)
                else:
                    break
            if x == 's':
                break
            if i == n-1 or self._kepler.gencom.mode == 0:
                break

        ncyc = self.kd.qparm.ncyc
        print(f' [CYCLE] Stop at cycle {ncyc:d}')

    def ttygets(self):
        """
        Interactive input from client.
        """
        return gets()

    # @property # broken autocompletion
    def s(self, n = 1):
        """
        Do a single step.  No need to call as function.
        """
        self.cycle(n)

    # @property # broken autocompletion
    def g(self):
        """
        Do run continuously ('go').
        """
        self.cycle(maxsize)

    def __del__(self):
        self.terminate()
        try:
            super().__del__()
        except:
            pass

    # these are identical with Kepler
    def plot(self, *args, **kwargs):
        self.kp.plot(*args, **kwargs)

    def update(self, *args, **kwargs):
        self.kp.update(*args, **kwargs)

    def closewin(self):
        self.kp.closewin()

    def pt(self, mode):
        try:
            return plot(self.kd, mode)
        except:
            pass

    # @property # broken autocompletion
    def run(self):
        """
        Interactive KEPLER mode

        TODO replace with something that allows plots to continue work.
        App?
        """
        if self.status == 'terminated':
            raise KeplerNotRunning()

        KeplerCmdLoop(self).cmdloop()

class LocalProxy(ProxyTemplate):
    def setup_process(
            self,
            params,
            connection = 'pipe',
            context = None,
            name = 'Kepler Parent',
            daemon = True,
            ):
        self._ctx = get_context(context)
        client_con, server_con = setupconnection(self._ctx, connection)
        self._connection = getconnection(client_con, name = 'client', debug = self._debug)
        self._daemon = daemon
        self._capture = None
        server_con = getconnection(server_con, name = 'server', debug = self._debug)
        redirect = self._redirect
        if self._redirect == True:
            if self._ctx._name == 'fork':
                # in this case the client may own the process and pass it on
                redirect = StreamRedirectorContext(server_con, 'stdout', debug = self._debug)
                self._capture = redirect
            self._daemon = False
        self._remote = self._ctx.Process(
            target = KepProcess,
            kwargs = dict(
                connection = server_con,
                params = params,
                redirect = redirect,
                ),
            name = name,
            daemon = self._daemon,
            )
        self._remote.start()
        print(f' [PROXY] Kepler started using {self._ctx._name!r} and {client_con[0]!r}.')

class ProxyObject():
    # slots are used to allow __setattr__ work for arbitraty other
    # attributes being passed on
    _slots = (
        '_name',
        '_proxy',
        '_cache',
        '_kwargs',
        '_debug',
        )
    _get_task = Get
    _set_task = Set
    _call_task = Call
    def __init__(self, name, proxy, **kwargs):
        self._name = str2attr(name)
        self._proxy = proxy
        self._kwargs = kwargs
        self._debug = kwargs.get('debug', False)
        self._cache = CacheProxy(debug = self._debug)
    def __getattr__(self, attr):
        if attr in ipythonfuck:
            raise AttributeError()
        attr_ = joinattr(self._name, attr)
        if self._get_task is None:
             raise AttributeError(f' Object "{self._name}" does not support attribute retrieval.')
        try:
            return self._cache[attr_]
        except (TypeError, KeyError):
            pass
        task = self._get_task(attr_, **self._kwargs)
        value = self._proxy.servercall(task)
        if isinstance(value, (RemoteComplexObjectError, RemoteMethodError)):
            proxy = self.__class__(attr_, self._proxy, **self._kwargs)
            self._cache[attr_] = proxy
            return proxy
        if isinstance(value, RemoteException):
            raise AttributeError from value
        return _adjust_ndarray(value)
    def _return(self, task):
        value = self._proxy.servercall(task)
        if isinstance(value, RemoteException):
            raise value
        return value
    def __setattr__(self, attr, value):
        if (attr in self._slots) or attr.startswith('_'):
            return super().__setattr__(attr, value)
        if self._set_task is None:
            raise AttributeError(f' Object "{self._name}" does not support item assignment.')
        attr_ = joinattr(self._name, attr)
        task = self._set_task(attr_, value, **self._kwargs)
        return self._return(task)
    def __getitem__(self, item):
        if self._get_task is None:
            raise AttributeError(f' Object "{self._name}" does not support retrieval.')
        attr_ = self._name
        task = self._get_task(attr_, index = item, **self._kwargs)
        value = self._proxy.servercall(task)
        if isinstance(value, RemoteException):
            raise value
        return _adjust_ndarray(value)
    def __setitem__(self, item, value):
        if self._set_task is None:
            raise AttributeError(f' Object "{self._name}" does not support item assignment.')
        attr_ = self._name
        task = self._set_task(attr_, value, index = item, **self._kwargs)
        return self._return(task)
    def __call__(self, *args, **kwargs):
        if self._call_task is None:
            AttributeError(f' Object "{self._name}" does not support calling.')
        attr_ = self._name
        task = self._call_task(attr_, args = args, kwargs = kwargs, **self._kwargs)
        return self._return(task)
    def get(self):
        if self._get_task is None:
            raise AttributeError(f' Object "{self._name}" does not support retrieval.')
        attr_ = self._name
        task = self._get_task(attr_, **self._kwargs)
        return self._proxy.servercall(task)
    def __len__(self):
        attr_ = joinattr(self._name, '__len__')
        task = Call(attr_, **self._kwargs)
        return self._return(task)
    def __repr__(self):
        return self.__class__.__name__ + f'({self._name})'
    def __str__(self):
        return self.__class__.__name__ + f'({self._name.__class__.__name__}{self._name})'

def _adjust_ndarray(value):
    """
    return write-protected version of ndarrays
    """
    # return self.__class__(self._name + (item,), self._proxy)
    # returning proxies for slices, though it actually works, causes issues with maptplotlib
    # it can be forced on the main proxy using, e.g., k['xm', slice(2,3)]
    if isinstance(value, ndarray):
        value.flags['WRITEABLE'] = False
    return value

class CacheProxy(dict):
    def __init__(self, debug = False, **kwargs):
        self._debug = debug
        super().__init__(**kwargs)
    def __setitem__(self, key, value):
        try:
            super().__setitem__(key, value)
            if self._debug:
                print(f'   [CacheProxy] setting dict[{key}] = {value}')
        except TypeError:
            if self._debug:
                print(f'   [CacheProxy] FAILED setting dict[{key}] = {value}')
    def __getitem__(self, key):
        if self._debug:
            try:
                value = super().__getitem__(key)
                print(f'   [CacheProxy] returning dict[{key}] --> {value}')
                return value
            except:
                raise
        else:
            return super().__getitem__(key)

class ProxyData(ProxyObject):
    _get_task = Data
    _set_task = None
    def __getitem__(self, attr):
        value = super().__getitem__(attr)
        return _adjust_ndarray(value)
    def __getattr__(self, attr):
        value = super().__getattr__(attr)
        return _adjust_ndarray(value)

class ProxyDataInterface(DataInterface):
    """
    Proxy object for KeplerData, used for plotting
    """
    _slots = (
        '_proxy',
        '_cache',
        '_kwargs',
        '_name',
        '_debug'
        )
    def __init__(self, proxy, name = 'kd', **kwargs):
        super().__init__()
        self._proxy = proxy
        self._kwargs = kwargs
        self._name = str2attr(name)
        self._debug = kwargs.get('debug', False)
        self._cache = CacheProxy(debug = self._debug)
    def __getattr__(self, attr):
        if attr in ipythonfuck:
            raise AttributeError()
        attr_ = joinattr(self._name, attr)
        try:
            return self._cache[attr_]
        except (TypeError, KeyError):
            pass
        task = Data(attr_, **self._kwargs)
        value =  self._proxy.servercall(task)
        if isinstance(value, (RemoteComplexObjectError, RemoteMethodError)):
            proxy = ProxyData(attr_, self._proxy, **self._kwargs)
            self._cache[attr_] = proxy
            return proxy
        if isinstance(value, RemoteAttributeError):
            raise AttributeError from value
        return _adjust_ndarray(value)
    def __setattr__(self, attr, value):
        if attr in self._slots:
            return super().__setattr__(attr, value)
        raise ValueError(f'{self.__class__.__name__} does not support assignment.')

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def set_cache_np(self, value):
        task = Set('_cache_np', value, base = ())
        self._proxy.serverpost(task)

    # ========================================
    # dictionary interface
    # ========================================
    def __delitem__(self, key, value):
        raise NotImplementedError('Deleting remote items not allowed.')

    def __missing__(self, key):
        raise NotImplementedError()

    def __getitem__(self, key):
        return self.__getattr__(str2attr(key))

    def __setitem__(self, key, value):
        return self.__setattr__(str2attr(key), value)
    # ========================================
