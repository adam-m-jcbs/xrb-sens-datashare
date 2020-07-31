from human import byte2human

from pickle import dumps, loads, PickleError

from contextlib import redirect_stdout
from sys import __stdout__, __stderr__

from queue import Empty, Full

# connection interaface

class ConnectionError(Exception):
    pass
class NoData(ConnectionError):
    pass
class NoSpace(ConnectionError):
    pass
class TooBig(ConnectionError):
    pass
class NoConnection(ConnectionError):
    pass

class ConnectionWrapper():
    """
    abstract class.

    derived class needs to implement at least
    _get - return NoData if channle is empty
    _put - return NoSpace if data is too big
    """
    def __init__(self, inchannel, outchannel, debug = False, name = None):
        self._input = inchannel
        self._output = outchannel
        self._debug = debug
        self._name = name
        self._header = f'   [{self.__class__.__name__}]'
        if name is not None:
            self._header += f' [{name.strip()}]'
        self._total__sent_bytes = 0
        self._total__recv_bytes = 0
        self._total__sent_max = 0
        self._total__recv_max = 0
        self._recent_sent_bytes = 0
        self._recent_recv_bytes = 0
        self._recent_sent_max = 0
        self._recent_recv_max = 0
    def get(self, *args, **kwargs):
        # need to treat load errors?
        with redirect_stdout(__stderr__):
            data = self._get(*args, **kwargs)
            if self._debug:
                print(self._header + f' received {byte2human(len(data))}.')
            size = len(data)
            self._total__recv_bytes += size
            self._total__recv_max = max(self._total__recv_max, size)
            self._recent_recv_bytes += size
            self._recent_recv_max = max(self._recent_recv_max, size)
            return loads(data)
    def put(self, obj, *args, **kwargs):
        with redirect_stdout(__stderr__):
            try:
                data = dumps(obj)
            except TypeError as error:
                if self._debug:
                    print(self._header + f' Error Pickling {obj!r}.')
                raise PickleError from error
            if self._debug:
                print(self._header + f' sending  {byte2human(len(data))}.')
            size = len(data)
            self._total__sent_bytes += size
            self._total__sent_max = max(self._total__sent_max, size)
            self._recent_sent_bytes += size
            self._recent_sent_max = max(self._recent_sent_max, size)
            return self._put(data, *args, **kwargs)
    def close(self):
        self._input.close()
        self._output.close()
    def join(self, timeout):
        if hasattr(self._output, 'join'):
            self._output.join(timeout)
    def _get(self, *args, **kwargs):
        raise NotImplementedError
    def _put(self, *args, **kwargs):
        raise NotImplementedError
    def oempty(self):
        raise NotImplementedError()
    def iempty(self):
        raise NotImplementedError()
    def reply(self, *args, **kwargs):
        return self.put(Reply(*args, **kwargs))
    def get_stats(self):
        return self._sent_bytes, self._recv_bytes
    def clear_stats(self, all = False):
        self._recent_sent_bytes = 0
        self._recent_recv_bytes = 0
        self._recent_sent_max = 0
        self._recent_recv_max = 0
        if all:
            self._total__sent_bytes = 0
            self._total__recv_bytes = 0
            self._total__sent_max = 0
            self._total__recv_max = 0
    def print_stats(self):
        print()
        print(self._header + f' recent sent     {byte2human(self._recent_sent_bytes):>7s},    max {byte2human(self._recent_sent_max  ):>7s}.')
        print(self._header + f' recent received {byte2human(self._recent_recv_bytes):>7s},    max {byte2human(self._recent_recv_max  ):>7s}.')
        print(self._header + f' total  sent     {byte2human(self._total__sent_bytes):>7s},    max {byte2human(self._total__sent_max  ):>7s}.')
        print(self._header + f' total  received {byte2human(self._total__recv_bytes):>7s},    max {byte2human(self._total__recv_max  ):>7s}.')

class QueueConnection(ConnectionWrapper):
    def task_done(self):
        """
        depending on queue type that may not be available
        """
        if hasattr(self._input, "task_done"):
            self._input.task_done()
    def _get(self, block = True, timeout = None):
        try:
            return self._input.get(block, timeout)
        except Empty as error:
            raise NoData from error
    def _put(self, obj, block = True, timeout = None):
        try:
            return self._output.put(obj, block, timeout)
        except Full as error:
            raise NoSpace from error
    def iempty(self):
        return self._input.empty()
    def oempty(self):
        return self._output.empty()

class PipeConnection(ConnectionWrapper):
    """
    may implement own queue-like interface on top, which may include
    dealing with sending too large objects (32MB+) that may raise
    ValueError.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_counter = 0
    def task_done(self):
        self._task_counter -= 1
    def _get(self, block = True, timeout = None):
        if block == False:
            if not self._input.poll():
                raise NoData()
        else:
            try:
                if not self._input.poll(timeout):
                    raise NoData()
            except OSError as error:
                raise NoConnection() from error
        try:
            data = self._input.recv_bytes()
            self._task_counter += 1
            return data
        except EOFError as error:
            raise NoData() from error
    def _put(self, obj, block = True, timeout = None):
        assert block, 'Only support blocking mode'
        try:
            return self._output.send_bytes(obj)
        except ValueError as error:
            raise TooBig() from error
    def iempty(self):
        return self._input.poll()

def getconnection(connection, name = None, debug = False):
    if connection[0] == 'queue':
        return QueueConnection(
            *(connection[1:]),
            debug = debug,
            name = name,
            )
    elif connection[0] == 'pipe':
        return PipeConnection(
            *(connection[1:]),
            debug = debug,
            name = name,
            )
    else:
        raise Exception(f'Invalid Connection Type "{connection[0]}".')

def setupconnection(context, connection):
    if connection == 'queue':
        task_queue = context.JoinableQueue()
        repl_queue = context.JoinableQueue()
        server_con = ('queue', task_queue, repl_queue)
        client_con = ('queue', repl_queue, task_queue)
    elif connection == 'pipe':
        get1, put1 = context.Pipe(False)
        get2, put2 = context.Pipe(False)
        server_con = ('pipe', get1, put2)
        client_con = ('pipe', get2, put1)
    else:
        raise Exception('No valid connection type specified {connection}')
    return client_con, server_con
