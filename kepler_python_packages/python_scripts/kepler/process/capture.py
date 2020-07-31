from multiprocessing import get_context, Process
from os import read, write, dup, dup2, pipe, linesep, getpid
from threading import Thread
from uuid import uuid4
from sys import stdout
from numpy.random import randint

from .api import Output, Done, Wait, _done_token, _ioid_token
from .connection import getconnection

class OutputStream():
    def __init__(self, connection, name, debug = False):
        self._connection = connection
        self._name = name
        self._buffer = ''
        self._debug = debug
        self._ioid = None
        self._stack = []
    def _send_data(self, data):
        if len(data) > 0:
            if data == _done_token:
                task = Done(ioid = self._ioid)
                self._ioid = self._stack.pop()
            elif data.startswith(_ioid_token):
                self._stack.append(self._ioid)
                self._ioid = data[len(_ioid_token):]
                return
            else:
                task = Output(
                    data = data,
                    name = self._name,
                    ioid = self._ioid,
                    )
            self._connection.put(task)
    def _send_buffer(self):
        if len(self._buffer) > 0:
            self._send_data(self._buffer)
            self._buffer = ''
    def write(self, data):
        self._buffer += data
        while True:
            i = self._buffer.find(linesep)
            if i >= 0:
                self._send_data(self._buffer[:i])
                self._buffer = self._buffer[i + len(linesep):]
            else:
                break
    def flush(self):
            self._send_buffer()

class StreamCapture():
    def __init__(self, fd, stream, sentinel, debug = False, run = True):
        self._debug = debug
        self._sentinel = sentinel
        self._stream = stream
        self._fd = fd
        if run:
            self.run()

    def run(self):
        while True:
            data = read(self._fd, 2**24)
            done = False
            while True:
                i = data.find(self._sentinel)
                if i >= 0:
                    data = data[:i] + data[i + len(self._sentinel):]
                    done = True
                else:
                    break
            self._stream.write(data.decode())
            if done:
                break
        self._stream.flush()

    def __call__(self):
        self.run()

class StreamRedirectorContext():
    def __init__(
            self,
            connection,
            name,
            debug = False,
            mode = 'process'):
        self._debug = debug
        self._name = name
        if isinstance(connection, tuple):
            connection = getconnection(connection, name, debug)
        self._connection = connection
        self._stream = OutputStream(connection, name, debug)
        self._readfd, self._writefd = pipe()
        self._sentinel = uuid4().hex.encode()
        if self._debug:
            print('   [StreamRedirector] Setting up redirect.')
        if mode == 'thread':
            self._capture = Thread(
                target = StreamCapture,
                args = (
                    self._read,
                    self._stream,
                    self._sentinel,
                    ),
                kwargs = dict(
                    debug = self._debug,
                    ),
                )
        elif mode == 'process':
            ctx = get_context('fork')
            self._capture = ctx.Process(
                target = StreamCapture,
                args = (
                    self._readfd,
                    self._stream,
                    self._sentinel,
                    ),
                kwargs = dict(
                    debug = self._debug,
                    ),
                daemon = True,
                )
        self._capture.start()
        self._stack = []
        self._orgfd = None
    def __enter__(self, ioid = None):
        # create IOID if None, return?
        if ioid is None:
            ioid = f'{randint(2**16):04X}'
        self._connection.put(Wait(ioid = ioid))
        if self._debug:
            print('   [StreamRedirector] Starting up redirect.')
        if not hasattr(self, '_org'):
            self._org = stdout
            self._orgfn = self._org.fileno()
        self._stack.append(self._orgfd)
        self._orgfd = dup(self._orgfn)
        dup2(self._writefd, self._orgfn)
        print(f'{_ioid_token}{ioid}')
    def __exit__(self, *args):
        print(_done_token)
        dup2(self._orgfd, self._orgfn)
        self._orgfd = self._stack.pop()
        if self._debug:
            print('   [StreamRedirector] Done redirect.')
        if isinstance(self._capture, Thread):
            self._stream.flush()
    def close(self, timeout = 0.001, tries = 100):
        count = 0
        own_capture = isinstance(self._capture, Thread)
        while True:
            write(self._writefd, self._sentinel)
            if isinstance(self._capture, Thread) or (
                    isinstance(self._capture, Process) and
                    (getpid == self._capture._parent_pid)):
                self._capture.join(timeout)
                if self._capture.is_alive():
                    if self._debug:
                        print('   [StreamRedirector] waiting ...')
            else:
                break
            if count > tries:
                break
            count += 1
        if isinstance(self._capture, Process):
            if getpid == self._capture._parent_pid:
                if self._capture.is_alive():
                    print('   [StreamRedirector] killing ...')
                    self._capture.terminate()
