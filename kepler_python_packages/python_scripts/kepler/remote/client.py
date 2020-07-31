"""
Client for remote Kepler server
"""

from multiprocessing import get_context

from ..process.client import ProxyTemplate
from ..process.connection import setupconnection, getconnection

from ..process.api import Status, Stop, Reply, Done, GetStartup, Cycle, Execute, \
     Call
from ..process.api import State

from ..hooks.framework_remote import  RemoteClientCycleHooks

from .defaults import *

from .tunnel import Tunnel

from .queueconnection import QueueConnectionClient


class RemoteProxy(ProxyTemplate, RemoteClientCycleHooks):
    """
    set up and manage ssh connection if need be

    connect to remote server

    set up and manage communication queue to to remote

    run 'g' and 's' local or remote?

    accept and manage old Output fro new cnnection

    timeout remote connection?
    """
    def setup_process(
            self,
            params,
            connection = 'pipe',
            context = None,
            name = 'Kepler Client',
            host = '',
            port = _default_port,
            daemon = True,
            ):
        self._port = port
        self._host = host
        self._ctx = get_context(context)
        client_con, server_con = setupconnection(self._ctx, connection)
        self._connection = getconnection(client_con, name = 'client', debug = self._debug)
        self._daemon = daemon
        server_con = getconnection(server_con, name = 'server', debug = self._debug)
        self._remote = self._ctx.Process(
            target = QueueConnectionClient,
            kwargs = dict(
                host = self._host,
                port = self._port,
                connection = server_con,
                name = self._name,
                debug = self._debug,
                ),
            name = name,
            daemon = self._daemon,
            )
        self._remote.start()
        # self.test_server()
        print(f' [CLIENT] Kepler Client started for {self._port}:{self._host}:{self._rport} using {self._ctx._name!r} and {client_con[0]!r}.')

    def test_server(self):
        # TODO - communicate with server to establish/test connection
        self._connection.put(Ping())
        try:
            reply = self.connection.get()
        # not sure this is useful, the child process should run, the
        # server, however, may not.
        except NoData:
            self.status = Status.REFUSED
        # TODO - weave in transparently with remote.client

    def capture_startup(self):
        if self._debug:
            print(' [RemoteProxy] Requestion startup message.')
        task = GetStartup()
        data = self.servercall(task)
        if data is None:
            data = list()
            if self._debug:
                print(' [RemoteProxy] WARNING - Received NO startup message.')
        self._startup = data

    def __init__(self, *args, **kwargs):
        host = kwargs.get('host', _default_host)
        port = None
        rport = None
        if host.count(':') == 0:
            pass
        elif host.count(':') == 1:
            host, port = host.split(':')
            port = int(port)
        elif host.count(':') == 2:
            port, host, rport = host.split(':')
            port = int(port)
            rport = int(rport)
        else:
            raise Exception('Unknown host format.')
        port = kwargs.get('port', port)
        rport = kwargs.get('rport', rport)
        if port is None:
            if rport is not None:
                port = rport
            else:
                port = _default_port
        if rport is None:
            rport = port
        self._tunnel = Tunnel(host, port, rport)
        # remove these from passing on, just set here
        kwargs['host'] = host
        kwargs['port'] = port
        self._rport = rport
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    pass
