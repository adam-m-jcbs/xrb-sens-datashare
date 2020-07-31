
from multiprocessing import get_context
from multiprocessing.managers import BaseManager

from ..process.connection import getconnection

from ..process.api import Status, Stop, Reply, Done, GetStartup, Cycle, Execute, \
     Call

from .tunnel import Tunnel


class QueueConnectionClient():
    """
    run remote client

    provide feedback on messages, includig failures
    """
    def __init__(
            self,
            host = '',
            port = _default_port,
            connection = None,
            name = None,
            debug = True,
            context = None,
            authkey = _default_authkey,
            ):
        self._debug = debug
        self._name = name
        self._host = host
        self._port = port
        if isinstance(connection, tuple):
            connection = getconnection(connection, name, debug)
        self._connection = connection
        self._authkey = authkey

        self._ctx = get_context(context)
        class ConnectionManager(BaseManager): pass

        ConnectionManager.register('get_connection_type')
        ConnectionManager.register('get_connection_out')
        ConnectionManager.register('get_connection_in')

        self._manager = ConnectionManager(
            address=(host, self._port),
            authkey=self._authkey,
            )
        try:
            self._manager.connect()
        except ConnectionRefusedError as error:
            if self._debug:
                print(f' [ConnectionClient] error: {error!r}')
            self.status = State.REFUSED
            self.run_notconnected()
            return
        except OSError as error:
            if self._debug:
                print(f' [ConnectionClient] error: {error!r}')
            self.status = State.REFUSED
            self.run_notconnected()
            return

        self.status = State.CONNECT

        server_type = self._manager.get_connection_type().__str__().strip('\'')
        if self._debug:
            print(f' [ConnectionClient] connection type: {server_type}')
        server_con_out = self._manager.get_connection_out()
        server_con_in = self._manager.get_connection_in()
        server_con = (server_type, server_con_in, server_con_out,)
        if isinstance(server_con, tuple):
            server_con = getconnection(server_con, self._name, self._debug)
        self._server = server_con
        self._stop = False
        self._done = False # should become counter
        if self._debug:
            print(f' [ConnectionClient] connected to {self._host}:{self._port}.')
        self.run()

    def run_notconnected(self):
        if self._debug:
            print(f' [ConnectionClient] connection REFUSED.')
        while True:
            task = self._connection.get()
            if isinstance(task, Stop):
                if self._debug:
                    print(f' [ConnectionClient] Terminating.')
                return
            if not task.noreply:
                if isinstance(task, Status):
                    reply = Reply(self.status)
                elif task_requires_done(task):
                    reply = Done()
                elif task.doreply:
                    reply = Reply(None)
                else:
                    reply = None
                if self._debug:
                    print(f' [ConnectionClient] NotConnected proxy reply: {reply}.')
                if reply is not None:
                    self._connection.put(reply)

    def run(self):

        # manage communication?

        # run two indpendent processes or tasks to handle queues?
        #
        # maybe not needed.  Each transaction should always be terminated by
        # by a done;
        # exception would be 'ping'
        #
        # for now - need to reflect recursive nested protocols of process/client|server

        while True:
            self._done = False
            if self._debug:
                print(f' [ConnectionClient] Waiting for tasks from client.')
            task = self._connection.get()
            self.handle(task)
            if self._stop:
                break

        # proper shutdown procedure?
        # notifiy client - wait for acknowledgement?

        self._manager.shutdown()
        if self._debug:
            print(f' [ConnectionClient] Terminating.')

    def handle(self, task):
        # stop from client
        if isinstance(task, Stop):
            if self._debug:
                print(f' [ConnectionClient] Stop from client.')
            self._stop = True
            return
        result = self.process(task)
        if result is not None:
            raise Exception('Action TBD')
        if self._debug:
            print(f' [ConnectionClient.cc] forwarding to SERVER: {task}')
        self._server.put(task)
        if isinstance(task, Reply):
            return
        done = task_requires_done(task)
        if task.noreply and not done:
            return
        self._done |= done # XXXXXXXXX Here we get burnt by recursive level!
        while True:
            if self._debug:
                print(f' [ConnectionClient] Waiting for reply from SERVER.')
            reply = self._server.get()
            if isinstance(reply, Call): # replace by ClientCall
                result = self.handle_client_call(reply)
                # deal with results
            else:
                if self._debug:
                    print(f' [ConnectionClient] Forwarding reply to client: {reply}')
                self._connection.put(reply)

            if done:
                if isinstance(reply, Done):
                    break
            else:
                if isinstance(reply, Reply):
                    break
        # Stop from Server(!)
        if isinstance(reply, Stop):
            if self._debug:
                print(f' [ConnectionClient] Stop from SERVER.')
            self._stop = True

    def handle_client_call(self, reply):
        result = self.process_client_call(reply)
        if result is not None:
            raise Exception('Action TBD')
        if self._debug:
            print(f' [ConnectionClient] Forwarding reply to client: {reply}')
        self._connection.put(reply)
        if reply.noreply:
            return
        while True:
            if self._debug:
                print(f' [ConnectionClient.cc] waiting for reply from client.')
            task = self._connection.get()
            result = self.handle(task)
            # deal with results
            if isinstance(task, Reply):
                return

    def process(self, task):
        pass

    def process_client_call(self, reply):
        return


class QueueConnectionServer():
    """
    run remote server

    provide feedback on messages, includig failures
    """
    def __init__(
            self,
            port,
            connection,
            server = 'queue',
            name = None,
            debug = True,
            context = None,
            authkey = _default_authkey,
            ):
        self._debug = debug
        self._name = name
        self._port = port
        if isinstance(connection, tuple):
            connection = getconnection(connection, name, debug)
        self._connection = connection
        self._authkey = authkey

        # use context?
        self._ctx = get_context(context)
        qi = self._ctx.JoinableQueue()
        qo = self._ctx.JoinableQueue()
        self._server = getconnection(
            ('queue', qi, qo),
            self._name,
            self._debug,
            )

        class ConnectionManager(BaseManager): pass
        ConnectionManager.register(
            'get_connection_type',
            callable = lambda: 'queue',
            )
        ConnectionManager.register(
            'get_connection_out',
            callable = lambda: qi,
            )
        ConnectionManager.register(
            'get_connection_in',
            callable = lambda: qo,
            )
        self._manager = ConnectionManager(
            address = ('', self._port),
            authkey = self._authkey,
            )

        self._stop = False
        self._done = False # should become counter
        self.run()

    def run(self):
        if self._debug:
            print(f' [ConnectionServer] Starting manager.')
        self._manager.start()

        # manage communication?

        # run two indpendent processes or tasks to handle queues?
        # not sure we need to manage on level of transactions
        # but should use synchronisation tools such as semaphore.
        # use asyncio?

        # put client request handle in method to allow recursive stack
        while True:
            self._done = False
            # clear!!! (old output)
            if self._debug:
                print(f' [ConnectionServer] waiting for commands from CLIENT.')
            task = self._server.get()
            self.handle(task)
            if self._stop:
                break

        # proper shutdown procedure?
        # notifiy client - wait for acknowledgement?

        self._manager.shutdown()
        if self._debug:
            print(f' [ConnectionServer] Terminating.')


    def handle(self, task):
        # stop from client
        if isinstance(task, Stop):
            if self._debug:
                print(f' [ConnectionServer] Stop from client.')
            self._stop = True
            return
        result = self.process(task)
        if result is not None:
            raise Exception('Action TBD')
        if self._debug:
            print(f' [ConnectionServer] forwarding to KeplerServer: {task}')
        self._connection.put(task)
        done = task_requires_done(task)
        if task.noreply and not done:
            return
        self._done |= done # XXXXXXXXX Here we get burnt by recursive level!
        while True:
            if self._debug:
                print(f' [ConnectionServer.handle] waiting for reply from KeplerServer.')
            reply = self._connection.get()
            # cache asynchronous calls until end of current transactions?
            # done by client?
            if isinstance(reply, Call):
                result = self.handle_client_call(reply)
                # deal with results
            else:
                # buffer output, erase if full?
                if self._debug:
                    print(f' [ConnectionServer.handle] forwarding reply to CLIENT: {reply}')
                self._server.put(reply)
            if done:
                if isinstance(reply, Done):
                    break
            else:
                if isinstance(reply, Reply):
                    break
        # stop from Server(!)
        if isinstance(reply, Stop):
            if self._debug:
                print(f' [ConnectionServer.handle] Stop from KeplerServer.')
            self._stop = True

    def handle_client_call(self, reply):
        result = self.process_client_call(reply)
        if result is not None:
            raise Exception('Action TBD')
        if self._debug:
            print(f' [ConnectionServer.cc] forwarding reply to CLIENT: {reply}')
        self._server.put(reply)
        if reply.noreply:
            return
        while True:
            if self._debug:
                print(f' [ConnectionServer.cc] waiting for reply from CLIENT.')
            task = self._server.get()
            result = self.handle(task)
            # deal with results
            if isinstance(task, Reply):
                return

    def process(self, task):
        return

    def process_client_call(self, reply):
        return
