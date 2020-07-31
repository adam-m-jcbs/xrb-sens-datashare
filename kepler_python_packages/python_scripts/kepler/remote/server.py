"""
Remote Kepler server
"""

import atexit

from sys import maxsize
from sys import __stderr__
from sys import argv
from contextlib import redirect_stdout

from multiprocessing import get_context

from ..process.server import KepProcess
from ..code.main import KepParm

from  ..process.connection import setupconnection, getconnection
# from  ..process.connection import NoData, NoConnection

from ..process.api import Stop, Done, Output, GetStartup, Reply, Task, \
     Cycle, Execute, Call
# from ..process.api import task_requires_done

from .defaults import *

from .queueconnection import QueueConnectionServer


# use redirect_stdout for debug messages

class KeplerServer():
    """
    start KEPLER process

    from command line parameters [for now]

    set up and manage communication queue to process and to remote

    run 'g' and 's' modes based on remote commands

    buffer Output if connection is lost

    timeout remote connection?
    """
    def __init__(self, *args, **kwargs):
        # inherit from ..process.LocalProxy to set up connection to Kepler.
        # start communicator process

        self._debug = kwargs.get('debug', False)
        self._redirect = kwargs.get('redirect', True)
        connection = kwargs.get('connection', 'pipe')

        self._port = kwargs.pop('port', _default_port)
        if 's' in args:
            initial_action = 'start'
            args = list(args).remove('s')
        else:
            initial_action = 'run'

        # set up Kepler parameters
        kepparm = KepParm(*args, **kwargs)
        portfile = kepparm.portfile
        with open(portfile, 'wt') as f:
            f.write(f'{self._port}\n')
            # add authentification token
        kepkw = kepparm.kwargs()
        kepkw['debug'] = self._debug
        params = dict(kwargs = kepkw)

        # set up communication with kepler and start it
        context = kwargs.get('context', None)
        self._ctx = get_context(context)
        client_con, server_con = setupconnection(self._ctx, connection)
        self._connection = getconnection(client_con, name = 'KeplerServer', debug = self._debug)
        server_con = getconnection(server_con, name = 'KepProcess', debug = self._debug)

        self._remote = self._ctx.Process(
            target = KepProcess,
            kwargs = dict(
                connection = server_con,
                params = params,
                redirect = self._redirect,
                ),
            name = f'Server for :{self._port}',
            daemon = False,
            )
        self._remote.start()
        print(f' [KeplerServer] Kepler started using {self._ctx._name!r} and {client_con[0]!r}.')
        self.capture_startup()

        # set up communication with communication interface and start it
        if self._debug:
            print(f'  [KeplerServer.init] Setting up Server on :{self._port}')
        client_con, server_con = setupconnection(self._ctx, connection)
        self._outward = getconnection(client_con, name = 'outward', debug = self._debug)
        server_con = getconnection(server_con, name = f'inward', debug = self._debug)

        name = kwargs.get('name', f'Kepler:{self._port}')
        self._server = self._ctx.Process(
            target = ConnectionServer,
            kwargs = dict(
                port = self._port,
                connection = server_con,
                name = name,
                debug = self._debug
                ),
            name = name,
            daemon = False
            )
        self._server.start()
        atexit.register(self.terminate)
        self._stop = False
        self._done = False
        self.run()

    def terminate(self):
        self._remote.terminate()
        self._server.close()

    def capture_startup(self, done = True, wait = True, timeout = None):
        # maybe this should be more like 'clear'
        self._startup = []
        while True:
            reply = self._connection.get(wait, timeout)
            if isinstance(reply, Done):
                break
            elif isinstance(reply, Output):
                self._startup.append(reply)
            else:
                # deal with clientcalls instead
                raise Exception(f'  [KeplerServer.capture_startup] unexpected reply: {replt}')

    def capture_output(self, done = True):
        output = ()
        other = ()
        return


    # def clear(self, wait = False, timeout = None, done = False):
    #     """
    #     clear output
    #     """
    #     while True:
    #         if done:
    #             wait = True
    #         try:
    #             if self._debug:
    #                 print(f' [KeplerServer.clear] waiting for reply from Kepler.')
    #             reply = self._connection.get(wait, timeout)
    #             wait = False
    #         except NoData:
    #             if self._debug:
    #                 print(f' [KeplerServer.clear] Kepler queue empty.')
    #             if not done:
    #                 break
    #             else:
    #                 continue
    #         except NoConnection:
    #             if self._debug:
    #                 print(f' [KeplerServer.clear] Kepler queue closed.')
    #             return
    #         if self._debug:
    #             print(f' [KeplerServer.clear] Forwarding reply to Connection {reply}')
    #         self._outward.put(reply)
    #         if isinstance(reply, Done):
    #             done = False


    def run(self):
        """
        do server command loop
        """
        # inital clear of startup info
        # self.clear(done = True)
        if self._debug:
            print(f' [KeplerServer.run] Starting command loop.')

        while True:
            self._done = False
            # clear!!! (old output)
            if self._debug:
                print(f' [KeplerServer.run] waiting for commands from SERVER.')
            task = self._outward.get()
            result = self.handle(task)
            # deal with results
            if self._stop:
                break

        # TODO - shut down / terminate KEPLER and server
        # setup atexit



    def handle(self, task):
        # stop from client
        if isinstance(task, Stop):
            if self._debug:
                print(f' [KeplerClient] Stop from client.')
            self._stop = True
            return
        result = self.process(task)
        if result is not None:
            if isinstance(result, (Reply, Done)):
                self._outward.put(result)
            else:
                raise Exception(' [KeplerServer.handle] unexpected reply')
            return
        if self._debug:
            print(f' [KeplerServer.handle] forwarding to Kepler: {task}')
        self._connection.put(task)
        done = task_requires_done(task) # change!!!
        if task.noreply and not done:
            return
        self._done |= done # XXXXXXXXX Here we get burnt by recursive level!
        while True:
            if self._debug:
                print(f' [KeplerServer.handle] waiting for reply from Kepler.')
            reply = self._connection.get()
            if isinstance(reply, Call):
                result = self.handle_client_call(reply)
                # deal with results
            else:
                # buffer output, erase if full?
                if self._debug:
                    print(f' [KeplerServer.handle] forwarding reply to Connection: {reply}')
                self._outward.put(reply)

            if done:
                if isinstance(reply, Done):
                    break
            else:
                if isinstance(reply, Reply):
                    break
        if isinstance(reply, Stop):
            # here we should just disconnect server or terminate KEPLER
            # self.stop = True
            print('  [KeplerServer.handle] Stop requested.  Ignoring.')
            self._stop = True

    def handle_client_call(self, reply):
        result = self.process_client_call(reply)
        if result is not None:
            raise Exception('Action TBD')
        if self._debug:
            print(f' [KeplerServer.cc] forwarding reply to SERVER {reply}')
        self._outward.put(reply)
        if reply.noreply:
            return
        while True:
            if self._debug:
                print(f' [KeplerServer.cc] waiting for reply from SERVER.')
            task = self._outward.get()
            result = self.handle(task)
            # deal with results
            if isinstance(task, Reply):
                return

    def process(self, task):
        if self._debug:
             print(f' [KeplerServer.process] processing {task}')
        if isinstance(task, GetStartup):
            return Reply(self._startup)
        elif isinstance(task, Cycle):
            self.cycle(task.n)


    def process_client_call(self, reply):
        if self._debug:
             print(f' [KeplerServer.pcc] processing {reply}')
        return

    def g(self):
        """
        run indefinitively, but listen for messages
        """
        self.cycle(n = maxsize)

    def cycle(self, n = 1):
        """
        go for defined numer of steps but listen for messages

        Where to collect output if not connected?
        """
        print(' [KepServer.cycle] called.')
        # need to produce exatly one 'Done.'
        # (caputure all originals and emit one whne finished.
        # also collect or forward output
        self._outward.put(Reply('[KepServer.cycle] called.'))
        return
        # if self.status == 'terminated':
        #     raise KeplerNotRunning()
        # self.clear()
        # for i in range(n):
        #     self._done = False
        #     task = Cycle()
        #     result = self.servercall(task)
        #     if isinstance(result, RemoteException):
        #         raise result
        #     self.clear(done = True)
        #     while True:
        #         x = gets()
        #         if x == 's':
        #             break
        #         elif x != '':
        #             x = str(x).strip()
        #             self.execute(x)
        #         else:
        #             break
        #     if x == 's':
        #         break
        #     if i == n-1 or self._kepler.gencom.mode == 0:
        #         break

        # ncyc = self.kd.qparm.ncyc
        # print(' [CYCLE] Stop at cycle {:d}'.format(ncyc))

    def s(self):
        # maybe this one is more atomic?
        self.cycle(n = 1)

def run(*argv):
    progname = args[0]
    args = argv[1:]
    kwargs = dict()
    for i,a in enumerate(args):
        try:
            args[i] = eval(a)
        except:
            pass
    for a in list(args):
        try:
            d = eval(f'dict({a})')
        except:
            pass
        else:
            kwargs.update(d)
            args.remove(a)
    kwargs['name'] = progname
    KeplerServer(*args, **kwargs)

if __name__ == "__main__":
    # use argparse instead, maybe based on name?
    # use atexit to terminate processes
    run(*argv)
