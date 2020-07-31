from multiprocessing import Process, Queue, JoinableQueue
from multiprocessing.managers import BaseManager
from multiprocessing.process import current_process
# import os
# import signal
# import sys
# from time import sleep


class Worker(Process):
    def __init__(self, input, output, server):
        self.input = input
        self.output = output
        self.server = server
        print(server, flush = True)
        super().__init__()
    def run(self):
        for params in iter(self.input.get, 'KILL'):
            out = self.processor(params)
            self.output.put(out)
            print(f'{params} -- {out}')
            if params == 'fuck':
                xxx
            self.input.task_done()
        print('server finished')
        self.input.task_done()
        # os.kill(self.pid, signal.SIGKILL)
        # sys.exit()

    def processor(self, attr):
        return len(attr)

class Server(Process):
    def __init__(self):
        super().__init__()
    def run(self):
        qi = JoinableQueue()
        qo = Queue()
        class QueueManager(BaseManager): pass
        QueueManager.register('get_in_queue', callable=lambda: qi)
        QueueManager.register('get_out_queue', callable=lambda: qo)
        m = QueueManager(address=('', 50000), authkey=b'abracadabra')
        # instead of this
        s = m.get_server()
        s.serve_forever()
        # we need something of the kind
        m.start()
        # wait for event
        m.shutdown()

"""
probably the kepler process (main process?) should own the both its
server and the proxy

But KEPLER should be run as a separate process.

Worker should be an interface to the queue - maybe own the server.
"""


def start():
    # s = Server()
    # s.start()
    # QueueManager.register('get_in_queue')
    # QueueManager.register('get_out_queue')
    # m = QueueManager(address=('', 50000), authkey=b'abracadabra')
    # sleep(1)
    # m.connect()
    # qi = m.get_in_queue()
    # qo = m.get_out_queue()
    # p = s.pid

    qi = JoinableQueue()
    qo = Queue()
    QueueManager.register('get_in_queue', callable=lambda: qi)
    QueueManager.register('get_out_queue', callable=lambda: qo)
    m = QueueManager(address=('', 50000), authkey=b'abracadabra')
    m.start()

    p = current_process().pid

    w = Worker(qi, qo, p)#s)
    w.start()

    w.join()
    m.shutdown()


if __name__ == "__main__":
    start()
