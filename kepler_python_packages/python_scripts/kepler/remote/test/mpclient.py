from multiprocessing import Process, Queue, JoinableQueue
from multiprocessing.managers import BaseManager
class QueueManagerClient(BaseManager): pass
QueueManagerClient.register('get_in_queue')
QueueManagerClient.register('get_out_queue')

# class Client():
#     def __init__(self):
#         m = QueueManagerClient(address=('', 50000), authkey=b'abracadabra')
#         m.connect()
#         self.m = m
#         self.qi = m.get_in_queue()
#         self.qo = m.get_out_queue()
#     def send(self, msg):
#         self.qi.put(msg)
#         result = self.qo.get()
#         print(f'{msg} --> {result}')
#     def stop(self):
#         self.qi.put('STOP')

class ClientProcess(Process):
    def __init__(self, inp, out):
        self.inp = inp
        self.out = out
        m = QueueManagerClient(address=('', 50000), authkey=b'abracadabra')
        self.connected = False
        try:
            m.connect()
            self.m = m
            self.qi = m.get_in_queue()
            self.qo = m.get_out_queue()
            self.connected = True
            self.out.put('Welcome')
        except:
            self.out.put('Failed')
        super().__init__()
    def run(self):
        if not self.connected:
            self.out.put(Exception('connection lost'))
            return
        for params in iter(self.inp.get, 'STOP'):
            self.qi.put(params)
            try:
                s = self.qo.get()
            except EOFError:
                self.out.put(Exception('connection lost'))
                return
            self.out.put(s)
            print(f'[CP] {params} -- {s}')
            self.inp.task_done()
        print('[CP] client finished')
        self.out.put('Done')
        self.inp.task_done()


class ProxyClient():
    def __init__(self):
        self.inp = JoinableQueue()
        self.out = Queue()
        self.p = ClientProcess(self.inp, self.out)
        self.p.start()
        out = self.out.get()
        print(out)
    def send(self, msg):
        self.inp.put(msg)
        result = self.out.get()
        print(f'{msg} --> {result}')
    def stop(self):
        self.send('STOP')
        self.p.join()
    def kill(self):
        self.send('KILL')
        self.stop()

def start():
    P = ProxyClient()
    return P
