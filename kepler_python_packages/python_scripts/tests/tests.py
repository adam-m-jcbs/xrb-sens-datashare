"""
Some tests
"""

import numpy as np

from logged import Logged

class Test(Logged):
    def __init__(self):
        self.setup_logger(silent=False)
        x = np.ndarray((10,100))
        for i in range(10000):
            y = x[0:2,0:-2]
        self.close_logger(timing='done')
        print(y.shape)


from math import log10

from multiprocessing import Process, Queue, JoinableQueue

def layer_processor(input, output, data):
        for value in iter(input.get, 'STOP'):
            x = 0
            for i in range((value + data)**3):
                x += i**2
            output.put(x)
            input.task_done()
        input.task_done()

class Test2(Logged):
    def __init__(self,
                 num_worker_threads = 12):
        
        self.setup_logger(silent=False)
        
        task_queue = JoinableQueue()
        done_queue = Queue()

        for task in range(10):
            task_queue.put(task)

        data = 100
        for i in range(num_worker_threads):
            p = Process(target=layer_processor, args=(task_queue, done_queue, data))
            p.start()
    
        for task in range(10):
            task_queue.put(task)

        task_queue.join()

        for i in range(num_worker_threads):
            task_queue.put('STOP')

        task_queue.join()

        for task in range(20):
            print(done_queue.get())

        self.close_logger(timing='done')
        


            # from Queue import Queue
            # from threading import Thread

            # def worker():
            #     while True:
            #         item = q.get()
            #         process_layer(item)
            #         q.task_done()

            # q = Queue
            # num_worker_threads = 4
            # for i in range(num_worker_threads):
            #     t = Thread(target=worker)
            #     t.daemon = True
            #     t.start()

            # for item in source():
            #         q.put(item)
                    
            # q.join()       # block until all tasks are done

class OpenHouse(object):
    def __init__(self):
        file = '/home/alex/x/x.csv'
        self.lines = []
        with open(file) as f:
            for line in f:
                self.lines.append(line.rstrip('\n'))
    def out(self):
        names = self.lines[0].split(';')[1:]
        for i,name in enumerate(names):
            print(name)
            for j in range(6):
                entries = self.lines[j+2].split(';')
                time = entries[0]
                prof = entries[i+1].split(' (')[0]
                location = entries[i+1].split(' (')[1].rstrip(')').split(',')
                if len(location) > 2:
                    host = location[0]
                    location = location[1:]
                else:
                    host = None
                
                room = location[0]
                phone = location[1]
                
                s = '{:s} {:s}, {:s} {:s}'.format(
                    time, prof, phone, room)

                if host is not None:
                    s += ' ({:s})'.format(host)
                print(s)
            print()

