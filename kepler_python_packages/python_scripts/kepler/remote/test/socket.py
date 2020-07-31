"""
modue to provide socket connection to a remote running kepler.

Design:

kepler will be run as 'server'

socket setup will provide pair of queue-like objects running through one socket
"""


import socket
from multiprocessing import Process, Queue, JoinableQueue

class RemoteQueue():
    def __init__(self, sock = None):
        if sock is None:
            self.sock = socket.socket(
                            socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host, port):
        self.sock.connect((host, port))

    def mysend(self, msg):
        totalsent = 0
        while totalsent < MSGLEN:
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent

    def myreceive(self):
        chunks = []
        bytes_recd = 0
        while bytes_recd < MSGLEN:
            chunk = self.sock.recv(min(MSGLEN - bytes_recd, 2048))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return b''.join(chunks)

class OutputQueue(RemoteQueue):
    def __init__(self, joinable = False):
        if joinable:
            self.queue = Queue()
        else:
            self.queue = JoinableQueue()

        super().__init__()
        self.connect('localhost', 4000)


class ImputQueue(RemoteQueue):
    def __init__(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
