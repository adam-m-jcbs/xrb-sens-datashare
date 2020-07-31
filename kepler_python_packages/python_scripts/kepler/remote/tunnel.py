import fabric
from contextlib import ExitStack

from .defaults import *

class Tunnel():
    """
    manage/establish ssh tunnel
    """
    def __init__(self, host, port = _default_port, rport = None):
        if rport is None:
            rport = port
        self._con = fabric.Connection(host)
        for k,v in self._con.ssh_config.items():
            print(f' [tunnel] {k}: {v}')
        self._fw = self._con.forward_local(port, rport)
        print(f' [tunnel] port: {port}')
        self._stack = ExitStack()
        self._stack.enter_context(self._fw)

    def close(self):
         self._stack.close()
         self._con.close()
