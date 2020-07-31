import fabric
from contextlib import ExitStack
from .mpclient import start

def communicate():
    con = fabric.Connection('w')
    fw = con.forward_local(50000)
    print(fw)
    stack = ExitStack()
    stack.enter_context(fw)
    x = start()
    x.send('xx')
    x.stop()
    stack.close()

    # with con.forward_local(50000):
    #     x = start()
    #     x.send('xx')
    #     x.stop()
