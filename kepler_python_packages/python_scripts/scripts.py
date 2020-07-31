"""
Collection of random scripts.
"""
import glob
import os
import os.path
import socket

import kepdump

def sollo03_core_data():
    """
    run on c
    """
    assert socket.getfqdn() == 'c.spa.umn.edu', 'wrong host'
    files = sorted(glob.glob('/g/alex/kepler/sollo03/s*r/s*r#presn'))
    for file in files:
        d = kepdump.loaddump(file)
        c = d.core()
        print("{:42s}: CO core: {.zm_sun:5.2f}; He core: {.zm_sun:5.2f}; star: {.zm_sun:5.2f}".format(
                file,
                c['C/O core'],
                c['He core'],
                c['star']))
