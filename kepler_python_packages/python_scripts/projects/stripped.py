import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import os.path

import kepdump
from physconst import XMSUN, RSUN


class PlotMR():
    def __init__(self):
        path = '/home/alex/kepler/lowmhe'
        masses = ['2','2.1','2.2','2.3','2.4','2.5','2.6','2.7','2.8','2.9','3']
        dumpfiles = [os.path.join(path, 'he' + m,'he' + m + 'z') for m in masses]
        self.dumps = [kepdump.load(d) for d in dumpfiles]
    def plot(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        m = [d.mass for d in self.dumps]
        r = [d.radius for d in self.dumps]
        ax.plot(m, r)
        ax.set_xlabel('initial He star mass (solar maasses)')
        ax.set_ylabel('pre-SN radius (solar radii)')
        f.tight_layout()
        f.show()

        self.f = f
        self.ax = ax
