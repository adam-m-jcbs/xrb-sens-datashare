import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

import lcdata
import scales

class Litecurve():
    """
    Install
    -------
    from kepler.code import Kepler as K
    from kepler.hooks.plots import Litecurve as LC
    k = K()
    lc = LC.install(k)

    TODO
    - detailed colour settings
    - y scale
    - save/restore/load from lc options
    - obtaining lc file data from remote host ...
    """

    def __init__(
            self, kepler,
            pltargs = dict(
               ls = '-',
               c = 'r',
               lw = None,
               ),
            load = None,
            ):
        self.l = []
        self.k = kepler
        self.t = []
        self.c = []
        self.pltargs = pltargs
        self.style = 'default'
        with plt.style.context(self.style):
            self.f = plt.figure()
            self.a = self.f.add_subplot(111)
            self.a.set_xscale('timescale')
            self.a.set_ylabel(r'$L$ ($\mathrm{L}_{\mathrm{Edd,\!\!\odot}}$)')
        if load is not None:
            self.loadlc(load)
    def plot(self):
        leddsun = 1.44e38
        with plt.style.context(self.style):
            self.a.plot(self.t, np.array(self.l) / leddsun, **self.pltargs)
            self.f.show()
    def update(self):
        self.l.append(self.k.xlum)
        self.t.append(self.k.time)
        self.c.append(self.k.ncyc)
        self.a.lines.clear()
        self.plot()
    def loadlc(self, filename = True):
        # TODO - find file automatically and load automatically ?
        # retrive file or data from remote hosts
        if filename == True:
            filename = self.k.kd.nameprob + '.lc'
        lc = lcdata.load(filename)
        recl = lc.data[-1]
        rec0 = lc.data[0]
        if len(self.c) == 0:
            assert self.k.ncyc <= recl.ncyc
            assert self.k.ncyc >= rec0.ncyc
            ii = lc.ncyc <= self.k.ncyc
        else:
            assert self.c[0] <= recl.ncyc
            assert self.c[1] >= rec0.ncyc
            ii = lc.ncyc < rec0.ncyc
        self.c = lc.ncyc[ii].tolist() + self.c
        self.l = lc.xlum[ii].tolist() + self.l
        self.t = lc.time[ii].tolist() + self.t
        self.plot()
    @classmethod
    def install(cls, kepler, *args, **kwargs):
        L = cls(kepler, *args, **kwargs)
        kepler.client_add_post_hook(L)
        return L
    def __call__(self):
        self.update()
