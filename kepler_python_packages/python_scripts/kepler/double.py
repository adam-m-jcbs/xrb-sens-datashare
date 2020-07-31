"""
Kepler Binary Module

... to be extended to multiple star module ...
"""

import numpy as np

from .process import Proxy, ProxyData

class Double():
    def __init__(self,
                 run1 = 'xxx',
                 gen1 = 'xxxz',
                 run2 = 'yyy',
                 gen2 = 'yyyz',
                 separation = 1.5e13,
                 eccentricity = 0,
                 ):
        """
        start double star
        """
        stars_parm = (
            (run1, gen1),
            (run2, gen2),
            )
        self.stars_parm = stars_parm
        self.stars = [
            Proxy(*parm) for parm in stars_parm
            ]

    def __getitem__(self, key):
        return self.stars[key]

    def s(self, concurrent = True):
        times = np.array([s.time for s in self.stars])
        dtnew = np.array([s.dtnew for s in self.stars])
        ii = np.argsort(times)
        i0, i1 = ii
        timesp = times + dtnew
        if timesp[i0] < times[i1] or not concurrent:
            self.stars[ii[0]].s()
        else:
            for s in self.stars:
                s.s(wait = False)
            for s in self.stars:
                s.retreive()

    def clear(self):
        for s in self.stars:
            s.clear()

    def terminate(self):
        for s in self.stars:
            s.terminate()

    def __del__(self):
        self.terminate()
        try:
            super().__del__()
        except:
            pass
