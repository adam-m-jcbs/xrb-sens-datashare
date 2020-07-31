"""
Defines plot data interface.
"""

import numpy as np

from physconst import Kepler as const

class DataInterface():
    """
    Plot interface that also provides select derived quantities.
    """
    @property
    def dy(self):
        """
        Zone column density.
        """
        # this has to be computed here because loadbuf does not return all values needed.
        # copy from kepdump
        jm = self.qparm.jm
        r = self.rn[0:jm+2]
        r2 = np.empty_like(r)
        r2[0] = r[0]**2
        r2[1:-1] = r[1:-1] * r[0:-2]
        r2[-1] = r[-2]**2
        if r[0] == 0:
            r2[1] = r[1]**2 / 3
            r2[0] = 1
        dm = self.xm[0:jm+2].copy()
        dm[-1] = self.parm.xmacrete
        dy = dm / (r2 * 4 * np.pi)
        return dy

    @property
    def y(self):
        """
        zone interface column density
        """
        return np.cumsum(self.dy[1:][::-1])[::-1]

    @property
    def pnf(self):
        """
        preessure on zone interface
        """
        jm = self.qparm.jm
        xm = self.pn[0:jm+2]
        x = np.ndarray(jm+1)
        x[1:-1] = 0.5*(xm[1:-2] + xm[2:-1])
        x[-1] = self.parm.pbound
        if x[-1] == 0:
            x[-1] = 2 * xm[-2] - x[-2]
        if x[-1] < 0:
            x[-1] = 0.5 * xm[-2]
        x[0] = 2 * xm[1] - x[1]
        return x

    @property
    def phi(self):
        """
        Newtonian gravitational potential at shell boundary (cm**2/sec**2)
        """
        jm = self.qparm.jm
        rn = self.rn[0:jm+2]
        zm = self.xm[0:jm+2]
        g = const.gee * self.parm.geemult
        p = np.empty_like(rn)
        p[-1] = 0
        p[-2] = - g * zm[-2] / rn[-2]
        x = - 0.5 * g * (
            (zm[1:-1] / rn[1:-1]**2 +
             zm[0:-2] / (rn[0:-2]**2 + 1.e-99)) *
            (rn[1:-1] - rn[0:-2]))
        p[0:-2] = p[-2] + np.cumsum(x[::-1])[::-1]
        return p

    @property
    def tau(self):
        """
        optical depth
        """
        jm = self.qparm.jm
        xkn = self.xkn[0:jm+2]
        dn = self.dn[0:jm+2]
        xm = self.xm[0:jm+2]
        rn = self.rn[0:jm+2]
        drdn = np.empty_like(rn)
        tau = np.empty_like(rn)
        drdn[1:-1] = xm[1:-1] / (2 * np.pi * (rn[1:-1]**2+rn[:-2]**2))
        tau[-2:] = 0
        if self.parm.isurf != 0:
            tau[-2] = 2/3
        tau[:-2] = np.cumsum((drdn[1:-1] * xkn[1:-1])[::-1])[::-1]
        return tau

    @property
    def xmtot(self):
        """
        total mass
        """
        return self.zm[self.jm]

    def set_cache_np(self, value):
        pass
