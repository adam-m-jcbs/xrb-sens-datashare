#! /bin/env python3

"""
Lane Emden Python interface.

Main routine:
   lane_emden_int(dz, n)
"""

import numpy as np
from . import _solver

def test():
    """
    A simple test.
    """
    n = 3.
    dz = 2.**(-14)
    _solver.lane(dz,n)
    out = _solver.laneout
    n = out.ndata
    t = out.theta
    return t,n

def lane_emden_int(dz = 2.**(-14), n = 3., w = 0.):
    """
    Interface to FORTRAN90 Lane-Emden Integrator.

    Call:
    ndata, data = laneemden.lane_emden_int(dz, n, w)

    INPUT:
        dz:
            step in z, maye use 2**(-14)
        n:
            polytropic index (use 3.)
        w:
            rotation parameter(use 0. for non-rot)
            w = 2 Omega^2 / (4 pi G rho_c)

    OUTPUT:
        ndata:
            number of last point (starts with 0)
        data:
            output data in form [0:ndata,0:1]
            index 0:
                equidistant grid with step size dz starting at 0
            index 1:
                0: theta(z)
                1: d theta(z) / dz
    """
    _solver.lane(dz, n, w)
    out = _solver.laneout
    n = int(out.ndata)
    t = out.theta
    return n,t[0:n+1,:]

def lane_emden_step(x,y,dx,n,w):
    """
    This allows a single call to the rk4 subroutine.

    It turns out to be *way* less efficient.
    Do not use.
    """
    _solver.rk4(x,y[0],y[1],dx,n,w)
    out = _solver.rk4out
    return np.array([out.z0,out.z1])

if __name__ == '__main__':
    t,n = test()
    print(t, n)
