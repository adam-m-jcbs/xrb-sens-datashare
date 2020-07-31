import numpy as np
import isotope

def tpow(i1, i2, t9):
    """
    Compute temperature sensitivity exponnent of binary nuclear reactions, d ln rate / d ln T.

    Only non-resonnant part of rate is considered.

    Input:
      i1, i2 :
        isotopes
      t9 :
        temperature in GK

    Returns:
      n = d ln rate / d ln T
    """
    i1 = isotope.ion(i1)
    i2 = isotope.ion(i2)
    mred = i1.A * i2.A / (i1.A + i2.A)
    tau = 4.2487 * (i1.Z**2 * i2.Z**2 * mred / t9)**(1/3)
    return  (tau - 2) / 3
