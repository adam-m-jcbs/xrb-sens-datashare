import numpy as np

m = np.array([12,13,14,15,16,17,18,20,22,25,27,30])

def I(m0, m1, gamma=-2.35):
    return (m1**(gamma+1) - m0**(gamma+1))/(gamma+1)

def I1(m0, m1, gamma=-2.35):
    return (m1**(gamma+2) - m0**(gamma+2))/(gamma+2)

# def Ifac0(m0, m1, gamma=-2.35):
#     return (m1 * I(m0, m1, gamma) - I1(m0, m1, gamma)) / (m1 - m0)

# def Ifac1(m0, m1, gamma=-2.35):
#     return (I1(m0, m1, gamma) - m0 * I(m0, m1, gamma)) / (m1 - m0)

def Ifac(m0, m1, gamma=-2.35):
    dmi = 1/(m1 - m0)
    i = I(m0, m1, gamma)
    i1 = I1(m0, m1, gamma)
    return (m1 * i - i1) * dmi, (i1 - m0 * i) * dmi

def weights(m, gamma = -2.35):
    w0, w1 = Ifac(m[:-1], m[1:], gamma)
    w = np.zeros_like(m, dtype = np.float64)
    w[:-1] = w0
    w[1:] += w1
    w /= I(m[0], m[-1])
    return w
