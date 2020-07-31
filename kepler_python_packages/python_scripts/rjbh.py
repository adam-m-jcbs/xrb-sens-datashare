'''
Library with BH routines to be translated from IDL
'''

import numpy as np
import matplotlib.pylab as plt

from reduce import reduce

from physconst import CLIGHT, GRAV, XMSUN

def reh(m, a = 0):
    """
    radius of event horizon of spinning black hole (cgs)

    a = J/M
    """
    return (GRAV * m + np.sqrt((GRAV * m)**2 - (a * CLIGHT)**2)) / CLIGHT**2

def reg(m, a = 0, theta = 0):
    """
    radius of ergosphere of spinning black hole (cgs)

    a = J/M

    theta in radians
    """
    return (GRAV * m + np.sqrt((GRAV * m)**2 - (np.cos(theta) * a * CLIGHT)**2)) / CLIGHT**2

def rs(m):
    """
    Schwarzschild radius (cgs)
    """
    return 2 * GRAV * m / CLIGHT**2

def jarm(a, r, m):
    """
    angular momentum of ISCO
    """

    rc = r * CLIGHT
    rc2 = rc**2
    gmr = GRAV * m * r
    sgmr = np.sqrt(gmr)

    p = rc2 - 3 * gmr + 2 * a * sgmr
    ii = p <= 0
    if np.shape(p) == ():
        if ii:
            p = 1.e-99
    else:
        p[ii] = 1.e-99
    j = sgmr * (rc2 - 2 * a * sgmr + a**2) / (rc * np.sqrt(p))
    if np.shape(j) == ():
        if ii:
            j = 0.
    else:
        j[ii] = 0
    return j

def earm(a, r, m):
    """
    specific energy of orbit (cgs)
    """

    rc = r * CLIGHT
    rc2 = rc**2
    gmr = GRAV * m * r
    sgmr = np.sqrt(gmr)

    p = np.maximum(rc2 - 3 * gmr + 2 * a * sgmr, 1.e-99)

    e = (rc2 - 2 * gmr + a * sgmr)/(rc * np.sqrt(p))
    return e

def ram(a, m):
    """
    radius of ISCO (cm)

    a, m in cgs units
    """
    x13 = 1 / 3
    z3 = a * CLIGHT / (GRAV * m)
    z1 = 1 + np.maximum(1 - z3**2, 0)**x13 *((1 + z3)**x13 + np.maximum(1 - z3, 0)**x13)
    z2 = np.sqrt(3 * z3**2 + z1**2)
    r = GRAV * m / CLIGHT**2 * (3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2)))
    return np.array(r, dtype=np.float)

def jam(a, m):
    """
    specific angular momentum of ISCO (cgs)

    a, m in scgs
    """
    r = ram(a, m)
    j = jarm(a, r, m)
    return j

def eam(a, m):
    """
    specific energy of ISCO
    """
    r = ram(a, m)
    e = earm(a, r, m)
    return e

def rjbh(j, a, m):
    """
    radius of equatorial orbit with given j
    """

    eps = 1.e-5
    x13=1/3
    z3 = a * CLIGHT / (GRAV * m)
    z1 = 1 + (1 - z3**2)**x13 *((1 + Z3)**x13 + (1 - z3)**x13)
    z2 = np.sqrt(3 * z3**2 + Z1**2)
    r0 = GRAV * m / CLIGHT**2 * (3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2))) * (1 + eps)
    r1 = 3 * j**2 / (GRAV * m)

    j0 = jarm(a, r0, m)
    j1 = jarm(a, r1, m)

    r = 0.5 * (r0 + r1)
    jx = jarm(a, r, m)

    i = 0
    imax = 50

    while (np.abs(j - jx) / np.abs(j + jx) > 1.e-7) and (i < imax) and (jx == jx):
        if jx < j:
            r0 = r
        else:
            r1=r
        r = 0.5 * (r0 + r1)
        jx = jarm(a, r, m)
        i = i + 1
    if (not (jx == jx)) or (i >= imax):
        r = 0
    return r

def arm(r, m):
    """
    Get a for given ISCO radius?
    """
    a1 = GRAV * m / CLIGHT
    a0 = 0

    a=0.5 * (a0 + a1)
    rx = ram(a, m)

    i = 0
    imax = 50

    while (np.abs(r - rx)/np.abs(r + rx) > 1.e-7) and (i < imax) and (rx == rx) and (rx < 1.e99):
        if rx > r:
            a0 = a
        else:
            a1=a
        a = 0.5 * (a0 + a1)
        rx = ram(a, m)
        i = i + 1

    if (not (rx == rx)) or (rx > 1.e99) or (i >= imax):
        r = 0
    a = 0.5 * (a0 + a1)
    return a


def areq():
    """
    what does this do?
    """
    m = 1.
    f = CLIGHT / (GRAV * m)

    a1 = 1 / f
    a0 = 0.

    a = 0.5 * (a0 + a1)
    jx = jam(a, m)

    i = 0
    imax = 50
    acc = 1.e-12

    while (np.abs(2 * a - jx) / np.abs(2 * a + jx) > acc) and (i < imax) and (jx == jx) and (jx < 1.e99):
        if jx > 2 * a:
            a0 = a
        else:
            a1 = a
        a = 0.5 * (a0 + a1)
        jx = jam(a, m)
        print(jx * f, a0 * f, a1 * f)
        i = i + 1
    if (not (jx == jx)) or (jx > 1.e99) or (i >= imax):
        r = 0
    a = 0.5 * (a0 + a1)
    print(a * f, ram(a, m) * CLIGHT**2/(GRAV * m))

#######################################################################
# plots
#######################################################################

def plot_e():
    """
    specific energy of ISCO
    """

    xm=XMSUN
    low = -10

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(10**low, 1)
    ax.set_xscale('log')
    ax.set_ylim(0.5, 1)
    ax.set_xlabel(r'$1-a\,c\,/\,(G\,M_\mathrm{BH})$'),
    ax.set_ylabel(r'$e_\mathrm{ISCO}\; (c^2)$')

    n = 1000
    x = np.arange(n+1) / n * low
    x = 10**x

    y = eam((1 - x) * xm * GRAV / CLIGHT, xm)
    ax.plot(x, y, ls='-', lw=2)
    ax.text(2.e-10, .6, r'$1/\sqrt{3}$', ha='left')
    ax.text(.5, .95, r'$\sqrt{8/9}$', ha='right')

def plot_ej():
    """
    energy and angular momentum of ISCO
    """
    xm = XMSUN

    low = -10

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(10**low, 1)
    ax.set_xscale('log')
    ax.set_ylim(0, 4)
    ax.set_xlabel(r'$1-a\,c\,/\,(G\,M_\mathrm{BH})$'),
    ax.set_ylabel(r'$j_\mathrm{ISCO}\; (G\,M_\mathrm{BH}\,/\,c),\quad e_\mathrm{ISCO}\; (c^2)$')

    n = 1000
    x = np.arange(n + 1) / n * low
    x = 10**x

    y = jam((1 - x) * xm * GRAV / CLIGHT, xm) * CLIGHT / GRAV / xm
    ax.plot(x, y, ls='-', label = r'$j_\mathrm{ISCO}$')

    y = eam((1 - x) * xm * GRAV / CLIGHT, xm)
    ax.plot(x, y, ls=':', label = r'$e_\mathrm{ISCO}$')

    ax.legend(loc='upper left')

    ax.text(2.e-10, .6, r'$1/\sqrt{3}$', ha='left')
    ax.text(.5, .95, r'$\sqrt{8/9}$', ha='right')

    ax.text(2.e-10, 1.25,r'$\sqrt{4/3}$', ha='left')
    ax.text(5.e-1, 3.5,r'$\sqrt{12}$', ha='right')


# ~/Plots/rjbh.pdf
# ~/Plots/rjbh_log.pdf

def plot_rjbh(log = False, dump = None):
    """
    plot j values for orbits of given radius as function of a/M
    """
    xm = XMSUN
    xms = r'M_\odot'
    xmbh = r'M_\mathrm{BH}'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    c = '#000000'

    if not log:
        ax.set_xlim(0,10)
        ax.set_xscale('linear')
        ax.set_ylim(0,1)
        ax.set_xlabel(rf'$j\,c\,/\,G\,{xmbh}$')
        ax.set_ylabel(rf'$a\,c\,/\,G\,{xmbh}$')
    else:
        ax.set_xlim(0, 10)
        ax.set_xscale('linear')
        ax.set_ylim(0, -10)
        ax.set_xlabel(rf'$j\,c\,/\,G\,{xmbh}$')
        ax.set_ylabel(rf'$\log\left(1-a\,c\,/\,G\,{xmbh}\right)$')
    n = 30000
    if log:
        y = 1 - 0.1**(10 * np.arange(n + 1) / n)
    else:
        y = np.arange(n + 1) / (n * (1+ 2.e-7)) + 1.e-7
    x = jam(y * xm * GRAV / CLIGHT, xm) * CLIGHT / GRAV / xm

    if log:
        yp = np.log10(1 - y)
    else:
        yp = y
    #x, yp = reduce(x, yp, axes=ax)
    ax.plot(x, yp, ls='--', c = c)

    x = eam(y * xm * GRAV / CLIGHT, xm) * y * 2
    if log:
        yp = np.log10(1 - y)
    else:
        yp = y
    #x, yp = reduce(x, yp, axes=ax)
    ax.plot(x, yp, ls='-.', c = c)

    # r = np.array([2.,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
    # r = np.array([.2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100])
    r = np.array([2.,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
    # r = np.arange(100)

    for ri in r:
        ys = y
        if ri * GRAV * xm / CLIGHT**2 < 6 * GRAV * xm / CLIGHT**2:
            a = arm(ri * GRAV * xm / CLIGHT**2, xm)*CLIGHT / GRAV / xm
            ii = ys > a
            ys = ys[ii]
        xs = jarm(ys * GRAV / CLIGHT * xm, ri * xm * GRAV / CLIGHT**2, xm) * CLIGHT / GRAV / xm
        ii = xs > 0
        xs = xs[ii]
        ys = ys[ii]
        t = 1
        l = '-'
        if ri % 10 == 0:
            t = 2
        if (ri > 10) and (ri % 10 == 5):
            l = ':'
        if log:
            ys = np.log10(1 - ys)
        #xs, ys = reduce(xs, ys, axes=ax)
        ax.plot(xs, ys, ls = l, lw = t, c = c)

    if log:
        ypos = 0.8
    else:
        ypos = 0.1
    ax.text(0.6, ypos,
            (fr'$G\,{xms}/\,c^2=1.476\times10^5\,\mathrm{{cm}}$'+'\n'+
             fr'$G\,{xms}/\,c\,\;=4.4\times10^{{15}}\,\mathrm{{cm}}^2\,\mathrm{{s}}^{{-1}}$'),
            ha='left',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.9),
            transform=ax.transAxes)

    ax.text(0.01, 0.99, fr'$a/{xmbh}$'+'\ndecr.',
            ha='left',
            va='top',
            transform=ax.transAxes)

    ax.text(0.19, 0.01, 'incr.\n'+ fr'$a/{xmbh}$',
            ha='center',
            va='bottom',
            transform=ax.transAxes)

    ax.text(0.98, 0.5, fr'$r_\mathrm{{circ}}\,c^2\,/\,G\,{xmbh}$',
            ha='right',
            va='center',
            bbox=dict(facecolor='white', alpha=0.9),
            transform=ax.transAxes)

    if log:
        pos = (0.8, -0.1)
        angle = 22
    else:
        pos = (1., 0.6)
        angle = 78
    ax.text(*pos, r'$2\,a\,e_\mathrm{ISCO}/c^2$',
            rotation=angle,
            ha='center',
            va='bottom')

    ypos = np.average(ax.get_ylim())
    labels = [fr'$^{{{p}}}$' for p in (10, 15, 20)]
    if log:
        xpos = (3.65, 4.3, 4.90)
    else:
        xpos = (3.85, 4.5, 5.05)
    for x,l in zip(xpos, labels):
        ax.text(x, ypos, l, rotation=90, ha='center')

    if log:
        ax.text(1.77, ypos, r'$^{3}$', rotation=90, ha='right')
        ax.text(2.09, ypos, r'$^{4}$', rotation=90, ha='right')
        ax.text(2.37, ypos, r'$^{5}$', rotation=90, ha='right')
    else:
        ax.text(2.35, 0.75, r'$^{3}$', rotation=0, ha='right')
        ax.text(2.75, 0.53, r'$^{4}$', rotation=0, ha='right')
        ax.text(3.10, 0.26, r'$^{5}$', rotation=0, ha='right')

    if dump is not None:
        d = kepdump.load(dump)
        angjn = d.angjn
        angambh = d.angambhn
        zm = d.zmm

        x = angjn * zm / (GRAV * XMSUN**2 / CLIGHT)
        y = angambh * CLIGHT/(GRAV * zm)
        # x, y = reduce(x, y, axes=ax)
        ax.plot(x, y)

    fig.tight_layout()

# ;-----------------------------------------------------------------------
# ;OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# ;-----------------------------------------------------------------------

def z_rk4(x, y):
    """
    rk4 integrand (non-vectorised) for z function
    """
    j = y[0]
    m = y[1]
    a = j / m
    dm = eam(a, m)
    dj = jam(a, m)
    return np.array([dj, dm])

def rk4(y, dydx, x, h, f):
    hh = h * 0.5
    h6 = h / 6
    xh = x + hh
    yt = y + hh * dydx
    dyt = f(xh, yt)
    yt = y + hh * dyt
    dym = f(xh, yt)
    yt = y + h * dym
    dym = dyt + dym
    dyt = f(x + h, yt)
    yout = y + h6 * (dydx + dyt + 2 * dym)
    return yout

def z():
    """
    Seems to determine mass and spin of BH accretion from LSO
    """
    dm0 = 0.5 * XMSUN
    eps = 1.e-7
    dmlim = 1.e-9 * XMSUN
    mlim = 10 * XMSUN

    n = 30000
    j = np.empty(n+1)
    m = np.empty(n+1)
    macc = np.empty(n+1)
    m[0] = XMSUN
    j[0] = 0.
    macc[0] = 0.

    dmi = dm0
    nn = 0
    print(f' [dmi] initial: {dmi}')
    i = -1
    while i < n:
        i += 2
        v = np.array([j[i-1], m[i-1]])
        res0 = rk4(v, z_rk4(0, v), 0, dmi, z_rk4)
        res1 = rk4(v, z_rk4(0, v), 0, dmi * 0.5, z_rk4)
        res2 = rk4(res1, z_rk4(0, res1), 0, dmi * 0.5, z_rk4)
        a0 = res0[0] * CLIGHT / (res0[1]**2 * GRAV)
        a2 = res2[0] * CLIGHT / (res2[1]**2 * GRAV)
        if dmi < dmlim:
             nn = i - 1
             i = n
             print(f' [dmlim] i = {nn}')
        elif ((np.abs(a0 - a2) / (2 - (a0 + a2)) > eps) or
              (np.abs(a0 - a2) / (a0 + a2 + eps) > eps)):
            dmi = dmi * 0.5
            i = i - 2
            print(f' [half] i = {i}, dmi = {dmi}')
        else:
            j[i:i+2] = [res1[0], res2[0]]
            m[i:i+2] = [res1[1], res2[1]]
            macc[i:i+2] = macc[i-1] + dmi * 0.5 * np.array([1, 2])
            if macc[i] > mlim:
                nn = i + 1
                i = n
                print(f' [mlim] i = {nn}')

    if nn == 0:
        nn = n
    macc = macc[0:nn+1]
    j = j[0:nn+1]
    m = m[0:nn+1]

    a = j / m

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xlabel(r'$M_\mathrm{acc}\,/\,M_\mathrm{BH,0}$')
    ax.set_ylabel(r'$1-a\,c\,/\,G\,M_\mathrm{BH}$')
    ax.plot(
        macc / XMSUN,
        1 - a * CLIGHT / ( m * GRAV),
        ls = '-',
        lw = 2)
    fig.tight_layout()

    print(f'M_acc/M_BH,initial = {macc[nn]/XMSUN}')
    print(f'M_BH/M_BH,initial = {m[nn]/XMSUN}')

    e = eam(a, m)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0.5, 1.)
    ax.set_yscale('linear')
    ax.set_xlabel(r'$M_\mathrm{acc}\,/\,M_\mathrm{BH,0}$')
    ax.set_ylabel(r'$e_\mathrm{LSO}\;(m_0\,c^2)$')
    ax.plot(
        macc / XMSUN,
        e,
        ls = '-',
        lw = 2)
    fig.tight_layout()
    #return a, m, macc
