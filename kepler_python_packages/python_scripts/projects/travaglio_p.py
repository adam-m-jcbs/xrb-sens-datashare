
import os.path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


from collections.abc import Iterable

import kepdump
import isotope
import ionmap
import abuset
import physconst
import color

extensions = ('eps','pdf','pgf','svg')

def plot1(iso = 'mo92',
          mass = 15,
          model = 'GG',
          Z = '00',
          path00 = '/m/chris/zdep',
          target = '~/Downloads'):
    """
    plots for production for Chris West grid
    """

    ions = abuset.IonList(iso)

    models = {
        'GG' : '4mass{:d}/sin{:s}/sin',
        'SS' : 'mass{:d}/sca{:s}/sca',
        'SG' : '4mass{:d}/sca{:s}/sca',
        'GS' : '6mass{:d}/sin{:s}/sin',
        }

    path0 = os.path.join(path00, models[model])
    path = path0.format(mass, Z)

    p = kepdump.load(path + '#presn')
    n = kepdump.load(path + 'D#nucleo')

    pa = ionmap.decay(p.abub)
    na = ionmap.decay(n.abub)

    fig = plt.figure(figsize=(8,6), tight_layout=True)
    ax = fig.add_subplot(111)

    llp = p.zm > p.parm.bmasslow
    llp[-1] = False # exlode wind

    ymax = 0.
    for ion in ions:
        ionmpl = ion.mpl()

        line, = ax.plot(
            p.zmm_sun[llp],
            pa.ion_abu(ion)[llp],
            linewidth=0.5,
            label=r'{ion} pre-SN'.format(ion=ionmpl))
        ax.plot(
            n.zmm_sun[1:-1],
            na.ion_abu(ion)[1:-1],
            '--',
            linewidth=1.5,
            label=r'{ion} post-SN'.format(ion=ionmpl),
            color = line.get_color())

        ymax = max(
            ymax,
            np.max(pa.ion_abu(ion)[llp]),
            np.max(na.ion_abu(ion)[1:-1])
            )

        ax.set_xlabel('enclosed mass in solar masses')
        ax.set_ylabel('mass fraction')

    ax.set_yscale('log')
    scale = ymax * 1.25
    ax.set_ylim(scale * 1e-4, scale)
    ax.legend(loc='best')
    plt.show()

    for ext in extensions:
        filename = os.path.join(
            os.path.expanduser(os.path.expandvars(target)),
            'PrePostSN_Msun{M:d}_logZsun{Z:s}_model{model:s}_{ion:s}.{ext:s}'.format(
                M = mass,
                model = model,
                Z = Z,
                ext = ext,
                ion = ''.join(ion.Name() for ion in ions)))
        print('writing to ', filename)
        fig.savefig(filename)


def plot2(dump = '#presn',
          mass = 15,
          model = 'GG',
          Z = '00',
          vmax = 1,
          xr = None,
          yr = None,
          name = 'Map',
          path00 = '/m/chris/zdep',
          target = '~/Downloads',
          ):
    """
    2D plots for production for Chris West grid
    """

    models = {
        'GG' : '4mass{:d}/sin{:s}/sin',
        'SS' : 'mass{:d}/sca{:s}/sca',
        'SG' : '4mass{:d}/sca{:s}/sca',
        'GS' : '6mass{:d}/sin{:s}/sin',
        }

    path0 = os.path.join(path00, models[model])
    path = path0.format(mass, Z)

    d = kepdump.load(path + dump)

    a = ionmap.decay(d.abub, isobars = True)

    fig = plt.figure(figsize=(8,6), tight_layout=True)
    ax = fig.add_subplot(111)

    # map
    A = isotope.ufunc_A(a.ions)
    Amax = max(A)
    Av = np.arange(Amax) + 1
    data = np.zeros((Amax, a.data.shape[0]))
    data[np.in1d(Av, A), :] = (a.data * A[np.newaxis,:]).transpose()
    Ap = np.insert(Av, 0, 0) + 0.5

    im = ax.pcolorfast(
        a.zm_sun,
        Ap,
        data[:,1:],
        vmax = vmax,
        cmap = color.ColorMapGal(4),
        )

    cbar = fig.colorbar(im, label = 'mass fraction')

    if xr is None:
        xr = [0, max(a.zm_sun)]
    if yr is None:
        yr = [0.5, Amax+0.5]

    ax.set_xlim(xr)
    ax.set_ylim(yr)

    ax.set_xlabel('enclosed mass in solar masses')
    ax.set_ylabel('mass number')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()

    fig.show()

    for ext in extensions:
        filename = os.path.join(
            os.path.expanduser(os.path.expandvars(target)),
            '{name:s}_Msun{M:d}_logZsun{Z:s}_model{model:s}_{dump:s}.{ext:s}'.format(
                name = name,
                M = mass,
                model = model,
                ext = ext,
                Z = Z,
                dump=dump.replace('#', '_')))
        print('writing to ', filename)
        fig.savefig(filename)
