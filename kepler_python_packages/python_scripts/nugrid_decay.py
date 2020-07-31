from utils import TextFile
from ionmap import DecayRate
from isotope import ion

def read_nugrid_network(filename = '~/NuGrid/test/decay/networksetup.txt.xz'):
    with TextFile(filename, mode = 'r') as f:
        section = 0
        while section < 2:
            l = f.readline()
            if l.startswith('*'):
                section += 1
        for i in range(3):
            x = f.readline()
        n = int(f.readline()[:14])
        lines = dict()
        for i in range(1, n + 1):
            l = f.readline()
            if l[8] == 'F':
                continue
            reaction = l[80:85]
            if not reaction in ('(+,g)', '(-,g)', '(g,a)'):
                continue
            rate = l[62:71]
            if rate.count('E') + rate.count('e') == 0:
                rate = rate.replace('-', 'E-').replace('+', 'E+')
            rate = float(rate)
            if rate < 1.e-30:
                continue
            lines[l[10:60]] = l, rate
        decaydata = list()
        for k,(l, rate)  in lines.items():
            ions_in  = [ion(l[14:19])]*int(l[11]) + [ion(l[27:32])] * int(l[24])
            ions_out = [ion(l[41:46])]*int(l[38]) + [ion(l[54:59])] * int(l[51])
            decaydata.append(DecayRate(ions_in, ions_out, rate))
    return decaydata

from abuset import AbuData, AbuDump
from NuGridPy import nugridse as mp
import numpy as np
import os.path

def read_nugrid_data(path = os.path.expandvars('${HOME}/NuGrid/test/decay'), model = None):
    sefiles = mp.se(path)
    isotopes = np.array(sefiles.se.isotopes)
    if model is None:
        model = sefiles.se.cycles[-1]
    massf = sefiles.get([model], 'iso_massf')[0]

    temp = sefiles.get([model], 'temperature')[0] * sefiles.get('temperature_unit')
    ii = np.where(temp == 1.)[0]
    if len(ii) > 0:
        bottom = ii[-1] + 1
    else:
        bottom = 0

    a = AbuData(
        data = massf,
        ions = isotopes,
        molfrac = False)

    m = sefiles.get([model], 'mass')[0] * sefiles.get('mass_unit')
    dm = np.empty_like(m)
    dm[1:] = m[1:] - m[:-1]
    dm[0] = 0

    d = AbuDump(
        a,
        bottom = 0,
        xm = dm,
        zm = m,
        )
    return d.cleaned(debug = True)

from ionmap import TimedDecay

def decay_test(time = 1.e9, n = 20):
    a = read_nugrid_data()
    decaydata = read_nugrid_network()
    d = TimedDecay(a, decaydata = decaydata)
    x = d(a, time = time)
    s = x.project()

    import matplotlib.pylab as plt
    import color
    colors = color.ColorBlindRainbow()(np.linspace(0,1,n))

    # total abundance plot
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(s.A(), s.X(), '.')
    ax.set_xlabel('mass number')
    ax.set_ylabel('mass fraction')

    # most abunant species
    f = plt.figure()
    ax = f.add_subplot(111)
    ii = np.argsort(s.X())

    for c, i in zip(colors, ii[-n:]):
        ix = s.iso[i]
        ax.plot(x.zm / 2.e33, x.ion_abu(ix), label = str(ix),
                color = c)
        ax.plot(a.zm / 2.e33, a.ion_abu(ix), ':', color = c)
    ax.set_xlabel('mass coordinate / solar masses')
    ax.set_ylabel('mass fraction')
    leg = ax.legend(loc='best', fontsize='small')
    leg.draggable()
    ax.set_xlim(0, 2)
