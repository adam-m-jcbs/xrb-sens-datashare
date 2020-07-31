"""
Load GCE results from Chris West
"""
import itertools
import os.path
import os
import lzma
import re
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import TextFile

import kepdump
import kepdata
import isotope
import physconst
import color

from abuset import AbuSet, AbuData
from abusets import SolAbu
from logged import Logged
from ionmap import Decay

from gcedata import z2s

# GG grid parameters
masses = [13, 15, 17, 20, 22, 25, 30]
version = '4'
series = 'sin'
Z = [ 0.3, 0.2, 0.1, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.5, -2.0,
      -2.5, -3.0, -4.0, -1000.]

path00 = '/West/chris/zdep/'
explosion = 'D'

def pplot(iso='hg196', decay = True):
    """
    plot isotope at different stages
    """
    np.seterr(all='warn')
    mass = 25
    metallicity = 0
    dumpnames = ['#odep', '#sidep', '#presn', explosion + '#nucleo', explosion + '#envel']
    dumps = []
    for dump in dumpnames:
        path = os.path.join(
            path00,
            version + 'mass{:d}'.format(mass),
            series + z2s(metallicity))
        dumpfilename = os.path.join(path, series + '{}'.format(dump))
        d = kepdump.load(dumpfilename)
        dumps += [d]

    f = plt.figure()
    ax = f.add_subplot(111)
    for d in dumps:
        zm = d.zm_sun
        if decay:
            x = d.abub.decayed(stable = True, elements = False).ion_abu(iso)
            D = 'decayed'
        else:
            x = d.abub.ion_abu(iso)
            D = 'undecayed'
        ax.plot(zm, x, label = d.filename.split('#')[-1])
    ax.set_xlabel('mass / solar masses')
    ax.set_ylabel('mass fraction')
    ax.set_yscale('log')
    ym = np.nanmax(x)
    text = r'${:d}\,\mathrm{{M}}_\odot$, [Z]={:4.1f}, {:s}'.format(
        mass,
        metallicity,
        D)
    ax.text(0.05, 0.95,
            text,
            transform=ax.transAxes)
    ax.set_ylim(1.e-12, None)
    ax.legend()

def pplot2(iso=('pb208', 'hg196'), decay = True, xlim = (1.5,5)):
    """
    plot isotope at different stages
    """
    np.seterr(all='warn')
    mass = 25
    metallicity = 0
    dumpnames = ['#presn', explosion + '#nucleo']
    dumps = []
    for dump in dumpnames:
        path = os.path.join(
            path00,
            version + 'mass{:d}'.format(mass),
            series + z2s(metallicity))
        dumpfilename = os.path.join(path, series + '{}'.format(dump))
        d = kepdump.load(dumpfilename)
        dumps += [d]

    C = color.isocolors(len(iso))
    T = ['solid', 'dashed', 'dashdot', 'dotted']

    f = plt.figure()
    ax = f.add_subplot(111)
    for d,t in zip(dumps, T):
        zm = d.zm_sun
        if decay:
            X = d.abub.decayed(stable = True, elements = False)
            D = 'decayed'
        else:
            X = d.abub.ion_abu(iso)
            D = 'undecayed'
        for i,c in zip(iso, C):
            i = isotope.Ion(i)
            x = X.ion_abu(i)
            lab = r'{} {}'.format(
                d.filename.split('#')[-1],
                i.LaTeX())
            ax.plot(zm, x, label = lab, color = c, linestyle = t)
    ax.set_xlabel('mass / solar masses')
    ax.set_ylabel('mass fraction')
    ax.set_yscale('log')
    ym = np.nanmax(x)
    text = r'${:d}\,\mathrm{{M}}_\odot$, [Z]={:4.1f}, {:s}'.format(
        mass,
        metallicity,
        D)
    ax.text(0.05, 0.95,
            text,
            transform=ax.transAxes)
    ax.set_xlim(xlim)
    ax.set_ylim(1.e-12, None)
    ax.legend()
