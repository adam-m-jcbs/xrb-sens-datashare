"""
Module for znuc special settings.
"""

import os
import os.path
import socket
import datetime
import string
import time
import numpy as np
import sys
import shutil
import glob
import psutil
import math
from collections import OrderedDict
from matplotlib.ticker import ScalarFormatter, FuncFormatter

import physconst
from logged import Logged
from abuset import AbuSet
from kepion import KepAbuSet
# from utils import MultiLoop, float2str
from kepdump import KepDump
from kepgen import TestExp, BurnExp, MassL, MassD, MassU, MassE
from convdata import ConvData
from keputils import MassFormatter

from human import time2human

from utils import cachedmethod, CachedAttribute, is_iterable

from abusets import SolAbu

# I think we do want a MassSet object

serpar = dict(
    composition = 'zero',
    yeburn = True,
    burn = True)


def testexp(mode = None,
            **kwargs):

    runpar = dict(serpar)
    runpar.update(kwargs)

    mass_list = MassL(
        9.6 + 0.01*np.arange(1,10,1),
        9.7 + 0.01*np.array([2,4,6,8]),
        9.8 + 0.01*np.array([2,4,6,8]),
        9.95)

    # # ======== u =========
    # parm = dict(
    #     exp = dict(
    #         u = dict(
    #             ekin = 0.01e51,
    #             scut = 4)),
    #     **runpar)

    #?9.64, 9.65, 9.67;
    #?9.7, 9.72, 9.766
    #?9.8, 9.84, 9.86, 9.88,
    #?9.95

    # OK = (9.61, 9.62, 9.63, 9.66, 9.68, 9.69, 9.74, 9.78, 9.82, 9.9)


    # mass = MassL(mass_list)
    # mass = MassL(mass, 9.9,9.8,9.7)

    # # cutsurf auto-detect is wrong
    # MassU(mass,
    #       (9.76, 9.9),
    #       cutsurf=0)
    # MassU(mass,
    #       (9.86),
    #       cutsurf=1)

    # # old runs require to remove extra inner zone (due to bug)
    # MassU(mass,
    #       (9.7, 9.8, 9.9),
    #       addcutzones=-1)
    # # runs require higher resolution for alpha
    # MassU(mass,
    #       precision = 4)
    # # ... some less
    # MassU(mass,
    #       (9.78, 9.62),
    #       precision = 3)
    # # ... and some don't
    # MassU(mass,
    #       (9.82, 9.74, 9.68, 9.63, 9.61),
    #       precision = 2)

    # TestExp(mass = mass, **parm)


    # TestExp(mass = mass, **A)
    # A = dict(
    #     exp = dict(
    #         A = dict(
    #            ekin = 0.3e51,
    #            scut = 4)),
    #     run = run,
    #     **runpar)
    # mass = list(mass_list)
    # MassU(mass, 9.76, cutsurf=0)
    # MassU(mass, 9.86, cutsurf=1)
    # MassU(mass,
    #       [9.61, 9.63, 9.65, 9.68, 9.74],
    #       envelalpha = 0.05)
    # TestExp(mass = mass, **A)

    # parm = dict(
    #     exp = [dict(
    #             exp = 'B',
    #             ekin = 0.6e51,
    #             envelalpha=0.05,
    #             scut = 4)],
    #     **runpar)
    # mass = list(mass_list)
    # MassU(mass, 9.76, cutsurf=0)
    # MassU(mass, 9.86, tstop=3.e6, cutsurf=1)
    # TestExp(mass = mass, **parm)

    # parm = dict(
    #     exp = [dict(
    #             exp = 'C',
    #             ekin = 0.9e51,
    #             envelalpha=0.05,
    #             scut = 4)],
    #     **runpar)
    # mass = list(mass_list)
    # MassU(mass, 9.76, cutsurf=0)
    # MassU(mass, 9.86, tstop=3.e6, cutsurf=1)
    # TestExp(mass = mass, **parm)

    # D=dict(exp={'D':dict(ekin=1.2e51,scut=4)}, **runpar)
    # D['run']=run
    # TestExp(mass=9.6+0.01*(np.array([1]), envelalpha=0.05), **D)
    # TestExp(mass=9.6+0.01*(np.arange(2,10,1)), **D)
    # TestExp(mass=9.7+0.01*np.array([2,4,8]), **D)
    # TestExp(mass=9.8+0.01*np.array([2,4,8]), **D)
    # TestExp(mass=9.76, cutsurf=0, **D)
    # TestExp(mass=9.86, cutsurf=1, tstop=3.e6, **D)
    # TestExp(mass=9.95, **D)

    # parm = dict(
    #     exp = [dict(
    #             exp = 'E',
    #             ekin = 1.5e51,
    #             envelalpha=0.05,
    #             tstop = 3.e6,
    #             scut = 4)],
    #     **runpar)
    # mass = list(mass_list)
    # MassU(mass, 9.76, cutsurf=0)
    # MassU(mass, 9.86, cutsurf=1)
    # TestExp(mass = mass, **parm)

    # parm = dict(
    #     exp = [dict(
    #             exp = 'F',
    #             ekin = 1.8e51,
    #             envelalpha=0.05,
    #             tstop = 3.e6,
    #             scut = 4)],
    #     **runpar)
    # mass = list(mass_list)
    # MassU(mass, 9.76, cutsurf=0)
    # MassU(mass, 9.86, cutsurf=1)
    # TestExp(mass = mass, **parm)

    # parm = dict(
    #     exp = [dict(
    #             exp = 'G',
    #             ekin = 2.4e51,
    #             envelalpha=0.05,
    #             tstop = 3.e6,
    #             scut = 4)],
    #     **runpar)
    # mass = list(mass_list)
    # MassU(mass, 9.76, cutsurf=0)
    # MassU(mass, 9.86, cutsurf=1)
    # TestExp(mass = mass, **parm)

    # parm = dict(
    #     exp = [dict(
    #             exp = 'H',
    #             ekin = 3.0e51,
    #             envelalpha=0.05,
    #             tstop = 3.e6,
    #             scut = 4)],
    #     **runpar)
    # mass = list(mass_list)
    # MassU(mass, 9.76, cutsurf=0)
    # MassU(mass, 9.86, cutsurf=1)
    # TestExp(mass = mass, **parm)

    # parm = dict(
    #     exp = [dict(
    #             exp = 'I',
    #             ekin = 5.0e51,
    #             envelalpha=0.05,
    #             tstop = 3.e6,
    #             scut = 4)],
    #     **runpar)
    # mass = list(mass_list)
    # MassU(mass, 9.76, cutsurf=0)
    # MassU(mass, 9.86, cutsurf=1)
    # TestExp(mass = mass, **parm)

    # J = dict(exp = {'J':dict(ekin=10.e51,
    #                          scut=4,
    #                          tstop=3.e6,
    #                          envelalpha=0.05)},
    #          **runpar)
    # J['run']=run
    # TestExp(mass=9.6+0.01*(np.arange(1,10,1)), **J)
    # TestExp(mass=9.7+0.01*np.array([2,4,8]), **J)
    # TestExp(mass=9.8+0.01*np.array([2,4,8]), **J)
    # TestExp(mass=9.76, cutsurf=0, **J)
    # TestExp(mass=9.86, cutsurf=1, **J)
    # TestExp(mass=9.95, **J)

    # P=dict(exp={'P':dict(ekin=1.2e51,scut=0)}, **runpar)
    # P['run']=run
    # TestExp(mass=9.6+0.01*(np.arange(1,10,1)), **P)
    # TestExp(mass=9.7+0.01*np.array([2,4,8]), **P)
    # TestExp(mass=9.8+0.01*np.array([4,8]), **P)
    # TestExp(mass=9.72, cutsurf=0, envelalpha=0.05, **P)
    # TestExp(mass=9.76, cutsurf=0, envelalpha=0.05, **P)
    # TestExp(mass=9.86, cutsurf=1, envelalpha=0.05, **P)
    # TestExp(mass=9.95, **P)

    # V=dict(exp={'V':dict(ekin=10.e51,scut=0,envelalpha=0.05)}, **runpar)
    # V['run']=run
    # TestExp(mass=9.6+0.01*(np.arange(1,10,1)), **V)
    # TestExp(mass=9.7+0.01*np.array([4,8]), **V)
    # TestExp(mass=9.8+0.01*np.array([2,4,8]), **V)
    # TestExp(mass=9.72, cutsurf=0, **V)
    # TestExp(mass=9.76, cutsurf=0, **V)
    # TestExp(mass=9.86, cutsurf=1, **V)
    # TestExp(mass=9.95, **V)
    pass





def burnexp(mode = None,
            **kwargs):

    runpar = dict(serpar)
    runpar.update(kwargs)

    # parm = dict(exp='u', **runpar)
    # parm['commands'] = "p 238 1.e-3\np 239 1.e-3\n"
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([0,2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([0,2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([0,5]), **parm)


    # parm = dict(exp='A', **runpar)
    # parm['commands'] = "p 238 1.e-3\np 239 1.e-3\n"
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

    # parm = dict(exp='B', **runpar)
    # parm['commands'] = "p 238 1.e-3\np 239 1.e-3\n"
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

    # parm = dict(exp='C', **runpar)
    # parm['commands'] = "p 238 1.e-3\np 239 1.e-3\n"
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

    # parm = dict(exp='D', **runpar)
    # parm['commands'] = "p 238 1.e-3\np 239 1.e-3\n"
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

    # parm = dict(exp='E', **runpar)
    # parm['commands'] = "p 238 1.e-3\np 239 1.e-3\n"
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

    # parm = dict(exp='F', **runpar)
    # parm['commands'] = "p 238 1.e-3\np 239 1.e-3\n"
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

    # parm = dict(exp='G', **runpar)
    # parm['commands'] = "p 238 1.e-3\np 239 1.e-3\n"
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

    # parm = dict(exp='H', **runpar)
    # parm['commands'] = "p 238, 1.e-3\np 239, 1.e-3\n"
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

    # parm = dict(exp='I', **runpar)
    # parm['commands'] = "p 238 1.e-3\np 239 1.e-3\n"
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

    # parm = dict(exp='J', **runpar)
    # parm['commands'] = "p 238 1.e-3\np 239 1.e-3\n"
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

    # parm = dict(exp='P', **runpar)
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

    # parm = dict(exp='V', **runpar)
    # BurnExp(mass=9.6+0.01*(np.arange(1,10,1)), **parm)
    # BurnExp(mass=9.7+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.8+0.01*np.array([2,4,6,8]), **parm)
    # BurnExp(mass=9.9+0.01*np.array([5]), **parm)

class RunData(object):
    @CachedAttribute
    def run_names(self):
        return np.array([self.sentinel + r for r in self.runs])

class ZRuns(RunData):
    metallicity = 0.
    sentinel = 'z'
    runs=[
        '9.6',
        '9.61',
        '9.62',
        '9.63',
        '9.64',
        '9.65',
        '9.66',
        '9.67',
        '9.68',
        '9.69',
        '9.7',
        '9.72',
        '9.74',
        '9.76',
        '9.78',
        '9.8',
        '9.82',
        '9.84',
        '9.86',
        '9.88',
        '9.9',
        '9.95',
        '10',
        '10.1',
        '10.2',
        '10.3',
        '10.4',
        '10.5',
        '10.6',
        '10.7',
        '10.8',
        '10.9',
        '11',
        '11.1',
        '11.2',
        '11.3',
        '11.4',
        '11.5',
        '11.6',
        '11.7',
        '11.8',
        '11.9',
        '12',
        '12.2',
        '12.4',
        '12.6',
        '12.8',
        '13',
        '13.2',
        '13.4',
        '13.6',
        '13.8',
        '14',
        '14.2',
        '14.4',
        '14.6',
        '14.8',
        '15',
        '15.2',
        '15.4',
        '15.6',
        '15.8',
        '16',
        '16.2',
        '16.4',
        '16.6',
        '16.8',
        '17',
        '17.1',
        '17.2',
        '17.3',
        '17.4',
        '17.5',
        '17.6',
        '17.7',
        '17.8',
        '17.9',
        '18',
        '18.1',
        '18.2',
        '18.3',
        '18.4',
        '18.5',
        '18.6',
        '18.7',
        '18.8',
        '18.9',
        '19',
        '19.2',
        '19.4',
        '19.6',
        '19.8',
        '20',
        '20.5',
        '21',
        '21.5',
        '22',
        '22.5',
        '23',
        '23.5',
        '24',
        '24.5',
        '25',
        '25.5',
        '26',
        '26.5',
        '27',
        '27.5',
        '28',
        '28.5',
        '29',
        '29.5',
        '30',
        '30.5',
        '31',
        '31.5',
        '32',
        '32.5',
        '33',
        '33.5',
        '34',
        '34.5',
        '35',
        '36',
        '37',
        '38',
        '39',
        '40',
        '41',
        '42',
        '43',
        '44',
        '45',
        '50',
        '55',
        '60',
        '65',
        '70',
        '75',
        '80',
        '85',
        '90',
        '95',
        '100']

    # this part may be superfluous and needs updating
    explosions=['A','B','C','D','E','F','G','H','I','J','P','V']
    energy=np.array([3, 6, 9, 12, 15, 18, 24, 30, 50, 100, 12, 100]) * 1.e50
    energies=['0.3', '0.6', '0.9', '1.2', '1.5', '1.8', '2.4', '3.0', '5.0', '10.0','1.2','10.0']
    cutlatex=['$S=4$']*10 + ['Ye core']*2
    cuts=['S4']*10 + ['Ye']*2

import winddata
import glob
import re

def make_run_lc(run):
    dir00 = '/m'
    dir00 = os.path.join(dir00,'kepler')
    dir0 = os.path.join(dir00,'znuc')

    tdir00 = '/m'
    tdir00 = os.path.join(tdir00,'web')
    tdir0 = os.path.join(tdir00,'firststars/znuc-old/lightcurve')
    target = os.path.join(tdir0,run+'.lc.txt.gz')

    dir1 = os.path.join(dir0,run)
    files = sorted(glob.glob(os.path.join(dir1,run+'.wnd*')))
    if len(files) > 1:
        num = [re.findall(run+'.wnd([0-9]*)', f)[0] for f in files ]
        num = [int(n) if n != '' else 999 for n in num]
        idx = np.argsort(num)
        files = [files[i] for i in idx]

    #source = os.path.join(dir1,run+'.wnd')
    zerotime = 0.
    for source in files:
        w = winddata.load(source,
                          zerotime = zerotime,
                          silent = False)
        zerotime = w.zerotime
        w.write_lc_txt(target,
                       append = (source != files[0]))

def make_all_run_lc():
    runs = [ZRuns().sentinel + r for r in ZRuns().runs]
    #    runs = ['z15','z30','z45','z60']
    # skip = ['z10.1']
    skip = []

    # models z10.1 and z10.3 had one large wnd file containing all
    # data *plus* numbered wnd files with the same data

    start = 'z10.4'
    i = runs.index(start)
    runs = runs[i:]

    # runs=['z10.2','z10.3']

    for run in runs:
        if not run in skip:
            make_run_lc(run)

def get_run_lifetimes(run, tms = False):
    dir00 = '/home/alex/'
    dir00 = os.path.join(dir00, 'kepler')
    dir0 = os.path.join(dir00, 'znuc')

    dir1 = os.path.join(dir0, run)

    print('{:6.2f}'.format(float(run[1:])), end=' ')

    files = sorted(glob.glob(os.path.join(dir1, run + '#presn')))
    assert len(files) == 1
    filename = files[0]
    d = KepDump(filename)

    print('{:12.5e}'.format(d.time), end = ' ')

    if tms:
        files = sorted(glob.glob(os.path.join(dir1, run + '.cnv*')))
        if len(files) > 1:
            num = [re.findall(run+'.wnd([0-9]*)', f)[0] for f in files ]
            num = [int(n) if n != '' else 999 for n in num]
            idx = np.argsort(num)
            files = [files[i] for i in idx]

        filename = files[0]
        c = ConvData(filename, silent = True)
        print('{:12.5e}'.format(c.tau_MS), end = ' ')

    print()


def get_all_run_lifetimes():
    for run in ZRuns().run_names:
        get_run_lifetimes(run)


def isoratplot():
    import stardb, isotope
    import matplotlib.pyplot as plt

    d = stardb.StarDB('/home/alex/kepler/znuc/znuc.S4.star.deciso.y.stardb.gz')
    ii = d.get_star_slice(energy=1.2, mixing=0.015)
    m = np.array([d.field_data['mass'][j] for j in ii])

    f = plt.figure()
    ax = f.add_subplot(111)

    ax.set_xscale('linear')
    ax.set_yscale('log')

    ax.set_xlabel(r'initial mass / solar masses')
    ax.set_ylabel(r'number ratio')

    xlim = [min(m)-1, max(m)+1]
    ax.set_xlim(xlim)

    ic12 = np.where(d.ions == isotope.Ion('C12'))[0][0]
    ic13 = np.where(d.ions == isotope.Ion('C13'))[0][0]
    c12 = d.abu_data[ic12, ii]
    c13 = d.abu_data[ic13, ii]
    ax.plot(m, c13/c12, 'r+', label = r'$^{13}\mathrm{C}/^{12}\mathrm{C}$')

    io16 = np.where(d.ions == isotope.Ion('O16'))[0][0]
    io17 = np.where(d.ions == isotope.Ion('O17'))[0][0]
    io18 = np.where(d.ions == isotope.Ion('O18'))[0][0]
    o16 = d.abu_data[io16, ii]
    o17 = d.abu_data[io17, ii]
    o18 = d.abu_data[io18, ii]
    ax.plot(m, o17/o16, 'b+', label = r'$^{17}\mathrm{O}/^{16}\mathrm{O}$')
    ax.plot(m, o18/o16, 'g+', label = r'$^{18}\mathrm{O}/^{16}\mathrm{O}$')

    in14 = np.where(d.ions == isotope.Ion('N14'))[0][0]
    in15 = np.where(d.ions == isotope.Ion('N15'))[0][0]
    n14 = d.abu_data[in14, ii]
    n15 = d.abu_data[in15, ii]
    ax.plot(m, n15/n14, 'c+', label = r'$^{15}\mathrm{N}/^{14}\mathrm{N}$')

    ax.legend(loc = 'best', numpoints = 1, handlelength=1, handletextpad=.25)
    f.tight_layout()
    plt.draw()

def psn_minit(m):
    mi = (m - 65) * (260 - 140) / (133 - 65) + 140
    return mi

def elratplot_he2sn():
    import stardb, isotope
    import matplotlib.pyplot as plt

    d = stardb.StarDB('/home/alex/kepler/znuc/he2sn.HW02.star.el.y.stardb.gz')
    ii = np.arange(d.nstar)
    m = np.array([d.field_data['mass'][j] for j in ii])

    f = plt.figure()
    ax = f.add_subplot(111)
    ax2 = ax.twinx()

    ax.set_xscale('linear')
    ax.set_yscale('log')
    # ax2.set_yscale('log')


    imn = np.where(d.ions == isotope.Ion('Mn'))[0][0]
    ife = np.where(d.ions == isotope.Ion('Fe'))[0][0]
    ymn = d.abu_data[imn, ii]
    yfe = d.abu_data[ife, ii]

    ax2.plot(m, m*yfe*56, 'g+')
    ax2.set_ylabel(r'approximate Fe ejecta mass / solar mass', color = 'g')

    ax.plot(m, ymn/yfe, 'r+', label = r'$\mathrm{Mn}/\mathrm{Fe}$', color = 'r')
    ax.set_ylabel(r'number ratio', color = 'r')

    ax.set_xlabel(r'helium core mass / solar masses')
    xlim = [min(m)-1, max(m)+1]
    ax.set_xlim(xlim)

    ax.legend(loc = 'best', numpoints = 1, handlelength=1, handletextpad=.25)
    f.tight_layout()
    plt.draw()

def elratplot(E=(1.2, 1.8, 3),
              mix=0.015,
              fontsize='large',
              figsize=(8,6),
              ratio = 'O/C',
              target = '/home/alex/LaTeX/SM0313-6708-Bessell/f4.pdf',
              yscale = '[]'):
    import stardb, isotope
    import matplotlib.pyplot as plt


    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111)

    ax.set_xscale('linear')

    if yscale == '[]':
        ytitle = r'[{}]'
        s = SolAbu('As12')
        if ratio == 'C/O':
            srat = s.Y('C') / s.Y('O')
        else:
            srat = s.Y('O') / s.Y('C')
        ax.set_yscale('linear')
    else:
        ytitle = r'{} number ratio'
        ax.set_yscale('log')

    ax.set_xlabel(r'initial mass / solar masses', fontsize=fontsize)
    ax.set_ylabel(ytitle.format(ratio), fontsize=fontsize)

    # SN data

    d = stardb.StarDB('/home/alex/kepler/znuc/znuc.S4.star.el.y.stardb.gz')
    if not is_iterable(E):
        E = (E,)
    labels = []
    artist = []
    for e,c in zip(E,['r+', 'g1', 'b2']):
        ii = d.get_star_slice(energy=e, mixing=mix)
        ms = np.array([d.field_data['mass'][j] for j in ii])
        ic = np.where(d.ions == isotope.Ion('C'))[0][0]
        io = np.where(d.ions == isotope.Ion('O'))[0][0]
        xc = d.abu_data[ic, ii]
        xo = d.abu_data[io, ii]
        if ratio == 'C/O':
            y = xc / xo
        else:
            y = xo / xc
        if yscale == '[]':
            y = np.log10(y / srat)
        artist += ax.plot(ms, y, c)
        labels.append('CCSN, E = {:3.1f} B, mix = {:4.1f}\%'.format(e, mix*100))
        try:
            ys = np.append(ys, y)
        except:
            ys = y

    # pair-SN data

    d = stardb.StarDB('/home/alex/kepler/znuc/he2sn.HW02.star.el.y.stardb.gz')
    ii = np.arange(d.nstar)
    mp = np.array([d.field_data['mass'][j] for j in ii])
    mp = psn_minit(mp)

    ic = np.where(d.ions == isotope.Ion('C'))[0][0]
    io = np.where(d.ions == isotope.Ion('O'))[0][0]
    xc = d.abu_data[ic, ii]
    xo = d.abu_data[io, ii]
    if ratio == 'C/O':
        y = xc / xo
    else:
        y = xo / xc
    if yscale == '[]':
        y = np.log10(y / srat)

    # artist += ax.plot(mp, y, 'ro', mfc='none', mec = 'r')
    artist += ax.plot(mp, y, 'r.', ms = 8)
    labels.append('pair instability supernova')

    xscale = 'log'
    m = np.append(ms, mp)
    xlim = np.array([min(m), max(m)])
    if xscale == 'linear':
        xlim += (np.max(m)-np.min(m)) * 0.025 * np.array([-1,1])
    else:
        xlim *= np.exp(np.log(np.max(m)/np.min(m)) * 0.025 * np.array([-1,1]))
    ax.set_xlim(xlim)
    ax.set_xscale(xscale)
    ax.xaxis.set_major_formatter(MassFormatter())

    y = np.append(ys, y)
    ylim = np.array([min(y), max(y)])
    if yscale == '[]':
        ylim +=  (np.max(y) - np.min(y)) * 0.025 * np.array([-1,1])
    else:
        ylim *=  np.exp(np.log(np.max(y)/np.min(y)) * 0.025 * np.array([-1,1]))
    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(MassFormatter())

    # oc_sm0313 = -0.25
    # val = oc_sm0313 + math.log10(ocs)

    oc_sm0313_err = 0.175
    err = oc_sm0313_err

    # val = +0.36
    val = 1.9
    if ratio == 'C/O':
        val = -val
    span = val * 10.**(np.array([-1,1]) * err)

    if yscale == '[]':
        val = np.log10(val / srat)
        span = np.log10(span / srat)

    # ax.axhline(10.**0.7, color = 'k')
    # ax.axhspan(*(10.**(0.7 + np.array([-1,1])*0.05)),
    le = ax.axhspan(*span,
                    color = '#dddddd')

    lv = ax.axhline(val,
                    lw = 4,
                    zorder = +1,
                    color = '#aaaaaa')

    labels.append('SM0313-6708')
    artist.append((le, lv))

    l = ax.legend(artist, labels,
                  loc = 'best',
                  numpoints = 1,
                  handlelength = 1,
                  handletextpad = 0.25,
                  fontsize = fontsize,
                  )

    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    f.tight_layout()
    plt.draw()

    if target is not None:
        f.savefig(target)

def Keller_Bessel_fit(mode = 1, save = False):
    from fit import Single

    starpath = os.path.expanduser('~/LaTeX/SM0313-6708-Bessell')
    datapath = os.path.expanduser('~/kepler/znuc')

    if mode == 2:
        starfile = os.path.join(starpath, 'SM0313-6708-Bessell-paper-tight.dat')
    else:
        starfile = os.path.join(starpath, 'SM0313-6708-Bessell-paper.dat')
    datafile = os.path.join(datapath, 'znuc2012.S4.star.el.y.stardb.gz')

    # change defaults to make nicer plot
    f = Single(
        filename = starfile,
        db = datafile,
        # z_exclude = [3, 24, 30],
        z_exclude = [3],
        # z_lolim = [21, 29],
        cdf = False,
        )

    print()
    print('     Index   Mass Energy Mixing Offset  chi**2')
    for i in range(20):
        sorted_stars = f.sorted_stars[i, 0]
        field_data = f.db.field_data[sorted_stars['index']]
        print('({:>2}) {:>5} {:>6.2f} {:>6.2f} {:>6.2f} {:>6.2f} {:>7.3f}'.format(
                i + 1,
                sorted_stars['index'],
                field_data['mass'],
                field_data['energy'],
                np.log10(field_data['mixing']),
                np.log10(sorted_stars['offset']),
                f.sorted_fitness[i],
            )
        )

    if save is True:
        savename = os.path.join(starpath, 'f4b.pdf')
    else:
        savename = None

    f.plot(
        figsize=(8, 6),
        ylim=(1.3, -10.5),
        savename = savename,
        fontsize = 'large',
        data_size = 3,
        )
    return f

# znuc text run for JINA-CEE
# kepgen.MakeRun(mass=60, yeburn=True, composition='zero', dirbase='~/kepler/znuc-jina-ca40', special = ('nugrid',))

def snplot():
    import matplotlib.pyplot as plt
    import color
    import kepdump

    dir0 = os.path.expanduser('~/kepler/znuc')
    run = 'z40'
    expl = 'F'
    dump_template = os.path.join(dir0, run, 'rerun', run + expl + '#{}s')

    f = plt.figure(figsize=(8, 6))
    ax = f.add_subplot(111)
    ax.set_xscale('log')

    n = 101

    c = color.isocolors(n)

    for i in range(n):
        filename = dump_template.format(i)
        d = kepdump.load(filename)
        ii = slice(1,-1)
        u = d.un[ii] * 1e-5
        r = d.rn[ii]
        ax.plot(r, u, color = c[i])

    ax.set_xlabel(r'radius (cm)')
    ax.set_ylabel(r'velocity (km/sec)')

    plt.draw()
