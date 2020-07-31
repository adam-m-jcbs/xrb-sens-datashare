"""
Interface to generate KEPLER link files.
"""

import os
import os.path
import string

import keppar
from physconst import SEC, XMSUN, RSUN
import logging
import collections
import datetime
import numpy as np

from human import byte2human
from human import time2human
from human import version2human
from logged import Logged
from utils import CachedAttribute, cachedmethod

import uuid
import logdata
from isotope import ion, Isotope
from abuset import AbuData
from kepion import mapburn

class KepLink(Logged):
    """
    Class to write KEPLER link files.
    """

    def __init__(self, rn, xm, tn, dn, angwn, un, abu,
                 ncyc = 0,
                 timesec = 0.,
                 jm = None,
                 radius0 = 0.):
        super().__init__()
        self.rn = rn
        self.xm = xm
        self.tn = tn
        self.dn = dn
        self.aw = angwn
        self.un = un
        self.abu = abu
        self.ncyc = ncyc
        self.timesec = timesec
        self.jm = jm
        self.radius0 = radius0

    def _write(self, filename):
        version = 20000
        with open(filename, 'wt') as f:
            f.write('Version {:7d}\n'.format(version))
            f.write('{:d}\n'.format(self.ncyc))
            f.write('{:25.17e}\n'.format(self.timesec))
            f.write('{:d}\n'.format(self.jm))
            f.write('{:25.17e}\n'.format(self.radius0))
            for j in range(self.jm):
                row = [self.rn[j],
                       self.xm[j],
                       self.tn[j],
                       self.dn[j],
                       self.aw[j],
                       self.un[j]] +\
                      [self.abu[j,i] for i in range(19)]
                f.write(' '.join(['{:25.17e}'.format(float(val)) for val in row])+'\n')


    def squeeze(self, factor = 2, truncate = True):
        self.logger.info(f'Squeezing data by {factor}.')
        if factor == 1:
            return
        nsurf = self.jm % factor
        if nsurf > 0:
            xm1 = self.xm [-nsurf:]
            rn1 = self.rn [-nsurf:]
            un1 = self.un [-nsurf:]
            dn1 = self.dn [-nsurf:]
            tn1 = self.tn [-nsurf:]
            aw1 = self.aw [-nsurf:]
            abu1= self.abu[-nsurf:,:]

            self.xm = self.xm [:-nsurf]
            self.rn = self.rn [:-nsurf]
            self.un = self.un [:-nsurf]
            self.dn = self.dn [:-nsurf]
            self.tn = self.tn [:-nsurf]
            self.aw = self.aw [:-nsurf]
            self.abu= self.abu[:-nsurf,:]

        iiouter = slice(factor-1,None,factor)

        def nsum(x):
            dim = (-1, factor) + x.shape[1:]
            return np.sum(x.reshape(dim), axis = 1)

        xm = nsum(self.xm)
        self.dn = xm/nsum(self.xm/self.dn)
        self.rn = self.rn[iiouter]
        self.un = self.un[iiouter]
        self.tn = nsum(self.xm*self.tn)/xm
        self.aw = nsum(self.xm*self.aw)/xm # roughly
        self.abu = nsum(self.xm[:,np.newaxis]*self.abu)/xm[:,np.newaxis]
        self.xm = xm
        self.jm //= factor

        if not truncate and nsurf > 0:
            xmt = np.sum(xm1)
            self.dn = np.append(self.dn, xmt/np.sum(xm1/dn1))
            self.rn = np.append(self.rn, rn1[-1])
            self.un = np.append(self.un, un1[-1])
            self.tn = np.append(self.tn, np.sum(xm1*tn1)/xmt)
            self.aw = np.append(self.aw, np.sum(xm1*aw1)/xmt)
            self.abu = np.append(
                self.abu,
                (np.sum(xm1[:,np.newaxis]*abu1, axis=0)/xmt)[np.newaxis,:],
                axis=0)
            self.xm = np.append(self.xm, xmt)
            self.jm += 1

    def dezone(self, xm_max, zm0 = None, zm1 = None):
        found = True
        while found:
            found = False
            found_last = False
            zm = np.cumsum(self.xm) / XMSUN
            if zm0 is None:
                zm0 = -1.e99
            if zm1 is None:
                zm1 = +1.e99
            ii, = np.where(np.logical_and(zm > zm0, zm < zm1))
            for i in ii[::-1]:
                if found_last:
                    continue
                if self.xm[i] < xm_max:
                    if i == self.jm-1:
                        j = -1
                    elif i == 0:
                        j = +1
                    elif self.xm[i-1] > self.xm[i+1]:
                        j = +1
                    else:
                        j = -1
                    found = found_last = True
                    i0 = min(i,i+j)
                    i1 = max(i,i+j)
                    xm = self.xm[i0] + self.xm[i1]
                    self.dn[i0] = xm/(self.xm[i0]/self.dn[i0] + self.xm[i1]/self.dn[i1])
                    self.rn[i0] = self.rn[i1]
                    self.tn[i0] = (self.xm[i0]*self.tn[i0] + self.xm[i1]*self.tn[i1])/xm
                    self.aw[i0] = (self.xm[i0]*self.aw[i0] + self.xm[i1]*self.aw[i1])/xm # roughly
                    self.un[i0] =  self.un[i1]
                    self.abu[i0,:] =  (self.xm[i0,np.newaxis]*self.abu[i0,:] + self.xm[i1,np.newaxis]*self.abu[i1,:])/xm
                    self.xm[i0] = xm
                    self.jm -= 1

                    self.xm [i1:-1] = self.xm [i1+1:]
                    self.dn [i1:-1] = self.dn [i1+1:]
                    self.rn [i1:-1] = self.rn [i1+1:]
                    self.tn [i1:-1] = self.tn [i1+1:]
                    self.aw [i1:-1] = self.aw [i1+1:]
                    self.un [i1:-1] = self.un [i1+1:]
                    self.abu[i1:-1,:] = self.abu[i1+1:,:]

            self.xm = self.xm [:self.jm]
            self.dn = self.dn [:self.jm]
            self.rn = self.rn [:self.jm]
            self.tn = self.tn [:self.jm]
            self.aw = self.aw [:self.jm]
            self.un = self.un [:self.jm]
            self.abu= self.abu[:self.jm,:]


    def write(self, filename, maxzones = 1983):
        factor = self.jm // maxzones + 1
        if factor > 1:
            self.squeeze(factor)
        self._write(filename)


    def xmplot(self, radius = False):
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if radius:
            x = self.rn
            ax.set_xscale('log')
        else:
            x = np.cumsum(self.xm) / XMSUN
            ax.set_xscale('linear')

        ax.plot(x, np.log10(self.xm), label = 'xm')
        ax.set_xlabel('mass coordinate (solar masses)')
        ax.set_ylabel('log zone mass (g)')

        r = np.ndarray(self.jm+1)
        r[0] = 0
        r[1:] = self.rn
        check = 4*np.pi*(r[1:]**3-r[:-1]**3)*self.dn/3
        ax.plot(x, np.log10(check), label = 'xm check')

        ax.legend(loc='best')
        fig.tight_layout()
        plt.draw()

    def fix_rn_from_dn_xm(self):
        dv = self.xm / self.dn
        v0 = 4 * np.pi * self.radius0**3 / 3
        v = v0 + np.cumsum(dv)
        self.rn = (v * 3 / (4 * np.pi))**(1/3)

    def write_burngen(self, filename):
        self.setup_logger()
        self.abub.write_burngen(filename)
        self.close_logger(timing=f'BURN generator written to "{filename}" in')


class ProfileGuillochon(KepLink):
    """
    Read data from James Guillochon

    rewrite using a class GuillchonData if proejct is ever to go ahead ...
    """
    def __init__(self, filename):
        f = open(filename, 'r')
        lines = f.read().split('\n')
        nfield = 17
        assert len(lines[0]) == 16*nfield
        fields = [(lines[0][16*i:16*(i+1)]).strip() for i in range(nfield)]
        xfields = ['mass', 'radius', 'radial vel.', 'h1', 'he3', 'he4', 'c12', 'n14', 'o16', 'ne20', 'mg24', 'si28', 'fe56', 'density', 'temperature', 'int. ener.', 'ang. vel.']
        assert fields == xfields
        jm = len(lines)-1
        if len(lines[-1]) == 0:
            jm -= 1

        val = np.array([[lines[j][i*16:(i+1)*16] for i in range(nfield)] for j in range(1,jm+1)], dtype='f8')

        xm = val[:,0]
        rn = val[:,1]
        # to be finished ....

class MesaDump(Logged):
    def __init__(self, filename):
        self._load(filename)

    class _Data(object):
        def __init__(self, parent):
            self._parent = parent
        def __getitem__(self, item):
            index = self._parent._columns[item]
            return self._parent._data[::-1, index]
        __call__ = __getitem__

    def _load(self, filename, header_only = False):
        self.setup_logger(False)
        with open(filename, 'rb') as f:
            lines = [f.readline().decode('ASCII') for i in range(6)]
            self.header = {x : float(v) for x,v in zip(
                    lines[1].split(),
                    lines[2].split())}
            if header_only:
                return
            self._columns = {name : int(col)-1 for col,name in zip(
                    lines[4].split(),
                    lines[5].split())}
            self._data = np.genfromtxt(f)
        self.data = self._Data(self)
        self.filename = filename
        self.close_logger(timing = "{} loaded in ".format(filename))

    def __getitem__(self, item):
        return self.data[item]

    __call__ = __getitem__

    def get_isotopes(self):
        isotopes = list()
        excludes = set()
        for c in self._columns:
            if c in excludes:
                continue
            i = ion(c)
            if isinstance(i, Isotope):
                isotopes.append(i)
        return isotopes

    @property
    def abub(self):
        ions = self.get_isotopes()
        jm = int(self.header['num_zones'])
        ppnb = np.ndarray((jm, len(ions)))
        for i,x in enumerate(ions):
            ppnb[:, i] = self.data[x.mesa()]
        return AbuData(
            data = ppnb,
            ions = ions,
            molfrac = False,
            )

    @property
    def approx19(self):
        return mapburn(self.abub)

class ProfileMesa20(KepLink):
    """
    Read data from Mesa / Sam Jones
    """
    def __init__(self, filename):
        mesa = MesaDump(filename)

        self.timesec = mesa.header['star_age'] * SEC
        self.ncyc = int(mesa.header['model_number'])
        self.radius0 = 0.
        self.velocity0 = 0.
        self.xm = mesa['mass'] * XMSUN
        self.xm[1:] = self.xm[1:] - self.xm[:-1]
        self.rn = mesa['radius'] * RSUN
        self.dn = 10.**mesa['logRho']
        self.tn = 10.**mesa['logT']
        self.aw = np.zeros_like(self.xm)
        self.un = mesa['velocity']
        self.jm = int(mesa.header['num_zones'])
        assert self.jm == len(self.xm)
        self.set_abu(mesa)

    def set_abu(self, mesa):
        abu = np.zeros((self.jm, 19), dtype = np.float64)
        abu[:, 0] = mesa['neut']
        abu[:, 1] = mesa['h1']
        abu[:, 2] = mesa['prot']
        abu[:, 3] = mesa['he3']
        abu[:, 4] = mesa['he4']
        abu[:, 5] = mesa['c12']
        abu[:, 6] = mesa['n14']
        abu[:, 7] = mesa['o16']
        abu[:, 8] = mesa['ne20']
        abu[:, 9] = mesa['mg24']
        abu[:,10] = mesa['si28']
        abu[:,11] = mesa['s32']
        abu[:,12] = mesa['ar36']
        abu[:,13] = mesa['ca40']
        abu[:,14] = mesa['ti44']
        abu[:,15] = mesa['cr48']
        abu[:,16] = mesa['fe52']
        abu[:,17] = mesa['fe54'] + mesa['fe56']
        abu[:,18] = mesa['ni56']
        self.abu = abu

class ProfileMesa(ProfileMesa20):
    """
    Read data from Mesa / Alejandro Vigna Gomez
    """
    def __init__(self, filename):
        mesa = MesaDump(filename)

        self.timesec = mesa.header['star_age'] * SEC
        self.ncyc = int(mesa.header['model_number'])
        self.radius0 = 0.
        self.velocity0 = 0.
        self.xm = mesa['mass'] * XMSUN
        self.xm[1:] = self.xm[1:] - self.xm[:-1]
        self.rn = 10.**mesa['logR'] * RSUN
        self.dn = 10.**mesa['logRho']
        self.tn = 10.**mesa['logT']
        self.aw = np.zeros_like(self.xm)
        self.un = mesa['velocity']
        self.jm = int(mesa.header['num_zones'])
        assert self.jm == len(self.xm)
        self.set_abu(mesa)

    def set_abu(self, mesa):
        self.abu = mesa.approx19.data
        self.abub = mesa.abub

from stern import BECData

class ProfileBEC(KepLink):
    def __init__(self, filename, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            kwargs = dict(kwargs)
            kwargs['nr'] = -1

        bec = BECData(filename, *args, **kwargs)

        self.timesec = bec.time
        self.ncyc = bec.modell
        self.radius0 = bec.r[0]
        self.velocity0 = bec.u[0]
        self.xm = bec.dm[:-1]
        self.rn = bec.r[1:]
        self.dn = bec.ro[:-1]
        self.tn = bec.t[:-1]
        self.aw = bec.aw[1:]
        self.un = bec.u[1:]
        self.jm = bec.n-1
        assert self.jm == len(self.rn)
        abu = np.zeros((self.jm+1, 19), dtype = np.float64)

        abu[:, 0] = bec.abu('n') # n
        abu[:, 1] = bec.abu('h1') # h1
        abu[:, 2] = 0 # prot
        abu[:, 3] = (bec.abu('d') + bec.abu('he3') + bec.abu('li6') +
                     bec.abu('li7') + bec.abu('be7') + bec.abu('be9') +
                     bec.abu('b8') + bec.abu('b10') + bec.abu('b11') +
                     bec.abu('c11')) # he3
        abu[:, 4] = bec.abu('he4') # he4
        abu[:, 5] = (bec.abu('c12') + bec.abu('c13') + bec.abu('n12')) # c12

        abu[:, 6] = (bec.abu('n14') + bec.abu('n15')) # n14
        abu[:, 7] = (bec.abu('o16') + bec.abu('o17') + bec.abu('o18') +
                     bec.abu('f19')) # o16
        abu[:, 8] = (bec.abu('ne20') + bec.abu('ne21') + bec.abu('ne22') +
                     bec.abu('na23')) # ne20
        abu[:, 9] = (bec.abu('mg24') + bec.abu('mg25') + bec.abu('mg26') +
                     bec.abu('al26') + bec.abu('al27')) # mg24
        abu[:,10] = (bec.abu('si28') + bec.abu('si29') + bec.abu('si30')) # si28
        abu[:,11] = 0 # s32
        abu[:,12] = 0 # ar36
        abu[:,13] = 0 # ca40
        abu[:,14] = 0 # ti44
        abu[:,15] = 0 # cr48
        abu[:,16] = 0 # fe52
        abu[:,17] = bec.abu('fe56') # fe54
        abu[:,18] = 0 # ni56
        self.abu = abu[:-1,:]

class NicoleLink(KepLink):
    """
    Read in data file from Nicole Rodrigues.
    """

    xmap = {
        0: (0,),
        1: (1,),
        4: (2,),
        5: (3,),
        6: (4,),
        7: (5,),
        8: (6,),
        9: (7,),
        10: (8,),
        11: (9,),
        12: (10,),
        13: (11,),
        14: (12,),
        15: (13,),
        16: (14,),
        17: (15, 17, 18),
        18: (16,),
        }

    def __init__(self, filename):
        import astropy.io.ascii as aascii
        data = aascii.read(filename)

        self.timesec = 0.
        self.ncyc = 0
        self.radius0 = 0.
        self.velocity0 = 0.
        self.xm = data['col2'].data*XMSUN
        self.rn = data['col4'].data
        self.dn = data['col5'].data
        self.tn = data['col6'].data
        self.aw = np.zeros_like(self.rn)
        self.un = np.zeros_like(self.rn)
        self.jm = len(self.xm)

        abu = np.zeros((self.jm, 19), dtype = np.float64)
        for i,jj in self.xmap.items():
            for j in jj:
                abu[:,i] = data['col{:d}'.format(j+8)].data

        self.abu = abu
        self.fix_rn_from_dn_xm()
