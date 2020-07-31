"""
Plotting directly from MESA text files

Plots 1 & 2 work in part.

maybe there is different restart dumps we should read?
"""


from keplink import MesaDump as Mess
import time
import os.path
import numpy as np
from .plot import plot as pt
from .datainterface import DataInterface
from physconst import RSUN, XMSUN

class MesaData(DataInterface):
    def __init__(self, dump = None):
        if not isinstance(dump, Mess):
            dump = Mess(dump)
        self.dump = dump
        self.jm = int(dump.header['num_zones'])
        self.jmcalc = self.jm
        self.ncyc = int(dump.header['model_number'])
        self.time = int(dump.header['star_age']) * 356.2425 * 3600 #'time_seconds'
        self.teff = dump.header['Teff']
        self.radius = dump.header['photosphere_r']
        self.xlum = dump.header['photosphere_L']
        self.xlumn = dump.header['power_neu']
        self.iwinsize = 8000600
        self.dt = dump.header['time_step'] * 356.2425 * 3600

        # dummy stuff for now
        self.idtcon = 0
        self.jdtc = [0]
        self.nameprob = os.path.splitext(os.path.basename(self.dump.filename))[0]
        self.nburnz = 0
        self.ninvl = 0
        self.inburn = 1
        self.iter = 0
        self.ent = 0.
        self.irtype = 3
        self.ipixtype = 1
        self.vlimset = -1.
        self.wimp = 0
        self.abunlim = 1.e-12
        self.iplotb = 3

        # APPROX
        self.ppn = np.zeros((19, self.jm + 2))
        self.ppn[:, 1:-1] = self.dump.approx19.abu(molfrac = True).transpose()
        self.ionn = ((np.arange(20) + 1) % 20).reshape((20,1))
        self.numi = np.array([19])
        self.netnum = np.tile(1, self.jm + 2)
        self.netnum[[0,-1]] = 0
        self.aion = self.dump.approx19.A

        # BURN
        nionb = self.dump.abub.ions.__len__()
        self.ppnb = np.zeros((nionb, self.jm + 2))
        self.ppnb[:, 1:-1] = self.dump.abub.abu(molfrac = True).transpose()
        self.ionnb = (np.arange(nionb) + 1).reshape((nionb,1))
        self.numib = np.array([nionb])
        self.ionsb = self.dump.abub.ions
        self.aionb = self.dump.abub.A
        self.zionb = self.dump.abub.Z
        self.netnumb = np.tile(1, self.jm + 2)
        self.netnumb[[0,-1]] = 0


    @property
    def filename(self):
        return os.path.basename(self.dump.filename)
    @property
    def runpath(self):
        return os.path.dirname(os.path.abspath(self.dump.filename))
    @property
    def datatime(self):
        return time.asctime(time.localtime(os.path.getctime(self.dump.filename)))

    def __getattr__(self, var):
        val = None
        # linear interface values
        _map = dict(un = 'velocity')
        key = self.dump._columns.get(_map.get(var, None), None)
        if key is not None:
            val = np.ndarray(self.jm + 2)
            val[1:-1] = self.dump._data[::-1, key]
            val[[0, -1]] = (0, np.nan)
            return val
        # linear zone values
        _map = dict(tn = 'T', dn = 'Rho', pn = 'P', sn = 'eps_nuc',
                    etan = 'eta', mu = 'mu')
        key = self.dump._columns.get(_map.get(var, None), None)
        if key is not None:
            val = np.ndarray(self.jm + 2)
            val[1:-1] = self.dump._data[::-1, key]
            val[[0, -1]] = (np.nan,) * 2
            return val
        # log interface values
        _map = dict()
        key = self.dump._columns.get(_map.get(var, None), None)
        if key is not None:
            val = np.ndarray(self.jm + 2)
            val[1:-1] = 10.**self.dump._data[::-1, key]
            val[[0, -1]] = (0, np.nan)
            return val
        # log zone values
        _map = dict(tn = 'logT', dn = 'logRho', pn = 'logP', )
        key = self.dump._columns.get(_map.get(var, None), None)
        if key is not None:
            val = np.ndarray(self.jm + 2)
            val[1:-1] = 10.**self.dump._data[::-1, key]
            val[[0, -1]] = (np.nan,) * 2
            return val
        if var == 'rn':
            key = self.dump._columns.get('logR', None)
            if key is not None:
                val = np.ndarray(self.jm + 2)
                val[1:-1] = 10.**self.dump._data[::-1, key] * RSUN
                val[[0, -1]] = (0, np.nan,)
                return val
            key = self.dump._columns.get('radius', None)
            if key is not None:
                val = np.ndarray(self.jm + 2)
                val[1:-1] = self.dump._data[::-1, key] * RSUN
                val[[0, -1]] = (0, np.nan,)
                return val
        if var == 'zm':
            key = self.dump._columns.get('mass', None)
            if key is not None:
                val = np.ndarray(self.jm + 2)
                val[1:-1] = self.dump._data[::-1, key] * XMSUN
                val[[0, -1]] = (0, np.nan,)
                return val
        if var == 'xm':
            key = self.dump._columns.get('dm', None)
            if key is not None:
                val = np.ndarray(self.jm + 2)
                val[1:-1] = self.dump._data[::-1, key] * XMSUN
                val[[0, -1]] = (np.nan, np.nan,)
                return val
        if var == 'snn':
            k1 = self.dump._columns.get('eps_nuc', None)
            k2 = self.dump._columns.get('non_nuc_neu', None)
            if k1 is not None and k2 is not None:
                snn = self.dump._data[::-1, k1] - self.dump._data[::-1, k2]
                val = np.ndarray(self.jm + 2)
                val[1:-1] = snn
                val[[0, -1]] = (np.nan,) * 2
                return val

        if var in ('sv', 'sadv', 'bfvisc', 'bfdiff', 'bfbr', 'bfbt', 'bfviscef', 'bfdiffef',):
            return np.zeros(self.jm + 2)
        raise AttributeError(var)
    @property
    def idtcsym(self):
        return ['mes']
    @property
    def iconv(self):
        return np.zeros(self.jm + 2, dtype = np.int)
    @property
    def entropies(self):
        """
        return array of entropes

        stored data only contains total entropy
        """
        data = np.zeros((self.qparm.jm+2, 6), dtype = np.float64)
        # set mesa values - data[:,0] = self.stot
        return data


def plot(dump = None, plot = None, **kwargs):
    """
    make kepler plot from dump
    """
    dd = MesaData(dump)

    if plot is None:
        plot = dump.ipixtype

    # this is to be adjusted for plot types
    return pt(dd, plot, **kwargs)
