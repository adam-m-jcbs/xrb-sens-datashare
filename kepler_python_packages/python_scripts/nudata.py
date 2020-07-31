'''
Python Module to read neutrino data
'''

from fortranfile import FortranReader
import physconst
import os.path

import numpy as np

from logged import Logged
from utils import CachedAttribute, cachedmethod
from loader import loader, _loader

class NuData(Logged):
    """
    Interface to load KEPLER nu data files.
    """
    def __init__(self,
                 filename,
                 zerotime = True,
                 silent = False):
        """
        Constructor; requires file name.

        PARAMETERS:
        zerotime (True):
            incorporate zerotime in time
        silent (False):
            reduce output
        """
        self.setup_logger(silent)
        filename = os.path.expanduser(filename)
        self.filename = os.path.expandvars(filename)
        self.file = FortranReader(self.filename)
        self._load()
        self.file.close()
        self.close_logger()

    def _load(self, zerotime = True):
        """
        Open file, call load data, time the load, print out diagnostics.
        """
        self.setup_logger()
        self.logger_file_info(self.file)
        self.data=[]
        while not self.file.eof():
            record = NuRecord(self.file)
            self.data.append(record)
        self.nmodels = len(self.data)

        self.logger_load_info(self.data[0].nvers,
                              self.data[ 0].ncyc,
                              self.data[-1].ncyc,
                              self.nmodels)
        if zerotime:
            self._remove_zerotime()
        self.close_logger()

    def _remove_zerotime(self, verbose = True):
        """
        Detect and remove resets of time.
        """
        zerotime = np.float64(0)
        time0 = self.data[0].time
        for i in range(1,self.nmodels):
            if self.data[i].time < time0:
                zerotime = self.data[i-1].time
                if verbose:
                    self.logger.info('@ model = {:8d} zerotime was set to {:12.5g}.'.format(
                        int(self.data[i].ncyc),
                        float(zerotime)))
            time0 = self.data[i].time
            self.data[i].time = time0 + zerotime

    @CachedAttribute
    def time(self):
        """
        Return time array. [s]
        """
        return np.array([rec.time for rec in self.data])

    @CachedAttribute
    def dt(self):
        """
        Return dt array. [s]
        """
        return np.array([rec.dt for rec in self.data])

    @cachedmethod
    def timecc(self, offset = 0.):
        """
        Return time till core collapse. [s]

        Set to 'None' to keep last dt as offset.
        """
        dt = (self.dt).copy()
        if offset is not None:
            dt[-1] = offset
        return ((dt[::-1]).cumsum())[::-1]

    @CachedAttribute
    def ncyc(self):
        """
        Return list of cycle numbers
        """
        return np.array([rec.ncyc for rec in self.data])

    @CachedAttribute
    def jm(self):
        """
        Return list of grid sizes

        Note ::
          each record contains jm + 1 data entries in spatial direction
        """
        return np.array([rec.jm for rec in self.data])

    @CachedAttribute
    def xlumn(self):
        """
        Return neutrino luminosity (from KEPLER) [erg/s]
        """
        return np.array([rec.xlumn for rec in self.data])

    @CachedAttribute
    def xm(self):
        """
        Return zone masses. [g]

        xm[0] is mass of 'core', if any.
        """
        n = np.max(self.jm) + 1
        xm = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            xm[i, :rec.jm + 1] = rec.xm
        return xm

    @CachedAttribute
    def ym(self):
        """
        Return zone surface mass coordiantes. [g]

        Mass coordinate as counted from surface.
        """
        n = np.max(self.jm) + 1
        ym = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            ym[i, :rec.jm + 1] = rec.ym
        return ym

    @CachedAttribute
    def zm(self):
        """
        Return zone mass coordinates. [g]

        zm[0] is mass of 'core', if any.
        """
        n = np.max(self.jm) + 1
        zm = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            zm[i, :rec.jm + 1] = rec.zm
        return zm

    @CachedAttribute
    def rn(self):
        """
        Return zone radii. [cm]

        xm[0] is mass of 'core', if any.
        """
        n = np.max(self.jm) + 1
        rn = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            rn[i, :rec.jm + 1] = rec.rn
        return rn

    @CachedAttribute
    def dn(self):
        """
        Return zone densities. [g/cm**3]
        """
        n = np.max(self.jm) + 1
        dn = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            dn[i, :rec.jm + 1] = rec.dn
        return dn

    @CachedAttribute
    def tn(self):
        """
        Return zone temperatures. [K]
        """
        n = np.max(self.jm) + 1
        tn = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            tn[i, :rec.jm + 1] = rec.tn
        return tn

    @CachedAttribute
    def ye(self):
        """
        Return zone Ye.
        """
        n = np.max(self.jm) + 1
        ye = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            ye[i,:rec.jm + 1] = rec.ye
        return ye

load = loader(NuData, __name__ + '.load')
_load = _loader(NuData, __name__ + '.load')
loadnu = load

class NuRecord(object):
    """
    Load individual nu record from nu file.
    """
    def __init__(self, file, data = True):
        self.file = file
        self._load(data)

    def _load(self,data=True):
        f = self.file
        f.load()
        self.nvers = f.get_i4()
        self.ncyc = f.get_i4()
        self.time = f.get_f8()
        self.dt = f.get_f8n()
        self.jm = f.get_i4()
        self.xlumn = f.get_f8n()
        summ0 = f.get_f8n()
        self.rn0   = f.get_f8n()
        n = self.jm
        self.xm   = f.get_f8n1d0(n)
        self.tn   = f.get_f8n1d0(n)
        self.dn   = f.get_f8n1d0(n)
        self.ye   = f.get_f8n1d0(n)
        f.assert_eor()

        self.xm[0] = summ0

    @CachedAttribute
    def ym(self):
        """
        Return outer mass coordinate. [g]
        """
        ym = np.empty_like(self.xm)
        ym[-1] = 0.
        ym[:-1] = np.cumsum(self.xm[:0:-1])[::-1]
        return ym

    @CachedAttribute
    def zm(self):
        """
        Return mass coordinate. [g]

        This formula is the same as implemented in KEPLER.
        """
        zm = self.xm[0] + (self.ym[0] - self.ym)
        return zm

    @CachedAttribute
    def rn(self):
        """
        Return radius coordinate. [cm]
        """
        rn = np.empty_like(self.xm)
        rn[0] = self.rn0
        rn[1:] = rn[0] + np.cumsum((3 * self.xm[1:] / (4 * np.pi * self.dn[1:]))**(1/3))
        return rn
