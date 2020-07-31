"""
Interface to read KEPLER str(ucture) files.
"""

import os.path
import datetime

import numpy as np

import physconst

from fortranfile import FortranReader
from human import byte2human
from human import time2human
from human import version2human
from logged import Logged

from utils import CachedAttribute, cachedmethod
from loader import loader, _loader

class StrData(Logged):
    """Interface to load KEPLER structure files."""

    def __init__(self,
                 filename,
                 zerotime = True,
                 silent = False):
        """
        Constructor; wants file name.
        """

        self.setup_logger(silent)
        self.filename = os.path.expandvars(os.path.expanduser(filename))
        self.file = FortranReader(self.filename, extension = 'str')
        self._load()
        self.file.close()
        self.close_logger(message = 'data loaded in')
        self.file = None


    def _load(self, zerotime = True):
        """
        Open file, call load data, time the load, print out diagnostics.
        """
        self.logger_file_info(self.file)
        starttime = self.get_timer()
        self.logger.info('time to open file:    {:s}'.format(time2human(starttime)))
        self.data=[]
        while not self.file.eof():
            record=StrRecord(self.file)
            self.data.append(record)
        self.nmodels = len(self.data)
        self.logger.info('version {:s}'.format(version2human(self.data[0].nvers)))
        self.logger.info('first model read {:>9d}'.format(int(self.data[ 0].ncyc)))
        self.logger.info(' last model read {:>9d}'.format(int(self.data[-1].ncyc)))

        self.nvers = self.data[0].nvers
        assert np.all(np.array([d.nvers for d in self.data]) == self.nvers), "different version sin same file"

        if zerotime:
            self.remove_zerotime()

    def remove_zerotime(self):
        """
        Detect and remove resets of time.
        """
        if self.nvers < 10002:
            zerotime = np.float64(0)
            time0 = self.data[0].time
            for i in range(1,self.nmodels):
                if self.data[i].time < time0:
                    zerotime = self.data[i-1].time
                    self.logger.info('@ model = {:8d} zerotime was set to {:12.5g}.'.format(
                        int(self.data[i].ncyc),
                        float(zerotime)))
                time0 = self.data[i].time
                self.data[i].time = time0 + zerotime
        else:
            time = self.dt.cumsum()
            time += self.data[0].time - self.data[0].dt
            for t,data in zip(time,self.data):
                data.time = t


    # some of these could be from a base class for all data ...
    @CachedAttribute
    def dt(self):
        """
        Return dt array.
        """
        return np.array([data.dt for data in self.data])

    @CachedAttribute
    def time(self):
        """
        Return time array.
        """
        return np.array([data.time for data in self.data])

    @cachedmethod
    def dn_zm(self, zm):
        """
        Get density trajectory for given mass coordinate (g).
        """
        dn = np.zeros(self.nmodels, dtype = np.float64)
        for i,data in enumerate(self.data):
            ii, = np.where(data.zm >= zm)
            if len(ii) > 2:
                dn[i] = data.dn[ii[1]]
                # should we add interpolation?
        return dn

    @cachedmethod
    def tn_zm(self, zm):
        """
        Get temperature trajectory for given mass coordinate (g).
        """
        tn = np.zeros(self.nmodels, dtype = np.float64)
        for i,data in enumerate(self.data):
            ii, = np.where(data.zm >= zm)
            if len(ii) > 2:
                tn[i] = data.tn[ii[1]]
                # should we add interpolation?
        return tn


load = loader(StrData, __name__ + '.load')
_load = loader(StrData, __name__ + '.load')

class StrRecord(Logged):
    """
    Individual cycle records of structure.

    Variables:
    ----------
    nvers - version number
    ncyc - cycle number
    timesec - current problem time (not including time offset) [s]
    dt - current time step for this cycle [s]
    jm - number of zones
    zm - mass coordinate on zone boundaries [g]
    rn - radius coordinate of zone boundaries [cm]
    tn - zone average temperature [K]
    dn - zone average density [g]
    """
    def __init__(self,file,data=True):
        self.file = file
        self.load(data)

    def load(self,data=True):
        f = self.file
        f.load()
        self.nvers = f.get_i4()
        self.ncyc = f.get_i4()
        self.time = f.get_f8n()
        if self.nvers >= 10002:
            self.dt = f.get_f8n()
        self.jm = f.get_i4()
        self.zm = f.get_f8n(self.jm + 1)
        self.rn = f.get_f8n(self.jm + 1)
        self.tn = f.get_f8n1d0(self.jm)
        if self.nvers >= 10001:
            self.dn = f.get_f8n1d0(self.jm)
        f.assert_eor()
