
from fortranfile import FortranReader
import physconst
import os.path

from human import byte2human
from human import time2human
from human import version2human

import datetime
import numpy as np
from logged import Logged
import utils

class EntData(Logged):
    """
    Interface to load KEPLER ent data files.
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
        start_time = datetime.datetime.now()
        self.logger_file_info(self.file)
        self.data=[]
        while not self.file.eof():
            record = EntRecord(self.file)
            self.data.append(record)
        self.nmodels = len(self.data)
        end_time = datetime.datetime.now()
        load_time = end_time - start_time

        self.logger_load_info(self.data[0].nvers,
                              self.data[ 0].ncyc,
                              self.data[-1].ncyc,
                              self.nmodels,
                              load_time.total_seconds())
        if zerotime:
            self._remove_zerotime()


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


    @utils.CachedAttribute
    def time(self):
        """
        Return time array.
        """
        return np.array([rec.time for rec in self.data])

    @utils.CachedAttribute
    def dt(self):
        """
        Return dt array.
        """
        return np.array([rec.dt for rec in self.data])

    @utils.cachedmethod
    def timecc(self, offset = 0.):
        """
        Return time till core collapse

        Set to 'None' to keep last dt as offset.
        """
        dt = (self.dt).copy()
        if offset is not None:
            dt[-1] = offset
        return ((dt[::-1]).cumsum())[::-1]



class EntRecord(object):
    """
    Load individual ent record from ent file.
    """
    def __init__(self, file, data = True):
        self.file = file
        self._load(data)

    def _load(self,data=True):
        f = self.file
        f.load()
        self.nvers = f.get_i4()
        self.ncyc = f.get_i4()
        self.time = f.get_f8n()
        self.dt = f.get_f8n()
        self.nlev = f.get_i4()
        n = self.nlev + 1
        self.zm   = f.get_f8n(n)
        self.rn   = f.get_f8n(n)
        self.sig  = f.get_f8n(n)
        self.sige = f.get_f8n(n)
        self.ye   = f.get_f8n(n)
        self.eta  = f.get_f8n(n)
        f.assert_eor()
