"""
Read in ev data file.

           REAL*8 Mass
           write(unit) MASS

           OPEN(UNIT=26,STATUS='OLD', ACCESS='APPEND',FORM=
     &          'UNFORMATTED',file=evolnfile)

           integer*4 NMODEL
           real*8 TIMEOUTDP, OUTPUT(84)
           WRITE(26) NMODEL,TIMEOUTDP,(OUTPUT(IOUTPUT),IOUTPUT=1,84)
           CLOSE(UNIT=26)
"""

from fortranfile import FortranReader
import physconst
import os.path

from human import byte2human
from human import time2human
from human import version2human
from logged import Logged
from utils import CachedAttribute, cachedmethod, make_cached_attribute
from loader import loader, _loader


import sys
import datetime
import numpy as np

import functools

class EVData(Logged):
    """Interface to load EV model files."""
    def __init__(self,
                 filename,
                 silent = False,
                 idx = 21):
        """
        Constructor; wants file name.
        """
        self.setup_logger(silent)
        filename = os.path.expanduser(filename)
        self.filename = os.path.expandvars(filename)
        self.file = FortranReader(self.filename, byteorder = "<")
        self._load(idx)
        self.file.close()
        self.close_logger()

    def _load(self, idx = 21):
        """Open file, call load data, time the load, print out diagnostics."""
        start_time = datetime.datetime.now()
        self.logger.info('Loading {} ({})'.format(
                self.file.filename,
                byte2human(self.file.filesize)))

        # load initial mass
        self.file.load()
        self.imass = self.file.get_f8n()
        self.file.assert_eor()

        self.data = []
        while not self.file.eof():
            record=EVRecord(self.file)
            self.data.append(record)
            if len(self.data) > 10:
                   break
        self.nmodels = len(self.data)

        end_time = datetime.datetime.now()
        load_time = end_time - start_time
        self.logger_load_info(
            0,
            self.data[ 0].nmodel,
            self.data[-1].nmodel,
            self.nmodels,
            load_time)

        def var(self, idx):
            print('being called', idx)
            return np.array([x.output[idx] for x in self.data])

        #       self.index(21,'xh','central XH')

        make_cached_attribute(self,
                             functools.partial(var,idx=idx),
                             'xh','central XH')

    @CachedAttribute
    def timeout(self):
        "time (s)."
        return np.array([x.timeout for x in self.data])

    @CachedAttribute
    def nmodel(self):
        "model number."
        return np.array([x.nmodel for x in self.data], dtype=np.int32)

    @CachedAttribute
    def lum(self):
        "luminosity (L_sun)."
        return np.array([x.output[13] for x in self.data])

    @CachedAttribute
    def temp(self):
        "log temperature (K)."
        return np.array([x.output[19] for x in self.data])

class EVRecord(object):
    def __init__(self, file, data = True):
        self.file = file
        self.load(data)

    def load(self,data=True):
        f = self.file
        f.load()
        self.nmodel = f.get_i4()
        self.timeout = f.get_f8n()
        self.output = f.get_f8n(84)
        f.assert_eor()

load = loader(EVData, 'loadev')
_load = _loader(EVData, 'loadev')
