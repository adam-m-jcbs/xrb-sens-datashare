"""
Read light curve data files.
"""

from fortranfile import FortranReader
import physconst
import os.path

from human import byte2human
from human import time2human
from human import version2human
from logged import Logged
from utils import CachedAttribute, cachedmethod, TextFile
from loader import loader, _loader

import datetime
import numpy as np
import gzip

class LCData(Logged):
    """Interface to load KEPLER light curve files."""
    _extension = 'lc'
    def __init__(self,
                 filename,
                 zerotime = True,
                 silent = False):
        """
        Constructor; wants file name.
        """
        self.setup_logger(silent)
        filename = os.path.expanduser(filename)
        self.filename = os.path.expandvars(filename)
        self.file = FortranReader(self.filename)
        self._load()
        self.file.close()
        self.close_logger()

    def _load(self, zerotime = True):
        """Open file, call load data, time the load, print out diagnostics."""
        start_time = datetime.datetime.now()
        self.logger.info('Loading {} ({})'.format(
                self.file.filename,
                byte2human(self.file.filesize)))
        self.data = []
        while not self.file.eof():
            record=LCRecord(self.file)
            self.data.append(record)
        self.nmodels = len(self.data)
        if zerotime:
            self.remove_zerotime()

        # NEED TO ADD _CLEAN ROUTINE - E.G., FROM WINDDATA.PY

        end_time = datetime.datetime.now()
        load_time = end_time - start_time
        self.logger_load_info(
            self.data[ 0].nvers,
            self.data[ 0].ncyc,
            self.data[-1].ncyc,
            self.nmodels,
            load_time)

    def remove_zerotime(self):
        """Detect and remove resets of time."""
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

    @CachedAttribute
    def time(self):
        "time (s)."
        return np.array([x.time for x in self.data])

    @CachedAttribute
    def rn(self):
        "outer zone radius (cm)."
        return np.array([x.rn for x in self.data])

    @CachedAttribute
    def un(self):
        "outer zone velicty (cm/sec)."
        return np.array([x.un for x in self.data])

    @CachedAttribute
    def xln(self):
        "outer zones luminosity (erg/sec)."
        return np.array([x.xln for x in self.data])

    @CachedAttribute
    def teffn(self):
        "outer zones effictive temperature (K)."
        fac = physconst.Kepler.a * physconst.Kepler.c * np.pi
        return (self.xln / (fac * self.rn**2))**0.25

    @CachedAttribute
    def xlphot(self):
        "luminosity (erg/s)."
        return np.array([x.xlphot for x in self.data])

    @CachedAttribute
    def teff(self):
        "effective temperature (K)."
        return np.array([x.teff for x in self.data])

    @CachedAttribute
    def vphot(self):
        "photospehere velocity (cm/s)."
        return np.array([x.vphot for x in self.data])

    @CachedAttribute
    def reff(self):
        "photospehere radius (cm)."
        fac = physconst.Kepler.a * physconst.Kepler.c * np.pi
        ii = np.where((self.teff > 0) & (self.xlphot > 0))[0]
        r = np.zeros_like(self.teff)
        r[ii] =  np.sqrt(self.xlphot[ii]/(self.teff[ii]**4 * fac))
        return r

    @CachedAttribute
    def xlphotmax(self):
        "maximum luminoisity outside photosphere (erg/s)."
        rp = self.reff
        ln = np.empty_like(rp)
        for j,x in enumerate(self.data):
            i = np.where(x.rn > rp[j])[0]
            if len(i) == 0:
                ln[j] = x.xln[-1]
            else:
                i = i[0]
                ln[j] = np.amax(x.xln[i:])
                if i > 0:
                    f = (rp[j] - x.rn[i-1]) / (rn[i] - rn[i-1])
                    ln[j] = max(ln[j], f * x.xln[i] + (1 - f) * xln[i-1])
        return ln

    @CachedAttribute
    def radius(self):
        "radius (cm)."
        return np.array([x.radius for x in self.data])

    @CachedAttribute
    def xlum(self):
        "luminosity (erg/s)."
        return np.array([x.xlum for x in self.data])

    @CachedAttribute
    def ncyc(self):
        "cycle number."
        return np.array([x.ncyc for x in self.data])

    @CachedAttribute
    def lbol(self):
        """
        bolometric luminosity estimator

        Compute max of luminosity outside photosphere.

        This can be useful for radioactive decay in SN simulations
        after the ejecta have become optically thin.  The code may
        formally still do some PdV work in the gas as it expands, but
        that may be a poor approcimation, hence we tak the maximum not
        the value from the outermost zone.
        """
        ii = [np.max(np.where(x.rn > x.radius)[0] - 1) for x in self.data]
        lbol = np.array([np.max(x.xln[i:]) for (x, i) in zip(self.data, ii)])
        return lbol


    def write_lc_txt(self,
                     filename = 'lc.txt',
                     silent = False,
                     **kwargs):
        """
        Write lc data to lc.txt file.

        Use 'compress' keyword (forwared to TextFile) to create compressed text files.
        Flag and be 'True' | 'False', or specify format ('gz', 'xz', 'bz2').
        Default is no compression.
        Defaul compression (if 'True') is 'xz'.
        """

        # SHOULD BE UPDATED, E.G., FROM WINDDATA.PY

        self.setup_logger(silent)
        version = 10100
        with TextFile(filename, mode = 'w', **kwargs) as f:
            f.write('VERSION {:6d}\n'.format(version))
            f.write('{:>25s}{:>25s}{:>25s}\n'.format(
                'time (s)',
                'R_eff (cm)',
                'L (erg/s)'))
            reff = self.reff
            time = self.time
            xlphot = self.xlphot
            for i in range(len(self.data)):
                f.write('{:25.17e}{:25.17e}{:25.17e}\n'.format(
                float(time[i]),
                float(reff[i]),
                float(xlphot[i])))
        self.close_logger()

load = loader(LCData, 'lcdata')
_load = _loader(LCData, 'lcdata')
loadlc = load

class LCRecord(object):

    def __init__(self, file, data = True):
        self.file = file
        self.load(data)

    def load(self,data=True):
        f = self.file
        f.load()
        self.nvers = f.get_i4()
        self.ncyc = f.get_i4()
        self.time = f.get_f8n()
        if self.nvers >= 10101:
            self.dt = f.get_f8n()
        (self.teff,
         self.radius,
         self.xlum) = f.get_f8n(3)
        if self.nvers >= 10100:
            self.vphot = f.get_f8n()
        if self.nvers >= 10102:
            self.xlphot = f.get_f8n()
        self.mdata = f.get_i4()
        if self.mdata > 0:
            x = f.get_f8n((3, self.mdata))
            self.rn  = x[0,:].copy()
            self.un  = x[1,:].copy()
            self.xln = x[2,:].copy()
        f.assert_eor()
