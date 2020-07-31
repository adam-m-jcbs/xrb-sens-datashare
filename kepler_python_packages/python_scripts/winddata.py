import datetime
import sys
import socket
import time
import os.path

import numpy as np

from numpy import linalg as LA
from numpy.lib import recfunctions as rfn

from fortranfile import FortranReader
import physconst
from human import byte2human
from human import time2human
from human import version2human
from logged import Logged
import isotope
from kepion import KepIon

from utils import CachedAttribute, cachedmethod
from loader import loader, _loader
from keputils import MissingModels

import gzip


class WindData(Logged):
    """
    Interface to read KEPLER wnd (wind) files.
    """
    _extension = 'wnd'
    def __init__(self,
                 filename,
                 silent = False,
                 zerotime = 0.,
                 lastmodel = sys.maxsize,
                 rec = True,
                 **kwargs):
        """
        Constructor; wants file name.
        """
        self.setup_logger(silent)
        filename = os.path.expanduser(filename)
        self.filename = os.path.expandvars(filename)
        self.file = FortranReader(self.filename)
        self.zerotime = zerotime
        self._load(rec, lastmodel)
        self.file.close()
        self._clean(**kwargs)
        self._remove_zerotime()
        self.close_logger()
        self.file = None


    def _load(self, rec = True, lastmodel = sys.maxsize):
        """Open file, call load data, time the load, print out
        diagnostics.

        If *rec* is True (default) we use a record array,
        if *rec* is False, load data as an object for each cycle.
        The latter method is about 200x slower for a full run."""

        self.logger_file_info(self.file)
        self.rec = rec
        if rec:
            self.load_recarray(lastmodel)
        else:
            self.load_record_objects(lastmodel)
        self.logger_load_info(self.data[ 0].nvers,
                              self.data[ 0].ncyc,
                              self.data[-1].ncyc,
                              self.nmodels)

    def load_record_objects(self, lastmodel = sys.maxsize):
        """Load data as object records."""
        self.data=[]
        while not self.file.eof():
            record=WindRecord(self.file)
            self.data.append(record)
            if record.ncyc >= lastmodel:
                break
        self.nmodels = len(self.data)


    def load_recarray(self, lastmodel = sys.maxsize):
        """Load the entire wind file as a record aray."""

        # start by reading version information
        f = self.file
        f.load()
        nvers = f.get_i4()
        ncyc = f.get_i4()
        f.rewind()
        if nvers == 20200:
            nsp = 19
            rectype = np.dtype([
                ("reclen0", ">i4"),
                ("nvers", ">i4"),
                ("ncyc", ">i4"),
                ("time", ">f8"),
                ("dt", ">f8"),
                ("xm", ">f8"),
                ("dms", ">f8"),
                ("r", ">f8"),
                ("u", ">f8"),
                ("aw", ">3f8"),
                ("t", ">f8"),
                ("sl", ">f8"),
                ("cap", ">f8"),
                ("teff", ">f8"),
                ("reff", ">f8"),
                ("yps", ">f8", (nsp)),
                ("reclen1", ">i4")])
        elif nvers == 20100:
            nsp = 19
            rectype = np.dtype([
                ("reclen0", ">i4"),
                ("nvers", ">i4"),
                ("ncyc", ">i4"),
                ("time", ">f8"),
                ("dt", ">f8"),
                ("xm", ">f8"),
                ("dms", ">f8"),
                ("r", ">f8"),
                ("u", ">f8"),
                ("awn", ">f8"),
                ("t", ">f8"),
                ("sl", ">f8"),
                ("cap", ">f8"),
                ("teff", ">f8"),
                ("reff", ">f8"),
                ("yps", ">f8", (nsp)),
                ("reclen1", ">i4")])
        elif nvers == 20000:
            nsp = 19
            rectype = np.dtype([
                ("reclen0", ">i4"),
                ("nvers", ">i4"),
                ("ncyc", ">i4"),
                ("time", ">f8"),
                ("xm", ">f8"),
                ("dms", ">f8"),
                ("r", ">f8"),
                ("u", ">f8"),
                ("awn", ">f8"),
                ("t", ">f8"),
                ("sl", ">f8"),
                ("cap", ">f8"),
                ("teff", ">f8"),
                ("reff", ">f8"),
                ("yps", ">f8", (nsp)),
                ("reclen1", ">i4")])
        elif nvers == 10100:
            nsp = 36
            rectype = np.dtype([
                ("reclen0", ">i4"),
                ("nvers", ">i4"),
                ("ncyc", ">i4"),
                ("time", ">f8"),
                ("xm", ">f8"),
                ("dms", ">f8"),
                ("uterm", ">f8"),
                ("r", ">f8"),
                ("u", ">f8"),
                ("awn", ">f8"),
                ("t", ">f8"),
                ("sl", ">f8"),
                ("gammax", ">f8"),
                ("cap", ">f8"),
                ("teff", ">f8"),
                ("reff", ">f8"),
                ("yps", ">f8", (nsp)),
                ("reclen1", ">i4")])
        elif nvers == 10000:
            nsp = 36
            rectype = np.dtype([
                ("reclen0", ">i4"),
                ("nvers", ">i4"),
                ("ncyc", ">i4"),
                ("time", ">f8"),
                ("xm", ">f8"),
                ("dms", ">f8"),
                ("uterm", ">f8"),
                ("r", ">f8"),
                ("u", ">f8"),
                ("awn", ">f8"),
                ("teff", ">f8"),
                ("sl", ">f8"),
                ("gammax", ">f8"),
                ("cap", ">f8"),
                ("yps", ">f8", (nsp)),
                ("reclen1", ">i4")])
        else:
            raise Error("Version not supported.")

        recsize = rectype.itemsize
        filesize = f.filesize
        filesize = min(filesize, recsize * lastmodel)
        assert np.mod(filesize, recsize) == 0,\
               "Inconsistent record length / file size."
        self.nmodels = filesize // recsize
        self.data = np.recarray(
            self.nmodels,
            dtype = rectype,
            buf = f.file.read()).copy()
        self.nvers = nvers

        if nvers >= 20200:
            awn = LA.norm(self.data['aw'], axis=1)
            self.data = rfn.rec_append_fields(self.data, 'awx', awn,
                                              dtypes='>f8')
        else:
            self.data = rfn.rec_append_fields(self.data, 'aw', None,
                                              dtypes='>3f8')
            self.data['aw'][:,0] = self.data['awn']
            self.data['aw'][:,1:] = 0.

        # recarrays suck


    def write(self,
              filename=None,
              composition=True):
        if filename is None:
            f = sys.stdout
        else:
            f = open(os.path.expandvars(os.path.expanduser(filename)),'w')
        version = 10000
        user = os.getlogin()
        host = socket.gethostname()
        f.write("# Version {version:s} created from file {file:s} by {user:s} on {host:s} at {time:s}\n".format(
                version=version2human(version),
                file=self.filename,
                user=user,
                host=host,
                time=time.asctime(time.gmtime())+' UTC'))
        layout = "{:>8s} {:>23s} {:>23s} {:>23s} {:>23s} {:>23s} {:>23s} {:>23s} {:>23s} {:>23s} {:>23s} {:>23s} {:>23s}"
        format = "{:8d} {:23.16e} {:23.16e} {:23.16e} {:23.16e} {:23.16e} {:23.16e} {:23.16e} {:23.16e} {:23.16e} {:23.16e} {:23.16e} {:23.16e}"
        header = layout.format(
            'cycle',
            'time',
            'dt',
            'stellar mass',
            'mass loss rate',
            'outer radius',
            'surface velocity',
            'angular velocity',
            'surface temperature',
            'luminosity',
            'surface opacity',
            'effective temperature',
            'effective radius')
        units = layout.format(
            'number',
            's',
            's',
            'g',
            'cm',
            'g/s',
            'cm/s',
            'rad/s',
            'K',
            'erg/s',
            'cm2/g',
            'K',
            'cm')
        ions = KepIon.approx_ion_names
        if composition:
            for ion in ions:
                header += ' {:>23s}'.format(ion)
                units += ' {:>23s}'.format('mass fraction')
        f.write(header+'\n')
        f.write(units +'\n')
        for record in self.data:
            # rec=dict()
            # for i in record.dtype.names:
            #     rec[i] = record[i]
            data = format.format(
                    int(record.ncyc),
                    float(record.time),
                    float(record.dt),
                    float(record.xm),
                    float(record.dms),
                    float(record.r),
                    float(record.u),
                    float(record.awn),
                    float(record.t),
                    float(record.sl),
                    float(record.cap),
                    float(record.teff),
                    float(record.reff))

            if composition:
                for i in range(len(ions)):
                    data += ' {:>23.16e}'.format(record.yps[i])
                f.write(data  +'\n')
        if filename is not None:
            f.close()

    def _clean(self,
               raise_exceptions = True,
               **kwargs):
        """Remove duplicate records."""
        if self.rec:
            ncyc = self.data.ncyc
        else:
            ncyc = np.array([d.ncyc for d in self.data], dtype=np.int64)
        if np.all(np.equal(ncyc[1:], ncyc[:-1]+1)):
            return
        u,ii = np.unique(ncyc[::-1],
                       return_index = True)
        ncyc_min = u[0]
        ncyc_max = u[-1]
        nncyc = len(u)
        if len(u) != ncyc_max - ncyc_min + 1:
            jj, = np.where(np.not_equal(u[1:], u[:-1]+1))
            missing = []
            for j in jj:
                missing += [x for x in range(u[j]+1, u[j+1])]
            self.logger.error('ERROR: Missing models: ' +
                              ', '.join([str(x) for x in missing]))
            if raise_exceptions:
                raise MissingModels(models = missing, filename = self.filename)
        self.logger.warning('WARNING: Removing {:d} duplicate models.'.format(
            self.nmodels - nncyc))

        # the following 2-step process is done to deal with record array
        self.data[0:nncyc] = self.data[(self.nmodels - 1) - ii]
        self.data = self.data[0:nncyc]
        self.nmodels = nncyc

    def _remove_zerotime(self, verbose = True):
        """
        Detect and remove resets of time.
        Reconstruct 'dt' for nvers < 20100.
        """
        if self.nvers < 20100:
            # this is very slow
            self.zerotime
            time0 = self.data[0].time
            self.dt = np.ndarray(self.nmodels, dtype = np.float64)
            self.time = np.ndarray(self.nmodels, dtype = np.float64)
            self.time[0] = self.data[0].time + self.zerotime
            self.dt[0] = time0
            for i in range(1, self.nmodels):
                if self.data[i].time < time0:
                    self.zerotime = self.data[i-1].time
                    if verbose:
                        self.logger.info('@ model = {:8d} zerotime was set to {:12.5g}.'.format(
                            int(self.data[i].ncyc),
                            float(self.zerotime)))
                    self.dt[i] = self.data[i].time
                else:
                    self.dt[i] = self.data[i].time - self.data[i-1].time
                time0 = self.data[i].time
                self.time[i] = time0 + self.zerotime
            if self.dt[0] > 10.*self.dt[1]:
                self.logger.info('Cannot determine first time step.')
                self.dt[0] = self.dt[1]
        else:
            if self.rec:
                self.dt = self.data.dt
                time = self.data.time
            else:
                self.dt = np.array([d.dt for d in self.data], dtype = np.float64)
                time = np.array([d.time for d in self.data], dtype = np.float64)
            self.time = self.dt.cumsum()
            self.time += self.data[0].time - self.data[0].dt + self.zerotime
            zerotime0 = self.zerotime
            ii, = np.where(time[1:] < time[:-1])
            self.zerotime += np.sum(time[ii])
            if verbose and self.zerotime != zerotime0:
                self.logger.info('zerotime was set to {:12.5g}.'.format(
                    float(self.zerotime)))

    @cachedmethod
    def tcc(self, tend = 0.25):
        """
        return time till core collapse
        """
        tcc = np.zeros_like(self.dt)
        tcc[:-1] = np.cumsum(self.dt[:0:-1])[::-1]
        return tcc + tend

    @CachedAttribute
    def ncyc(self):
        """
        return model numbers
        """
        return np.array([d.ncyc for d in self.data])


    @CachedAttribute
    def radius(self):
        """
        photospheric radius (cm)
        """
        return np.array([d.reff for d in self.data])

    @CachedAttribute
    def teff(self):
        """
        effective temperature (K)
        """
        return np.array([d.teff for d in self.data])


    @CachedAttribute
    def sl(self):
        """
        stellar photon luminosity (erg/s)
        """
        return np.array([d.sl for d in self.data])

    def __getattr__(self, attr):
        if 'data' in self.__dict__:
            if attr in self.data[0].dtype.fields:
                return np.array([d[attr] for d in self.data])
            try:
                index = KepIon.approx_ion_names.index(attr)
                return np.array([d.yps[index] for d in self.data])
            except ValueError:
                pass
        raise AttributeError(attr)

    def dmsion(self, ion):
        ionabu = self.__getattr__(ion)
        return KepIon(ion).A * ionabu * self.dms

    @CachedAttribute
    def xmlost(self):
        """
        Cummulative mass loss (g)
        """
        return np.cumsum([d.dt * d.dms for d in self.data])

    def write_lc_txt(self,
                     filename,
                     silent = True,
                     append = False,
                     compress = None):
        "Write lc data to lc.txt file."
        self.setup_logger(silent)
        version = 10100
        if (compress is None) and (filename.endswith('.gz')):
            compress = 'gz'
        if (compress is None) or (compress == False):
            opener = open
        elif (compress == 'gz') or (compress == True):
            # could use partial to set compression defaults
            opener = gzip.open
            if not filename.endswith('.gz'):
                filename = filename + '.gz'
        else:
            self.logger.error('Compression type not supported.')
            self.close_logger()
            return
        if append:
            mode = 'at'
        else:
            mode = 'wt'
        with opener(filename,mode, encoding='ASCII') as f:
            if not append:
                f.write('VERSION {:6d}\n'.format(version))
                f.write('{:>25s}{:>25s}{:>25s}\n'.format(
                    'time (s)',
                    'R_eff (cm)',
                    'L (erg/s)'))
            for i,d in enumerate(self.data):
                f.write('{:25.17e}{:25.17e}{:25.17e}\n'.format(
                float(self.time[i]),
                float(d.reff),
                float(d.sl)))
        self.close_logger()



load = loader(WindData, 'winddata')
_load = _loader(WindData, 'winddata')
loadwind = load

class WindRecord(object):
    def __init__(self,
                 loadfile,
                 data = True):
        self.load(loadfile)

    def load(self, f):
        f.load()
        self.nvers = f.get_i4()
        self.ncyc = f.get_i4()
        if self.nvers == 10000:
            nsp = 36
            (self.time,
             self.xm,
             self.dms,
             self.uterm,
             self.r,
             self.u,
             self.aw,
             self.teff,
             self.sl,
             self.gammax,
             self.cap)= f.get_f8n(11)
            self.yps = f.get_f8an([nsp-1])
        elif self.nvers == 10100:
            nsp = 36
            (self.time,
             self.xm,
             self.dms,
             self.uterm,
             self.r,
             self.u,
             self.aw,
             self.t,
             self.sl,
             self.gammax,
             self.cap,
             self.teff,
             self.reff) = f.get_f8n(13)
            self.yps = f.get_f8n(nsp-1)
        elif self.nvers == 20000:
            nsp = 19
            (self.time,
             self.xm,
             self.dms,
             self.r,
             self.u,
             self.aw,
             self.t,
             self.sl,
             self.cap,
             self.teff,
             self.reff) = f.get_f8n(11)
            self.yps = f.get_f8an([nsp])
        elif self.nvers == 20100:
            nsp = 19
            (self.time,
             self.dt,
             self.xm,
             self.dms,
             self.r,
             self.u,
             self.aw,
             self.t,
             self.sl,
             self.cap,
             self.teff,
             self.reff) = f.get_f8n(12)
        elif self.nvers == 20200:
            nsp = 19
            (self.time,
             self.dt,
             self.xm,
             self.dms,
             self.r,
             self.u,
             self.awx,
             self.awy,
             self.awz,
             self.t,
             self.sl,
             self.cap,
             self.teff,
            self.reff) = f.get_f8n(14)
            self.yps = f.get_f8n(nsp)
        if self.nvers < 20200:
            self.awx = self.aw
            self.awy, self.awz = 0., 0.
            self.aw = np.array([self.awx, self.awy, self.awz])
            self.awn = LA.norm(self.aw)
        f.assert_eor()
