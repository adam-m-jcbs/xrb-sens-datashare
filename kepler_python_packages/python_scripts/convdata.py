"""
Python module to load KEPLER cnv data.

Currently only versions 10400+ are supported.
Maybe some others.
"""

import os.path
import sys

from multiprocessing import Process, Queue, JoinableQueue, cpu_count, active_children

import numpy as np

from numpy import linalg as LA

from human import time2human
from utils import cachedmethod, CachedAttribute, bit_count, queue_processor
from loader import loader, _loader
from logged import Logged
from kepion import KepAbuData
from fortranfile import FortranReader
from fortranfile.errors import RecLenError
from keputils import RecordVersionMismatch, UnkownVersion, MissingModels

# import physconst
# datlabel= '0=rad 1=neut 2=osht 3=semi 4=conv 5=thal'

conv_types = {
    ' ' : 0,
    'N' : 1,
    'O' : 2,
    'S' : 3,
    'C' : 4,
    'T' : 5,
    }

conv_names = {
    '    ' : 0,
    'neut' : 1,
    'osht' : 2,
    'semi' : 3,
    'conv' : 4,
    'thal' : 5,
    }

full_conv_names = {
    ' ' : 'radiative',
    'C' : 'convection',
    'N' : 'convectively neutral',
    'S' : 'semiconvection',
    'O' : 'overshooting',
    'T' : 'thermohaline convection',
    }

logmin = 1.e-99

def extract_layer(
    ydata = None,
    ilayer = None,
    level = None,
    radius = False,
    surface = False,
    offset = None,
    logarithmic = False,
    column = False,
    scale = None,
    **kwargs):
    """
    extract outside object hirachy to allow multi-processing
    """
    nnc = 0
    for i,data in enumerate(ydata):
        if data.nx[ilayer] > 0:
            nnc = max(nnc,data.nx[ilayer])
    nc = np.ndarray(len(ydata), dtype = np.uint8)
    c = np.ndarray((len(ydata), nnc+1, 2),
                   dtype = np.float64)
    for i, data in enumerate(ydata):
        if data.nx[ilayer] > 0:
            # actual extraction
            x = data.nnx[ilayer,0:data.nx[ilayer]+1]
            if level > 0:
                i0 = np.where(np.logical_and(x[0:-1] <  level, x[1:] >= level))
                i1 = np.where(np.logical_and(x[0:-1] >= level, x[1:] <  level))
            else:
                i0 = np.where(np.logical_and(x[0:-1] >  level, x[1:] <= level))
                i1 = np.where(np.logical_and(x[0:-1] <= level, x[1:] >  level))
            ii = data.inx[ilayer, np.transpose(np.array([i0, i1])).reshape(-1)] - 1
            nc[i] = len(i1[0])

            # transformations
            if radius:
                y = data.rncoord[ii]
            else:
                y = data.xmcoord[ii]
            if surface:
                if radius:
                    y = data.rncoord[-1] - y[::-1]
                else:
                    y = data.xmcoord[-1] - y[::-1]
                    if column:
                        y /= 4. * np.pi * (data.rncoord[ii]**2)[::-1]
            elif column:
                y /= 4. * np.pi * data.rncoord[0]**2
            if offset is not None:
                y += offset[i]
            if scale is not None:
                y *= scale
            if logarithmic:
                y = np.log10(np.maximum(logmin, y))

            # final asignment
            c[i,0:nc[i],:] = y.reshape((-1, 2))
        else:
            nc[i] = 0
    c = c[:,0:np.max(nc),:]
    return nc, c

def extract_conv(
    ydata = None,
    ilayer = None,
    radius = False,
    surface = False,
    offset = None,
    logarithmic = False,
    column = False,
    scale = None,
    **kwargs):
    """
    extract outside object hirachy to allow multi-processing
    """
    nc = np.array([len((np.where(data.convtype == ilayer))[0])
                   for data in ydata],
                  dtype = np.uint16)

    c = np.ndarray((len(ydata), np.max(nc), 2),
                   dtype = np.float64)
    # hmm, it would seem we can't have radius *and* colum
    # there can be three modes: mass, radius, column
    # column is fixed facor it could be in scale and does not need to be
    # computed here, e.g. for NS models, 4 pi (r=1e6)**2

    # below this is split up for performance
    # it could be done by
    # 1) computing mass or radius
    # 2) if surface subtract [-1]
    # 3) if column divide by 4 pi r**2

    for i, data in enumerate(ydata):
        if nc[i] > 0:
            # actual extraction
            x = (np.where(data.convtype == ilayer))[0]
            ii = np.transpose(np.array([x, x + 1])).reshape(-1)

            # transformations
            if radius:
                y = data.rc[ii]
            else:
                y = data.mc[ii]
            if surface:
                if radius:
                    y = data.rc[-1] - y[::-1]
                else:
                    y = data.mc[-1] - y[::-1]
                    if column:
                        y /= 4 * np.pi * (data.rc[ii]**2)[::-1]
            elif column:
                y /= 4 * np.pi * data.rc[0]**2
            if offset is not None:
                y += offset[i]
            if scale is not None:
                y *= scale
            if logarithmic:
                y = np.log10(np.maximum(logmin, y))

            # final assignment
            c[i,0:nc[i],:] = y.reshape((-1, 2))
    return nc, c


class ConvExtractor(Logged):
    """
    extract outside object hirachy to allow multi-processing
    """
    def __init__(self,
                 silent = False,
                 **kwargs):
        self.setup_logger(silent)
        self.layer_data = extract_conv(**kwargs)
        self.close_logger(timing = ' Conv  {:>3d} extracted in'.format(kwargs['ilayer']))

class LayerExtractor(Logged):
    """
    extract outside object hirachy to allow multi-processing
    """
    def __init__(self,
                 silent = False,
                 **kwargs):
        self.setup_logger(silent)
        self.layer_data = extract_layer(**kwargs)
        self.close_logger(timing = 'Level {:>3d} extracted in'.format(kwargs['level']))

class DataProcessr(Logged):
    def __init__(self,
                 data):
        self.data = data
    def __call__(self,
                 task,
                 silent = True):
        # maybe 'task' should be dict and contain 'silent'?
        # (it should)
        self.setup_logger(silent)
        self.record = ConvRecord(task[0], **self.data)
        self.id = task[1]
        self.close_logger(timing = 'Record {:d} loaded in'.format(
            self.record.ncyc))
        return (self.record, self.id)

class ConvData(Logged):
    """
    Interface to load KEPLER cnv data files.
    """
    _extension = 'cnv'
    def __init__(self,
                 filename,
                 silent = False,
                 **kwargs):
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
        self.file = FortranReader(self.filename, **kwargs)
        self._load(**kwargs)
        self.file.close()
        self.close_logger()
        self.file = None

    def _load(self,
              zerotime = True,
              timecc = 0.25,
              message_time = 5,
              threads = 1,
              endtime = sys.maxsize,
              **kwargs):
        """
        Open file, call load data, time the load, print out diagnostics.
        """
        self.logger_file_info(self.file)
        starttime = self.get_timer()
        self.logger.info('time to open file:    {:s}'.format(time2human(starttime)))

        self.stepmodel = kwargs.setdefault('stepmodel', 1)

        lasttime = starttime
        if threads is None:
            threads = 1
        if threads < 0:
            threads = cpu_count()
        if threads > 100000 : # too slow
            self.add_timer('thread')
            self.logger.info('Setting up worker processes.')
            task_queue = JoinableQueue()
            done_queue = Queue()
            args = dict(kwargs)
            params = dict(processor = DataProcessr,
                          data = args)
            nproc = threads
            for i in range(nproc):
                Process(target = queue_processor,
                        args   = (task_queue,
                                  done_queue,
                                  params)
                        ).start()
            self.logger.info('worker processes set up in {:s}'.format(time2human(self.finish_timer('thread'))))
            self.add_timer('setup')
            self.logger.info('Setting up tasks.')
            nin = 0
            for record in self.file.iterrecords():
                task = (record, nin)
                task_queue.put(task)
                nin += 1

            self.logger.info('tasks set up in {:s}'.format(time2human(self.finish_timer('setup'))))
            self.logger.info('Running tasks.')
            self.add_timer('task')

            task_queue.join()

            self.logger.info('Tasks finished in {:s}'.format(time2human(self.finish_timer('task'))))
            self.add_timer('collect')
            self.logger.info('Collecting data.')

            self.data = np.ndarray((nin), dtype = np.object)
            for i in range(nin):
                result = done_queue.get()
                self.data[result[1]] = result[0]

            self.logger.info('Closing threads.')

            for i in range(nproc):
                task_queue.put('STOP')

            task_queue.join()
            task_queue.close()
            done_queue.close()
            self.logger.info('Collecting data finished in {:s}'.format(time2human(self.finish_timer('collect'))))
        else:
            self.data=[]
            # try:
            #     for record in self.file.iterrecords():
            #         record = ConvRecord(record, **kwargs)
            #         # rewrite using 'timer' model
            #         curtime = self.get_timer()
            #         dlasttime = curtime - lasttime
            #         if dlasttime.total_seconds() > message_time:
            #             eta = ((self.file.filesize / self.file.fpos - 1) *
            #                    (curtime - starttime).total_seconds())
            #             self.logger.info('Time: {:>10s}, ETA: {:>10s}'.format(
            #                 self.get_timer_human(),
            #                 time2human(eta)))
            #             lasttime = curtime
            #         if record.time is None:
            #             continue
            #         self.data.append(record)
            #         if endtime <= record.time:
            #             break
            # except Exception as e:
            #     print(e)
            #     aboard_pos = self.file.fpos - self.file.reclen + self.file.pos
            #     self.logger.critical('ERROR in file {} in record {:d} (byte {:d} / {:d}, {:5.2f}%).  ABOARDING.'.format(
            #         self.filename,
            #         self.file.rpos,
            #         aboard_pos,
            #         self.file.filesize,
            #         aboard_pos / self.file.filesize * 100))
            #     if kwargs.get('raise_exceptions', True):
            #         raise Exception('File {} is corrupt.'.format(self.filename))

            while not self.file.eof():
                try:
                    record = ConvRecord(self.file, **kwargs)
                except Exception as e:
                    if isinstance(e, RecLenError):
                        aboard_pos = self.file.fpos
                    else:
                        aboard_pos = (
                            self.file.fpos
                            - self.file.reclen
                            - self.file.fortran_reclen
                            + self.file.pos)
                    self.logger.critical('ERROR in file {} in record {:d} (byte {:d} / {:d}, {:5.2f}%).  ABOARDING.'.format(
                        self.filename,
                        self.file.rpos,
                        aboard_pos,
                        self.file.filesize,
                        aboard_pos / self.file.filesize * 100))
                    if kwargs.get('skip_broken', False):
                        self.file.seek_noncorrupt()
                        continue
                    if kwargs.get('raise_exceptions', True):
                        raise Exception('File {} is corrupt.'.format(self.filename)) from e
                    break
                if record is None:
                    break
                # rewrite using 'timer' model
                curtime = self.get_timer()
                dlasttime = curtime - lasttime
                if dlasttime.total_seconds() > message_time:
                    eta = ((self.file.filesize / self.file.fpos - 1) *
                           (curtime - starttime).total_seconds())
                    self.logger.info('Time: {:>10s}, ETA: {:>10s}'.format(
                        self.get_timer_human(),
                        time2human(eta)))
                    lasttime = curtime
                if record.time is None:
                    continue
                self.data.append(record)
                if endtime <= record.time:
                    break
            curtime = self.get_timer()
            self.logger.info('time to load records: {:s}'.format(time2human(curtime - starttime)))
        self.add_timer('compile')
        self.logger.info('Compiling data.')

        self._sort_models(**kwargs)
        self.nvers = self.data[0].nvers
        self.nmodels = len(self.data)
        if self.nvers >= 10401:
            self.layer_types = ConvRecord.layer_types_10401
        elif self.nvers >= 10400:
            self.layer_types = ConvRecord.layer_types_10400
        elif self.nvers >= 10100:
            self.layer_types = ConvRecord.layer_types_10100
        else:
            raise UnkownVersion(self.nvers)

        self._set_timecc()
        if zerotime:
            self._remove_zerotime(**kwargs)
        self.close_timer('compile', timing='Data comliled in')

        # we may need our own...
        self.logger_load_info(self.data[0].nvers,
                              self.data[ 0].ncyc,
                              self.data[-1].ncyc,
                              self.nmodels)

        # kill worker processes that _still_ run (should be none)
        for p in active_children():
            self.logger.info('terminating process {:d}'.format(p.pid))
            p.terminate()

    def _sort_models(self,
                     fast = False,
                     raise_exceptions = True,
                     **kwargs):
        """
        Remove models that are superseeded by later model.

        This is something that would not be necessary in the current KEPLER output.
        """
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
        valid = np.array([data.time is not None for data in self.data])
        self.data = self.data[valid]
        if (self.data[0].nvers >= 10403) and fast:
            # speed up cases that should not require fixing
            return
        # find unique models, keeping *last* one in file
        models = np.array([-data.ncyc for data in self.data])
        u, ii = np.unique(models,
                          return_index = True)
        # remove duplicate models
        if len(ii) <= len(models):
            ii = ii[::-1]
            idup = np.setdiff1d(np.arange(len(models)), ii)
            for data in self.data[idup]:
                self.logger.info('Removing duplicate model {:9d}'.format(
                    data.ncyc))
            self.data = self.data[ii]
        u = -u[::-1]
        # find missing models
        ncyc_min = u[0]
        ncyc_max = u[-1]
        # try find step
        if len(u) < ncyc_max - ncyc_min + 1:
            nstep, excess = np.divmod(ncyc_max - ncyc_min, len(u) - 1)
            if excess == 0 and np.allclose(u, np.arange(ncyc_min, ncyc_max+1, nstep)):
                    self.stepmodel = nstep
                    self.logger.info('Setting model step to {:d}'.format(
                        self.stepmodel))
        if len(u) != (ncyc_max - ncyc_min + self.stepmodel) // self.stepmodel:
            jj, = np.where(np.not_equal(u[1:], u[:-1] + self.stepmodel))
            missing = []
            for j in jj:
                missing += [x for x in range(u[j] + self.stepmodel, u[j+1], self.stepmodel)]
            self.logger.error('ERROR: Missing models: ' +
                              ', '.join([str(x) for x in missing]))
            if raise_exceptions:
                raise MissingModels(models = missing, filename = self.filename)
            # else:
            #     TODO - add fix for dt etc.

        versions = np.unique([data.nvers for data in self.data])
        if len(versions) > 1:
            self.logger.warning('--------------------------------------------------')
            self.logger.warning('WARNING: record versions differ - errors may occur')
            self.logger.warning(', '.join(['{:d}'.format(v) for v in versions]))
            self.logger.warning('--------------------------------------------------')
            if raise_exceptions:
                raise RecordVersionMismatch(filename = self.filename, versions = versions)

    def _set_timecc(self, **kwargs):
        """
        Set timecc in case dt is not defined.
        """
        if self.nvers <= 10402 or self.stepmodel != 1:
            self.data[-1].timecc = 0.
            time = self.data[-1].time
            timecc = time
            for data in self.data[-2::-1]:
                if time < data.time:
                    timecc += data.time
                data.timecc = timecc - data.time
                time = data.time
        else:
            dt = (self.dt).copy()
            timecc = ((dt[::-1]).cumsum())[::-1]
            for tcc,data in zip(timecc, self.data):
                data.timecc = tcc

    def _remove_zerotime(self, verbose = True, **kwargs):
        """
        Detect and remove resets of time.

        Reconstruct "dt" for nvers < 10403 or discontineous model list, e.g.,
        when using 'stepmodel' on loading.
        """
        if self.nvers <= 10402 or self.stepmodel != 1:
            zerotime = np.float64(0)
            time0 = self.data[0].time
            self.data[0].dt = time0
            for i in range(1,self.nmodels):
                if self.data[i].time < time0:
                    zerotime = self.data[i-1].time
                    if verbose:
                        self.logger.info('@ model = {:8d} zerotime was set to {:12.5g}.'.format(
                            int(self.data[i].ncyc),
                            float(zerotime)))
                    self.data[i].dt = self.data[i].time
                else:
                    self.data[i].dt = self.data[i].time - self.data[i-1].time
                time0 = self.data[i].time
                self.data[i].time = time0 + zerotime
        else:
            time = self.dt.cumsum()
            time += self.data[0].time - self.data[0].dt
            for t,data in zip(time,self.data):
                data.time = t

    @CachedAttribute
    def xtime(self):
        """
        Return plot time array.
        """
        xtime = np.ndarray(self.nmodels+1, dtype = np.float64)
        xtime[1:] = self.time
        if self.nvers >= 10403 and self.stepmodel == 1:
            xtime[0] = xtime[1] - self.dt[0]
        else:
            if self.data[0].ncyc == 1:
                xtime[0] = 0.
            else:
                xtime[0] = max(0., 2*xtime[1] - xtime[2])
        return xtime

    @CachedAttribute
    def models(self):
        """
        Return Model numbers
        """
        models = np.array([data.ncyc for data in self.data])
        return models

    ncyc = models

    @CachedAttribute
    def xmodels(self):
        """
        Return Model Numbers Coordinate for plotting
        """
        xmodels = np.insert(self.models, 0, 2 * self.models[0] - self.models[1])
        return xmodels

    @CachedAttribute
    def summ0(self):
        """
        Return summ0.
        """
        return np.array([data.summ0 for data in self.data])

    # some of these could be from a base class for all data ...
    @CachedAttribute
    def time(self):
        """
        Return time array.
        """
        return np.array([data.time for data in self.data])

    # use this as a template for any other variable
    @CachedAttribute
    def dt(self):
        """
        Return dt array.
        """
        return np.array([data.dt for data in self.data])

    @cachedmethod
    def timecc(self, offset = 0.):
        """
        Return time till core collapse.

        Set offset to 'None' to use last dt of last time step taken as
        offset.
        """
        if self.nvers >= 10403:
            dt = np.empty_like(self.dt)
            dt[:-1] = self.dt[1:]
            if offset is not None:
                dt[-1] = offset
            else:
                dt[-1] = dt[-2]
            return ((dt[::-1]).cumsum())[::-1]
        else:
            timecc = np.array([data.timecc for data in self.data])
            if offset is not None:
                timecc += offset
            return timecc

    @cachedmethod
    def xtimecc(self, offset = 0.):
        """
        Return time till core collapse including Cycle 0.

        Set offset to 'None' to keep last dt as offset.
        """
        if self.nvers >= 10403:
            dt = np.ndarray(self.nmodels + 1, dtype = np.float64)
            dt[:-1] = self.dt
            if offset is None:
                dt[-1] = dt[-2]
            else:
                dt[-1] = offset
            return ((dt[::-1]).cumsum())[::-1]
        else:
            timecc = np.ndarray(self.nmodels+1, dtype = np.float64)
            timecc[1:] = np.array([data.timecc for data in self.data])
            if offset is not None:
                timecc += offset
            timecc[0] = timecc[1]+self.time[0]
            return timecc

    @CachedAttribute
    def rstar(self):
        """
        outer radius of outer grid point (cm)
        """
        if self.nvers < 10500:
            return np.array([data.rc[-1] for data in self.data])
        else:
            return np.array([data.rncoord[-1] for data in self.data])

    @CachedAttribute
    def rstar0(self):
        """
        inner radius if inner grid point (cm)
        """
        if self.nvers < 10500:
            return np.array([data.rc[0] for data in self.data])
        else:
            return np.array([data.rncoord[0] for data in self.data])

    @CachedAttribute
    def xmstar(self):
        """
        mass of star (g)
        Q - does this include summ0?
        """
        if self.nvers < 10500:
            return np.array([data.mc[-1] for data in self.data])
        else:
            return np.array([data.xmcoord[-1] for data in self.data])

    @CachedAttribute
    def cstar(self):
        """
        column depth of star (g/cm**2)
        """
        return self.xmstar / (4 * np.pi * self.rstar0**2)

    @CachedAttribute
    def tc(self):
        """
        central temperature (K)
        """
        return np.array([data.tc for data in self.data])

    @CachedAttribute
    def dc(self):
        """
        central density (g/cm**3)
        """
        return np.array([data.dc for data in self.data])

    @CachedAttribute
    def vc(self):
        """
        central specific volume (cm**3/g)
        """
        return 1/self.dc

    @CachedAttribute
    def pc(self):
        """
        central pressure (dyn/cm**2)
        """
        return np.array([data.pc for data in self.data])

    @CachedAttribute
    def aw(self):
        """
        central angular velocity vector (rad/s)
        """
        return np.array([data.aw for data in self.data])

    @CachedAttribute
    def awn(self):
        """
        central angular velocity magnitude (rad/s)
        """
        return LA.norm(self.aw, axis=1)

    @CachedAttribute
    def awcst(self):
        """
        central rotation parameter vector \tilde(\omega) (rad s**(-1) g**(2/3) cm**(-1))

        omega_c * rho_c**(-2/3)
        """
        return np.array([data.aw * data.dc**(-2./3.) for data in self.data])

    @CachedAttribute
    def awcstn(self):
        """
        central rotation parameter magnitude \tilde(\omega) (rad s**(-1) g**(2/3) cm**(-1))

        omega_c * rho_c**(-2/3)
        """
        return LA.norm(self.awcst, axis=1)

    @CachedAttribute
    def sc(self):
        """
        central entropy (k_B/baryon)
        """
        return np.array([data.sc for data in self.data])

    stot = sc

    @CachedAttribute
    def ye(self):
        """
        central Y_e (electrons/baryon)
        """
        return np.array([data.ye for data in self.data])

    @CachedAttribute
    def ab(self):
        """
        central \bar(A)
        """
        return np.array([data.ab for data in self.data])

    @CachedAttribute
    def et(self):
        """
        central degeneracy parameter eta
        """
        return np.array([data.et for data in self.data])

    @CachedAttribute
    def sn(self):
        """
        central energy generation rate (erg/g/s)
        """
        return np.array([data.sn for data in self.data])

    @CachedAttribute
    def su(self):
        """
        central nuclear energy generation rate (erg/g/s)
        """
        return np.array([data.su for data in self.data])

    @CachedAttribute
    def g1(self):
        """
        central \Gamma_1
        """
        return np.array([data.g1 for data in self.data])


    @CachedAttribute
    def g2(self):
        """
        central \Gamma_2
        """
        return np.array([data.g2 for data in self.data])

    @CachedAttribute
    def s1(self):
        """
        central Ledoux criterion
        """
        return np.array([data.s1 for data in self.data])

    @CachedAttribute
    def s2(self):
        """
        central Schwarzschild  criterion
        """
        return np.array([data.s2 for data in self.data])

    @CachedAttribute
    def an(self):
        """
        central network type
        This will be needed to constuct network/abu data
        """
        return np.array([data.an for data in self.data], dtype = np.int)

    @CachedAttribute
    def abun(self):
        """
        central composition (mass fraction)
        """
        return np.array([data.abun for data in self.data])

    @CachedAttribute
    def net(self):
        """
        central composition KepAbuData object
        """
        return KepAbuData(self.abun,
                         netnum = self.an,
                         molfrac = False)

    @CachedAttribute
    def xlumn(self):
        """
        neutrino luminosity (erg/sec)
        """
        return np.array([data.xlumn for data in self.data])

    @CachedAttribute
    def eni(self):
        """
        Total internal energy.
        """
        return np.array([data.eni for data in self.data])

    @CachedAttribute
    def enk(self):
        """
        Total kinetic energy.
        """
        return np.array([data.enk for data in self.data])

    @CachedAttribute
    def ent(self):
        """
        Total current energy.
        """
        return np.array([data.ent for data in self.data])

    @CachedAttribute
    def enp(self):
        """
        Total current potential energy.
        """
        return np.array([data.enp for data in self.data])

    @CachedAttribute
    def epro(self):
        """
        Total net energy produced so far by nuclear reactions less neutrino losses.
        """
        return np.array([data.epro for data in self.data])

    @CachedAttribute
    def enn(self):
        """
        Total neutrino energy lost from the star.
        """
        return np.array([data.enn for data in self.data])

    @CachedAttribute
    def enr(self):
        """
        Current rotational energy.
        """
        return np.array([data.enr for data in self.data])

    @CachedAttribute
    def ensc(self):
        """
        Total energy deposited so far from input ?source?.
        """
        return np.array([data.ensc for data in self.data])

    @CachedAttribute
    def enes(self):
        """
        Total non-neutrino energy that has so far escaped from the star?s surface.
        """
        return np.array([data.enes for data in self.data])

    @CachedAttribute
    def enc(self):
        """
        Energy check.
        """
        return np.array([data.enc for data in self.data])

    @CachedAttribute
    def enpist(self):
        """
        Total energy input by the piston.
        """
        return np.array([data.enpist for data in self.data])

    @CachedAttribute
    def enid(self):
        """
        Rate of change in the total internal energy.
        """
        return np.array([data.enid for data in self.data])

    @CachedAttribute
    def enkd(self):
        """
        Rate of change in the total kinetic energy.
        """
        return np.array([data.enkd for data in self.data])

    @CachedAttribute
    def enpd(self):
        """
        Rate of change in the total potential energy.
        """
        return np.array([data.enpd for data in self.data])

    @CachedAttribute
    def entd(self):
        """
        Rate of change in the total energy.
        """
        return np.array([data.entd for data in self.data])

    @CachedAttribute
    def eprod(self):
        """
        Total rate of nuclear energy production less neutrino losses.
        """
        return np.array([data.eprod for data in self.data])

    @CachedAttribute
    def xlumn(self):
        """
        Neutrino luminosity.
        """
        return np.array([data.xlumn for data in self.data])

    @CachedAttribute
    def enrd(self):
        """
        Rate of change of rotational energy during last step.
        """
        return np.array([data.enrd for data in self.data])

    @CachedAttribute
    def enscd(self):
        """
        Rate of energy deposition by input source.
        """
        return np.array([data.enscd for data in self.data])

    @CachedAttribute
    def enesd(self):
        """
        Total rate of energy escape from the star (in photons).
        """
        return np.array([data.enesd for data in self.data])

    @CachedAttribute
    def encd(self):
        """
        Total rate of change in the energy check.
        """
        return np.array([data.encd for data in self.data])

    @CachedAttribute
    def enpistd(self):
        """
        Energy input rate by the piston.
        """
        return np.array([data.enpistd for data in self.data])

    @CachedAttribute
    def xlum(self):
        """
        Surface luminosity in electromagnetic radiation
        """
        return np.array([data.xlum for data in self.data])

    @CachedAttribute
    def xlum0(self):
        """
        Energy incout at bottom.
        """
        return np.array([data.xlum0 for data in self.data])

    @CachedAttribute
    def angit(self):
        """
        Total momentum of inertia of the star.
        """
        return np.array([data.angit for data in self.data])

    @CachedAttribute
    def anglt(self):
        """
        Current total angular momentum vector.
        """
        return np.array([data.anglt for data in self.data])

    @CachedAttribute
    def angltn(self):
        """
        Current total angular momentum magnitude.
        """
        return LA.norm(self.anglt, axis=1)


    ### some routines for plots

    @staticmethod
    def conv_index(layer = 'C'):
        """
        Return index, name of layer
        """
        if isinstance(layer, str):
            if len(layer) == 1:
                layer = conv_types.get(layer, 0)
            else:
                layer = conv_names.get(layer, 0)
        assert layer < len(conv_types), 'unkown convection type index'
        layer_sentinel = None
        for n,i in conv_types.items():
            if i == layer:
                layer_sentinel = n
        layer_name = None
        for n,i in conv_names.items():
            if i == layer:
                layer_name = n
        return layer, layer_sentinel, layer_name

    @cachedmethod
    def extract_conv(self,
                     layer = 'C',
                     radius = False):
        """
        Extract convection layer.

        Supports numeric and character parameters as defined in conv_types

        For now this is just for demo ...
        """
        self.setup_logger()
        self.add_timer('conv')
        layer, layer_sentinel, layer_name = self.conv_index(layer)
        if layer == 0:
            # though we could return radiative regions as well
            return None

        nc, c = extract_conv(
            ydata = self.data,
            layer = layer,
            radius = radius,
            )

        self.logger.info('{:>4s} layer    extraced in {:s}'.format(layer_name,time2human(self.finish_timer('conv'))))
        self.close_logger()
        return nc, c

    def layer_index(self,
                    layer = 'nuc'):
        """
        Return index, whether radial, name of layer
        """
        if isinstance(layer, str):
            layer = self.layer_types.get(layer, None)
        layer_name = None
        for n,i in self.layer_types.items():
            if i == layer:
                layer_name = n
        radial = None
        if isinstance(layer_name, str):
            radial = layer_name.endswith('d')
        return layer, radial, layer_name


    # @cachedmethod
    # def extract_layer(self,
    #                   layer = 'nuc',
    #                   level = 1,
    #                   radius = False):
    #     """
    #     Extract convection layer.

    #     Supports numeric and character parameters as defined in ConvRecord.conv_types

    #     For now this is just for demo ...
    #     """
    #     self.setup_logger()
    #     self.add_timer('layer')
    #     layer, radial, layer_name = self.layer_index(layer)

    #     # current algorithm requires the following
    #     assert level != 0

    #     nc = np.ndarray(self.nmodels, dtype = np.uint8)
    #     for i,data in enumerate(self.data):
    #         if data.nx[layer] > 0:
    #             x = data.nnx[layer,1:data.nx[layer]+1]
    #             if level > 0:
    #                 i1 = np.where(np.logical_and(x[0:-1] >= level,x[1:] <  level))
    #             else:
    #                 i1 = np.where(np.logical_and(x[0:-1] <= level,x[1:] >  level))
    #             nc[i] = len(i1[0])
    #         else:
    #             nc[i] = 0
    #     c = np.ndarray((self.nmodels, np.max(nc), 2),
    #                    dtype = np.float64)
    #     for i, data in enumerate(self.data):
    #         if data.nx[layer] > 0:
    #             x = data.nnx[layer,0:data.nx[layer]+1]
    #             if level > 0:
    #                 i0 = np.where(np.logical_and(x[0:-1] <  level,x[1:] >= level))
    #                 i1 = np.where(np.logical_and(x[0:-1] >= level,x[1:] <  level))
    #             else:
    #                 i0 = np.where(np.logical_and(x[0:-1] >  level,x[1:] <= level))
    #                 i1 = np.where(np.logical_and(x[0:-1] <= level,x[1:] >  level))
    #             ii = data.inx[layer, np.transpose(np.array([i0,i1])).reshape(-1)] - 1
    #             if radius:
    #                 c[i,0:nc[i],:] = data.rncoord[ii].reshape((-1,2))
    #             else:
    #                 c[i,0:nc[i],:] = data.xmcoord[ii].reshape((-1,2))

    #     self.logger.info('{:>4s} layer{:>3d} extraced in {:s}'.format(
    #         layer_name,
    #         level,
    #         time2human(self.finish_timer('layer'))))
    #     self.close_logger()
    #     return nc, c

    @cachedmethod
    def extract_layer(self,
                      layer = 'nuc',
                      level = 1,
                      radius = False):
        """
        Extract convection layer.

        Supports numeric and character parameters as defined in conv_types

        For now this is just for demo ...
        """
        self.setup_logger()
        self.add_timer('layer')
        layer, radial, layer_name = self.layer_index(layer)

        # current algorithm requires the following
        assert level != 0

        nc, c = extract_layer(
            ydata = self.data,
            layer = layer,
            level = level,
            radius = radius,
            )

        self.logger.info('{:>4s} layer{:>3d} extraced in {:s}'.format(
            layer_name,
            level,
            time2human(self.finish_timer('layer'))))
        self.close_logger()
        return nc, c

    @cachedmethod
    def level_range(self,
                    layer = 'nuc'):
        """
        Return range of levels for a given layer.
        """
        layer, radial, layer_name = self.layer_index(layer)
        maxlevel = 0
        minlevel = 0
        for data in self.data:
            n = data.nx[layer]
            if n > 0:
                maxlevel = max(maxlevel, np.max(data.nnx[layer,0:n]))
                minlevel = min(minlevel, np.min(data.nnx[layer,0:n]))
        return -minlevel, maxlevel

    @cachedmethod
    def level_min(self,
                  layer = 'nuc'):
        """
        Return minimum values of levels for a given layer.
        """
        layer, radial, layer_name = self.layer_index(layer)
        return self.data[0].minx[layer,:]


    @CachedAttribute
    def tau_MS(self):
        """
        MS lifetime in sec
        """
        xh1 = self.abun[:,1]
        ii = np.where(xh1 < 1.e-4)[0][0]
        return self.time[ii]

    @CachedAttribute
    def dmdec(self):
        """
        Get time-integrated decretion mass

        This routine is to be generalised
        """
        dm = np.ndarray(self.nmodels + 1, dtype = np.float64)
        # todo - add some more checks

        dm[0] = 0
        for i, data in enumerate(self.data):
            idx = bin(data.ladv & (ConvRecord.adv_dec - 1)).count('1')
            assert data.iadv[idx] == 2 # for now only this case
            dm[i+1] = data.dmadv[idx]
        return np.cumsum(dm)

    @CachedAttribute
    def dcdec(self):
        """
        Get time-integrated decretion column depth
        """
        dc = self.dmdec
        dc[1:] /= (4. * np.pi * self.rstar0**2)
        return dc

load = loader(ConvData, 'loadconv')
_load = _loader(ConvData, __name__ + '.load')
loadconv = load

class ConvRecord(object):
    """
    Load individual record from cnv file.
    """
    def __init__(self, file, **kwargs):
        assert file is not None
        self._load(file, **kwargs)

    layer_types_10100 = {'nuc' : 0,
                         'nuk' : 1}

    layer_types_10400 = {'nuc' : 0,
                         'nuk' : 1,
                         'nucd': 2,
                         'nukd': 3}

    layer_types_10401 = {'nuc' : 0,
                         'nuk' : 1,
                         'neu' : 2,
                         'nucd': 3,
                         'nukd': 4,
                         'neud': 5}

    # advection flags
    adv_loss = 1
    adv_acc  = 2
    adv_dec  = 4

    # number of APPROX species
    nhiz = 20

    @staticmethod
    def _int_kind(kind_len):
        if kind_len <= 2:
            return np.dtype(np.int8)
        if kind_len <= 4:
            return np.dtype(np.int16)
        if kind_len <= 9:
            return np.dtype(np.int32)
        if kind_len <= 18:
            return np.dtype(np.int64)
        raise Exception('Unkown kind_len')

    @staticmethod
    def _yzip2type(yzip):
        x = np.zeros_like(yzip, dtype = np.uint8)
        for t,i in conv_types.items():
            x[np.where(yzip == t)] = i
        return x

    def _load(self,
              f,
              firstmodel = -1,
              lastmodel = sys.maxsize,
              stepmodel = 1,
              **kwargs):
        f.load()
        self._byteorder = f.byteorder
        self.nvers = f.get_i4()
        self.ncyc = f.get_i4()

        if ((self.ncyc < firstmodel) or
            (self.ncyc > lastmodel) or
            ((self.ncyc - firstmodel) % stepmodel != 0)):
            self.time = None
            return

        if self.nvers < 10400:
            self._load_10300(f,**kwargs)
        else:
            self._load_10400(f,**kwargs)

        f.assert_eor()

    def _load_10400(self,
                    f,
                    consitent_convection = False,
                    **kwargs):

        # just te be sure (others to be added later)
        assert self.nvers >= 10400

        self.time = f.get_f8n()
        if self.nvers >= 10403:
            self.dt = f.get_f8n()
        self.nconv = f.get_i4()
        if self.nvers < 10401:
            layer_types = self.layer_types_10400
        else:
            layer_types = self.layer_types_10401
        nlayer = len(layer_types)
        self.nx = f.get_i4n(nlayer)
        self.ncoord = f.get_i4()
        self.idx_kind_len, self.nuc_kind_len = f.get_i4(2)

        nuc_kind = self._int_kind(self.nuc_kind_len)
        idx_kind = self._int_kind(self.idx_kind_len)

        # here we add "0" element to make layer extraction more efficient
        # ("padding" at surface is already included)
        self.nnx = np.ndarray((self.nx.size, self.nx.max() + 1),
                              dtype = nuc_kind)
        for i in range(self.nx.size):
            if self.nx[i] > 0:
                self.nnx[i,1:self.nx[i]+1] = f.get_n(self.nx[i], nuc_kind)
        self.nnx[:,0] = 0

        yzip = f.get_sn(self.nconv, 1)

        self.convtype = self._yzip2type(yzip)
        if self.nvers < 10500:
            self.mc = f.get_f8n1d0(self.nconv)
            self.rc = f.get_f8n1d0(self.nconv)
        self.xmcoord = f.get_f8n(self.ncoord)
        self.rncoord = f.get_f8n(self.ncoord)
        if self.nvers >= 10500:
            # flags in ladv
            # 1 - mass loss
            # 2 - accretion
            # 4 - decretion
            self.ladv = f.get_i4()
            self.nadv = bit_count(self.ladv)
            if self.nadv > 0:
                self.iadv  = f.get_n  (self.nadv, idx_kind)
                self.dmadv = f.get_f8n(self.nadv)
                self.dvadv = f.get_f8n(self.nadv)
                # fix bug in versions or 10500 prior to 20160417:
                #
                #     flag adv_dec (4) was always set the same as flag
                #     adv_acc (2); this cannot fix cases where flag
                #     adv_dec (4) should have been set but flag
                #     adv_acc (2) was not set, but only cases where
                #     flag adv_acc (2) was set and flag adv_dec (4)
                #     should not have been set.
                if self.nvers == 10500 and self.ladv & self.adv_dec:
                    if self.dmadv[-1] == 0:
                        self.dmadv = self.dmadv[:-1]
                        self.dvadv = self.dvadv[:-1]
                        self.iadv  = self.iadv [:-1]
                        self.ladv  = self.ladv - self.adv_dec
        self.inx = np.ndarray((self.nx.size, self.nx.max()),
                              dtype = idx_kind)
        for i in range(self.nx.size):
            if self.nx[i] > 0:
                self.inx[i,0:self.nx[i]] = f.get_n(self.nx[i], idx_kind)
        if self.nvers >= 10500:
            if self.nconv > 0:
                self.iconv = f.get_n(self.nconv, idx_kind)

                # there was a bug in versions 10500 prior to 20121018
                # where the information on the outermost zone was
                # overwritten, we try a crude fix
                #
                # 1) if outermost zone is 'O' but zone below that is
                # not 'C' then likely there was a convection zone
                # above, likely to the surface.
                if kwargs.get('fix_surf_10500', False):
                    if self.nconv >= 2:
                        if self.convtype[-2] in (2, 3):
                            self.convtype = np.insert(self.convtype, self.nconv - 1, 4)
                            self.iconv = np.insert(self.iconv, self.nconv - 1, self.iconv[-1] - 1)
                            self.nconv += 1

                self.mc = np.ndarray((self.nconv+1), dtype = np.float64)
                self.rc = np.ndarray((self.nconv+1), dtype = np.float64)
                # in principle we do not have to reconstruct
                # xmc and rc arrays here but only when/if needed.
                self.mc[1:] =  0.5 * (self.xmcoord[self.iconv-1]    + self.xmcoord[np.minimum(self.iconv,self.ncoord-1)])
                self.rc[1:] = (0.5 * (self.rncoord[self.iconv-1]**3 + self.rncoord[np.minimum(self.iconv,self.ncoord-1)]**3))**(1/3)
                # here we _could_ use same as above if we wanted to be consitent
                # but it _may_ not look as nice on plots [...]
                if consitent_convection:
                    self.mc[0] =  0.5 * (self.xmcoord[0]    + self.xmcoord[1])
                    self.rc[0] = (0.5 * (self.rncoord[0]**3 + self.rncoord[1]**3))**(1/3)
                else:
                    self.mc[0] = self.xmcoord[0]
                    self.rc[0] = self.rncoord[0]
                # TODO - integrate advection terms?
                # * they would need to go into a separate variable
                #   that represents the 'reconstructed end of time step'

        self.levcnv = f.get_i4()
        self.minx = f.get_i4n((2,nlayer)).transpose()
        if self.nvers == 10400:
            (self.tc,
             self.dc,
             self.pc,
             self.ec,
             self.sc,
             self.ye,
             self.ab,
             self.et,
             self.sn,
             self.su,
             self.g1,
             self.g2,
             self.s1,
             self.s2,
             self.xn,
             self.aw,
             self.en,
             self.summ0,
             self.radius0,
             self.an) = f.get_f8n(20)
        elif self.nvers >= 10401:
            (self.tc,
             self.dc,
             self.pc,
             self.ec,
             self.sc,
             self.ye,
             self.ab,
             self.et,
             self.sn,
             self.su,
             self.g1,
             self.g2,
             self.s1,
             self.s2) = f.get_f8n(14)
            if self.nvers < 10600:
                self.aw = np.array([f.get_f8n(), 0., 0.])
            else:
                self.aw = f.get_f8n(3)
            (self.summ0,
             self.radius0,
             self.an) = f.get_f8n(3)
        self.abun = f.get_f8n(self.nhiz)
        if self.nvers >= 10401:
            (self.eni,
             self.enk,
             self.enp,
             self.ent,
             self.epro,
             self.enn,
             self.enr,
             self.ensc,
             self.enes,
             self.enc,
             self.enpist,
             self.enid,
             self.enkd,
             self.enpd,
             self.entd,
             self.eprod,
             self.xlumn,
             self.enrd,
             self.enscd,
             self.enesd,
             self.encd,
             self.enpistd,
             self.xlum,
             self.xlum0,
             self.entloss,
             self.eniloss,
             self.enkloss,
             self.enploss,
             self.enrloss,
             self.angit) = f.get_f8n(30)
            if self.nvers < 10600:
                self.anglt = np.array([f.get_f8n(), 0., 0.])
            else:
                self.anglt = f.get_f8n(3)
        if self.nvers >= 10402:
            self.xmacc = f.get_f8n()

        # Do we need to add summ0 ?
        # would need to offset all
        #
        if False:
            self.mc += self.summ0
        # but we do need to set radius for older versions;
        # radius is always absolute
        if self.nvers < 10500:
            self.rc[0] = self.radius0
            # TODO:
            # reconstruct iconv - including updating xmcoord/rncoord?

        f.assert_eor()

    def _load_10300(self,
                    f,
                    **kwargs):

        # print(self.nvers,self.ncyc,self._byteorder)

        # just te be sure (others to be added later)
        assert 10100 <= self.nvers < 10400, f'DEBUG - {self.nvers:g}'

        if self.nvers < 10300:
            # these are versions from STERN
            # check time unit!
            pass
        self.time = f.get_f8n()
        (self.nconv,
         nnfak,
         nnfac,
         nnvak,
         self.nber) = f.get_i4(5)

        # first just load info
        nxt = np.array([nnfak,
                        nnfac,
                        nnvak])
        nlayert = nxt.size
        nnxt = np.zeros((nlayert, nxt.max()),
                        dtype = np.int32)
        for ilayer in range(nlayert):
            if nxt[ilayer] > 0:
                nnxt[ilayer,0:nxt[ilayer]] = f.get_i4n(nxt[ilayer])

        yzip = f.get_sn(self.nconv, 1)
        self.convtype = self._yzip2type(yzip)
        self.mc = f.get_f8n1d0(self.nconv)
        if self.nvers >= 10200:
            self.rc = f.get_f8n1d0(self.nconv)

        # we have never used this for anything (...)
        self.ber = f.get_f8n(self.nber)

        # load level mass coordinates
        nxt_max = nxt.max()
        if nxt_max > 0:
            nnxt_max = nnxt.max()
        else:
            nnxt_max = 0
        if nnxt_max > 0:
            me = np.zeros((nlayert, nxt_max, nnxt.max(), 2),
                          dtype = np.float64)
            for ilayer in range(nlayert):
                if nxt[ilayer] > 0:
                    for ilevel in range(nxt[ilayer]):
                        me[ilayer,
                           ilevel,
                           0:nnxt[ilayer,ilevel],
                           0:2
                           ] = f.get_f8n((2,nnxt[ilayer, ilevel])).transpose()

        # load extra data
        if self.nvers >= 10300:
            (self.tc,
             self.dc,
             self.pc,
             self.ec,
             self.sc,
             self.ye,
             self.ab,
             self.et,
             self.sn,
             self.su,
             self.g1,
             self.g2,
             self.s1,
             self.s2,
             self.an) = f.get_f8n(15)
        self.abun = f.get_f8n(self.nhiz)
        if self.nvers >= 10301:
            self.xn = f.get_f8n()
        if self.nvers >= 10302:
            self.aw = f.get_f8n()
        if self.nvers >= 10303:
            self.summ0 = f.get_f8n()
            (levcnv,
             minloss,
             mingain,
             minnucl) = f.get_i4(4)
        else:
            summ0 = np.float64(0.)
            (levcnv,
             minloss,
             mingain,
             minnucl) = (0,-1,-1,-1)
        if self.nvers >= 10304:
            self.en = f.get_f8n()

        f.assert_eor()

        # convert mass layer data into level/coordinate format (104xx)
        layer_types = self.layer_types_10100
        nlayer = len(layer_types)

        self.minx = np.array([[minloss, mingain],
                              [minnucl,       0]])

        nx = np.zeros((nlayer),
                      dtype = np.int32)

        # check for empty layer data
        if nnxt_max == 0:
            self.nx = nx
            return

        # coordinates and index
        self.xmcoord, idxcoord = np.unique(me,
                                           return_inverse = True)
        self.ncoord = len(self.xmcoord)
        ie = idxcoord.reshape(me.shape)

        # construct full levels
        sx = np.zeros((nlayer,self.ncoord+2), dtype = np.int32)
        ilayer = 0
        ilayert = 0
        for ilevel in range(nxt[ilayert]):
            for iregion in range(nnxt[ilayert,ilevel]):
                for icoord in range(*ie[ilayert,ilevel,iregion,:]):
                    sx[ilayer,icoord+1] = ilevel+1
        ilayer = 0
        ilayert = 1
        for ilevel in range(nxt[ilayert]):
            for iregion in range(nnxt[ilayert,ilevel]):
                for icoord in range(*ie[ilayert,ilevel,iregion,:]):
                    sx[ilayer,icoord+1] = -(ilevel+1)
        ilayer = 1
        ilayert = 2
        for ilevel in range(nxt[ilayert]):
            for iregion in range(nnxt[ilayert,ilevel]):
                for icoord in range(*ie[ilayert,ilevel,iregion,:]):
                    sx[ilayer,icoord+1] = ilevel+1

        # construct layer level arrays
        nnx = np.zeros((nlayer,self.ncoord+1),
                       dtype = np.int32)
        inx = np.zeros((nlayer,self.ncoord+1),
                       dtype = np.int32)
        for ilayer in range(nlayer):
            for icoord in range(1,self.ncoord+2):
                if sx[ilayer, icoord] != sx[ilayer, icoord-1]:
                    nnx[ilayer,nx[ilayer]] = sx[ilayer, icoord-1]
                    inx[ilayer,nx[ilayer]] = icoord-1
                    nx[ilayer] += 1

        self.nx = nx
        self.nnx = nnx[:, 0:self.nx.max() + 1]
        self.inx = inx[:, 0:self.nx.max() + 1] + 1

        # Do we need to add summ0 ?
        # would need to offset all
        if False:
            self.mc += self.summ0
        # but we do need to set radius for older versions;
        # radius is always absolute
        # self.rc[0] = self.radius0
        # unfortunately this is not stored prior to 10400
        # rc i only saved above 10200, let's take the best we have
        if self.nvers >= 10200:
            if len(self.rc) > 0:
                self.rc[0] = self.rc[1]
