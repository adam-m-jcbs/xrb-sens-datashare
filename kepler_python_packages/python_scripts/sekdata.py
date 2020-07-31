#! /bin/env python3

"""
read sek files and write to h5 for NuGrid
"""

import datetime
import numpy as np
import os.path
import sys

from human import byte2human
from human import time2human
from human import version2human
from fortranfile import FortranReader

from se import seFile
from se_filename import se_filename

import uuid
import utils
from utils import CachedAttribute, cachedmethod
from loader import loader, _loader
from logged import Logged

UUID_NBYTES = 16

import physconst

#    Convective structure (plus stability)
# 0      ... stable
# 1 neut ... "neutral" -- weak stabilization
# 2 osht ... convective overshooting
# 3 semi ... semiconvective
# 4 conv ... convective
# 5 thal ... thermohaline

class SekData(Logged):
    """
    Class for converting KEPLER sek files into se.h5 files.
    """
    _extension = 'sek'
    def __init__(self, filename, silent = False, **kwargs):
        """
        Load *.sek file from disk.  Requires file name.
        """
        self.setup_logger(silent)
        filename = os.path.expanduser(filename)
        self.filename = os.path.expandvars(filename)
        self.file = FortranReader(self.filename)
        self.load_records(**kwargs)
        self.file.close()
        self.close_logger(timing = 'Data loaded in')

    def load_records(self, lastmodel = None, **kwargs):
        """Load the KEPLER sek file."""
        start_time = datetime.datetime.now()

        self.logger.info('Loading {} ({})'\
              .format(self.file.filename,\
                      byte2human(self.file.stat.st_size)))
        self.data = []
        while not self.file.eof():
            record = SekRecord(self.file)
            if lastmodel is not None and record.ncyc > lastmodel:
                break
            self.data.append(record)
        end_time = datetime.datetime.now()
        load_time = end_time - start_time
        self.models = len(self.data)
        self.logger.info('version {}'.format(version2human(self.data[0].nvers)))
        self.logger.info('first model red {:>9d}'.format(self.data[ 0].ncyc))
        self.logger.info(' last model red {:>9d}'.format(self.data[-1].ncyc))
        self.logger.info('{:8d} models loaded in {}'.format(self.models,time2human(load_time.total_seconds())))

    def sewrite(self,
                path = None,
                group_size = 1000,
                first = 0,
                last = -1,
                compression_level = 0,
                comment = 'kepler',
                metallicity = None,
                mass = None,
                progress = True):
        """Write the data into SE file."""

        self.setup_logger(False)

        self.progress = progress

        if path is None:
            path = os.path.dirname(self.filename)

        path = os.path.expanduser(path)
        path = os.path.expandvars(path)

        icyc = self.data[first].ncyc
        ncyc = self.data[last].ncyc -icyc + 1
        ngroups = (ncyc - 1) // group_size + 1
        groups = [group_size * i for i in range(ngroups+1)]
        groups[-1] = ncyc
        if mass is None:
            mass = self.data[0].totm0 / physconst.Kepler.solmass
        if metallicity is None:
            metallicity = self.data[0].zinit
        for i in range(ngroups):
            nstart = groups[i]
            nstop = groups[i+1]-1
            hdf5file = os.path.join(
                path,
                se_filename(
                    mass = mass,
                    metallicity = metallicity,
                    cycle = self.data[nstart].ncyc,
                    comment = comment))
            self.logger.info('Creating {}'.format(hdf5file))
            self._se_write_group(hdf5file,
                                 nstart,
                                 nstop,
                                 compression_level = compression_level)

        self.close_logger(timing = 'data written in ')

    def _se_write_group(self,
                        hdf5file,
                        nstart,
                        nstop,
                        compression_level = 0):
        """Write out global attributes range of records."""

        self.setup_logger(False)

        codev = "kepler {}".format(version2human(self.data[0].nvers))
        modname = self.filename
        mini = self.data[0].totm0 / physconst.Kepler.solmass
        zini = self.data[0].zinit
        rotini = self.data[0].anglint
        overini = self.data[0].woversht

        try:
            os.remove(hdf5file)
        except:
            pass
        se = seFile(hdf5file)
        se.compression_level = compression_level

        # write global attributes
        se.writeattr(-1, "codev", codev)
        se.writeattr(-1, "modname", modname)
        se.writeattr(-1, "mini", mini)
        se.writeattr(-1, "zini", zini)
        se.writeattr(-1, "rotini", rotini)
        se.writeattr(-1, "overini", overini)

        # write some extra data from KEPLER
        se.writeattr(-1, "UUID_run", \
                     str(uuid.UUID(bytes=self.data[0].uuidrun)))

        # units
        age_unit = 1.e0
        mass_unit = physconst.Kepler.solmass
        radius_unit = 1.e0
        rho_unit = 1.e0
        temperature_unit = 1.e0
        pressure_unit = 1.e0
        velocity_unit = 1.e0
        dcoeff_unit = 1.e0
        energy_unit = 1.e0

        se.writeattr(-1, "age_unit",age_unit)
        se.writeattr(-1, "mass_unit",mass_unit)
        se.writeattr(-1, "radius_unit",radius_unit)
        se.writeattr(-1, "rho_unit",rho_unit)
        se.writeattr(-1, "temperature_unit",temperature_unit)
        se.writeattr(-1, "pressure_unit",pressure_unit)
        se.writeattr(-1, "velocity_unit",velocity_unit)
        se.writeattr(-1, "dcoeff_unit",dcoeff_unit)
        se.writeattr(-1, "energy_unit",energy_unit)

        # write out cycle information
        ncyclenb = nstop - nstart + 1
        ifirstcycle = self.data[nstart].ncyc
        ilastcycle = self.data[nstop].ncyc
        se.writeattr(-1, "icyclenb",ncyclenb)
        se.writeattr(-1, "ifirstcycle", ifirstcycle)
        se.writeattr(-1, "ilastcycle", ilastcycle)

        for i in range(nstart, nstop + 1):
            self.data[i].sewrite(se)
            if self.progress:
                print('.', end = '', flush = True)
        if self.progress:
            print('!')
        se.close()
        self.close_logger(timing = 'file written in ')


load = loader(SekData, 'sekdata')
_load = _loader(SekData, 'sekdata')
loadwind = load

class SekRecord(object):
    """Read/Write individual records for sek/se.h5 files."""
    def __init__(self,file,data=True):
        self.file = file
        self.load(data)

    def load(self, data = True):
        """Load sek file record."""
        f = self.file
        f.load()
        self.nvers = f.get_i4()
        self.ncyc = f.get_i4()
        if data:
            self.timesec = f.get_f8n()
            self.dt = f.get_f8n()
            jm = f.get_i4()
            self.jm = jm
            self.totm0,\
                       self.anglint,\
                       self.zinit,\
                       self.woversht = \
                       f.get_f8n(4)
            self.uuidrun, self.uuidcycle = \
                           f.get_buf(2, length = UUID_NBYTES)
            self.teff, self.xlum, self.radius =\
                       f.get_f8n(3)
            nysel = f.get_i4()
            self.nysel = nysel
            self.zysel = f.get_f8n(nysel)
            self.aysel = f.get_f8n(nysel)
            self.xm = f.get_f8n(jm+1)
            self.rn = f.get_f8n(jm+1)
            self.un = f.get_f8n(jm+1)
            self.dn = f.get_f8n1d0(jm)
            self.tn = f.get_f8n1d0(jm)
            self.pn = f.get_f8n1d0(jm)
            self.sn = f.get_f8n1d0(jm)
            self.xdifi = f.get_f8n1d0n(jm-1)
            self.iconv = f.get_i4n1d0n(jm-1)
            self.abu = f.get_f8n((nysel, jm))
            self.abu = np.insert(self.abu,0,0,axis=1)

            # check we read all the data
            f.assert_eor()

    def sewrite(self, se):
        """Write cycle data to SE file."""
        mass =  self.xm.cumsum()
        structure = dict()
        structure["mass"] = mass[::-1] / physconst.Kepler.solmass
        structure["delta_mass"] = self.xm[::-1] / physconst.Kepler.solmass
        structure["pressure"] = self.pn[::-1]
        structure["temperature"] = self.tn[::-1]
        structure["radius"] = self.rn[::-1]
        structure["rho"] = self.dn[::-1]
        structure["velocity"] = self.un[::-1]
        structure["dcoeff"] = self.xdifi[::-1]
        structure["convection_indicator"] = self.iconv[::-1]
        structure["energy_generation"] = self.sn[::-1]
        structure["iso_massf"] = self.abu[:,::-1].swapaxes(0,1)
        se.write(self.ncyc, structure)
        se.writearrayattr(self.ncyc, "Z", self.zysel)
        se.writearrayattr(self.ncyc, "A", self.aysel)

        se.writeattr(self.ncyc, "model_number", self.ncyc)
        se.writeattr(self.ncyc, "total_mass", sum(self.xm) / physconst.Kepler.solmass)
        se.writeattr(self.ncyc, "age", self.timesec)
        se.writeattr(self.ncyc, "deltat", self.dt)
        se.writeattr(self.ncyc, "shellnb", self.jm)
        se.writeattr(self.ncyc, "logL", np.log10(max(self.xlum / physconst.XLSUN,1.e0)))
        se.writeattr(self.ncyc, "logTeff", np.log10(max(self.teff,1.e0)))

        # write some extra data from KEPLER
        se.writeattr(self.ncyc, "UUID_cycle", str(uuid.UUID(bytes=self.uuidcycle)))

if __name__ == '__main__':
    import sys
    s = SekData(sys.argv[1])
    if len(sys.argv) > 2:
        path = sys.argv[2]
    else:
        path = os.path.dirname(sys.argv[1])
    s.sewrite(path)
