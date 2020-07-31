#! /bin/env python3


"""
This module provides an interface to from KEPLER dump files to
NuGrid SE files.

The provided class SeKepDump extends the DumpFile class and provides
the sewrite routine.
"""

import os
import os.path
import datetime
import uuid
import numpy as np

from human import byte2human
from human import time2human
from human import version2human

import physconst

from kepion import KepIon
from kepdump import KepDump

# for se_writing
from se_filename import se_filename
from se import seFile

def load(*args, **kwargs):
    return SeKepDump(*args, **kwargs)

class SeKepDump(KepDump):
    """
    Write KEPLER dump files to SE files.

    Method: sewrite

    KEYWORDS
    dir - where to write files  [.]
    burn - write out BURN abundances [True].
    compression_level - passed to se before first write[0].
    """

    def sewrite(self,
                path = None,
                burn = True,
                compression_level = 0,
                comment = 'kepler',
                metallicity = None,
                silent = None):
        """
        Write the data into SE file.
        """

        self.setup_logger(silent)
        if path is None:
            path = os.getcwd()
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)

        jm = self.qparm.jm
        ncyc = self.qparm.ncyc

        if metallicity is None:
            metallicity = self.qparm.zinit

        hdf5file = os.path.join(path, se_filename(
            mass = self.parm.totm0 / physconst.Kepler.solmass,
            metallicity = metallicity,
            cycle = ncyc,
            comment = comment))
        try:
            os.remove(hdf5file)
        except:
            pass
        se = seFile(hdf5file)
        se.compression_level = compression_level

        self.logger.info('Creating SE File {}'.format(hdf5file))

        codev = "kepler {}".format(version2human(self.parm.nsetparm))
        modname = self.filename
        mini = self.parm.totm0 / physconst.Kepler.solmass
        zini = self.qparm.zinit
        rotini = self.qparm.anglint
        overini = self.parm.woversht

        # write global attributes
        se.writeattr(-1, "codev", codev)
        se.writeattr(-1, "modname", modname)
        se.writeattr(-1, "mini", mini)
        se.writeattr(-1, "zini", zini)
        se.writeattr(-1, "rotini", rotini)
        se.writeattr(-1, "overini", overini)

        # units
        age_unit = 1.e0
        mass_unit = physconst.Kepler.solmass
        radius_unit=1.e0
        rho_unit=1.e0
        temperature_unit=1.e0
        pressure_unit=1.e0
        velocity_unit=1.e0
        dcoeff_unit=1.e0
        energy_unit=1.e0

        se.writeattr(-1, "age_unit", age_unit)
        se.writeattr(-1, "mass_unit", mass_unit)
        se.writeattr(-1, "radius_unit", radius_unit)
        se.writeattr(-1, "rho_unit", rho_unit)
        se.writeattr(-1, "temperature_unit", temperature_unit)
        se.writeattr(-1, "pressure_unit", pressure_unit)
        se.writeattr(-1, "velocity_unit", velocity_unit)
        se.writeattr(-1, "dcoeff_unit", dcoeff_unit)
        se.writeattr(-1, "energy_unit", energy_unit)

        # write out cycle information
        ncyclenb = 1
        ifirstcycle = ncyc
        ilastcycle = ncyc
        se.writeattr(-1, "icyclenb",ncyclenb)
        se.writeattr(-1, "ifirstcycle", ifirstcycle)
        se.writeattr(-1, "ilastcycle", ilastcycle)

        # write some extra data from KEPLER
        try:
            se.writeattr(-1, "UUID_run",
                          str(uuid.UUID(bytes=self.uuidrun)))
        except:
            pass

        # Now write out per-cycle data
        iconv = np.zeros(jm + 2, dtype = np.int32)
        for j in range(1,jm - 1):
            icon = self.icon[j].strip()
            if icon ==  'conv':
                iconv[j] = 1
            elif icon == 'semi':
                iconv[j] = 2
            elif icon == 'osht':
                iconv[j] = 3
            elif icon == 'that':
                iconv[j] = 4
            elif icon == 'neut':
                iconv[j] = 5

        xm =  self.xm.copy()
        xm[0] = self.parm.summ0
        mass = xm.cumsum()

        if not hasattr(self,"bfdiffef"):
            self.bfdiffef = np.zeros(jm + 2, dtype = np.float64)
        if not hasattr(self,"angdgeff"):
            self.angdgeff = np.zeros(jm + 2, dtype = np.float64)
        if not hasattr(self,"difieff"):
            self.difieff = np.zeros(jm + 2, dtype = np.float64)
        xdifi = self.parm.difim * (self.difieff + self.parm.angfc * self.angdgeff + self.bfdiffef)

        dn = self.dn.copy()
        pn = self.pn.copy()
        tn = self.rn.copy()

        dn[0] = 0
        pn[0] = 0
        tn[0] = 0

        structure = dict()
        structure["mass"] = mass[-2::-1] / physconst.Kepler.solmass
        structure["delta_mass"] = xm[-2::-1] / physconst.Kepler.solmass
        structure["pressure"] = pn[-2::-1]
        structure["temperature"] = tn[-2::-1]
        structure["radius"] =   self.rn[-2::-1]
        structure["rho"] = dn[-2::-1]
        structure["velocity"] = self.un[-2::-1]
        structure["dcoeff"] = xdifi[-2::-1]
        structure["convection_indicator"] = iconv[-2::-1]
        structure["energy_generation"] = self.sn[-2::-1]
        if burn:
            A = self.aionb
            Z = self.zionb
            iso_name = np.array([ion.strip() for ion in self.ionsb])
            y = self.ppnb[:,-2::-1] * A[:,np.newaxis]
        else:
            nionmax = len(self.ions)
            y = np.ndarray([nionmax,jm+2], dtype = np.float64)
            isel = np.zeros(nionmax)
            for j in range(1,jm):
                net = self.netnum[j]
                iion = self.ionn[0:18,net-1]-1
                y[iion,j] = self.ppn[0:18,j]
                isel[iion] = 1
            isel = np.where(isel == 1)[0]
            ions = [KepIon(self.ions[i]) for i in isel]
            A = [ion.A for ion in ions]
            Z = [ion.A for ion in ions]
            iso_name = [ion.name() for ion in ions]
            y = y[isel,-2::-1] * np.maximum(A,1)[:,np.newaxis]

        iso_name = [i.encode() for i in iso_name]
        structure["iso_massf"] = y.swapaxes(0,1)
        se.write(ncyc, structure)

        se.writearrayattr(ncyc, "Z", Z)
        se.writearrayattr(ncyc, "A", A)
        se.writearrayattr(ncyc, "iso_name", iso_name)

        age = self.parm.time + self.parm.toffset

        totm = self.qparm.totm + self.parm.summ0

        se.writeattr(ncyc, "model_number", ncyc)
        se.writeattr(ncyc, "total_mass", totm / physconst.Kepler.solmass)
        se.writeattr(ncyc, "age", age )
        se.writeattr(ncyc, "deltat", self.qparm.dt)
        se.writeattr(ncyc, "shellnb", jm)
        se.writeattr(ncyc, "logL", np.log10(max(self.qparm.xlum / physconst.XLSUN,1.e0)))
        se.writeattr(ncyc, "logTeff", np.log10(max(self.qparm.teff,1.e0)))

        # write some extra data from KEPLER
        try:
            se.writeattr(ncyc, "UUID_run",
                         str(uuid.UUID(bytes=self.uuidrun)))
            se.writeattr(ncyc, "UUID_cycle",
                         str(uuid.UUID(bytes=self.uuidcycle)))
            se.writeattr(ncyc, "UUID_dump",
                         str(uuid.UUID(bytes=self.uuiddump)))
            se.writeattr(ncyc, "UUID_prog",
                         str(uuid.UUID(bytes=self.uuidprog)))
        except:
            pass
        se.close()

        self.close_logger(timing = 'SE data written in ')

if __name__ == '__main__':
    import sys
    s = loaddump(sys.argv[1])
    if len(sys.argv) > 2:
        path = sys.argv[2]
    else:
        path = os.path.dirname(sys.argv[1])
    s.sewrite(path)
