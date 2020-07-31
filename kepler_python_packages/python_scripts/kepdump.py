"""
Interface to KEPLER restart dump files.
"""

import sys
import os
import os.path
import string
import builtins
import collections
import datetime
import numpy as np
import re
from numpy import linalg as LA

import keppar
import physconst
import logging
from functools import partial

from fortranfile import FortranReader, FortranWriter
from indexdict import IndexDict
from human import byte2human
from human import time2human
from human import version2human
from logged import Logged
from utils import CachedAttribute, cachedmethod
from loader import loader, _loader

import uuid
import logdata
from isotope import ion, Ion
from abuset import IonList, AbuSet, AbuDump
from kepion import KepIon, KepAbuDump
from keputils import mass_string

from rotation.rot import w2p, w2c

from uuidtime import UUID_NBYTES

# current kepler version - TODO: Try to generate/get from kepler source
ncurvers = 171000

class KeplerWriteError(Exception):
    def __init__(self, reason = None):
        s = 'KEPLER Dump write error'
        if reason is not None:
            s += ':\nReason: ' + reason
        super().__init__(s)

class KepDumpSlice(object):
    @CachedAttribute
    def center_slice(self):
        """
        Return slice for indices of zone centered quantities.
        """
        return slice(1, self.jm + 1)

    slice = center_slice
    zone_slice = center_slice

    @CachedAttribute
    def upper_slice(self):
        """
        Return slice for indices of zone upper boundaries.
        """
        return slice(1, self.jm + 1)

    @CachedAttribute
    def lower_slice(self):
        """
        Return slice for indices of zone lower boundaries.
        """
        return slice(0, self.jm)

    @CachedAttribute
    def boundary_slice(self):
        """
        Return slice for indices of all zone boundaries.
        """
        return slice(0, self.jm + 1)

    @CachedAttribute
    def interface_slice(self):
        """
        Return slice for indices of all zone interfaces.
        """
        return slice(1, self.jm)

    @CachedAttribute
    def yield_slice(self):
        """
        Return slice for indices of all zones with yields including wind.
        """
        return slice(1, self.jm + 2)

    @CachedAttribute
    def all_slice(self):
        """
        Return slice for indices of all zones including wind.
        """
        return slice(0, self.jm + 2)

    @CachedAttribute
    def wind_zone(self):
        """
        Return index of wind zone.
        """
        return self.jm + 1

    @CachedAttribute
    def core_zone(self):
        """
        Return index of core zone.
        """
        return 0

    @staticmethod
    def face2center(x):
        """
        create zone centered quantity by averaging face-defined values
        """
        y = x.copy()
        y[1:] += x[:-1]
        y     *= 0.5
        y[ 0]  = x[0]
        y[-1]  = x[-2]
        return y

    @staticmethod
    def center2face(x, logarithmic=False):
        """
        create zone interface quantity by averaging zone values
        """
        y = x.copy()
        if logarithmic:
            y[:-1] *= x[1:]
            y[1:-2] = np.sqrt(y[1:-2])
        else:
            y[:-1] += x[1:]
            y      *= 0.5
        y[0]    = x[1]
        y[-2]   = x[-2]
        y[-1]   = np.nan
        return y

class KepDump(Logged, KepDumpSlice):
    """Access to KEPLER dump files."""
    head_names = ('nvers0', 'ncyc0', 'lenhead0', 'jmz0', 'jmzb0', 'nburn0', 'iratioz0', 'nvar0', 'nheadz0', 'nhead0', 'nparmz0', 'nparm0', 'nqparmz0', 'nqparm0', 'nniz0', 'nhiz0', 'nitz0', 'nnizb0', 'nhizb0', 'nitzb0', 'nreacz0', 'ndtz0', 'ndt0', 'npistz0', 'nyez0', 'nsubz0', 'nsub0', 'nzonei0', 'nzonec0', 'nzoneb0', 'nsmall0', 'nsmallc0', 'imax0', 'imaxb0', 'nreac0', 'jmsave', 'lencom0', 'lencomc0', 'nedtcom0', 'ndatqz0', 'ngridz0', 'nylibz0', 'nyoffst0', 'lenshed0', 'lenqhed0', 'nzedz0', 'ncsavdz0')

    def __init__(self,
                 filename,
                 killburn = False,
                 silent = True):
        """Constructor - provide name of dump file."""
        self.setup_logger(silent)
        filename = os.path.expanduser(filename)
        self.filename = os.path.expandvars(filename)
        self.file = FortranReader(self.filename)
        self._load(killburn = killburn)
        self.file.close()
        self.close_logger()

    def _load(self, killburn = False):
        """
        Open file, call load data, time the load, print out diagnostics.
        """
        start_time = datetime.datetime.now()
        self.logger.info('Loading {} ({})'.format(
            self.file.filename,
            byte2human(self.file.stat.st_size)))
        self.load_data(killburn = killburn)
        self.file.close()
        end_time = datetime.datetime.now()
        load_time = end_time - start_time

        self.logger.info('Loaded model  {:12d}, time (s)   = {:12.5g}'.format(
            int(self.qparm.ncyc),self.parm.time))
        self.logger.info('  Tc (K)    = {:12.5g}, Dc (g/cm3) = {:12.5g}'.format(
            self.tn[1],self.dn[1]))
        self.logger.info('mass (Msun) = {:12.5g}, Mi (Msun)  = {:12.5g}'.format(
            self.qparm.totm / physconst.Kepler.solmass, self.parm.totm0 / physconst.Kepler.solmass))
        s = '   Version  = {:>12s}'.format(version2human(self.parm.nsetparm))
        if self.qparm.imaxb > 0:
            s += ', # burn iso = {:12d}'.format(int(self.qparm.imaxb))
        self.logger.info(s)
        self.logger.info('Data loaded in ' + time2human(load_time.total_seconds()))

    def load_burn(self, killburn = False):
        """
        Load BURN data.
        """
        nvers = self.nvers
        f = self.file

        # problematic because we can't jump ....
        if killburn:
            # do not load BURN data if killburn was requested
            self.parm['nsaveb']  = 0
            self.qparm['imaxb']  = 0
            self.parm['irecb']   = 0
            self.parm['inburn']  = 0
            self.parm['lburn']   = 0
            self.parm['iburnye'] = 0

        # burn co-processor arrays and abundances
        imaxb = self.qparm.imaxb
        nsaveb = self.parm.nsaveb
        jmzb0 = self.head.jmzb0
        jmsave1 = self.head.jmsave + 1
        jm = int(self.qparm.jm)
        if (imaxb == 0) or (jmzb0 == 1) or (nsaveb == 0):
            return

        nbdum1=jmsave1
        # but really, it should be 12 ...
        assert nsaveb == 10,\
               "ERROR bdum: nsaveb = {:d}, required = {:d}"\
               .format(nsaveb, 10)
        self.netnumb = f.load_f8_kep_i4n([nbdum1])
        self.limab   = f.load_f8_kep_i4n([nbdum1])
        self.limzb   = f.load_f8_kep_i4n([nbdum1])
        self.limcb   = f.load_f8_kep_i4n([nbdum1])
        self.timen   = f.load_f8n([nbdum1])
        self.dtimen  = f.load_f8n([nbdum1])
        self.dnold   = f.load_f8n([nbdum1])
        self.tnold   = f.load_f8n([nbdum1])
        self.ymb     = f.load_f8n([nbdum1])
        self.sburn   = f.load_f8n([nbdum1])
        # it seems the following have dropped because by default
        # nsaveb parameter is set to 10.  Should be 12
        if nsaveb == 12:
            self.etab    = f.load_f8n([nbdum1])
            self.pbuf    = f.load_f8n([nbdum1])
        # mabe older versions of KEPLER do this differently.

        # load ppnb
        self.ppnb = f.load_f8n([imaxb,jm])
        self.ppnb = np.insert(self.ppnb,[0,jm],0,axis=1)

        # burn recording data
        if nvers > 160350:
            irecb = self.parm.irecb
            if irecb == 1:
                if nvers < 160850:
                    nbmax=imaxb
                    self.nbmax = np.int32(nmaxb)
                    self.nabmax = np.array(int(round(self.aionb(i))), dtype=np.int32)
                    self.nzbmax = np.array(int(round(self.zionb(i))), dtype=np.int32)
                    self.nibmax = np.array(range(nbmax)+1, dtype=np.int32)
                    self.ionbmax = np.array(self.ionsb, dtype=np.dtype((np.str, 8)))
                else:
                    self.nbmax = f.load_i4n()
                    nbmax = int(self.nbmax)
                    self.nabmax = f.load_i4n([nbmax])
                    self.nzbmax = f.load_i4n([nbmax])
                    self.nibmax = f.load_i4n([imaxb])
                    self.ionbmax = f.load_sn([nbmax], 8)
                self.burnamax = f.load_f8n([nbmax])
                self.burnmmax = f.load_f8n([nbmax])
                self.ibcmax = f.load_i4n([nbmax])

        # accreation composition
        if nvers > 162250:
            self.compsurfb = f.load_f8n([imaxb])



    def load_data(self, killburn = False):
        """The routine that actually loads the data"""
        f = self.file

        # load head
        f.load()
        # starting with KEPLER 16.92 we write version and cycle number
        # in the first 8 bytes.  This can be used to modify all the
        # subsequent layout.  For now, only, we just read this in as
        # part of ihdum (stored in self.head).
        lenhead2 = f.reclen // 4
        ihdum = f.get_i4(lenhead2)
        self.head = IndexDict(self.head_names,ihdum[:len(self.head_names)])
        self.nhead = lenhead2
        assert self.nhead == 2 * self.head.lenhead0
        f.assert_eor()

        nvers = self.head.nvers0
        ncyc  = self.head.ncyc0

        if nvers//100 > ncurvers//100:
            self.logger.warn('Loading binary version that is newer than script:')
            self.logger.warn('  This may or may not work.')

        # load p parameters
        f.load()
        nparmz = self.head.nparmz0
        if nvers < 170000:
            xversion = f.peek_f8(offset=66*8)
        pk = keppar.p.copy()
        nparmk = len(pk)
        if nparmk < nparmz:
            self.logger.warn('Parameter list too short.')
            for i in range(nparmk + 1, nparmz + 1):
                pk['px{:04d}'.format(i - 1)] = -1
        parm = f.get_kep_parm(list(pk.values())[:nparmz+1])
        self.parm = IndexDict(list(pk.keys()), parm)
        # in old versions the number of parameters stored used to be 500
        nparmz_old = 500
        if nparmz < nparmz_old:
            f.skip_bytes((nparmz_old - nparmz) * 8)
        f.assert_eor()

        # convert version number
        if nvers < 170000:
            nvers = int(10000 * xversion + 0.1)
            self.parm.nsetparm = nvers
        self.nvers = nvers

        # load q parameters
        f.load()
        nqparmz = self.head.nqparmz0
        qk = keppar.q.copy()
        qparm = f.get_kep_parm(list(qk.values())[:nqparmz+1])
        nqparmk = len(qk)
        if nqparmk < nqparmz:
            self.logger.warn('Edit parameter list too short.')
            for i in range(nqparmk + 1, nqparmz + 1):
                qk['qx{:04d}'.format(i - 1)] = -1
        self.qparm=IndexDict(list(qk.keys()), qparm)
        f.skip_bytes(max(0,(nqparmz - len(self.qparm) + 1)*8))
        f.assert_eor()

        if nvers < 170000:
            ncyc = self.qparm.ncyc
        self.ncyc = ncyc

        # load sdum
        f.load()
        assert self.head.nsmall0 == f.reclen // 8, \
               "ERROR sdum: small0 = {:d}, reclen = {:d}"\
               .format(self.head.nsmall0, f.reclen // 8)
        f.skip_bytes(8)
        #piston position and ye initialization arrays
        npistz = self.head.npistz0
        self.tpist = f.get_f8n([npistz])
        self.rpist = f.get_f8n([npistz])
        nyez = self.head.nyez0
        self.yemass = f.get_f8n([nyez])
        self.yeq0 = f.get_f8n([nyez])
        # ion arrays
        nitz = self.head.nitz0
        self.aion = f.get_f8n([nitz])
        self.zion = f.get_f8n([nitz])
        nniz = self.head.nniz0
        nhiz = self.head.nhiz0
        self.numi = f.get_f8_kep_i4n([nniz])
        self.ionn = f.get_f8_kep_i4n([nhiz, nniz])
        # burn ion arrays
        if nvers < 159950:
            imaxb = self.head.nitzb0
        else:
            imaxb = self.head.imaxb0
        nnizb = self.head.nnizb0
        self.aionb = f.get_f8n([imaxb])
        self.zionb = f.get_f8n([imaxb])
        self.numib = f.get_f8_kep_i4n([nnizb])
        self.ionnb = f.get_f8_kep_i4n([imaxb,nnizb])
        # time-step controller arrays
        ndtz = self.head.ndtz0
        self.dtc = f.get_f8n([ndtz])
        self.jdtc = f.get_f8_kep_i4n([ndtz])
        # subroutine timing array
        nsubz = self.head.nsubz0
        self.timeused = f.get_f8n([3,nsubz+1])
        #  reaction arrays
        nreacz = self.head.nreacz0
        self.totalr = f.get_f8n([nreacz])
        self.rater  = f.get_f8n([nreacz])
        self.qval   = f.get_f8n([nreacz])
        self.jrate  = f.get_f8_kep_i4n([nreacz])
        self.rrx    = f.get_f8n([nreacz])
        # accretion composition array
        self.compsurf = f.get_f8n([nhiz])
        # post-processor dump arrays (those should be removed)
        ndatqz = self.head.ndatqz0
        self.locqz    = f.get_f8_kep_i4n([ndatqz])
        self.locqz0   = f.get_f8_kep_i4n([ndatqz])
        self.ratzdump = f.get_f8n([ndatqz])
        self.ratiodez = f.get_f8n([ndatqz])
        self.ratioadz = f.get_f8n([ndatqz])
        # user-specified edit arrays (are these still used?)
        nzedz = self.head.nzedz0
        self.ndatzed  = f.get_f8_kep_i4n([nzedz])
        self.ncyczed  = f.get_f8_kep_i4n([nzedz])
        self.zedmass1 = f.get_f8n([nzedz])
        self.zedmass2 = f.get_f8n([nzedz])
        # record of isotope mass lost in wind
        if nvers > 151350:
            self.wind = f.get_f8n([nitz])
            self.windb = f.get_f8n([imaxb])
        f.assert_eor()

        # load cdum
        f.load()
        if nvers < 159950:
            ncdum0=self.head.nsmallc0 + 1 + self.head.jmsave
        else:
            ncdum0=self.head.nsmallc0
        assert ncdum0 == f.reclen // 8, \
               "ERROR cdum: ncdum0 = {:d}, reclen = {:d}"\
               .format(ncdum0, f.reclen // 8)
        f.skip_bytes(8)
        # names, flags, and id-words (char*8) (except namec0 is char*16)
        self.namep0 = f.get_s(8)
        self.namec0 = f.get_s(16)
        self.iflag80, self.iqbrnflg, self.craybox, self.idword\
            = f.get_s(4, 8)
        # storage directory, last run and code mod dates (char*16)
        self.nxdirect, self.lastrun, self.lastmod0\
            = f.get_s(3, 16)
        # arrays of symbols for isotopes, burn isotopes, time-step
        # controlers, and reactions (char*8)
        self.ions = f.get_sn(nitz, 8)
        self.ionsb = f.get_sn(imaxb, 8)
        self.idtcsym = f.get_sn(ndtz, 8)
        self.isymr = f.get_sn(nreacz, 8)
        # post-processor file names (char*16)
        self.nameqq, self.nameqlib, self.nameolds, self.namenews\
            = f.get_s(4, 16)
        # arrays of post-processor dump variable names (char*8)
        # and labels (char*48)
        self.namedatq = f.get_sn(ndatqz, 8)
        self.labldatq = f.get_sn(ndatqz, 48)
        # array of edit variable names for user-specified edits (char*8)
        self.namedzed = f.get_sn((nzedz, 10), 8)
        # array of aliases
        self.savdcmd0 = f.get_sn(30, 80)
        # remembered 'look' post-processing variable names (char*8) and
        # labels (char*48)
        self.namedatl = f.get_sn(ndatqz, 8)
        self.labldatl = f.get_sn(ndatqz, 48)
        # 'look'-read post-processor file names (char*16)
        self.nameqql0, self.nameqql1, self.nameqlbl\
            =  f.get_sn(3, 16)
        # output storage directory (char*48)
        self.nsdirect = f.get_s(48)
        # set of isotopes to be plotted (char*8) and their
        # plot icons (char*16)
        self.isosym = f.get_sn(50, 8)
        self.isoicon = f.get_sn(50, 16)
        # array of remembered tty command strings (char*80)
        ncsavedz = self.head.ncsavdz0
        self.savedcmd = f.get_sn(ncsavedz, 80)
        # path for location of data file (char*80)
        self.datapath = f.get_s(80)
        # array of zonal convection sentinels (char*8)
        jmsave1 = self.head.jmsave + 1
        # this array was one longer than needed.
        if nvers > 149950:  # I know it has to be < 152600
            self.icon = f.get_sn(jmsave1, 8)
        # finally check length
        f.assert_eor()

        # load zdum
        nsavez = int(self.parm.nsavez)
        if nvers < 171000:
            nangjdz = 1
        else:
            nangjdz = 3
        nsavez_actual = 32 + nangjdz
        self.nangjd = 3
        self.nangmd = 5
        # version ~> 150100
        nangmdz = self.nangmd
        assert nsavez == nsavez_actual,\
               "ERROR zdum: nsavez = {:d}, loading = {:d}"\
               .format(nsavez, nsavez_actual)
        self.ym     = f.load_f8n([jmsave1])
        self.rn     = f.load_f8n([jmsave1])
        self.rd     = f.load_f8n([jmsave1])
        self.un     = f.load_f8n([jmsave1])
        self.xln    = f.load_f8n([jmsave1])
        self.qln    = f.load_f8n([jmsave1])
        self.qld    = f.load_f8n([jmsave1])
        self.difi   = f.load_f8n([jmsave1])
        self.netnum = f.load_f8_kep_i4n([jmsave1])
        self.xm     = f.load_f8n([jmsave1])
        self.dn     = f.load_f8n([jmsave1])
        self.tn     = f.load_f8n([jmsave1])
        self.td     = f.load_f8n([jmsave1])
        self.en     = f.load_f8n([jmsave1])
        self.pn     = f.load_f8n([jmsave1])
        self.zn     = f.load_f8n([jmsave1])
        self.etan   = f.load_f8n([jmsave1])
        self.sn     = f.load_f8n([jmsave1])
        self.snn    = f.load_f8n([jmsave1])
        self.abar   = f.load_f8n([jmsave1])
        self.zbar   = f.load_f8n([jmsave1])
        self.xkn    = f.load_f8n([jmsave1])
        self.xnei   = f.load_f8n([jmsave1])
        self.stot   = f.load_f8n([jmsave1])
        self.angj = np.ndarray((jmsave1, self.nangjd))
        for i in range(nangjdz):
            self.angj[:, i] = f.load_f8n([jmsave1])
        self.angj[:, nangjdz:] = 0.
        self.angdg  = f.load_f8n([jmsave1])
        self.angd = np.ndarray((jmsave1, self.nangmd))
        for i in range(nangmdz):
            self.angd[:, i] = f.load_f8n([jmsave1])
        self.dsold  = f.load_f8n([jmsave1])
        self.tsold  = f.load_f8n([jmsave1])

        # ppn
        jm = int(self.qparm.jm)
        imax = int(self.qparm.imax)
        self.ppn = f.load_f8n([imax, jm])
        self.ppn = np.insert(self.ppn,[0,jm],0,axis=1)

        # magnetic field data
        #
        # TODO [KEPLER] - To allow future physics, these field should
        #                 also be written out [0...jmsave1]
        #
        # TODO [DUMP] - provide consitent empty arrays for interface
        magnet = self.parm.magnet
        if magnet > 0:
            if nvers >= 170000:
                f.load()
                self.bfvisc = f.get_f8n([jmsave1])
                self.bfdiff = f.get_f8n([jmsave1])
                f.assert_eor()
                f.load()
                self.bfbr = f.get_f8n([jmsave1])
                self.bfbt = f.get_f8n([jmsave1])
                f.assert_eor()
                # effective mixing coefficients
                f.load()
                self.bfviscef = f.get_f8n([jmsave1])
                self.bfdiffef = f.get_f8n([jmsave1])
                f.assert_eor()
            else:
                if nvers > 161150:
                    f.load()
                    self.bfvisc = f.get_f8n1d0n(jm)
                    self.bfdiff = f.get_f8n1d0n(jm)
                    f.assert_eor()
                if nvers > 162650:
                    f.load()
                    self.bfbr = f.get_f8n1d0n(jm)
                    self.bfbt = f.get_f8n1d0n(jm)
                    f.assert_eor()
                # effective mixing coefficients
                if nvers > 162150:
                    f.load()
                    self.bfviscef = f.get_f8n1d0n(jm)
                    self.bfdiffef = f.get_f8n1d0n(jm)
                    f.assert_eor()
        # effective viscosities needed for pre-cycle mixing
        if nvers >= 170000:
            f.load()
            self.angdgeff = f.get_f8n([jmsave1])
            self.difieff = f.get_f8n([jmsave1])
            f.assert_eor()
        elif nvers > 162150:
            f.load()
            self.angdgeff = f.get_f8n1d0n(jm)
            self.difieff = f.get_f8n1d0n(jm)
            f.assert_eor()

        # read flame data
        if nvers > 161350:
            sharp1 = self.parm.sharp1
            if sharp1 > 0:
                nflamez = 50
                # this should go into header file as well!
                f.load()
                self.xmburn = f.get_f8n([nflamez])
                self.fc12mult = f.get_f8n1d0n(jm)
                f.assert_eor()

        # load BURN (old location)
        if nvers < 169250:
            self.load_burn(killburn = killburn and (nvers < 167550))

        # WIMP energy deposition data
        wimp = self.parm.wimp
        if (nvers > 167550) and (wimp > 0):
            self.snw = f.load_f8n1d0n(jm)
            self.snwcrsi = f.load_f8n1d0n(jm)
            self.snwcrsd = f.load_f8n1d0n(jm)

        # advection energy deposition data
        if nvers > 168150:
            self.sadv = f.load_f8n1d0n(jm)

        # read in UUIDs and log data
        if nvers >= 170600:
            f.load()
            self.uuidrun, self.uuidcycle, self.uuiddump, self.uuidprev, self.uuidprog, self.uuidexec \
                = f.get_buf(6, length = UUID_NBYTES)
            self.gitsha = f.get_s(40)
            self.hostname = f.get_s(64, strip = True)
            self.username = f.get_s(16, strip = True)
            self.gitbranch = f.get_s(16, strip = True)
            self.nuuidhist = f.get_i8()
            self.uuidhist = f.get_buf(
                (13, self.nuuidhist),
                length = UUID_NBYTES,
                order = 'C',
                output = 'list',
                )
            f.assert_eor()
        elif nvers >= 170100:
            f.load()
            self.uuidrun, self.uuidcycle, self.uuiddump, self.uuidprev, self.uuidprog, self.uuidexec \
                = f.get_buf(6, length = UUID_NBYTES)
            self.gitsha = f.get_s(40)
            self.hostname = f.get_s(64, strip = True)
            self.username = f.get_s(16, strip = True)
            self.nuuidhist = f.get_i8()
            self.uuidhist = f.get_buf(
                (12, self.nuuidhist),
                length = UUID_NBYTES,
                order = 'C',
                output = 'list',
                )
            f.assert_eor()
        elif nvers >= 170004:
            f.load()
            self.uuidrun, self.uuidcycle, self.uuiddump, self.uuidprev, self.uuidprog, self.uuidexec \
                = f.get_buf(6, length = UUID_NBYTES)
            self.nuuidhist = f.get_i8()
            self.uuidhist = f.get_buf(
                (6, self.nuuidhist),
                length = UUID_NBYTES,
                order = 'C',
                output = 'list',
                )
            f.assert_eor()
        elif nvers > 168450:
            f.load()
            self.uuidrun, self.uuidcycle, self.uuiddump, self.uuidprev, self.uuidprog \
                = f.get_buf(5, length = UUID_NBYTES)
            f.assert_eor()

        # initialize old versions
        if nvers < 168450:
            self.uuidrun, self.uuidcycle, self.uuiddump, self.uuidprev, self.uuidprog \
                =  [ b'\0'*16 ] * 5
        if nvers < 170004:
            self.uuidexec = b'\0'*16
            self.nuuidhist = 0
            self.uuidhist = [[]]*6
        if nvers < 170100:
            self.gitsha = '0'*40
            self.hostname = ''
            self.username = ''
            # initialize array values 6-11
            if self.nuuidhist > 0:
                for x in [b' '*16] + [b'\0'*16]*2 + [b' '*16]*3:
                    self.uuidhist.append([x] * self.nuuidhist)
            else:
                self.uuidhist += [[]]*6
        if nvers < 170600:
            self.gitbranch = ''
            # initialize array value(s) 12
            if self.nuuidhist > 0:
                self.uuidhist.append([b' '*16] * self.nuuidhist)
            else:
                self.uuidhist += [[]]
        # read in log data
        if nvers > 168450:
            f.load()
            self.nlog = f.get_i4n()
            nlog = int(self.nlog)
            self.ilog = f.get_i4n([nlog])
            self.llog = f.get_i4n([nlog])
            llog = self.llog.tolist()
            self.clog = f.get_sln(llog)
            f.assert_eor()

        # viscous heating
        if nvers > 169750:
            self.sv = f.load_f8n1d0n(jm)

        # optional user-defined parameters
        if nvers >= 170002:
            f.load()
            self.noparm = f.get_i4()
            nameoprm = f.get_sn(self.noparm, 8)
            self.iotype = f.get_i4n(self.noparm)
            oparm = f.get_kep_parm64(self.iotype)
            self.oparm = IndexDict(list(str(name).strip() for name in nameoprm), oparm)
            f.assert_eor()

        # command lines
        if nvers >= 170100:
            f.load()
            self.ncmd = f.get_i4n()
            ncmd = int(self.ncmd)
            self.lcmd = f.get_i4n([ncmd])
            lcmd = self.lcmd.tolist()
            self.cmd = f.get_sln(lcmd)
            f.assert_eor()

        # load BURN - new location
        if nvers > 169250:
            self.load_burn(killburn = killburn)

            if not killburn:
                # check if we are all done
                f.assert_eof()
        else:
            # check if we are all done
            if not (killburn and (nvers < 167550)):
                f.assert_eof()

        self.jm = int(jm)

        # add wind and core to xm
        self.xm[0] = self.parm.summ0
        self.xm[jm+1] = self.qparm.xmlost

    def __getattr__(self, var):
        """
        This is to supply variables that may not have been loaded, but
        to save memory, we do not create them if not used.

        More variables to be added as needed.
        """
        if var in ('bfvisc', 'bfdiff', 'bfbr', 'bfbt', ):
            val = np.zeros(self.jm + 2, dtype = np.float64)
            setattr(self, var, val)
            return val
        raise AttributeError(var)

    def write(self, filename = None, silent = False):
        """
        Write out KEPLER restart dump.

        By default, save to original file.
        """
        self.setup_logger(level = logging.ERROR)

        # Writing old versions to new would require initialization etc.
        # starting with 17.01.00 second group is increased whenever format changes
        if self.nvers // 100 != ncurvers // 100:
            s = 'Saving of non-current model version is not supported.'
            self.logger.error(s)
            raise KeplerWriteError(s + '\nnvers    = {:6d}\nncurvers = {:6d}'.format(
                self.nvers, ncurvers))
            return

        # use loaded file name by default
        if filename is None:
            filename = self.filename

        # open file with compression, etc.
        self.file = FortranWriter(filename, byteorder = '>')
        self._write()
        self.file.close()
        self.close_logger(timing = 'Model written in ')

    def _write(self):
        """
        Write out KEPLER restart dump data
        """

        f = self.file

        # this would be used if there was a change of version allowed
        # self.head.nvers = self.nvers

        ihdum = np.zeros(self.nhead, dtype=np.int32)
        ihdum[0:len(self.head)] = np.array(self.head.data, dtype=np.int32)
        f.write_i4(ihdum)
        f.write_kep_parm(self.parm.data)

        f.put_kep_parm(self.qparm.data)
        nqparmz = self.head.nqparmz0
        f.skip_bytes(max(0,(nqparmz - len(self.qparm) + 1)*8))
        f.write()

        # sdum
        f.skip_bytes(8)
        f.put_f8(self.tpist)
        f.put_f8(self.rpist)
        f.put_f8(self.yemass)
        f.put_f8(self.yeq0)
        # ion arrays
        f.put_f8(self.aion)
        f.put_f8(self.zion)
        f.put_f8_kep_i4(self.numi)
        f.put_f8_kep_i4(self.ionn)
        # burn ion arrays
        f.put_f8(self.aionb)
        f.put_f8(self.zionb)
        f.put_f8_kep_i4(self.numib)
        f.put_f8_kep_i4(self.ionnb)
        # time-step controller arrays
        f.put_f8(self.dtc)
        f.put_f8_kep_i4(self.jdtc)
        # subroutine timing array
        f.put_f8(self.timeused)
        # #  reaction arrays
        f.put_f8(self.totalr)
        f.put_f8(self.rater)
        f.put_f8(self.qval)
        f.put_f8_kep_i4(self.jrate)
        f.put_f8(self.rrx)
        # accretion composition array
        f.put_f8(self.compsurf)
        # post-processor dump arrays (those should be removed)
        f.put_f8_kep_i4(self.locqz)
        f.put_f8_kep_i4(self.locqz0)
        f.put_f8(self.ratzdump)
        f.put_f8(self.ratiodez)
        f.put_f8(self.ratioadz)
        # user-specified edit arrays (are these still used?)
        f.put_f8_kep_i4(self.ndatzed)
        f.put_f8_kep_i4(self.ncyczed)
        f.put_f8(self.zedmass1)
        f.put_f8(self.zedmass2)
        # record of isotope mass lost in wind
        f.put_f8(self.wind)
        f.put_f8(self.windb)
        f.write()

        f.skip_bytes(8)
        # names, flags, and id-words (char*8) (except namec0 is char*16)
        f.put_s(self.namep0, 8)
        f.put_s(self.namec0, 16)
        f.put_s((self.iflag80,
                 self.iqbrnflg,
                 self.craybox,
                 self.idword), 8)
        # storage directory, last run and code mod dates (char*16)
        f.put_s((self.nxdirect,
                 self.lastrun,
                 self.lastmod0), 16)
        # arrays of symbols for isotopes, burn isotopes, time-step
        # controlers, and reactions (char*8)
        f.put_s(self.ions, 8)
        f.put_s(self.ionsb, 8)
        f.put_s(self.idtcsym, 8)
        f.put_s(self.isymr, 8)
        # post-processor file names (char*16)
        f.put_s((self.nameqq,
                 self.nameqlib,
                 self.nameolds,
                 self.namenews), 16)
        # arrays of post-processor dump variable names (char*8)
        # and labels (char*48)
        f.put_s(self.namedatq, 8)
        f.put_s(self.labldatq, 48)
        # array of edit variable names for user-specified edits (char*8)
        f.put_s(self.namedzed, 8)
        # array of aliases
        f.put_s(self.savdcmd0, 80)
        # remembered 'look' post-processing variable names (char*8) and
        # labels (char*48)
        f.put_s(self.namedatl, 8)
        f.put_s(self.labldatl, 48)
        # 'look'-read post-processor file names (char*16)
        f.put_s((self.nameqql0,
                  self.nameqql1,
                  self.nameqlbl), 16)
        # output storage directory (char*48)
        f.put_s(self.nsdirect, 48)
        # set of isotopes to be plotted (char*8) and their
        # plot icons (char*16)
        f.put_s(self.isosym, 8)
        f.put_s(self.isoicon, 16)
        # array of remembered tty command strings (char*80)
        f.put_s(self.savedcmd, 80)
        # path for location of data file (char*80)
        f.put_s(self.datapath, 80)
        # array of zonal convection sentinels (char*8)
        f.put_s(self.icon, 8)
        f.write()

        # structure
        f.write_f8(self.ym)
        f.write_f8(self.rn)
        f.write_f8(self.rd)
        f.write_f8(self.un)
        f.write_f8(self.xln)
        f.write_f8(self.qln)
        f.write_f8(self.qld)
        f.write_f8(self.difi)
        f.write_f8_kep_i4(self.netnum)
        jm = self.jm
        self.xm[0] = self.xm[jm+1] = 0
        f.write_f8(self.xm)
        f.write_f8(self.dn)
        f.write_f8(self.tn)
        f.write_f8(self.td)
        f.write_f8(self.en)
        f.write_f8(self.pn)
        f.write_f8(self.zn)
        f.write_f8(self.etan)
        f.write_f8(self.sn)
        f.write_f8(self.snn)
        f.write_f8(self.abar)
        f.write_f8(self.zbar)
        f.write_f8(self.xkn)
        f.write_f8(self.xnei)
        f.write_f8(self.stot)
        if self.nvers >= 171000:
            for i in range(self.nangjd):
                f.write_f8(self.angj[:, i])
        else:
            f.write_f8(self.angj[:, 0])
        f.write_f8(self.angdg)
        for i in range(5):
            f.write_f8(self.angd[:, i])
        f.write_f8(self.dsold)
        f.write_f8(self.tsold)

        f.write_f8(self.ppn[:,1:-1])

        # magnetic field data - NEED TEST
        magnet = self.parm.magnet
        if magnet > 0:
            f.put_f8(self.bfvisc)
            f.put_f8(self.bfdiff)
            f.write()
            f.put_f8(self.bfbr)
            f.put_f8(self.bfbt)
            f.write()
            # effective mixing coefficients
            f.put_f8(self.bfviscef)
            f.put_f8(self.bfdiffef)
            f.write()

        # effective viscosities needed for pre-cycle mixing
        f.put_f8(self.angdgeff)
        f.put_f8(self.difieff)
        f.write()

        # flame data
        sharp1 = self.parm.sharp1
        if sharp1 > 0:
            f.put_f8(self.xmburn)
            f.put_f8_1d_0n(self.fc12mult)
            f.write()

        # WIMP energy deposition data
        wimp = self.parm.wimp
        if (wimp > 0):
            f.write_f8_1d_0n(self.snw)
            f.write_f8_1d_0n(self.snwcrsi)
            f.write_f8_1d_0n(self.snwcrsd)

        # advection energy deposition data
        f.write_f8_1d_0n(self.sadv)

        # UUIDs
        f.put_buf((self.uuidrun,
                   self.uuidcycle,
                   self.uuiddump,
                   self.uuidprev,
                   self.uuidprog,
                   self.uuidexec))
        f.put_s(self.gitsha, 40)
        f.put_s(self.hostname, 64, fill = b' ')
        f.put_s(self.username, 16, fill = b' ')
        f.put_s(self.gitbranch, 16, fill = b' ')
        f.put_i8(self.nuuidhist)
        f.put_buf(self.uuidhist, order = 'C')
        f.write()

        # log data
        f.put_i4(self.nlog)
        f.put_i4(self.ilog)
        # should check llog is correct (unicode, ...)
        f.put_i4(self.llog)
        f.put_s(self.clog, self.llog)
        f.write()

        f.write_f8_1d_0n(self.sv)

        # user-defined parameters
        f.put_i4(self.noparm)
        f.put_s(self.oparm.list, 8, fill = b' ')
        f.put_i4(self.iotype)
        f.put_kep_parm64(self.oparm.data)
        f.write()

        # command file
        f.put_i4(self.ncmd)
        f.put_i4(self.lcmd)
        f.put_s(self.cmd, self.lcmd)
        f.write()

        self.write_burn()

    def write_burn(self):
        """Write BURN infor to file."""
        imaxb = self.qparm.imaxb
        nsaveb = self.parm.nsaveb
        jmzb0 = self.head.jmzb0
        if (imaxb == 0) or (jmzb0 == 1) or (nsaveb == 0):
            return
        f = self.file
        f.write_f8_kep_i4(self.netnumb)
        f.write_f8_kep_i4(self.limab)
        f.write_f8_kep_i4(self.limzb)
        f.write_f8_kep_i4(self.limcb)
        f.write_f8(self.timen)
        f.write_f8(self.dtimen)
        f.write_f8(self.dnold)
        f.write_f8(self.tnold)
        f.write_f8(self.ymb)
        f.write_f8(self.sburn)
        # it seems the following have dropped because by default
        # nsaveb parameter is set to 10.  Should be 12
        if nsaveb == 12:
            f.write_f8(self.etab)
            f.write_f8(self.pbuf)
            # mabe older versions of KEPLER do this differently.

        # ppnb - BURN abundances
        f.write_f8(self.ppnb[:,1:-1])

        irecb = self.parm.irecb
        if irecb == 1:
            f.write_i4(self.nbmax)
            f.write_i4(self.nabmax)
            f.write_i4(self.nzbmax)
            f.write_i4(self.nibmax)
            f.write_s(self.ionbmax, 8)
            f.write_f8(self.burnamax)
            f.write_f8(self.burnmmax)
            f.write_i4(self.ibcmax)

        # accreation composition
        f.write_f8(self.compsurfb)


    def show_log(self, events = None, head = True, hist=True):
        """
        show the log and file info
        """
        if events is None:
            events = logdata.log_name
        divider = '-' * 50
        print(divider)
        if hist:
            for i in range(self.nuuidhist):
                print('UUID DUMP  : {!s}'.format(uuid.UUID(bytes=self.uuidhist[0][i] )))
                print('UUID CYCLE : {!s}'.format(uuid.UUID(bytes=self.uuidhist[1][i] )))
                print('UUID PROG  : {!s}'.format(uuid.UUID(bytes=self.uuidhist[2][i] )))
                print('UUID EXEC  : {!s}'.format(uuid.UUID(bytes=self.uuidhist[3][i] )))
                print('CYCLE      : {:>40n}'.format(int(self.uuidhist[4][i])))
                print('FILE       : {:>40s}'.format(self.uuidhist[5][i].decode('ASCII').strip()))
                print('VERSION    : {:>40s}'.format(self.uuidhist[6][i].decode('ASCII').strip()))
                print('SHA-1      : {:>40s}'.format((self.uuidhist[7][i][0:10]+self.uuidhist[8][i][0:10]).hex()))
                print('USER       : {:>40s}'.format(self.uuidhist[9][i].decode('ASCII').strip()))
                print('HOST       : {:>40s}'.format((self.uuidhist[10][i]+self.uuidhist[11][i]).decode('ASCII').strip()))
                print('BRANCH     : {:>40s}'.format(self.uuidhist[12][i].decode('ASCII').strip()))
                print(divider)
        if head:
            print('CYCLE      : {:>40n}'.format(int(self.qparm.ncyc)))
            print('UUID DUMP  : {!s:>40s}'.format(uuid.UUID(bytes=self.uuiddump )))
            print('UUID CYCLE : {!s:>40s}'.format(uuid.UUID(bytes=self.uuidcycle)))
            print('UUID PREV  : {!s:>40s}'.format(uuid.UUID(bytes=self.uuidprev )))
            print('UUID RUN   : {!s:>40s}'.format(uuid.UUID(bytes=self.uuidrun  )))
            print('UUID PROG  : {!s:>40s}'.format(uuid.UUID(bytes=self.uuidprog )))
            print('UUID EXEC  : {!s:>40s}'.format(uuid.UUID(bytes=self.uuidexec )))
            print('USER       : {:>40s}'.format(self.username ))
            print('HOST       : {:>40s}'.format(self.hostname ))
            print('GIT SHA-1  : {:>40s}'.format(self.gitsha   ))
            print('GIT BRANCH : {:>40s}'.format(self.gitbranch))
            print(divider)
        if self.nlog > 0:
            for i,c in zip(self.ilog,self.clog):
                if logdata.log_name[i] in events:
                    print(r'{:<6s} {:s}'.format('['+logdata.log_name[i]+']',c))
            print(divider)

    @CachedAttribute
    def nameprob(self):
        """
        Return problem name.
        """
        return self.namep0.strip()

    @CachedAttribute
    def mass_raw(self):
        """
        Return non-rounded initial mass. (Msun)
        """
        return self.parm.totm0 / physconst.Kepler.solmass

    @CachedAttribute
    def mass_string(self):
        """
        Return string for initial mass (Msun).
        """
        return mass_string(self.mass_raw)

    @CachedAttribute
    def mass(self):
        """
        Return rounded initial mass (Msun).
        """
        return float(self.mass_string)

    @CachedAttribute
    def radius(self):
        """
        Photosphere radius (Rsun)
        """
        return self.qparm.radius / physconst.Kepler.solrad

    @CachedAttribute
    def isob(self):
        """
        Return array of BURN ions.
        """
        if self.qparm.imaxb > 0:
            return np.array([ion(s) for s in self.ionsb])
        else:
            return None

    @CachedAttribute
    def IonList(self):
        """
        Return IonList of BURN ions.
        """
        if self.qparm.imaxb > 0:
            return IonList(self.ionsb)
        else:
            return None

    @CachedAttribute
    def ni56(self):
        """
        Return APPROX Ni56 mass fraction
        """
        return self.KepAbuDump(ion('Ni56'))

    @CachedAttribute
    def y(self):
        """
        column depth (g/cm**2).
        """
        y = np.empty_like(self.rn)
        y[:-1] = np.cumsum(self.dy[:0:-1])[::-1]
        y[ -1] = 0
        return y

    @CachedAttribute
    def y_m(self):
        """
        zone centered column depth (g/cm**2).
        """
        return self.face2center(self.y)

    @CachedAttribute
    def dy(self):
        """
        zone column thickness (g/cm**2).
        """
        r = self.rn
        r2 = np.empty_like(self.rn)
        r2[0] = r[0]**2
        r2[1:-1] = r[1:-1] * r[0:-2]
        r2[-1] = r[-2]**2
        dm = self.xm.copy()
        dm[-1] = self.parm.xmacrete
        dy = dm / (r2 * 4 * np.pi)
        return dy

    @CachedAttribute
    def xm_sun(self):
        """
        Zone mass (Msun).
        """
        return self.xm / physconst.Kepler.solmass

    @CachedAttribute
    def ym_sun(self):
        """
        Exterior mass coordinate (Msun).
        """
        return self.ym / physconst.Kepler.solmass

    @CachedAttribute
    def zm(self):
        """
        Interior mass coordinate (g).
        """
        return np.cumsum(self.xm)

    @CachedAttribute
    def zm_(self):
        """
        Interior mass coordinate on grid (g).
        """
        x = np.empty_like(self.xm)
        x[1:] = np.cumsum(self.xm[1:])
        x[0] = 0.
        return x

    @CachedAttribute
    def zm_sun(self):
        """
        Interior mass coordinate (Msun).
        """
        return self.zm / physconst.Kepler.solmass

    @CachedAttribute
    def zmm(self):
        """
        Interior mass coordinate in middle of zone (g).
        """
        return self.face2center(self.zm)

    @CachedAttribute
    def zmm_(self):
        """
        Interior mass coordinate on grid in middle of zone (g).
        """
        return self.face2center(self.zm_)

    @CachedAttribute
    def rnm(self):
        """
        Radius coordinate in middle of zone (cm).
        """
        return self.face2center(self.rn)

    @CachedAttribute
    def rm(self):
        """
        Radius of middle(mass) of zone assuming const density (cm).
        """
        rm = np.empty_like(self.rn)
        rm[1:] = 0.5 * (self.rn[1:]**3 + self.rn[:-1]**3)
        rm = np.power(rm, 1/3)
        rm[0] = self.rn[0]
        rm[-1] = self.rn[-2]
        return rm

    @CachedAttribute
    def rn_sun(self):
        """
        Radius coordinate (Rsun).
        """
        return self.rn / physconst.Kepler.solrad

    @CachedAttribute
    def rm_sun(self):
        """
        Radius of middle(mass) of zone assuming const density (Rsun).
        """
        return self.rm / physconst.Kepler.solrad

    @CachedAttribute
    def rnm_sun(self):
        """
        Radius coordinate in middle of zone (Rsun).
        """
        return self.rnm / physconst.Kepler.solrad

    @CachedAttribute
    def dnf(self):
        """
        Density at zone interface (flat extrapolation for boundaries) (g/cm**3).
        """
        # it may be desirable to get density from the conservative
        # interpolation function and then use that for the formula.
        return self.center2face(self.dn)

    @CachedAttribute
    def pnf(self):
        """
        Pressure at zone interface (flat extrapolation for boundaries) (dyn/cm**2).
        """
        # it may be desirable to use hydrostatic considerations
        return self.center2face(self.pn)

    @CachedAttribute
    def zmm_sun(self):
        """
        Interior mass coordinate in middle of zone (Msun).
        """
        return self.zmm / physconst.Kepler.solmass

    @CachedAttribute
    def uesc(self):
        """
        Local escape velocity of shell center (cm/sec).

        This takes only into account gravity from zones inside.
        """
        uesc = np.sqrt(2 * self.gee * self.zmm / (self.rm + 1.e-99))
        if self.rm[0] == 0:
            uesc[0] = 0.
        uesc[-1] = 0.
        return uesc

    @CachedAttribute
    def uescf(self):
        """
        Local escape velocity of shell face (cm/sec).

        This takes only into account gravity from zones inside.
        """
        uesc = np.sqrt(2 * self.gee * self.zm / (self.rn + 1.e-99))
        if self.rn[0] == 0:
            uesc[0] = 0.
        uesc[-1] = 0.
        return uesc

    @CachedAttribute
    def hpm(self):
        """
        Pressure scale height at zone center (cm).

        Returns np.inf for invalid zones.
        """
        hp = np.empty_like(self.pn)
        ii = self.center_slice
        hp[ii] = - self.pn[ii] / (self.dn[ii] * (self.gravm[ii] + 1.e-99))
        if self.grav[0] == 0.:
            hp[0] = np.inf
        else:
            hp[0] = hp[1]
        hp[-1] = np.inf
        return hp

    @CachedAttribute
    def hp(self):
        """
        Pressure scale height at zone interface (cm).

        Returns np.inf for invalid zones.
        """
        hp = np.empty_like(self.pn)
        ii = self.boundary_slice
        hp[ii] = - self.pnf[ii] / (self.dnf[ii] * (self.grav[ii] + 1.e-99))
        if self.grav[0] == 0:
            hp[0] = np.inf
        hp[-1] = np.inf
        return hp

    @CachedAttribute
    def gamma(self):
        """
        Polytropic gamma, d ln P / d ln rho, on zone boundary.

        Returns np.nan for invalid zones.
        """
        gamma = np.empty_like(self.pn)
        ih = slice(2, self.jm + 1)
        il = slice(1, self.jm)
        gamma[il] = (self.pn[ih] - self.pn[il]) * (self.dn[ih] + self.dn[il])/(
            (self.pn[ih] + self.pn[il]) * (self.dn[ih] - self.dn[il]))
        gamma[1] = gamma[2]
        gamma[self.jm] = gamma[self.jm - 1]
        gamma[0] = gamma[self.jm + 1] = np.nan
        return gamma

    @CachedAttribute
    def difim(self):
        """
        Zone centered diffusion coefficient (cm**2/sec)
        """
        v = np.zeros_like(self.difi)
        v[self.center_slice] = 0.5 * (
            self.difi[self.lower_slice] + self.difi[self.upper_slice])
        return v

    @CachedAttribute
    def vconvm(self):
        """
        Convective velocity at zone center (cm/sec).
        """
        v = np.zeros_like(self.dn)
        ii = self.center_slice
        v[ii] = 3 * self.difim[ii] / (self.parm.xmlen * self.hpm[ii])
        return v

    @CachedAttribute
    def vconv(self):
        """
        Convective velocity at zone interface (cm/sec).
        """
        v = np.zeros_like(self.dn)
        ii = self.boundary_slice
        v[ii] = 3 * self.difi[ii] / (self.parm.xmlen * self.hp[ii])
        return v

    @CachedAttribute
    def tconvm(self):
        """
        Convective time scale at zone center (sec).
        """
        t = np.empty_like(self.dn)
        ii = self.difim > 0
        t[ii] =  (self.parm.xmlen * self.hpm[ii])**2 / (3 * self.difim[ii])
        t[np.logical_not(ii)] = np.inf
        return t

    @CachedAttribute
    def tconv(self):
        """
        Convective time scale at zone face (sec).
        """
        t = np.empty_like(self.dn)
        ii = self.difi > 0
        t[ii] =  (self.parm.xmlen * self.hp[ii])**2 / (3 * self.difi[ii])
        t[np.logical_not(ii)] = np.inf
        return t

    @CachedAttribute
    def erot(self):
        """
        Specific rotational energy (cm**2/sec**2).
        """
        # return 0.5 * self.angjn * self.angwn
        # return np.sum(d.angw * d.angj, axis=1)
        return 0.5 * np.einsum('ij,ij->i', self.angj, self.angw)

    @CachedAttribute
    def gee(self):
        """
        KEPLER effecive gravitational constant (cm**2/sec**2/g)
        """
        return physconst.Kepler.gee * self.parm.geemult

    @CachedAttribute
    def rs(self):
        """
        Schwarzschild radius at zone interface (cm).
        """
        return 2 * self.zm * self.gee / physconst.Kepler.c**2

    @CachedAttribute
    def rsm(self):
        """
        Schwarzschild radius at zone center (cm).
        """
        return 2 * self.zmm * self.gee / physconst.Kepler.c**2

    @CachedAttribute
    def gravm(self):
        """
        Gravitational acceleration at zone center (cm/sec**2).
        """
        g = -self.gee * self.zmm / (self.rm + 1.e-99)**2
        if self.parm.relmult != 0:
            g /= 1 - self.rsm / (self.rm + 1e-99)
        return g

    @CachedAttribute
    def grav(self):
        """
        Gravitational acceleration at zone interface (cm/sec**2).
        """
        g = -self.gee * self.zm / (self.rn + 1.e-99)**2
        if self.parm.relmult != 0:
            g /= 1 - self.rs / (self.rn + 1.e-99)
        return g

    @CachedAttribute
    def egrav(self):
        """
        Gravitational potential of enclosed mass (cm**2/sec**2).
        """
        return - self.gee * self.zmm / (self.rm + 1.e-99)

    @CachedAttribute
    def ekin(self):
        """
        Specific kinetic energy (cm**2/sec**2).
        """
        ekin = np.empty_like(self.un)
        ekin[1:] = ((self.un[1:] + self.un[:-1]) * self.un[1:]
                    + self.un[:-1]**2)
        ekin[0] = self.un[0]**2
        return ekin / 6

    @CachedAttribute
    def xbind(self):
        """
        Zonal binding energy (egs).
        """
        x = self.xm * (
            self.en +
            self.erot +
            self.egrav +
            self.ekin)
        x[[0,-1]] = np.nan
        return x

    @CachedAttribute
    def ybind(self):
        """
        Integrated external binding energy (erg).  Definition from KEPLER.
        """
        x = np.zeros_like(self.xbind)
        x[1:-1] = self.xbind[-2:0:-1].cumsum()[::-1]
        return x

    @CachedAttribute
    def zbind(self):
        """
        Integrated internal binding energy (erg).  Definition from KEPLER.
        """
        x = np.zeros_like(self.xbind)
        x[1:-1] = self.xbind.cumsum()
        return x

    @CachedAttribute
    def angvk(self):
        """
        shell Keplerian ('orbital') velocity (cm/sec)
        """
        g = physconst.Kepler.gee * self.parm.geemult
        vk = np.sqrt(g * self.zm / (self.rm + 1.e-99))
        if self.rm[0] == 0:
            vk[0] = 0
        return vk

    @CachedAttribute
    def angwk(self):
        """
        shell keplerian angular velocity (rad/sec)
        """
        g = physconst.Kepler.gee * self.parm.geemult
        wk = np.sqrt(g * self.zm / (self.rm**3 + 1.e-99))
        if self.rm[0] == 0:
            wk[0] = np.sqrt(4 / 3 * np.pi * g * self.dn[1])
        return wk

    @CachedAttribute
    def angwwk(self):
        """
        shell angular velocity vector relative to keplerian
        """
        wwk = self.angw / (self.angwk[:, np.newaxis] + 1.e-99)
        return wwk

    @CachedAttribute
    def angwwkn(self):
        """
        shell angular velocity magnitude relative to keplerian
        """
        return LA.norm(self.angwwk, axis=1)

    angvvk = angwwk
    angvvkn = angwwkn

    @CachedAttribute
    def angi(self):
        """
        shell specific moment of inertia (cm**2)
        """
        ai = np.empty_like(self.rn)
        ai[1:-1] = calcai(self.rn[:-2], self.rn[1:-1])
        ai[0] = ai[-1] = np.nan
        return ai

    @CachedAttribute
    def angw(self):
        """
        shell angular velocity vector (rad/sec)
        """
        s = slice(1,-1)
        aw = np.empty_like(self.angj)
        aw[s] = self.angj[s] / self.angi[s, np.newaxis]
        if 'angw0' in self.parm:
            aw[0, 0 ] = self.parm.angw0
            aw[0, 1:] = 0.
        else:
            aw[0] = np.nan
        aw[-1] = np.nan
        return aw

    @CachedAttribute
    def angwn(self):
        """
        shell angular velocity amplitude (rad/sec)
        """
        return LA.norm(self.angw, axis=1)

    @CachedAttribute
    def angwx(self):
        """
        shell angular velocity amplitude (rad/sec)
        """
        return self.angw[:, 0]

    @CachedAttribute
    def angwy(self):
        """
        shell angular velocity amplitude (rad/sec)
        """
        return self.angw[:, 1]

    @CachedAttribute
    def angwz(self):
        """
        shell angular velocity amplitude (rad/sec)
        """
        return self.angw[:, 2]

    @CachedAttribute
    def angjn(self):
        """
        shell specific angular momentum amplitude (erg*sec/g)
        """
        return LA.norm(self.angj, axis=1)

    @CachedAttribute
    def angjx(self):
        """
        shell specific angular momentum amplitude (erg*sec/g)
        """
        return self.angj[:, 0]

    @CachedAttribute
    def angjy(self):
        """
        shell specific angular momentum amplitude (erg*sec/g)
        """
        return self.angj[:, 1]

    @CachedAttribute
    def angjz(self):
        """
        shell specific angular momentum amplitude (erg*sec/g)
        """
        return self.angj[:, 2]

    @CachedAttribute
    def snu(self):
        """
        specific neutrino loss rate (erg/g/sec)
        """
        return self.sn - self.snn

    @CachedAttribute
    def xlnu(self):
        """
        neutrino luminosity (erg/sec)
        """
        return np.cumsum(self.snu)

    @CachedAttribute
    def fnu(self):
        """
        neutrino flux (erg/sec/cm**2)
        """
        fnu = np.zeros_like(self.snu)
        if self.rn[0] == 0:
            i = 1
        else:
            i = 0
        fnu[i:] = self.snu[i:] / (np.pi * 4 * self.rn[i:]**2)
        return fnu

    @CachedAttribute
    def angwcst(self):
        """
        rotation constant vercor (cm/s*g**(-2/3))
        """
        s = slice(1,-1)
        awt = self.angjt[s] / self.angit[s, np.newaxis]
        dnt = self.zm[s] * 3 / (4 * np.pi * self.rn[s]**3)
        awx = np.zeros_like(self.angj)
        awx[s] = awt * dnt[:, np.newaxis]**(-2 / 3)
        return awx

    @CachedAttribute
    def angwcstn(self):
        """
        rotation constant (cm/s*g**(-2/3))
        """
        return LA.norm(self.angwcst, axis=1)

    @CachedAttribute
    def angit(self):
        """
        total I (g*cm**2)
        """
        s = slice(1,-1)
        angI = np.zeros_like(self.xm)
        angI[s] = self.angi[s] * self.xm[s]
        angI[0] = (self.parm.summ0 *
                   calcai(0., self.rn[0]))
        angI = angI.cumsum()
        angI[-1] = np.nan
        return angI

    @CachedAttribute
    def angit_sun(self):
        """
        total I (Msun*Rsun**2)
        """
        return self.angit /(physconst.Kepler.solmass * physconst.Kepler.solrad**2)

    @CachedAttribute
    def angjt0(self):
        """
        total angular momentum of core (erg*sec)
        """
        angw0 = np.zeros((3,))
        if 'angw0' in self.parm:
            angw0[0] = self.parm.angw0
        else:
            angw0[0] = 0
        return angw0 * self.parm.summ0 * calcai(0, self.rn[0])

    @CachedAttribute
    def angjt0n(self):
        """
        total angular momentum magnitude of core (erg*sec)
        """
        return LA.norm(self.angjt0, axis=1)

    @CachedAttribute
    def angjbm(self):
        """
        average specific angular momentum vector (erg*sec/g)

        zone center value
        """
        s = slice(0, None)
        j = self.angjtm[s] / self.zmm[s, np.newaxis]
        return j

    @CachedAttribute
    def angjbmn(self):
        """
        average specific angular momentum magnitude (erg*sec/g)

        zone center value
        """
        return LA.norm(self.angjbm, axis=1)

    @CachedAttribute
    def angjt(self):
        """
        interior total angular momentum vector (erg*sec)
        """
        s = slice(1, -1)
        angJ = np.zeros_like(self.angj)
        angJ[ 0] = self.angjt0
        angJ[ s] = self.angj[s] * self.xm[s, np.newaxis]
        angJ = angJ.cumsum(axis=1)
        angJ[-1] = np.nan
        return angJ

    @CachedAttribute
    def angjtn(self):
        """
        interior total angular momentum magnitude (erg*sec)
        """
        return LA.norm (self.angjt, axis=1)

    @CachedAttribute
    def angjtm(self):
        """
        total J vector at zone center (erg*sec)

        """
        jtm = np.empty_like(self.angjt)
        jtm[[0,-1]] = self.angjt[[0,-1]]
        jtm[ 1:-1 ] = 0.5 * (self.angjt[0:-2] + self.angjt[1:-1])
        return jtm

    def angjtmn(self):
        """
        total J magnitude at zone center (erg*sec)

        """
        return LA.norm(self.angjtm, axis=1)

    @CachedAttribute
    def anglstn(self):
        """
        j(J_below) of last stable orbit (cm**2/sec)

        This would need to be refined in non-trivial way if there is
        no alignment of the angular momentum.
        """
        g = physconst.Kepler.gee * self.parm.geemult
        c = physconst.Kepler.c
        eps = 1.e-7
        ii = slice(1,-1)
        a = self.angabhn[ii]
        m = self.zmm[ii]
        x, = np.where(a >= (1 - eps) * m)
        a[x] = m[x] * (1 - eps)
        x13 = 1 / 3
        z1 = (1 + (1 - a**2 / m**2)**x13 *
              ((1 + a/m)**x13 + (1 - a / m)**x13))
        z2 = np.sqrt(3 * a**2 / m**2 + z1**2)
        r = m * (3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2)))
        l = (np.sqrt(m * r) * (r**2 - 2 * a * np.sqrt(m * r) + a**2) /
             (r * np.sqrt(r**2 - 3 * m * r + 2 * a * np.sqrt(m * r))))
        v = l * g / c
        v[x] = self.anglstk[x]
        j = np.empty_like(self.zmm)
        j[:] = np.nan
        j[ii] = v
        return j

    @CachedAttribute
    def phi(self):
        """
        Newtonian gravitational potential at shell boundary (cm**2/sec**2)
        """
        g = physconst.Kepler.gee * self.parm.geemult
        p = np.empty_like(self.rn)
        p[-1] = 0
        p[-2] = - g * self.zm[-2] / self.rn[-2]
        x = - 0.5 * g * (
            (self.zm[1:-1] / self.rn[1:-1]**2 +
             self.zm[0:-2] / (self.rn[0:-2]**2 + 1.e-99)) *
            (self.rn[1:-1] - self.rn[0:-2]))
        p[0:-2] = p[-2] + np.cumsum(x[::-1])[::-1]
        return p

    @CachedAttribute
    def phim(self):
        """
        Newtonian gravitational potential at shell center (cm**2/sec**2)
        """
        x = self.phi
        p = np.empty_like(x)
        p[0] = x[0]
        p[-1] = x[-1]
        p[1:-1] = 0.5 * (x[0:-2] + x[1:-1])
        return p

    @CachedAttribute
    def anglsts(self):
        """
        j of last stable orbit of Schwarzschild BH (cm**2/sec)

        Zone centered value
        """
        g = physconst.Kepler.gee * self.parm.geemult
        return (np.sqrt(12) * g / physconst.Kepler.c * self.zmm)

    @CachedAttribute
    def anglstk(self):
        """
        j of last stable orbit of Kerr BH (cm**2/sec)

        Zone centered value
        """
        g = physconst.Kepler.gee * self.parm.geemult
        return (np.sqrt(4 / 3) * g / physconst.Kepler.c * self.zmm)

    @CachedAttribute
    def angabh(self):
        """
        black hole spin parameter vector (g)
        """
        g = physconst.Kepler.gee * self.parm.geemult
        a = np.zeros_like(self.angj)
        s = self.center_slice
        a[s] = ( self.angjtm[s] / self.zmm[s, np.newaxis] *
                 physconst.Kepler.c / g )
        return a

    @CachedAttribute
    def angabhn(self):
        """
        black hole spin parameter magnitude (g)
        """
        return LA.norm(self.angabh, axis=1)

    @CachedAttribute
    def angambh(self):
        """
        normalized black hole spin parameter verctor a/m
        """
        am = np.zeros_like(self.xm)
        s = self.center_slice
        am[s] = self.angabh[s] / self.zmm[s]
        return am

    @CachedAttribute
    def angambhn(self):
        """
        normalized black hole spin parameter norm a/m
        """
        return LA.norm(self.angambh, axis=1)

    @CachedAttribute
    def xmwind(self):
        """
        mass in wind (g)
        """
        return self.qparm.xmlost

    @CachedAttribute
    def KepAbuDump(self):
        """
        KepAbuDump instance from ppn

        Assume wind is APPROX
        """
        netnum = self.netnum
        netnum[-1] = 1
        netnum[0] = 0
        wind = self.wind
        if self.xmwind != 0:
            wind /= self.xmwind
        return KepAbuDump(
            self.ppn,
            netnum,
            self.ionn,
            wind = wind,
            molfrac = True,
            xm = self.xm,
            zm = self.zm,
            )

    @CachedAttribute
    def jburnmin(self):
        """
        minimum valid BURN zone (above bmasslow)
        """
        if self.parm.bmasslow <= 0:
            return 1
        j = np.argwhere(self.zm > self.parm.bmasslow)
        return j.flatten()[0]

    @CachedAttribute
    def Ionsb(self):
        """
        return numpy array of BURN ions
        """
        return np.array([ion(i) for i in self.ionsb])

    @CachedAttribute
    def IonList(self):
        """
        return BURN IonList
        """
        return IonList(self.ionsb)

    @CachedAttribute
    def AbuDump(self):
        """
        AbuDump instance from ppnb

        Maybe this should be modified to pnly retunr abundances above
        bmasslow?
        """
        windb = self.windb
        if self.xmwind != 0:
            windb /= self.xmwind
        return AbuDump(
            self.ppnb,
            self.IonList,
            windb = windb,
            xm = self.xm,
            zm = self.zm,
            molfrac = True,
            )

    # aliases
    netb = AbuDump
    net = KepAbuDump
    abu = net
    abub = netb

    @CachedAttribute
    def p_F(self):
        """
        p_F - Fermi momentum of electrons (g*cm/s)
        """
        return physconst.Kepler.h * (self.ye * self.dn * physconst.Kepler.n0 * 3 / (8 * np.pi))**(1/3)

    @CachedAttribute
    def p_F_mec(self):
        """
        p_F - Fermi momentum of electrons electrons (m_e*c)
        """
        return self.p_F / (physconst.Kepler.me * physconst.Kepler.c)

    @CachedAttribute
    def E_F_mec2(self):
        """
        Fermi energy of electrons in (m_e*c**2)
        """
        x = self.p_F_mec
        ii = x < 1.e-7
        ef = np.empty_like(x)
        ef[ii] = x[ii]**2 * 0.5
        jj = x > 1.e7
        ef[jj] = x[jj]
        kk = ~(ii | jj)
        ef[kk] = (np.sqrt(x[kk]**2 + 1) - 1)
        return ef

    @CachedAttribute
    def E_F(self):
        """
        E_F - Fermi energy of electrons (erg)
        """
        c = physconst.Kepler.c
        me = physconst.Kepler.me
        mec2 = c**2 * me
        return self.E_F_mec2 * mec2

    @CachedAttribute
    def E_F_MeV(self):
        """
        Fermi energy of electrons (MeV)
        """
        return self.E_F / physconst.MEV

    @CachedAttribute
    def T_F(self):
        """
        T_F - Fermi temperature of electrons (K)
        """
        return self.E_F / physconst.Kepler.k

    @CachedAttribute
    def P_e_deg(self):
        """
        P_(e,deg) - degenerate electron Fermi pressure (T=0) (erg/ccm)
        """
        x = self.p_F_mec
        const = np.pi * physconst.Kepler.me**4 * physconst.Kepler.c**5 / (3 * physconst.Kepler.h**3)
        x2 = x**2
        f = x * (2 * x2 - 3) * np.sqrt(x2 + 1) + 3 * np.arcsinh(x)
        return f * const

    @CachedAttribute
    def U_e_deg(self):
        """
        U_(e,deg) - degenerate electron energy density (T=0) (erg/ccm)
        """
        x = self.p_F_mec
        const = np.pi * physconst.Kepler.me**4 * physconst.Kepler.c**5 / (3 * physconst.Kepler.h**3)
        x2 = x**2
        x3 = x * x2
        g = (6 * x3 + 3 * x) * np.sqrt(x2 + 1) - 8 * x3 - 3 * np.arcsinh(x)
        return g * const

    @CachedAttribute
    def u_e_deg(self):
        """
        u_(e,deg) - degenerate electron specific energy (T=0) (erg/g)
        """
        return self.U_e_deg / self.dn

    @CachedAttribute
    def P_rad(self):
        """
        P_rad - Radiation pressure (erg/ccm)
        """
        return self.U_rad / 3

    @CachedAttribute
    def U_rad(self):
        """
        U_rad - Radiation energy density (erg/ccm)
        """
        return self.tn**4 * physconst.Kepler.a

    @CachedAttribute
    def u_rad(self):
        """
        u_rad - Radiation specific energy (erg/g)
        """
        return self.tn**4 * physconst.Kepler.a

    @CachedAttribute
    def P_ions(self):
        """
        P_ions - Ion pressure (erg/ccm)
        """
        return physconst.Kepler.rk * self.tn * self.dn / self.abar

    @CachedAttribute
    def U_ions(self):
        """
        U_ions - Ion energy density (erg/ccm)
        """
        return 1.5 * self.P_ions

    @CachedAttribute
    def u_ions(self):
        """
        U_ions - Ion specific energy (erg/g)
        """
        return 1.5 * physconst.Kepler.rk * self.tn / self.abar


    @CachedAttribute
    def mui(self):
        """
        mean molecular weight per ion
        """
        return self.abar

    @CachedAttribute
    def mue(self):
        """
        mean molecular weight per electron
        """
        return self.zbar

    @CachedAttribute
    def mu(self):
        """
        mean molecular weight
        """
        x = np.empty_like(self.abar)
        ii = slice(1, None)
        x[ii] = 1/(1 / self.abar[ii] + 1 / self.zbar[ii])
        x[0] = np.nan
        return x

    @CachedAttribute
    def P_e_nd(self):
        """
        P_ions - non-degenerate electron pressure (erg/ccm)
        """
        return physconst.Kepler.rk * self.tn * self.dn * self.ye

    @CachedAttribute
    def U_e_nd(self):
        """
        U_e_nd - non-degenerate electron energy density (erg/ccm)
        """
        return 1.5 * self.P_e_nd

    @CachedAttribute
    def u_e_nd(self):
        """
        u_e_nd - non-degenerate electron specific energy (erg/g)
        """
        return 1.5 * physconst.Kepler.rk * self.tn * self.ye

    @CachedAttribute
    def ye(self):
        """
        Y_e
        """
        x = self.zbar / (self.abar + 1.e-99)
        # x[0] = np.nan
        return x

    @CachedAttribute
    def yeb(self):
        """
        Ye from BURN
        """
        return self.abub.Ye

    @CachedAttribute
    def etab(self):
        """
        eta = 1 - 2 Y_e from BURN
        """
        return self.abub.eta

    @CachedAttribute
    def angjeq(self):
        """
        specific equatorial angular momentum (cm**2/sec)
        """
        return self.angj * 1.5

    @CachedAttribute
    def angjeqn(self):
        """
        specific equatorial angular momentum (cm**2/sec)
        """
        return self.angjn * 1.5

    @CachedAttribute
    def icon_stripped(self):
        """
        strip spaces from icon sentinels
        """
        return np.array([x.strip() for x in self.icon])

    @CachedAttribute
    def conv(self):
        """
        full convection name
        """
        names = {
            ''     : 'radiative',
            'neut' : 'neutral',
            'osht' : 'overshooting',
            'semi' : 'semiconvective',
            'conv' : 'convective',
            'thal' : 'thermohaline',
            }
        return np.array([names.get(i, '') for i in self.icon_stripped])

    @CachedAttribute
    def iconv(self):
        """
        full convection name
        """
        numbers = {
            ''     : 0,
            'neut' : 1,
            'osht' : 2,
            'semi' : 3,
            'conv' : 4,
            'thal' : 5,
            }
        return np.array([numbers.get(i, '') for i in self.icon_stripped])

    @CachedAttribute
    def network(self):
        """
        reduced network name
        """
        netid=['APPROX', 'QSE', 'NSE']
        return np.array([netid.get(i, '') for i in self.netnum])

    @CachedAttribute
    def j_conv_core(self):
        """
        Zone of outer edge of convective core.
        """
        j = np.where(self.icon_stripped[1:] != 'conv')[0]
        return int(j[0])

    @CachedAttribute
    def s4core(self):
        """
        Return mass of core with S < 4 k_B / baryon (solar masses)

        TODO - updated version to check of base of O shell, raise, etc.
        """
        ii = np.where(self.stot>4)[0]
        if len(ii) > 0:
            return self.zm_sun[ii[0]-1]
        return 0.

    class Core(collections.OrderedDict):

        class CoreData(object):
            def __init__(self, **kwargs):
                self.j = kwargs.pop('j', None)
                self.zm_sun = kwargs.pop('zm_sun', None)
                self.rn = kwargs.pop('rn', None)
                self.ybind = kwargs.pop('ybind', None)
                self.stot = kwargs.pop('stot', None)
                self.angjt = kwargs.pop('angjt', None)
                self._nang3d = kwargs.pop('nang3d', 1)
                self._mode3d = kwargs.pop('mode3d', 'cartesian')
                if len(kwargs) != 0:
                    raise AttributeError("unexpected parameters " + str(kwargs))
            def _name(self):
                s = (
                    "j = {core.j}, " +
                    "zm_sun = {core.zm_sun}, " +
                    "rn = {core.rn}, " +
                    "ybind = {core.ybind}, " +
                    "stot = {core.stot}, "
                    )
                extra = dict()
                if self._nang3d == 1 and self._mode3d == 'cartesian':
                    s += (
                        "angjtx = {core.angjt[0]}, " +
                        "angjty = {core.angjt[1]}, " +
                        "angjtz = {core.angjt[2]}"
                        )
                elif self._nang3d == 1 and self._mode3d == 'polar':
                    s += (
                        "angjtn = {coord[0]}, " +
                        "angjtt = {coord[1]}, " +
                        "angjtp = {coord[2]}"
                        )
                    extra['coord'] = w2p(self.angjt)
                elif self._nang3d == 1 and self._mode3d == 'cylindrical':
                    s += (
                        "angjtn = {coord[0]}, " +
                        "angjtp = {coord[1]}, " +
                        "angjtz = {coord[2]}"
                        )
                    extra['coord'] = w2c(self.angjt)
                else:
                    s += (
                        "angjtn = {coord} "
                        )
                    extra['coord'] = LA.norm(self.angjt)
                return s.format(core = self, **extra)

            def __str__(self):
                return self._name()

            def __repr__(self):
                return self.__class__.__name__ + "('" + self._name() + "')"

            def __getitem__(self, key):
                if key in self.__slots__:
                    return self.__getattribute__(key)
                else:
                    raise KeyError(key)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.default_factory = self.CoreData

        head = (
            "         core     j   zm_sun          rn      ybind       stot     angjt",
            "         core     j   zm_sun          rn      ybind       stot     angjtx     angjty     angjtz",
            "         core     j   zm_sun          rn      ybind       stot     angjtn     angjtt     angjtp",
            "         core     j   zm_sun          rn      ybind       stot     angjtr     angjtp     angjtz",
            )

        @classmethod
        def core_line(cls, name = None, core_data = None, nang3d = None, mode3d = None):
            mode = 0
            if nang3d == 1:
                if mode3d == 'cartesian':
                    mode = 1
                elif mode3d == 'polar':
                    mode = 2
                elif mode3d == 'cylindrical':
                    mode = 3
            if name is None:
                return cls.head[mode]
            if core_data is None:
                return
            extra = dict()
            s = (
                " {name:>12s}" +
                "{core.j:6d}" +
                "{core.zm_sun:9.3f} " +
                "{core.rn:11.3E}" +
                "{core.ybind:11.3E}" +
                "{core.stot:11.3E}"
                )
            if mode == 0:
                s += "{angjt:11.3E}}"
                extra['angjt'] = LA.norm(core_data.angjt)
            elif mode == 1:
                s += (
                    "{core.angjt[0]:11.3E}" +
                    "{core.angjt[1]:11.3E}" +
                    "{core.angjt[2]:11.3E}" +
                    "")
            elif mode == 2:
                s += (
                    "{angjt[0]:11.3E}" +
                    "{angjt[1]:11.5F}" +
                    "{angjt[2]:11.5F}" +
                    "")
                angjt = w2p(core_data.angjt)
                angjt[1:] *= 180/np.pi
                extra['angjt'] = angjt
            elif mode == 3:
                s += (
                    "{angjt[0]:11.3E}" +
                    "{angjt[1]:11.5F}" +
                    "{angjt[2]:11.3E}" +
                    "")
                angjt = w2c(core_data.angjt)
                angjt[1] *= 180/np.pi
                extra['angjt'] = angjt
            else:
                raise Exception('Unknonw core angular momentum mode')

            return s.format(
                name = name,
                core = core_data,
                **extra)

        def text(self, *args, **kwargs):
            """
            Return text for core so it can be used for logging, etc.
            """
            text = None
            for name, core in self.items():
                if name.startswith('_'):
                    continue
                if text is None:
                    kwargs.setdefault('nang3d', core._nang3d)
                    kwargs.setdefault('mode3d', core._mode3d)
                    text = self.core_line(*args, **kwargs)
                text += '\n' + self.core_line(name, core, *args, **kwargs)
            return text

        def Print(self, mode3d = 'polar', *args, **kwargs):
            kwargs['mode3d'] = mode3d
            print(self.text(*args, **kwargs))

        def __str__(self, *args, **kwargs):
            return self.text(*args, **kwargs)


    @cachedmethod
    def core(self, xlim = 0.01, felim = 0.1, mode3d = 'cartesian'):
        """
        Compute core information and return core object.

        Parameters:
            xlim = 0.01
            felim = 0.1
        """
        def add_core(name, j):
            if j > 0:
                core[name] = self.Core.CoreData(\
                    j = min((int(j), self.jm)),
                    zm_sun = zm_sun[j-1],
                    rn = rn[j-1],
                    ybind = ybind[j],
                    stot = 0.5 * (stot[max((j-1,1))]+stot[min((j,self.jm))]),
                    angjt = angjt[j-1],
                    mode3d = mode3d,
                    nang3d = self.parm.get('nang3d', 0)
                    )

        joshell = max((0,self.jrate[11]))
        k = self.net
        z  = k.metallicity()
        fe = k.iron()
        x1h   = k.ion_abu(KepIon('h1'), missing = 0.)
        x4he  = k.ion_abu('he4')
        x12c  = k.ion_abu('c12', missing = 0.)
        x16o  = k.ion_abu('o16')
        x28si = k.ion_abu('si28')
        xiron = fe/(z + 1.e-99)
        ye = self.ye
        zm_sun = self.zm_sun
        rn = self.rn
        ybind = self.ybind
        stot = self.stot
        angjt = self.angjt
        jm = self.jm

        core = self.Core(_nang3d = self.parm.get('nang3d', 0), _mode3d = mode3d)
        add_core('center', 1)

        ii = np.where(ye > 0.49)[0]
        jye = ii[0] if len(ii) > 0 else 0
        if jye == 1:
            jye = 0
        add_core('ye core', jye)

        ii = np.where(self.netnum > 1)[0]
        japprox = ii[-1] + 1 if len(ii) > 0 else 0
        add_core('approx netw', japprox)

        ii = np.where(xiron[1:] > 0.5)[0] + 1
        jiron = ii[-1] + 1 if len(ii) > 0 else 0
        add_core('iron core', jiron)

        # the following 4 may need checking for 2nd occurrences
        ii = np.where(np.logical_and(x1h[1:] > xlim, fe[1:] < felim))[0] + 1
        jhecore = ii[0] if len(ii) > 0 else 0
        if jhecore == 1:
            jhecore = 0

        ii = np.where(np.logical_and(x4he[1:] > xlim, xiron[1:] < felim))[0] + 1
        jcocore = ii[0] if len(ii) > 0 else 0
        if jcocore == 1:
            jcocore = 0

        ii = np.where(np.logical_and(x12c[1:] > xlim, xiron[1:] < felim))[0] + 1
        jneocore = ii[0] if len(ii) > 0 else 0
        if jneocore == 1:
            jneocore = 0

        if x1h[jm] < xlim:
            jhecore = jm
        if x4he[jm] < xlim:
            jcocore = jm

        ii = np.where(np.logical_and(x16o[1:] < x28si[1:], x28si[1:] > felim))[0] + 1
        ij = np.where(ii < min((jhecore, jcocore)))[0]
        ii = ii[ij]
        jsicore = ii[-1] + 1 if len(ii) > 0 else 0

        jsicore  = max((jsicore, jiron))
        jneocore = max((jneocore, jsicore))
        jcocore  = max((jcocore, jneocore))
        jhecore  = max((jhecore, jcocore))

        if jneocore == 0:
            joshell = 0

        add_core('O shell', joshell)
        add_core('Si core', jsicore)
        add_core('Ne/Mg/O core', jneocore)
        add_core('C/O core', jcocore)
        add_core('He core', jhecore)
        add_core('star', self.jm + 1)

        return core

    def sn_result(self):
        """
        Determine SN explosion result.
        """
        ii = slice(1, -1)
        zones, = np.where((self.xbind[ii] > 0.) &
                          (self.rn[ii] > 1.e10) &
                          (self.un[ii] > 0.01 * self.uescf[ii]))
        if len(zones) > 0:
            zone = zones[0] + 1
        else:
            zone = 1
        mass = self.zm[zone-1] * physconst.Kepler.solmassi
        ekin = self.qparm.enk
        mni = (np.sum(self.xm[zone:-1] *
                      self.ni56[zone:-1])
               * physconst.Kepler.solmassi)
        mpist = (self.parm.summ0
                 * physconst.Kepler.solmassi)
        result = dict(mni = mni,
                      ekin = ekin,
                      zone = zone,
                      mass = mass,
                      mpist = mpist)
        return result


    @cachedmethod
    def xb_ion(self, ix):
        """
        return isotope mol fractions
        """
        if not isinstance(ix, Ion):
            ix = ion(ix)
        return self.yb_ion(ix) * ix.A

    @cachedmethod
    def yb_ion(self, ix):
        """
        return isotope mol fractions
        """
        if not isinstance(ix, Ion):
            ix = ion(ix)
        idx, = np.argwhere(np.equal(ix, self.isob))[0]
        x = self.ppnb[idx]
        x[-1] = self.windb[idx]
        if self.xmwind != 0:
            x[-1] /= self.xmwind
        return x

    def zone2j(self, zone):
        """
        Convert zone to array index similar to KEPLER.

        > jm+1: jm + 1 (wind)
        < 1   : jm + zone (zone counted from surface)
                lower limit is 1
        else  : zone
        """
        jm  = self.jm
        if zone > jm:
            return jm + 1
        if zone < 1:
            return max(1, jm + zone)
        return zone

    def zoner2jj(self, zone1, zone2 = None):
        """
        Convert zone range to array indices similar to KEPLER.

        > jm+1: jm + 1 (wind)
        < 1   : jm + zone (zone counted from surface)
                lower limit is 1
        else  : zone
        """
        if zone2 is None:
            if len(zone1) == 2:
                zone1, zone2 = zone1
            else:
                raise AttributeError('Wrong arguments')
        assert len(zone1) == len(zone2) == 1
        return sorted(np.array([self.zone2j(zone1), self.zone2j(zone2)]))

    @cachedmethod
    def AbuSet(self, zone):
        """
        Return AbuSet of zone, starting at 1, wind = jm+1
        """
        j = self.zone2j(zone)
        return AbuSet(self.ionsb,
                      self.xb_zone(j),
                      silent = True,
                      mixture = "Zone {:d}".format(zone),
                      comment = self.filename)

    @cachedmethod
    def xb_zone(self, zone):
        """
        Return mass fraction of zone, starting at 1, wind = jm+1
        """
        return self.yb_zone(zone) * self.Ab

    @cachedmethod
    def yb_zone(self, zone):
        """
        Return mol fraction of zone, starting at 1, wind = jm+1
        """
        j = self.zone2j(zone)
        return self.ppnb[:, j]

    @CachedAttribute
    def xb_all(self):
        """
        Return BURN mol fractions
        """
        return self.yb_all * self.Ab[:,np.newaxis]

    @CachedAttribute
    def yb_all(self):
        """
        Return BURN mol fractions
        """
        x = self.ppnb
        if self.xmwind != 0:
            x[:,-1] = self.windb / self.xmwind
        return x.transpose()

    @CachedAttribute
    def Ab(self):
        """
        Return BURN isotope mass number.
        """
        return np.array([i.A for i in self.isob])

    @CachedAttribute
    def Zb(self):
        """
        Return BURN charge mass number.
        """
        return np.array([i.Z for i in self.isob])

    @CachedAttribute
    def Nb(self):
        """
        Return BURN isotope neutron number.
        """
        return self.Ab - self.Zb

    @CachedAttribute
    def time(self):
        """
        Return current time (sec).
        """
        return self.parm.time + self.parm.toffset

    @cachedmethod
    def compactness(self, mass:'solar masses' = 2.5):
        """
        Return Compactness Parameter, O'Connor, E., & Ott, C. 2011, ApJ, 730, 70
        """
        i = np.where(self.zm_sun > mass)[0][0]
        zeta = mass / self.rn[i] * 1.e8
        return zeta

    @CachedAttribute
    def ertl(self):
        """
        Return two-parameter SN criterion, Ertl+ 2016

        https://ui.adsabs.harvard.edu/abs/2016ApJ...818..124E

        return the value pair (x = M_4 \mu_4 , y = \mu_4)

        where M4 is the location where entropy first exceeds a value of
        4 k_B / baryon and \mu_4 is the compactness of the 0.3 M_sun above it
        """
        dm = 0.3
        i = np.where(self.stot > 4)[0][0]
        m4 = self.zm_sun[i]
        j = np.where(self.zm_sun > m4 + dm)[0][0]
        mu4 = (self.zm_sun[j] - m4) / (self.rn[j] - self.rn[i]) * 1.e8
        return (m4 * mu4, mu4)

    def ertl_bh(self, k1 =  0.274, k2 = 0.0470):
        """
        Return whether star makes BH (True)

        defaul values from Ertl+ 2015, arXiv:1503.07522, s19.8 normalization
        https://ui.adsabs.harvard.edu/abs/2016ApJ...818..124E
        """
        x, y = self.ertl
        return k1 * x + k2 < y

    @CachedAttribute
    def ertl2(self):
        """
        Return MODIFIED two-parameter SN criterion, Ertl+ 2015, arXiv:1503.07522

        return the value pair (x = M_4, y = 1/\mu_4)

        where M4 is the location where entropy first exceeds a value of
        4 k_B / baryon and \mu_4 is the compactness of the 0.3 M_sun above it

        https://ui.adsabs.harvard.edu/abs/2016ApJ...818..124E
        """
        dm = 0.3
        i = np.where(self.stot > 4)[0][0]
        m4 = self.zm_sun[i]
        j = np.where(self.zm_sun > m4 + dm)[0][0]
        mu4 = (self.zm_sun[j] - m4) / (self.rn[j] - self.rn[i]) * 1.e8
        return (m4, 1/mu4)

    @CachedAttribute
    def is_presn(self):
        """
        Return whether model is pre-SN

        This is still very limited at this time and only tested for
        limited mass range.  Refinements may be needed for various mass ranges.
        """
        ok = True
        if self.dn[1] < 1.e9:
            ok = False
        if self.tn[1] < 5.e9:
            ok = False
        if self.qparm.enp > -3.e51:
            ok = False
        if self.un[1] > -1.e5:
            ok = False
        if self.core().get('iron core', None) is None:
            ok = False
        if self.netnum[1] != 3:
            ok = False
        return ok

    @CachedAttribute
    def is_agb(self):
        """
        Return whether model is AGB and wont't reach pre-SN

        This is still very limited at this time and only tested for
        limited mass range.  Refinements may be needed for various mass ranges.
        """
        ok = True
        if self.dn[1] > 1.e9:
            ok = False
        if self.tn[1] > 5.e9:
            ok = False
        if self.qparm.enp < -3.e51:
            ok = False
        if self.un[1] < -1.e5:
            ok = False
        if self.core().get('Si core', None) is not None:
            ok = False
        if np.any(self.netnum[1:-1] != 1):
            ok = False
        hecore = self.core().get('He core', None)
        if hecore is not None:
            if hecore.zm_sun > 1.37:
                ok = False
        if self.core().get('C/O core', None) is None:
            ok = False
        return ok

    @CachedAttribute
    def rhor3m(self):
        """rho * r**3 in cell center"""
        return self.dn * self.rm**3

    @CachedAttribute
    def rhor3f(self):
        """rho * r**3 on cell face"""
        x = self.rn**3
        x[1:-2] *= (self.dn[1:-2] + self.dn[2:-1]) * 0.5
        x[0] *= self.dn[1]
        x[-2] *= self.dn[-2]
        x[-1] = 0
        return x

    # return parm and qparm where possible
    def __getattr__(self, attr):
        if 'parm' in self.__dict__:
            try:
                return self.parm[attr]
            except AttributeError:
                pass
        if 'qparm' in self.__dict__:
            try:
                return self.qparm[attr]
            except AttributeError:
                pass
        raise AttributeError(attr)


load = loader(KepDump, __name__ + '.load')
_load = _loader(KepDump, __name__ + '.load')
loaddump = load

def calcai(rix, rax):
    """
    Compute specific moment of inertia.
    """
    # np.seterr(all='raise')
    ri = np.atleast_1d(rix)
    ra = np.atleast_1d(rax)
    ii = ri <= 0.
    ai = np.empty_like(ri)
    ai[ii] = 0.4 * ra[ii]**2
    ii = np.logical_not(ii)
    rai = ra[ii]*ri[ii]
    ra2 = ra[ii]**2
    ri2 = ri[ii]**2
    rm2 = ri2 + rai + ra2
    ai[ii] = 0.4 * (ri2**2 + rai * rm2 + ra2**2)/rm2
    if (not (isinstance(rix, np.ndarray) and isinstance(rax, np.ndarray))
        and len(ai) == 1):
        return float(ai[0])
    return ai

# TODO - update to current version
# TODO - check for version of kepdump and routine

def compdump(d1, d2, detail = True):
    """Compare KEPLER binary dumps."""
    def cmp_parm(name):
        v1 = d1.__getattribute__(name)
        v2 = d2.__getattribute__(name)
        for (i,(x1,x2)) in enumerate(zip(v1.data,
                                         v2.data)):
            if x1 != x2:
                print('{:s} {:s}({:s} {:d}) mismatch'.format(
                    name, v1.list[i], name[0], i))
                print('1: ', x1)
                print('2: ', x2)

    def cmp_arr(name):
        v1 = d1.__getattribute__(name)
        v2 = d2.__getattribute__(name)
        if type(v1) != type(v2):
            print(f'{name:} have different typess:')
            print(f'1: {type(v1)}')
            print(f'2: {type(v2)}')
            return
        if np.shape(v1) != np.shape(v2):
            print(f'{name:s} have different shapes:')
            print(f'1: {np.shape(v1)}')
            print(f'2: {np.shape(v2)}')
            return
        if not np.alltrue(v1 == v2):
            print('{name:s} differs')
            if detail:
                x = np.array(v1) != np.array(v2)
                if isinstance(x, np.ndarray):
                    v1 = v1.flat
                    v2 = v2.flat
                    x = x.flat
                    ii, = np.where(x)
                    for i in ii:
                        if isinstance(v1[i], str):
                            print('1: {:s}[{:d}] = {:}'.format(name,i,v1[i]))
                            print('2: {:s}[{:d}] = {:}'.format(name,i,v2[i]))
                        else:
                            print('1: {:s}[{:d}] = {:} = {:}'.format(
                                name,i,("{:08b}"*len(v1[i].data.tobytes())).format(*v1[i].data.tobytes()[::-1]),v1[i]))
                            print('2: {:s}[{:d}] = {:} = {:}'.format(
                                name,i,("{:08b}"*len(v2[i].data.tobytes())).format(*v2[i].data.tobytes()[::-1]),v2[i]))
                else:
                    print('.. different length.')

    def cmp_scalar(name):
        v1 = d1.__getattribute__(name)
        v2 = d2.__getattribute__(name)
        if v1 != v2:
            print('{:s} differ'.format(name))
            if isinstance(v1, builtins.bytes):
                print('1: {:s}'.format(('{:02X}'*len(v1)).format(*(v1))))
                print('2: {:s}'.format(('{:02X}'*len(v2)).format(*(v2))))
            else:
                print('1: ', v1)
                print('2: ', v2)

    if not isinstance(d1, KepDump):
        d1 = load(d1)
    if not isinstance(d2, KepDump):
        d2 = load(d2)

    cmp_scalar('nvers')
    # head and parameters
    cmp_parm('head')
    cmp_parm('parm')
    cmp_parm('qparm')
    # sdum
    cmp_arr('tpist')
    cmp_arr('rpist')
    cmp_arr('yemass')
    cmp_arr('yeq0')
    # ion arrays
    cmp_arr('aion')
    cmp_arr('zion')
    cmp_arr('numi')
    cmp_arr('ionn')
    # time-step controller arrays
    cmp_arr('dtc')
    cmp_arr('jdtc')
    # subroutine timing array
    cmp_arr('timeused')
    # #  reaction arrays
    cmp_arr('totalr')
    cmp_arr('rater')
    cmp_arr('qval')
    cmp_arr('jrate')
    cmp_arr('rrx')
    # accretion composition array
    cmp_arr('compsurf')
    # post-processor dump arrays (those should be removed)
    cmp_arr('locqz')
    cmp_arr('locqz0')
    cmp_arr('ratzdump')
    cmp_arr('ratiodez')
    cmp_arr('ratioadz')
    # user-specified edit arrays (are these still used?)
    cmp_arr('ndatzed')
    cmp_arr('ncyczed')
    cmp_arr('zedmass1')
    cmp_arr('zedmass2')
    # record of isotope mass lost in wind
    cmp_arr('wind')
    cmp_arr('windb')
    # names, flags, and id-words (char*8) (except namec0 is char*16)
    cmp_scalar('namep0')
    cmp_scalar('namec0')
    cmp_scalar('iflag80')
    cmp_scalar('iqbrnflg')
    cmp_scalar('craybox')
    cmp_scalar('idword')
    # storage directory, last run and code mod dates (char*16)
    cmp_scalar('nxdirect')
    cmp_scalar('lastrun')
    cmp_scalar('lastmod0')
    # arrays of symbols for isotopes, burn isotopes, time-step
    # controlers, and reactions (char*8)
    cmp_arr('ions')
    cmp_arr('ionsb')
    cmp_arr('idtcsym')
    cmp_arr('isymr')
    # post-processor file names (char*16)
    cmp_scalar('nameqq')
    cmp_scalar('nameqlib')
    cmp_scalar('nameolds')
    cmp_scalar('namenews')
    # arrays of post-processor dump variable names (char*8)
    # and labels (char*48)
    cmp_arr('namedatq')
    cmp_arr('labldatq')
    # array of edit variable names for user-specified edits (char*8)
    cmp_arr('namedzed')
    # array of aliases
    cmp_arr('savdcmd0')
    # remembered 'look' post-processing variable names (char*8) and
    # labels (char*48)
    cmp_arr('namedatl')
    cmp_arr('labldatl')
    # 'look'-read post-processor file names (char*16)
    cmp_scalar('nameqql0')
    cmp_scalar('nameqql1')
    cmp_scalar('nameqlbl')
    # output storage directory (char*48)
    cmp_scalar('nsdirect')
    # set of isotopes to be plotted (char*8) and their
    # plot icons (char*16)
    cmp_arr('isosym')
    cmp_arr('isoicon')
    # array of remembered tty command strings (char*80)
    cmp_arr('savedcmd')
    # path for location of data file (char*80)
    cmp_scalar('datapath')
    # array of zonal convection sentinels (char*8)
    cmp_arr('icon')

    # structure
    cmp_arr('ym')
    cmp_arr('rn')
    cmp_arr('rd')
    cmp_arr('un')
    cmp_arr('xln')
    cmp_arr('qln')
    cmp_arr('qld')
    cmp_arr('difi')
    cmp_arr('netnum')
    cmp_arr('xm')
    cmp_arr('dn')
    cmp_arr('tn')
    cmp_arr('td')
    cmp_arr('en')
    cmp_arr('pn')
    cmp_arr('zn')
    cmp_arr('etan')
    cmp_arr('sn')
    cmp_arr('snn')
    cmp_arr('abar')
    cmp_arr('zbar')
    cmp_arr('xkn')
    cmp_arr('xnei')
    cmp_arr('stot')
    cmp_arr('angj')
    cmp_arr('angdg')
    cmp_arr('angd')
    cmp_arr('dsold')
    cmp_arr('tsold')
    cmp_arr('ppn')

    # magnetic field data
    if d1.parm.magnet > 0:
        cmp_arr('bfvisc')
        cmp_arr('bfdiff')
        cmp_arr('bfbr')
        cmp_arr('bfbt')
        cmp_arr('bfviscef')
        cmp_arr('bfdiffef')
    # effective viscosities needed for pre-cycle mixing
    cmp_arr('angdgeff')
    cmp_arr('difieff')

    # flame data
    if d1.parm.sharp1 > 0:
        cmp_arr('xmburn')
        cmp_arr('fc12mult')

    # WIMP energy deposition data
    if (d1.parm.wimp > 0):
        cmp_arr('snw')
        cmp_arr('snwcrsi')
        cmp_arr('snwcrsd')

    # advection energy deposition data
    cmp_arr('sadv')

    # UUIDs and log data
    cmp_scalar('uuidrun')
    cmp_scalar('uuidcycle')
    cmp_scalar('uuiddump')
    cmp_scalar('uuidprev')
    cmp_scalar('uuidprog')
    cmp_scalar('uuidexec')
    cmp_scalar('nuuidhist')
    cmp_arr('uuidhist')

    # log
    cmp_scalar('nlog')
    cmp_arr('ilog')
    cmp_arr('llog')
    cmp_arr('clog')

    # viscous heating
    cmp_arr('sv')

    cmp_scalar('noparm')
    cmp_parm('oparm')

    # BURN
    imaxb = d1.qparm.imaxb
    nsaveb = d1.parm.nsaveb
    jmzb0 = d1.head.jmzb0
    if not ( (imaxb == 0) or (jmzb0 == 1) or (nsaveb == 0)):
        cmp_arr('netnumb')
        cmp_arr('limab')
        cmp_arr('limzb')
        cmp_arr('limcb')
        cmp_arr('timen')
        cmp_arr('dtimen')
        cmp_arr('dnold')
        cmp_arr('tnold')
        cmp_arr('ymb')
        cmp_arr('sburn')
        # it seems the following have dropped because by default
        # nsaveb parameter is set to 10.  Should be 12
        if nsaveb == 12:
            cmp_arr('etab')
            cmp_arr('pbuf')

        # ppnb - BURN abundances
        cmp_arr('ppnb')
        irecb = d1.parm.irecb
        if irecb == 1:
            cmp_scalar('nbmax')
            cmp_arr('nabmax')
            cmp_arr('nzbmax')
            cmp_arr('nibmax')
            cmp_arr('ionbmax')
            cmp_arr('burnamax')
            cmp_arr('burnmmax')
            cmp_arr('ibcmax')

        # accreation composition
        cmp_arr('compsurfb')

    print('\nComparison done.')

def compparm(d1, d2):
    """
    Compare parameter settings of two dumps
    """

    if not d1.__class__.__name__ == KepDump.__name__:
        d1 = load(d1)
    if not d2.__class__.__name__ == KepDump.__name__:
        d2 = load(d2)

    v1 = d1.parm
    v2 = d2.parm
    print(' '*14 + '{:>16s} {:>16s}'.format(
        os.path.basename(d1.filename),
        os.path.basename(d2.filename)))
    for (i,(x1,x2)) in enumerate(zip(v1.data,
                                     v2.data)):
        if x1 != x2:
            if np.issubdtype(x1, np.int):
                s1 = '{:>16d}'.format(int(x1))
            else:
                s1 = '{:>16g}'.format(x1)
                if len(re.findall('[.e]',s1)) == 0:
                    s1 = s1[1:] + '.'
            if np.issubdtype(x2, np.int):
                s2 = '{:>16d}'.format(int(x2))
            else:
                s2 = '{:>16g}'.format(x2)
                if len(re.findall('[.e]',s2)) == 0:
                    s2 = s2[1:] + '.'
            print('{:8s}({:>3d}) {:s} {:s}'.format(v1.list[i],  i, s1, s2))
