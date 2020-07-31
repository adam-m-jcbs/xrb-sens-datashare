#!/usr/bin/env python3

"""
Write out link file format 'kepdata'.
"""

import sys
import os
import os.path
import datetime
import uuid
import time
import gzip
import bz2
import lzma
import importlib
import argparse

import numpy as np

import uuidtime
import isotope
import utils
import ionmap

from logged import Logged
from human import byte2human
from human import version2human
from kepion import KepIon
from utils import TextFile

__version__ = 10204


class MissingBurnError(Exception):
    pass

class KepData(Logged):
    """
    KEPLER link file data output '@' files.
    """

    version = __version__

    abu_missing = -1.

    abu_unit =  {True : 'mol fraction',
                 False: 'mass fraction'}

    class DataColumn(object):
        """
        Class for data of one column

        Contains name, unit, format
        """
        def __init__(self,
                     data,
                     name = '',
                     unit = '',
                     format = 'e',
                     first = False,
                     last = True,
                     wind = False,
                     center = True,
                     missing = None,
                     width = 25,
                     void = None,
                     valid_min = 0,
                     ):
            """
            Set up Column Data.
            """
            self.name = name
            self.unit = unit
            self.width = width
            self.missing = missing
            if void is None:
                void = '-'
            if isinstance(void, str) and void.lower() == 'nan':
                void = np.nan
            if isinstance(void, str) and format == 'e':
                try:
                    void = float(void)
                except:
                    pass
            if isinstance(void, str) and format == 'd':
                try:
                    void = int(void)
                except:
                    pass
            if not isinstance(void, str) and format == 's':
                void = '{:>{:d}g}'.format(self.width, void)
            if format == 'e':
                self.format = ('{:>' +
                               '{:d}'.format(self.width) +
                               '.' +
                               '{:d}'.format(self.width-8) +
                               'e}')
            elif format == 's':
                self.format = '{:>'+'{:d}'.format(self.width)+'s}'
            elif format == 'd':
                self.format = '{:>'+'{:d}'.format(self.width)+'d}'
            else:
                raise Exception('Unsupported field format')
            self.first = first
            self.last = last
            self.wind = wind
            if not isinstance(void, str):
                self.void = void = self.format.format(void)
            else:
                self.void = void
            self.center = center
            self.data = data
            self.ndata = len(data)
            self.fmt = '{:>'+'{:d}'.format(self.width)+'s}'
            self.valid_min = valid_min

        def out(self, idx):
            """
            Return formatted column data.

            idx:
             'name'
             'unit'
             zone number
             'wind'
            """

            if idx == 'name':
                return self.fmt.format(self.name)
            if idx == 'unit':
                return self.fmt.format(self.unit)
            if idx == 'wind':
                idx = self.ndata - 1

            if self.first:
                start = 0
            else:
                start = 1
            start += self.valid_min

            end = self.ndata - 1
            if not self.wind:
                end -= 1
            if not self.last:
                end -= 1

            if start <= idx <= end:
                if self.data[idx] == self.missing:
                    s = self.void
                else:
                    s = self.format.format(self.data[idx])
            else:
                s = self.void
            return self.fmt.format(s)

    class Grid(DataColumn):
        """
        Write grid data.
        """

        def __init__(self,
                     ndata = 0,
                     name = 'grid',
                     unit = 'unit',
                     width = 5,
                     ):
            self.width = width
            self.format = '{:>'+'{:d}'.format(self.width)+'d}'
            self.fmt = '{:>'+'{:d}'.format(self.width)+'s}'
            self.name = name
            self.unit = unit
            self.ndata = ndata

        def out(self, idx):
            if idx == 'name':
                return self.fmt.format(self.name)
            if idx == 'unit':
                return self.fmt.format(self.unit)
            if idx == 'wind':
                s = self.fmt.format('wind')
            else:
                s = self.format.format(idx)
            return s


    def __init__(self,
                 filename = None,
                 silent = False,
                 dump = None,
                 loader = 'kepdump.load',
                 **kwargs):
        """
        this routine does all the work
        """

        self.setup_logger(silent)

        if isinstance(dump, str):
            filename = dump
            dump = None

        if not isinstance(filename, str):
            dump = filename
            filename = None

        if dump is not None:
            filename = dump.filename

        filename = os.path.expanduser(filename)
        filename = os.path.expandvars(filename)
        self.filename = os.path.abspath(filename)

        self.kw = dict(kwargs)

        self.username = self.kw.get('inusername', os.getlogin())
        self.hostname = self.kw.get('inhostname', os.uname().nodename)
        inpathname = self.kw.get('inpathname', None)
        if inpathname is None:
            self.infilename = self.kw.get('infilename', self.filename)
        else:
            inpathname = os.path.expanduser(inpathname)
            inpathname = os.path.expandvars(inpathname)
            self.infilename = os.path.join(
                inpathname,
                os.path.basename(self.filename),
                )

        self.outfilename = self.kw.get('outfilename', None)
        self.outpathname = self.kw.get('outpathname', None)
        if self.outpathname is not None:
            self.outpathname = os.path.expanduser(self.outpathname)
            self.outpathname = os.path.expandvars(self.outpathname)

        burn = False  # maybe change to True by default?
        for iburn in ['radiso', 'deciso', 'elements', 'sn', 'burn', 'alliso' ]:
            if self.kw.get(iburn, False):
                burn = True

        # load model data
        if dump is None:
            if isinstance(loader, str):
                modulename, loadername = loader.rsplit('.', 1)
                module = importlib.import_module(modulename)
                loader = module.__dict__[loadername]
            if hasattr(loader, '__call__'):
                try:
                    dump = loader(
                        self.filename,
                        killburn = not burn,
                        silent = silent)
                except:
                    raise Exception("""Require loader that supports interface
                        loader(<filename>, killburn=<bool>, silent=<bool>)
                        """)
            else:
                raise Exception('Require callable loader')
        self.dump = dump

        self.silent = silent

        output = False
        if self.kw.get('outfile', None) or self.kw.get('at', None) or self.kw.get('screen', None) :
            output = True

        if output:
            self.output(silent = silent, **self.kw)
        self.close_logger()


    def header(self):
        """
        Return file header string.
        """
        fileuuid = uuidtime.UUID1()

        s = '{:s} VERSION {:6d}\n'.format(self.comment,
                                          self.version)
        s += '{:s}\n'.format(self.comment)
        if self.format is not None:
            x = self.format
        else:
            x = '-'
        s += '{:s} {:<25s} {:<25s}\n'.format(self.comment,
                                             'format',
                                             x)
        s += '{:s} {:<25s} {:<25s}\n'.format(self.comment,
                                             'created at',
                                             time.asctime(time.gmtime()))
        s += '{:s} {:<25s} {:<25s}\n'.format(self.comment,
                                             'created by',
                                             self.username)
        s += '{:s} {:<25s} {:<25s}\n'.format(self.comment,
                                             'created on',
                                             self.hostname)
        s += '{:s} {:<25s} {:<25s}\n'.format(self.comment,
                                             'created from',
                                             self.infilename)
        s += '{:s} {:<25s} {:<25s}\n'.format(self.comment,
                                             'file date',
                                             time.asctime(time.gmtime(os.stat(self.filename).st_mtime)))
        if self.outfilename is not None:
            x = self.outfilename
        else:
            x = '-'
        s += '{:s} {:<25s} {:<25s}\n'.format(self.comment,
                                             'created as',
                                             x)
        s += '{:s} {:<25s} {:<25d}\n'.format(self.comment,
                                             'cycle',
                                             self.dump.qparm.ncyc)
        s += '{:s} {:<25s} {:<25.17e}\n'.format(self.comment,
                                                'time (sec)',
                                                self.dump.parm.time)
        s += '{:s} {:<25s} {:<25.17e}\n'.format(self.comment,
                                                'time offset (sec)',
                                                self.dump.parm.toffset)
        # kepler version info
        s += '{:s} {:<25s} {:<s}\n'.format(self.comment,
                                           'KEPLER Version',
                                           version2human(self.dump.parm.nsetparm))
        # add git info
        if hasattr(self.dump, 'gitsha'):
            branch = self.dump.gitbranch
            if branch == '':
                branch = '(NOT ON BRANCH)'
            s += '{:s} {:<25s} {:s}\n'.format(self.comment,
                                        'GIT  SHA',
                                        self.dump.gitsha)
            s += '{:s} {:<25s} {:s}\n'.format(self.comment,
                                        'GIT  BRANCH',
                                        branch)
        # add times, machine name (IP) library
        s += '{:s} {:<25s} {!s} {:s}\n'.format(self.comment,
                                          'UUID PROG',
                                          uuidtime.UUID(bytes=self.dump.uuidprog ),
                                          uuidtime.UUID(bytes=self.dump.uuidprog ).ctimex())
        s += '{:s} {:<25s} {!s} {:s}\n'.format(self.comment,
                                          'UUID RUN',
                                          uuidtime.UUID(bytes=self.dump.uuidrun  ),
                                          uuidtime.UUID(bytes=self.dump.uuidrun  ).ctimex())
        s += '{:s} {:<25s} {!s} {:s}\n'.format(self.comment,
                                          'UUID CYCLE',
                                          uuidtime.UUID(bytes=self.dump.uuidcycle),
                                          uuidtime.UUID(bytes=self.dump.uuidcycle).ctimex())
        s += '{:s} {:<25s} {!s} {:s}\n'.format(self.comment,
                                          'UUID DUMP',
                                          uuidtime.UUID(bytes=self.dump.uuiddump ),
                                          uuidtime.UUID(bytes=self.dump.uuiddump ).ctimex())
        s += '{:s} {:<25s} {!s} {:s}\n'.format(self.comment,
                                          'UUID EXEC',
                                          uuidtime.UUID(bytes=self.dump.uuidexec ),
                                          uuidtime.UUID(bytes=self.dump.uuidexec ).ctimex())
        s += '{:s} {:<25s} {!s} {:s}\n'.format(self.comment,
                                          'UUID THIS FILE',
                                          fileuuid,
                                          fileuuid.ctimex())
        s += '{:s}\n'.format(self.comment)
        return s


    def output(self,
               outfile = None,
               outdir = None,
               compress = True,
               comment = '#',
               void = None,
               wind = None,
               center = None,
               at = None,
               wind_numeric = None,
               perm = None,
               **kw):
        """
        Open and write out data file.

        If 'outfile' is specified it supersedes 'at'.

        'comress' can be '\.?gz', '\.?bz2', '\.?xz', or
           True (same as 'gz' by default).
        """
        xkw = dict(self.kw)
        xkw.update(kw)

        silent = xkw.get('silent', False)

        self.setup_logger(silent)

        self.format = None
        self.comment = comment

        # structure
        cols = ['grid', 'dm', 'm', 'r', 'u', 'rho', 'T', 'P', 'e', 's',
                'omega', 'Abar', 'Ye']

        if xkw.get('column', False):
            cols[cols.index('dm')] = 'dy'
            cols[cols.index( 'm')] =  'y'

        if xkw.get('luminosity', False):
            cols += ['L']
        if xkw.get('diffusion', False):
            cols += ['D']
        if xkw.get('magnet', False):
            cols += ['Bt', 'Br']
        if xkw.get('neutrino', False):
            cols += ['eps_nu', 'L_nu', 'F_nu']
        if xkw.get('opacity', False):
            cols += ['kappa']

        # determine angular velocity mode
        nang3d = self.dump.parm.get('nang3d', 0)
        if 'omega' in cols:
            iomega = cols.index('omega')
            if nang3d == 0:
                cols[iomega] = 'omegan'
            else:
                cols[iomega] = 'omega3'

        # network and structure info
        if xkw.get('structure', False):
            cols += ['structure']
        if xkw.get('network', False):
            cols += ['network']

        # find which network info to write
        if xkw.get('burn', False) or self.kw.get('alliso', False):
            cols += ['burn']
        elif xkw.get('approx', False):
            cols += ['approx']
            if void is None:
                void = 0.
        elif xkw.get('radiso', False):
            cols += ['radiso']
        elif xkw.get('deciso', False):
            cols += ['deciso']
        elif xkw.get('elements', False):
            cols += ['elements']
            xkw.setdefault('molfrac', True)
        elif xkw.get('sn', False):
            cols += ['sn']
        elif xkw.get('hhez', False):
            cols += ['hhez']
            xkw.setdefault('molfrac', False)
        elif xkw.get('structure', False):
            self.format = 'structure'
        else:
            cols += ['ions']

        if self.format is None:
            self.format = cols[-1]

        molfrac = xkw.get('molfrac', False)
        if self.format != 'structure':
            self.format += ' ({:s})'.format(self.abu_unit[molfrac])

        # add quantities
        columns = []
        for col in cols:
            if col == 'grid':
                columns += [self.Grid(
                    ndata = self.dump.jm)]
            elif col == 'm':
                columns += [self.DataColumn(
                    data = self.dump.zm,
                    name = 'cell outer total mass',
                    unit = 'g',
                    center = False,
                    first = True,
                    wind = True,
                    void = void,
                    )]
            elif col == 'dm':
                columns += [self.DataColumn(
                    data = self.dump.xm,
                    name = 'cell mass',
                    unit = 'g',
                    center = True,
                    first = True,
                    wind = True,
                    void = void,
                    )]
            elif col == 'y':
                columns += [self.DataColumn(
                    data = self.dump.y,
                    name = 'cell outer column depth',
                    unit = 'g/cm**2',
                    center = False,
                    first = True,
                    wind = True,
                    void = void,
                    )]
            elif col == 'dy':
                columns += [self.DataColumn(
                    data = self.dump.dy,
                    name = 'cell column density',
                    unit = 'g/cm**2',
                    center = True,
                    first = True,
                    wind = True,
                    void = void,
                    )]
            elif col == 'r':
                columns += [self.DataColumn(
                    data = self.dump.rn,
                    name = 'cell outer radius',
                    unit = 'cm',
                    center = False,
                    first = True,
                    wind = False,
                    void = void,
                    )]
            elif col == 'u':
                columns += [self.DataColumn(
                    data = self.dump.un,
                    name = 'cell outer velocity',
                    unit = 'cm/sec',
                    center = False,
                    first = True,
                    wind = False,
                    void = void,
                    )]
            elif col == 'rho':
                columns += [self.DataColumn(
                    data = self.dump.dn,
                    name = 'cell density',
                    unit = 'g/cm**3',
                    center = True,
                    first = False,
                    wind = False,
                    void = void,
                    )]
            elif col == 'T':
                columns += [self.DataColumn(
                    data = self.dump.tn,
                    name = 'cell temperature',
                    unit = 'K',
                    center = True,
                    first = False,
                    wind = False,
                    void = void,
                    )]
            elif col == 'P':
                columns += [self.DataColumn(
                    data = self.dump.pn,
                    name = 'cell pressure',
                    unit = 'dyn/cm**2',
                    center = True,
                    first = False,
                    wind = False,
                    void = void,
                    )]
            elif col == 'e':
                columns += [self.DataColumn(
                    data = self.dump.en,
                    name = 'cell spec. int. energy',
                    unit = 'erg/g',
                    center = True,
                    first = False,
                    wind = False,
                    void = void,
                    )]
            elif col == 's':
                columns += [self.DataColumn(
                    data = self.dump.stot,
                    name = 'cell specific entropy',
                    unit = 'kb/baryon',
                    center = True,
                    first = False,
                    wind = False,
                    void = void,
                    )]
            elif col == 'omegan':
                columns += [self.DataColumn(
                    data = self.dump.angwn,
                    name = 'cell angular velocity',
                    unit = 'rad/sec',
                    center = True,
                    first = True,
                    wind = False,
                    void = void,
                    )]
            elif col == 'omega3':
                for i,d in enumerate('xyz'):
                    columns += [self.DataColumn(
                        data = self.dump.angw[:, i],
                        name = f'cell {d} angular velocity',
                        unit = 'rad/sec',
                        center = True,
                        first = True,
                        wind = False,
                        void = void,
                        )]
            elif col == 'Abar':
                columns += [self.DataColumn(
                    data = self.dump.abar,
                    name = 'cell A_bar',
                    unit = 'amu',
                    center = True,
                    first = False,
                    wind = False,
                    void = void,
                    )]
            elif col == 'Ye':
                columns += [self.DataColumn(
                    data = self.dump.ye,
                    name = 'cell Y_e',
                    unit = '',
                    center = True,
                    first = False,
                    wind = False,
                    void = void,
                    )]
            elif col == 'Br':
                columns += [self.DataColumn(
                    data = self.dump.bfbr,
                    name = 'cell outer B_poloidal',
                    unit = 'G',
                    center = False,
                    first = False,
                    wind = False,
                    void = void,
                    )]
            elif col == 'Bt':
                columns += [self.DataColumn(
                    data = self.dump.bfbt,
                    name = 'cell outer B_toroidal',
                    unit = 'G',
                    center = False,
                    first = True,
                    wind = False,
                    void = void,
                    )]
            elif col == 'L':
                columns += [self.DataColumn(
                    data = self.dump.xln,
                    name = 'cell outer luminosity',
                    unit = 'erg/s',
                    center = False,
                    first = True,
                    wind = False,
                    void = void,
                    )]
            elif col == 'D':
                columns += [self.DataColumn(
                    data = self.dump.xln,
                    name = 'cell outer diffusion',
                    unit = 'cm**2/sec',
                    center = False,
                    first = True,
                    wind = False,
                    void = void,
                    )]
            elif col == 'kappa':
                columns += [self.DataColumn(
                    data = self.dump.xkn,
                    name = 'cell opacity',
                    unit = 'cm**2/g',
                    center = True,
                    first = False,
                    wind = False,
                    void = void,
                    )]
            elif col == 'eps_nu':
                columns += [self.DataColumn(
                    data = self.dump.snu,
                    name = 'cell spec. nu loss rate',
                    unit = 'erg/g/sec',
                    center = True,
                    first = False,
                    wind = False,
                    void = void,
                    )]
            elif col == 'L_nu':
                columns += [self.DataColumn(
                    data = self.dump.xlnu,
                    name = 'cell outer nu luminosity',
                    unit = 'erg/sec',
                    center = False,
                    first = True,
                    wind = False,
                    void = void,
                    )]
            elif col == 'F_nu':
                columns += [self.DataColumn(
                    data = self.dump.xlnu,
                    name = 'cell outer nu flux',
                    unit = 'erg/sec',
                    center = False,
                    first = True,
                    wind = False,
                    void = void,
                    )]
            elif col == 'stability':
                columns += [self.DataColumn(
                    data = self.dump.conv,
                    name = 'outer cell stability',
                    unit = '',
                    format = 's',
                    center = False,
                    first = False,
                    last = False,
                    wind = False,
                    void = void,
                    )]
            elif col == 'network':
                columns += [self.DataColumn(
                    data = self.dump.network,
                    name = 'cell reduced network',
                    unit = '',
                    format = 's',
                    center = False,
                    first = False,
                    last = True,
                    wind = False,
                    void = void,
                    )]
            elif col == 'ions':
                ions = KepIon.ion_names.copy()
                del ions[2]
                abu = self.dump.KepAbuDump
                for i in ions:
                    columns += [self.DataColumn(
                        data = abu.ion_abu(i,
                                           missing = self.abu_missing,
                                           molfrac = molfrac),
                        name = i,
                        missing = self.abu_missing,
                        unit = self.abu_unit[molfrac],
                        center = True,
                        first = False,
                        wind = True,
                        void = void,
                        )]
            elif col == 'hhez':
                ions = KepIon.ion_names.copy()
                abu = self.dump.KepAbuDump.XYZ(
                    missing = self.abu_missing,
                    molfrac = molfrac)
                for i,x in zip(['X','Y','Z'], abu.transpose()):
                    columns += [self.DataColumn(
                        data = x,
                        name = i,
                        missing = self.abu_missing,
                        unit = self.abu_unit[molfrac],
                        center = True,
                        first = False,
                        wind = True,
                        void = void,
                        )]
            elif col == 'approx':
                ions = KepIon.approx_ion_names
                abu = self.dump.KepAbuDump
                for i in ions:
                    columns += [self.DataColumn(
                        data = abu.ion_abu(KepIon(i),
                                           missing = self.abu_missing,
                                           molfrac = molfrac),
                        name = i,
                        missing = -1,
                        unit = self.abu_unit[molfrac],
                        center = True,
                        first = False,
                        wind = True,
                        void = void,
                        )]
            elif col in [ 'radiso', 'deciso', 'elements', 'burn'] :
                if self.dump.qparm.imaxb == 0:
                    raise MissingBurnError()
                dkw = { 'burn' : dict(decay = False),
                        'radiso' : dict(),
                        'deciso' : dict(stable = True),
                        'elements' : dict(elements = True),
                        }[col]
                dkw['silent'] = silent
                valid_min = self.dump.jburnmin
                # abu = self.dump.AbuDump
                # decay = ionmap.Decay(
                #     abu,
                #     molfrac_out = molfrac,
                #     **dkw)
                # abu *= decay
                # self.decay = decay
                # abu = ionmap.Decay.Map(
                #     self.dump.AbuDump,
                #     molfrac_out = molfrac,
                #     **dkw)
                # for i,a in abu:
                for i,a in ionmap.Decay.Map(self.dump.AbuDump,
                                            molfrac_out = molfrac,
                                            **dkw):
                    columns += [self.DataColumn(
                        data = a,
                        name = str(i),
                        unit = self.abu_unit[molfrac],
                        center = True,
                        first = False,
                        wind = True,
                        valid_min = valid_min,
                        void = void,
                        )]
            else:
                raise Exception(f'column type "{col}" not found.')

        if outdir is None:
            if outfile:
                xpath, outfile = os.path.split(outfile)
                if xpath is not '':
                    outdir = xpath

        if outdir is None:
            outdir = os.path.dirname(self.filename)

        if at == True:
            if len({'burn', 'deciso', 'radiso', 'elements'} & set(xkw)):
                at = '@@'
            else:
                at = '@'

        if at:
            if not outfile:
                outfile = os.path.basename(self.filename)
            else:
                if not outdir:
                    outdir = os.path.split(outfile)[0]
                outfile = os.path.basename(outfile)
            if outfile.count('#') == 1:
                outfile = outfile.replace('#', at)
            elif outfile.count('#') == 0:
                outfile = '{:s}{:s}{:d}'.format(outfile, at, self.dump.qparm.ncyc)
            else:
                outfile = outfile[::-1].replace('#', at, 1)[::-1]
                self.logger.warning('Only replaceing last occurrence of `' + at + '`.')

        if outfile:
            outfile = os.path.join(outdir, outfile)

        self.outfile = outfile

        if self.outpathname is not None:
            self.outfilename = os.path.join(
                self.outpathname,
                os.path.basename(self.outfile),
                )
        elif self.outfilename is None:
            self.outfilename = self.outfile

        # decide about inner boundary
        zones = ['name', 'unit']
        if center is None:
            center = self.dump.rn[0] != 0.
        if center:
            zones += [0]

        # add zones
        zones += list(range(1, self.dump.jm+1))

        # decide about outer boundary
        if wind is None:
            if wind_numeric is None:
                wind = self.dump.xmwind != 0
            else:
                wind = wind_numeric
        if wind_numeric is None:
            wind_numeric = False
        if wind:
            if wind_numeric:
                zones += [self.dump.jm+1]
            else:
                zones += ['wind']

        with TextFile(self.outfilename,
                      mode = 'w',
                      compress = compress,
                      return_filename = True) as (fout, fout_filename):

            # write header
            fout.write(self.header())

            # output zones
            for j in zones:
                if isinstance(j, str):
                    s = self.comment
                else:
                    s = ' ' * len(self.comment)
                for col in columns:
                    s += col.out(j)
                fout.write(s + '\n')

        # close file
        if fout != sys.stdout:
            self.logger_timing('Data written in')
            self.logger.critical(
                'Wrote {fn:s} ({fs:s}): {fm:s}.'.format(
                fn = fout_filename,
                fs = byte2human(os.path.getsize(fout_filename)),
                fm = self.format))
            if perm:
                os.chmod(fout_filename, perm)
        else:
            self.logger_timing('Data generated in')
            self.logger.info('Format: ' + self.format)

        self.close_logger()


        # add accmass to xm for XRBs?


def kepdata(**args):
    return KepData(**args)

def command(args, progname = None):
    if isinstance(args, str):
        args = [args]

    parser = argparse.ArgumentParser(
        description = 'Common KEPLER dump link generator "@" aka kepdata.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog = 'Please keep the comment lines on top for model reference.',
        fromfile_prefix_chars = '!')

    parser.add_argument(
        'filename',
        nargs = '+',
        metavar = 'infile',
        action = 'store',
        help = 'input KEPLER dump file')

    parser.add_argument(
        '-l', '--loader',
        action = 'store',
        dest = 'loader',
        default = argparse.SUPPRESS,
        help = 'callable loader to create interface object; specify full module name including object/function name')

    outfile = parser.add_mutually_exclusive_group(
        required = False)
    outfile.add_argument(
        '-o', '--outfile',
        dest = 'outfile',
        nargs = '?',
        action = 'store',
        default = argparse.SUPPRESS,
        help = 'output ASCII data file')
    outfile.add_argument(
        '-@','--at',
        nargs = '?',
        const = True,
        metavar = 'X',
        default = argparse.SUPPRESS,
        help = "replace '#' with X (default: X = '@' or '@@')")
    outfile.add_argument(
        '-', '--screen',
        dest = 'screen',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = "print to screen")

    parser.add_argument(
        '-D','--outdir',
        action = 'store',
        dest = 'outdir',
        default = argparse.SUPPRESS,
        help = 'output directory for ASCII data file')

    parser.add_argument(
        '-p','--permission',
        action = 'store',
        dest = 'perm',
        default = argparse.SUPPRESS,
        help = 'output file permissions (numeric)')

    abu_format = parser.add_mutually_exclusive_group()
    abu_format.add_argument(
        '-a','--approx',
        dest = 'approx',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = "use APPROX network ")
    abu_format.add_argument(
        '-b','--burn',
        dest = 'burn',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = 'output BURN isotopes')
    abu_format.add_argument(
        '--sn',
        dest = 'sn',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = 'make SN format output')
    abu_format.add_argument(
        '-e','--elements',
        dest = 'elements',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = 'make elements format output')
    abu_format.add_argument(
        '-d','--deciso',
        dest = 'deciso',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = 'make decayed isotope format output')
    abu_format.add_argument(
        '-r','--radiso',
        dest = 'radiso',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = 'make decayed isotope format output')
    abu_format.add_argument(
        '-x','--hhez', '--xyz',
        dest = 'hhez',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = 'make XYZ format output')
    abu_format.add_argument(
        '-S','--structure',
        dest = 'structure',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = 'only output structure, suppress composition output')

    massfrac = parser.add_mutually_exclusive_group()
    massfrac.add_argument(
        '-m','--molfrac',
        dest = 'molfrac',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = 'output mol fractions instead of mass fractions; default, e.g., for elements')
    massfrac.add_argument(
        '-M','--massfrac',
        dest = 'molfrac',
        action = 'store_false',
        default = argparse.SUPPRESS,
        help = ( 'output mass fractions instead of mol fractions; ' +
                 'overwrite molfrac where this is the natural default behaviour'))

    wind = parser.add_mutually_exclusive_group()
    wind.add_argument(
        '-w','--wind',
        dest = 'wind',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = 'write out wind')
    wind.add_argument(
        '--no-wind',
        dest = 'wind',
        action = 'store_false',
        default = argparse.SUPPRESS,
        help = 'do not write out wind')

    parser.add_argument(
        '-W', '--wind-numeric',
        dest = 'wind_numeric',
        action = 'store_true',
        default = False,
        help = 'write wind zone as number, implies -w')

    center = parser.add_mutually_exclusive_group()
    center.add_argument(
        '-0', '--center',
        dest = 'center',
        action = 'store_true',
        default = argparse.SUPPRESS,
        help = 'write out central zone')
    center.add_argument(
        '--no-center',
        dest = 'center',
        action = 'store_false',
        default = argparse.SUPPRESS,
        help = 'do not write out central zone')

    compress = parser.add_mutually_exclusive_group()
    compress.add_argument(
        '-c', '--compress',
        dest = 'compress',
        nargs = '?',
        choices = ('gz', 'bz2', 'xz'),
        default = argparse.SUPPRESS,
        const = 'xz',
        help = 'compress output (default: xz)')
    compress.add_argument(
        '-z', '--gz',
        dest = 'compress',
        action = 'store_const',
        const = 'gz',
        default = argparse.SUPPRESS,
        help = 'compress output in gz format (default)')
    compress.add_argument(
        '-J', '--xz',
        dest = 'compress',
        action = 'store_const',
        const = 'xz',
        default = argparse.SUPPRESS,
        help = 'compress output in xz format')
    compress.add_argument(
        '-j', '--bz2',
        dest = 'compress',
        action = 'store_const',
        const = 'bz2',
        default = argparse.SUPPRESS,
        help = 'compress output in bz2 format')
    compress.add_argument(
        '-u', '--uncompressed',
        dest = 'compress',
        default = argparse.SUPPRESS,
        const = '',
        action = 'store_const',
        help = 'do not compress output')

    parser.add_argument(
        '--comment',
        default = argparse.SUPPRESS,
        metavar = 'C',
        help = 'comment character, default is "#"')
    parser.add_argument(
        '-V', '--void',
        dest = 'void',
        action = 'store',
        metavar = 'VOID',
        # nargs = 1,
        default = argparse.SUPPRESS,
        help = 'value to use for invalid entries, default is "-"')

    parser.add_argument(
        '-y', '--column',
        dest = 'column',
        action = 'store_true',
        default = False,
        help = 'use column depth for y axis, default "False"')

    parser.add_argument(
        '-L', '--luminosity',
        dest = 'luminosity',
        action = 'store_true',
        default = False,
        help = 'output luminosity')

    parser.add_argument(
        '-F', '--diffusion',
        dest = 'diffusion',
        action = 'store_true',
        default = False,
        help = 'output diffusion coefficient')

    parser.add_argument(
        '-B', '--magnet',
        dest = 'magnet',
        action = 'store_true',
        default = False,
        help = 'output magnetic fields from dynamo model')

    parser.add_argument(
        '-N', '--neutrino',
        dest = 'neutrino',
        action = 'store_true',
        default = False,
        help = 'output neutrino quantities')

    parser.add_argument(
        '-X', '--opacity',
        dest = 'opacity',
        action = 'store_true',
        default = False,
        help = 'output opacity')

    parser.add_argument(
        '-s', '--stability',
        dest = 'stability',
        action = 'store_true',
        default = False,
        help = 'output stability information (text)')

    parser.add_argument(
        '-n', '--network',
        dest = 'network',
        action = 'store_true',
        default = False,
        help = 'output short network information APPROX|QSE|NSE (text)')

    # overwrite some comments
    overwritesource = parser.add_mutually_exclusive_group(
        required = False)
    overwritesource.add_argument(
        '--infilename',
        dest = 'infilename',
        metavar = 'PATH',
        nargs = '?',
        action = 'store',
        default = argparse.SUPPRESS,
        help = 'overwrite file name of source file (including path)')
    overwritesource.add_argument(
        '--inpathname',
        dest = 'inpathname',
        metavar = 'PATH',
        nargs = '?',
        action = 'store',
        default = argparse.SUPPRESS,
        help = 'overwrite path name of source file (not including file name)')

    overwritetarget = parser.add_mutually_exclusive_group(
        required = False)
    overwritetarget.add_argument(
        '--outfilename',
        dest = 'outfilename',
        metavar = 'PATH',
        nargs = '?',
        action = 'store',
        default = argparse.SUPPRESS,
        help = 'overwrite file name of source file (including path)')
    overwritetarget.add_argument(
        '--outpathname',
        dest = 'outpathname',
        metavar = 'PATH',
        nargs = '?',
        action = 'store',
        default = argparse.SUPPRESS,
        help = 'overwrite path name of source file (not including file name)')

    parser.add_argument(
        '--inusername',
        dest = 'inusername',
        metavar = 'USER',
        nargs = '?',
        action = 'store',
        default = argparse.SUPPRESS,
        help = 'overwrite user name')
    parser.add_argument(
        '--inhostname',
        dest = 'inhostname',
        metavar = 'HOST',
        nargs = '?',
        action = 'store',
        default = argparse.SUPPRESS,
        help = 'overwrite host name')

    parser.add_argument(
        '-v','--version',
        action = 'version',
        version = version2human(__version__))

    quiet = parser.add_mutually_exclusive_group()
    quiet.add_argument(
        '-Q','--really-quiet',
        action = 'store_true',
        dest = 'silent',
        default = argparse.SUPPRESS,
        help = 'no output whatsoever')
    quiet.add_argument(
        '-q','--quiet',
        action = 'store_const',
        const = 50,
        dest = 'silent',
        default = argparse.SUPPRESS,
        help = 'quiet operation')
    quiet.add_argument(
        '--verbose',
        action = 'store_false',
        dest = 'silent',
        default = argparse.SUPPRESS,
        help = 'verbose operation')

    # set default values
    parser.set_defaults(compress = 'xz')

    # do the parsing
    args = parser.parse_args(args)

    args = vars(args)

    if not { 'outfile', 'at', 'screen'} & set(args):
        args['at'] = True

    if 'perm' in args:
        try:
            args['perm'] = int(args['perm'], base = 8)
        except Exception as e:
            print(' [kepdata] Error in permissions:', e)
            return

    filenames = args.pop('filename')
    if len (filenames) > 1:
        if args.get('infilename', None):
            raise Exception('--infilename parameter may only be used with single file name')
        if args.get('outfilename', None):
            raise Exception('--outfilename parameter may only be used with single file name')
    for filename in filenames:
        try:
            kepdata(filename = filename, **args)
        except MissingBurnError:
            print(' ERROR: NO BURN NETWORK FOUND.\n ABORDING.')

if __name__ == "__main__":
    args = sys.argv
    command(args[1:], progname = args[0])
