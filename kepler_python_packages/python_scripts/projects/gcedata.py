"""
Load GCE results from Chris West
"""
import itertools
import os.path
import os
import lzma
import re
import time
import numpy as np

from utils import TextFile

import kepdump
import kepdata
import isotope
import physconst

from abuset import AbuSet, AbuData
from abusets import SolAbu
from logged import Logged
from ionmap import Decay
from stardb import StarDB

from sn_analytic import SN

masses = [13, 15, 17, 20, 22, 25, 30]
version = '4'
series = 'sin'
Z = [ 0.3, 0.2, 0.1, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.5, -2.0,
      -2.5, -3.0, -4.0, -1000.]
# Z = [0.3]
# masses = [13,15]
path00 = '/x/chris/zdep/'
path00 = '/travel2/chris/zdep/'
explosion = 'D'
dump = 'final'

def z2s(z):
    s = '{:02d}'.format(int(abs(z)*10 + 0.001))
    if z < 0:
        s = '_' + s
    return s


class Grid(Logged):
    def __init__(self,
                 silent = False,
                 sn_cutoff = 'None',
                 add_x0 = False,
                 series = series,
                 version = version,
                 ):
        self.setup_logger(silent = silent)
        models = []
        solref = SolAbu('Lo09')
        Z0 = solref.metallicity()
        for mass, metallicity in itertools.product(masses, Z):
            self.logger.info('Loading mass = {}, [Z] = {}'.format(mass, metallicity))
            if metallicity < -9.99:
                path = '/home/alex/kepler/znuc/z{}'.format(mass)
                dumpfile = os.path.join(path, 'z{:d}{}#{}'.format(mass, explosion, dump))
                presnfile = os.path.join(path, 'z{:d}#presn'.format(mass))
                linkfile = os.path.join(path, '{}.link'.format(explosion))
                # outfilename = 'z{:d}@presn'.format(mass)
                bgfile = '/home/alex/kepler/local_data/z0cbg'
            else:
                path = os.path.join(
                    path00,
                    version + 'mass{:d}'.format(mass),
                    series + z2s(metallicity))
                dumpfilename = os.path.join(path, series + '{}#{}'.format(explosion, dump))
                presnfilename = os.path.join(path, series + '#presn')
                linkfilename = os.path.join(path, '{}.link'.format(explosion))
                # outfilename = 'x{:2d}{:+3.1f}@presn'.format(mass, metallicity)
                bgfile = os.path.join(
                    path00,
                    'local_data',
                    'sol{}{}g'.format(series, z2s(metallicity)))
            # kepdata.command(['-D', '/home/alex/grid', '-o', outfilename, presnfilename])
            # continue

            d = kepdump._load(dumpfilename)
            p = kepdump._load(presnfilename)
            if d is None:
                self.logger.info('{} not found'.format(dumpfilename))
            else:
                zonecut = self.get_fallback(linkfilename)
                a = d.AbuDump
                if sn_cutoff == 'xi2.5=0.25':
                    if p.compactness(2.5) > 0.25:
                        zonecut = d.jm + 1
                elif sn_cutoff == 'xi2.5=0.45':
                    if p.compactness(2.5) > 0.45:
                        zonecut = d.jm + 1
                elif sn_cutoff == 'ertl_s19.8':
                    if p.ertl_bh(k1 = 0.274,
                                 k2 = 0.0470):
                        zonecut = d.jm + 1
                elif sn_cutoff == 'muller':
                    result = SN(p).get_explosion()
                    if result['e_expl'] == 0:
                        zonecut = d.jm + 1
                    print(' [Grid] E', result['e_expl'])
                elif sn_cutoff == 'muller_cutoff':
                    result = SN(p).get_explosion()
                    zonecut = result['i_final'] + 1
                    zonecut = np.where(d.zm > p.zmm[zonecut])[0][0]
                    print(' [Grid] j', d.jm, zonecut)
                elif sn_cutoff is None:
                    pass
                else:
                    raise Exception(f'sn_cutoff "{sn_cutoff}" not found.')

                ii = slice(zonecut, None)
                s = a.project(zones = ii)
                m = dict(
                    mass      = mass,
                    metallicity = 10**metallicity * Z0,
                    data      = s,
                    remnant   = d.zm_sun[zonecut - 1],
                    energy    = 1.2,
                    mixing    = 0.1,
                    time      = p.time / physconst.SEC,
                    sn_cutoff = sn_cutoff,
                    he_core   = p.core()['He core'].zm_sun,
                    co_core    = p.core()['C/O core'].zm_sun,
                    iron_core = p.core()['iron core'].zm_sun,
                    )
                if add_x0 is True:
                    # this is due to current network limitations in KEPLER
                    x0 = AbuSet(bg_file = bgfile)
                    x0.abu[isotope.ufunc_Z(x0.iso) > 83] = 0
                    m['x0'] = x0
                models += [m]
        self.models = models
        self.close_logger('Loading data finished in')

    def make_stardb(self,
                    filename = None,
                    silent   = False,
                    mode     = 'isotopes',
                    db_class = StarDB,
                    series   = 'GG',
                    output   = 'Travaglio',
                    ):
        self.setup_logger(silent = silent)
        # to be efficient, put all models on same isotope basis.
        self.logger.info('Creating isotope list.')
        ii = set()
        for d in self.models:
            ii |= set(d['data'].isotopes())
        self.logger.info('Making compatible AbuData')
        a = AbuSet(list(ii), np.zeros((len(ii),)), sorted = True)
        data = np.ndarray((len(self.models), len(a)), dtype = np.float64)
        metadata = np.ndarray((len(self.models),), dtype = object)
        for i,d in enumerate(self.models):
            y = d.copy()
            x = y.pop('data') + a
            data[i, :] = x.Y()
            metadata[i] = y

        sn_cutoff = self.models[0]['sn_cutoff']
        abu = AbuData(data, a.iso.copy(), molfrac = True)
        self.logger.info('Mapping')

        if mode == 'elements':
            mpar = dict(
                stable = True,
                elements = True,
                )
            dpar = dict(
                abundance_type   = StarDB.AbundanceType.element,
                abundance_class  = StarDB.AbundanceClass.dec,
                )
        elif mode == 'isotopes':
            mpar = dict(
                stable = True,
                elements = False,
                )
            dpar = dict(
                abundance_type   = StarDB.AbundanceType.isotope,
                abundance_class  = StarDB.AbundanceClass.dec,
                )
        elif mode == 'radioactive':
            mpar = dict(
                stable = False,
                elements = False,
                )
            dpar = dict(
                abundance_type   = StarDB.AbundanceType.isotope,
                abundance_class  = StarDB.AbundanceClass.rad,
                )

        mapped = Decay.Map(abu, **mpar)

        fields  = ['metallicity', 'mass', 'energy', 'mixing', 'remnant', 'time']

        if output == 'Travaglio':
            fields += ['he_core', 'co_core', 'iron_core']

        # could match these automatically from field type
        types   = [np.float64] * len(fields)
        units   = ['absolute', 'solar masses', 'B', 'He core fraction', 'M_sun', 'yr']
        formats = ['8.2G', '2.0F', '3.1F', '3.1F', '6.3F', '9.3E']
        flags   = ([StarDB.Flags.parameter] * 4) + ([StarDB.Flags.property] * (len(fields) - 4))

        if output == 'Travaglio':
            units += ['M_sun'] * 3
            formats += ['6.3F'] * 3

        fielddata = np.ndarray(
            mapped.data.shape[0],
            dtype = {'names': fields,
                     'formats': types,
                     }
            )

        if 'x0' in self.models[0]:
            x0 = []
        for i,d in enumerate(metadata):
            for f in fields:
                fielddata[i][f] = d[f]
            try:
                x0 += [d['x0']]
            except:
                pass

        self.logger.info('Creating DB')
        db = db_class(
            data = mapped,
            fielddata = fielddata,
            fieldunits = units,
            fieldformats = formats,
            fieldflags = flags,
            name = '{} West+ 2015+, SN cutoff is {}'.format(series, sn_cutoff),
            comments = ['model grid based on West+ 2013 GCE model',
                        'Pop III models from Heger+ 2010',
                        ],
            abundance_unit   = StarDB.AbundanceUnit.mol_fraction,
            abundance_total  = StarDB.AbundanceTotal.ejecta,
            abundance_norm   = None,
            abundance_data   = StarDB.AbundanceData.all_ejecta,
            abundance_sum    = StarDB.AbundanceSum.number_fraction,
            **dpar
            )

        self.close_logger('Creating StarDB finished in')

        if filename is not None:
            db.write(filename)
        if 'x0' in self.models[0]:
            return db, x0
        return db


    def get_fallback(self, filename):
        with open(filename) as f:
            lines = f.read()
        zonecut = int(re.findall(r'c #  zone cut: (\d+)\s', lines)[0])
        masscut = float(re.findall(r'c #  mass cut: ([-+e\.0-9]+)\s', lines)[0])
        # self.logger.info('mass cut {} (zone {})'.format(masscut, zonecut))

        return zonecut

class TravaglioStarDB(StarDB):
    def write_ascii(self, filename):
        filename = os.path.expanduser(filename)
        if os.path.exists(filename):
            os.remove(filename)
        self.setup_logger(silent = False,
                     logfile = filename)
        self.print_info()
        self.close_logger()
        with open(filename, 'at+') as f:
            f.write('\nData:\nejected solar masses\n')
            f.write(
                (' '.join(['{:>12s}'] * len(self.fieldnames))).format(*self.fieldnames) +
                (' '.join(['{!s:>12s}'] * len(self.ions))).format(*self.ions) +
                '\n'
                )
            for d, a in zip(self.fielddata, self.data):
                m = a * (d['mass'] - d['remnant']) * isotope.ufunc_A(self.ions)
                f.write(
                    (' '.join(['{:>12.5e}'] * len(d))).format(*d) +
                    (' '.join(['{:>12.5e}'] * len(m))).format(*m) +
                    '\n'
                    )

class NuGridStarDB(StarDB):
    def __init__(self,
                 db = None,
                 x0 = None,
                 ):
        super().__init__(db = db)
        self.x0 = x0

    def write_ascii_header_line(self, s):
        self._fout.write('H {:s}\n'.format(s))

    def write_ascii(self,
                    filename = None,
                    compress = True,
                    ):
        self.iZ = np.where(self.fieldnames == 'metallicity')[0][0]
        self.iM = np.where(self.fieldnames == 'mass')[0][0]
        self.iT = np.where(self.fieldnames == 'time')[0][0]
        self.iR = np.where(self.fieldnames == 'remnant')[0][0]

        with TextFile(filename,
                      mode = 'w',
                      compress = compress,
                      return_filename = True) as (fout, fout_filename):
            self._fout = fout
            self.write_ascii_file_header()
            for index in range(self.nstar):
                self.write_ascii_model_header(index)
                self.write_ascii_model_data(index)

    def write_ascii_file_header(self):
        self.write_ascii_header_line('Yield Set: {}'.format(self.name))
        self.write_ascii_header_line('Data prepared by: {}'.format(os.getlogin()))
        self.write_ascii_header_line('Data prepared date: {}'.format(time.strftime('%d %b %Y', time.gmtime())))
        self.write_ascii_header_line('Isotopes: {}'.format(', '.join([i.NuGrid() for i in self.ions])))
        self.write_ascii_header_line('Number of metallicities: {:d}'.format(self.nvalues[self.iZ]))
        self.write_ascii_header_line('Units: Msun, year, erg')

    def write_ascii_model_header(self, index):
        self.write_ascii_header_line('Table: (M={:g},Z={:g})'.format(
            self.fielddata[index][self.iM],
            self.fielddata[index][self.iZ],
            ))
        self.write_ascii_header_line('Lifetime: {:9.3E}'.format(
            self.fielddata[index][self.iT],
            ))
        self.write_ascii_header_line('Mfinal: {:9.3E}'.format(
            self.fielddata[index][self.iR],
            ))

    def write_ascii_model_data(self, index):
        self._fout.write('&Isotopes &Yields    &X0        &Z &A  \n')
        for i,ion in enumerate(self.ions):
            try:
                x0 = self.x0[index].X(ion)
            except:
                x0 = 0.
            self._fout.write('&{:<9s}&{:<10.3e}&{:<10.3e}&{:<2d}&{:<3d}\n'.format(
                ion.NuGrid(),
                self.data[index, i] * ion.A * (self.fielddata[index][self.iM] - self.fielddata[index][self.iR]),
                x0,
                ion.Z,
                ion.A,
                ))

    def _write_other(self):
        # raise NotImplementedError()
        x = np.ndarray((self.nstar, self.nabu), dtype = np.float64)
        for index in range(self.nstar):
            for i,ion in enumerate(self.ions):
                try:
                    x0 = self.x0[index].X(ion)
                except:
                    x0 = 0.
                x[index, i] = x0
        self._write_dbl(x.transpose())

    def _read_other(self):
        # raise NotImplementedError()
        x0 = self._read_dbl((self.nabu, self.nstar))
        x0 = np.ascontiguousarray(x0.transpose())
        self.x0 = [AbuSet(iso = self.ions,
                          abu = x0[index]) for index in range(self.nstar)]

def make_NuGrid(mode = None, nugrid=False):
    dbparm = dict()
    if mode is not None:
        dbparm['mode'] = mode
    cases = {
        'GG':
        dict(version = '4',
             series = 'sin'
             ),
        # 'SS':
        # dict(version = '',
        #      series = 'sca'
        #      ),
        }

    if mode == 'Travaglio':
        path = os.path.expanduser('~/NuGrid/GCE_Travaglio')
    else:
        path = os.path.expanduser('~/NuGrid/GCE')
    sne = {
        # 'no_cutoff': None,
        # 'xi25': 'xi2.5=0.25',
        # 'xi45': 'xi2.5=0.45',
        # 'ertl': 'ertl_s19.8',
         'mu16': 'muller',
        #'mc16': 'muller_cutoff',
        }
    for cname, case in cases.items():
        for name, sn in sne.items():
            bname = os.path.join(path, cname + '-' + name)
            if mode is not None:
                bname += '-' + mode
            if mode is not 'Travaglio':
                g = Grid(add_x0 = True, sn_cutoff = sn, **case)
                dbparm['series'] = cname
                d, x0 = g.make_stardb(**dbparm)
                d.write(bname + '.stardb.xz')
                if nugrid:
                    n = NuGridStarDB(d, x0)
                    n.write(bname + '.nugrid.stardb.xz')
                    n.write_ascii(bname  + '.txt')
            else:
                g = Grid(sn_cutoff = sn, **case)
                dbparm['series'] = cname
                d, x0 = g.make_stardb(**dbparm)
                d.write(bname + '.travaglio.stardb.xz')
                t = TravaglioStarDB(db = d)
                filename = bname + '.travaglio.txt'
                c.write_ascii(filename)
                with open(filename, 'rb') as f, lzma.open(filename+'.xz', 'wb') as g:
                    g.write(f.read())
                os.remove(filename)
