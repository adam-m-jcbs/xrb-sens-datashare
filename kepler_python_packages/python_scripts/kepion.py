"""
Collection of routines for KEPLER APPROX/NSE/ISO isotopes
"""

import string
import re
import os
import io
import copy
import numpy as np
#from matplotlib.cbook import is_numlike
from collections import Iterable
import time
import textwrap
import os.path
from abc import abstractmethod
import datetime

import physconst

from logged import Logged
from utils import ClearCache, CachedAttribute, cachedmethod, project, OutFile, stuple, iterable, np_array, is_iterable
from human import version2human
from isotope import Ion, ion, register_other_bits, NEUTRON, PROTON, ALPHA, VOID, ufunc_A, ufunc_Z
from abuset import AbuSet, AbuData

class KepIon(Ion):
    """
    Special Ions for KEPLER APPOX/NSE/QSE networks.

    TODO:
        need to overwrite add etc. to make sure nothing bad happens ...?
    """

    # define KepIon bits
    F_OTHER_KEPION = 2
    F_KEPION = Ion.F_OTHER * F_OTHER_KEPION

    IONLIST={  #   B         F   Z   A   E
        'nt1'  : ( 1, F_KEPION,  0,  1,  0),
        'h1'   : ( 2, F_KEPION,  1,  1,  0),
        'pn1'  : ( 3, F_KEPION,  1,  1,  0),
        'he3'  : ( 4, F_KEPION,  2,  3,  0),
        'he4'  : ( 5, F_KEPION,  2,  4,  0),
        'c12'  : ( 6, F_KEPION,  6, 12,  0),
        'n14'  : ( 7, F_KEPION,  7, 14,  0),
        'o16'  : ( 8, F_KEPION,  8, 16,  0),
        'ne20' : ( 9, F_KEPION, 10, 20,  0),
        'mg24' : (10, F_KEPION, 12, 24,  0),
        'si28' : (11, F_KEPION, 14, 28,  0),
        's32'  : (12, F_KEPION, 16, 32,  0),
        'ar36' : (13, F_KEPION, 18, 36,  0),
        'ca40' : (14, F_KEPION, 20, 40,  0),
        'ti44' : (15, F_KEPION, 22, 44,  0),
        'cr48' : (16, F_KEPION, 24, 48,  0),
        'fe52' : (17, F_KEPION, 26, 52,  0),
        'fe54' : (18, F_KEPION, 26, 54,  0),
        'ni56' : (19, F_KEPION, 28, 56,  0),
        'ye'   : (30, F_KEPION,  0,  0,  0),
        'yq'   : (31, F_KEPION,  0,  0,  0),
        'eb0'  : (32, F_KEPION,  0,  0,  0),
        'yf'   : (33, F_KEPION,  0,  0,  0),
        'fe56' : (34, F_KEPION, 26, 56,  0),
        "'fe'" : (35, F_KEPION, 26, 56,  0)}

    BE = {
        'nt1'  :   0.,
        'h1'   :   0.,
        'pn1'  :   0.,
        'he3'  :   7.71819,
        'he4'  :  28.29603,
        'c12'  :  92.16294,
        'n14'  : 104.65998,
        'o16'  : 127.62093,
        'ne20' : 160.64788,
        'mg24' : 198.2579,
        'si28' : 236.5379,
        's32'  : 271.7825,
        'ar36' : 306.7202,
        'ca40' : 342.0568,
        'ti44' : 375.4772,
        'cr48' : 411.469,
        'fe52' : 447.708,
        'fe54' : 471.7696,
        'ni56' : 484.003,
        'ye'   :   0.,
        'yq'   :   0.,
        'eb0'  :   0.,
        'yf'   :   0.,
        'fe56' : 492.2539,
        "'fe'" : 492.2539,
        }

    ME = {
        'nt1'  :  +8.0714165,
        'h1'   :  +7.2890735,
        'pn1'  :  +7.2890735,
        'he3'  : +14.9313735,
        'he4'  :  +2.42495,
        'c12'  :   0.,
        'n14'  :  +2.86345,
        'o16'  :  -4.73701,
        'ne20' :  -7.04298,
        'mg24' : -13.93202,
        'si28' : -21.49104,
        's32'  : -26.01466,
        'ar36' : -30.23138,
        'ca40' : -34.847,
        'ti44' : -37.54642,
        'cr48' : -42.81724,
        'fe52' : -48.33526,
        'fe54' : -56.254027,
        'ni56' : -53.90928,
        'ye'   :  +0.782343,
        'yq'   :   0.,
        'eb0'  :   0.,
        'yf'   :   0.,
        'fe56' : -60.595494,
        "'fe'" : -60.595494,
        }

    nseqse_ion_names = [
        "nt1", "ye", "pn1", "yq", "he4", "yf", "eb0", "o16",
        "fe56", "mg24", "si28", "s32", "ar36", "ca40", "ti44",
        "cr48", "'fe'", "fe54", "ni56"]


    burn_ion_names = [
        'h1', 'he3', 'he4', 'n14', 'c12', 'o16', 'ne20', 'mg24',
        'si28', 's32', 'ar36', 'ca40', 'ti44', 'cr48', 'fe52',
        'ni56', 'fe54', 'pn1', 'nt1']

    # approx_ion_names = ['pn1', 'h1', 'nt1', 'he3', 'he4', 'c12', 'n14', 'o16',
    #                     'ne20', 'mg24', 'si28', 's32', 'ar36', 'ca40', 'ti44',
    #                     'cr48', 'fe52', 'fe54', 'ni56']

    approx_ion_names = [name
                        for name,i in sorted(IONLIST.items(),
                                             key = lambda x: x[1][0])
                        if i[0] <= 20]

    ion_names = [name
                 for name,ion in sorted(IONLIST.items(),
                                        key = lambda x: x[1][0])
                 if not ion[0] in {3, 30, 31, 32, 33}]

    network_names = [ 'APPROX', 'NSE', 'QSE' ]

    def networks(self):
        networks = []
        if self.name() in self.approx_ion_names:
            networks += [ 1 ]
        if self.name() in self.nseqse_ion_names:
            networks += [ 2, 3 ]
        return networks

    @classmethod
    def network_name(cls, network):
        return cls.network_names[network - 1]

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            ix = args[0]
            if isinstance(ix, Ion) and ix != self.VOID:
                for n,t in self.IONLIST.items():
                    if (t[2] == ix.Z and
                        t[3] == ix.A and
                        t[4] == ix.E and
                        ((ix.F == self.F_KEPION and ix.B == t[0]) or
                         ix.F != self.F_KEPION)):
                        name = n
                        super(KepIon, self).__init__(name)
                        return
                # super(KepIon, self).__init__(self.VOID)
                super().__init__(self.VOID)
                return
        super().__init__(*args, **kwargs)

    @classmethod
    def ion2bfzae(cls, s='', element = None, isomer = None):
        s = s.strip()
        if s.startswith('ion'):
            s=s[3:]
        return cls.IONLIST.get(s.lower(), cls.VOID)

    @classmethod
    def ionn2bfzae(cls, ionn):
        for ix in cls.IONLIST.values():
            if ix[0] == ionn:
                return ix
        return cls.VOID

    @classmethod
    def from_ionn(cls, ionn):
        ix = cls.ionn2bfzae(ionn)
        if ix != cls.VOID:
            return KepIon(
                B = ix[0],
                F = ix[1],
                Z = ix[2],
                A = ix[3],
                E = ix[4])
        return KepIon(cls.VOID)

    @classmethod
    def nuclei(cls):
        # should be replaced by a set
        return np.array([KepIon(n) for n,t in cls.IONLIST.items() if t[3] > 0])

    def ion(self):
        return ion(A = self.A, Z = self.Z)

    ufunc_ion = np.frompyfunc(lambda x: x.ion(), 1, 1)

    @classmethod
    def from_ion(cls, ion):
        if not isinstance(ion, Ion):
            raise TypeError()
        return KepIon(ion.name())

    def _name(self, upcase = True):
        t  = self.tuple()
        for k,v in self.IONLIST.items():
            if v == t:
                return k
        return self.VOID_STRING

    LaTeX_names = {
         1 : '^{  }\mathrm{n }',
         2 : '^{ 1}\mathrm{H }',
         3 : '^{  }\mathrm{p }',
         4 : '^{ 3}\mathrm{He}',
         5 : '^{ 4}\mathrm{He}',
         6 : '^{12}\mathrm{C }',
         7 : '^{14}\mathrm{N }',
         8 : '^{16}\mathrm{O }',
         9 : '^{20}\mathrm{Ne}',
        10 : '^{24}\mathrm{Mg}',
        11 : '^{28}\mathrm{Si}',
        12 : '^{32}\mathrm{S }',
        13 : '^{36}\mathrm{Ar}',
        14 : '^{40}\mathrm{Ca}',
        15 : '^{44}\mathrm{Ti}',
        16 : '^{48}\mathrm{Cr}',
        17 : '^{52}\mathrm{Fe}',
        18 : '^{54}\mathrm{Fe}',
        19 : '^{56}\mathrm{Ni}',
        30 : '^{  }\mathrm{- }',
        31 : '^{  }\mathrm{- }',
        32 : '^{  }\mathrm{- }',
        33 : '^{  }\mathrm{- }',
        34 : '^{56}\mathrm{Fe}',
        35 : '^{}\mathrm{\'Fe\'}',
        }

    @CachedAttribute
    def mpl(self):
        return '${}$'.format(self.LaTeX_names[self.B])


        #needs to be updated
        #return self.ion().mpl()

    def ionn(self):
        return self.B

KepIon.factory = KepIon

register_other_bits(KepIon)


class KepAbu(object):
    """
    Single KepIon with abundance.
    """
    pass

class KepAbuSet(ClearCache, Logged):
    """
    Single Zone Kepler abudance.
    """
    def __init__(self,
                 iso = None,
                 abu = None,
                 comment = None,
                 mixture = None,
                 silent = False,
                 sentinel = None,
                 **kwargs):
        """
        Initialize KEPLER abundance set.

        Currently allow call with
         * list of isotopes only (abu set to 0)
         * list of isotopes and list of abundances
         * keyword iso=abu
         * dictionary {'iso':abu, ...}
         * abuset

        TODO - should always be sorted?
        TODO - should this be subclass of AbuSet?
               would need to define isotope class to use (KepIon)
        """

        if isinstance(iso, AbuSet) and abu is None:
            iso, abu = abu, iso
        if isinstance(abu, AbuSet):
            approx_map = MapBurn(abu.iso)
            if mixture is None:
                try:
                    mixture = abu.mixture
                except AttributeError:
                    mixture = 'x'
            comment = stuple(abu.comment, comment)
            abu = approx_map(abu.abu)
            iso = approx_map.decions
        self.comment = stuple(comment)
        self.mixture = mixture
        self.sentinel = sentinel
        if isinstance(iso, dict):
            self.iso, self.abu = self._ion_abu_from_dict(**iso)
        elif iso is None:
            assert abu is None, "Need isotope name"
            self.iso = np.array([], dtype=np.object)
            self.abu = np.array([], dtype=np.float64)
        else:
            self.iso = np.array([KepIon(i) for i in np.atleast_1d(iso)], dtype=np.object)
            if abu is not None:
                self.abu = np.array(np.atleast_1d(abu), dtype=np.float64)
                assert len(self.abu) == len(self.iso), "Need equal number of elements."
            else:
                self.abu = np.zeros_like(self.iso, dtype=np.float64)
        if len(kwargs) > 0:
            self._append(*self._ion_abu_from_dict(**kwargs))

    @staticmethod
    def _ion_abu_from_dict(**kwargs):
        #todo: add sorting?
        return (
            np.array([KepIon(i) for i in kwargs.keys()],
                     dtype=np.object),
            np.array(list(kwargs.values()),
                     dtype=np.float64))

    def _append(self, iso, abu):
        # sorting?
        self.iso = np.append(self.iso, iso)
        self.abu = np.append(self.abu, abu)

    def __str__(self):
        return ("ion(" +
                ", ".join(['{:s}: {:f}'\
                           .format(iso.Name(),abu)
                           for iso, abu in zip(self.iso,self.abu)]) +
                ")")
    __repr__ = __str__

    def iteritems(self): # <-- TODO: replace by "items" and return "view"
        """
        Return pairs of (isotope, value).
        """
        if self.iso.size == 0:
            raise StopIteration
        for iso, abu in zip(self.iso, self.abu):
            yield (iso, abu)

    def metallicity(self):
        """
        Return 'metallicity' Z of composition.
        """
        z = sum([abu for abu,iso in zip(self.abu,self.iso) if iso.Z >= 3])
        return z

    def mu(self):
        """
        Return mean molecular weight of composition.
        """
        xmu = sum([abu/iso.A*(iso.Z+1) for abu,iso in zip(self.abu,self.iso)])
        return 1./xmu

    def normalize(self, total = None):
        """
        Normalize abundances to one.

        If sum == 0 just return.

        Note: Clone of AbuSet.normalize()
        """
        abusum = self.abu.sum()
        if abusum == 0.:
            return
        self.abu /= abusum
        if total is not None:
            self.abu *= total

    # maybe this routine is not needed ... ?
    def write_approx(self,
                     outfile,
                     comment = None,
                     mixture = None,
                     overwrite = False,
                     silent = False):
        """
        Write APPROX abundances generator to file.

        If outfile is file use this.
        If outfile is a filename open outfile for writing.
        """
        self.setup_logger(silent = silent)
        # this code seems to repeat
        with OutFile(outfile) as f:
            # # TODO - test utils.OutFile context manager
            # if not isinstance(outfile, io.IOBase):
            #     filename = os.path.expanduser(os.path.expandvars(outfile))
            #     assert overwrite or not os.path.exists(filename)
            #     f = open(filename,'w')
            # else:
            #     f = outfile
            if not mixture:
                mixture = self.mixture
            if mixture is None:
                mixture = 'comp'
            for c in stuple(comment, self.comment):
                f.write('c {:s}\n'.format(c))
            for iso, abu in zip(self.iso, self.abu):
                f.write('m {:s} {:25.17E} {:s}\n'.format(
                    mixture,
                    abu,
                    iso.name()))

        # if not isinstance(outfile, io.IOBase)
        #     f.close()
        self.close_logger(timing='BURN generator written to "{:s}" in'.format(f.name))

def _ions_from_beyond(ionn):
    """
    Return list of ions for all networks
    """
    ions = np.ndarray(ionn.shape, dtype = np.object)
    nflat = ionn.reshape(-1)
    iflat = ions.reshape(-1)
    for i in range(len(nflat)):
        iflat[i] = KepIon.from_ionn(nflat[i])
    return ions

def _calc_oburn_map():
    """
    Return APPROX19 mapping matrix for o_burn.
    """
    burn_map = np.zeros((19,19))
    xx = [KepIon('si28'), KepIon('s32')]
    xf = [0.6, 0.4]
    for i,x in enumerate(KepIon.approx_ion_names):
        x = KepIon(x)
        if x.A < 28:
            for y,f in zip(xx, xf):
                burn_map[i, y.B-1] = x.A/y.A*f
        else:
            burn_map[i, i] = 1
    return burn_map


def _calc_approx_map():
    """
    Return APPROX19 mapping matrix for o_burn.

    From KEPLER:
    in order to use new ise network must also zero yc12 and yne
    and pack all iron isotopes more neutron rich than fe54 in fe52
    """
    burn_map = np.zeros((19, 19))
    burn_map[np.array([8, 16]), 16] = 56 / 52
    diag = np.array([0, 2, 4, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18])
    burn_map[diag, diag] = 1
    return burn_map

def _calc_nseqse_data():
    """
    compute maps for NSE/QSE quantities
    """
    ions = np.array([KepIon(x) for x in KepIon.nseqse_ion_names])
    Z = np.array([x.Z if x.A > 0 else 0 for x in ions])
    EB = np.array([KepIon.BE[x] for x in KepIon.nseqse_ion_names])
    return Z, EB

class KepAbuData(ClearCache):
    """
    Kepler abudance set.  For use with, e.g., cnv.

    Helpful to convert ppn data with different networks as a function
    of zone into usable data.
    """

    ionn = np.array(
        [[ 1,  1,  1],
         [ 2, 30, 30],
         [ 3,  3,  3],
         [ 4, 31, 31],
         [ 5,  5,  5],
         [ 6, 33, 33],
         [ 7, 32, 32],
         [ 8,  8,  8],
         [ 9, 34, 34],
         [10, 10, 10],
         [11, 11, 11],
         [12, 12, 12],
         [13, 13, 13],
         [14, 14, 14],
         [15, 15, 15],
         [16, 16, 16],
         [17, 35, 35],
         [18, 18, 18],
         [19, 19, 19],
         [ 0,  0,  0],
         ])

    BE_APPROX = np.array([KepIon.BE[x] for x in KepIon.approx_ion_names])
    ME_APPROX = np.array([KepIon.ME[x] for x in KepIon.approx_ion_names])

    APPROX_IONS = np.array([KepIon(x)  for x in KepIon.approx_ion_names])

    ions = _ions_from_beyond(ionn)

    _oburn_map  = _calc_oburn_map()
    _approx_map = _calc_approx_map()
    _nseqse_Z, _nseqse_EB = _calc_nseqse_data()

    def __init__(self, ppn, netnum = None, molfrac = None):
        """
        Initialize with data and try to deduce missing info.

        Assume [20,:] layout
        """
        assert len(ppn.shape) == 2
        assert ppn.shape[1] == 20

        self.ppn = ppn.copy()
        self.n = ppn.shape[0]
        if netnum is not None:
            if len(netnum) == 1:
                netnum = np.tile(netnum, self.n)
            netnum[:] = np.minimum(netnum, 3)
        else:
            i, = np.where(np.logical_or(ppn[:,6] > 1., ppn[:,2] > 0.1))
            netnum = np.tile(1, self.n)
            netnum[i] = 2
        self.netnum = netnum
        if molfrac is None:
            ii, = np.where(netnum == 1)
            if len(ii) > 0:
                x = np.sum(ppn[0,ii[0]])
                if netnum[0] > 1:
                    x -= ppn[0,6]
                molfrac =  x < 0.9
            else:
                # assume NSE/QSE nets
                molfrac = True
        self.molfrac = molfrac
        self.i0 = 0
        self.i1 = self.n

    # @cachedmethod - it is not clear what is wrong with this ... !
    def ion_abu(self, ion, molfrac = False, missing = np.nan):
        """
        Return ion or kepion abundace, depending on type.

        Sort of the cheap version of ionmap for APPROX/QSE/NSE.
        """
        if not isinstance(ion, KepIon):
            kepion = KepIon(ion)
            if kepion == Ion.VOID:
                value = np.ndarray(self.ppn.shape[0], dtype = np.float64)
                value.fill(missing)
                return value
            if kepion.B == 2:
                # pn1 seems to be defined in all three netwoks, but h1 not
                value  = self.ion_abu(kepion, molfrac = molfrac, missing = 0. )
                value += self.ion_abu(KepIon.from_ionn(3), molfrac = molfrac, missing = missing)
                return value
            return self.ion_abu(kepion, molfrac = molfrac, missing = missing)
        yfac = max(1, ion.A)
        if molfrac == self.molfrac:
            yfac = 1
        if molfrac and not self.molfrac:
            yfac = 1 / yfac
        value = np.ndarray(self.ppn.shape[0], dtype = np.float64)
        value.fill(missing)
        netlist = np.argwhere(ion.ionn() == self.ionn)
        for inet in range(netlist.shape[0]):
            zonelist = np.argwhere((netlist[inet,1] + 1) == self.netnum[self.i0:self.i1])
            if len(zonelist) > 0:
                value[zonelist + self.i0] = self.ppn[zonelist + self.i0, netlist[inet,0]] * yfac
        return value

    def _extract(self, ext_func, molfrac = False, missing = np.nan):
        """
        extract abundace combinations based on extfunc that filters for valid ion indices
        """
        value = np.ndarray(self.ppn.shape[0], dtype = np.float64)
        value[:] = missing
        for netnum in range(self.ionn.shape[1]):
            ion_index = ext_func(self.ions[:,netnum])
            zone_index = np.where(self.netnum[self.i0:self.i1] == (netnum + 1))[0] + 1
            if molfrac:
                yfac = np.ones(len(ion_index))
            else:
                yfac = ufunc_A(self.ions[ion_index,netnum])
            if len(zone_index) > 0:
                value[zone_index] = np.tensordot(
                    self.ppn[np.ix_(zone_index,ion_index)], yfac, axes=(1,0))
        value[:self.i0] = missing
        value[self.i1:] = missing
        return value

    @cachedmethod
    def iron(self, **kw):
        """
        Return ion mass fraction (A > 46).
        """
        def ext_func(ions):
            return np.where(ufunc_A(ions) > 46)[0]
        return self._extract(ext_func, **kw)

    @cachedmethod
    def metallicity(self, **kw):
        """
        Return mass fraction sum of A >= 5.
        """
        def ext_func(ions):
            return np.where(ufunc_Z(ions) > 2)[0]
        return self._extract(ext_func, **kw)

    @cachedmethod
    def hydrogen(self, **kw):
        """
        Return sum of mass fraction sum of A = 1,2 (nt1, h1, pn1).
        """
        def ext_func(ions):
            return np.where(np.in1d(ufunc_A(ions), [1,2]))[0]
        return self._extract(ext_func, **kw)

    @cachedmethod
    def helium(self, **kw):
        """
        Return sum of mass fraction sum of A = 3,4 (h3, he3, he4).
        """
        def ext_func(ions):
            return np.where(np.in1d(ufunc_A(ions), [3,4]))[0]
        return self._extract(ext_func, **kw)

    @cachedmethod
    def XYZ(self, **kw):
        """
        Return 'astronomical' X, Y, Z of composition.
        """
        return np.vstack([
            self.hydrogen(**kw),
            self.helium(**kw),
            self.metallicity(**kw),
            ]).transpose()

    @cachedmethod
    def __getitem__(self, ion):
        return self.ion_abu(ion)

    @cachedmethod
    def __call__(self, *args, **kw):
        return self.ion_abu(*args, **kw)

    def __getattr__(self, attr):
        try:
            return self.ion_abu(attr)
        except:
            pass
        return super().__getattribute__(attr)

    def ions_netnum(self, netnum):
        """
        Return the ion list for a given netnum
        """
        assert 0 < netnum <= self.ionn.shape[1]
        ions = self.ions[:,netnum-1]
        massive = []
        for i in ions:
            if i.A > 0:
                massive += [i]
        return np.array(massive)

    @CachedAttribute
    def netnum_ranges(self):
        """
        Return a list of network numbers and ranges, including wind.
        """
        ranges = []
        netnum = self.netnum
        ii, =  np.where(netnum[1:] != netnum[:-1])
        ii = [0] + (ii+1).tolist() + [len(netnum)]
        for i in range(len(ii)-1):
            n = netnum[ii[i]]
            if n > 0:
                ranges.append((n, np.array([ii[i], ii[i + 1]-1])))
        return ranges

    @CachedAttribute
    def BE(self):
        """
        Return nuclear binding energy in MeV per nucleon.

        Note that the value is positive for more tightly bound nuclei.
        """
        BE = np.tile(np.nan, self.n)
        if self.molfrac:
            norm = 1
        else:
            norm = 1/np.array([x.A for x in self.APPROX_IONS])
        for n, r in self.netnum_ranges:
            ii = slice(r[0], r[1]+1)
            if n == 1:
                BE[ii] = np.dot(self.ppn[ii,:19], self.BE_APPROX * norm)
            else:
                BE[ii] = self.ppn[ii, 6] # location of eb0
        return BE

    @CachedAttribute
    def BE_cgs(self):
        """
        Return nuclear binding energy in erg/g.

        Note that the value is positive for more tightly bound nuclei.
        """
        return self.BE * physconst.NA * physconst.MEV

    @CachedAttribute
    def ME(self):
        """
        Return nuclear mass excess in MeV per nucleon.

        Note that the value is negative for more tightly bound nuclei.
        """
        ME = np.tile(np.nan, self.n)
        if self.molfrac:
            norm = 1
        else:
            norm = 1/np.array([x.A for x in self.APPROX_IONS])
        for n, r in self.netnum_ranges:
            ii = slice(r[0], r[1]+1)
            if n == 1:
                ME[ii] = np.dot(self.ppn[ii,0:19], self.ME_APPROX * norm)
            else:
                ME[ii] = (
                    - self.ppn[ii, 6] # BE/nucleon
                    + KepIon.BE['c12'] / 12  # BE of C12 reference (per nucleon)
                    + KepIon.ME['ye'] * (self.ppn[ii, 1] - 0.5) # ME from n-(p+e)-mass difference
                    )
        return  ME

    @CachedAttribute
    def ME_cgs(self):
        """
        Return nuclear mass excess in erg/g.

        Note that the value is negative for more tightly bound nuclei.
        """
        return self.ME * physconst.NA * physconst.MEV

    def approx(self, zones = None, _clear_cache = True):
        """
        Convert zones to APPROX19 network.

        Currently only converts Networks 2 and 3, which are equal, to
        Network 1.
        """
        if zones is None:
            zones = np.arange(self.i0, self.i1)
        zones = np_array(zones)
        jj = zones[np.in1d(self.netnum[zones], [2,3])]
        self.netnum[jj] = 1
        self.ppn[jj, :] = np.tensordot(
            self.ppn[jj, :],
            self._approx_map,
            axes=[1, 0])
        if _clear_cache:
            self.clear_cache()

    invalid_index = np.iinfo(np.int64).min

    def ion_index(self, ion, net = None):
        """
        Return index of ion in network(s).

        If no network number is specified, return array for all network.

        If array of networks is specified, return array of indices.

        Indices are 0-based, and self.invalid_index =
        np.iinfo(np.int64).min = -9223372036854775808 is used for
        invalid indices.
        """
        if not isinstance(ion, KepIon):
            i = KepIon(ion)
        if net is None:
            net = np.arange(self.ions.shape[1]) + 1
        if is_iterable(net):
            idx = []
            for n in net:
                ii = np.where(self.ions[:, n - 1] == ion)
                if len(ii[0]) == 0:
                    idx += [self.invalid_index]
                else:
                    idx += [ii[0][0]]
            return np.array(idx)
        else:
            ii = np.where(self.ions[:, net - 1] == ion)
            if len(ii[0]) == 0:
                return self.invalid_index
            else:
                return ii[0][0]


    def si_burn(self, zones = None):
        """
        Convert all stuff to Ni56, APPROX19 network.
        """
        # self.set_abu(zones,
        #              netnum = 1,
        #              abu = {'ni56' : 1.0},
        #              normalize = False,
        #              )

        netnum = 1
        idx_ni56 = self.ion_index('ni56', netnum)
        if zones is None:
            zones = np.arange(self.i0, self.i1)
        zones = np_array(zones)
        self.netnum[zones] = netnum
        self.ppn[zones, :] = 0
        self.ppn[zones, idx_ni56] = 1 / 56
        self.clear_cache()

    def alpha_burn(self, zones = None):
        """
        Convert all stuff to alphas, APPROX19 network.
        """
        # self.set_abu(zones,
        #              netnum = 1,
        #              abu = {'he4' : 1.0},
        #              normalize = False,
        #              )

        netnum = 1
        i_he4 = self.ion_index('he4', netnum)
        if zones is None:
            zones = np.arange(self.i0, self.i1)
        zones = np_array(zones)
        self.netnum[zones] = netnum
        self.ppn[zones, :] = 0
        self.ppn[zones, i_he4] = 1 / 4
        self.clear_cache()

    def o_burn(self, zones = None):
        """
        Convert all stuff lighter than Si to Si/S mixture.
        """
        if zones is None:
            zones = np.arange(self.i0, self.i1)
        zones = np_array(zones)
        self.approx(zones, _clear_cache = False)
        self.ppn[zones, :] = np.tensordot(
            self.ppn[zones, :],
            self._oburn_map,
            axes=[1, 0])
        self.clear_cache()


    def _nseqse_update(self, zones = None, clear_cache = True):
        """
        update ye and eb0 in NSE/QSE networks.

        Not clear how to fix yf and yq at this time.
        """
        if zones is None:
            zones = np.arange(self.i0, self.i1)
        zones = np_array(zones)

        assert np.alltrue(np.in1d(self.netnum[zones], [2,3]))
        assert np.alltrue(self.netnum[zones] == self.netnum[zones[0]])
        netnum = self.netnum[zones[0]]
        idx_ye = self.ion_index('ye', netnum)
        self.ppn[zones, idx_ye] = np.sum(self.ppn[zones] *
                                         self._nseqse_Z[np.newaxis,:],
                                         axis = 1)
        idx_eb0 = self.ion_index('eb0', netnum)
        self.ppn[zones, idx_eb0] = np.sum(self.ppn[zones] *
                                          self._nseqse_EB[np.newaxis,:],
                                          axis = 1)
        if clear_cache:
            self.clear_cache()


    # TODO - if we change NSE/QSE: update eb0, yf ???
    def set_abu(self,
                zones = None,
                abu = None,
                netnum = None,
                clear = True,
                normalize = True,
                molfrac = False,
                ):
        """
        Set abundace of zones.  Assume input is mass fraction.
        """
        if zones is None:
            zones = np.arange(self.i0, self.i1)
        zones = np_array(zones)
        assert isinstance(abu, dict)
        if clear:
            self.ppn[zones, :] = 0
        if netnum is not None:
            self.netnum[zones] = netnum
        for i,a in list(abu.items()):
            if not isinstance(i, KepIon):
                del abu[i]
                i = KepIon(i)
                abu[i] = a
            if not molfrac:
                A = i.A
                if A > 0:
                    abu[i] /= A
        for n in np.unique(self.netnum[zones]):
            jj = zones[np.where(self.netnum[zones] == n)[0]]
            for i,a in abu.items():
                ii = np.argwhere(self.ionn[:, n - 1] == i.B)
                assert len(ii) > 0, '"{}" is not in network {}.'.format(
                    i, n)
                self.ppn[jj, ii[0,0]] = a
            if normalize:
                A = ufunc_A(self.ions[:self.ppn.shape[1], n - 1])
                tot = np.sum(self.ppn[jj, :] * A[np.newaxis, :], axis = 1)
                assert np.allclose(tot, 1., atol = 1.e-3), 'abundance not normalized'
                self.ppn[jj, :] /= tot[:, np.newaxis]
            if n in [2,3]:
                self._nseqse_update(jj, clear_cache = False)
        self.clear_cache()


class KepAbuDump(KepAbuData):
    """
    Kepler abudance set.  For use with ppn from KEPLER dumps.

    Helpful to convert ppn data with different networks as a function
    of zone into usable data.
    """
    def __init__(self, ppn, netnum,
                 ionn = None,
                 wind = None,
                 molfrac = True,
                 xm = None,
                 zm = None,
                 copy = False,
                 ):
        """
        Initialize with KEPLER network information.
        """

        self.ppn = ppn.transpose()
        self.netnum = netnum
        if self. ionn is not None:
            self.ionn = ionn
            self.ions = _ions_from_beyond(ionn)
        self.xm = xm
        self.zm = zm
        if copy:
            self.netnum = self.netnum.copy()
            self.ppn = self.ppn.copy()
            if self.xm is not None:
                self.xm = self.xm.copy()
            if self.zm is not None:
                self.zm = self.zm.copy()
        self.n = len(zm)
        self.molfrac = molfrac
        self.i0 = 1
        self.i1 = self.ppn.shape[0] - 1
        if wind is not None:
            self.ppn[-1,:] = wind.reshape((5,-1))[netnum[-1]-1,:self.ppn.shape[1]]
            self.i1 += 1

    @cachedmethod
    def xE(self, func = None, zones = None):
        """
        Return burning energy for zones.

        call as, e.g.,

        >>> xE(func='o_burn')
        """
        if zones is None:
            zones = np.arange(0, self.n)
        zones = np_array(zones)
        ppn = self.ppn
        self.ppn = ppn.copy()
        netnum = self.netnum
        self.netnum = netnum.copy()
        E0 = self.ME_cgs
        if func is not None:
            getattr(self, func)()
        E = ((E0 - self.ME_cgs) * self.xm)[zones]
        self.ppn = ppn
        self.netnum = netnum
        self.clear_cache()
        return E

    @cachedmethod
    def xNi56(self, func = None, zones = None):
        """
        Return Ni56 burning yields for zones.

        call as, e.g.,

        >>> xNi56(func='o_burn')
        """
        if zones is None:
            zones = np.arange(0, self.n)
        zones = np_array(zones)
        ppn = self.ppn
        self.ppn = ppn.copy()
        netnum = self.netnum
        self.netnum = netnum.copy()
        if func is not None:
            getattr(self, func)()
        Ni56 = (self.ion_abu('ni56') * self.xm)[zones]
        self.ppn = ppn
        self.netnum = netnum
        self.clear_cache()
        return Ni56


    @cachedmethod
    def xENi56(self, func = None, zones = None):
        """
        Return burning energy and Ni56 burning yields for zones.

        call as, e.g.,

        >>> xENi56(func='o_burn')
        """
        if zones is None:
            zones = np.arange(0, self.n)
        zones = np_array(zones)
        ppn = self.ppn
        self.ppn = ppn.copy()
        netnum = self.netnum
        self.netnum = netnum.copy()
        E0 = self.ME_cgs
        if func is not None:
            getattr(self, func)()
        E = ((E0 - self.ME_cgs) * self.xm)[zones]
        Ni56 = (self.ion_abu('ni56') * self.xm)[zones]
        self.ppn = ppn
        self.netnum = netnum
        self.clear_cache()
        return E, Ni56


class MapBurn(object):
    """
    Map isotope distribution to APPROX-19 network

    ideally this should become similar to Decay, maybe inherit from common abstract base class?
    """

    approx19 = ('nt1' ,
                'h1'  ,
                'pn1' ,
                'he3' ,
                'he4' ,
                'c12' ,
                'n14' ,
                'o16' ,
                'ne20',
                'mg24',
                'si28',
                's32' ,
                'ar36',
                'ca40',
                'ti44',
                'cr48',
                'fe52',
                'fe54',
                'ni56')

    approx19_ions = [ion(i) for i in approx19]

    def __init__(self, ions):
        """
        Create mapping matrix.

        MAY NEED TO ADD mass/mol mapping.
        Currently mass only.
        """
        self.decmatrix = np.zeros((len(ions),19),
                                 dtype=np.float64)
        self.decions = np.array([KepIon(i) for i in self.approx19])
        self.ions = ions

        for i,ix in enumerate(ions):
            if ix == NEUTRON:
                self.decmatrix[i,0] = 1.
            elif ix == PROTON:
                self.decmatrix[i,1] = 1.
            elif ix == ALPHA:
                self.decmatrix[i,4] = 1.
            elif ix == self.approx19_ions[18]:
                self.decmatrix[i,18] = 1.
            elif ix.Z < 6:
                self.decmatrix[i,3] = 1.
            elif ix.Z < 7:
                self.decmatrix[i,5] = 1.
            elif ix.Z < 8:
                self.decmatrix[i,6] = 1.
            elif ix.Z < 10:
                self.decmatrix[i,7] = 1.
            elif ix.Z < 12:
                self.decmatrix[i,8] = 1.
            elif ix.Z < 14:
                self.decmatrix[i,9] = 1.
            elif ix.Z < 16:
                self.decmatrix[i,10] = 1.
            elif ix.Z < 18:
                self.decmatrix[i,11] = 1.
            elif ix.Z < 20:
                self.decmatrix[i,12] = 1.
            elif ix.Z < 22:
                self.decmatrix[i,13] = 1.
            elif ix.Z < 24:
                self.decmatrix[i,14] = 1.
            elif ix.Z < 26:
                self.decmatrix[i,15] = 1.
            elif (ix.Z < 26) and (ix.A < 54):
                self.decmatrix[i,16] = 1.
            else:
                self.decmatrix[i,17] = 1.

    def __call__(self, abu):
        """
        Map a data set and return result.

        Add checks?
        """
        return np.dot(abu, self.decmatrix)

def mapburn(abub, kep = False):
    assert isinstance(abub, AbuData)
    assert abub.molfrac == False
    mapper = MapBurn(abub.ions)
    mapped = mapper(abub.data)
    if kep:
        mapped = np.hstack((mapped, np.zeros((mapped.shape[0], 1))))
        return KepAbuData(mapped, molfrac = abub.molfrac)
    return AbuData(
        data = mapped,
        ions = mapper.decions,
        molfrac = abub.molfrac,
        )
