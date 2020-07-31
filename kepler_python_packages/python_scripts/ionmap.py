"""
 the following is the final plan; this is not implemented as of 20100425.
-----------------------------------------------------------------------
            Treat as     Keep in    Add to      compatible
            stable       output     EL/A sum    with solabu
            (no decays)

 ionlist    DEPENDS      YES        DEPENDS     CHECK
 stabions   YES          YES        YES         IONS: NO; EL/A:CHECK
 addions    NO           YES        NO          NO
 keepions   YES          YES        NO          NO
 orgions    NO           YES        NO          CHECK
-----------------------------------------------------------------------
It is not clear this list does not duplicate things between
addions/keepions/orgions

"""

import os
import os.path
import datetime
import inspect
import itertools
from scipy.linalg import expm

from collections import defaultdict
from copy import copy, deepcopy

import numpy as np

from human import byte2human
from human import time2human
from human import version2human

from isotope import Ion, ion, VOID, PROTON, NEUTRON, ALPHA, isomermap
from isotope import ufunc_A, ufunc_Z, ufunc_N
from isotope import ufunc_idx, ufunc_isotope_idx
from isotope import ufunc_is_isomer, ufunc_is_isobar, ufunc_is_isotone, ufunc_is_element


from reaction import DecIon
from abuset import IonList, AbuData, IonSet, AbuSet
from abusets import SolAbu
from utils import contract, project, iterable
from utils import CachedAttribute, cachedmethod
from logged import Logged

# from physconst import *

class IonMap(Logged):
    """
    base class for ion mapping
    """
    def __init__(self, ions, decions, ionmap,
                 molfrac_in = None,
                 molfrac_out = None,
                 silent = False):
        """
        create new map object
        """
        # update to handle other classes

        self.setup_logger(silent)

        # convert input data
        ions, molfrac_in, molfrac_out = self.input2ionarr(
            ions, molfrac_in, molfrac_out)

        # convert outut data
        decions, molfrac_out = self.output2ionarr(
            decions, molfrac_out)

        self.set_molfrac(molfrac_in, molfrac_out)

        self.ions = ions
        self.decions = decions

        if ionmap is not None:
            assert ionmap.shape == (len(ion), len(decions))
        self.map = ionmap

        self.close_logger()

    def set_molfrac(self, molfrac_in, molfrac_out):
        if molfrac_in is None and molfrac_out is not None:
            molfrac_in = molfrac_out
        if molfrac_out is None and molfrac_in is not None:
            molfrac_out = molfrac_in
        if molfrac_in == molfrac_out == None:
            molfrac_in = molfrac_out = False
        self.molfrac = [molfrac_in, molfrac_out]

    @staticmethod
    def input2ionarr(ions, molfrac_in = None, molfrac_out = None):
        """
        extract ions from input data

        # TODO add IonMap
        """
        if isinstance(ions, AbuData):
            molfrac_in = ions.molfrac
            ions = ions.ions
        elif isinstance(ions, AbuSet):
            assert molfrac_in in (None, False)
            assert molfrac_out in (None, False)
            molfrac_in = False
            molfrac_out = False
            ions = ions.iso
        if isinstance(ions, IonList):
            ions = np.array(ions)
        ions = np.atleast_1d(ions)
        if not isinstance(ions[0], Ion):
            ions = np.array([ion(ix) for ix in ions])
        return ions, molfrac_in, molfrac_out

    @staticmethod
    def output2ionarr(ions, molfrac_out = None):
        """
        extract ions from input data

        # TODO add IonMap
        """
        if isinstance(ions, AbuData):
            molfrac_out = ions.molfrac
            ions = ions.ions
        elif isinstance(ions, AbuSet):
            assert molfrac_out in (None, False)
            molfrac_out = False
            ions = ions.iso
        if isinstance(ions, IonList):
            ions = np.array(ions)
        ions = np.atleast_1d(ions)
        if isinstance(ions[0], str):
            ions = np.array([ion(ix) for ix in ions])
        return ions, molfrac_out


    def __add__(self, other):
        """
        Add 2 Ion mappings
        """
        assert self.map.shape[1] == other.map.shape[1]
        assert np.all(self.ions == other.ions)
        assert np.all(self.molfrac == other.molfrac)

        return IonMap(
            self.ions,
            np.hstack(self.decions, other.decions),
            np.hstack(self.map, other.map),
            self.molfrac)

    def __mul__(self, other):
        """
        Add multiplication of matrices or call, depending on type
        """
        if isinstance(other, IonMap):
            assert np.all(self.decions == other.ions)
            assert self.molfrac[1] == other.molfrac[0]

            return IonMap(
                self.ions,
                other.decions,
                np.dot(self.map, other.map),
                [self.molfrac[0], other.molfrac[1]])
        elif isinstance(other, (AbuData, AbuSet)):
            return self(other)
        raise AttributeError()

    def __rmul__(self, other):
        """
        Add call transform
        """
        if isinstance(other, (AbuData, AbuSet)):
            return self(other)
        raise AttributeError()

    def __call__(self, data,
                 ions = None,
                 molfrac = None,
                 silent = False):
        """
        Map a data set and return result.

        Add checks!
        Eat AbuData & AbuSet - return same?
        """
        self.setup_logger(silent)

        # checks
        out_type = type(None)
        if isinstance(data, AbuData):
            assert ions is None
            assert molfrac is None or molfrac == data.molfrac
            ions = data.ions
            molfrac = data.molfrac
            out_type = type(data)
            org = data
            # update after adding conversion routine to basic AbuData
            assert data.molfrac == self.molfrac[0]
            data = data.data
        elif isinstance(data, AbuSet):
            assert ions is None
            assert molfrac is None or molfrac == data.molfrac
            ions = data.iso
            # AbuSet should be updated to (allow) use molfrac
            molfrac = data.molfrac
            out_type = type(data)
            org = data
            data = data.abu
            assert molfrac == self.molfrac[0] == self.molfrac[1]
        if isinstance(ions, IonList):
            ions = ions.ions()
        if ions is not None:
            assert np.all(ions == self.ions)
        if molfrac is not None:
            assert molfrac == self.molfrac[0]

        result =  np.dot(data, self.map)

        if issubclass(out_type, AbuData):
            result = org.updated(result, self.decions, self.molfrac[1])
        elif issubclass(out_type, AbuSet):
            for itype in inspect.getmro(out_type):
                try:
                    result = itype(abu = result, iso = self.decions)
                    break
                except TypeError:
                    pass
            else:
                raise AttributeError('Could not mtach type.')
        elif not out_type == type(None):
            raise AttributeError('Unkonw input data type')

        self.close_logger(timing = 'mapping computed in')
        return result

    @classmethod
    def Map(cls, ions = None, **kw):
        assert isinstance(ions, (AbuData, AbuSet))
        silent = kw.get('silent', False)
        return cls(ions = ions, **kw)(ions, silent = silent)

    def molfrac_convert(self, decmatrix):

        if not self.molfrac[1]:
            f_out = ufunc_A(self.decions)
            decmatrix *= f_out[np.newaxis,:]
        if not self.molfrac[0]:
            f_in  = 1/ufunc_A(self.ions)
            decmatrix *= f_in[:,np.newaxis]
        return decmatrix

def assert_isomer(ionlist):
    # set local map if necessary
    imap = isomermap
    ions = np.array([i.isomer(E = imap) for i in ionlist])
    return ions

def assert_isotope(ionlist):
    ions = copy(ionlist)
    for i, ion in enumerate(ions):
        if ion.is_isomer():
            if ion.E == 0:
                ions[i] = ions.isotope()
            elif ion == 'ta180m':
                ions[i] = ions.isotope()
            else:
                ions[i] = None
    assert len(np.unique(ions)) == len(ions)
    return ions

# convenience function
def decay(*args, **kwargs):
    return Decay.Map(*args, **kwargs)

class Decay(IonMap):
    """
    Object to compute decays/mappings for a given set of ions.

    Initialize the mapping with input ion list atnd desired output
    format.  Then call as a function (__call__ method) to do the
    mapping.  *decions* contains the list of output ions.  Currently
    np.array of isotope.Ions.

    What is not implememented at this time are to pass isobar,
    element, or isotones as output in ionlist etc.
    """

    def __init__(self,
                 ions = None,
                 molfrac_in = None,
                 molfrac_out = None,
                 decay = True,
                 isobars = False,
                 elements = False,
                 isotones = False,
                 isotopes = True,
                 solprod = False,
                 stable = False,
                 solions = False,
                 sort = True,
                 ionlist = None,
                 addions = None,
                 keepions = None,
                 stabions = None,
                 orgions = None,
                 solabu = None,
                 keepall = True,
                 decayfile = '~/kepler/local_data/decay.dat',
                 isomers = None,
                 silent = False,
                 debug = False,
                 ):
        """
        Initialize decay map.

        ions - list of input ions.
               Currently np.array of isotope.Ion.
               Probably should allow (or requre?) composition instead?
               Allow any string, array of strings, ...
               Same for all ion lists.

        decay - whether to do decay
                if disabled and no other option is used,
                the matrix will just do sorting, if specified

        molfrac_in - whether input is mass or abundance.
        molfrac_out - whether output is mass or abundance.
        TODO - add molfrac, used for both

        isobars - return result mapped to isobars   [only stable or undecayed isotopes]
        elements - return result mapped to elements [only stable or undecayed isotopes]
        isotones - return result mapped to isotones [only stable or undecayed isotopes]
        isotopes - return result mapped to isotopes - if processing isomers

        ADD - isotopes - return result mapped to isotopes; isomers otherwise
        ADD - decay - do decay or not

        solabu - solar abundance set to use: filename, token, or isotope.SolAbu
        solprod - return result in terms of solar production factor [only stable isotopes]
        solions - return exactly ions in solar abundace set [plus addions, stabions, keepions]

        ionlist - output *exactly* these isotops

        stable - output only stable isotopes, discard chains
        addions - output these extra ions, but not add to EL/A sums
        keepions - ions to keep as stable but not add ion EL/A sums

        stabions - ions to fully treat as stable - may conflict with solprod

        orgions - ions for which to return the initial value w/o decays

        sort - sort output ions (Z > A > E), mostly relevant when
               customized ions are provided.

        keepall - if set to false, do not keep all ions even if stable set is used.

        decayfile - decay file to use. [should add allowing to pass object]

        isomers - process as isomers if input is isotopes

        silent - whether to be verbose or not
        """
        self.setup_logger(silent)

#        ions = np.array([ion('pb220')])
#        ions = np.array([ion('n13'), ion('o14'), ion('pb209')])
#        ions = np.array([ion('n13'), ion('o14')])

        ions, molfrac_in, molfrac_out = self.input2ionarr(
            ions, molfrac_in, molfrac_out)

        if isomers is None:
            isomers = np.any(ufunc_is_isomer(ions))

        if not isomers and np.any(ufunc_is_isomer(ions)):
            new_ions = assert_isotope(ions)
            assert len(np.unique(ufunc_idx(new_ions))) == len(ions), 'Failed to project ions uniquely'
            ions = new_ions

        if isomers and not np.all(ufunc_is_isomer(ions)):
            ions = assert_isomer(ions)

        # all need to be the same.  TODO - more sophisticated
        assert np.all(ufunc_is_isomer(ions) == ions[0].is_isomer())

        if molfrac_in is None:
            molfrac_in = False
        if molfrac_out is None:
            molfrac_out = molfrac_in

        assert isinstance(ions, np.ndarray), "ions need to be np.array type"
        assert ions.ndim == 1, "ions need to be 1D array"
        assert isinstance(ions[0], Ion), "ions must be of type Ion"

        # find out whether we need list of stable ions
        need_stable = stable

        # TODO - check ionlist should exclude isobars, isotopes, ...
        need_isotopes = isotopes
        need_isobars  = isobars
        need_isotones = isotones
        need_elements = elements

        # add things from ionlist
        if ionlist:
            need_isotopes |= np.any(ufunc_is_isotope(ionlist))
            need_isobars  |= np.any(ufunc_is_isobar(ionlist))
            need_isotones |= np.any(ufunc_is_isotone(ionlist))
            need_elements |= np.any(ufunc_is_element(ionlist))
        if addions:
            need_isotopes |= np.any(ufunc_is_isotope(addions))
            need_isobars  |= np.any(ufunc_is_isobar(addions))
            need_isotones |= np.any(ufunc_is_isotone(addions))
            need_elements |= np.any(ufunc_is_element(addions))

        need_orgions = orgions or not decay

            ### [WORK HERE]

        # THIS IS WRONG - ONLY IF DECAY
        # if need_elements or need_isobars or need_isotones or (need_isotopes and isomers):
        #     need_stable = decay
        if need_elements or need_isobars or need_isotones:
            need_stable = decay

        self.ions = ions
        self.amax = np.max(ufunc_A(ions))
        self.zmax = np.max(ufunc_Z(ions))

        # check keepions
        if keepions is not None:
            if isinstance(keepions, IonSet):
                keepions = np.array(keepions)
            keepions = np.atleast_1d(keepions)
            assert isinstance(keepions, np.ndarray), "keepions need to be np.array type"
            assert keepions.ndim == 1, "keepions need to be 1D array"
            assert issubclass(type(keepions[0]),Ion), "keepions must be of type Ion"

            self.amax = max(self.amax, np.max(ufunc_A(keepions)))
            self.zmax = max(self.zmax, np.max(ufunc_Z(keepions)))

        # check stabions
        if stabions is not None:
            if isinstance(stabions, IonSet):
                stabions = np.array(stabions)
            stabions = np.atleast_1d(stabions)
            assert isinstance(stabions, np.ndarray), "stabions need to be np.array type"
            assert stabions.ndim == 1, "stabions need to be 1D array"
            assert issubclass(type(stabions[0]),Ion), "stabions must be of type Ion"

            self.amax = max(self.amax, np.max(ufunc_A(stabions)))
            self.zmax = max(self.zmax, np.max(ufunc_Z(stabions)))

        # check ionlist
        if ionlist is not None:
            if not isinstance(ionlist, np.ndarray):
                ionlist = np.array(ionlist)
            ionlist = np.atleast_1d(sn.squeeze(ionlist))
            assert ionlist.ndim == 1, "ionlist need to be 1D array"
            assert issubclass(type(ionlist[0]),Ion), "ionlist must be of type Ion"

            self.amax = max(self.amax, np.max(ufunc_A(ionlist)))
            self.zmax = max(self.zmax, np.max(ufunc_Z(ionlist)))

        # generate decay data
        self.decdata = DecayData(
            filename = decayfile,
            amax = self.amax,
            zmax = self.zmax,
            isomers = isomers,
            silent = silent,
            debug = debug,
            )
        # self.decdata = DecayData(decayfile, isomers = isomers, silent = silent)

        # now let's add keepions - not sure about this one - probably wrong
        if keepions is not None:
            # the strategy is to modify the decdata
            for ix in keepions:
                self.decdata.add_stable(ix)

        # ...and stabions
        if stabions is not None:
            # the strategy is modify the decdata
            for ix in stabions:
                self.decdata.add_stable(ix)

        # construct table from ions
        self.dectable = {}
        if decay:
            d0 = [self.iter_add_dectable(ix) for ix in self.ions]
        else:
            d0 = [self.identity_dectable(ix) for ix in self.ions]

        # let us just use indices into the array to speed things up
        self.d = np.ndarray(len(self.dectable), dtype = object)
        for dec in self.dectable.values():
            self.d[dec[0][1]] = dec

        self.decions = np.array([ix for ix in self.dectable.keys()], dtype = object)
        self.indices = np.array([dec[0][1] for dec in self.dectable.values()], dtype = np.int64)

        # construct decay matrix - this should be its separate method!!!
        nions = len(ions)
        ndec = len(self.dectable)
        self.decmatrix = np.zeros([nions,ndec], dtype=np.float64)

        # compute needed decay matrix
        #
        # currently this is fixed when isomers are mapped to isotopes,
        # but this loses the 'maxradio' info for isomers.
        #
        # Instead, in the future, a full square decay matrix needs to
        # be constractued that can invert the isomer submatrix for
        # isotopes, and similar for all cases, releasing the
        # requirement for stable or raw isotopes only.
        #
        # the extra isotopes should be added after a first full decay
        # attempt.
        #
        # not sure this will ever be useful other than for isotopes.
        #
        if need_isotopes and not stable and decay and isomers:
            self.revind = np.argsort(self.indices)
            for i,ix in enumerate(self.ions):
                self.iter_add_decmatrix_iso(i, np.float64(1), d0[i])
        else:
            for i,ix in enumerate(self.ions):
                self.iter_add_decmatrix(i, np.float64(1), d0[i])

        m = self.decions.argsort()
        self.decions = self.decions[m]
        self.decmatrix = self.decmatrix[:,self.indices[m]]

        self.molfrac = [molfrac_in, molfrac_out]

        self.molfrac_convert(self.decmatrix)

        # we need to check whether we need solar abundances.
        need_solabu = solions or solprod
        # since we can get the list of stable ions from the decay table,
        # we do not really need this for definition of stable isotopes.
        if need_solabu:
            if isinstance(solabu, str):
                solabu = SolAbu(solabu)
            elif solabu is None:
                solabu = SolAbu()
            assert isinstance(solabu, SolAbu), "Need solar abundace data."
            # let us assure solar abundance pattern is sorted
            assert not np.any(solabu.iso.argsort() - np.arange(len(solabu))), "Solar pattern not sorted."

            # now we construct the map to solar, smap, and the map
            # from solabu to decions, rmap
            k = 0
            smap = []
            rmap = []
            for i,iso in enumerate(solabu.iso):
                while self.decions[k] < iso:
                    k += 1
                    if k == len(self.decions):
                        break
                if k == len(self.decions):
                    break
                if self.decions[k] == iso:
                    smap.append(k)
                    rmap.append(i)

        # why this ... not need_solabu ???
        # why not have need_solabu inply need_stable?
        if need_stable and not need_solabu:
            # here we are overwring smap from above!!!
            # WHY???
            # DELETE ???
            smap = [i for i,ix in enumerate(self.decions) if len(self.dectable[ix]) == 1]


        # before we do any of the following, we still need to make
        # sure to only include stable isotopes.
        # proably best to keep old array and construct new one piece
        # by piece.

        if need_stable or need_solabu:
            stable_decions = self.decions[smap]
            stable_decmatrix = self.decmatrix[:,smap]

        # add missing solar ions
        # ??? UPDATE to include stab...
        if solions and len(solabu) > len(stable_decions):
            ndec = len(solabu)
            d = np.zeros([nions,ndec], dtype=np.float64)
            d[:,rmap] = stable_decmatrix[:,:]
            stable_decions = solabu.iso
            stable_decmatrix = d

        ### add missing ions from ion list
        # ....
        # can ion list be elements, isobars, isotones?
        # yes!
        #
        # 1) the stuff below needs to be termined which to compute
        # 2) store computed values separately - not right in decions
        # 3) then select the ecessary ones for final decions

        # maybe able to construct decay/isotop map?
        if need_isotopes and not stable and decay and isomers:
            # construct a decay matrix that does not double-count.
            # The key is here a function that finds entries with
            # identical projected values
            decidx = ufunc_isotope_idx(self.decions)
            idx = np.unique(
                decidx,
                )
            decmatrix = self.decmatrix.copy()
            # but how to discard double counts???
            # ??? mat coeff in range, -I, subtract entire column (mat mult)
            # submatrix = np.zeros([len(self.decions)]*2)
            for i in idx:
                ii = np.where(i == decidx)[0]
                if len(ii) > 1:
                    deciso = self.decions[ii]
                    # do we need to assume isomeric states are ordered, or that order matters?
                    ions = []
                    iions = []
                    for jj,ix in enumerate(deciso):
                        j = np.where(self.ions == ix)[0]
                        if len(j) == 1:
                            ions += [j[0]]
                            iions += [ii[jj]]
                        elif len(j) > 1:
                            raise Exception('something is wrong here')
                    print(self.decmatrix[np.ix_(ions,ii)].transpose())

                    # next: sort by chain?
                    # find things that depend on each other, in order
                    # then subtract in order

                    # maybe subtract where things depend on self?

                    # OK, for this to work we need to ensure all ions
                    # on 'out' channels are also on 'in' channel so
                    # the matrix can be 'inverted'.

                    assert len(ii) == len(ions), 'matrix cannot be inverted'

                    # currently, further up, we create a differnt
                    # matrix for this case that does not require this
                    # correction.  In practice, we want to have both
                    # options, the 'maxradio' for all sub-sets and
                    # supersets.



            raise NotImplementedError()

        # *** temporary fix:
        if need_stable:
            raw_ions = stable_decions
            raw_matrix = stable_decmatrix
        else:
            raw_ions = self.decions
            raw_matrix = self.decmatrix


        if need_elements:
            decionz = ufunc_Z(raw_ions)
            elements_decmatrix, z = project(raw_matrix, decionz, return_values = True, axis = -1)
            elements_decions = np.array([ion(Z = Z) for Z in z])
        if need_isobars:
            deciona = ufunc_A(raw_ions)
            isobars_decmatrix, a = project(raw_matrix, deciona, return_values = True, axis = -1)
            isobars_decions = np.array([ion(A = A) for A in a])
        if need_isotones:
            decionn = ufunc_N(raw_ions)
            isotones_decmatrix, n = project(raw_matrix, decionn, return_values = True, axis = -1)
            isotones_decions = np.array([ion(N = N) for N in n])

        # *** likely, this should not use raw_ions but self.decions
        if need_isotopes:
            deciiso = ufunc_isotope_idx(raw_ions)
            isotopes_decmatrix, i = project(raw_matrix, deciiso, return_values = True, axis = -1)
            isotopes_decions = np.array([ion(idx = I) for I in i])

        if need_orgions:
            orgions_decmatrix = np.identity(len(self.ions))
            orgions_decions = self.ions


        if solprod:
            # TODO
            raise NotImplementedError('solprod')


        # compile final decions set; make it an IonList
        if isobars:
            decmatrix = isobars_decmatrix
            decions = isobars_decions
        elif isotones:
            decmatrix = isotones_decmatrix
            decions = isotones_decions
        elif elements:
            decmatrix = elements_decmatrix
            decions = elements_decions
        elif isotopes:
            decmatrix = isotopes_decmatrix
            decions = isotopes_decions
        elif ionlist is not None:
            # TODO
            raise NotImplementedError('ionlist')
        else:
            decmatrix = self.decmatrix
            decions = self.decions

        # clean up
        self.decions = decions
        self.map = decmatrix
        del self.decmatrix

        self.close_logger(timing = 'matrix constructed in')

    def iter_add_decmatrix_iso(self, i, weight, j, chain = None):
        ii = self.ions[i]
        ij = self.decions[self.revind[j]]
        ok = True
        if chain is not None:
            for ic in chain:
                if ij.isotope() == ic.isotope():
                    ok = False
                    break
            chain += [ij]
        else:
            chain = [ij]
        if (ii == ij or ok):
            print(ii,ij, weight, chain)
            self.decmatrix[i,j] += weight
        for b, jj in self.d[j][1:]:
            w = b * weight
            self.iter_add_decmatrix_iso(i, w, jj, chain)

    def iter_add_decmatrix(self, i, weight, j):
        self.decmatrix[i,j] += weight
        for b, jj in self.d[j][1:]:
            w = b * weight
            self.iter_add_decmatrix(i, w, jj)

    def iter_add_dectable(self, ix):
        if ix not in self.dectable:
            idx0 = len(self.dectable)
            declist = [(np.float64(1), idx0)]
            self.dectable[ix] = declist
            for b,*d in self.decdata(ix):
                for D in d:
                    idx = self.iter_add_dectable(D)
                    declist.append((b, idx))
            return idx0
        return self.dectable[ix][0][1]

    def identity_dectable(self, ix):
        idx0 = len(self.dectable)
        declist = [(np.float64(1), idx0)]
        self.dectable[ix] = declist
        return idx0

    # some analysis utility routines
    def parents(self, ix):
        """
        return dictionary of parent nuclei (including self) and their weights
        """
        ix = ion(ix)
        ii = np.where(ix == self.decions)[0][0]
        col = self.map[:,ii]
        ii = np.where(col > 0)[0]
        p = {}
        for i in ii:
            p[self.ions[i]] = col[i]
        return p

    def daughters(self, ix):
        """
        return dictionary of daughter nuclei (including self) and their weights
        """
        ix = ion(ix)
        ii = np.where(ix == self.ions)[0][0]
        col = self.map[ii,:]
        ii = np.where(col > 0)[0]
        p = {}
        for i in ii:
            p[self.decions[i]] = col[i]
        return p


# one change we may want to do is to make a copy of the raw data
# before we modify the decay with "add_stable"

# There should be variants that work for different decay data
# maybe one could, e.g., be generated from reaclib?
# One form current bdat would not be complete due to special decays.
class DecayData(Logged):
    """
    Class to interface with decay.dat file.
    """
    def __init__(self,
                 filename = None,
                 amax = 999,
                 zmax = 999,
                 isomers = False,
                 silent = False,
                 debug = True,
                 nuclei = True,
                 ):
        """
        Load decay file.

        format
          c12 [BR] [decay | output]+

        examples

          c9 1 p b8m3
          b8m3 g
          b8m2 b- c8
          c8 EC b8m1
          b8m1 g
          b8 b+ be8
          be8 2a
          he4
          h1


        """

        self.setup_logger(silent)
        self.isomers = isomers

        self.decayfile = os.path.expandvars(os.path.expanduser(filename))


        decdata = {}
        with open(self.decayfile,'rt') as f:
            self.logger.info('Loading {} ({})'
                  .format(f.name,
                          byte2human(os.fstat(f.fileno()).st_size)))
            for s in f:
                if debug:
                    print('\nprocessing: ' + s.strip())
                if s.startswith(';'):
                    continue
                if len(s.strip()) == 0:
                    continue
                reac = s.split()

                ix = ion(reac[0], isomer = isomers)
                if debug:
                    print('ion = ', ix)
                if ix.A > amax and ix.Z > zmax:
                    continue
                if isomers:
                    ix = assert_isomer([ix])[0]
                else:
                    ix = assert_isotope([ix])[0]
                    if ix is None:
                        continue
                if debug:
                    print('ion = ', ix)

                assert ix != Ion.VOID, "Ion name not valid: " + str(ix)
                i = 1
                if len(reac) > i:
                    try:
                        br = np.float64(reac[i])
                        i += 1
                    except ValueError:
                        br = np.float64(1.)
                else:
                    br = np.float64(1.)
                # we shall accept "empty" as stable....
                decays = []
                ions = []
                for r in reac[i:]:
                    try:
                        d = DecIon(r)
                    except:
                        d = VOID
                    if d == VOID:
                        jx = ion(r, isomer = isomers)
                        ions += [jx]
                    else:
                        decays += [d]
                if len(reac) == 1:
                    decays += [DecIon('s')]
                for dec in decays:
                    for p,n in dec.particles().items():
                        if n < 0:
                            p = -p
                            n = -n
                        ions = [p] * n + ions
                for jj,jx in enumerate(ions):
                    if jx.is_nucleus():
                        if isomers:
                            jx = assert_isomer([jx])[0]
                        else:
                            jx = jx.isotope()
                        ions[jj] = jx
                Z,A,E = ix.ZAE()
                for i in ions:
                     Z -= i.Z
                     A -= i.A
                     E -= i.E
                if not (A == 0 and Z == 0):
                    assert Z <= A, 'something is wrong'
                    if isomers:
                        if not (ix.A == A and ix.Z == Z):
                            E = 0
                    else:
                        E = None
                    out = ion(Z=Z, A=A, E=E)
                    if not out == ix:
                        ions += [out]

                # filter for nucleons
                if nuclei:
                    decions = [i for i in ions if i.is_nucleus()]
                else:
                    decions = ions

                decinfo = [br] + decions

                # store info
                if ix not in decdata:
                    decdata[ix] = []
                decdata[ix].append(tuple(decinfo))

        # check BRs and ion lists
        decions = set()
        for ix, products in decdata.items():
            BR = np.sum([x[0] for x in products])
            assert BR == 1, "BR don't add up {!s:}: {}".format(ix, BR)
            for x in products:
                for i in x[1:]:
                    decions |= {i}
        missing = decions - set(decdata)
        if len(missing) != 0:
            s = "missing ions from decdata: {}".format(', '.join([str(i) for i in missing]))
            if debug:
                self.logger.critical(s)
            else:
                self.logger.debug(s)

        self.decdata = decdata
        self.close_logger(timing = 'data loaded in')

    def get_decay(self, ix):
        """
        Return decay table entry.

        If the decay is not in stored data, extrapolate from last
        available decay at same element (Z value).

        Use gs decays only for extrapolation and assume unspecified
        excited states do single 'g' decay.

        TODO: should we store new entries?
        """
        try:
            return self.decdata[ix]
        except:
            if ix.is_isomer():
                if ix.E > 0:
                    return [(1., ix.isomer(E = ix.E - 1))]
            Z = ix.Z
            a = np.array([ix.A for ix in self.decdata.keys() if ix.Z == Z and ix.E == 0])
            assert len(a) > 0, "Problem finding isotopes for decay."
            assert not (min(a) < ix.A < max(a)), 'decay chain has gaps, no unique extrapolation possible'
            A = min(max(a), max(min(a), ix.A))
            refion = ion(Z = Z,
                         A = A,
                         isomer = ix.is_isomer())
            try:
                refdec = self.decdata[refion]
            except KeyError:
                raise Exception('Decay Extrapolation: Could not determine reference nucleus')
            dec = copy(refdec)
            dA = ix.A - refion.A
            for i, branch in enumerate(refdec):
                br = branch[0]
                products = list(branch[1:])
                # stable
                if len(products) == 0:
                    assert len(refdec) == 0, 'can only be stable or not'
                    continue
                #Q: use last or most heavy nucleus?
                #A: let's check last one is most heavy
                ap = ufunc_A(products)
                maxa = np.where(ap == np.max(ap))[0]
                assert len(maxa) == 1, 'cannot determine decay nucleus'
                # or I could just use the most heavy nucleus
                idec = maxa[0]
                refdecion = products[idec]
                newdecion = ion(Z = refdecion.Z,
                                A = refdecion.A + dA,
                                isomer = ix.is_isomer())
                products[idec] = newdecion
                dec[i] = tuple([br] + products)
            return dec

    def __call__(self, ix):
        return self.get_decay(ix)

    def __getitem__(self, ix):
        return self.get_decay(ix)

    def add_stable(self, ix):
        """
        Flag an ion as stable, extending decays from edge.

        Assume all isotopes are either isomers or isotopes.
        Extra gap fillers will be assumed to be isomers in gs (if isomeric)
        """
        Z = ix.Z
        A = ix.A
        E = ix.E
        a = np.array([jx.A for jx in self.decdata.keys() if jx.Z == Z and jx.E == E])
        if len(a) > 0:
            amin = min(a)
            amax = max(a)
            if self.isomers:
                E = 0
            else:
                E = None
            for a in range(A+1, amin):
                newion = ion(Z = Z, A = a, E = E)
                self.decdata[newion] = self.get_decay(newion)
            for a in range(amax+1, A):
                newion = ion(Z = Z, A = a, E = E)
                self.decdata[newion] = self.get_decay(newion)
        # finally add/overwrite "stable"
        self.decdata[ix] = [(np.float64(1),)]


# t-dependent decay routine(s)

class DecayRate(object):
    """ hold decays with fixed rate """
    def __init__(self, i_in, i_out, rate):
        i_in = list(iterable(i_in))
        i_out = list(iterable(i_out))
        for j,i in enumerate(i_in):
            if isinstance(i, str):
                i_in[j] = ion(i)
        for j,i in enumerate(i_out):
            if isinstance(i, str):
                i_out[j] = ion(i)
        self.i_in = i_in
        self.i_out = i_out
        self.rate = rate

    @CachedAttribute
    def ions(self):
        return set(self.i_in + self.i_out)

    @CachedAttribute
    def ions_in(self):
        return set(self.i_in)

    @CachedAttribute
    def ions_out(self):
        return set(self.i_out)

    def __str__(self):
        return self._name()

    def _name(self):
        s = '{} ==> {}: {}'.format(
            ', '.join([str(i) for i in self.i_in]),
            ', '.join([str(i) for i in self.i_out]),
            self.rate)
        return s

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            self._name(),
            )

    def __eq__(self, other):
        return (self.ions_in == other.ions_in and
                self.ions_out == other.ions_out)

# TODO - implement collection object
class DecayRateData(object):
    """
    goal: allow easy (fast) finding of rates, e.g., hash dictionaly by
    ions
    """
    def __init__(self, reactions):
        self._reactions = list(reactions)
        self._ions_in = dict()
        for r in self._reactions:
            for i in r.ions_in:
                l = list()
                self._ions_in.setdefault(i, l).append(r)

    def add(self, r):
        self._reactions.append(r)
        for i in rreactions.ions_in:
            l = list()
            self._ions_in.setdefault(i, l).append(r)

    def get_reactions_in(self, i):
        return self._ions_in.get(i, [])

    def __next__(self):
        self._index = -1
        return self

    def __iter__(self):
        self._index += 1
        if index >= len(self._index):
            raise StopIteration()
        return self._reactions(self._index)

    def __len__(self):
        return len(self._reactions)

    def __getitem__(self, index):
        return self._reactions[index]


class TimedDecay(IonMap):
    def __init__(self,
                 ions = None,
                 decays = None, # you should usually not need to use this
                 decaydata = None,
                 molfrac_in = None,
                 molfrac_out = None,
                 decions = None,
                 time = None,
                 extend = True,
                 silent = False,
                 ):
        """
        INPUT:
        expect decaydata to be list of DecayRate objects
        """

        self.setup_logger(silent)

        # get ion data
        ions, molfrac_in, molfrac_out = self.input2ionarr(
            ions, molfrac_in, molfrac_out)

        self.start_timer('compose')

        # get ion_list
        decay_ions = set(ions.copy())

        if decays is None:
            assert decaydata is not None
            decays = self.decay_data(
                ions = ions,
                extend = extend,
                decaydata = decaydata,
                silent = silent,
                )

        for d in decays:
            decay_ions |= d.ions
        decay_ions = sorted(decay_ions)
        ion_map = {i:j for j,i in enumerate(decay_ions)}

        n = len(decay_ions)
        a = np.zeros((n, n), dtype = np.float64)

        for d in decays:
            ii = list(ion_map[i] for i in d.i_in)
            io = list(ion_map[i] for i in d.i_out)
            # TODO - for ions that have multiplicity in 'in' chanel:
            # need to devide by n! (not relevant for decays)
            for i,j in itertools.product(ii, io):
                a[i, j] += d.rate
            for i in ii:
                a[i, i] -= d.rate

        self.logger_timing(
            timing = 'matrix composition finished in',
            timer = 'compose',
            finish = True)

        if decions is None:
            decions = decay_ions
        elif decions == Ellipsis:
            decions = ions
            molfrac_out = molfrac_in
        else:
            decions, molfrac_out = self.output2ionarr(
                decions, molfrac_out)

        self.set_molfrac(molfrac_in, molfrac_out)

        self.a0 = a
        self.ions = ions
        self.decions = decions
        self.ion_map = ion_map
        self.time = np.nan
        self.decays = decays # for debug etc.

        self.update_time(time)

        self.close_logger(timing = 'decay matrix constructed in')

    def update_time(self, time = None):
        # compute reduced matrix
        if time is None:
            return
        if time == self.time:
            return
        with self.timeenv(timing = 'decay matrix computed in'):
            a = expm(self.a0 * time)
            self._project(a)
        self.time = time

    def _project(self, a):
        ii = [self.ion_map[i] for i in self.ions]
        io = [self.ion_map[i] for i in self.decions]

        a = a[ii, :][:,io]
        self.molfrac_convert(a)
        self.map = a

    def __call__(self, *args, **kwargs):
        """
        call routine allows extra time parameter
        """
        kw = kwargs.copy()
        time = kw.pop('time', None)
        if time is None and len(args) > 1:
            time = args[1]
            args = args[:1] + args[2:]
        if time is not None:
            self.update_time(time)
        return super().__call__(*args, **kw)

    def timeseries(self,
                   ions = None,
                   start=None, stop=None, num=None, endpoint=None,
                   **kwargs):
        # this is TOO inefficient except for small isotope vectors
        # use of identity matrix is not a good choice.
        silent = kwargs.get('silent', False)
        self.setup_logger(silent = silent)
        kwm = dict(start=start, stop=stop, num=num, endpoint=endpoint)
        kw = kwargs.copy()
        from scipy.sparse.linalg import expm_multiply
        _a = self.map
        x = expm_multiply(
            self.a0,
            np.identity(self.a0.shape[0]),
            **kwm)
        out = []
        for a in x:
            self._project(a)
            out += [self.__call__(ions)]
        self.map = _a
        self.close_logger(timing = 'time series completed in {}.')
        return out

    def decay_data(self,
                 ions = None,
                 decaydata = None,
                 extend = True,
                 silent = False):
        self.setup_logger(silent = silent)
        assert ions is not None
        selected = list()
        ions = set(ions)
        additions = False
        ions = ions.copy()
        for d in decaydata:
            if d.ions <=  ions:
                selected += [d]
            # check convexity
            x = d.ions_out - ions
            if d.ions_in <= ions and len(x) > 0:
                s = str(d)
                if extend:
                    selected += [d]
                    self.logger.warn(
                        "adding " +
                        ' ,'.join(str(i) for i in x) +
                        ' due to ' + s)
                    additions = True
                    ions |= d.ions_out
                else:
                    self.logger.warn(
                        "ion list not convex for " + s)

        # iteratively add all ions on "out" channel
        iteration = 1
        while additions:
            iteration += 1
            self.logger.warn(
                "Pass number {}".format(iteration))
            additions = False
            for d in decaydata:
                # check convexity
                x = d.ions_out - ions
                if d.ions_in <= ions and len(x) > 0:
                    s = str(d)
                    selected += [d]
                    self.logger.warn(
                        "adding " +
                        ' ,'.join(str(i) for i in x) +
                        ' due to ' + s)
                    additions = True
                    ions |= d.ions_out

        self.close_logger()
        return selected

#######################################################################
# tests

def test_isomer_decay():
    import tempfile
    ions = ['c9', 'b8m3', 'c8', 'b8m1', 'be8']
    data = """
           c9 1 p b8m3
           b8m3 g
           b8m2 b- c8
           c8 EC b8m1
           b8m1 g
           b8 b+ be8
           be8 2a
           he4
           h1
          """
    with tempfile.NamedTemporaryFile(mode='w+t') as f:
        for l in data.split('\n'):
            f.write(l.strip() + '\n')
        f.flush()
        d = Decay(ions,
                  #decayfile = '~/kepler/local_data/decay_test.dat',
                  decayfile = f.name,
                  molfrac_in = True,
                  molfrac_out = True,
                  debug = True,
                  )
    return d

def test_tdep_dec_abuset():
    import bdat
    import kepdump

    np.seterr(all = 'warn')
    k = kepdump.load('/home/alex/kepler/test/z15D#nucleo')
    a = k.abub
    b = bdat.BDat('/home/alex/kepler/local_data/bdat').decaydata
    d = TimedDecay(ions = a, decaydata = b)

    import matplotlib.pylab as plt

    f = plt.figure()
    ax = f.add_subplot(111)
    for x in np.arange(4,9, 0.5):
        t = 10**x
        s = time2human(t)
        ax.plot(k.zm_sun, d(a, time = t).ion_abu('co56'), label = s)
    ax.set_xlabel('mass coordinate / solar masses')
    ax.set_ylabel('mass fraction')
    leg = ax.legend(loc='best')
    leg.set_draggable(True)
    ax.set_xlim(1.4, 1.6)

def test_tdep_dec_simple():
    import abusets
    s = abusets.AbuSet({'ni56':0.5})
    b = [
        DecayRate(['Ni56'], ['Co56'], 1.320582e-06),
        DecayRate(['Co56'], ['Fe56'], 1.038745e-07),
        ]
    d = TimedDecay(ions = s, decaydata = b)
    for i in range(10):
        t = 10**i
        print(t,': ', d(s, time=t))
