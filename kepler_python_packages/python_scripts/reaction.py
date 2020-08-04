"""
Module for reaction based on Ion class
"""

import numpy as np
from copy import copy

from isotope import Ion, ion, register_other_bits, GAMMA
from utils import CachedAttribute, cachedmethod

class DecIon(Ion):
    """
    Class for decays.

    Will store A, Z, E in ion class portion
    Will set flag F_REACTION
    Will store extra info in "B" field, for now just the 12 recognized decay modes

    Provides list of ions, a dictionary with numbers -
        number of ions/particles that come out.

    The general reaction class should have these be negative for
    things that go in and positive for things that come out whereas the
    A, Z, E fields will store IN - OUT net values.

    LIMITATION:
        Currently this class only handles a pre-defined set of decays.

    PLAN:
        Extension for sf - complicated decays - maybe a separate
        derived class?

    TODO:
        For general class provide a generalized *index*.
        probably multiply the FZAE indices
        (plus BMUL/2 to take care of negative values in Z)
        with according numbers of BMUL?
        Add in flag for IN/OUT?
        It does not have to be nice, just unique...
        General reactions need have dE = 0 but still connect and 2 (Z,A,E) states.
    """

    # define KepIon bits
    F_OTHER_REACTION = 1
    F_REACTION = Ion.F_OTHER * F_OTHER_REACTION

    # list of decay names accepted
    reaction_list = {
        ''     :  0,
        's'    :  0,
        'g'    :  1,
        'g1'   :  1,
        '1g'   :  1,
        'b-'   :  2,
        'b+'   :  3,
        'ec'   :  4,
        'n'    :  5,
        'p'    :  6,
        'a'    :  7,
        '2a'   :  8,
        'a2'   :  8,
        '2p'   :  9,
        'p2'   :  9,
        'n2'   : 10,
        '2n'   : 10,
        'np'   : 11,
        'pn'   : 11,
        'bn'   : 12,
        'b2n'  : 13,
        'g2'   : 14,
        '2g'   : 14,
        'g3'   : 15,
        '3g'   : 15,
        'g4'   : 16,
        '4g'   : 16,
        }

    # OK, we want names to be unique on output
    reaction_names = {
          0 : 's'  ,
          1 : 'g'  ,
          2 : 'b-' ,
          3 : 'b+' ,
          4 : 'ec' ,
          5 : 'n'  ,
          6 : 'p'  ,
          7 : 'a'  ,
          8 : '2a' ,
          9 : '2p' ,
         10 : '2n' ,
         11 : 'pn' ,
         12 : 'bn' ,
         13 : 'b2n',
         14 : 'g2' ,
         15 : 'g3' ,
         16 : 'g4' ,
        }

    # list of particles that come out...
    particle_list = {
         0 : {},
         1 : {ion('g')   : +1},
         2 : {ion('e-')  :  1},
         3 : {ion('e+')  :  1},
         4 : {ion('e-')  : -1},
         5 : {ion('nt1') :  1},
         6 : {ion('h1')  :  1},
         7 : {ion('he4') :  1},
         8 : {ion('he4') :  2},
         9 : {ion('h1')  :  2},
        10 : {ion('nt1') :  2},
        11 : {ion('nt1') :  1, ion('h1')  : 1},
        12 : {ion('e-')  :  1, ion('nt1') : 1},
        13 : {ion('e-')  :  1, ion('nt1') : 2},
        14 : {ion('g')   : +2},
        15 : {ion('g')   : +3},
        16 : {ion('g')   : +4},
        }

    def is_photon(self):
        return self.E != 0

    _custom_add = True

    def _add(self, x, sign1 = +1, sign2 = +1):
        """
        Add reaction to Ion of two reactions

        if Ion is an isomer, return isomer, otherwise return isotope
        """
        if isinstance(x, self.__class__):
            new = self.__class__('')
            new.B = -1
            new._particles = {}
            for p,m in self._particles.items():
                new._particles[p] = sign1 * m
            for p,n in x._particles.items():
                n *= sign2
                m = new._particles.get(p, 0)
                if m == 0:
                    new._particles[p] = n
                else:
                    m += n
                    if m == 0:
                        del new._particles[p]
                    else:
                        new._particles[p] = m
            for b,p in new.particle_list.items():
                if p == new._particles:
                    new.B = b
            new.Z, new.A, new.E = new.particles2zae(new._particles)
            return new

        if not isinstance(x, Ion):
            x = ion(x)
        A = sign1 * self.A + sign2 * x.A
        Z = sign1 * self.Z + sign2 * x.Z
        if x.is_isomer():
            E = max(sign2 * x.E + sign1 * self.E, 0)
        else:
            E = None
        return ion(Z = Z, A = A, E = E)

    def __add__(self, x):
        return self._add(x)
    __radd__ = __add__

    def __call__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        return self._add(x, +1, -1)

    def __rsub__(self, x):
        return self._add(x, -1, +1)

    def __mul__(self, x):
        assert np.mod(x, 1) == 0, " should only mutiply integers"

        new = self.__class__('')
        new.B = -1
        new._particles = {}
        for p,m in self._particles.items():
            new._particles[p] = m * x
        for b,p in new.particle_list.items():
            if p == new._particles:
                new.B = b
        new.Z, new.A, new.E = new.particles2zae(new._particles)
        return new

    __rmul__ = __mul__

    def __init__(self, s):
        """
        Set up decay reaction.

        Currently only a string is allowed for initialization.
        The main purpose is the interface to the decay.dat file
        and the Decay class setup in ionmap.py.
        """
        if isinstance(s, str):
            self.B, self.F, self.Z, self.A, self.E = self.ion2bfzae(s)
        elif isinstance(s, type(self)):
            self.B, self.F, self.Z, self.A, self.E = s.tuple()
        else:
            raise AttributeError('Wrong type')
        assert 0 <= self.B < len(self.reaction_names), "Unknown decay/reaction."
        self._particles = self.particle_list[self.B]
        self.update_idx()

    @classmethod
    def from_reactants(cls, input, output, hint = None):
        """
        e.g., call with 'c14','n14m1'
        """
        pass



    # # for deepcopy:
    # def __getstate__(self):
    #     return self.B
    # def __setstate__(self, x):
    #     self.__init__(self.reaction_names[x])

    @classmethod
    def ion2bfzae(cls, s=''):
        s = s.strip()
        if s.startswith('(') and s.endswith(')'):
            s=s[1:-1].strip()
        b =  cls.reaction_list.get(s.lower(), -1)
        assert b >= 0, "decay not found"
        z, a, e = cls.particles2zae(cls.particle_list.get(b,{}))
        return b, cls.F_REACTION, z, a, e

    @staticmethod
    def particles2zae(particles):
        z, a, e = 0, 0, 0
        for i,k in particles.items():
            a -= i.A * k
            z -= i.Z * k
            e -= i.E * k
        return z, a, e

    def _name(self, upcase = None):
        """
        Return pre-defined name from list.
        """
        if self.B >= 0:
            return self.reaction_names.get(self.B, self.VOID_STRING)
        else:
            i = []
            o = []
            for p,n in self._particles.items():
                if n < 0:
                    i += [p._name(upcase)] * (-n)
                else:
                    o += [p._name(upcase)] * (+n)
            if len(i) > 0 and len(o) == 0:
                o += [GAMMA._name(upcase)]
            if len(i) > 0:
                s = ' '.join(i) + ', '
            else:
                s = ''
            s += ' '.join(o)
            return s


    def particles(self):
        """
        Returns all paricles in the reaction/decay.
        """
        return copy(self._particles)

    def hadrons(self):
        """
        Returns just the hardrons in the reaction/decay.

        This is useful for networks like decay where photons and
        leptons usually are not accounted for.
        """
        h = dict()
        for i,j in self._particles.items():
            if i.is_hadron():
                h[i] = j
        return h

    def nuclei(self):
        """
        Returns just the nuclei in the reaction/decay.

        This is useful for networks like decay where photons and
        leptons usually are not accounted for.
        """
        h = dict()
        for i,j in self._particles.items():
            if i.is_nucleus():
                h[i] = j
        return h

    def leptons(self):
        """
        Returns just the leptons in the reaction/decay.
        """
        h = dict()
        for i,j in self._particles.items():
            if i.is_lepton():
                h[i] = j
        return h

    def photons(self):
        """
        Returns just the photons (if any) in the reaction/decay.
        """
        h = dict()
        for i,j in self._particles.items():
            if i.is_photon():
                h[i] = j
        return h

    def isstable(self):
        """
        Return if 'reaction'/'decay' is 'stable'.
        """
        return self.B == 0

register_other_bits(DecIon.F_OTHER_REACTION, DecIon)

# Q: not sure this will become sub- or super-class of DecIon
# A: should become a superclass
class ReactIon(DecIon):
    """
    Class for pre-defined set of reactions.

    I suppose we don't need separate 'In' and 'Out' list - just
    positive and negative values in the dictionary of particles.
    We could support 'In' and 'Out', though.
    We should add pre-defined strings for the common set of p,n,a,g,b+/b-.

    Will store A, Z, E in ion class portion
    Will set flag F_REACTION
    Will store extra info in "B" field, for now just the recognized reactions

    Provides list of ions, a dictionary with numbers -
        number of ions/particles that come out or come in.

    The general reaction class should have these be negative for
    things that go in and positive for things that come out whereas the
    A, Z, E fields will store IN - OUT net values.

    LIMITATION:
        Currently this class only handles a pre-defined set of reactions

    PLAN:
        Extension for sf - complicated reactions - maybe a separate
        derived class?

    TODO:
        For general class provide a generalized *index*.
        probbaly multiply the FZAE indices
        (plus BMUL/2 to take care of negative values in Z)
        with according numbers of BMUL?
        Add in flag for IN/OUT?
        It does not have to be nice, just unique...
    TODO:
        'g,...' and '...,g' should be used only when level
            change is intended
    """

    # list of reactions names accepted
    reaction_list = {
        ''     :  0,
        'b-'   :  1,
        'ec'   :  2,
        'b+'   :  3,
        'pc'   :  4,
        'g,n'  :  5,
        'gn'   :  5,
        'n,g'  :  6,
        'ng'   :  6,
        'g,p'  :  7,
        'gp'   :  7,
        'p,g'  :  8,
        'pg'   :  8,
        'g,a'  :  9,
        'ga'   :  9,
        'a,g'  : 10,
        'ag'   : 10,
        'n,p'  : 11,
        'np'   : 11,
        'p,n'  : 12,
        'pn'   : 12,
        'n,a'  : 13,
        'na'   : 13,
        'a,n'  : 14,
        'an'   : 14,
        'p,a'  : 15,
        'pa'   : 15,
        'a,p'  : 16,
        'ap'   : 16,
        'g'    : 17,
        'g*'   : 18,
        'ad'   : 19,
        'ac'   : 20,
        }

    # OK, we want names to be unique on output
    reaction_names = {
          0 : ''   ,
          1 : 'b-' ,
          3 : 'ec' ,
          2 : 'b+' ,
          4 : 'pc' ,
          5 : 'g,n',
          6 : 'n,g',
          7 : 'g,p',
          8 : 'p,g',
          9 : 'g,a',
         10 : 'a,g',
         11 : 'n,p',
         12 : 'p,n',
         13 : 'n,a',
         14 : 'a,n',
         15 : 'p,a',
         16 : 'a,p',
         17 : 'g'  ,
         18 : 'g*' ,
         19 : 'ad',
         20 : 'ac',
        }

    reaction_names_latex = {
          0 : r''   ,
          1 : r'\beta^-' ,
          3 : r'\mathsf{EC}' ,
          2 : r'\beta^+' ,
          4 : r'\mathsf{PC}' ,
          5 : r'\gamma,\mathsf{n}',
          6 : r'\mathsf{n},\gamma',
          7 : r'\gamma,\mathsf{p}',
          8 : r'\mathsf{p},\gamma',
          9 : r'\gamma,\alpha',
         10 : r'\alpha,\gamma',
         11 : r'\mathsf{n},\mathsf{p}',
         12 : r'\mathsf{p},\mathsf{n}',
         13 : r'\mathsf{n},\alpha',
         14 : r'\alpha,\mathsf{n}',
         15 : r'\mathsf{p},\alpha',
         16 : r'\alpha,\mathsf{p}',
         17 : r'\gamma'  ,
         18 : r'\gamma^*' ,
         19 : r'\mathsf{AD}',
         20 : r'\mathsf{AC}',
        }

    # list of particles that come out...
    particle_list = {
         0 : {},
         1 : {                 ion('e-')  : 1},
         2 : {                 ion('e+')  : 1},
         3 : {ion('e-')  : -1                },
         4 : {ion('e+')  : -1,               },
         5 : {                 ion('nt1') : 1},
         6 : {ion('nt1') : -1                },
         7 : {                 ion('h1')  : 1},
         8 : {ion('h1')  : -1                },
         9 : {                 ion('he4') : 1},
        10 : {ion('he4') : -1                },
        11 : {ion('nt1') : -1, ion('h1')  : 1},
        12 : {ion('h1')  : -1, ion('nt1') : 1},
        13 : {ion('nt1') : -1, ion('he4') : 1},
        14 : {ion('he4') : -1, ion('nt1') : 1},
        15 : {ion('h1')  : -1, ion('he4') : 1},
        16 : {ion('he4') : -1, ion('h1')  : 1},
        17 : {ion('g')   : -1                },
        18 : {                 ion('g')   : 1},
        19 : {                 ion('he4') : 1},
        20 : {ion('he4') : -1                },
        }

    def _name(self, upcase = None):
        """
        Return pre-defined name from list.
        """
        return "({})".format(self.reaction_names.get(self.B, self.VOID_STRING))

    @cachedmethod
    def mpl(self):
        s = self.reaction_names_latex.get(self.B, self.VOID_STRING)
        s = s.replace(',', '$,$')
        #s = f'$({s})$'
        s = '$({})$'.format(s)
        return s

    def __mul__(self, x):
        raise NotImplementedError("Can't multiply tabulated reactions yet.")

    def __neg__(self):
        return self.rev()

    def rev(self):
        return self.__class__(self.reaction_names[self.B - 1 + 2 * (self.B % 2)])

    # later we could return a general reaction here

# TODO - add split function

# define more general class

from collections import Counter, OrderedDict

class Reaction(object):
    """
    General reaction class, including all that goes in and out.
    """
    def __init__(self, nuc_in, nuc_out, flags = None):
        """
        nuc_in, nuc_out - ion, list of ions, [list of] tuple[s] of (#, ion)

        TODO - maybe use OrderedDict?
        """

        if isinstance(nuc_in, (Ion, str)):
            nuc_in = (1, nuc_in)
        if isinstance(nuc_in, tuple):
            nuc_in = [nuc_in]
        if isinstance(nuc_in, (list, np.ndarray)):
            x_in = Counter()
            for x in nuc_in:
                if isinstance(x, tuple):
                    i,n = x
                else:
                    i,n = 1,x
                if isinstance(n, str):
                    n = ion(n)
                assert isinstance(i, int)
                assert isinstance(n, Ion)
                x_in.update({n:i})
            nuc_in = x_in
        assert isinstance(nuc_in, Counter)
        for n,i in nuc_in.items():
            if isinstance(n, str):
                del nuc_in[n]
                nuc_in.update({ion(n):i})
        self.nuc_in = OrderedDict(sorted(nuc_in.items(), reverse = True))

        if isinstance(nuc_out, (Ion, str)):
            nuc_out = (1, nuc_out)
        if isinstance(nuc_out, tuple):
            nuc_out = [nuc_out]
        if isinstance(nuc_out, (list, np.ndarray)):
            x_out = Counter()
            for x in nuc_out:
                if isinstance(x, tuple):
                    i,n = x
                else:
                    i,n = 1,x
                if isinstance(n, str):
                    n = ion(n)
                assert isinstance(i, int)
                assert isinstance(n, Ion)
                x_out.update({n:i})
            nuc_out = x_out
        assert isinstance(nuc_out, Counter)
        for n,i in nuc_out.items():
            if isinstance(n, str):
                del nuc_out[n]
                nuc_out.update({ion(n):i})
        self.nuc_out = OrderedDict(sorted(nuc_out.items(), reverse = True))

        self.flags = flags

    @CachedAttribute
    def inlist(self):
        ions = []
        for n,i in self.nuc_in.items():
            ions.extend([n]*i)
        return ions

    @CachedAttribute
    def outlist(self):
        ions = []
        for n,i in self.nuc_out.items():
            ions.extend([n]*i)
        return ions

    @cachedmethod
    def __str__(self):
        s = (' + '.join([str(n) for n in self.inlist]) +
             ' --> ' +
             ' + '.join([str(n) for n in self.outlist]))
        return(s)

    @cachedmethod
    def mpl(self):
        s = (' $+$ '.join([n.mpl() for n in self.inlist]) +
             ' $\longmapsto$ ' +
             ' $+$ '.join([n.mpl() for n in self.outlist]))
        return(s)

    @cachedmethod
    def __repr__(self):
        return '[' + str(self) + ']'

    def __lt__(self, x):
        assert isinstance(x, self.__class__)
        for n,i, nx, ix in zip(self.nuc_in.keys(),
                               self.nuc_in.values(),
                               x.nuc_in.keys(),
                               x.nuc_in.values(),
                               ):
            if n < nx:
                return True
            elif n > nx:
                return False
            if i < ix:
                return True
            elif i > ix:
                return False
        if len(self.nuc_in) < len(x.nuc_in):
            return True
        elif len(self.nuc_in) > len(x.nuc_in):
            return False

        for n,i, nx, ix in zip(self.nuc_out.keys(),
                               self.nuc_out.values(),
                               x.nuc_out.keys(),
                               x.nuc_out.values(),
                               ):
            if n < nx:
                return True
            elif n > nx:
                return False
            if i < ix:
                return True
            elif i > ix:
                return False
        if len(self.nuc_out) < len(x.nuc_out):
            return True
        elif len(self.nuc_out) > len(x.nuc_out):
            return False

        if self.flags == None and x.flags != None:
            return True
        elif self.flags != None and x.flags == None:
            return False
        elif self.flags != None and x.flags != None:
            if self.flags != x.flags:
                return self.flags < x.flags

        # equal
        assert self.__eq__(x)
        return False

    def __eq__(self, x):
        assert isinstance(x, self.__class__)
        if len(self.nuc_in) != len(x.nuc_in):
            return False
        if len(self.nuc_out) != len(x.nuc_out):
            return False
        if not np.alltrue(np.array(list(self.nuc_in.values())) ==
                          np.array(list(x.nuc_in.values()))):
            return False
        if not np.alltrue(np.array(list(self.nuc_out.values())) ==
                          np.array(list(x.nuc_out.values()))):
            return False
        if not np.alltrue(np.array(list(self.nuc_in.keys())) ==
                          np.array(list(x.nuc_in.keys()))):
            return False
        if not np.alltrue(np.array(list(self.nuc_out.keys())) ==
                          np.array(list(x.nuc_out.keys()))):
            return False
        if not self.flags == x.flags:
            return False
        return True

    @CachedAttribute
    def reverse(self):
        """
        Return reverse reaction.
        """
        return self.__class__(self.nuc_out, self.nuc_in, self.flags)
