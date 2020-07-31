"""
Module to make fission data for KEPLER
"""

import numpy as np
import collections

import isotope
import utils
from logged import Logged
from reaction import Reaction

class EmptyRecord(Exception):
    pass

# both ReacLibData and ReacLibRecord shoud be drived from general base
# classes that provide writing.


class ReacLibData(Logged):
    def __init__(self,
                 reac = True,
                 mode = None,
                 nuc = None,
                 nuc_mode = None,
                 ):

        if reac == True or isinstance(reac, (str,tuple, list)):
            kwargs = {}
            if not isinstance(reac, bool):
                kwargs['filename'] = reac
            if mode is not None:
                kwargs['mode'] = mode
            self.load_rate_data(**kwargs)
        else:
            self.rates = None

        if nuc == True or isinstance(nuc, str):
            kwargs = {}
            if isinstance(nuc, str):
                kwargs['filename'] = nuc
            if nuc_mode is not None:
                kwargs['mode'] = nuc_mode
            self.load_nuclear_data(**kwargs)
        else:
            self.nucdata = None

    def load_rate_data(self,
                          filename = (
        '/home/alex/kepler/fission/netsu_nfis_Roberts2010rates',
        '/home/alex/kepler/fission/netsu_sfis_Roberts2010rates',
        ),
                          mode = 'reaclib2',
                          ):
        self.setup_logger(silent = False)
        rates = []
        filename = utils.iterable(filename)
        assert mode in ('reaclib', 'reaclib2')
        for fn in filename:
            with open(fn, 'rt') as f:
                self.logger.info(f'loading {fn} ...')
                if mode == 'reaclib':
                    chapter = int(f.readline())
                    f.readline()
                    f.readline()
                else:
                    chapter = None
                while True:
                    try:
                        rate = ReacLibRecord(f, chapter)
                        if rate is not None:
                            rates.append(rate)
                    except Exception as e:
                        if isinstance(e, EmptyRecord):
                            break
                        else:
                            raise
        self.rates = rates
        self.close_logger(r'loaded {} reactions in '.format(len(rates)))


    def load_nuclear_data(self,
                          filename = '/home/alex/kepler/fission/winvne_v2.0.dat',
                          mode = 'winvne2',
                          ):
        """
        load nuclear data for ReacLib
        """
        self.setup_logger(silent = False)
        assert mode == 'winvne2'
        nucdata = []
        with open(filename) as f:
            self.logger.info(f'loading {filename} ...')
            # read header: grid and nuclei info
            l = f.readline()
            assert len(l.strip()) == 0, '{}'.format(len(l)) + ' >' + l + '<'
            l = f.readline()
            assert l.rstrip() == '010015020030040050060070080090100150200250300350400450500600700800900100'
            t9grid = [int(l[3*i:3*(i+1)])/10 for i in range(len(l.rstrip())//3)]
            nt9 = len(t9grid)
            nuclei = []
            while True:
                l = f.readline()
                n = isotope.ion(l)
                # equal nuclei is used as sentinel for end of list
                if len(nuclei) > 1 and nuclei[-1] == n:
                    break
                nuclei.append(n)
            nnuclei = len(nuclei)
            self.logger.info('{} temperature points for {} nuclei.'.format(nt9, nnuclei))
            while True:
                try:
                    nuc = ReacLibNuc(f, mode = mode, nt9 = nt9)
                    if nuc is not None:
                        nucdata.append(nuc)
                except Exception as e:
                    if isinstance(e, EmptyRecord):
                        break
                    else:
                        raise
            assert len(nucdata) == len(nuclei)
        self.nucdata = nucdata
        self.close_logger(r'loaded {} nuclei in '.format(len(nucdata)))


    def combine_records(self):
        """
        combine resonnant and non-resonnat contributionss
        combine forward and reverse rates
        """
        # this is trick, we do need a dictionary
        # mybe we do need to do combinations first

        self.logger.info('Combining rates ...')
        n_in = len(self.rates)
        rates = {}
        for i,r in enumerate(self.rates):
            formula = r.formula # may need mapping of compatible formulae
            label = (tuple(r.reaction.inlist), tuple(r.reaction.outlist), formula)
            if label in rates:
                rates[label] += (i,)
            else:
                rates[label] = (i,)
        crates = []
        for ii in rates.values():
            rate = self.rates[ii[0]]
            for i in ii[1:]:
                rate.coeff.extend(self.rates[i].coeff)
            crates.append(rate)
        self.rates = crates
        n_out = len(self.rates)
        self.logger.info('Combining {:d} rates to {:d}'.format(n_in, n_out))

    def sort_records(self):
        self.logger.info('Sorting rates ...')
        self.rates = sorted(self.rates, key = lambda x: x.reaction)

    def write_header(self, f):
        version = 10000
        f.write('version {:>6d}\n'.format(version))
        f.write('ReacLib rates\n')

    def write_rates(self, f):
        self.combine_records()
        self.sort_records()
        self.logger.info('Writing rates ...')
        f.write('RATES\n')
        f.write('{:>6d}\n'.format(len(self.rates)))
        f.write('RATE DATA\n')
        for r in self.rates:
            r.write(f)

    def write_nucdata(self, f):
        # need to sort data ?
        self.logger.info('Writing nuclear data ...')
        f.write('NUCLEI\n')
        f.write('{:>6d}\n'.format(len(self.nucdata)))
        f.write('NUCLEI DATA\n')
        for x in self.nucdata:
            x.write(f)

    def write(self, filename, rates = True, nucdata = True):
        self.setup_logger(silent = False)
        with open(filename, 'wt') as f:
            self.write_header(f)
            if nucdata and self.nucdata is not None:
                self.write_nucdata(f)
            if rates and self.rates is not None:
                self.write_rates(f)
        self.close_logger('data written in ')

class ReacLibRecord(object):
    chapter_info = {
         1: (1, 1),
         2: (1, 2),
         3: (1, 3),
         4: (2, 1),
         5: (2, 2),
         6: (2, 3),
         7: (2, 4),
         8: (3, 1),
         9: (3, 2),
        10: (4, 2),
        11: (1, 4),
        }
    n_coeff = (4, 3)
    type_names = {
        'n': 'non-resonant',
        'r': 'resonant',
        'w': 'weak',
        's': 'spontaneous',
        ' ': 'non-resonant',
        }

    def __init__(self, f, chapter = None, check = True):
        """
        Read Reaclib data record.

        Based on description in

        https://groups.nscl.msu.edu/jina/reaclib/db/help.php?topic=reaclib_format&intCurrentNum=0
        """
        if chapter is None:
            l = f.readline()
            if len(l) == 0:
                raise EmptyRecord()
            chapter = int(l.strip())
            firstline = False
        else:
            firstline = True
        self.formula = chapter
        n_in, n_out = self.chapter_info[chapter]
        n = n_in + n_out
        l = f.readline()
        if len(l) == 0 and firstline:
            raise EmptyRecord()
        nuc = [l[5+5*i:5+5*(i+1)] for i in range(n)]
        nuc_in = isotope.ufunc_ion(nuc[:n_in])

        # deal with panov modification to reaclib
        nuc_out = []
        for n in nuc[n_in:]:
            if n.count('#') > 0:
                ni,nt = n.split('#')
                nuc_out.extend([nt] * int(ni))
            else:
                nuc_out.append(n)

        nuc_out = isotope.ufunc_ion(nuc_out)
        self.label = l[43:47]
        self.type = l[47]

        # we added type ' ' not in REACLIB documentation
        assert self.type in ('n', 'r', 'w', 's', ' '), 'ERROR type: '+ l
        reverse = l[48]
        assert reverse in ('v', ' '), 'ERROR reverse: ' + l
        self.reverse = reverse == 'v'
        self.Q = float(l[52:64])
        coeff = []
        for nci in self.n_coeff:
            l = f.readline()
            for i in range(nci):
                coeff.append(l[i*13:(i+1)*13])
        coeff = [float(i) for i in coeff]
        self.reaction = Reaction(nuc_in, nuc_out)
        self.coeff = coeff

        # some check - this does increase time by 35%
        # if check:
        #     assert np.sum(isotope.ufunc_A(self.nuc_in)) == np.sum(isotope.ufunc_A(self.nuc_out))
        #     sum_Z_in = np.sum(isotope.ufunc_Z(self.nuc_in))
        #     sum_Z_out = np.sum(isotope.ufunc_Z(self.nuc_out))
        #     if self.type == 'w':
        #         assert sum_Z_in in (sum_Z_out - 1, sum_Z_out + 1)
        #     else:
        #         assert sum_Z_in == sum_Z_out

    def __str__(self):
        s = str(self.reaction)
        s1 = self.type_names[self.type]
        if self.reverse:
            s1 += ', reverse'
        s += ' (' + s1 + ')'
        return s

    __repr__ = __str__

    def eval(self, t9):
        return np.exp(
            self.coeff[0] +
            np.sum(self.coeff[1:6] * t9**((2 * np.arange(5) - 4) / 3)) +
            self.coeff[6] * np.log(t9))

    def write(self, f):
        """
        Write to new bdat2 format

        TODO
          - combine res and non-res rates as in bdat
          - compute/set/ceck reverse rates
          - add nuclear data including weight functions
        """

        # the reac_type may need to be adjusted based in res/nres, reverse
        # may want to combine rates as done in bdat (extra clean step in parent)
        formula = self.formula

        f.write(('{:>4d}{:>4d}{:>8d}{:>8d}' + ' {:>4s}          {:s}\n').format(
            len(self.reaction.nuc_in),
            len(self.reaction.nuc_out),
            formula,
            len(self.coeff),
            self.label,
            str(self.reaction),
            ))

        for n,i in self.reaction.nuc_in.items():
            if isinstance(n, isotope.Isomer):
                E = n.E
            else:
                E = -1
            f.write(('{:>4d}'*4).format(i, n.Z, n.N, E))
        f.write('\n')
        for n,i in self.reaction.nuc_out.items():
            if isinstance(n, isotope.Isomer):
                E = n.E
            else:
                E = -1
            f.write(('{:>4d}'*4).format(i, n.Z, n.N, E))
        f.write('\n')

        for i,c in enumerate(self.coeff):
            f.write('{:>13.6e}'.format(float(c)))
            if i % 7 == 6 or i == len(self.coeff) - 1:
                f.write('\n')

class ReacLibNuc(Logged):
    """
    load nuclear data for ReacLib
    """
    winvne2_type = 5
    def __init__(self, f, mode = 'winvne2', nt9 = None):
        assert mode == 'winvne2'
        assert nt9 is not None
        l = f.readline()
        if len(l.strip()) == 0:
            raise EmptyRecord()
        nuc, A, Z, N, S, ME, label = l.split()
        self.ion = isotope.ion(nuc)
        self.A = int(round(float(A)))
        self.Z = int(Z)
        self.N = int(N)
        self.S = float(S)
        self.ME = float(ME)
        self.E = 0.
        self.label = label
        assert self.N + self.Z == self.A
        assert self.ion.N == self.N
        assert self.ion.Z == self.Z
        assert self.ion.E == 0
        data = []
        for i in range(nt9):
            j = i % 8
            if j == 0:
                l = f.readline()
            data.append(float(l[12*j:12*(j+1)]))
        self.coeff = data
        self.Q = round(
            self.Z * 7.28898454697355
            + self.N * 8.071317791830353
            - self.ME, 3)
        self.formula = self.winvne2_type

    def write(self, f):
        n = self.ion
        if isinstance(n, isotope.Isomer):
            E = n.E
        else:
            E = -1
        f.write(('{:>4d}'*3 + ' ' + '{:>13.7f}'*2 + '{:8d}'*2 + ' {:s}\n').format(
            n.Z, n.N, E,
            self.E, self.Q,
            self.formula,
            len(self.coeff) + 1,
            self.label,
                ))
        coeff = [self.S] + self.coeff
        for i,c in enumerate(coeff):
            f.write('{:>13.6e}'.format(float(c)))
            if i % 7 == 6 or i == len(coeff) - 1:
                f.write('\n')

    def __str__(self):
        s = str(isotope.ion) + 'Q={}, S={} '.format(self.Q, self.S)
        return s

    def __repr__(self):
        return '[{}]'.format(str(self))
