"""
Load bdat file.

Provide class for time-dependent decay based on bdat
"""

import os.path
import string
import numpy as np
from logged import Logged
from human import version2human
import isotope
from reaction import DecIon
from ionmap import TimedDecay, DecayRate
from utils import CachedAttribute, cachedmethod, is_iterable
from loader import loader, _loader
from reaction import ReactIon
from abuset import IonList

reaction_desc=["ng","pn","bx","pg","ap","an","ag","ga"]

class BDat(Logged):
    """
    Class to contain BDAT data.

    In first card of element parameter deck
    Z, A, (pairs or formula #, # constants)

    ic(j,0) = type formula to be used to calculate rate

    ic(j,1) = number of constants in fitting formula
              for reaction j on species i
    j = 1 for ng              j = 6 for an
    j = 2 for pn              j = 7 for ag
    j = 3 for weak decay      j = 8 for ground state alpha decay
    j = 4 for pg
    j = 5 for ap

    Writing of bdat files thanks to Andre Sieverding (2016).
    """

    maxsize = 2**14
    maxreac = 10

    def __init__(self,
                 filename = 'bdat',
                 silent = False,
                 check = False):
        self.setup_logger(silent = silent)

        filename_ = filename
        filename = os.path.expanduser(os.path.expandvars(filename))
        if not os.path.exists(filename):
            path = os.getenv('KEPLER_DATA')
            if not path:
                path = os.path.join(os.path.expanduser('~'), 'kepler', 'local_data')
                #self.logger.warning(f'Using default path "{path}".')
                self.logger.warning('Using default path "{}".'.format(path))
            else:
                path = os.path.expanduser(os.path.expandvars(path))
            filename = os.path.join(path, filename)
            if not os.path.exists(filename):
                #msg = f'Could not find data file "{filename_}".'
                msg = 'Could not find data file "{}".'.format(filename_)
                self.logger.error(msg)
                raise FileNotFoundError(msg)

        self.check = check

        data = np.ndarray(self.maxsize, dtype = np.object)

        with open(filename) as f:
            self.f = f

            # set version information
            self.version = -1
            self._read_record()

            self.count = 0
            while True:
                record = self._read_record()
                if record is None:
                    break
                data[self.count] = record
                self.count += 1

        self.data = data[:self.count]
        self.filename = filename
        self.close_logger(timing='BDAT "{:s}" loaded {:d} records in'.format(filename,self.count))
        del self.f

    def _read_record(self):
        # read formula deck
        # (2i6,20i3,1X,A80)
        line = self.f.readline()
        nz = int(line[0:6])
        na = int(line[6:12])
        ic = [line[i:i+3] for i in range(12,72,3)]
        ic = np.array(ic, dtype=np.int64).reshape((-1,2))
        cm = line[72:].strip()
        if (nz == 999) or ((nz == 99) and (na == 99)):
            self.logger.info('Read end of file record.')
            return None
        if self.version == -1:
            if nz == -1:
                self.version = na
                if self.version >= 810:
                    self.version = ((self.version // 100)*10000 +
                               ((self.version // 10) % 10)*100 +
                               (self.version % 10))
                else:
                    self.version = (
                        10000*na +
                        100*ic.flatten()[0] +
                        ic.flatten()[1])
                self.version_name = cm
            if self.version == -1:
                self.version = 0
                self.version_name = 'unkown'
                f.seek(0)
            self.logger.info('Version {:s} - {:s}'.format(
                version2human(self.version),
                self.version_name))
            return True

        # read temperature dependent partition function information
        # <  80100 (2i3,f11.4,f5.1,4e12.3,i2,1X,A8)
        # >= 80100 (2i3,f11.4,f5.1,5e12.3,i2,1X,A8)
        line = self.f.readline()
        nzx = int(line[0:3])
        nn  = int(line[3:6])
        q   = np.array(line[6:17], dtype = np.float64)
        if self.version < 80100:
            sa  = [line[17:22],line[22:34],line[34:46],line[46:58],line[58:70]]
            ist = int(line[70:72])
            iso = line[72:].strip()
        else:
            sa  = [line[17:22],line[22:34],line[34:46],line[46:58],line[58:70],line[70:82]]
            ist = int(line[82:84])
            iso = line[84:].strip()
        sa = np.array(sa, dtype = np.float64)

        # read extra partition function information
        # (f10.4,f10.3,f10.4,f10.3,f10.4,f10.3,f10.4,f10.3)
        if ist > 0:
            n = 2 * ist
            gs = list()
            while n > 0:
                line = self.f.readline()
                dm = min(n, 8)
                i1 = 0
                for i in range(dm):
                    i0 = i1
                    i1 += 10
                    gs.append(line[i0:i1])
                n -= dm
            gs = np.array(gs, dtype=np.float64)
        else:
            gs = None

        # read reaction coefficients
        # (7e13.6,1X,A8)
        c = np.ndarray(self.maxreac, dtype = np.object)
        c[:] = [np.array([], dtype = np.float64)]
        creac = np.array([''] * self.maxreac, dtype = np.object)
        cref  = np.array([''] * self.maxreac, dtype = np.object)
        for j in range(self.maxreac):
            n = ic[j,1]
            if n > 0:
                cx = list()
                m = 0
                ref = ''
                while n > 0:
                    line = self.f.readline()
                    dm = min(n, 7)
                    i1 = 0
                    for i in range(0, dm):
                        i0 = i1
                        i1 += 13
                        cx.append(line[i0:i1])
                    if m == 0:
                        reac = line[92:].strip()
                    elif n <= 7:
                        ref = line[92:].strip()
                    m += dm
                    n -= dm
                c[j] = np.array(cx, dtype=np.float64)
                cref[j] = ref
                creac[j] = reac

        return BDatRecord(
            nz, nzx, nn, na, iso,
            q,
            sa, ist, gs,
            ic, c,
            cm, cref, creac,
            check = self.check)

    def write(self, filename, version = None, version_name = None, silent = True):
        """
        Write the currently stored bdat into a specified file.

        TODO - add general header function writer methods.
        """

        if self.count == 0:
             raise Exception("No data. Read in a bdat file first.")

        # Write version header. This is not (yet) consistent with the verion control of KEPLER!
        # Just copy from the input file for now.
        if version_name is None:
            version_name = self.version_name
        if version is None:
            version = self.version
        version_header = "    -1    {:>2d} {:>2d} {:>2d}  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 {:<s}\n".format(
            version // 10000,
            (version // 100) % 100,
            version % 100,
            version_name)
        last_line = "   999   999  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n"
        # TO DO: ask to overwrite if file exists
        with open(os.path.expanduser(os.path.expandvars(filename)),"wt") as f:
            f.write(version_header)
            for i, rec in enumerate(self.data):
                if not silent:
                    print("Writing record {:d} of {:d}.".format(i + 1, self.count))
                rec.write(f)
            f.write(last_line)

    def write_mass_table(self, filename):
        """
        Write out isotope mass table based on "q" values.

        See also:
        http://www.nndc.bnl.gov/masses/mass.mas03
        n   1.008664916
        H1  1.00782504672146
        Be8 8.00530516464127 (56.49948 MeV mass excess)
        """
        filename = os.path.expanduser(os.path.expandvars(filename))
        with open(filename,'w') as f:
            f.write('; Generated from BDAT "{:s}" file {:s}\n'.format(
                self.version_name,
                self.filename))
            format = '{:<8s} {:12.8f}\n'
            i = 0
            if self.data[i].ion != 'nt1':
                f.write(format.format('nt1', 1.008664916))
            else:
                i = 1
            if (self.data[i].ion != 'h1'):
                f.write(format.format('H1', 1.00782504672146))
            be8 = False
            for d in self.data:
                if be8 and d.ion != 'be8':
                    f.write(format.format('Be8', 8.00530516464127))
                    be8 = False
                f.write(format.format(d.ion.Name(), d.mass()))
                if d.ion == 'Be7':
                    be8 = True

    def __getitem__(self, ion):
        if not hasattr(self, '_ion_hash'):
            self._ion_hash = {d.ion : i for i,d in enumerate(self.data)}
        if not isinstance(ion, isotope.Ion):
            ion = isotope.ion(ion)
        try:
            return self.data[self._ion_hash[ion]]
        except:
            raise KeyError('{} not in data base'.format(ion))

    @CachedAttribute
    def decaydata(self):
        decays = list()
        for d in self.data:
            r = d.get_ad()
            if r is not None:
                decays += [r]
            r = d.get_bd()
            if r is not None:
                decays += [r]
        decays += [DecayRate([isotope.ion('be8')],
                             [isotope.ion('he4'), isotope.ion('he4')],
                             np.log(2)/6.7e-8),
                   DecayRate([isotope.ion('nt1')],
                             [isotope.ion('pn1')],
                             np.log(2)/881.5),
                   ]
        return decays

    @cachedmethod
    def get_decays(self, **kwargs):
        return TimedDecay.decay_data(decaydata = self.decaydata, **kwargs)

    rate_map = {
        '(n,g)' : 0,
        '(p,n)' : 1,
        '(b-)'  : 2,
        '(pc)'  : 2, # rev b+
        '(p,g)' : 3,
        '(a,p)' : 4,
        '(a,n)' : 5,
        '(a,g)' : 6,
        '(ac)'  : 7, # rev ad (since we list lighter nucleus)
        }

    def find_rate(self, ion, reac = None, return_other = False, silent = False):
        if reac is None:
            i = ion.find('(')
            if i < 0:
                i = len(ion) - 2
            reac = ion[i:]
            ion = ion[:i]
        if not isinstance(ion, isotope.Ion):
            ion = isotope.ion(ion)
        if isinstance(reac, int):
            return self[ion], rate
        reac = ReactIon(reac)
        prod = ion + reac
        forward = True
        if ion.Z > prod.Z or (ion.Z == prod.Z and ion.A > prod.A):
            ion, prod = prod, ion
            reac = -reac
            forward = False
        rec = self[ion]
        ind = self.rate_map[str(reac)]
        s = '{}{}{}: {}'.format(ion, reac, prod, rec[ind].creac)
        if rec[ind].cref != '':
            s +=' ({})'.format(rec[ind].cref)
        s += ' {} {} {}'.format(ind, *rec[ind].ic)
        if forward:
            s += ' forward'
        else:
            s += ' reverse'
        if not silent:
            print(' [BDAT] ' + s)
        ret_val = rec, ind, forward
        if return_other:
            ret_val += (self[prod],)
        return ret_val

    def scale_rate(self, fac, ion, reac = None):
        '''
        find rate and scale it.
        '''
        rec, ind, forward = self.find_rate(ion, reac)
        rec.scale(fac, ind)

    def eval(self, ion, t9, rho, reac = None, ):
        rec, ind, forward, rev = self.find_rate(ion, reac, return_other = True)
        val = rec.eval(t9, rho, ind, forward, rev)
        return val

    def mass_excess(self, ions):
        """
        TODO:
          generally supplement by data from
             http://www.nndc.bnl.gov/masses/mass.mas12
        """
        if not is_iterable(ions):
            shape = ()
            ions = ions,
        else:
            shape = np.shape(ions)
        if not hasattr(self, '_mass_excess_data'):
            self._mass_excess_data = dict()
        me = []
        spec_ions_me = {
            isotope.ion('nt1') : 8.07131714,
            isotope.ion('h1' ) : 7.28897059,
            isotope.ion('he4') : 2.42491561,
            isotope.ion('be8') : 4.941671,
            }
        for i in ions:
            x = self._mass_excess_data.get(i, None)
            if x is None:
                try:
                    x = self.__getitem__(i).mass_excess()
                except KeyError:
                    try:
                        x = spec_ions_me[isotope.ion(i)]
                    except KeyError:
                        x = 0
                        #print(f' [ERROR] NOT FOUND: {i} (returning {x})')
                        print(' [ERROR] NOT FOUND: {} (returning {})'.format(i,x))
                self._mass_excess_data[i] = x
            me.append(x)
        if shape == ():
            return me[0]
        return np.array(me)

    @CachedAttribute
    def ions(self):
        return IonList(list(d.ion for d in self.data))

    @CachedAttribute
    def ext_ions(self):
        addions = ('nt1', 'h1', 'he4', 'be8')
        ions = set(self.ions) | set(isotope.ion(i) for i in addions)
        return IonList(sorted(list(ions)))


load = loader(BDat, __name__ + '.load')
_load = _loader(BDat, __name__ + '.load')
loadbdat = load

class BDatRecord(object):
    """
    BDAT reaction dec record for one isotope

    TODO - add evaluation function (but needs weights)

    TODO - make individual reactions their own objects?
    """
    maxreact = 10

    def __init__(self,
                 *args,
                 check = False,
                 silent = False):

        (nz, nzx, nn, na, iso,
         q,
         sa, ist, gs,
         ic, c,
         cm, cref, creac) = args

        self.ion = isotope.ion(Z=nz, A=na)
        self.q     = q
        self.sa    = sa
        self.ist   = ist
        self.gs    = gs
        self.cm    = cm

        self.rates = [BDatRate(ic[i], c[i], creac[i], cref[i]) for i in range(len(ic))]

        if not check:
            return

        assert nz == nzx
        assert nn == na - nz
        assert self.ion == iso

    def write(self, f):
        """
        Write lines to bdat file
        """
        lines = list()
        #--------------------------------------------------
        #HEADER .....
        #header line with 2I6,20I3, i.e., Z,A,8x(ic) + 0s
        line = "{:>6d}{:>6d}".format(self.ion.Z, self.ion.A)
        line += "".join("{:>3d}{:>3d}".format(*r.ic) for r in self)
        lines.append(line)

        #isotope information and partition function
        line = "{:>3d}{:>3d}".format(self.ion.Z, self.ion.N)
        line += "{:11.4f}{:5.1f}".format(float(self.q), float(self.sa[0]))
        line += "".join("{:12.3e}".format(self.sa[i]) for i in range(1,6))
        line += "{:>2d}".format(self.ist)
        line += "{:>6s}".format(self.ion.name())
        lines.append(line)
        if self.ist > 0:
            m = 0
            line = ''
            for i in range(self.ist):
                if m % 4 == 0:
                    if m > 0:
                        lines.append(line)
                    line = ''
                line += '{:>10.3f}{:>10.4f}'.format(*gs[2*i:2*(i+1)])
            lines.append(line)
        #END HEADER
        #-------------------------------------------------
        for i in range(self.maxreact):
            ic = self[i].ic
            c = self[i].c
            if ic[0] > 0 and ic[1] > 0:
                nlines, rest = divmod(ic[1], 7)
                for j in range(max(nlines, 1)):
                    values = c[7 * j: min(7 * (j + 1), ic[1] + 1)].tolist()
                    if j == 0:
                        values += [0.] * (7 - ic[1])
                    line = "".join("{:13.6e}".format(value) for value in values)
                    if j == 0:
                        reac = self[i].creac
                        if reac == '':
                            reac = self.ion.name() + reaction_desc[i]
                        line += " {:>7s}".format(reac)
                    lines.append(line)
                if rest > 0 and nlines > 0:
                    line = "".join("{:13.6e}".format(c[j + 7 * nlines]) for j in range(rest))
                    line += " " * (92 - rest * 13)
                    line += self[i].cref
                    lines.append(line)
        f.write('\n'.join(lines)+'\n')

    def mass(self):
        """
        Return isotope mass based on 'q' value. (u)
        """
        return (self.ion.Z * 1.00782504672146
                + self.ion.N * 1.008664916
                - self.q * 1.0735441502217242e-3)

    def mass_excess(self):
        """
        Return isotope mass excess based on 'q' value. (MeV)
        """
        return (self.ion.Z * 7.28898454697355
                + self.ion.N * 8.071317791830353
                - self.q)

    def get_ad(self):
        ic = self[7].ic
        if ic[0] == 0:
            return None
        assert ic[0] == 19
        i_in = (self.ion + 'he4', )
        i_out = (self.ion, isotope.ion('he4'))
        rate = self[7].c[0]

        return DecayRate(i_in, i_out, rate)

    def get_bd(self, tdep = False):
        """
        return b+, b-, EC rates
        """
        ic = self[2].ic
        jdex = ic[0]
        if jdex == 0:
            return None

        # jdex of 16,17,26,27,36,37 are T-dep and maybe should not be used for t=0?
        if jdex in (16, 17, 26, 27, 36, 37) and not tdep:
            raise # for debug
            return None

        # decay type
        if jdex in (6, 8, 17, 27, 37): # b+/EC
            i_in = (self.ion - DecIon('b+'), )
            i_out = (self.ion,)
        elif jdex in (7, 20, 16, 26, 36): # b-
            i_in = (self.ion, )
            i_out = (self.ion + DecIon('b-'), )

        # decay rate
        if jdex in (6, 20, 28): # GS + excited states
            rate = self[2].c[1]
        if jdex in (7, 8): # GS decay
            rate = self[2].c[0]
        if jdex in (16, 17, 26, 27, 36, 37):
            rate = np.sum(np.exp(self[2].c[0::7]))

        return DecayRate(i_in, i_out, rate)

    def scale(self, fac, ind):
        '''
        scale rate with index "ind"
        '''
        self[ind].scale(fac)

    def __getitem__(self, index):
        return self.rates[index]

    def __str__(self):
        s = self.__class__.__name__ + '('+self.ion.Name()
        r = ','.join(reac.creac for reac in self if reac.creac != '')
        if r != '':
            s += '[' + r + ']'
        s += ')'
        return s

    __repr__ = __str__

    def part_func_g(self, t9):
        '''
        return relative partition function g based on t9
        '''
        g = 1
        if self.sa[1] != 0:
            sa = self.sa
            g += np.exp(
                sa[1] / t9 +
                sa[2] +
                (sa[3] + (sa[4] + self.sa[5] * t9) * t9) * t9
                )
        ist = self.ist
        for i in range(ist):
            g += np.sum(self.gs[2 * i + 1] * np.exp(-self.gs[2 * i] / t9))
        return g

    def part_func_w(self, t9 = None, g = None):
        '''
        return total partition function w based on t9
        '''
        assert (g is None) != (t9 is None), 'Specify excactly one of T9 and g,'
        if g is None:
            g = self.part_func_g(t9)
        w = g * self.sa[0]
        return w

    lrevpar = (False, True, False, False, True, True, False, False, False, False)

    def eval(self, t9, rho, ind, forward, rev):
        '''
        evaluate reaction rate with index ind
        '''
        g = self.part_func_g(t9)
        gr = rev.part_func_g(t9)
        revpar = self.lrevpar[ind]
        return self[ind].eval(t9, rho, forward, g, gr, revpar)

class BDatRate(object):
    '''
    class to hold individual reaction
    '''
    def __init__(self, ic, c, creac, cref):
        self.ic = ic
        self.c = c
        self.creac = creac
        self.cref = cref

    @property
    def ic0(self):
        return self.ic[0]

    @property
    def ic1(self):
        return self.ic[1]

    def scale(self, fac):
        '''
        scale rate with index "ind"
        '''
        ic0 = self.ic0
        ic1 = self.ic1
        c = self.c
        if ic0 == 0 or ic1 == 0:
            raise Exception('Empty rate.')
        if ic0 in (14,15,16,17, 24,25,26,27, 34,35,36,37):
            # reaclib
            for i in range(ic1 // 7):
                c[7 * i] += np.log(fac)
        elif ic0 == 28:
            # lugaro fit
            c[1] *= fac
            for i in range((ic1 - 2) // 3):
                c[3 * i + 3] *= fac
        elif ic0 in (21, 22):
            # Woosley fits
            c[0] += np.log(fac)
        elif ic0 in (6, 19):
            # gs + 1st ex decay
            c[1] *= fac
            c[3] *= fac
        elif ic0 == 18:
            # special reaclib capture - neg Q
            c[0] += np.log(fac)
            c[7] += np.log(fac)
        elif ic0 in (11, 12, 4, 3, ):
            c[0] += np.log(fac)
        elif ic0 == 10:
            nres = (ic1 - 4) // 2
            for i in range(nres + 1):
                c[2*i] *= fac
        elif ic0 in (8, 9, 7, 5, 13, 19, 2, 1):
            c[0] *= fac
        else:
            raise Exception('Rate formula not implemented.')

    @staticmethod
    def _hunt(t9):
        '''
        find index and interpolation for n-dimensional array or scalar

        (this is why it looks so strange)
        '''
        t9ga = np.array([
            1.0e-2,1.5e-2,2.0e-2,3.0e-2,4.0e-2,
            5.0e-2,6.0e-2,7.0e-2,8.0e-2,9.0e-2,
            1.0e-1,1.5e-1,2.0e-1,3.0e-1,4.0e-1,
            5.0e-1,6.0e-1,7.0e-1,8.0e-1,9.0e-1,
            1.0e-0,1.5e-0,2.0e-0,3.0e-0,4.0e-0,
            5.0e-0,6.0e-0,7.0e-0,8.0e-0,9.0e-0,
            1.0e+1])
        t9x = np.array(t9)
        t9g = np.empty_like(t9x, dtype = np.float)
        jlo = np.empty_like(t9x, dtype = np.int)
        ct9 = np.empty_like(t9x, dtype = np.float)
        ii = t9x <= t9ga[0]
        t9g[ii] = t9ga[0]
        jlo[ii] = 0
        ct9[ii] = 0.
        jj = ii
        ii = t9x >= t9ga[-1]
        t9g[ii] = t9ga[-1]
        jlo[ii] = len(t9ga)-2
        ct9[ii] = 1.
        jj |= ii
        kk = ~jj
        t9g[kk] = t9x[kk]
        for i in range(t9x.size):
            if jj.flat[i]:
                continue
            jlo.flat[i] = np.where(t9x.flat[i] > t9ga)[0][-1]
        jl = jlo[kk]
        ct9[kk] = (t9x[kk] - t9ga[jl])/(t9ga[jl + 1] - t9ga[jl])
        return jlo[()], ct9[()], t9g[()], len(t9ga)

    def eval(self, t9, rho, forward, g, gr, revpar):
        '''
        evaluate reaction rate
        '''
        ic0 = self.ic0
        ic1 = self.ic1
        c = self.c
        tvec = np.ones_like(t9)
        dvec = np.ones_like(rho)
        vec = dvec * tvec
        zero = 0 * dvec * tvec
        if ic0 == 7:
            frate = c[0] * vec
            rrate = zero
        elif ic0 == 8:
            rrate = c[0] * vec
            frate = zero
        elif ic0 == 28:
            nrate = (ic1 - 2) // 3
            zsum = c[0]
            rate = c[0] * c[1]
            for k in range(nrate):
                j = 3*k + 2
                t9y = c[j]*exp(c[j+2] / t9)
                zsum += t9y
                rate += c[j+1] * t9y
            rate /= zsum
            frate = rate * dvec
            rrate = zero
        elif ic0 in (14,15,16,17, 34,35,36,37):
            nrate = (ic1 - 2) // 7
            rate = 0.
            for k in range(nrate):
                j = 7 * k
                rate += np.exp(
                    c[j] +
                    c[j+1] / t9 +
                    c[j+2] * t9 ** (-1/3) +
                    c[j+3] * t9 ** (1/3) +
                    c[j+4] * t9 +
                    c[j+5] * t9 ** (5/3) +
                    c[j+6] * np.log(t9)
                    )
            j = 7 * nrate
            if ic0 in (14, 34):
                frate = rate * rho
                rrate = rate * g / gr * c[j] * np.exp(-c[j+1] / t9)
                if revpar:
                    rrate = rrate * rho
                else:
                    rrate = rrate * t9 ** (3/2) * dvec
            elif ic0 in (15, 35):
                rate = rate * rho
                frate = rate * gr / g * c[j] * np.exp(-c[j+1] / t9)
                rrate = rate
            elif ic0 in (16, 36):
                frate = rate
                rrate = zero
            elif ic0 in (17, 37):
                frate = zero
                rrate = rate
            else:
                raise Exception()
        elif ic0 in (18,):
            frate = np.exp(
                    c[0] +
                    c[1] / t9 +
                    c[2] * t9 ** (-1/3) +
                    c[3] * t9 ** (1/3) +
                    c[4] * t9 +
                    c[5] * t9 ** (5/3) +
                    c[6] * np.log(t9)
                    ) * rho
            rrate = np.exp(
                    c[7] +
                    c[8] / t9 +
                    c[2] * t9 ** (-1/3) +
                    c[3] * t9 ** (1/3) +
                    c[4] * t9 +
                    c[5] * t9 ** (5/3) +
                    (c[6] + 1.5) * np.log(t9)
                    ) * g / gr
        elif ic0 in (24,25,26,27):
            nrate = (ic1 - 2) // 8
            rate = 0.
            for k in range(nrate):
                j = 8 * k
                rate += np.exp(
                    c[j] +
                    c[j+1] * t9 **(-5/3) +
                    c[j+2] / t9 +
                    c[j+3] * t9 ** (-1/3) +
                    c[j+4] * t9 ** (1/3) +
                    c[j+5] * t9 +
                    c[j+6] * t9 ** (5/3) +
                    c[j+7] * np.log(t9)
                    )
            j = 8 * nrate
            if ic0 in (24,):
                frate = rate * rho
                rrate = rate * g / gr * c[j] * np.exp(-c[j+1] / t9)
                if revpar:
                    rrate = rrate * rho
                else:
                    rrate = rrate * t9 ** (3/2) * dvec
            elif ic0 in (25,):
                rate = rate * rho
                frate = rate * gr / g * c[j] * np.exp(-c[j+1] / t9)
                rrate = rate
            elif ic0 in (26, ):
                frate = rate
                rrate = zero
            elif ic0 in (27, ):
                frate = zero
                rrate = rate
            else:
                raise Exception()
        elif ic0 in (21, 22):
            jlo, ct9, t9g, nt9 = self._hunt(t9)
            klo = ic1 - nt9 + jlo
            cfi = c[klo] + (c[klo + 1] - c[klo]) * ct9

            t92g = t9g * t9g
            t93g = t92g * t9g
            t913g = t9g**(1/3)
            t923g = t913g * t913g
            t9m13g = 1. / t913g
            t9m23g = t923g * t923g
            rt9 = 11.6048 / t9
            t932 = t9 ** (3/2)

            rate = t9m23g * np.exp(
                c[0] -
                c[1] * t9m13g * (
                    1. + c[2] * t9g + c[3] * t92g + c[4] * t93g
                    )
                + cfi)
            if ic0 == 21:
                frate = rate * rho
                rrate = rate * c[6] * np.exp(-rt9 * c[5]) * g / gr
                if revpar:
                    rrate = rrate * rho
                else:
                    rrate = rrate * t932 * dvec
            else:
                rate = rate * rho
                frate = rate * c[6] * np.exp(-rt9 * c[5]) * gr / g
                rrate = rate
        elif ic0 == 20:
            rt9 = 11.6048 / t9
            t9y = c[2] * np.exp(-c[4] * rt9)
            frate = (c[0] * c[1] + c[3] * t9y) / ( c[0] + t9y) * dvec
            rrate = zero
        elif ic0 == 19:
            rrate = c[0] * vec
            frate = zero
        elif ic0 == 12:
            rt9 = 11.6048 / t9
            rate = rho * t9**(-2/3) * np.exp(
                c[0] - c[1] * t9**(-1/3) *
                (1. + (c[2] + (c[3] + (c[4] + c[8] * t9) * t9) * t9) * t9))
            rrate = rate
            frate = rate * c[6] * np.exp(-c[5] * rt9) * gr / g
        elif ic0 == 11:
            rt9 = 11.6048 / t9
            rate = t9**(-2/3) * np.exp(c[0] - c[1] * t9**(-1/3) *
               (1. + (c[2] + (c[3] + (c[4] + c[8] * t9) * t9) * t9) * t9))
            frate = rate * rho
            rrate = rate * c[6] * np.exp(-c[5] * rt9) * g / gr
            if revpar:
                rrate = rrate * rho
            else:
                rrate = rrate * t9**(3/2) * dvec
        elif ic0 == 10:
            nres = (ic1 - 4) // 2
            sigrs = 0.
            for nr in range(res):
                ir = 2 * nr
                sigrs += c[ir] * np.exp(-c[ir+1] / t9) * t9**(-3/2)
            ir = ic1 - 4
            rate = sigrs + c[ir] * np.exp(-c[ir+1] * t9**(-1/3)) * t9**(-2/3)
            rrate = rate * c[ir + 2] * g / gr * np.exp(c[ir+3] / t9)
            frate = rate * rho
            if revpar:
                rrate = rrate * rho
            else:
                rrate = rrate * t9**(3/2) * dvec
        elif ic0 == 9:
            rate = (c[0] * t9**(2/3) * (1. + c[1] * t9)**(-5/6) *
                    np.exp(c[2] - c[3] * (1/t9 + c[1])**(1/3)))
            rrate = rate * c[4] * g / gr * np.exp(c[5] / t9)
            frate = rate * rho
            if revpar:
                rrate = rrate * rho
            else:
                rrate = rrate * t9**(3/2) * dvec
        elif ic0 == 6:
            rt9 = 11.6048 / t9
            t9y = c[2] * np.exp(-rt9 * c[4])
            rrate = (c[0] * c[1] + c[3] * t9y) / (c[0] + t9y) * dvec
            frate = zero
        elif ic0 == 5:
            rate = rho * c[0] * np.exp(-c[8] * (t9 + c[4])**(-1/3) *
               (1. + (c[1] + (c[2] + c[3] * t9) * t9) * t9)
                + c[9] / (t9 + c[5]))
            rrate = rate
            frate = rate * c[7] * np.exp(-c[9] / t9) * gr / g
        elif ic0 == 4:
            rt9 = 11.6048 / t9
            rate = rho * t9**(-2/3) * np.exp(c[0] - c[1] * t9**(-1/3) *
                (1. + (c[2] + (c[3] + c[4] * t9) * t9) *t9))
            rrate = rate
            frate = rate * c[6] * np.exp(-rt9 * c[5]) * gr / g
        elif ic0 == 3:
            rt9 = 11.6048 / t9
            rate = t9**(-2/3) * np.exp(c[0] - c[1] * t9**(-1/3) *
                (1. + (c[2] + (c[3] + c[4] * t9) * t9) * t9))
            frate = rate * rho
            rrate = rate * c[6] * np.exp(-rt9 * c[5]) * g / gr
            if revpar:
                rrate = rrate * rho
            else:
                rrate = rrate * t9**(3/2) * dvec
        elif ic0 == 2:
            rate = rho * c[0] * np.exp((c[1] + (c[2] + c[3] * t9) * t9) * t9)
            rrate = rate
            frate = rate * c[7] * gr / g * np.exp(-c[9] / t9)
        elif ic0 == 1:
            t9y = (t9 - 0.348)
            rate = c[0] * (t9 / 0.348)**c[3] * np.exp((c[2] * t9y + c[1]) * t9y)
            frate = rate * rho
            rrate = rate * t9**(3/2) * c[7] * exp(-c[9] / t9) * g / gr
        elif ic0 == 0:
            rrate = frate = zero
            # raise Exception('Rate not in bdat')
        else:
            raise Exception('Rate not implemented.')
        if forward:
            return frate
        return rrate

class BDatTimedDecay(TimedDecay):
    """
    make time-dependent decay based on bdat data file
    """
    def __init__(self, **kwargs):
        """
        generate time-dependent decay from bdat file
        """
        kw = kwargs.copy()
        bdat = kw.pop('bdat', None)

        # get data file
        if isinstance(bdat, str):
            bdat = BDat(bdat)
        elif bdat is None:
            bdat = BDat()
        assert isinstance(bdat, BDat)

        # get decays
        kw['decaydata'] = bdat.decaydata

        super().__init__(**kw)
