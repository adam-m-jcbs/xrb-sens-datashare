import re
import os
import time
import socket
import itertools

import numpy as np

from isotope import ion
from bdat import BDatRecord, BDat
from reaction import ReactIon, DecIon

class Nucleus(object):
    """
    nuclear data

    iso:
      isotope.ion
    ME:
      mass excess in MeV of state
   sweight:
     2 * spin + 1
   EL:
     energy of level above ground state (MeV)
   gfunc:
     formula for t-dependent weight function relative to sweight
     flags:
       1: statistical model fit
       2: individual levels
   gconst:
     np.ndarray of formula constants
   """
    def __init__(self, iso, sweight=1, ME=0, EL=0, gfunc=0, gconst=np.ndarray((0))):
        self.iso = iso
        self.ME = ME
        self.EL = EL
        self.sweight = sweight
        self.gfunc = gfunc
        self.gconst = gconst

class IonGroup(object):
    def __init__(self, *args):
        self.n = args[0]
        self.iso = args[1]
    def __repr__(self):
        return '+'.join((self.iso.Name(),)*self.n)

class ReactionSide(list):
    def __init__(self, groups):
        super().__init__(groups)
    def __repr__(self):
        return '+'.join((repr(x) for x in self))

class Reaction(object):
    """
    basic type for each reaction

    kind flags:
      REVERSE = 1
      WEAK    = 2 (provide nu loss)
    """
    def __init__(self, educts, products, formula, constants,
                 kind = 0,
                 priority = 0,
                 tmin = 0, tmax=1e99, dmin=0, dmax=1e99,
                 reference=None, reaction=None):
        self.educts = educts
        self.products = products
        self.kind = kind
        self.priority = priority
        self.formula = formula
        self.constants = constants
        self.reference = reference
        self.reaction = reaction
        self.tmin = tmin
        self.tmax = tmax
        self.dmin = dmin
        self.dmax = dmax

    def __repr__(self):
        sym = '--->'
        if self.kind & 2 == 2:
            ze = sum(x.iso.Z * x.n for x in self.educts)
            zp = sum(x.iso.Z * x.n for x in self.products)
            if ze > zp:
                sym = '(b+)'
            else:
                sym = '(b-)'
        r = f'{self.educts!r}{sym}{self.products!r}'
        if self.reference:
            r += f' ({self.reference})'
        return r

class RDat(object):
    """
    Class to store reaction rate data
    """
    def __init__(self, filename = None):
        self.filename = filename
        if filename is not None:
            self.load()
            return
        self.version = 10000
        self.format = 1
        self.comment = []
        self.nuclei = [] # replace by container class providing insert and sort
        self.reactions = []

    @classmethod
    def from_file(cls, filename):
        return cls(filename = filename)

    def load(self):
        with open(self.filename, 'rt') as f:
            lines = f.read().splitlines()
        # read header
        i = 0
        line = lines[i]
        p = re.compile(r'^VERSION ( *\d+)$')
        items = p.findall(line)
        assert len(items) == 1 and len(items[0]) == 8
        self.version = int(items[0])
        i += 1
        p = re.compile(r'^FORMAT  ( *\d+)$')
        line = lines[i]
        items = p.findall(line)
        assert len(items) == 1 and len(items[0]) == 8
        self.format = int(items[0])
        i += 1
        p = re.compile(r'^COMMENT ( *\d+)$')
        line = lines[i]
        items = p.findall(line)
        assert len(items) == 1 and len(items[0]) == 8
        ncomment = int(items[0])
        i += 1
        self.comment = lines[i:i+ncomment]
        i += ncomment
        p = re.compile(r'^NUCLEI  ( *\d+)$')
        line = lines[i]
        items = p.findall(line)
        assert len(items) == 1 and len(items[0]) == 8
        nnuclei = int(items[0])
        i += 3
        nuclei = []
        for _ in range(nnuclei):
            line = lines[i]
            Z = int(line[0:4])
            A = int(line[4:8])
            E = int(line[8:12])
            SW = int(line[12:16])
            ME = float(line[16:33])
            EL = float(line[33:50])
            gfunc = int(line[50:58])
            ngconst = int(line[58:66])
            name = line[66:].strip()
            iso = ion(Z=Z, A=A, E=E)
            assert iso == ion(name)
            i += 1
            gconst = []
            for j in range((ngconst + 4) // 5):
                line = lines[i]
                gconst += [float(line[k*16:(k+1)*16]) for k in range(min(5, ngconst - j * 5))]
                i += 1
            gconst = np.array(gconst)
            self.nuclei.append(
                Nucleus(
                    iso,
                    sweight=SW,
                    ME=ME,
                    EL=EL,
                    gfunc=gfunc,
                    gconst=gconst,
                    )
                )
        self.nuclei = nuclei
        p = re.compile('^REACTS  ( *\d+)$')
        line = lines[i]
        items = p.findall(line)
        assert len(items) == 1 and len(items[0]) == 8
        nreactions = int(items[0])
        i += 6
        reactions = []
        for _ in range(nreactions):
            line = lines[i]
            items = line.lsplit(' ', 1)
            reaction = items[0]
            if reaction == '':
                reaction = None
            if len(items) > 1:
                reference = items[1]
            else:
                reference = None
            i += 1
            line = lines[i]
            neducts = int(line[0:6])
            nproducts = int(line[6:12])
            kind = int(line[12:18])
            priority = int(line[18:24])
            formula = int(line[24:36])
            nconst = int(line[36:44])
            i += 1
            groups = []
            for _ in range(neducts + nproducts):
                line = lines[i]
                n = int(line[0:4])
                Z = int(line[4:8])
                A = int(line[8:12])
                E = int(line[12:16])
                iso = ion(Z=Z, A=A, E=E)
                groups.append(
                    IonGroup(n, ion)
                    )
                i += 1
            educts = ReactionSide(groups[:neducts])
            products = ReactionSide(groups[neducts:])
            line = lines[i]
            dmin, dmax, tmin, tmax = (float(line[k*16:(k+1)*16]) for k in range(4))
            i += 1
            constants = []
            for j in range((nconst + 4) // 5):
                line = lines[i]
                constants += [float(line[k*16:(k+1)*16]) for k in range(min(5, nconst - j * 5))]
                i += 1
            constants = np.array(constants)
            reactions.append(
                Reaction(
                    educts,
                    products,
                    formula,
                    constants,
                    kind = kind,
                    priority = priority,
                    tmin = tmin,
                    tmax = tmax,
                    dmin = dmin,
                    dmax = dmax,
                    reference=reference,
                    reaction=reaction,
                    )
                )
        self.reactions = reactions

    def write(self, filename = None):
        """
        write out data file

        TODO - sort data entries first?
        """
        if filename is None:
            filename = self.filename
        with open(filename, 'wt') as f:
            f.write(f'VERSION {self.version:8d}\n')
            f.write(f'FORMAT  {self.format:8d}\n')
            f.write(f'COMMENT {len(self.comment):8d}\n')
            for c in self.comment:
                f.write(f'{c}\n')
            f.write(f'NUCLEI  {len(self.nuclei):8d}\n')
            f.write(f'   Z   A   E  SW               ME               EL   gfunc ngconst <name>\n')
            f.write(f'<weight constants>\n')
            for n in self.nuclei:
                f.write(f' {n.iso.Z:3d} {n.iso.A:3d} {n.iso.E:3d} {n.sweight:3.0f} {n.ME:16.9e} {n.EL:16.9e} {n.gfunc:7d} {len(n.gconst):7d} {n.iso.Name():s}\n')
                nc = len(n.gconst)
                for i in range(nc):
                    f.write(f'{n.gconst[i]:16.9e}')
                    if (i // 5 == 4 and nc > i+1) or i == nc - 1:
                        f.write('\n')
            f.write(f'REACTS  {len(self.reactions):8d}\n')
            f.write(f'REACTION REFERENCE/COMMENT\n')
            f.write(f'   NIN  NOUT  KIND  PRIO     FORMULA  NCONST\n')
            f.write(f'            TMIN            TMAX            DMIN            DMAX\n')
            f.write(f' NUM   Z   A   E\n')
            f.write(f'<constants>\n')
            for r in self.reactions:
                f.write(f'{r.reaction} {r.reference}\n')
                f.write(f'{len(r.educts):6d}{len(r.products):6d}{r.kind:6d}{r.priority:6d}{r.formula:12d}{len(r.constants):8d}\n')
                f.write(f'{r.tmin:16.9e}{r.tmax:16.9e}{r.dmin:16.9e}{r.dmax:16.9e}\n')
                for i in itertools.chain(r.educts, r.products):
                    f.write(f'{i.n:4d}{i.iso.Z:4d}{i.iso.A:4d}{i.iso.E:4d}\n')
                nc = len(r.constants)
                for i in range(nc):
                    f.write(f'{r.constants[i]:16.9e}')
                    if (i // 5 == 4 and nc > i+1) or i == nc - 1:
                        f.write('\n')

    @classmethod
    def from_bdat(cls, bdat):
        """
        construct ion list
        add missing ions
        add hard-code kepler rates  (later)
        add weak rates (much later)
        """
        rdat = RDat()
        rdat.comment += [
            f'Generated from:',
            f'  BDAT:         {bdat.version_name}',
            f'  Version:      {bdat.version:>6d}',
            f'  File:         {bdat.filename}',
            f'  Generated at: {time.asctime()}',
            f'  Generated by: {os.getlogin()}',
            f'  Generated on: {socket.gethostname()}',
            ]
        nuc = Nucleus(
            iso = ion(N=1, Z=0, E=0),
            sweight = 2,
            ME = 8.07131714,
            EL = 0,
            gfunc = 0,
            gconst = [],
            )
        rdat.nuclei.append(nuc)
        nuc = Nucleus(
            iso = ion(N=0, Z=1, E=0),
            sweight = 2,
            ME = 7.28897059,
            EL = 0,
            gfunc = 0,
            gconst = [],
            )
        rdat.nuclei.append(nuc)
        nuc = Nucleus(
            iso = ion(N=2, Z=2, E=0),
            sweight = 1,
            ME = 2.42491561,
            EL = 0,
            gfunc = 0,
            gconst = [],
            )
        rdat.nuclei.append(nuc)
        nuc = Nucleus(
            iso = ion(N=4, Z=4, E=0),
            sweight = 1,
            ME = 4.941671,
            EL = 0,
            gfunc = 0,
            gconst = [],
            )
        rdat.nuclei.append(nuc)

        for record in bdat.data:
            rdat.add_bdat_record(record)

        #raise NotImplementedError()
        return rdat

    lrevpar = (False, True, False, False, True, True, False, False, False, False)

    def add_bdat_record(self, record):
        assert isinstance(record, BDatRecord)
        Z, A, E = record.ion.ZAE()
        N = A - Z
        iso = ion(Z=Z, N=N, E=E)
        ME = record.q - 8.07131714 * N - 7.28897059 * Z
        if iso == 'c12g':
            ME = 0
        sweight = record.sa[0]
        gfunc = 0
        gconst = []
        if record.sa[1] != 0:
            gfunc += 1
            assert len(record.sa) == 6
            gconst += record.sa[1:].tolist()
        if record.ist > 0:
            gfunc += 2
            gconst += record.gs.tolist()
        nuc = Nucleus(
            iso = iso,
            sweight = sweight,
            ME = ME,
            EL = 0,
            gfunc = gfunc,
            gconst = gconst,
            )
        self.nuclei.append(nuc)
        for i,rate in enumerate(record.rates):
            if rate.ic[1] == rate.ic[0] == 0:
                continue
            kind = 0
            # determine which rates have reverse rates --> kind || 1
            #    !!! reverse formula varies based on type
            # determine which rates are weak rates --> kind || 2
            reaction = rate.creac
            reference = rate.cref
            constants = rate.c
            formula = rate.ic[0]
            priority = 0
            tmin = dmin = 0
            tmax = dmax = 1e99
            educts = [IonGroup(1, iso)]
            rev = lrevpar[i]:
            if i == 2:
                # weak decay
                kind += 2
                products = [IonGroup(1, iso + DecIon('b-'))]
                if formula in (6, 8,):
                    products, educts = educts, products
                elif formula in (7, 20, 28):
                    pass
                else:
                    raise Exception('Unknown weak decay')
            else:
                if i in (34,35,36,37):
                    tmin = 1e8
                    tmax = 1e10
                    formula -= 20
                if i in (14,15,16,17):
                    tmin = 1e7
                if rev and formula in (14,15,):
                    formula += 30
                if formula == 17:
                    kind += 2
                    raise (16,17)
            self.reactions.append(
                Reaction(
                   ReactionSide(educts),
                   ReactionSide(products),
                   formula,
                   constants,
                   kind = kind,
                   reference = reference,
                   reaction = reaction,
                   tmin = tmin,
                   tmax = tmax,
                   dmin = dmin,
                   dmax = dmax,
                   )
                )
