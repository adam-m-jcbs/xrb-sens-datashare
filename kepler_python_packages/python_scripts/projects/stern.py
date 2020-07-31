"""
Python tool to map stern data to KEPLER link file
"""

from fortranfile import FortranReader, FortranWriter
import numpy as np
import isotope

class BECData(object):
    """
    Read BEC stellar model from Tom Tauris
    """
    NMOX = 5
    def __init__(self, filename, model = None, nr = None):
        self.filename = filename
        if nr is -1:
            print('loading last model in file')
        if nr is not None:
            assert nr > 0 or nr == -1, 'record numbers start at 1'
        if nr is not None or model is not None:
            self.load(model=model, nr=nr)

    @classmethod
    def find_model(cls, f, model = None, nr = None):
        # search model
        if nr is not None and nr > 0:
            f.skip(nr-1)
            f.load()
        elif nr == -1:
            # load last model
            while not f.eof():
                f.load()
        else:
            nr = 0
            while True:
                f.load()
                nr += 1
                nvers = f.peek_i4()
                if not nvers in (10003, 10002, 10001, 10000):
                    offset = 8
                else:
                    offset = 12
                modell = f.peek_i4(offset=offset)
                if modell < model:
                    continue
                elif modell == model:
                    break
                elif modell > model:
                    raise Exception('model not found')

    def load(self, model = None, nr = None):
        with FortranReader(self.filename) as f:
            self.find_model(f, model=model, nr=nr)
            self._load(f)

    def toc(self):
        with FortranReader(self.filename) as f:
            i = 0
            while not f.eof():
                f.load()
                i += 1
                self._load(f, silent = True)
                print('{nr:4d} {model:6d} {time:12.5e} {mass:12.5e} {tc:12.5e} {dc:12.5e}'.format(
                    nr = f.rpos,
                    model = self.modell,
                    mass = self.gms,
                    time = self.time,
                    dc = self.t[0],
                    tc = self.ro[0],
                    ))

    def _load(self, f, silent = False):

        # load model
        nvers = f.peek_i4()

        if nvers in (10003, 10002, 10001, 10000):
            nvers = f.get_i4()
        else:
            nvers = 0

        self.nvers = nvers
        self.gms = f.get_f8()
        self.modell = f.get_i4()
        self.dtn = f.get_f8()
        self.time = f.get_f8()
        self.n = f.get_i4()
        n = self.n

        if not silent:
            print('[load] Loading nr {}, model {}, version {}, M = {} M_sun'.format(\
                        f.rpos, self.modell, nvers, self.gms))

        if nvers == 10003:
            self.dncore = f.get_f8()
            self.nsp1 = f.get_i4()
            self.windmd = f.get_f8()
            self.dntotal = f.get_f8()
            self.dtalt = f.get_f8()
            raise NotImplementedError()
        elif nvers in (10002, 10001):
            self.n1 = f.get_i4()
            self.nsp1 = f.get_i4()
            self.windmd = f.get_f8()
            self.vvcmax = f.get_f8()
            self.dtalt = f.get_f8()
        elif nvers in (10000, 0):
            self.n1 = f.get_i4()
            self.nsp1 = f.get_i4()
            self.ilow = f.get_i4()
            self.ihigh = f.get_i4()
        else:
            raise NotImplementedError('No such version.')

        if nvers in (10003, 10002):
            self.yzi = np.ndarray(n, dtype = (np.str_, 1))
            self.u = np.ndarray(n, dtype = np.float64)
            self.r = np.ndarray(n, dtype = np.float64)
            self.ro = np.ndarray(n, dtype = np.float64)
            self.t = np.ndarray(n, dtype = np.float64)
            self.sl = np.ndarray(n, dtype = np.float64)
            self.e = np.ndarray(n, dtype = np.float64)
            self.al = np.ndarray(n, dtype = np.float64)
            self.vu = np.ndarray(n, dtype = np.float64)
            self.vr = np.ndarray(n, dtype = np.float64)
            self.vro = np.ndarray(n, dtype = np.float64)
            self.vt = np.ndarray(n, dtype = np.float64)
            self.vsl = np.ndarray(n, dtype = np.float64)
            self.dm = np.ndarray(n, dtype = np.float64)
            self.bfbr = np.ndarray(n, dtype = np.float64)
            self.bfbt = np.ndarray(n, dtype = np.float64)
            self.bfvisc = np.ndarray(n, dtype = np.float64)
            self.bfdiff = np.ndarray(n, dtype = np.float64)
            self.ibflag = np.ndarray(n, dtype = np.int32)
            self.bfq = np.ndarray(n, dtype = np.float64)
            self.bfq0 = np.ndarray(n, dtype = np.float64)
            self.bfq1 = np.ndarray(n, dtype = np.float64)
            self.ediss = np.ndarray(n, dtype = np.float64)
            self.cap = np.ndarray(n, dtype = np.float64)
            self.diff = np.ndarray(n, dtype = np.float64)
            self.dg = np.ndarray(n, dtype = np.float64)
            self.d = np.ndarray((self.NMOX, n), dtype = np.float64)
            for k in range(n):
                self.yzi[k] = f.get_sn(length=1)
                (self.u[k], self.r[k], self.ro[k],
                    self.t[k], self.sl[k], self.e[k], self.al[k],
                    self.vu[k], self.vr[k], self.vro[k], self.vt[k],
                    self.vsl[k], self.dm[k], self.bfbr[k],
                    self.bfbt[k], self.bfvisc[k], self.bfdiff[k]) = \
                    f.get_f8n(17)
                self.ibflag[k] = f.get_i4n()
                (self.bfq[k], self.bfq0[k], self.bfq1[k],
                     self.ediss[k], self.cap[k], self.diff[k],
                     self.dg[k]) = \
                     f.get_f8n(7)
                self.d[:,k] = f.get_f8n(self.NMOX)
            self.ypstmp = f.get_f8n((self.nsp1-1, n))
            self.vertmp = f.get_f8n(self.nsp1)
            self.xnint = np.ndarray(n, dtype=np.float64)
            self.istory = np.ndarray(n, dtype=np.int32)
            self.insrt = np.ndarray(n, dtype=np.int32)
            self.indel = np.ndarray(n, dtype=np.int32)
            for k in range(n):
                self.xnint[k] = f.get_f8n()
                self.istory[k] = f.get_i4n()
                self.insrt[k] = f.get_i4n()
                self.indel[k] = f.get_i4n()
            ions = (' n  H1  D  3He  4He  6Li 7Li 7Be 9Be 8B 10B 11B 11C '+
                    '12C 13C 12N 14N 15N 16O 17O 18O 20Ne 21Ne 22Ne 23Na '+
                    '24Mg 25Mg 26Mg 27Al 28Si 29Si 30Si 56Fe 19F 26Al').split()
            self.ions = np.array([isotope.ion(i) for i in ions])
            assert len(ions) == self.nsp1
        elif nvers in (10001,):
            self.yzi = np.ndarray(n, dtype = (np.str_, 1))
            self.u = np.ndarray(n, dtype = np.float64)
            self.r = np.ndarray(n, dtype = np.float64)
            self.ro = np.ndarray(n, dtype = np.float64)
            self.t = np.ndarray(n, dtype = np.float64)
            self.sl = np.ndarray(n, dtype = np.float64)
            self.e = np.ndarray(n, dtype = np.float64)
            self.al = np.ndarray(n, dtype = np.float64)
            self.vu = np.ndarray(n, dtype = np.float64)
            self.vr = np.ndarray(n, dtype = np.float64)
            self.vro = np.ndarray(n, dtype = np.float64)
            self.vt = np.ndarray(n, dtype = np.float64)
            self.vsl = np.ndarray(n, dtype = np.float64)
            self.dm = np.ndarray(n, dtype = np.float64)
            self.diff = np.ndarray(n, dtype = np.float64)
            self.dg = np.ndarray(n, dtype = np.float64)
            self.d = np.ndarray((self.NMOX, n), dtype = np.float64)
            for k in range(n):
                self.yzi[k] = f.get_sn(length=1)
                (self.u[k], self.r[k], self.ro[k],
                    self.t[k], self.sl[k], self.e[k], self.al[k],
                    self.vu[k], self.vr[k], self.vro[k], self.vt[k],
                    self.vsl[k], self.dm[k], self.diff[k],
                    self.dg[k]) = f.get_f8n(15)
                self.d[:,k] = f.get_f8n(self.NMOX)
            self.ypstmp = f.get_f8n((self.nsp1-1, n))
            self.vertmp = f.get_f8n(self.nsp1)
            self.xnint = np.ndarray(n, dtype=np.float64)
            self.istory = np.ndarray(n, dtype=np.int32)
            self.insrt = np.ndarray(n, dtype=np.int32)
            self.indel = np.ndarray(n, dtype=np.int32)
            for k in range(n):
                self.xnint[k] = f.get_f8n()
                self.istory[k] = f.get_i4n()
                self.insrt[k] = f.get_i4n()
                self.indel[k] = f.get_i4n()
            ions = (' n  H1  D  3He  4He  6Li 7Li 7Be 9Be 8B 10B 11B 11C '+
                    '12C 13C 12N 14N 15N 16O 17O 18O 20Ne 21Ne 22Ne 23Na '+
                    '24Mg 25Mg 26Mg 27Al 28Si 29Si 30Si 56Fe 19F 26Al').split()
            self.ions = np.array([isotope.ion(i) for i in ions])
            assert len(ions) == self.nsp1
        elif nvers in (10001,):
            self.yzi = np.ndarray(n, dtype = (np.str_, 1))
            self.u = np.ndarray(n, dtype = np.float64)
            self.r = np.ndarray(n, dtype = np.float64)
            self.ro = np.ndarray(n, dtype = np.float64)
            self.t = np.ndarray(n, dtype = np.float64)
            self.sl = np.ndarray(n, dtype = np.float64)
            self.e = np.ndarray(n, dtype = np.float64)
            self.al = np.ndarray(n, dtype = np.float64)
            self.vu = np.ndarray(n, dtype = np.float64)
            self.vr = np.ndarray(n, dtype = np.float64)
            self.vro = np.ndarray(n, dtype = np.float64)
            self.vt = np.ndarray(n, dtype = np.float64)
            self.vsl = np.ndarray(n, dtype = np.float64)
            self.dm = np.ndarray(n, dtype = np.float64)
            self.diff = np.ndarray(n, dtype = np.float64)
            self.dg = np.ndarray(n, dtype = np.float64)
            self.d = np.ndarray((self.NMOX, n), dtype = np.float64)
            for k in range(n):
                self.yzi[k] = f.get_sn(length=1)
                (self.u[k], self.r[k], self.ro[k],
                    self.t[k], self.sl[k], self.e[k], self.al[k],
                    self.vu[k], self.vr[k], self.vro[k], self.vt[k],
                    self.vsl[k], self.dm[k], self.diff[k],
                    self.dg[k]) = f.get_f8n(15)
                self.d[:,k] = f.get_f8n(self.NMOX)
            self.ypstmp = np.zeros((self.nsp1-1, n))
            self.ypstmp[:,self.ilow-1:self.ihigh] = f.get_f8n((self.nsp1-1, self.ihigh-self.ilow + 1))
            self.vertmp = f.get_f8n(self.nsp1)
            self.xnint = f.get_f8n(n)
            ions = (' n  H1  D  3He  4He  6Li 7Li 7Be 9Be 8B 10B 11B 11C '+
                    '12C 13C 12N 14N 15N 16O 17O 18O 20Ne 21Ne 22Ne 23Na '+
                    '24Mg 25Mg 26Mg 27Al 28Si 29Si 30Si 56Fe 19F 26Al').split()
            self.ions = np.array([isotope.ion(i) for i in ions])
            assert len(ions) == self.nsp1
        elif nvers in (0, ):
            self.yzi = np.ndarray(n, dtype = (np.str_, 1))
            self.u = np.ndarray(n, dtype = np.float64)
            self.r = np.ndarray(n, dtype = np.float64)
            self.ro = np.ndarray(n, dtype = np.float64)
            self.t = np.ndarray(n, dtype = np.float64)
            self.sl = np.ndarray(n, dtype = np.float64)
            self.e = np.ndarray(n, dtype = np.float64)
            self.vu = np.ndarray(n, dtype = np.float64)
            self.vr = np.ndarray(n, dtype = np.float64)
            self.vro = np.ndarray(n, dtype = np.float64)
            self.vt = np.ndarray(n, dtype = np.float64)
            self.vsl = np.ndarray(n, dtype = np.float64)
            self.dm = np.ndarray(n, dtype = np.float64)
            for k in range(n):
                self.yzi[k] = f.get_sn(length=1)
                (self.u[k], self.r[k], self.ro[k],
                    self.t[k], self.sl[k], self.e[k],
                    self.vu[k], self.vr[k], self.vro[k], self.vt[k],
                    self.vsl[k], self.dm[k]) = f.get_f8n(12)
            self.ypstmp = np.zeros((self.nsp1-1, n))
            self.ypstmp[:,self.ilow-1:self.ihigh] = f.get_f8n((self.nsp1-1, self.ihigh-self.ilow+1))
            self.vertmp = f.get_f8n(self.nsp1)
            self.xnint = f.get_f8n(n)
            ions = (' n  H1  D  3He  4He  6Li 7Li 7Be 9Be 8B 10B 11B 11C '+
                    '12C 13C 12N 14N 15N 16O 17O 18O 20Ne 21Ne 22Ne 23Na '+
                    '24Mg 25Mg 26Mg 27Al 28Si 29Si 30Si 56Fe 19F 26Al').split()
            self.ions = np.array([isotope.ion(i) for i in ions])
            assert len(ions) == self.nsp1

            self.al = np.zeros(n)
            self.difi = np.zeros(n)
            self.dg = np.zeros(n)
            self.d = np.zeros((NMOX, n))
        f.assert_eor()

        # add missing species (poor idea)
        others = 1 - np.sum(self.ypstmp, axis=0)
        self.yps = np.append(self.ypstmp, others[np.newaxis,:], axis=0)
        del self.ypstmp

        self.calcai()
        self.al[0] = 1.e-40
        self.aw = self.al / self.ai

    def calcai(self):
        ai = np.empty_like(self.dm)
        ai[0] = 1e-20
        ri = self.r[:-1]
        ra = self.r[1:]
        dm = self.dm[:-1]
        rai=ra*ri
        ra2=ra**2
        ri2=ri**2
        rm2=ri2+rai+ra2
        ai[1:]=0.4*dm*(ri2**2+rai*rm2+ra2**2)/rm2
        self.ai = ai

    def abu(self, ion):
        ii = np.where(self.ions == isotope.ion(ion))[0]
        return self.yps[ii]

def extract_model(filein, fileout, nr = None, model = None):
    with FortranReader(filein) as f:
        BECData.find_model(f, model=model, nr=nr)
        data = f.get_data()
        byteorder = f.byteorder
    with FortranWriter(fileout, byteorder = byteorder) as f:
        f.write_data(data)
