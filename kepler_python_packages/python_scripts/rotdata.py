'''
Python Module to read rotation data
'''

from fortranfile import FortranReader
import physconst
import os.path

import numpy as np
from numpy.linalg import norm


from logged import Logged
from utils import CachedAttribute, cachedmethod
from loader import loader, _loader
from kepdump import calcai, KepDumpSlice

from dsiplot import fw_min, fw_mint

# for debugging (disable caching)
CachedAttribute = property
cachedmethod = lambda x:x

class RotData(Logged):
    """
    Interface to load KEPLER nu data files.
    """
    def __init__(self,
                 filename,
                 zerotime = True,
                 silent = False):
        """
        Constructor; requires file name.

        PARAMETERS:
        zerotime (True):
            incorporate zerotime in time
        silent (False):
            reduce output
        """
        self.setup_logger(silent)
        filename = os.path.expanduser(filename)
        self.filename = os.path.expandvars(filename)
        self.file = FortranReader(self.filename)
        self._load()
        self.file.close()
        self.close_logger()

    def _load(self, zerotime = True):
        """
        Open file, call load data, time the load, print out diagnostics.
        """
        self.setup_logger()
        self.logger_file_info(self.file)
        self.data=[]
        while not self.file.eof():
            record = RotRecord(self.file)
            self.data.append(record)
        self.nmodels = len(self.data)

        self.logger_load_info(self.data[0].nvers,
                              self.data[ 0].ncyc,
                              self.data[-1].ncyc,
                              self.nmodels)
        if zerotime:
            self._remove_zerotime()
        self.close_logger()

    def _remove_zerotime(self, verbose = True):
        """
        Detect and remove resets of time.
        """
        zerotime = np.float64(0)
        time0 = self.data[0].time
        for i in range(1,self.nmodels):
            if self.data[i].time < time0:
                zerotime = self.data[i-1].time
                if verbose:
                    self.logger.info('@ model = {:8d} zerotime was set to {:12.5g}.'.format(
                        int(self.data[i].ncyc),
                        float(zerotime)))
            time0 = self.data[i].time
            self.data[i].time = time0 + zerotime

    @CachedAttribute
    def time(self):
        """
        Return time array. [s]
        """
        return np.array([rec.time for rec in self.data])

    @CachedAttribute
    def dt(self):
        """
        Return dt array. [s]
        """
        return np.array([rec.dt for rec in self.data])

    @cachedmethod
    def timecc(self, offset = 0.):
        """
        Return time till core collapse. [s]

        Set to 'None' to keep last dt as offset.
        """
        dt = (self.dt).copy()
        if offset is not None:
            dt[-1] = offset
        return ((dt[::-1]).cumsum())[::-1]

    @CachedAttribute
    def ncyc(self):
        """
        Return list of cycle numbers
        """
        return np.array([rec.ncyc for rec in self.data])

    @CachedAttribute
    def jm(self):
        """
        Return list of grid sizes

        Note ::
          each record contains jm + 1 data entries in spatial direction
        """
        return np.array([rec.jm for rec in self.data])

    @CachedAttribute
    def xm(self):
        """
        Return zone masses. [g]

        xm[0] is mass of 'core', if any.
        """
        n = np.max(self.jm) + 1
        xm = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            xm[i, :rec.jm + 1] = rec.xm
        return xm

    @CachedAttribute
    def ym(self):
        """
        Return zone surface mass coordiantes. [g]

        Mass coordinate as counted from surface.
        """
        n = np.max(self.jm) + 1
        ym = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            ym[i, :rec.jm + 1] = rec.ym
        return ym

    @CachedAttribute
    def zm(self):
        """
        Return zone mass coordinates. [g]

        zm[0] is mass of 'core', if any.
        """
        n = np.max(self.jm) + 1
        zm = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            zm[i, :rec.jm + 1] = rec.zm
        return zm

    @CachedAttribute
    def rn(self):
        """
        Return zone radii. [cm]

        xm[0] is mass of 'core', if any.
        """
        n = np.max(self.jm) + 1
        rn = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            rn[i, :rec.jm + 1] = rec.rn
        return rn

    @CachedAttribute
    def dn(self):
        """
        Return zone densities. [g/cm**3]
        """
        n = np.max(self.jm) + 1
        dn = np.zeros((len(self.data), n), dtype = np.float64)
        for i, rec in enumerate(self.data):
            dn[i, :rec.jm + 1] = rec.dn
        return dn

load = loader(RotData, __name__ + '.load')
_load = _loader(RotData, __name__ + '.load')
loadrot = load

class RotRecord(KepDumpSlice):
    """
    Load individual nu record from nu file.
    """
    def __init__(self, file, data = True):
        self.file = file
        self._load(data)

        self.geemult = 1

        self.gee = self.geemult * physconst.GRAV
        self.relmult = 0

    def _load(self,data=True):
        # Version 10000
        #  WRITE(iunit) nvers,ncyc,timesec,dt,jm,
        # &     rn(0), zm(0), xm(1:jm), xmlost,
        # &     dn(1:jm),
        # &     pn(1:jm), gamma1(1:jm),
        # &     angw0x, angw(1:jm,1),awcorotx,
        # &     angw0y, angw(1:jm,2),awcoroty,
        # &     angw0z, angw(1:jm,3),awcorotz,
        # &     difi(0:jm),iconv(0:jm),
        # &     angd(0:jm,1:nangmd),bfvisc(0:jm),
        # &     bfbr(0:jmz), bfbt(0:jmz),
        # &     sv(0:jm)

        f = self.file
        f.load()
        self.nvers = f.get_i4()
        self.ncyc = f.get_i4()
        self.time = f.get_f8()
        self.dt = f.get_f8n()
        self.jm = f.get_i4()
        jm = self.jm

        self.rn0    = f.get_f8n()
        self.xm     = f.get_f8n1d(jm+2)
        self.dn     = f.get_f8n1d0n(jm)
        self.pn     = f.get_f8n1d0n(jm)
        self.gamma1 = f.get_f8n1d0n(jm)
        self.angw   = f.get_f8n([jm+2, 3])
        self.difi   = f.get_f8n1dn(jm+1)
        self.iconv  = f.get_i4n1dn(jm+1, padval=0)
        self.angd   = f.get_f8n([5,jm+1], pad=((0,0),(0,1)), order='C', padval=0.).transpose()
        self.bfvisc = f.get_f8n1dn(jm+1)
        self.bfbr   = f.get_f8n1dn(jm+1)
        self.bfbt   = f.get_f8n1dn(jm+1)
        self.sv     = f.get_f8n1dn(jm+1)
        f.assert_eor()

    @CachedAttribute
    def ym(self):
        """
        Return outer mass coordinate. (g)
        """
        ym = np.empty_like(self.xm)
        ym[-1] = np.nan
        ym[-2] = 0.
        ym[:-3] = np.cumsum(self.xm[1:-2])[::-1]
        return ym

    @CachedAttribute
    def zm(self):
        """
        Return mass coordinate. (g)
        """
        return np.cumsum(self.xm)

    @CachedAttribute
    def zmm(self):
        """
        Return mass coordinate at zone center. (g)
        """
        zm = np.empty_like(self.xm)
        zm[1:] = 0.5 * (self.zm[:-1] + self.zm[1:])
        zm[0] = 0.5 * zm[0]
        return zm

    @CachedAttribute
    def zm_sun(self):
        """
        Return mass coordinate. (Msun)
        """
        return self.zm / physconst.XMSUN

    @CachedAttribute
    def zmm_sun(self):
        """
        Return mass coordinate at zone center. (Msun)
        """
        return self.zmm / physconst.XMSUN

    @CachedAttribute
    def rn(self):
        """
        Return radius coordinate. (xm)
        """
        rn = np.empty_like(self.xm)
        rn[0] = self.rn0
        rn[1:-1] = (rn[0]**3 + 3 / (4 * np.pi)
                    * np.cumsum(self.xm[1:-1] / self.dn[1:-1]))**(1/3)
        rn[-1] = np.nan
        return rn

    @CachedAttribute
    def angi(self):
        """
        Return specific moment of intertia. (cm**2)
        """
        ai = np.empty_like(self.xm)
        ai[self.core_zone] = calcai(0., self.rn0)
        ai[self.zone_slice] = calcai(self.rn[self.lower_slice], self.rn[self.upper_slice])
        ai[self.wind_zone] = np.nan
        return ai

    @CachedAttribute
    def angj(self):
        """
        Return specific moment of intertia. (cm**2)
        """
        aj = np.empty_like(self.xm)
        ii = slice(0,-1)
        aj[ii] = self.angi[ii] * self.angw[ii]
        aj[-1] = np.nan
        return aj

    @CachedAttribute
    def rs(self):
        """
        Schwarzschild radius at zone interface. (cm)
        """
        return 2 * self.zm * self.gee / physconst.CLIGHT**2

    @CachedAttribute
    def grav(self):
        """
        Gravitational acceleration at zone interface. (cm/sec**2)
        """
        g = np.empty_like(self.xm)
        g[:-1] = -self.gee * self.zm[:-1] / (self.rn[:-1] + 1.e-99)**2
        if self.relmult != 0:
            g[:-1] /= 1 - self.rs[:-1] / (self.rn[:-1] + 1.e-99)
        g[-1] = np.nan
        return g

    @CachedAttribute
    def gamma1f(self):
        """
        Adiabatic index gamma1 at zone interface. (1)
        """
        g1 = np.empty_like(self.xm)
        g1[1:-2] = 0.5 * (self.gamma1[1:-2] + self.gamma1[2:-1])
        g1[[0,-1,-2]] = np.nan
        return g1

    @CachedAttribute
    def drf(self):
        """
        Radius differential at zone interface. (cm)
        """
        dr = np.empty_like(self.xm)
        dr[1:-2] = 0.5 * (self.rn[2:-1] - self.rn[:-3])
        dr[[0,-1,-2]] = np.nan
        return dr

    @CachedAttribute
    def xmf(self):
        """
        Mass differential at zone interface. (g)
        """
        xm = np.empty_like(self.xm)
        xm[1:-2] = 0.5 * (self.xm[1:-2] + self.xm[2:-1])
        xm[[0,-1,-2]] = np.nan
        return xm

    @CachedAttribute
    def dp(self):
        """
        Pressure differential at zone interface. (erg/cm**3)
        """
        dp = np.empty_like(self.xm)
        dp[1:-2] = self.pn[2:-1] - self.pn[1:-2]
        dp[[0,-1,-2]] = np.nan
        return dp

    @CachedAttribute
    def dd(self):
        """
        Density differential at zone interface. (g/cm**3)
        """
        dd = np.empty_like(self.xm)
        dd[1:-2] = self.dn[2:-1] - self.dn[1:-2]
        dd[[0,-1,-2]] = np.nan
        return dd

    @CachedAttribute
    def pf(self):
        """
        Pressure at zone interface. (erg/cm**3)
        """
        pf = np.empty_like(self.xm)
        pf[1:-2] = 0.5 * (self.pn[2:-1] + self.pn[1:-2])
        pf[[0,-1,-2]] = np.nan
        return pf

    @CachedAttribute
    def df(self):
        """
        Density at zone interface. (g/cm**3)
        """
        df = np.empty_like(self.xm)
        df[1:-2] = 0.5 * (self.dn[2:-1] + self.dn[1:-2])
        df[[0,-1,-2]] = np.nan
        return df

    @CachedAttribute
    def hpr(self):
        """
        Inverse of Pressure scale height. (1/cm)
        """
        h = np.empty_like(self.xm)
        h[1:-2] = self.dp[1:-2] / (self.pf[1:-2] * self.drf[1:-2] + 1.e-99)
        h[[0,-1,-2]] = np.nan
        return h

    @CachedAttribute
    def dlddlp(self):
        """
        d ln rho / d ln P [inverse of polytropic index]. (1)
        """
        g1 = np.empty_like(self.xm)
        g1[1:-2] = self.pf[1:-2] * self.dd[1:-2] / (self.df[1:-2] * self.dp[1:-2])
        g1[[0,-1,-2]] = np.nan
        return g1

    @CachedAttribute
    def n2(self):
        """
        (BV Frequency)**2. (1/sec**2)
        """
        n2 = np.empty_like(self.xm)
        n2[1:-2] = -self.grav[1:-2] * self.hpr[1:-2] * (1 / self.gamma1f[1:-2] - self.dlddlp[1:-2])
        n2[[0,-1,-2]] = np.nan
        return n2

    @CachedAttribute
    def angwa(self):
        """
        Magnitude of angular velocity. (rad/sec)
        """
        return norm(self.angw, axis=1)

    @CachedAttribute
    def angwaf(self):
        """
        Magnitude of angular velocity at zone interface. (rad/sec)
        """
        # needs to be consistent with angwf
        w = np.empty_like(self.xm)
        w[:-1] =  0.5 * (self.angwa[:-1] + self.angwa[1:])
        w[-1] = np.nan
        return w

    @CachedAttribute
    def angwf(self):
        """
        Angular velocity at zone center. (rad/sec)

        This is tricky as average does not take into account rotations, but
        for anti-alignment rotations become undefined...
        """
        wm = np.empty_like(self.angw)
        wm[:-1,:] = 0.5 * (self.angw[1:,:] + self.angw[:-1,:])
        wm[:-1,:] *= (self.angwaf[:-1] / (norm(wm[:-1,:], axis=1) + 1.e-99))[:,np.newaxis]
        #TODO - use sin(theta) to transition to average for theta > pi/2
        wm[-1,:] = np.nan
        return wm

    @CachedAttribute
    def angdw(self):
        """
        Angular velocity differential. (rad/sec)
        """
        w = np.empty_like(self.angw)
        w[:-1,:] =  self.angw[1:,:] - self.angw[:-1,:]
        w[-1,:] = np.nan
        return w

    @CachedAttribute
    def angdwdr(self):
        """
        Angular velocity gradient. (rad/sec/cm)
        """
        w = np.empty_like(self.angw)
        w[1:-2,:] = self.angdw[1:-2, :] / ((self.drf[1:-2] + 1.e-99)[:, np.newaxis])
        w[[0,-1,-2]] = np.nan
        return w

    @CachedAttribute
    def angdwdra(self):
        """
        Angular velocity gradient magnitude. (rad/sec/cm)
        """
        return norm(self.angdwdr, axis=1)

    @CachedAttribute
    def angpw(self):
        """
        Angular velocity shear parameter. (1)
        """
        p = np.empty_like(self.xm)
        p[1:-2] = self.rn[1:-2] * self.angdwdra[1:-2] / self.angwaf[1:-2]
        p[[0,-1,-2]] = np.nan
        return p

    @CachedAttribute
    def angmwdw(self):
        """
        Angular velocity gradient angle cosine. (1)
        """
        m = np.empty_like(self.xm)
        m[1:-2] = np.einsum('ij,ij->i', self.angwf[1:-2], self.angdwdr[1:-2]) / (
            self.angwaf[1:-2] * self.angdwdra[1:-2] + 1.e-99)
        # m[1:-2] = np.maximum(np.minimum(m[1:-2], 1), -1)
        m[[0,-1,-2]] = np.nan
        return m

    @CachedAttribute
    def angtwdw(self):
        """
        Angular velocity gradient angle cosine. (1)
        """
        t = np.empty_like(self.xm)
        s = norm(np.cross(self.angwf[1:-2], self.angdwdr[1:-2], axis=-1), axis=-1)
        c = np.einsum('ij,ij->i', self.angwf[1:-2,:], self.angdwdr[1:-2])
        t[1:-2] = np.arctan2(s,c)
        t[[0,-1,-2]] = np.nan
        return t

    @CachedAttribute
    def angfmin(self):
        """
        Dynamical instability factor. (1)
        """
        f = np.empty_like(self.xm)
        f[1:-2] = fw_mint(self.angpw[1:-2], self.angtwdw[1:-2])
        f[[0,-1,-2]] = np.nan
        return f

    @CachedAttribute
    def angnw(self):
        """
        Radio of N**2 to W**2. (1)
        """
        n = np.empty_like(self.xm)
        n[1:-2] = self.n2[1:-2] / (self.angwaf[1:-2]**2 + 1.e-99)
        n[[0,-1,-2]] = np.nan
        return n

    @CachedAttribute
    def n2x(self):
        """
        Square of dynamical buyancy frequency. (rad**2/sec**2)
        """
        n = np.empty_like(self.xm)
        n[1:-2] = self.n2[1:-2] + self.angwaf[1:-2]**2 * self.angfmin[1:-2]
        n[[0,-1,-2]] = np.nan
        return n

    @CachedAttribute
    def tdyn(self):
        """
        Dynamical (free-fall) timescale. (sec)
        """
        n = np.empty_like(self.xm)
        n[1:-2] = np.sqrt(self.rnf[1:-2] / self.g[1:-2])
        n[[0,-1,-2]] = np.nan
        return n

    @CachedAttribute
    def n2dyn(self):
        """
        Fundamental frequency squared (1/sec**2)

        from surface wave dispersion relation
        w**2 = g * k * tanh(k h)
        k = 2 pi / lambda
        set
        h = r, lambda = 2 pi r (<==> k = 1/r)
        w**2 = g/r * tanh(1)
        tanh(1) = (e**2-1)/(e**2+1)=0.7615941559557649 ~ 1
        (to be done properly for spherical harmonics)

        For higher-order modes we have
        lambda = 2 pi r / n, n = integer
        w**2 = g * n / r * tanh(n)
        tanh(2) = 0.9640275800758169
        tanh(3) = 0.9950547536867305
        """
        n = np.empty_like(self.xm)
        n[1:-2] = -self.grav[1:-2] / self.rn[1:-2]
        n[[0,-1,-2]] = np.nan
        return n
