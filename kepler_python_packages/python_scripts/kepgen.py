#! /bin/env python3

"""
Module to generate 'generator' (g) files for KEPLER.

TODO - allow generation from KepAbuSet for APPROX and AbuSet for BURN
"""

import os
import os.path
import re

# what-the-hack bugfix
import pwd
os.getlogin = lambda: pwd.getpwuid(os.getuid())[0]

import socket
import datetime
import string
import time
import numpy as np
from collections import Iterable
import sys
import shutil
from configparser import SafeConfigParser
import glob
import psutil
import subprocess

import physconst
from logged import Logged
from human import time2human
from human import version2human
from laneemden import lane_emden_int
from keppar import p as parm
from abuset import AbuSet
from abusets import ScaledSolar, ScaledSolarHelium
from kepion import KepAbuSet
from utils import MultiLoop, float2str, stuple, touch
from explosion import Explosion, Result, State, Link
from kepdump import KepDump
from gch_wh13 import GCHWH13 as GCHAbu
from keputils import mass_string, mass_equal

np.seterr(all='raise')


def get_spin(angular,
             mass,
             metallicity = 0.,
             version = 2,
             silent = False,
             ):
    """
    Compute rotation rate for given mass and metallicity.

    INPUT:
    mass in solar masses
    metalliicity in mass fraction
    angular absolute or fraction

    OUTPUT
    if angular > 1e6 just return angular.
    if angular > 0   scale with maximum rotation
    if angular < 0   scale to metallicity-dependent typical value
    """
    if angular > 1.e6:
        return angular
    if version == 1:
        # old fit (Heger 1998)
        angular = abs(angular)*2.5e53*( mass / 20)**1.65
        print(' [get_spin] using Heger 1998')
    elif version == 2:
        # Now: new fit (Woosley & Heger 2012)
        if metallicity <= 6.53160290478554e-08:
            J20 = 6.79954e52
            print(' [get_spin] using Z=0 value')
        else:
            print(' [get_spin] using Z={:g}'.format(metallicity))
            lm = np.log10(metallicity)
            J20 = (-0.0196226e53*lm**2
                   -0.11150409e53*lm
                   + 0.89179526e53)
        J100 = J20 * (mass/20.)**1.78
        if angular < 0:
            # do metallicity scaling
            # Jt = -J100 * (+0.629960
            #               -0.287965 *
            #               (metallicity/0.02)**(1/3)
            #               )**(1.5)
            Jt = -J100 * (
                +0.629960
                -1.060872130505979 * metallicity**(1/3)
                )**(1.5)
        else:
            Jt = J100
        angular *= Jt
    return angular

def explosion_defaults(parm):
    """
    update default explosion settings
    """
    defaults = dict(
        u = dict(ekin =  0.01e51, scut = 4),
        v = dict(ekin =  0.02e51, scut = 4),
        w = dict(ekin =  0.03e51, scut = 4),
        x = dict(ekin =  0.05e51, scut = 4),
        y = dict(ekin =  0.10e51, scut = 4),
        Z = dict(ekin =  0.20e51, scut = 4),
        A = dict(ekin =  0.30e51, scut = 4),
        a = dict(ekin =  0.40e51, scut = 4),
        b = dict(ekin =  0.50e51, scut = 4),
        B = dict(ekin =  0.60e51, scut = 4),
        c = dict(ekin =  0.70e51, scut = 4),
        d = dict(ekin =  0.80e51, scut = 4),
        C = dict(ekin =  0.90e51, scut = 4),
        e = dict(ekin =  1.00e51, scut = 4),
        f = dict(ekin =  1.10e51, scut = 4),
        D = dict(ekin =  1.20e51, scut = 4),
        E = dict(ekin =  1.50e51, scut = 4),
        F = dict(ekin =  1.80e51, scut = 4),
        G = dict(ekin =  2.40e51, scut = 4),
        H = dict(ekin =  3.00e51, scut = 4),
        I = dict(ekin =  5.00e51, scut = 4),
        J = dict(ekin = 10.00e51, scut = 4),
        #K = dict(ekin = 20.00e51, scut = 4),
        #L = dict(ekin = 30.00e51, scut = 4),
        M = dict(ekin =  0.30e51, scut = 0),
        N = dict(ekin =  0.60e51, scut = 0),
        O = dict(ekin =  0.90e51, scut = 0),
        P = dict(ekin =  1.20e51, scut = 0),
        Q = dict(ekin =  1.50e51, scut = 0),
        R = dict(ekin =  1.80e51, scut = 0),
        S = dict(ekin =  2.40e51, scut = 0),
        T = dict(ekin =  3.00e51, scut = 0),
        U = dict(ekin =  5.00e51, scut = 0),
        V = dict(ekin = 10.00e51, scut = 0),
        #W = dict(ekin = 20.00e51, scut = 0),
        #X = dict(ekin = 30.00e51, scut = 0),
    )

    exp = parm['exp']
    if exp in defaults:
        default = defaults[exp]
    else:
        default = dict(ekin = 1.2e51, scut = 4)
    p = dict(parm)
    parm.update(default)
    parm.update(p)



# def rk4(x0,y0,yp,dx,*args):
#     """
#     4th order Runge Kutta integration step
#     """

#     k1 = dx * yp(x0, y0, *args)
#     k2 = dx * yp(x0 + 0.5 * dx, y0 + 0.5 * k1, *args)
#     k3 = dx * yp(x0 + 0.5 * dx, y0 + 0.5 * k2, *args)
#     k4 = dx * yp(x0 + dx, y0 + k3, *args)

#     return y0 + (k1 + 2 * (k2 + k3) + k4) / 6

class Theta(Logged):
    """
    Theta function of Lane-Emden Equation.
    """
    def __init__(self,
                 n = 3.,
                 w = 0.,
                 silent = False):
        """
        Set up and integrate theta from Lane Emden Equation.

        PARAMETERS:
            n = 3.
                polytropic index
            w = 0.
                rotation parameter
            silent = False
                verbosity

        NOTES:
        Classical LE:
        1/z**2 d/dz (z**2 d/dz theta(z)) + theta(z)**n = 0
        for small z one can approximate
        theta(z) = 1. + (-1/6.)*z**2 + (n/120.)*z**4 + O(z**6)
        Therefore lim(z)-->0 d**2 theta(z)/d z**2 = -1/3

        If we include a constant rotation rate Omega the equation becomes
        1/z**2 d/dz (z**2 d/dz theta(z)) + theta(z)**n - w = 0
        where
        w = W/rho_c
        W = 2 Omega**2 / 4 pi G
        for small z one can approximate
        theta(z) = 1. + (w - 1)/6 * z**2 + (1.-w)*(n/120.)*z**4 + O(z**6)
        Therefore lim(z)-->0 d**2 theta(z)/d z**2 = (w-1.)/3


        """
        self.setup_logger(silent)

        self.n = n
        self.w = w

        dz = 0.5**(12)
        fz = 0.5
        acc = 1.e-9
        z0 = -1.
        z  = +1.

        while abs(z - z0) > acc:
            dz *= fz
            ndata, data = lane_emden_int(dz, n, w)

            z1 = dz * ndata
            t0 = data[ndata-1,0]
            t1 = data[ndata  ,0]
            d0 = data[ndata-1,1]
            d1 = data[ndata  ,1]
            z0 = z
            f = t1 / (t0-t1)
            z = z1 + f * dz
            t = t1 + f * (t1-t0)
            d = d1 + f * (d1-d0)

            self.logger.info(r"{:20.17f} {:12.9f} {:12.5e}".format(z, d, z - z0))

        # calculate the coefficients for a third order interpolation
        data10 = data[1:ndata+1, 0]
        data00 = data[0:ndata  , 0]
        data11 = data[1:ndata+1, 1]
        data01 = data[0:ndata  , 1]

        self.coeff = np.ndarray([4,ndata])
        self.coeff[0,...] = data00
        self.coeff[1,...] = data01 * dz
        self.coeff[2,...] = 3. * data10 - data11 * dz - 3. * data00 - 2. * data01 * dz
        self.coeff[3,...] = data11 * dz - 2. * data10 + 2. * data00 + data01 * dz

        self.data = data[0:ndata+1,:]
        self.zmax = z
        self.dz = dz
        self.ndata = ndata

        self.tmax = t
        self.dmax = d

        self.close_logger(timing='integration completed in')

    def __call__(self, z):
        """
        return theta(z)
        """

        rh = 1. / self.dz
        zh = np.minimum(z,self.zmax) * rh
        ii = np.floor(zh)
        f = zh-ii
        ii = np.array(ii,dtype='i8')

        a = self.coeff[0,:]
        b = self.coeff[1,:]
        c = self.coeff[2,:]
        d = self.coeff[3,:]

        return (a[ii]+f*(b[ii]+f*(c[ii]+f*d[ii])),
                (b[ii]+f*(2.*c[ii]+f*3.*d[ii]))*rh)


class LaneEmdenGrid(Logged):
    """
    Compute Lane Emden stellar grid.
    """
    def __init__(self,
                 mass = 10.,
                 rho_c = 0.1,
#                 mu = 1.3342, # helium star
                 mu = 0.59202714, # Pop III star
                 n = 3.,
                 ngrid = 500,
                 m_core = 0.001 * physconst.Kepler.solmass,
                 cutoff = 1.e-9,
                 rho_change = 0.1,
                 dm_change = 0.1,
                 theta = None,
                 silent = False,
                 geemult = 1.,
                 Omega = 0.):
        """
        Set up Lane Emden Stellar Grid.

        PARAETERS (default):
            mass = 10.
                mass of star in solar masses
            rho_c = 0.1
                central density
            mu = 0.59202714,
                mean moelcular weight
            n = 3.
                polytropic index
            ngrid = 500
                initila number of grid points
            m_core = 0.001 * physconst.Kepler.solmass
                maximum mass of central zone
            cutoff = 1.e-9
                desnity cutoff
            rho_change = 0.1
                maximum change of density between zones
            dm_change = 0.1
                maximum change of mass in surface zones
            theta = None
                theta function object (for reuse)
            silent = False
                verbosity
            geemult = 1.
                multiplier on gravity
            Omega = 0.
                angular velocity of star

        """
        start_time = datetime.datetime.now()
        self.setup_logger(silent)

        # define special power
        def xpow(x, n):
            return np.sign(x) * np.abs(x) ** n

        # define constants
        arad = physconst.Kepler.a
        grav = physconst.Kepler.gee * geemult
        rk   = physconst.Kepler.rk

        # calculate rotation parameters
        W = 2. * Omega**2 / (4. * np.pi * grav)
        w = W / rho_c
        w3 = w / 3.

        # use old or update
        if (theta is None) or (theta.n != n) or (theta.w != w):
            theta = Theta(n, w)
        self.theta = theta
        zmax = theta.zmax
        dmax = theta.dmax

        # calculates mass of star for polytrope
        mstar = mass * physconst.Kepler.solmass

        rfac = (- mstar / (4. * np.pi * rho_c * zmax**2 * (dmax - zmax * w3)))**(1./3.)
        mfac = -4. * np.pi * rfac**3 * rho_c
        vfac = 4. / 3. * np.pi

        radius = rfac * zmax
        self.radius = radius

        z = np.linspace(0, zmax, ngrid+1)
        t,d = theta(z)

        self.logger.info('USING mu = {:12.8f}'.format(mu))

        # refine core
        m0 = mfac * z[1]**2 * ( d[1] - z[1] * w3)
        nzone = ngrid

        while m0 > m_core:
           # let's assume in the center rho is about constant, then m~r^3,
           # r=rfac*x
           # *** needs to be refined ***
           # so, to have equal zone masses, one would use about equal
           # volumes, i.e., equal values or z^3.
           # Split inner two zones into three:
           znew = (np.array([1.,2.])/3.)**(1./3.) * z[2]
           tnew, dnew = theta(znew)
           z = np.insert(z,1,0)
           t = np.insert(t,1,0)
           d = np.insert(d,1,0)
           z[1:3] = znew
           t[1:3] = tnew
           d[1:3] = dnew
           nzone += 1
           m0 = mfac * z[1]**2 * (d[1] - z[1] * w3 )
           # it turns out that for reasonable values of m_core this is never called...

           # for small z one can approximate
           # theta(z) = 1. + (w-1)/6 * z**2 + (1-w)*(n/120.)*z**4 + O(z**6)

        iter = 0
        ii_refine = np.array([0])
        while ii_refine.size > 0:

            #   nzone = N_ELEMENTS(x)-1

            # create physical mass profile from polytrope
            m = mfac * z**2 * ( d - z * w3)
            r = z * rfac

            # Volume profile
            V = vfac * r**3
            m1 = m[1:]
            m0 = m[0:-1]
            r1 = r[1:]
            r0 = r[0:-1]
            dM = m1 - m0
            dV = vfac * (r1 - r0) * (r1**2 + r1 * r0 + r0**2)
            rho = dM / dV
            rho[0] = rho_c

            mratio = 2. * abs(dM) / (m1 + m0)
            ii, = np.where(mratio < 1.e-7)

            if len(ii) == 0:
                ii = np.array([len(mratio)-1])

            # # alternative integration method:
            # dRhodV = (rho_c * n * t**(n - 1.) * d)/(4. * np.pi * r**2 * rfac)

            # # normalize
            # rhox = rho_c * t**n
            # rho0 = rhox[:-1]
            # rho1 = rhox[1: ]
            # dRhodV_n0 = dRhodV[:-1] * dV
            # dRhodV_n1 = dRhodV[1: ] * dV


            # # interpolation coefficient
            # ai = rho0[ii]
            # bi = dRhodV_n0[ii]
            # ci = 3. * rho1[ii] - dRhodV_n1[ii] - 3. * rho0[ii] - 2. * dRhodV_n0[ii]
            # di = dRhodV_n1[ii] - 2. * rho1[ii] + 2. * rho0[ii] + dRhodV_n0[ii]

            # alternative integration method:
            iix = np.append(ii, ii[-1] + 1)
            ii0 = ii
            ii1 = ii + 1

            # normalize
            rho0 = rho_c * xpow(t[ii0], n)
            rho1 = rho_c * xpow(t[ii1], n)
            dRhodV = (rho_c * n * xpow(t[iix], n - 1.) * d[iix])/(4. * np.pi * r[iix]**2 * rfac)
            dRhodV_n0 = dRhodV[:-1] * dV[ii]
            dRhodV_n1 = dRhodV[1: ] * dV[ii]

            # interpolation coefficient
            ai = rho0
            bi = dRhodV_n0
            ci = 3. * rho1 - dRhodV_n1 - 3. * rho0 - 2. * dRhodV_n0
            di = dRhodV_n1 - 2. * rho1 + 2. * rho0 + dRhodV_n0

            # integration to find mass of zones
            rho[ii] = ai + bi / 2. + ci / 3. + di / 4.
            dM [ii] = dV[ii] * rho[ii]

            m = dM.cumsum()
            np.insert(m,0,0)

            # outer zone refinement (density)
            rho1 = rho[1:-1]
            rho0 = rho[0:-2]
            ym = np.append((dM[::-1]).cumsum()[::-1],[0.])

            # determines change in density in adjacent zones
            delta_rho = (rho1 - rho0)/(rho1 + rho0)
            # determines where to create new zones
            ii_refine, = np.where(np.logical_and(
                np.logical_or(abs(delta_rho) > rho_change,
                              dM[1:-1]/ym[1:-2] > dm_change),
                              (rho[1:-1] > cutoff)))

            if ii_refine.size > 0:
                for ii in ii_refine[::-1]:
                    i = ii+1
                    znew = 0.5 * (z[i] + z[i+1])
                    tnew,dnew = theta(znew)
                    z = np.insert(z,i+1,znew)
                    t = np.insert(t,i+1,tnew)
                    d = np.insert(d,i+1,dnew)
                istop = 0
                if nzone > 1500:
                    istop=1
                    raise ValueError()

                iter += 1
                nzone += ii_refine.size
                self.logger.info('iteration = {:d}, zones = {:d}'.format(
                    iter,nzone-1))
            else:
                ii_refine = np.array([])

        truncate, = np.where(rho > cutoff)
        rho = np.insert(rho[truncate],0,rho_c)
        dM = dM[truncate]

        # OK, we lost some part of the grid ... the mass
        # so we should re-normalize total mass, maybe ... ?

        totm = dM[::-1].sum()
        scale = mstar / totm
        dM *= scale

        ym = np.append(dM[::-1].cumsum()[::-1],0.) / physconst.Kepler.solmass
        # ... and just to make sure
        ym[0] = mstar
        # we have no mass of "zone 0"
        xm = np.insert(dM,0,0.)

        # create pressure profile
        K = rfac**2 * rho_c**((n - 1.)/n) * 4. * np.pi * grav / (n + 1.)
        P = K * rho**((n + 1.)/n)

        # create temperature profile
        u1 = ((3. * rk * rho)/(4. * mu * arad))**2
        U = (u1 + np.sqrt(u1**2 + (P/arad)**3))**(1./3.)
        t1 = 2. * (U - P/(arad * U))
        T = (np.sqrt((6. * rk * rho)/(mu * arad * np.sqrt(t1))-t1) - np.sqrt(t1) ) * 0.5

        # store values
        self.ngrid = dM.size
        self.xm = xm
        self.ym = ym
        self.tn = T
        self.dn = rho
        self.pn = P
        self.en = U
        self.rn = r[0:max(truncate)+2]
        self.zm = m[0:max(truncate)+2] * scale

        self.z = z[0:max(truncate)+2]
        self.t = t[0:max(truncate)+2]
        self.d = d[0:max(truncate)+2]

        self.jm = max(truncate) + 1

        end_time = datetime.datetime.now()
        load_time = end_time - start_time
        self.logger.info('n   = {}'.format(n))
        self.logger.info('M   = {}'.format(self.zm[-1] / physconst.Kepler.solmass))
        self.logger.info('rho = {}'.format(self.dn[0]))
        self.logger.info('T   = {}'.format(self.tn[0]))
        self.logger.info('R   = {}'.format(self.rn[-1] / physconst.Kepler.solrad))
        self.logger.info('profile created in {:s} seconds.'.format(time2human(load_time)))
        self.close_logger()

class KepEnv(Logged):
    """
    Setup KEPLER environment.

    To derive your own class/environement, it needs to set
    self.dir00    - basic KEPLER directory
    self.progfile - KPELER executable
    self.expfile  - explosion file (usually we may no longer need that)
    self.datadir  - data directory

    INITIALIZATION:
    You may set the parameters on initialization or from environment.

    TODO
    Mabe have kepler.ini in local directory?
    Or home directory?
    """
    def __init__(self,
                 dir00 = None,
                 progfile = None,
                 datadir = None,
                 expfile = None,
                 batchfile = None,
                 version = None,
                 kepler = None,
                 program = None,
                 data = None,
                 explosion = None,
                 local_prog = None,
                 batch_cfg = None,

                 silent = False):
        """
        Default setup for Alex.
        """
        self.setup_logger(silent)

        user = os.getlogin()
        host = socket.gethostname()
        system = os.uname()[0]
        home = os.path.expanduser('~')

        # default values
        if version is None:
            version = os.getenv('KEPLER_VERSION','gfortran')
        if kepler is None:
            kepler = os.getenv('KEPLER_KEPLER','kepler')
        if program is None:
            program = os.getenv('KEPLER_PROGRAM','keplery')
        if data is None:
            data = os.getenv('KEPLER_DATA','local_data')
        if explosion is None:
            explosion = os.getenv('KEPLER_EXPLOSION','explosion.py')
        if local_prog is None:
            local_prog = os.getenv('KEPLER_LOCAL_PROG','./k')
        if batch_cfg is None:
            batch_cfg = os.getenv('KEPLER_BATCH_CFG','batch.cfg')

        # here specific settings could be added
        if host.endswith('.monash.edu'):
            pass

        self.logger.info('Using default KepEnv settings.')

        self.dir00      = dir00      or os.getenv('KEPLER_BASEDIR'  , os.path.join(home, kepler))
        self.progfile   = progfile   or os.getenv('KEPLER_PROGFILE' , os.path.join(self.dir00,version,program))
        self.datadir    = datadir    or os.getenv('KEPLER_DATADIR'  , os.path.join(self.dir00,data))
        self.expfile    = expfile    or os.getenv('KEPLER_EXPFILE'  , os.path.join(home,'python','source',explosion))
        self.batchfile  = batchfile  or os.getenv('KEPLER_BATCHFILE', os.path.join(self.dir00,'batch',batch_cfg))
        self.local_prog = local_prog

        self.close_logger()

    def datafile(self,
                 filename,
                 local_path = None,
                 silent = False):
        """
        find data file from environement, local, or absolute path
        """
        assert filename
        self.setup_logger(silent = silent)
        if os.path.isfile(filename):
            return filename
        if local_path:
            if os.path.isdir(local_path):
                file_name = os.path.join(local_path, filename)
                if os.path.isfile(file_name):
                    return file_name
            path = os.path.dirname(local_path)
            if os.path.isdir(path):
                file_name = os.path.join(path, filename)
                if os.path.isfile(file_name):
                    return file_name
        paths = os.getenv('KEPLER_DATA')
        for path in paths.split(':'):
            if os.path.isdir(path):
                file_name = os.path.join(path, filename)
                if os.path.isfile(file_name):
                    return file_name
        self.logger.error('File "{:s}" not found.'.format(filename))
        self.close_logger()
        raise IOError('File Not Found: ' + filename)


class KepSeries(Logged):
    """
    Define Default KEPLER directories.

    Store Output in object.

    'Outputs' (stores)
    self.dir0
    self.series
    self.suffix
    self.bgdir
    self.dirtarget - only needed to overwrite default generation

    To overwrite you need to provide your own routine that sets these variables.
    """
    def __init__(self, **kwargs):
        """
        Set defaults for Alex historic projects.
        Only allow kw arguments.
        """

        # REWRITE in old style
        # interface/defaults
        kw = dict(kwargs)
        mass = kw.setdefault('mass', None)
        composition = kw.setdefault('composition', None)
        burn = kw.setdefault('burn', None)
        if burn is None:
            burn = kw.setdefault('BURN', False)
        yeburn = kw.setdefault('yeburn', False)
        lburn = kw.setdefault('lburn', False)
        magnet = kw.setdefault('magnet', None)
        angular = kw.setdefault('angular', None)
        massloss = kw.setdefault('massloss', None)
        mu12 = kw.setdefault('mu12', None)
        axion = kw.setdefault('axion', None)
        special = kw.setdefault('special', None)
        kepenv = kw.setdefault('kepenv', KepEnv(silent = True))
        dirsuffix = kw.setdefault('dirsuffix', None)
        dirbase = kw.setdefault('dirbase', None)
        dirtarget = kw.setdefault('dirtarget', None)
        basename = kw.setdefault('basename', None)
        exp = kw.setdefault('exp', None)
        bgdir = kw.setdefault('bgdir', None)
        bdat = kw.setdefault('bdat', None)
        bdatcopy = kw.setdefault('bdatcopy', False)
        genburn = kw.setdefault('genburn', None)
        silent = kw.setdefault('silent', False)
        series_ = kw.setdefault('series', None)
        bgcopy = kw.setdefault('bgcopy', False)
        subdir_ = kw.setdefault('subdir', None)
        projectdir =  kw.setdefault('projectdir', None)
        name =  kw.setdefault('name', None)

    # WIMP_MASS=wimp_mass, $

        self.setup_logger(silent)
        self.mass = mass
        self.exp = exp

        self.kepenv = kepenv

        if genburn is not None:
            xpath,xfile = os.path.split(os.path.expanduser(os.path.expandvars(genburn)))
            if not bgdir:
                bgdir = xpath
                genburn = xfile

        bgdir1 = None

        dir00 = self.kepenv.dir00

        # set defaults
        if composition is None:
            composition = 'solar'
            self.logger.warning('Using {} composition.'.format(composition))
        if not yeburn:
            yeburn = False
            self.logger.info('NOT using yeburn.')
        if not lburn:
            lburn = False
            self.logger.info('NOT using lburn.')
        if yeburn:
            self.logger.info('yeburn forces use of BURN.')
            burn = True
        if lburn:
            self.logger.info('lburn forces use of BURN.')
            burn = True
        if not burn:
            self.logger.info('NOT using BURN.')

        if magnet is None:
            self.logger.info('NOT using magnetic fields.')
        if magnet is not None:
            if angular is None:
                self.logger.info('Using rotation.')
                angular = magnet
        if (angular == 0) or (angular is None):
            self.logger.info('NOT using rotation.')
            angular = None

        # type of grid
        if burn:
            dir1 = 'nuc'
        else:
            dir1 = 'grid'

        suffix = ''
        subdir = None

        # rotation / magnetic fields
        if angular is not None:
            if magnet is not None:
                suffix = 'b'
            else:
                suffix='r'
            if angular < 1.e4:
                suffix += '{:04.0f}'.format(1000.*angular)

        if massloss is not None:
            suffix = '-{:03.0f}'.format(100.*massloss)

        # particle physics
        if mu12 is not None:
            suffix += 'mu{:d}'.format(mu12)
        if axion is not None:
            if issubclass(int,type(axion)):
                snumber = '{:d}'.format(axion)
            else:
                snumber = '{:04.1f}'.format(axion)
            suffix += 'ax' + snumber


        if special is not None:
            for case in special:
                if case == 'z0a1':
                    suffix += 'a'
                else:
                    pass

        # composition
        if composition == 'double':
            series = 'd'
        elif composition == 'solar':
            series = 's'
        elif composition == 'solag89':
            series = 's'
            subdir = os.path.join('solag89', 's'+dir1)
        elif composition == 'solas09':
            series = 's'
            subdir = os.path.join('solas09', 's'+dir1)
        elif composition == 'sollo03':
            series = 's'
            subdir = os.path.join('sollo03', 's'+dir1)
        elif composition == 'sollo09':
            series = 's'
            subdir = os.path.join('sollo09', 's'+dir1)
        elif composition == 'solgn93':
            series = 's'
            subdir = os.path.join('solgn93', 's'+dir1)
        elif composition == 'solas12':
            series = 's'
            subdir = os.path.join('solas12', 's'+dir1)
        elif composition == 'half':
            series = 'h'
        elif composition == 'third':
            series = 'r'
        elif composition == 'fifth':
            series = 'f'
        elif composition == 'lmc3':
            series = 'l3-'
        elif composition == 'lmc':
            series = 'l-'
        elif composition == 'lmc25':
            series = 'l25-'
        elif (composition == 'tenth') or (composition == 'old'):
            composition='old'
            series = 'o'
        elif composition == 'twentyth':
            series = 'w'
        elif composition == 'hundreds':
            series = 't'
        elif composition == 'very': # 1/1000 -- this should be e for extremely instead
            series = 'v'
        elif composition == 'ultra': # 1E-4
            series = 'u'
        elif composition == 'hyper': # 1E-5
            series = 'y'
        elif composition == 'mega': # 1E-6
            series = 'm'
            bdat = 'rath00_10.1.bdat_jlf34' # maybe update for *future* runs
        elif composition == 'zero':
            series = 'z'
        elif composition == 'CO':
            series = 'co'
        elif composition == 'scaled':
            series = 'x'
        elif composition == 'scaled_he':
            series = 'e'
        elif composition == 'gch':
            series = 'g'
        elif composition == 'hez':
            series = 'he'
            subdir = os.path.join('he2sn', 'z'+dir1)
            # bgdir1 ='..'
        else:
            if composition[0] == 'x':
                series = 'x'
                subdir = os.path.join('x' + dir1)
            if composition[0] == 'e':
                series = 'e'
                subdir = os.path.join('e' + dir1)
                if len(composition) > 1:
                    suffix += composition[1:]
            elif composition[0] == 'g':
                series = 'g'
                subdir = os.path.join('g' + dir1, composition)
            elif composition[0:3] == 'hex':
                series = 'hex'
                subdir = os.path.join('he2sn', 'hex' + dir1, composition)
                bgdir1 = '..'
            elif composition[0:3] == 'hen':
                series = 'hen'
                subdir = os.path.join('he2sn', 'hex' + dir1, composition)
                bgdir1 = '..'
            else:
                self.logger.warning('Invalid composition.')
                self.series = 'm'

        if series_ is not None:
            series = series_

        if suffix == '':
            suffix = None

        if special is not None:
            if isinstance(special, str):
                special = {special,}
            for case in special:
                if case == 'hez-ppsn':
                    subdir = os.path.join('heppsn', 'hez')
                    if burn:
                        subdir += 'nuc'
                    else:
                        subdir += 'grid'
                elif case == 'z-ppsn':
                    subdir = os.path.join('zppsn', 'z')
                    if burn:
                        subdir += 'nuc'
                    else:
                        subdir += 'grid'
                elif case == 'gridb':
                    subdir = os.path.join('gridb', subdir, series+dir1)
                    if suffix is not None:
                        subdir += suffix
                elif case == 'EtaCar':
                    if subdir is not None:
                        subdir = os.path.join('EtaCar', subdir)
                    else:
                        subdir = 'EtaCar'
                    if suffix is not None:
                        subdir += suffix
                else:
                    pass

        # IF N_ELEMENTS(dirsuffix) EQ 0 THEN dirsuffix=''
        # ;; IF dirsuffix EQ 1 THEN BEGIN
        # ;;    dir1=dir1+suffix
        # ;;    suffix=''
        # ;; ENDIF

        if isinstance(composition, (AbuSet, KepAbuSet)):
            series = composition.sentinel
            if series is None:
                try:
                    series = composition.mixture[0]
                except:
                    series = 'a'

        if subdir is None:
            subdir = series + dir1

        if suffix is not None:
            subdir += suffix
            suffix = None
        if dirsuffix is not None:
            suffix = dirsuffix


        # lburn - fully coupled big network
        if lburn:
            series = series.upper()
            dirs = os.path.split(subdir)
            subdir = os.path.join(dirs[0], dirs[1].upper())

        if bgdir is None:
            bgdir = bgdir1

        if not bgdir:
            bgdir = ''
        else:
            bgdir = os.path.expanduser(os.path.expandvars(bgdir))

        if subdir_ is not None:
            subdir = subdir_

        if projectdir is not None:
            subdir = os.path.join(projectdir, subdir)

        # return values
        self.dir0 = os.path.join(dir00, subdir)
        self.bgdir = bgdir
        self.series = series
        self.suffix = suffix
        self.bdat = bdat
        self.angular = angular
        self.magnet = magnet
        self.genburn = genburn
        self.bgcopy = bgcopy
        self.bdatcopy = bdatcopy

        # overwrite dirs, names, ...
        if dirbase is not None:
            self.dir0 = os.path.expanduser(os.path.expandvars(dirbase))
        if dirtarget is not None:
            dirtarget = os.path.normpath(os.path.expandvars(os.path.expanduser(dirtarget)))
            if not os.path.isabs(dirtarget):
                dirtarget = os.path.join(dir00, dirtarget)
        self.dirtarget = dirtarget

        self.basename = basename

        self._name = name

        # done
        self.close_logger()

    def set_mass(self, mass):
        self.mass = mass

    def set_exp(self, exp):
        self.exp = exp

    def name(self):
        if self._name is not None:
            return self._name
        smass = mass_string(self.mass)
        return self.series + smass

    def dir(self):
        if self.dirtarget is not None:
            return self.dirtarget
        else:
            path = os.path.join(self.dir0, self.name())
            if self.suffix is not None:
                path += self.suffix
            return path

    def make_dir(self, overwrite = False):
        dir = self.dir()
        if not os.path.isdir(dir):
            os.makedirs(dir)
        elif not overwrite:
            self.logger.error('Directory {} alreay exists.'.format(dir))
            raise IOError('Directory {} alreay exists.'.format(dir))


    def parent_dir(self):
        """
        Return parent directory of run.

        Usually this should be dir0,
        but could have been overwritten by "targetdir".
        """
        return os.path.split(os.path.normpath(self.dir()))[0]

    def make_gseries_file(self):
        """
        generate the geseries file in the parent directory of the run.

        This may not be needed in future python batch.
        """
        filename = os.path.join(self.parent_dir(),'gseries')
        with open(filename,'w') as f:
            f.write("{:s}\n".format(self.series))

    def gfilename(self):
        if self.basename is None:
            name = self.name()
        else:
            name = self.basename
        return os.path.join(self.dir(), name+'g')

    def copy_kepler(
            self,
            overwrite = False,
    ):
        target = os.path.join(self.dir(),'k')
        if os.path.isfile(target) and not overwrite:
            self.logger.error('File exists {}'.format(target))
            raise IOError('File exists: {}'.format(target))
        shutil.copy2(self.kepenv.progfile, target)

    def copy_cmd(
            self,
            cmdfile,
            overwrite = False,
    ):
        target = os.path.join(self.dir(), cmdfile)
        if os.path.isfile(target) and not overwrite:
            self.logger.error('File exists {}'.format(target))
            raise IOError('File exists: {}'.format(target))
        shutil.copy2(cmdfile, target)

    def link_bg(
            self,
            genburn,
            silent = False,
            overwrite = False,
    ):
        self.setup_logger(silent)
        bgdir = self.bgdir
        if genburn is None:
            return
        if not self.bgcopy:
            if bgdir is None:
                return
            if bgdir == '':
                return
        else:
            if not bgdir:
                bgdir = os.path.dirname(self.kepenv.datafile(genburn, self.dir()))
        if os.path.samefile(bgdir, self.dir()):
            return
        source = os.path.join(bgdir, genburn)
        target = os.path.join(self.dir(), genburn)
        if os.path.exists(target):
            if not overwrite:
                self.logger.error('File exists {}'.format(target))
                raise IOError('File exists: {}'.format(target))
            self.logger.warning('Overwriting {}'.format(target))
            os.remove(target)
        if self.bgcopy:
            self.logger.info('Copying {} to {}'.format(source,target))
            shutil.copy2(source, target)
        else:
            self.logger.info('Linking {} to {}'.format(source,target))
            os.symlink(source, target)
        self.close_logger()

    def link_bdat(
            self,
            silent = False,
            overwrite = False):
        self.setup_logger(silent)
        if self.bdat is None:
            bdat = 'bdat'
        else:
            bdat = self.bdat
        bdatfile = self.kepenv.datafile(bdat, self.dir())
        while os.path.islink(bdatfile):
            link = os.readlink(bdatfile)
            if not os.path.isabs(link):
                link = os.path.join(os.path.dirname(bdatfile), link)
            bdatfile = link
        source = bdatfile
        target = os.path.join(self.dir(), 'bdat')
        if self.bdatcopy:
            targetx = os.path.join(self.dir(), os.path.basename(source))
            if os.path.exists(targetx):
                if not overwrite:
                    self.logger.error('File exists {}'.format(target))
                    raise IOError('File exists: {}'.format(target))
                self.logger.warning('Overwriting {}'.format(targetx))
                os.remove(targetx)
            shutil.copy2(source, targetx)
            source = os.path.relpath(targetx, self.dir())
        if os.path.exists(target):
            if not overwrite:
                self.logger.error('File exists {}'.format(target))
                raise IOError('File exists: {}'.format(target))
            if os.path.islink(target):
                if os.readlink(target) == source:
                    return
            self.logger.warning('Overwriting {}'.format(target))
            os.remove(target)
        self.logger.info('Linking {} to {}'.format(source,target))
        os.symlink(source, target)
        self.close_logger()

    def expdir(self, exp = None):
        if exp is None:
            exp = self.exp
        return os.path.join(self.dir(),'Expl'+exp)

    def make_expdir(self):
        expdir = self.expdir()
        if not os.path.isdir(expdir):
            os.makedirs(expdir)

    def presndump(self):
        dumpfilename = os.path.join(
            self.dir(),
            self.name()+'#presn')
        return dumpfilename

    def expkepfile(self):
        expkepfile = os.path.join(self.expdir,'k')
        return expkepfile

    def exppresndump(self):
        dumpfilename = os.path.join(
            self.expdir(),
            self.name()+'#presn')
        return dumpfilename

    def exppresndumprel(self):
        dumpfilename = os.path.join(
            '..',
            self.name()+'#presn')
        return dumpfilename

    def explogfile(self):
        logfilename = os.path.join(
            self.expdir(),
            'explosion.log')
        return logfilename

    # interface routines
    def gen_special(self, **kw):
        pass
    # def gen_cmd_special(self, **kw):
    #     pass
    # def exp_special(self, **kw):
    #     pass
    # def exp_cmd_special(self, **kw):
    #     pass
    # def nuc_special(self, **kw):
    #     pass
    # def nuc_cmd_special(self, **kw):
    #     pass


class KepparFormatter(object):
    """
    Output formatted KEPLER parameter for command/parameter files.
    """
    def __init__(self,
                 numeric = False):
        """
        Set up formatter.

        numeric (boolean) [False]:
           set default behavior of parameter output
        """
        self.d = {k:i for i,k in enumerate(parm.keys())}
        self.l = [[k,f] for k,f in parm.items()]
        self.n = len(self.l)
        self.numeric = numeric

    def _get(self, index):
        """
        Return value by key or index.
        """
        try:
            i = int(index)
        except:
            i = self.d.get(index, -1)
        if (i < 0) or (i >= self.n):
            raise AttributeError("Key not found.")
        return i,self.l[i][0],self.l[i][1]

    def __call__(self,
               p,
               value,
               numeric = None):
        """
        Format parameter.

        Currently only integer and float values are supported;
        other appear not be used...
        """
        ip, sp, fp = self._get(p)
        if fp == 1:
            v = float2str(value, 15)
        else:
            v = '{:d}'.format(value)
        if numeric is None:
            numeric = self.numeric
        if numeric:
            s = 'p {:d} {:s}'.format(ip,v)
        else:
            s = 'p {:s} {:s}'.format(sp,v)
        return s

class GenFile(Logged):
    """
    Write out KEPLER generator/link/command files.
    """
    def __init__(self,
                 outfile = None,
                 silent = False):

        self.setup_logger(silent)
        self.open_genfile(outfile)
        self.pf = KepparFormatter()
        self.close_logger()

    def write_alias(self, name, commands):
        """
        Write out an alias definition line.

        INPUT:
        _list_ of definition strings

        TODO:
        automatic truncation to 71 character
        ... and generation of continuation commands
        ... as needed
        Done?
        """
        if isinstance(commands, (list, tuple)):
            s = ','.join(commands)
        else:
            s = commands
        assert isinstance(s,str)
        lines = None
        if len(s) > 70:
            tokens = s.split(',')
            tokens = [t.strip() for t in tokens]
            lines = [tokens[0]]
            tokens = tokens[1:]
            for t in tokens:
                if len(t) + len(lines[-1]) < 69:
                    lines[-1] = ','.join([lines[-1],t])
                else:
                    lines += [t]
            if len(lines) == 1:
                s = lines[0]
                lines = None
        if lines is None:
            self.genline('alias {:s} "{:s}"'.format(name,s))
        else:
            name = name.strip()
            alist = list()
            for i,line in enumerate(lines):
                alias = '{:s}{:d}'.format(name,i)
                alist += [alias]
                self.write_alias(alias, line)
            alist = ','.join(alist)
            assert len(alist) <= 70
            self.write_alias(name,alist)

    def parm_line(self, parm, value):
        """
        Write parameter to file.
        """
        self.genline(self.pf(parm, value))

    def genraw(self, s):
        """
        write out raw data to generator file
        """
        self.fout.write(s)

    def genline(self, s):
        """
        write out one line to generator file
        """
        assert len(s) == 0 or s[0] == 'c' or len(s) <= 120, \
            'Line length exceeds KEPLER limit of 120.'
        self.fout.write(s.strip() + '\n')

    def comment(self, s = ''):
        """
        write out one comment line to generator file
        """
        for line in stuple(s):
            self.genline('c ' + line)

    def open_genfile(
            self,
            outfile = None,
            silent = False):
        self.setup_logger(silent)
        if outfile:
            self.outfile = os.path.expandvars(os.path.expanduser(outfile))
            path = os.path.dirname(self.outfile)
            if not os.path.exists(path):
                os.makedirs(path)
            fout = open(self.outfile,'wt')
            self.logger.info('Generating ' + self.outfile)
        else:
            fout = sys.stdout
        self.fout = fout
        self.close_logger()

    def close_genfile(self, timestamp = None):
        if self.fout != sys.stdout:
            self.fout.close()
            if timestamp is not None:
                touch(self.fout.name, timestamp = timestamp)

    def write_extra_lines(self, **kw):
        extra = kw.get('extra', None)
        if extra is not None:
            self.comment('----------------------------------------------------------------------')
            self.comment('Extra Commands')
            self.comment('----------------------------------------------------------------------')
            if not extra.endswith('\n'):
                extra += '\n'
            self.genraw(extra)

    def write_commands(self, **kw):
        try:
            commands = self.commands
        except AttributeError:
            commands = None
        commands_ = kw.get('commands', None)
        if commands == '':
            commands = None
        if commands_ == '':
            commands_ = None
        if commands is None:
            commands = commands_
        elif commands_ is not None:
            if not commands.endswith('\n'):
                commands += '\n'
            commands += commands_
        if commands is not None:
            self.comment('----------------------------------------------------------------------')
            self.comment('Command File (additions)')
            self.comment('----------------------------------------------------------------------')
            if not commands.endswith('\n'):
                commands += '\n'
            if not commands.startswith('//'):
                self.genraw('//*\n')
            self.genraw(commands)
            # if not commands.endswith('\\\\\n'):
            #     self.genraw('\\\\\n')

class KepGen(GenFile, Logged):
    """
    Class to generate KEPLER generator "g" files.
    """

    version = 10101
    default_solar = 'sollo09'

    def __init__(self, **kwargs):
        """
        Set up and write out the KEPLER grid.

        SOME DEFAULS:
        wimp_mass  = 0.D0    [GeV]
        wimp_rhodm = 2.D13   [GeV/cm^3]
        wimp_vdisp = 1.D6    [cgs]
        wimp_vstar = 0.D0    [cgs]
        wimp_si    = 1.D-43  [cgs]
        wimp_sd    = 1.D-41  [cgs]
        wimp_nsi   = wimp_si [cgs]
        wimp_psi   = wimp_si [cgs]
        wimp_nsd   = wimp_sd [cgs]
        wimp_psd   = wimp_sd [cgs]
        wimp_burn  = False


        TODO: Maybe change to just set up a class have a 'write' function.
        This would also allow to make more models based on one object
        and with similar settings.

        In Fact, this should become the write routine of a makerun object?
        """
        # parameter defaults (filter those needed)
        # TODO - carry through to subroutines
        kw = dict(kwargs)
        self.ks = kw.get('kepseries', KepSeries(**kw))

        mass = kw['mass']

        # we could get this from ks as well.  CHECK
        # in some caes it does not even work otherwise
        kepenv = kw.setdefault('kepenv', KepEnv(silent = True))

        genburn = kw.get('genburn', None)
        bgdir = kw.get('bgdir', None)
        if genburn is None:
            genburn = self.ks.genburn
            bgdir = self.ks.bgdir

        composition = kw.setdefault('composition', 'solar')
        burn = kw.setdefault('burn', False)
        yeburn = kw.setdefault('yeburn', False)
        lburn = kw.setdefault('lburn', False)
        mapburn = kw.setdefault('mapburn', True)

        angular = self.ks.angular
        magnet = self.ks.magnet
        massloss = kw.setdefault('massloss', None)

        stan = kw.setdefault('stan', False)
        special = kw.setdefault('special', None)

        mu12 = kw.setdefault('mu12', None)
        axion = kw.setdefault('axion', None)

        wimp_mass = kw.setdefault('wimp_mass', None)
        wimp_rhodm = kw.setdefault('wimp_rhodm', None)
        wimp_si = kw.setdefault('wimp_si', None)
        wimp_sd = kw.setdefault('wimp_sd', None)
        wimp_nsi = kw.setdefault('wimp_nsi', None)
        wimp_psi = kw.setdefault('wimp_psi', None)
        wimp_nsd = kw.setdefault('wimp_nsd', None)
        wimp_psd = kw.setdefault('wimp_psd', None)
        wimp_vdisp = kw.setdefault('wimp_vdisp', None)
        wimp_vstar = kw.setdefault('wimp_vstar', None)
        wimp_burn = kw.setdefault('wimp_burn', None)

        lane_rhoc = kw.setdefault('lane_rhoc', None)
        lane_rho_cutoff = kw.setdefault('lane_rho_cutoff', None)
        lane_n = kw.setdefault('lane_n',None)
        lane_theta = kw.setdefault('lane_theta', None)
        lane_Omega = kw.setdefault('lane_Omega', None)

        silent = kw.setdefault('silent', False)
        overwrite = kw.setdefault('overwrite', False)

        accretion = kw.setdefault('accrete', None)
        parm = kw.setdefault('parm', {})

        commands = kw.setdefault('commands', None)

        # the code...
        self.setup_logger(silent)

        super().__init__(self.ks.gfilename())

        self.kepenv = kepenv

        # burning
        self.burn = burn
        if yeburn:
            self.burn = True
        if lburn:
            self.burn = True
        if lburn:
            yeburn = False

        self.mass = mass
        self.composition = composition
        self.commands = None

        self.write_head(parm)
        self.set_composition(
            genburn,
            bgdir,
            overwrite = overwrite)
        self.set_spin(angular)
        self.interpolate_parms()
        self.write_grid(
            lane_rhoc = lane_rhoc,
            lane_rho_cutoff = lane_rho_cutoff,
            lane_n = lane_n,
            lane_theta = lane_theta,
            lane_Omega = lane_Omega)
        self.write_burn(mapburn)
        self.write_hstat(lane_n)
        self.write_spin()
        self.write_parm()
        self.write_aliases()
        self.write_extensions(
            massloss,
            stan,
            yeburn,
            lburn,
            magnet,
            mu12,
            axion,
            wimp_mass,
            wimp_si,
            wimp_sd,
            wimp_psi,
            wimp_nsi,
            wimp_psd,
            wimp_nsd,
            wimp_rhodm,
            wimp_vdisp,
            wimp_vstar,
            wimp_burn,
            special)
        self.write_special(self.ks, **kw)

        self.write_extra_lines(**kw)

        self.write_commands(**kw)

        # done
        self.close_genfile(timestamp = kw.get('genfiletime', None))
        self.close_logger(timing = 'profile created in')

    def write_special(self, ks, **kw):
        """
        Add extra special lines to generator.
        """
        special = ks.gen_special(**kw)
        if special is not None:
            for line in special:
                self.genline(line)

    def write_extensions(self,
                         massloss,
                         stan,
                         yeburn,
                         lburn,
                         magnet,
                         mu12,
                         axion,
                         wimp_mass,
                         wimp_si,
                         wimp_sd,
                         wimp_psi,
                         wimp_nsi,
                         wimp_psd,
                         wimp_nsd,
                         wimp_rhodm,
                         wimp_vdisp,
                         wimp_vstar,
                         wimp_burn,
                         special):

        """
        extensions beyond WW95
        """

        if special is not None:
            if isinstance(special, str):
                special = {special}
            else:
                assert isinstance(special, (list, set, tuple, np.ndarray))
                special = set(special)
        else:
            special = set()

        self.comment()
        self.comment('----------------------------------------------------------------------')
        self.comment('EXTENSIONS beyond WW95')
        self.comment('----------------------------------------------------------------------')
        self.comment()
        self.comment('write out convection data file')
        self.parm_line(376, 1)
        self.comment()
        self.comment('write out wind data file')
        self.parm_line(390, 1)

        self.comment()
        self.comment('no convective surface layers')
        self.parm_line(408, 0.67)
        self.comment()

        # use this template to add all mass loss
        if massloss is not None:
            self.massloss = massloss
        if self.massloss == 1:
            self.comment('switch ON Niewenhuijzen & de Jager mass loss ')
            self.parm_line(363, 1.)
        else:
            self.comment('switch OFF Niewenhuijzen & de Jager mass loss ')
            self.parm_line(363, 0.)
        self.comment()
        self.comment('maximum APPROX network number for BURN coprocessing')
        self.parm_line(240, 2)
        self.comment()
        self.comment('small surface boundary pressure')
        if 'p69' not in self.__dict__:
            self.p69 = 10.
        self.parm_line(69, self.p69)
        self.comment()
        self.comment('under-relaxation for Newton-Raphson solver')
        self.parm_line(375, .33)
        self.comment()
        # new versions of KEPLER allow BURN burning down to T=0
        # the low T part is crucial for radioactive decays in the envelope of
        # stars.
        self.p233 = 1.e-3
        self.comment('turn on burn co-processing down to T = {:g}'.format(self.p233))
        self.parm_line(233, self.p233)
        self.parm_line(235, -1.)
        self.comment()
        self.comment('1.2 times Buchmann et al. (2000, priv. com) C12(a,g) rate')
        self.parm_line(208, 1.2)
        self.comment()
        self.comment('use Jaeger et al. Ne22(a,g) rate')
        self.parm_line(421, 6)
        self.comment()
        self.comment('switch on adaptive network')
        self.parm_line(137, 1)
        self.comment()
        self.comment('allow more BURN backups')
        self.parm_line(263, 20)
        self.comment()
        self.comment('allow more backups')
        self.parm_line(52, 10)
        self.comment()
        self.comment('undo mixing in case of backup')
        self.parm_line(433, 2)

        self.comment()
        self.comment('try restrictive bachups to prevent mass n.c.')
        self.parm_line(204, 1.e-13)
        self.parm_line(442, 1.e-13)
        self.comment()
        self.comment('we do not usually need these warnings')
        self.parm_line('ibwarn', 0)

        if self.composition == 'solar':
            # maybe use above a certain metallicity instead?
            if self.mass > 25.:
                self.comment()
                self.comment('[SPECIAL] artifical viscosity in the outer layers')
                self.comment('[SPECIAL] to fix He burning crash (m > 25)')
                self.parm_line(109, 10.)
                zonevisk = zone-100
                self.parm_line(110, zonevisk)
                self.parm_line(111, zonevisk)
                self.parm_line(112, zonevisk)

            # this seems too special
            # if self.mass > 17.19 and self.mass < 17.21:
            #     self.comment()
            #     self.comment('[SPECIAL] artifical viscosity in the outer layers')
            #     self.comment('[SPECIAL] to fix He burning crash (m = 17.2)')
            #     self.parm_line(109, 10.)
            #     zonevisk = zone-100
            #     self.parm_line(110, zonevisk)
            #     self.parm_line(111, zonevisk)
            #     self.parm_line(112, zonevisk)

            # this seems too special
            # if self.mass > 24.59 and self.mass < 24.81:
            #     self.comment()
            #     self.comment('[SPECIAL] artifical viscosity in the outer layers')
            #     self.comment('[SPECIAL] to fix He burning crash (m = 24.6-24.8)')
            #     self.parm_line(109, 10.)
            #     zonevisk = zone-100
            #     self.parm_line(110, zonevisk)
            #     self.parm_line(111, zonevisk)
            #     self.parm_line(112, zonevisk)

        if stan:
            self.comment()
            self.comment('Stan specials')
            self.comment()
            self.parm_line(14, 80000)
            self.parm_line(16, 1000000)
            self.parm_line(18, 100)
            self.parm_line(156, 50)
            self.comment()
            self.parm_line(88, 5.e11)

        if yeburn:
            self.comment()
            self.comment('use Y_e from BURN')
            self.parm_line(357, 1)
            self.parm_line(429, 1)
            self.p65 = 1.e7

        if lburn:
            self.comment()
            self.comment('SWITCH ON FULL IMPLICIT BURN NETWORK')
            self.parm_line(434, 1)
            self.parm_line(443, 1)
            self.p65 = self.p233

        if 'p65' in self.__dict__:
            self.comment()
            self.comment('minimum nuclear burning temperature (K)')
            self.parm_line(65, self.p65)

        if self.composition[0:1] in ('x', 'g', 'e'):
            self.comment()
            self.comment('resolve fine abundances')
            self.parm_line(246, 1.e-10)
            self.parm_line(206, 1.e-4)
            self.parm_line(47, 1.e-4)
            self.comment()
            self.comment('mass conservation')
            self.parm_line(204, -1.e-6)
            self.comment()
            self.comment('allow more backups')
            self.parm_line(52, 20)
            self.comment()
            self.comment('20080415:')
            self.comment('no adzoning at the surface of RSGs')
            self.parm_line(88, 1.e14)

        if self.composition == 'zero':
            self.comment()
            self.comment('resolve fine abundances in Z=0 stars')
            self.parm_line(246, 1.e-10)
            self.parm_line(206, 1.e-4)
            self.parm_line(47, 1.e-4)
            self.comment()
            self.comment('mass conservation')
            self.parm_line(204, -1.e-6)
            self.comment()
            self.comment('do not trace Al26 in Z=0 stars')
            self.parm_line(266, 1.e0)
            self.comment()
            self.comment('allow more backups')
            self.parm_line(52, 20)
            self.comment()
            self.comment('20080415:')
            self.comment('no adzoning at the surface or RSGs')
            self.parm_line(88, 1.e14)

        if ((self.composition[0:3] == 'hex') or
            (self.composition[0:3] == 'hen')):
            scale = self.composition[3,3]
            self.comment()
            if scale == '000':
                self.comment('resolve fine abundances')
                self.parm_line(246, 1.e-10)
                self.parm_line(206, 1.e-4)
                self.parm_line(47, 1.e-4)
                self.comment()
                self.comment('mass conservation')
                self.parm_line(204, -1.e-6)
                xscale = 0.
            else:
                self.comment('resolve fine abundances')
                self.parm_line(246, 1.e-6)
                self.parm_line(206, 1.e-3)
                self.parm_line(47, 1.e-3)
                self.comment()
                self.comment('mass conservation')
                self.parm_line(204, -1.e-5)
                xscale = 10.**(float(scale)*0.1)
            self.comment()
            self.comment('do not trace Al26 in Z = 0 stars')
            self.parm_line(266, 1.)
            self.comment()
            self.comment('allow more backups')
            self.parm_line(52, 20)
            self.comment()
            self.comment('we need some surface pressue')
            self.p69 = 5.e7 * mass/120.
            self.p69 *= (1.+(min(xscale * 1.e3, 2.)) *
                max(min((self.mass - 100)/20,2),0))
            self.parm_line(69, self.p69)

            special += {'he-psn'}

            self.comment()
            self.comment('set the time at which to make zero-age-main-sequence parameter changes')
            self.parm_line(308, 1.e11)
            self.comment()
            self.comment('really stop in case of collapse')
            self.parm_line(304, 1.5e10)

        if self.composition == 'hez':
            self.comment()
            self.comment('resolve fine abundances in Z=0 stars')
            self.parm_line(246, 1.e-10)
            self.parm_line(206, 1.e-4)
            self.parm_line( 47, 1.e-4)
            self.comment()
            self.comment('mass conservation')
            self.parm_line(204, -1.e-6)
            self.comment()
            self.comment('do not trace Al26 in Z=0 stars')
            self.parm_line(266, 1.e0)
            self.comment()
            self.comment('allow more backups')
            self.parm_line( 52, 20)
            self.comment()
            self.comment('we need some surface pressue')
            if 'p69' not in self.__dict__:
                self.p69 = 5.e7 * self.mass / 120.
                self.parm_line(69, self.p69)
            self.comment()
            self.comment('set the time at which to make zero-age-main-sequence parameter changes')
            self.parm_line(308, 1.e11)

        if self.composition == 'very':
            self.comment()
            self.comment('resolve fine abundances in log(Z/Zsun)=-4 stars')
            self.parm_line(246, 1.e-10)
            self.parm_line(206, 1.e-4)
            self.parm_line(47, 1.e-4)
            self.comment()
            self.comment('mass conservation')
            self.parm_line(204, -1.e-6)
            self.comment()
            self.comment('do not trace Al26 in Z=0 stars')
            self.parm_line(266, 1.e0)
            self.comment()
            self.comment('allow more backups')
            self.parm_line(52, 20)
            self.comment()
            self.comment('special settings Stan 20060119')
            self.parm_line(18, 50)
            self.parm_line(16, 1000000000)
            self.parm_line(156, 20)
            self.comment()
            self.comment('20080415:')
            self.comment('no adzoning at the surface or RSGs')
            self.parm_line(88, 1.e14)

        if self.composition == 'ultra':
            self.comment()
            self.comment('resolve fine abundances in log(Z/Zsun)=-4 stars')
            self.parm_line(246, 1.e-10)
            self.parm_line(206, 1.e-4)
            self.parm_line(47, 1.e-4)
            self.comment()
            self.comment('mass conservation')
            self.parm_line(204, -1.e-6)
            self.comment()
            self.comment('do not trace Al26 in Z=0 stars')
            self.parm_line(266, 1.)
            self.comment()
            self.comment('allow more backups')
            self.parm_line(52, 20)
            self.comment()
            self.comment('special settings Stan 20060119')
            self.parm_line(18, 50)
            self.parm_line(16, 1000000000)
            self.parm_line(156, 20)
            self.comment()
            self.comment('20080415:')
            self.comment('no adzoning at the surface or RSGs')
            self.parm_line(88, 1.e14)

        if self.aw is not None and self.aw > 0:
            self.comment()
            self.comment('swich on rotational mixing processes (nangmix)')
            self.parm_line(364, 1)
            self.parm_line(365, 5.00e-2)
            self.parm_line(366, 3.33e-2)
            self.parm_line(367, 1.0)
            self.parm_line(368, 2.5e3)
            self.parm_line(369, 0.25)
            self.parm_line(370, 1.0)
            self.parm_line(371, 1.0)
            self.parm_line(372, 0.9)
            self.parm_line(373, 0.9)
            self.parm_line(374, 0.9)
            self.comment('smoothing of gradients and time derivative')
            self.parm_line(380, 0.2)
            self.parm_line(381, 2)
            self.parm_line(382, 1.0e-3)
            self.parm_line(383, 1.0e-3)

        if magnet is not None and magnet > 0:
            self.comment()
            self.comment('use magnetic fields + fix for SC')
            self.parm_line(423, 2)

        if mu12 is not None:
            self.comment()
            self.comment('neutrino magnetic moment')
            self.parm_line(487, mu12)

        if axion is not None:
            self.comment()
            self.comment('axion mass')
            self.parm_line(487, axion)

        if wimp_mass is not None:
            if wimp_rhodm is None: wimp_rhodm = 2.e+13,
            if wimp_si    is None: wimp_si    = 1.e-43,
            if wimp_sd    is None: wimp_sd    = 1.e-41,
            if wimp_vdisp is None: wimp_vdisp = 1.e+06,
            if wimp_vstar is None: wimp_vstar = 0.,
            if wimp_burn  is None: wimp_burn  = 0,
            if wimp_nsi   is None: wimp_nsi   = wimp_si
            if wimp_psi   is None: wimp_psi   = wimp_si
            if wimp_nsd   is None: wimp_nsd   = wimp_sd
            if wimp_psd   is None: wimp_psd   = wimp_sd

            self.comment()
            self.comment('WIMP annihilation')
            self.parm_line('wimp'    ,wimp_mass )
            self.parm_line('wimpsip' ,wimp_psi  )
            self.parm_line('wimpsin' ,wimp_nsi  )
            self.parm_line('wimpsdp' ,wimp_psd  )
            self.parm_line('wimpsdn' ,wimp_nsd  )
            self.parm_line('wimprho0',wimp_rhodm)
            self.parm_line('wimpv0'  ,wimp_vdisp)
            self.parm_line('wimpvelo',wimp_vstar)
            self.parm_line('iwimpb'  ,wimp_burn )

        if ((self.angular is not None) and
            (self.angular != 0.) and
            (self.composition == 'zero')):
            self.comment()
            self.comment('SPECIAL SETTINGS FOR ROTATING GRID')
            self.comment()
            self.comment(' surface hydrostatic equilibrium')
            self.parm_line(386, 1.0e32)
            self.comment('minimum zoning')
            self.parm_line(336, 2.0e30)

        # not sure what this does ... when ...
        if ((self.angular is not None) and
            (self.angular != 0.) and
            (self.composition == '')):
            self.comment()
            self.comment('SPECIAL SETTINGS FOR ROTATING GRID')
            self.comment()
            self.comment(' surface hydrostatic equilibrium')
            self.parm_line(386, 1.0e32)
            self.comment('minimum zoning')
            self.parm_line(336, 2.0e30)

        # SPECIAL section - add to or overwrite default behavior

        if 'z0a1' in special:
            self.comment()
            self.comment('SPECIAL SETTING FOR NO OVERSHOOTING')
            self.parm_line(148, 0.)
            self.parm_line(326, 0.)
        if set(('z-ppsn', 'hez-ppsn')) & special > set():
            self.comment()
            self.comment('SPECIAL SETTING FOR hez-ppsn')
            self.write_alias('heign',[self.pf(146,0.1),self.pf(147,0.1)])
            self.parm_line(462, -1.)
            self.parm_line(64, 1)
            self.parm_line(497, 1)
        if 'he-psn' in special:
            # this is copied from
            self.comment()
            self.comment('some extra settings')
            self.parm_line(64, 1)
            self.parm_line(113, 31200)
            self.parm_line(497, 1)
            self.parm_line(375, 1.)
            self.parm_line(185, -1.)
            self.parm_line(462, -1.)
            self.odep += [self.pf(13,0.),
                          self.pf(69,0.),
                          self.pf(86,0),
                          self.pf(146,.1),
                          self.pf(147,.1),
                          self.pf(442,1.e-10),
                          self.pf('tstop',100.),
                          self.pf(425, 0.),
                          self.pf(412,0.),
                          self.pf(325,0.),
                          self.pf(24,0.)]
        if 'sms' in special:
            self.comment()
            self.comment('some extra settings')
            self.parm_line(64, 1)
            self.parm_line(113, 31200)
            self.parm_line(497, 1)
            self.parm_line(185, -1.)
            self.parm_line(462, -1.)
            self.parm_line(375, .3)
            self.parm_line(485, 0)
            self.odep += [self.pf(13,0.),
                          self.pf(69,0.),
                          self.pf(86,0),
                          self.pf(146,.1),
                          self.pf(147,.1),
                          self.pf(442,1.e-10),
                          self.pf(425,0.),
                          self.pf(412,0.),
                          self.pf(325,0.),
                          self.pf(24,0.)]
            self.comment()
            self.comment('set the time at which to make zero-age-main-sequence parameter changes')
            self.parm_line(308, 1.e11)
            self.comment()
            self.comment('really stop in case of collapse')
            self.parm_line(304, 1.5e10)
            self.comment()
            self.comment('switch on GR')
            self.parm_line('relmult', 1.)
            self.comment()
            self.comment('log r plot')
            self.parm_line(132, 1)
            self.comment()
            self.comment('limit convection')
            self.parm_line(146, .1)
            self.parm_line(147, .1)
            self.comment()
            self.comment('non-convective atmosphere')
            self.parm_line('optconv', .67)
            self.comment()
            self.comment('low abundance limit')
            self.parm_line('abunlim', 1.e-10)
            self.comment()
        if 'wimp' in special:
            self.comment()
            self.comment('SPECIAL SETTING FOR WIMP RUNS')
            self.comment()
            self.comment('graphics in log radius')
            self.parm_line(132, 1)
            self.comment()
            self.comment('max time EQ Hubble Time')
            self.parm_line(15, 4.e17)
        if 'ns10' in special:
            self.comment()
            self.comment('frequent output')
            self.parm_line('nsdump', 10)
        if 'gridb' in special:
            if self.massloss == 0:
                self.massloss = 1
            self.comment()
            self.comment('leave on massloss')
            self.parm_line('xmlossn', self.massloss)
            self.parm_line('xmlossw', self.massloss)
            self.comment()
            self.comment('let us try with just increased mass loss')
            self.parm_line('lossrot', 1)
            self.comment()
            self.write_alias('zams',[self.pf('awwkloss', 1.)])
            self.comment()
            self.comment('we omit this for now')
            self.comment('p centmult 0.9')
        if 'EtaCar' in special:
            if self.massloss == 0:
                self.massloss = 1
            self.comment()
            self.comment('leave on massloss')
            self.parm_line('xmlossn', self.massloss)
            self.parm_line('xmlossw', self.massloss)
            self.comment()
            self.comment('let us try with just increased mass loss')
            self.parm_line('lossrot', 1)
            self.comment()
            self.write_alias('zams',[self.pf('awwkloss', 1.)])
        if 'nugrid' in special:
            # if self.massloss == 0:
            #     self.massloss = 1
            # self.comment()
            # self.comment('leave on massloss')
            # self.parm_line('xmlossn', self.massloss)
            # self.parm_line('xmlossw', self.massloss)
            self.comment()
            self.comment('write out sek file')
            self.parm_line(536, 1)
        if 'sun' in special:
            self.comment()
            self.comment('=== sun extra ==')
            self.parm_line('dypmin', 0.)
            self.parm_line('tnucmin', 1.e6)
            self.parm_line('pbound', 0.1)
            self.parm_line('dnmin', 1.e-8)
            self.parm_line('dtmax', 1.e14)
            self.parm_line('timezms', 1.e14)
            self.parm_line('maxbak', 20)
            self.parm_line('irtype', 4)
            self.write_alias('zams0', "p fmin .1")
            self.write_alias('hign', "p tnmin 5.d3")
        if 'summer2018' in special:
            if self.commands is None:
                self.commands = ''
            self.commands += '{}\n'.format(self.pf(497, 0))
            self.commands += '@ ent > 0.\n'
            self.commands += 'link CMD.SN\n'
            self.commands += '* zams\n'
            self.commands += '{}\n'.format(self.pf(69, 5.e7 * self.mass/120))

        self.comment()
        self.comment('remaining aliases')
        self.write_alias('odep', self.odep)

        if 'testing' in special:
            self.comment()
            self.comment('=== testing ====')
            self.genline('plot')
            self.genline('g')

    def write_parm(self):
        self.comment()
        self.comment('reset default parameter values:')
        self.comment()
        self.comment('time-step and back-up controls')
        self.parm_line(  6, .05)
        self.parm_line(  7, .035)
        self.parm_line(  8, .1)
        self.parm_line(  9, .05)
        if 'p25' in self.__dict__:
            self.parm_line(25, self.p25)
        self.parm_line( 46, .15)
        self.parm_line( 47, .001)
        if 'p55' in self.__dict__:
            self.parm_line(55, self.p55)
        self.parm_line(205, .3)
        self.parm_line(206, .001)
        self.comment()
        self.comment('turn off postprocessor edits')
        self.parm_line(299, 1000000)
        self.comment()
        self.comment('convergence control parameters')
        self.parm_line( 11, 1.e-7)
        self.parm_line( 12, 1.e-7)
        self.comment()
        self.comment('problem termination criteria')
        self.parm_line(158, 999999)
        self.parm_line(306, 9.e7)
        self.comment()
        self.comment('turn on sparse matrix inverter')
        self.parm_line(258,1)
        self.comment()
        self.comment('special command execution')
        self.parm_line(331, 1.2e9)
        self.parm_line(332, .05)
        self.comment()
        self.comment('linear artificial viscosity coefficient (reset to 0.1 at zero-age ms)')
        self.parm_line(13, 1000.)
        self.comment()
        self.comment('edit and dump  controls')
        self.parm_line( 16, 1000000)
        self.parm_line( 18, 10)
        self.parm_line(156, 100)
        self.parm_line(197, 5000)
        self.parm_line(268, 53)
        self.comment()
        self.comment('equation of state parameters')
        self.parm_line(92,1.e-8)
        self.comment()
        self.comment('semiconvection and overshoot mixing coefficients')
        self.parm_line( 24, 0.1)
        self.parm_line(148, 0.01)
        self.parm_line(324, 4.)
        self.parm_line(325, 0.1)
        self.parm_line(326, 0.01)
        self.comment()
        self.comment('graphics parameters')
        self.parm_line( 42, 10240750)
        self.parm_line( 64, 1)
        self.parm_line(497, 1)
        self.parm_line(113, self.p113)
        self.comment()
        self.comment('rezoning criteria')
        self.parm_line( 78, .2)
        self.parm_line( 79, .08)
        self.parm_line( 80, .2)
        self.parm_line( 81, .08)
        self.parm_line( 83, 1.e4)
        self.parm_line( 84, 1.e-4)
        self.parm_line( 86, 0)
        self.parm_line( 87, 1)
        if 'p138' in self.__dict__:
            self.parm_line(138,self.p138)
        if 'p139' in self.__dict__:
            self.parm_line(139,self.p139)
        if 'p150' in self.__dict__:
            self.parm_line(150,self.p150)
        if 'p151' in self.__dict__:
            self.parm_line(151,self.p151)
        if 'p152' in self.__dict__:
            self.parm_line(152,self.p152)
        if 'p193' in self.__dict__:
            self.parm_line(193,self.p193)
        if 'p195' in self.__dict__:
            self.parm_line(195,self.p195)
        self.parm_line(216,3)
        self.comment()
        self.comment('ise control parameters')
        self.parm_line(184, 1.5e9)
        self.parm_line(185, .04)
        self.parm_line(203, 1.e5)
        self.comment()
        self.comment('c12(a,g) rate multipliers')
        self.comment('(obsolete)')
        self.parm_line(227, 1.7)
        self.parm_line(228, 1.7)
        self.comment()
        self.comment('post-processor-dump control parameters')
        self.parm_line(44,6000000)
        self.comment('p 300 8192')
        self.parm_line(303, 0.5)
        self.comment()
        self.comment('set the time at which to make zero-age-main-sequence parameter changes')
        self.parm_line(308, 1.e12)
        self.comment()
        self.comment('turn on rezoner at the zero-age main sequence by reseting p 86')
        self.comment('to the value of p 309')
        self.parm_line(309, 1)
        self.comment()
        self.comment('turn down the linear artificial viscosity at the zero-age main')
        self.comment('sequence by reseting p 13 to the value of p 310')
        self.parm_line(310, .1)
        self.comment()
        self.comment('set the core temperature at which to make pre-carbon-burning')
        self.comment('parameter changes')
        self.parm_line(311, 5.e8)
        self.comment()
        self.comment('raise floor on abundances considered in calculating the time-step')
        self.comment('just before carbon ignition by reseting p47 to the value of p312')
        self.parm_line(312, .003)
        self.comment()
        self.comment('finely zone zone #2 just before carbon ignition')
        self.comment('by reseting p195 to the value of p313 and p150 to the value of p314')
        self.comment('(currently not used)')
        if 'p313' not in self.__dict__ and 'p195' in self.__dict__:
            self.p313 = self.p195
        if 'p314' not in self.__dict__ and 'p150' in self.__dict__:
            self.p314 = self.p150
        if 'p313' in self.__dict__:
            self.parm_line(313,self.p313)
        if 'p314' in self.__dict__:
            self.parm_line(314,self.p314)
        self.comment()
        self.comment()

    def write_aliases(self):
        """
        write out alias definition
        """
        self.comment('----------------------------------------------------------------------')
        self.comment('[ALIASES]')
        self.comment('----------------------------------------------------------------------')
        self.comment()
        self.comment('Definitions of aliased commands...')
        self.comment('The tnchar command is executed when the central temperature')
        self.comment('exceeds tempchar (p333) degK.')
        self.comment('The cdep command is executed when the central temperature')
        self.comment('exceeds tempcdep (p331) degK.')
        self.comment('The odep command is executed when the oxygen abundance drops below ')
        self.comment('o16odep (p332) in weight%, provided that the central temperature')
        self.comment('exceeds tqselim (p184) degK.')
        self.comment('The presn command is executed when the infall velocity exceeds ')
        self.comment('vinfall (p306) cm/sec.  ')
        self.comment()

        self.cign = []
        self.cdep = [self.pf(206,.003),self.pf(331,1.e99)]
        # apparently some mass range (12.2, 12.4, 12.6) has convergence
        # problems with central C burning; the follwing seems to fix this
        # IF (mass GT 12.01) AND (mass LT 12.79) THEN BEGIN
        # self.comment('[SPECIAL] keep under-relaxation till #cdep (12.0 < m <12.79)')
        # cdep=cdep+',p 375 1.'
        # ENDIF ELSE BEGIN
        self.cign += [self.pf(375,1.)]

        # limit minimum zone mass for off-center Ne/O ignition
        if self.mass <= 13.:
            self.comment('[SPECIAL] limit minimum zone mass for off-center Ne/O ignition (m <= 13.)')
            self.cign += [self.pf(336, 2.e30)]

        self.write_alias('tnchar',self.pf(87,1))
        self.write_alias('cign', self.cign)
        self.write_alias('cdep', self.cdep)
        self.odep = [self.pf(  6,     .02),
                     self.pf(  7,     .02),
                     self.pf(  8,     .02),
                     self.pf( 11,   1.e-8),
                     self.pf( 12,   1.e-8),
                     self.pf( 54,     10.),
                     self.pf( 55,     10.),
                     self.pf( 70,  1.e99),
                     self.pf( 73,  1.e99),
                     self.pf(206,   3.e-3),
                     self.pf(332, -1.e99),
                     'zerotime']
        self.comment()
        self.comment('for our convenience (...)')
        self.write_alias('t1',
                         ['tq','1','1 i'])
        self.comment()

    def write_spin(self):
        if self.angular > 0.0:
            self.comment()
            self.comment('set up rigid rotation')
            self.genline('rigidl {:12.5e}'.format(self.angular))

    def write_burn(self, mapburn):
        self.comment()
        self.comment('specify burn-generator-file name to turn on isotopic co-processing')
        if self.burn:
            self.genline('genburn {:s}'.format(self.burngen))
        else:
            self.comment('genburn <to be specified>')
        if mapburn:
            self.comment()
            self.comment('map BURN abundances to APPROX network')
            self.genline('mapburn')

    def write_hstat(self, lane_n):
        # the Lane-Emdean Integrator does a better job than hstat
        if lane_n != 0:
            return
        self.comment()
        self.comment('adjust initial temperature to yield hydrostatic equilibrium')
        self.genline('hstat')


    def set_spin(self, angular):
        """
        Set rotation rate.

        Update plot layout (p113).
        """
        self.p113 = 31
        self.aw = 0
        if angular is None:
            angular = 0
        self.angular = get_spin(angular, self.mass, self.metallicity)
        if self.angular != 0:
            self.aw = 1
            self.p113 = 13800

    def write_grid(self,
                   lane_rhoc = None,
                   lane_rho_cutoff = None,
                   lane_n = None,
                   lane_theta = None,
                   lane_Omega = None):
        """
        We only use Lane-Emden setpus for now.

        The old grid interpolation from the IDL program was discarded.
        """
        self.comment()
        self.comment('initial grid (zone #, exterior mass(g), network #,...')
        self.comment('... temp(K), rho(g/cc), [omega(1/s)[, u(cm/s)]])')
        aw = self.aw
        parms = dict()
        if lane_rhoc is not None:
            parms['rho_c'] = lane_rhoc
        if lane_rho_cutoff is not None:
            parms['cutoff'] = lane_rho_cutoff
        if lane_n is not None:
            parms['n'] = lane_n
        if lane_Omega is not None:
            parms['Omega'] = lane_Omega
            aw = lane_Omega
        if lane_theta is not None:
            parms['theta'] = lane_theta
        grid = LaneEmdenGrid(
            mass = self.mass,
            mu = self.mu,
            **parms)

        net = 1
        self.comment()
        self.comment('write out zone mass data in g')
        self.genline('zonemass g')
        self.comment()
        self.grid_line(
            0,
            grid.xm[0],
            grid.tn[0],
            grid.dn[0],
            net,
            self.comp,
            aw,
            massunit=1.)
        for i in range(1,grid.jm):
            self.grid_line(
                i,
                grid.xm[i],
                grid.tn[i],
                grid.dn[i],
                net,
                self.comp,
                massunit=1.)
        self.grid_line(
            grid.jm,
            grid.xm[grid.jm],
            grid.tn[grid.jm],
            grid.dn[grid.jm],
            net,
            self.comp,
            aw,
            massunit=1.,
            last = True)
        self.comment()
        self.comment('grid boundary pressure')
        self.p69 = grid.pn[grid.jm]
        self.parm_line(69, self.p69)
        self.lane_theta = grid.theta

    def interpolate_parms(self):
        """
        OUTDATED - Interpolate parameters from grid.

        (from IDL routine)
        """
        star_mass = np.array([12,15,20,25], dtype = np.float64)
        n_masses = star_mass.size

        # center
        mcenter = np.array([ 0.00, 0.002, 0.003,  0.005,0.0075], dtype = np.float64)
        ncenter = np.array([    0,     1,      1,     5,     4], dtype = np.int64)
        tcenter = np.array([ 6.e6,  5.e6,   5.e6,  5.e6,  5.e6], dtype = np.float64)
        dcenter = np.array([1.e-1, 8.e-2,  8.e-2, 8.e-2, 8.e-2], dtype = np.float64)

        # core(s)
        xxmass = np.array([[ 2.4, 2.4, 3.0,  3.5],
                           [ 3.0, 3.5, 5.5,  8.0],
                           [ 4.0, 5.0, 7.0, 10.0]],
                          dtype = np.float64)

        xxres = np.array([[0.005, 0.005, 0.010, 0.030],
                          [0.006, 0.007, 0.012, 0.035],
                          [0.008, 0.010, 0.015, 0.040],
                          [0.010, 0.015, 0.020, 0.050]],
                         dtype = np.float64)

        xxtime = np.array([7.e11, 5.e11, 4.e11, 3.e11], dtype = np.float64)
        xxtbac = np.array([   5.,    4.,     2.,   2.], dtype = np.float64)

        # do we need these?
        #xtem = np.array([5.0e6, 1.0e6, 1.0e6, 1.e6], dtype = np.float64)
        #xden = np.array([8.0e-2, 5.5e-2, 5.0e-2, 5.e-2], dtype = np.float64)

        # surface
        # msurf=[5.0D-1,5.0d-2,5.0D-3,5.0D-4,5.0D-4,5.0D-6,5.0D-7,0.0d+0]
        # tsurf=[1.0d+4,1.0d+4,1.0d+4,5.0d+3]
        # dsurf=[2.0d-2,1.0d-2,6.0d-3,1.0d-3]
        # dzsurf=[10,10,10,10,10]
        #msurf  = np.array([5.0D-1,5.0d-2,5.0D-3,5.0D-4,5.0D-5,5.0D-6,2.0D-7,0.0d+0], dtype = np.float64)
        #tsurf  = np.array([7.0d+4,5.0d+4,2.0d+4,1.0d+4,7.0d+3,5.0d+3,5.0d+3,5.0d+3], dtype = np.float64)
        #dsurf  = np.array([2.0d-2,1.0d-2,6.0d-3,6.0d-3,6.0d-3,6.0d-3,6.0d-3,1.0d-3], dtype = np.float64)
        #dzsurf = np.array([10    ,10    ,10    ,10    ,10    ,10    ,1     ,0     ], dtype = np.int64)

        # interpolate core(s)
        ii = n_masses-2
        for i in range(n_masses-2,0,-1):
            if self.mass < star_mass[i]:
                ii=i-1

        fmass = (self.mass - star_mass[ii])/(star_mass[ii+1] - star_mass[ii])
        xmass = xxmass[:,ii] + (xxmass[:,ii+1] - xxmass[:,ii])*fmass
        xres  = xxres [ii,:] + (xxres [ii+1,:] - xxres [ii,:])*fmass
        with np.errstate(under='ignore'):
            xtime = xxtime[ii]*np.exp(np.log(xxtime[ii+1]/xxtime[ii])*fmass)
            xtbac = xxtbac[ii]*np.exp(np.log(xxtbac[ii+1]/xxtbac[ii])*fmass)

        if self.composition == 'zero':
            xtime *= 0.25
        xtime = max(xtime, 5.0e10)

        for i in range(xres.size-1):
            xres[i+1] = np.maximum(xres[i+1],xres[i])
        for i in range(xmass.size-1):
            xmass[i+1] = np.maximum(xmass[i+1],xmass[i])

        # --------------------
        #  determine grid parameters
        # --------------------

        # grid
        resmult = 2.4

        xfrac = xmass / self.mass

        self.p193 = xfrac[0]
        self.p138 = xfrac[1]
        self.p139 = xfrac[2]

        xfres = xres / self.mass * resmult

        self.p195 = xfres[0]
        self.p150 = xfres[1]
        self.p151 = xfres[2]
        self.p152 = xfres[3]

        # time-step
        self.p25 = xtime
        self.p55 = xtbac


    def set_composition(self,
                        genburn,
                        bgdir,
                        overwrite = False):
        """
        Find, compute, set APPROX and BURN compositions.
        """
        approx = None
        approx_comment = ()
        comp = None
        burn = None
        burngen = None
        massloss = None

        if isinstance(self.composition, AbuSet):
            approx_comment = ('generated from abuset',)
            burn = self.composition
            if burn.comp is not None:
                comp = burn.comp
            else:
                comp = 'gen'
            burngen = self.ks.name() + 'bg'
            approx = KepAbuSet(abu = burn, mixture = comp)
            comment = burn.comment
            if comment is not None:
                if isinstance(comment, str):
                    comment = (comment,)
                approx_comment += comment
            if self.burn:
                burn.write_bg(os.path.join(self.ks.dir(), burngen),
                              mixture = comp,
                              overwrite = overwrite)
            if burn.mixture is not None:
                self.composition = burn.mixture
            else:
                self.composition = '(custom)'
            if burn.metallicity() > 1.e-6:
                massloss = 1
            else:
                massloss = 0

        # the sun
        if self.composition == 'solar':
            self.composition = self.default_solar
        if self.composition == 'solag89':
            approx_comment = ('solar abundances (weight %):',)
            comp = 'sol'
            burngen = 'solag89g'
            massloss = 1
        if self.composition == 'solgn93':
            comp = 'sol'
            burngen = 'solgn93g'
            massloss = 1
        if self.composition == 'sollo03':
            comp = 'sol'
            burngen = 'sollo03g'
            massloss = 1
        if self.composition == 'sollo09':
            comp = 'sol'
            burngen = 'sollo09g'
            massloss = 1
        if self.composition == 'solas09':
            comp = 'sol'
            burngen = 'solas09g'
            massloss = 1
        if self.composition == 'solas12':
            comp = 'sol'
            burngen = 'solas12g'
            massloss = 1

        # some "traditional" mixtures
        if self.composition == 'double':
            approx_comment = ('define "double" composition -- 2.0 *solar metallicity',)
            comp = 'double'
            burngen = 'dubbg'
            massloss = 1
        if self.composition == 'half':
            approx_comment = (
                'Abundances derived from isotopic "halfcomp" where:',
                'H and He and Li7 taken as (z+s)/2 where z are the zero metallitcity',
                ' abundances of Walker et. al. and s are the solar abundances of',
                ' anders and grevesse.',
                '016, Ne20, and Mg24 are scaled by a factor of 0.54064 ("O-like")',
                'Si28, S32, Ar36, and Ca40 are scaled by a factor of 0.48184 ("inter")',
                'The clearly secondary isotopes ("secondaries":O17, O18, Ne21, Ne22,',
                '  S36, Ar40, Ca46, Fe58, Cu63, Cu65, Ni64, & A.gt.66)',
                '  are scaled by a factor of 0.27032 (= 0.5 * O-like scaling)',
                'All other isotopes are scaled by a factor of 0.42945 reflecting',
                ' their mostly roughly flat observed scaling with respect to iron in',
                ' in the 1/2 Zsun to 2 Zsun range.',
                'The factor for the "O-like" isotopes corresponds to the 0.1 dex',
                ' observed excess of these isotopes relative to iron at 0.5 Zsun.',
                'The factor for the "intermediate" isotopes represents the roughly',
                ' 0.05 dex excess of these isotopes relative to iron at 0.5 Zsun.',
                'The graphs of observations in Timmes, Woosley, and Weaver (1994)',
                ' were used to deduce these factors (which include renormalization).',
                '',
                'define "half" composition -- 0.5 *solar metallicity (see above))')
            comp = 'half'
            burngen = 'halfbg'
            massloss = 1
        if self.composition == 'third':
            approx_comment = ('third=((7.D0/30.D0)*half+(1.D0/6.D0)*old)*2.5D0',)
            comp = 'thrd'
            burngen = 'thirdbg'
            massloss = 1
        if self.composition == 'fifth':
            approx_comment = ('fifth=(0.3D0*old+0.1D0*half)*2.5D0',)
            comp = 'fif'
            burngen = 'fifthbg'
            massloss = 1
        if self.composition == 'old':
            approx_comment = (
                'c define "old disk" composition -- 1.e-1 *solar metallicity',
                'c intermediate energy explosive nucleosynthesis case ("b3")')
            approx = dict(
                h1   =    75.621,
                he3  =  1.237e-2,
                he4  =    24.180,
                c12  =  1.515e-2,
                n14  =  6.358e-4,
                o16  = 1.2528e-1,
                ne20 =  1.272e-2,
                mg24 =  5.567e-3,
                si28 =  1.199e-2,
                s32  =  5.599e-3,
                ar36 =  1.239e-3,
                ca40 =  8.125e-4,
                cr48 =  1.685e-5,
                fe52 =  1.119e-4,
                fe54 =  7.450e-3)
            # metallicity = 0.0018657215812206013e0
            # mu = 5.900e-01
            comp='oldi'
            # so, do we need the above?
            burngen='z2e-3abg'
            massloss = 1
        if self.composition == 'twentyth':
            approx_comment = ('define 1E-4 absolute metallicity composition -- 1/20 *solar metallicity',)
            approx = dict(
                h1   =       76.,
                he3  =   1.25e-2,
                he4  =       24.,
                c12  = 1.2395e-2,
                n14  =  1.683e-5,
                o16  =  6.454e-2,
                ne20 =  1.202e-2,
                mg24 =  2.055e-3,
                si28 =    5.9e-4,
                s32  =  1.902e-4,
                ar36 =  3.762e-5,
                ca40 = 3.6345e-5,
                cr48 = 1.9995e-6,
                fe52 = 1.2785e-5,
                fe54 = 1.5625e-3)
            comp = 'twy'
            burngen = None
            massloss = 1
        if self.composition == 'hundreds':
            approx_comment = ('define Pop II metallicity composition -- 1.e-2 *solar metallicity',)
            approx = dict(
                h1   =        76.,
                he3  =     1.7e-2,
                he4  =        24.,
                c12  =  0.3069e-2,
                n14  =  0.1109e-2,
                o16  =  0.9618e-2,
                ne20 =  0.1753e-2,
                mg24 = 0.06935e-2,
                si28 =   7.688e-4,
                s32  =   4.259e-4,
                ar36 =   9.619e-5,
                ca40 =   6.571e-5,
                cr48 =   2.947e-6,
                fe52 =   1.813e-5,
                fe54 =  0.1285e-2)
            # metallicity = 0.00018898391498524416e0
            # mu = 5.884e-01
            comp = 'pop2'
            massloss = 1
            comp_done = 1
            burngen = None
        if self.composition == 'very':
            approx_comment = (
                'define "very old halo" composition -- 1.e-3 *solar metallicity',
                'intermediate energy explosive nucleosynthesis case ("b3")',
                'logarithmic interpolation between old and ulti')
            approx = dict(
                h1   = 7.59966e+01,
                he3  = 1.24988e-02,
                he4  = 2.40016e+01,
                c12  = 1.61053e-04,
                n14  = 5.76127e-06,
                o16  = 1.25656e-03,
                ne20 = 1.38418e-04,
                mg24 = 5.42261e-05,
                si28 = 1.09187e-04,
                s32  = 5.08184e-05,
                ar36 = 1.12367e-05,
                ca40 = 7.39185e-06,
                cr48 = 1.55765e-07,
                fe52 = 1.03345e-06,
                fe54 = 7.02140e-05)
            # mu = 5.883e-01
            # metallicity = 1.865821098046324e-05
            massloss = 0
            comp='vohi'
            # so, do we really need explicit approx ?
            burngen = 'z2e-5abg'
        if  self.composition == 'ultra':
            approx_comment = (
                'define ultra-low metallicity composition -- 1.e-4 *solar metallicity',
                'intermediate energy explosive nucleosynthesis case ("b3")')
            comp = 'ultr'
            burngen = 'uburn05bg'
            massloss = 0.
            # metallicity = 3.0358235476600409e-06
        if self.composition == 'zero':
            approx_comment = (
                'zero metallicity abundances (weight %):',
                'data from B Fields, p.c. 2002',
                'mapping H1+H2-->h1',
                'mapping H3+Li6...B11-->he3',
                'mapping C12+C13+N15-->c12',
                'mapping O16+O17+O18-->o16')
            # metallicity = 0.
            comp = 'zero'
            burngen = 'z0cbg'
            massloss = 0.
        if self.composition == 'hez':
            approx_comment = (
                'zero metallicity abundances (weight %):',
                'Z120 he core abundances.',
                '  (layer at edge of convective core when N14 is at maximum.)',
                'made by he2sn.pro, option 0, he2sn/znuc/Z120/Z120#n14max')
            comp = 'he'
            burngen = 'znuchebg'
            massloss = 0.
        if self.composition == 'hes':
            approx_comment = (
                'solar metallicity abundances (weight %):',
                'S120 he core abundances.',
                '  (layer at edge of convective core when H1 is depleted.)',
                'made by he2sn.pro, option 1, he2sn/znuc/Z120/Z120#hecore')
            comp = 'he'
            burngen = 'snuchebg'
            massloss = 0.
        if self.composition == 'lmc3':
            approx_comment = (
                'New lmc, Z=0.006,alpha-enhanced abundances',)
            # metallicity = 0.0018657215812206013e0
            # mu = 5.900e-01
            mapburn=True
            comp='lmc3'
            burngen='lmc6.0E-3.alphag'
            massloss = 1
        if self.composition == 'lmc':
            approx_comment = (
                'lmc, Z=0.0055, scaled solar, Asplund 2009',)
            # metallicity = 0.0018657215812206013e0
            # mu = 5.900e-01
            mapburn=True
            comp='lmc'
            burngen='lmc5.5E-3.scaled'
            massloss = 1
        if self.composition == 'lmc25':
            approx_comment = (
                'lmc, Z=0.0055, Alex',)
            # metallicity = 0.0018657215812206013e0
            # mu = 5.900e-01
            mapburn=True
            comp='lmc25'
            burngen='lmc25bg'
            massloss = 1

        # generated compositions from name
        if self.composition[0:1] == 'g' :
            # GCH abundances
            scale = self.composition[1:]
            if re.fullmatch('0+', scale) is not None:
                scale = 0.
            elif scale[0] in ('+', '-',):
                assert re.fullmatch('-0+', scale) is None, 'require positive zero'
                scale = 10.**(float(scale) * 0.1)
            else:
                assert re.fullmatch('\d+', scale) is not None
                scale = -0.1**(float(scale) * 0.1)
            comp = 'gch'
            burngen = self.composition + 'bg'
            burn = GCHAbu(scale)
            approx = KepAbuSet(abu = burn, mixture = comp)
            approx_comment = burn.comment
            if self.burn:
                burn.write_bg(
                    os.path.join(self.ks.dir(),burngen),
                    mixture = comp,
                    overwrite = overwrite,
                    )
            massloss = 1.
        elif self.composition[0:1] == 'e':
            # helium-enhanced scaled solar abundances
            scale = self.composition[1:4]
            if scale == '000':
                scale = 0.
            else:
                scale = 10.e0**(float(scale) * 0.1)
            helium = float(self.composition[4:7]) / 1000
            if helium == 0:
                helium = 1.
            comp = 'xsunhe'
            burngen = self.composition + 'bg'
            burn = ScaledSolarHelium(scale, helium, scale_light=True)
            approx = KepAbuSet(abu = burn, mixture = comp)
            approx_comment = burn.comment
            if self.burn:
                burn.write_bg(os.path.join(self.ks.dir(),burngen),
                              mixture = comp,
                              overwrite = overwrite)
            massloss = 1.
        elif self.composition[0:1] == 'x':
            # scaled solar abundances
            scale = self.composition[1:4]
            if scale == '000':
                scale = 0.
            else:
                scale = 10.e0**(float(scale) * 0.1)
            comp = 'xsun'
            burngen = self.composition + 'bg'
            burn = ScaledSolar(scale)
            approx = KepAbuSet(abu = burn, mixture = comp)
            approx_comment = burn.comment
            if self.burn:
                burn.write_bg(os.path.join(self.ks.dir(),burngen),
                              mixture = comp)
            massloss = 1.
        elif self.composition[0:3] == 'hex':
            # scaled solar he abndances
            # TODO - where do these come from?
            #        Include/compute directly?
            scale = self.composition[3:6]
            burngen = self.composition+'bg'
            comp = 'hex'
            # get data file from top-level directory
            burn = AbuSet(bg_file =
                os.path.normpath(
                os.path.join(
                self.ks.dir(),bgdir,burngen)))
            approx_comment = (
                'scaled solar He core; abundances SCALE = ' + scale,
                '')
            massloss = 0.
        elif self.composition[0:3] == 'hen':
            # Pop III abundance + N14
            # TODO - where do these come from?
            #        Include/compute directly?
            scale = self.composition[3:6]
            burngen = self.composition+'bg'
            comp = 'hen'
            # get data file from top-level directory
            burn = AbuSet(bg_file =
                          os.path.join(
                self.ks.dir(),bgdir,burngen))
            approx_comment = (
                'Pop III He core abundance + N14 mass fraction of ' + scale,
                '')
            massloss = 0.
        elif self.composition == 'CO':
            approx_comment = (
                'c CO WD composition',
                '')
            approx = dict(
                c12  =  0.5,
                o16  =  0.5)
            comp = 'oxcomp'
            burngen = None
            massloss = 0

        # external provided generator
        if genburn is not None:
            burngen = genburn
        if approx is None:
            if burn is None:
                assert isinstance(burngen, str)
                if bgdir:
                    xdir = os.path.expandvars(os.path.expanduser(bgdir))
                    if not os.path.isabs(xdir):
                        xdir = os.path.normpath(os.path.join(self.ks.dir(),bgdir))
                    filename = os.path.expandvars(os.path.expanduser(os.path.join(xdir, burngen)))
                else:
                    filename = self.kepenv.datafile(
                        burngen,
                        local_path = self.ks.dir())
                burn = AbuSet(bg_file = filename)
            if comp is None:
                comp = burn.mixture
            approx = KepAbuSet(abu = burn, comment = approx_comment)
        elif isinstance(approx, dict):
            approx = KepAbuSet(approx, comment = approx_comment)
        elif not isinstance(approx, KepAbuSet):
            raise Exception('Require APPROX abundance.')

        approx.mixture = comp
        self.write_approx(approx)

        self.mu = approx.mu()
        self.metallicity = approx.metallicity()
        # Should/need we use mu values from BURN if IBURNYE==1?
        # (need to check KEPLER)

        self.burngen = burngen
        self.comp = comp
        self.massloss = massloss

    def write_head(self, parm = None):
        """
        Write out generator head.

        Used mass, composition, and burn flag.
        """
        self.comment('COMPUTER-GENERATED GENERATOR FILE')
        self.comment('VERSION {:s}'.format(version2human(self.version)))
        self.comment(time.asctime(time.gmtime())+' UTC')
        self.comment()
        self.comment(mass_string(self.mass) + ' solar mass star')
        if isinstance(self.composition, str):
            self.comment(self.composition + ' composition')
        elif isinstance(self.composition, AbuSet) and self.composition.mixture is not None:
            self.comment(self.composition.mixture + ' composition')
        elif isinstance(self.composition, AbuSet):
            self.comment('(custom) composition')
        if self.burn:
            self.comment('BURN coprocessing')
        self.comment()
        if parm is not None:
            self.comment('GENERATOR COMMAND PARAMETERS')
            self.comment(str(parm)[1:-1])
            self.comment()
        self.comment('box and id information:')
        self.genline('box a02')
        self.comment()
        self.comment('input for approx network (network #1):')
        self.genline('net 1  h1    he3   he4   n14   c12   o16   ne20  mg24  si28  s32')
        self.genline('net 1  ar36  ca40  ti44  cr48  fe52  ni56  fe54  pn1   nt1')
        self.comment()
        self.comment('include ise and nse networks (networks #2 and #3)')
        self.genline('isenet')

    def write_approx(self, approx):
        """
        Write out approx abundance set.
        """
        approx.normalize()
        self.comment()
        for c in approx.comment:
            self.comment(c)
        self.comment()
        for ion,abu in approx.iteritems(): # <-- replace with items after fixing KepAbuSet
            if abu > 0:
                self.comp_line(approx.mixture,
                               ion,
                               abu)

    def comp_line(self, comp, ion, abu):
        """
        Write out one composition line.
        """
        self.genline('m {:s} {:24.16g} {:s}'.format(
            comp,
            abu,
            ion.name()))

    def grid_line(self,
                  zone,
                  ym,
                  t,
                  rho,
                  net = 1,
                  comp = 'sol',
                  aw = 0.,
                  u = None,
                  last = False,
                  massunit = -1.):
        """
        write out one grid line
        """
        s = 'g'
        s += r" {:>4d}".format(zone)
        s += r" {:17.10e}".format(ym*massunit)
        s += r" {:1d}".format(net)
        s += r" {:4s}".format(comp)
        s += r" {:16.9e}".format(t)
        s += r" {:16.9e}".format(rho)
        if (zone == 0) or last or (aw != 0.) or (u is not None):
            s += r" {:16.9e}".format(aw)
        if u is None:
            u = 0.
        if (zone == 0) or last or (u != 0.):
            s += r" {:16.9e}".format(u)
        self.genline(s)


def _check_run(cmdline, cwd):
    """
    check whether KEPLER is already running for this run

    This currently tests also for the *same* executable name.
    """
    for p in psutil.process_iter():
        try:
            if p.cwd() == cwd and p.cmdline()[:2] == cmdline[:2]:
                return p
        except psutil.AccessDenied:
            pass
    return None

class MakeRun(Logged, MultiLoop):
    """
    Generate a run or run series.
    """

    def __init__(self,
                 **kwargs):
        """
        Create Initialization Object, write file.

        Calls KepGen to generate the 'g' file.
        """

        silent = kwargs.setdefault('silent', False)
        kepenv = kwargs.setdefault('kepenv', KepEnv(silent = True))
        kwargs.setdefault('no_resolve_add', kwargs.get('no_resolve', tuple()) + (KepAbuSet, AbuSet))
        self.kepenv = kepenv

        self.setup_logger(silent)


        # if isinstance(composition,str):
        #     composition = [composition]
        # if not isinstance(mass, Iterable):
        #     mass = (mass,)

        # set up Lane-Emden object
        self.lane_theta = None

        self.multi_loop(self._make_runs, **kwargs)
#        self.make_runs(*args,**kwargs)

        # done
        self.close_logger(timing = 'finished in')

#    @loopedmethod
    def _make_runs(self, **kwargs):
        kw = dict(kwargs)
        program   = kw.setdefault('program',   True)
        makedir   = kw.setdefault('makedir',   True)
        genonly   = kw.setdefault('genonly',   False)
        kepseries = kw.setdefault('kepseries', KepSeries)
        cmd = kw.setdefault('cmd', None)
        run = kw.setdefault('run', False)
        overwrite = kw.setdefault('overwrite', False)

        if isinstance(kepseries, type):
            ks = kepseries(**kwargs)
        else:
            ks = kepseries

        self.logger.info('---------------------------------------------------------------')
        self.logger.info('generating model in {:s}'.format(ks.dir()))
        self.logger.info('---------------------------------------------------------------')

        if genonly:
            program = False
            makedir = False

        # create directory and set up info files
        # this may become different in the PYTHON framework
        #
        #    IF makedir EQ 1 THEN BEGIN
        #       PRINT,' [MAKERUN] WARNING: Creation of base directory needs to be updated.  SKIPPED.'
        # ;     xxx=findfile(dir0,COUNT=count)
        # ;     IF count EQ 0 THEN BEGIN
        # ;         spawn,'mkdir '+dir0
        # ;         OPENW,unit,dir0+'gseries',/GET_LUN
        # ;         PRINTF,unit,series,FORMAT="(A1)"
        # ;         CLOSE,unit
        # ;         OPENW,unit,dir0+'gdir'
        # ;         PRINTF,unit,STRMID(dir0,0,STRLEN(dir0)-1)
        # ;         CLOSE,unit
        # ;         FREE_LUN,unit
        # ;         CD,dir0,CURRENT=current
        # ;         SPAWN,"ln -s "+dir00+"batch/?op ."
        # ;         SPAWN,"ln -s "+dir00+"batch/?new ."
        # ;         SPAWN,"ln -s "+dir00+"batch/?check ."
        # ;         SPAWN,"ln -s "+dir00+"batch/?top ."
        # ;         CD,current
        # ;    ENDIF
        #    ENDIF

        if makedir:
            ks.make_dir(overwrite = overwrite)
            # ks.make_gseries_file()

        kgparm = dict(kwargs)
        kgparm['kepseries'] = ks
        # kgparm['outfile'] = ks.gfilename()
        # kgparm['bgdir'] = ks.bgdir
        kgparm['lane_theta'] = self.lane_theta
        kgparm['genfiletime'] = kwargs.get('genfiletime', None)
        kgparm['parm'] = self.clean(kwargs, ('silent', 'kepenv', 'overwrite', ))
        kg = KepGen(**kgparm)

        # save Lane-Emden solution
        self.lane_theta = kg.lane_theta

        if program:
            ks.copy_kepler(overwrite = overwrite)

        if cmd is not None:
            ks.copy_cmd(cmd, overwrite = overwrite)

        if kg.burn:
            ks.link_bg(kg.burngen, overwrite = overwrite)
            ks.link_bdat(overwrite = overwrite)

        self.dir = ks.dir()
        self.name = ks.name()

        if run:
            self._run(
                path = ks.dir(),
                name = ks.name(),
                prog = self.kepenv.local_prog,
                overwrite = overwrite)

    def _run(self,
             path = None,
             prog = KepEnv(silent = True).local_prog,
             name = None,
             nice = 19,
             overwrite = False):
        """
        start run
        """
        run   = name
        dump = 'g'
        args = [prog, run, dump]
        argsz = [prog, run, 'z']
        restart_dumpfile = os.path.join(path, run + 'z')

        p = _check_run(args, path)
        if p is None:
            p = _check_run(argsz, path)
        if p is not None:
            if overwrite:
                self.logger.info(
                    'Killing PID={pid:d}'.format(
                        pid = p.pid))
                p.kill()
                p = None
            else:
                self.logger.info(
                    'Process PID={pid:d} is already running.'.format(
                        pid = p.pid))
                self.p = p
                return

        if not overwrite:
            if os.path.isfile(restart_dumpfile):
                args = argsz

        p = psutil.Popen(args,
                         shell  = False,
                         cwd    = path,
                         stdin  = subprocess.DEVNULL,
                         stdout = subprocess.DEVNULL,
                         stderr = subprocess.DEVNULL,
                         start_new_session = True)

        if (nice is not None) and (nice > 0):
            p.nice(nice)
        self.p = p
        self.logger.info(
            'Starting {run:s} with PID {pid:d}'.format(
                run = ' '.join(self.p.cmdline()),
                pid = self.p.pid))





class MakeExplosionCfg(Logged):
    """
    Make explosion config file, explosion.cfg
    """
    def __init__(self,
                expdir    = None,
                name      = None,
                dump      = None,
                alpha     = 1.,
                precision = None,
                ekin      = 0.,
                mni       = 0.,
                exp       = 'a',
                start     = 'a',
                silent    = False,
                **kwargs):
        """
        Write file.

        NOTES:
        dump is only needed when mni < 0
          (Ni mass relative to He core mass,
           never used)

        OUTPUT
        self.flag
        self.goal
        """

        self.setup_logger(silent)

        section = 'explosion'
        cp = SafeConfigParser()
        cp.add_section(section)
        cp.set(section,'link', exp + start)
        cp.set(section,'alpha','{:g}'.format(alpha))
        if (mni == 0.) and (ekin == 0.):
            ekin = 1.2e51
        if ekin > 0.:
            if (ekin < 1.e40) or (ekin > 1.e60):
                self.logger.warning('ekin = {:g} erg!'.format(ekin))
            self.flag = 'ekin'
            self.goal = ekin
        else:
            if mni < 0.:
                mhe = KepDump(dump).core()['He core'].zm_sun
                mni = max(-mni * mhe, 0.3)
            if (mni < 1.e-10) or (mni > 1.e10):
                self.logger.warning('mni = {:g} Msun!'.format(mni))
            self.flag = 'mni'
            self.goal = mni
        cp.set(section,self.flag,'{:g}'.format(self.goal))
        cp.set(section,'base', name)
        if precision is not None:
            cp.set(section,'precision','{:g}'.format(precision))
        self.precision = precision
        self.configfile = os.path.join(expdir,'explosion.cfg')
        with open(self.configfile,'w') as f:
            cp.write(f)
        self.cp = cp

        # done
        self.close_logger(timing = 'finished in')

class KepExpLink(GenFile, Logged):
    """
    Write supernova generator file.
    """
    sentinel = Explosion.sentinel

    def __init__(self,
                 expdir = None,
                 rundir = None,
                 dump = None,
                 scut = 0,
                 mix = 2,
                 accel = .25,
                 mcut = 0,
                 tstop = 3.e7,
                 composition = None,
                 mass = None,
                 addcutzones = 0,
                 envelalpha = 1.,
                 norelativistic = None,
                 cutsurf = None,
                 test_commands = None,
                 silent = False,
                 mixing = 0.1,
                 mixrep = 4,
                 **kwargs):

        self.setup_logger(silent)

        kepdump = KepDump(dump)
        core = kepdump.core()
        zm_sun = kepdump.zm_sun
        if mcut > 0.:
            icut = (np.argwhere(zm_sun > mcut))[0][0]
            icut += addcutzones
            mcut = zm_sun[icut]
            self.logger.info('mass cut at  m = {:g} (j = {:d})'.format(float(mcut),int(icut)))
        elif scut == 0:
            icut = core['ye core'].j+addcutzones
            mcut = zm_sun[icut]
            self.logger.info('mass cut at Ye core: m = {:g} (j = {:d})'.format(float(mcut),int(icut)))
        else:
            stot = kepdump.stot
            icut = (np.argwhere(stot > scut))[0][0]
            icut += addcutzones
            mcut = zm_sun[icut]
            self.logger.info('mass cut at S = {:g}: Ye core: m = {:g} (j = {:d})'.format(float(scut),float(mcut),int(icut)))
            if stot[icut] / stot[icut-1] < 1.1:
                self.logger.warning('small entropy jump: S = {:g}, {:g}, {:g} !!!'.format(
                    stot[icut-1],
                    stot[icut  ],
                    stot[icut+1]))
        self.icut = icut
        self.mcut = mcut
        outfile = os.path.join(expdir,'explosion.link')
        super(KepExpLink, self).__init__(outfile)

        self.comment('link for explosion with alpha = xxx')
        self.comment()
        self.comment('determine piston and cut inner part')
        self.comment('bounce <j_cut/ye_cut> <t_min> <r_min> <r_max> <alpha> ')

        if accel == 0:
            self.genline('bounce {:d} 0.45 5.d7 1.d9 {:s} cut'.format(
                int(icut),
                self.sentinel))
        else:
            self.genline('bounce {:d} {:6.3f} 5.e7 1.e9 {:s} cut accel'.format(
                int(icut),
                float(accel),
                self.sentinel))
        self.comment()
        self.comment('resize graphics window')
        self.genline('mlim {:8.3f} 5.'.format(float(mcut)))
        self.parm_line(113, 12300)
        self.parm_line(191, 1.5e9)
        self.comment()
        self.comment('set time to execute the tshock command (at piston bounce) defined below')
        self.comment('(p 343 now set by the bounce command)')
        self.comment()
        self.comment('turn off hydrostatic equilibrium for the envelope')
        self.parm_line(386, -1.e99)
        self.comment()
        self.comment('no opal opacities')
        self.comment('p 377 0')
        self.comment()
        self.comment('limit burn co-processing maximum temperature')
        self.parm_line(234, 1.e10)
        self.comment()
        self.comment('turn off rotationally induced mixing')
        self.parm_line(364, 0)
        self.comment()
        self.comment('turn off convection plot')
        self.parm_line(376, 0)
        self.comment()
        self.comment('set time to execute the tnucleo command(post exp.nucleo)defined below')
        self.parm_line(344, 100.)
        self.comment()
        self.comment('set time to execute the tenvel command(shock in envelop)defined below')
        self.parm_line(345, 2.5e4)
        self.comment()
        self.comment('change any remaining ise or nse zones to approx')
        self.genline('approx 1 99999')
        self.comment()
        self.comment('zero problem time (add old time to toffset)')
        self.genline('zerotime')
        self.comment()
        self.comment('turn off coulomb corrections and reset zonal energies')
        self.parm_line(65, 1.e99)
        self.parm_line(215, 0.)
        self.genline('newe')
        self.parm_line(65, 1.e7)
        self.comment()
        self.comment('dump only every 1000th model')
        self.parm_line(156, 100)
        self.comment()
        self.comment('reset other parameter values as required:')
        self.comment()
        self.comment('reset time-step and back-up controls')
        self.parm_line(1, 1.e-4)
        self.parm_line(6, .05)
        self.parm_line(7, .03)
        self.parm_line(8, 10.)
        self.parm_line(25, .002)
        self.comment()
        self.comment('make less frequent dumps')
        self.parm_line(18, 1000)
        self.parm_line(156, 1)
        self.parm_line(16, 100000000)
        self.comment()
        self.comment('reset problem termination criteria')
        self.parm_line( 15, tstop)
        self.parm_line(306, 1.e99)
        self.comment()
        self.comment('turn off linear artificial viscosity ')
        self.parm_line(13, 0.)
        self.comment()
        self.comment('turn off iben opacities')
        self.parm_line(29, 0.)
        self.parm_line(30, 0.)
        self.comment()
        self.comment('set opacity floor')
        self.parm_line(50, 1.e-5)
        self.comment()
        self.comment('turn off any boundary temperature or pressure')
        self.parm_line(68, 0.)
        self.parm_line(69, 0.)
        self.comment()
        self.comment('turn off rezoning ')
        self.parm_line(86, 0)
        self.comment()
        self.comment('turn off convection')
        self.parm_line(146, 0.)
        self.parm_line(147, 0.)
        self.comment()
        self.comment('turn off transition to ise')
        self.parm_line(184, 1.e99)
        self.comment()
        self.comment('reset burn-coprocessing parameters')
        self.parm_line(229, 10)
        self.parm_line(230, .1)
        self.parm_line(272, .001)
        self.comment()
        self.comment('make sure sparse matrix inverter is turned on')
        self.parm_line(258, 1)
        self.comment()
        self.comment('set timescale for neutrino pulse used by burn for the nu-process')
        self.parm_line(286, 3.)
        self.comment()
        self.comment('set temperature of mu and tau neutrinos for the nu-process')
        self.parm_line(288, 6.)
        self.comment()
        self.comment('rest toffset and reference time to zero and use linear time in timeplots')
        self.parm_line(315, 0.)
        self.parm_line(319, 0.)
        self.parm_line(327, 1)
        self.comment()
        self.comment('reset all dump file names and delete all old dump variables')
        self.genline('newdumps')
        self.comment()
        self.comment('list of dump  variables (name# dump  ratio# dezone ratio# adzone ratio):')
        self.comment()
        self.comment('thermodynamic quantities')
        self.comment()
        self.comment()
        self.comment('Definitions of aliased commands (72 characters max)...')
        self.comment('The tshock command is executed when time reaches tshock (p343)')
        self.comment('The tnucleo command is executed when time reaches tnucleo (p344)')
        self.comment('The tenvel command is executed when time reaches tenvel (p345)')
        self.comment('The tfinal command is executed when time reaches tstop (p15)')
        self.comment()
        self.comment('               *********1*********2*********3*********4*********5*********6*********7**')
        self.write_alias('t1',
                         ['tq',
                          '1',
                          '1 i'])
        self.comment()
        self.write_alias('tshock',
                         [self.pf( 25, 1.e99),
                          self.pf(229, 20)])
        self.comment()
        self.write_alias('tnucleo',
                         [self.pf(229, 100),
                          self.pf( 38, 0.),
                          self.pf(286, 0.),
                          self.pf( 64, 50),
                          'mixnuc',
                          'newe'])
        self.comment()
        self.comment('turn off OPAL95')
        self.write_alias('tenvel',
                         [self.pf(337, 1),
                          self.pf(377, 0),
                          'mixenv',
                          'newe',
                          self.pf(375, float(envelalpha))])
        self.comment()
        # do we really still want that?
        # self.write_alias('tfinal',
        #                  ['editiso'])
        self.comment()
        self.parm_line(375, .33)
        self.parm_line(  5, 40)
        self.parm_line(425, 0.)
        self.comment()
        self.comment('output wind file to make light curve plots')
        self.parm_line(390, 1)
        self.comment()
        self.comment('switch off ye taken from BURN -- bad for fallback')
        self.parm_line(357, 0)
        self.comment()
        self.comment('allow many steps')
        self.parm_line( 14, 1000000000)
        self.comment()
        # self.comment('long lc output')
        # self.parm_line(437, 50)
        self.comment('full lc output')
        self.parm_line(437, 1000000000)
        self.comment()
        self.comment('5 MeV nu_e_bar')
        self.parm_line(446, 5.)
        self.comment()
        self.comment('turn off mass loss')
        self.parm_line(363, 0.)
        self.parm_line(387, 0.)
        self.parm_line(519, 0)

        self.comment()
        self.comment('less restrictive abundance check for explosions')
        self.parm_line(442, 1.e-10)

        # envel dump needs to be earlier for Z=0 stars
        if composition == 'zero':
            self.comment()
            self.comment('special Z=0 early execution of the #envel command')
            if mass < 15:
                self.parm_line(345, 500.)
            elif mass < 20:
                self.parm_line(345, 1000.)
            else:
                self.parm_line(345, 2000.)

        if mix is None: mix = 0
        mixnucdum = 'alias mixnuc "p 1"' # a dummy
        mixenvdum = 'alias mixenv "p 1"' # a dummy
        mixnuc = 'c'
        mixenv = 'c'

        if mix > 0:
            # terminate if mixing fails (last "1")
            mixit = 'mix 1 {:d} {:g} 1'.format(
                int(core['star'].j - icut),
                core['He core'].zm_sun * mixing)
            if mix & 1:
                mixenvdum = 'c'
                mixenv = 'alias mixnuc "{}"'.format(','.join(['mixit'] * mixrep))
                self.logger.info('applying "{:s}" {:d} times at #tenvel.'.format(mixit, mixrep))
            if mix & 2:
                mixnucdum = 'c'
                mixnuc = 'alias mixnuc "{}"'.format(','.join(['mixit'] * mixrep))
                self.logger.info('applying "{:s}" {:d} times at #nucleo.'.format(mixit, mixrep))

            self.comment()
            self.comment('alias for no mixing')
            if mixnucdum != 'c': self.genline(mixnucdum)
            if mixenvdum != 'c': self.genline(mixenvdum)
            self.comment()
            self.comment('aliases for mixing')
            if mixnuc != 'c': self.genline(mixnuc)
            if mixenv != 'c': self.genline(mixenv)
            self.write_alias('mixit', [mixit])

        if cutsurf is not None:
            if cutsurf == 0:
                self.comment()
                self.comment('removing NO "detached" surface zones...')
                self.logger.info('REMOVING ***NO*** SURFACE ZONES')
            else:
                self.comment()
                self.comment('removing "detached" surface zones...')
                self.genline('cutsurf {:d}'.format(int(cutsurf)))
                self.parm_line(68, 0.)
                self.parm_line(69, 0.)
                if cutsurf < 0:
                    self.logger.info('REMOVING OUTER {:d} ZONES'.format(-int(cutsurf)))
                else:
                    self.logger.info('REMOVING ZONES OUTSIDE {:d}'.format(int(cutsurf)))
        else:
            # cut away surface density jumps ("detached zones")
            dn = kepdump.dn
            jm = kepdump.jm
            x, = np.where(
                np.logical_and(
                    np.logical_and(
                        (dn[0:-1] > 2. * dn[1:]),
                        (dn[1:] < 1.e-4)),
                    (kepdump.ym_sun[1:] < 0.1)))
            if len(x) > 0 and x[0] < jm:
                i = jm - x[0]
                self.comment()
                self.comment('removing "detached" surface zones...')
                self.genline('cutsurf {:d}'.format(-int(i)))
                self.parm_line(68, 0.)
                self.parm_line(69, 0.)
                self.logger.info('REMOVING OUTER {:d} ZONES'.format(int(i)))

        # the following is in particular to fix pulsational pair-SN runs
        # in which surface zones may have been cut away for previous pulses

        if norelativistic is None:
            norelativistic = True
        if norelativistic:
           self.comment()
           self.comment('do not go relativistic')
           self.parm_line('vloss', 1.e10)
        else:
           self.comment()
           self.comment('do not cut off surface zones')
           self.parm_line(271, 1.e99)
           self.parm_line(409, 1.e99)

        # add refined nucleosynthesis settings
        self.comment()
        self.comment('refined nucleosynthesis settings (alex 20120510')
        self.parm_line(238, 1.e-3)
        self.parm_line(239, 1.e-3)

        # add extra lines
        self.write_extra_lines(**kwargs)

        # add extra commands to link file (TODO - test whether this works)
        self.write_commands(**kwargs)

        self.close_genfile()

        # also make special explosion.cmd file for test explosions
        self.logger.info('Making explosion cmd file.')
        cmdfile = os.path.join(expdir,'explosion.cmd')
        self.open_genfile(cmdfile)
        self.parm_line(16, 100000000)
        self.parm_line(18, 1000)
        self.parm_line(156, 10)
        self.parm_line(390, 0)
        self.parm_line(437, 0)
        self.parm_line(536, 0)

        # add extra commands
        if test_commands is not None:
            self.genraw(test_commands)
            if not test_commands.endswith('\n'):
                self.genraw('\n')

        self.close_genfile()

        # done
        self.close_logger(timing = 'linkfile created in')


class GetLinkData(Logged):
    """
    Read data from head of Link file.

    Use expsion.Result(fromlink = True) instead

    Output (stored fields):
    ALPHA=alpha,
    MNI=mni,
    EKIN=ekin,
    MCUT=mcut,
    ZONECUT=zonecut

    NOTES/TODO:
    Why not use the code in explosion.py that generates that data?
    """
    def __init__(self,
                 linkfile = None,
                 silent = False):

        self.setup_logger(silent)

        with open(os.path.expandvars(os.path.expanduser(outfile)),'r') as f:
            # check version is 2.00.00
            for i in range(5):
                f.readline()
            self.alpha = float(f.readline()[14:])
            self.mni = float(f.readline()[14:])
            self.ekin = float(f.readline()[14:])
            self.mcut = float(f.readline()[14:])
            self.zonecut = int(f.readline()[14:])
            self.piston  = int(f.readline()[14:])

        # done
        self.close_logger(
            timing = 'link information from "{:s}" read in'.format(
            linkfile))


class TestExp(Logged, MultiLoop):
    """
    Generate test explosion setup.
    """
    def __init__(self,
                 *args,
                 **kwargs):
        """
        Create Test Explosion setup, write file.

        Calls KepExpLink to generate the 'explosion.link' file.

        One of the specific changes will be to use dictionary to
        associate explosion models and energies, e.g., {'A': 1.2e51}

        We also have a more extended syntax for exp:
        {'A':dict(ekin=0.3e51,scut=4),
         'B':dict(ekin=0.6e51,scut=4)}

        We also allow a syntax with one level less of nesting:
        [dict(exp='A',ekin=0.3e51,scut=4),
         dict(exp='P',ekin=1.2e51,scut=0)]

          IMPORTANT:
            Due to multi-loop resolution exp needs to be a list in
            this case so that things can work

        We also allow specific settings associated with
        with the mass parameter:
          mass = [{mass = 9.5, cursurf=0},
                  {mass = 9.6, cursurf=1},
                  9.7, 9.8, 9.9]
          Note that if mass is a dictionary, the mass has to be given
          using the "mass" key in it.

          IMPORTANT:
            Due to multi-loop resolution mass needs to be a list in
            this case so that things can work

        We also allow mass to be a dictionary where the mass values
        are keys, either float or string
          mass = {9.5: {},
                  9.6: {cutsurf: 0},
                  '9.7': {}}

          NOTE:
            Whereas this is saver because the elements are never
            scalars, extracting a specific element is less easy than
            in a list where they can be indexed by position.


        """
        silent = kwargs.setdefault('silent', False)
        self.setup_logger(silent)

        kepenv = kwargs.setdefault('kepenv', KepEnv())
        self.kepenv = kepenv

        # DO MULTILOOP
        # IDL had one on mass and "base" [now exp] (explosion A,B,...)
        self.multi_loop(self._make_exp, *args, **kwargs)

        # done
        self.close_logger(timing = 'testexp generated in')


    def _make_exp(self,
                  genonly = False,
                  kepler = 0,
                  explosion = False,
                  makedir = True,
                  silent = False,
                  kepseries = KepSeries,
                  getadir = '.',
                  dump = False,
                  run = False,
                  **kwargs):

        kw = dict(kwargs)

        if genonly:
            kepler = 0
            explosion = False
            makedir = False
            dump = False

        exp = kw['exp']
        # here magic to follow
        if isinstance(exp, dict):
            # make copy of dictionary so that subsequent calls
            # can sill use the original object
            exp = dict(exp)
            expname = exp.pop('exp', None)
            if expname is not None:
                expvalues = exp
            else:
                expname, = list(exp.keys())
                expvalues, = list(exp.values())
            if isinstance(expvalues, dict):
                kw.update(expvalues)
            else:
                kw['ekin'] = expvalues
            kw['exp'] = exp = expname

        # more magic
        mass = kw['mass']
        if isinstance(mass, dict):
            if 'mass' in mass:
                kw.update(mass)
                mass = kw['mass']
            else:
                assert len(mass) == 1
                massvalue,masspar = list(mass.items())[0]
                if isinstance(massvalue, str):
                    massvalue = float(mass)
                kw.update(masspar)
                kw['mass'] = mass = massvalue

        explosion_defaults(kw)

        ksparm = dict(kw)
        # set extra values
        ksparm['exp'] = exp
        ks = kepseries(**ksparm)

        # Not sure we need that, done in link file writer
        # IF (N_ELEMENTS(mni) EQ 0) AND (N_ELEMENTS(ekin) EQ 0) THEN ekin=1.2D51

        if getadir == '.':
            getadir = ks.dir()

        if makedir:
            ks.make_expdir()

        elparm = dict(kw)
        elparm['expdir'] = ks.expdir()
        elparm['rundir'] = ks.dir()
        elparm['dump']   = ks.presndump()
        el = KepExpLink(**elparm)

        # config file will be updated later in _get_adir
        meparm = dict(kw)
        meparm['expdir'] = ks.expdir()
        meparm['name']   = ks.name()
        meparm['dump']   = ks.presndump()
        ec = MakeExplosionCfg(**meparm)

        # this makes no sense since explosion is a script
        # that links in with many things rather than being
        # a self-contained program
        if explosion:
            shutil.copy2(self.kepenv.expfile, ks.expdir())

        # we should not need to have to create that link
        if (kepler & 1):
            expkepfile = ks.expkepfile()
            if os.path.exists(expkepfile):
                os.remove(expkepfile)
            os.symlink(
                Explosion.default_config['program'],
                expkepfile)

        # also, this should use the same KEPLER that is used later to
        # run the explosion for nucleosynthesis
        if (kepler >> 1) & 1:
            expkepfile = ks.expkepfile()
            if os.path.exists(expkepfile):
                os.remove(expkepfile)
            shutil.copy2(ks.progfile,expkepfile)

        # I do not think we need to make a link for the dump
        if dump:
            dumpfile = ks.exppresndump()
            if os.path.exists(dumpfile):
                os.remove(dumpfile)
            os.symlink(
                ks.exppresndumprel(),
                dumpfile)

        if getadir is not None:
            gaparm = dict()
            gaparm['getadir']   = getadir
            gaparm['mpist']     = el.mcut
            gaparm['ipist']     = el.icut
            gaparm['expconfig'] = ec
            # gaparm['flag']      = ec.flag
            # gaparm['configfile']= ec.configfile
            # gaparm['goal']      = ec.goal
            # gaparm['precision'] = ec.precision
            if 'getarun' in kw:
                gaparm['getarun']   = kw['getarun']
            self._get_adir(**gaparm)

        if run:
            self._run(path = ks.expdir(),
                      prog = self.kepenv.expfile)

    def _get_adir(self,
                  getadir = None,
                  getarun = '?',
                  expconfig = None,
                  flag = None,
                  mpist = None,
                  ipist = None,
                  configfile = None,
                  goal = None,
                  precision = 2):
        """
        find set of best links and add min/max info to config file as
        well as a better guess for alpha.
        """
        if isinstance(   expconfig, MakeExplosionCfg):
            flag       = expconfig.flag
            goal       = expconfig.goal
            if expconfig.precision is not None:
                precision = expconfig.precision
            configfile = expconfig.configfile
            cp         = expconfig.cp
        else:
            cp         = None
        if not isinstance(getadir, (list, tuple)):
            getadir = [getadir]
        if not isinstance(getarun, (list, tuple)):
            getarun = [getarun]
        links = []
        for adir in getadir:
            for arun in getarun:
                links += glob.glob(
                    os.path.join(adir,
                                 'Expl' + arun,
                                 arun + '?.link'))
        results = []
        for link in links:
            result = Result(from_link = True,
                            linkfile = link)

            # checks
            if result.alpha is None:
                continue
            if result.mpist is None:
                if ipist == result.ipist:
                    result.mpist = mpist
                else:
                    continue
            else:
                if abs(1 - result.mpist / mpist) > 1.e-6:
                    continue
            results += [result]
        if len(results) == 0:
            self.logger.info('getadir: No suitable runs found.')
            return

        # we shall assume results are monotonuous in alpha
        results = sorted(results, key = lambda x: x.val(flag))

        # check for monotonicity (program fix later)
        for r0,r1 in zip(results[:-1],results[1:]):
            assert r0.val(flag) <= r1.val(flag), 'Cannot sort links.'
            if r0.val(flag) > r1.val(flag):
                self.logger.info('getadir: Cannot sort links.')
                return

        rmin = results[0]
        for r in results:
            if r.val(flag) > goal:
                break
            rmin = r
        rmax = results[-1]
        for r in results[::-1]:
            if r.val(flag) < goal:
                break
            rmax = r

        # just to be sure ...
        assert rmax.val(flag) >= rmin.val(flag)

        alpha = State(results = [rmin,rmax],
                      precision = precision,
                      flag = flag,
                      goal = goal).alpha_forecast()

        # update config file
        if cp is None:
            cp = SafeConfigParser()
            with open(configfile,'r') as f:
                cp.readfp(f)
        section = 'explosion'
        cp.set(section,'alpha','{:g}'.format(alpha))

        rmin.add_to_config(config  = cp,
                           section = 'min')
        rmax.add_to_config(config  = cp,
                           section = 'max')
        with open(configfile,'w') as f:
            cp.write(f)

    @staticmethod
    def run(path = None,
            prog = KepEnv(silent=True).expfile,
            nice = 19):
        """
        start run

        TODO:
        check if things are already running
        """
        args = [prog]
        p = psutil.Popen(args,
                         shell  = False,
                         cwd    = path,
                         stdin  = subprocess.DEVNULL,
                         stdout = subprocess.DEVNULL,
                         stderr = subprocess.DEVNULL,
                         start_new_session = True)
        if (nice is not None) and (nice > 0):
            p.nice(nice)
        return p

    def _run(self,
             **kwargs):
        """
        start run and log PID
        """
        kw = dict(kwargs)
        kw.setdefault('prog', self.kepenv.expfile)
        self.p = self.run(**kw)
        self.logger.info('Starting {run:s} with PID {pid:d}'.format(
            run = ' '.join(self.p.cmdline()),
            pid = self.p.pid,
            ))

########################################################################

class RerunBurnExp(object):
    def __init__(self,
                 rundir = None,
                 expdir = None,
                 exp    = None):
        cmdfile = os.path.join(rundir,'CMD' + exp)
        if os.path.exists(cmdfile):
            os.remove(cmdfile)

        linkfile = os.path.join(rundir, exp + '.link')
        if os.path.exists(linkfile):
            os.remove(linkfile)

        logfile = os.path.join(expdir,'explosion.log')
        if os.path.exists(logfile):
            os.remove(logfile)

        TestExp.run(path = expdir)


class MakeCmd(GenFile, Logged):
    def __init__(self,
                 rundir = None,
                 commands = None,
                 exp = None):
        cmdfile = os.path.join(rundir,'CMD' + exp)
        self.open_genfile(cmdfile)
        self.genline('link {:s}.link'.format(exp))

        # add extra commands
        if commands is not None:
            if not commands.endswith('\n'):
                commands += '\n'
            self.genraw(commands)

        self.close_genfile()

class CopyLink(Logged):
    def __init__(self,
                 expdir = None,
                 rundir = None,
                 exp    = None,
                 name   = None,
                 overwrite = False,
                 silent = False,
                 **kwargs):

        self.setup_logger(silent)

        self.ok = True

        linkname = None

        # here we want to look for a "explosion.res" file in the future
        logfile = os.path.join(expdir,'explosion.log')
        with open(logfile,'r') as f:
            for line in f:
                if line.startswith(' in link: '):
                    linkname = line.rsplit(' ',1)[-1].strip()
                    break

        if linkname is None:
            self.logger.error('no valid link file found: {:s}'.format(expdir))
            self.ok = False
            return

        newlink = os.path.join(rundir, exp + '.link')
        if not overwrite:
            if os.path.exists(newlink):
                self.logger.error('WARNING: file {:s} exists.'.format(newlink))
                self.logger.error('WARNING: Not overwriting.')
                self.ok = False
            if self.ok:
                run = name + exp
                if len(glob.glob(os.path.join(rundir,run + '*'))) >0:
                    self.logger.error('WARNING: previous run files exists for {:s}.'.format(run))
                    self.logger.error('WARNING: Not overwriting.')
                    self.ok = False
            if not self.ok:
                return

        shutil.copy2(os.path.join(expdir,linkname),newlink)
        self.close_logger()


class BurnExp(Logged, MultiLoop):
    """
    I assume this was done by the batch script in the past instead
    """
    def __init__(self,
                 *args,
                 **kwargs):

        silent = kwargs.setdefault('silent', False)
        self.setup_logger(silent)

        kepenv = kwargs.setdefault('kepenv', KepEnv())
        self.kepenv = kepenv

        # DO MULTILOOP
        self.multi_loop(self._make_exp, *args, **kwargs)

        # done
        self.close_logger(timing = 'nucleosynthesis explosions generated in')

    def _make_exp(self, **kwargs):

        # check which we need...
        kw = dict(kwargs)

        mass = kw['mass']
        exp  = kw['exp']
        makecommand = kw.setdefault('makecommand', True)
        kepseries = kw.setdefault('kepseries', KepSeries)
        rerun = kw.setdefault('rerun', False)
        overwrite = kw.setdefault('overwrite', False)
        copylink = kw.setdefault('copylink', True)
        run = kw.setdefault('run', False)
        run_only = kw.setdefault('run_only', False)
        commands = kw.setdefault('commands', None)
        copygen = kw.setdefault('copygen', None)


        ksparm = dict(kw)
        ks = kepseries(**ksparm)

        expdir = 'Expl' + exp

        smass = mass_string(mass)
        name = ks.name()

        if rerun:
            RerunBurnExp(
                rundir = ks.dir(),
                expdir = ks.expdir(),
                exp    = ks.exp)
        if copygen:
            ok = self._copy_gen(
                target = ks.dir(),
                source = copygen,
                exp = exp,
                overwrite = overwrite)
            if not ok:
                self.logger.warning('Error copying files.')
                return
            copylink = False
            makecommand = False
        if run_only:
            run = True
            copylink = False
            makecommand = False
        if copylink:
            cl = CopyLink(
                overwrite = overwrite,
                expdir = ks.expdir(),
                rundir = ks.dir(),
                name   = ks.name(),
                exp    = ks.exp)
            if not cl.ok:
                makecommand = False
        if makecommand:
            MakeCmd(
                rundir = ks.dir(),
                commands = commands,
                exp = exp)
        elif not run_only and not copygen:
            run = False
            self.logger.warning('Not starting run.')
        if run:
            self._run_exp(
                path = ks.dir(),
                exp  = ks.exp,
                name = ks.name(),
                prog = self.kepenv.local_prog,
                overwrite = overwrite)

    def _copy_gen(self,
            target = None,
            source = None,
            exp = None,
            overwrite = False):
        if not os.path.isdir(target):
            self.logger.warning('Target directory not found: {}'.format(target))
            return False
        cmd = 'CMD' + exp
        link = exp + '.link'
        files = [cmd, link]
        target = os.path.expanduser(os.path.expandvars(target))
        source = os.path.expanduser(os.path.expandvars(source))
        if not os.path.isabs(source):
            sx = os.path.normpath(os.path.join(target, source))
            if not os.path.isdir(sx):
                sx = os.path.normpath(os.path.join(self.ks.ke.dir00, source))
            source = sx
        if not os.path.isdir(source):
            self.logger.warning('Source directory not found: {}'.format(source))
            return False
        for f in files:
            source_file = os.path.join(source, f)
            if not os.path.isfile(source_file):
                self.logger.warning('Source file not found: {}'.format(source_file))
                return False
            target_file = os.path.join(target, f)
            if os.path.isfile(target_file):
                if not overwrite:
                    self.logger.warning('Not overwriting target file ' + target_file)
                    return False
                os.remove(target_file)
            self.logger.info('Copying {} to {}'.format(source_file, target_file))
            shutil.copy2(source_file, target_file)
        return True

    def _run_exp(self,
                 path = None,
                 prog = KepEnv(silent = True).local_prog,
                 name = None,
                 exp  = None,
                 nice = 19,
                 overwrite = False):
        """
        start run
        """
        run   = name + exp
        dump = name+'#presn'
        args = [prog, run, dump]
        argsz = [prog, run, 'z']
        restart_dumpfile = os.path.join(path, run + 'z')

        p = _check_run(args, path)
        if p is None:
            p = _check_run(argsz, path)
        if p is not None:
            if overwrite:
                self.logger.info(
                    'Killing PID={pid:d}'.format(
                        pid = p.pid))
                p.kill()
                p = None
            else:
                self.logger.info(
                    'Process PID={pid:d} is already running.'.format(
                        pid = p.pid))
                self.p = p
                return

        if not overwrite and os.path.isfile(restart_dumpfile):
            args = argsz
        else:
            shutil.copy2(os.path.join(path, 'CMD' + exp),
                         os.path.join(path, run + '.cmd'))

        p = psutil.Popen(args,
                         shell  = False,
                         cwd    = path,
                         stdin  = subprocess.DEVNULL,
                         stdout = subprocess.DEVNULL,
                         stderr = subprocess.DEVNULL,
                         start_new_session = True)

        if (nice is not None) and (nice > 0):
            p.nice(nice)
        self.p = p
        self.logger.info(
            'Starting {run:s} with PID {pid:d}'.format(
                run = ' '.join(self.p.cmdline()),
                pid = self.p.pid))

#######################################################################
# some utilities to make grids

from collections import Iterable

def MassL(*args, **kwargs):
    """
    convenince function to generate mass List
    """
    mass = []
    for a in args:
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if not isinstance(a, Iterable):
            a = [a]
        mass += a
    if len(kwargs) == 0:
        return mass
    masses = []
    for m in mass:
        mx = dict(mass = m)
        mx.update(kwargs)
        masses.append(mx)
    return masses

def MassD(*args, **kwargs):
    """
    convenince function to generate mass dictionary
    """
    mass = []
    for a in args:
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if not isinstance(a, Iterable):
            a = [a]
        mass += a
    masses = OrderedDict()
    for m in mass:
        masses[m] = dict(kwargs)
    return masses

def MassU(masses, *args, **kwargs):
    """
    convenince function to update mass list/dictionary

    the entries are created as new objects
    """
    if len(kwargs) == 0:
        return masses
    mass = []
    for a in args:
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if not isinstance(a, Iterable):
            a = [a]
        mass += list(a)
    if len(mass) == 0:
        # assume all masses (make mass list)
        if isinstance(masses, dict):
            for k in masses.keys():
                mass += [k]
        else:
            for m in masses:
                if isinstance(m, dict):
                    mass += [m['mass']]
                else:
                    mass += [m]
    if isinstance(masses, dict):
        for j, mx in enumerate(mass):
            for k, v in masses.items():
                if mass_equal(k, mx):
                    v = dict(v)
                    v.update(kwargs)
                    masses[k] = v
                    break
    else:
        for j, mx in enumerate(mass):
            for i, m in enumerate(masses):
                if isinstance(m, dict):
                    if mass_equal(m['mass'], mx):
                        m = dict(m)
                        m.update(kwargs)
                        masses[i] = m
                        break
                else:
                    if mass_equal(m, mx):
                        my = dict(mass = m)
                        my.update(kwargs)
                        masses[i] = my
                        break
    return masses

def MassE(masses, *args, **kwargs):
    """
    convenince function to extract mass from list/dictionary

    the entries are created as new objects
    """
    mass = []
    for a in args:
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if not isinstance(a, Iterable):
            a = [a]
        mass += a
    if len(mass) > 0:
        if isinstance(masses, dict):
            subset = OrderedDict()
            for j, mx in enumerate(mass):
                for k, v in masses.items():
                    if mass_equal(k, mx):
                        v = dict(v)
                        subset[k] = v
                        break
        else:
            subset = []
            for j, mx in enumerate(mass):
                for i, m in enumerate(masses):
                    if isinstance(m, dict):
                        if mass_equal(m['mass'], mx):
                            my = dict(m)
                            subset += [my]
                            break
                    else:
                        if mass_equal(m, mx):
                            subset += [m]
                            break
            return subset
    elif len(kwargs) > 0:
        if isinstance(masses, dict):
            subset = OrderedDict()
            for kx, vx in kwargs.items():
                for k, v in masses.items():
                    if isinstance(v, dict):
                        if kx in v and v[kx] == vx:
                            v = dict(v)
                            subset[k] = v
        else:
            subset = []
            for kx, vx in kwargs.items():
                for i, m in enumerate(masses):
                    if isinstance(m, dict):
                        if kx in m and m[kx] == vx:
                            my = dict(m)
                            subset += [my]
        return subset
    else:
        return None

#######################################################################

if __name__ == "__main__":
    argv = sys.argv[1:]
    command = argv[0]
    function = sys.modules[__name__].__dict__[command]
    args = []
    kwargs = {}
    for x in argv[1:]:
        try:
            arg = eval(x)
            args += [arg]
            continue
        except:
            pass
        try:
            kw = eval('dict({})'.format(x))
            kwargs.update(kw)
            continue
        except:
            pass
        raise Exception('Cannot evaluate {}'.format(x))
    print(' [KEPGEN] Calling {} with {} {}.'.format(command, args, kwargs))
    function(*args, **kwargs)

# kepgen.KepGen(composition='CO',mass=1.2)

# kepgen.MakeRun(composition='solar',mass=1.,lane_rhoc=.01)
# kepgen.MakeRun(composition='solag89',mass=1.,lane_rhoc=.01,special={'sun','testing'})
# kepgen.MakeRun(composition='solag89',mass=1.,lane_rhoc=.01,special={'sun','testing'},lburn=True,magnet=1.9e48)


# kepgen.MakeRun(composition='zero',mass=1.e5,lane_rhoc=.001,dirtarget='SMS',special={'sms'})

# kepgen.MakeRun(composition='zero', mass = 250, yeburn = True, subdir = 'PopIIImix')

# commands = """//*
# xmlossn 0.
# p 69 10.
# p 336 1.d20
# p dnmin 1.d-12
# p taumin 0.1
# p 132 11
# p 113 31200
# p iwinsize 10241750
# p optconv 1.
# plot
# killburn
# p isurf 1
# * heign
# p isurf 2
# * hedep
# p 375 .1
# p vloss -2.
# """
# kepgen.MakeRun(mass=2.5, series='he', dirtarget='/home/alex/kepler/lowmhe/he2.5', genburn='../hes15bg')
