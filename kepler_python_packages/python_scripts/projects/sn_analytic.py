"""
Project analytic supernova models
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import copy
import logging
import pylab as P
from scipy.interpolate import interp1d
from logged import Logged
import kepdata
import physconst
from human import time2human
import color as xcolor
from prog3d import Presn
import kepdump
import sys
import glob


#######################################################################

class SN(Logged):
    def __init__(self, dump = None):
        self.dump = dump

#-------------------------------------------------------------------------------
    @staticmethod
    def rhoav(r, m):
        """
        average density in g/cm**3
        """
        rhoav = (0.75 / np.pi) * m * np.maximum(r, 1.e-99)**(-3)
        if rhoav[0] == 0:
            rhoav[0] = rhoav[1]
        return rhoav
#-------------------------------------------------------------------------------
    @classmethod
    def t_infall(cls, r, m):
        """
        infall time in sec
        """
        rhoav = cls.rhoav(r, m)
        time = 0.5 * np.sqrt(np.pi / physconst.Kepler.gee) * rhoav**(-0.5)
        return time
#-------------------------------------------------------------------------------
    @classmethod
    def mdot(cls, r, m, rho):
        """
        accretion rate in g/s
        """
        tff = cls.t_infall(r, m)
        rhoav = cls.rhoav(r, m)
        acc = 2 * m / tff * (rho/(rhoav-rho))
        return acc
#-------------------------------------------------------------------------------
    @staticmethod
    def r_gain(mdot, m, time, offset):
        """
        gain radius in cm
        """
        rg = (mdot / m**3 * (1.2e7**3 * physconst.Kepler.solmass**2) + (1.2e6)**3)**(1/3)
        return rg
#-------------------------------------------------------------------------------
    @staticmethod
    def luminosity(mdot, m, r_gain, time, zeta = 0.7, t_15 = 1.2, **kwargs):
        """
        luminosity inoin erg/s
        """
        #t_cool = 5 * (m / (1.5 * physconst.Kepler.solmass))
        #t_cool = t_15 * (m / (1.5 * physconst.Kepler.solmass)) ** (5.0/3.0)
        t_cool = t_15 * (m / (1.5 * physconst.Kepler.solmass)) ** (5.0/3.0)
        t_cool = np.maximum(t_cool, 0.1)
        l0 = ((0.3 * 0.084e0 / physconst.Kepler.solmass * physconst.Kepler.c**2 ) * m**2 / t_cool)
        # initial luminosity \approx binding energy/cooling time scale
        #alpha = np.maximum(0, (1 - physconst.Kepler.gee * m / (5.0/7.0*r_gain * physconst.Kepler.c**2)))         # redshift factor
        lum = (zeta * physconst.Kepler.gee * mdot * m / r_gain + l0 * np.exp(-np.minimum(time / t_cool,100))) #* alpha ** 3
        return lum

#-------------------------------------------------------------------------------

    def get_explosion(
            self,
            dump = None,
            offset = 0,
            ma_conv = None,
            silent = True,
            **kwargs):

        if dump is None:
            dump = self.dump

        if isinstance(dump, str):
            dump = kepdump.load(dump)

        rn = dump.rn[1:-1]
        zm = dump.zm[1:-1]
        dn = dump.dnf[1:-1]
        st = dump.stot[1:-1]

        if ma_conv is not None:
            ma_conv = ma_conv * dump.vconvm[1:-1]**2 / (4 / 3 * dump.pn [1:-1] / dump.dn [1:-1])

        ybind = dump.ybind[1:-1]
        return self.criterion(
            dump.abu,
            rn,
            zm,
            dn,
            st,
            ybind,
            dump.core(),
            offset,
            silent,
            ma_conv = ma_conv,
            **kwargs)

#----------------------------------------------------------------------------------------------------
# Add extra mass and energy to the explosion due to neutrino driven wind (after accretion has ceased)

    def add_wind(self,m_by,lum0,rg,e_expl,m_grav,i_final,radius=1.2e6,t_15=1.2,**kwargs):
        """
        Modifies the explosion energy and remenant mass by adding in the neutrino driven wind phase.

        """
        C=1                                                         # Proportionality constant for neutrino heating i   n the wind phase
        e_rec_alpha = 5 * physconst.MEV / physconst.AMU             # Binding (recombination energy) ???
        epsilon_v = 9.5 * m_by/physconst.Kepler.solmass             # Mean neutrino energy
        #t_cool = 5 * (m_by / (1.5 * physconst.Kepler.solmass))      # Cooling timescale
        t_cool = t_15 * (m_by / (1.5 * physconst.Kepler.solmass)) ** (5.0/3.0)
        L_0 = lum0[i_final]                                         # Neutrino luminosity at the neutron star surface
        #radius = 1.2e6                                             # Neutron star radius
        radius = rg[i_final]
        #radius = 2.2e5
        e_neutrino = L_0 * t_cool                                   # Total energy released by neutrino luminosity (assuming exp L decay)

        # Initial mass loss rate (m_dot) after accretion ceases (at the mass coordinate given by i_final)
        m_dot_0 = 1.14e-10 * C**(5/3) * (L_0/1e51)**(5/3) * epsilon_v**(10/3) * (radius/1e6) **(5/3) * (1.4*physconst.Kepler.solmass/m_by)**2 * physconst.Kepler.solmass

        m_wind = m_dot_0 * t_cool                                   # Mass lost in wind (baryonic)
        e_wind = m_wind * e_rec_alpha                               # Energy contribution in wind

        e_expl =  e_expl + e_wind                               # Modifed explosion energy
        m_by   =  m_by-m_wind                                   # Modified neutron star remenant mass (baryonic)
#        m_grav =  (m_by-m_wind) - 0.084 * (m_by-m_wind)**2 / physconst.Kepler.solmass #(Convert to gravitational mass)
        m_grav = (-1.0 + np.sqrt (1.0 + 4.0 * 0.084* m_by / physconst.Kepler.solmass)) / (2 * 0.084)  * physconst.Kepler.solmass

        if e_expl <=0:  # <-- No modification necessary if there is no explosion (i.e. black hole)
            e_wind     =  0.
            m_wind     =  0.

        return {'e_expl'      : e_expl,
                'm_by'        : m_by,
                'm_grav'      : m_grav,
                'e_wind'      : e_wind,
                'm_wind'      : m_wind,
                }

#-------------------------------------------------------------------------------

    def criterion(
            self,
            abu,
            r,
            m,
            rho,
            stot,
            ybind,
            core,
            offset = 0,
            silent = False,
            threshold = 1,

            # # old parameters
            # alpha_outflow = 0.5,
            # alpha_turb = 1.37557,
            # beta = 7.0,
            # eta_outflow = 1.5,
            # t_15 = 1.2,
            # zeta = 0.7,

            # new set
            alpha_outflow = 0.5,
            alpha_turb = 1.18,
            beta = 4,
            eta_outflow = 0.5,
            t_15 = 1.2,
            zeta = 0.7,

            addwind = True,
            ma_conv = None,
            **kwargs):
        """
        What does this function do???

        INPUT:

        r, m, rho: radius [cm], mass coordinate [g], density [g/cm^3]
            for pre-SN model

        offset: if != 0, this introduces a time delay between the
            response of the neutrino luminosity to changes in the
            accretion rate

        OUTPUT:

        time: infall time [s] (~free-fall time for a specific
           mass shell)

        macc: accretion rate [g/sec]

        mass: macc coordinate [g]

        tadv, theat: advection, heating time-scale [s]

        rsh, rgain: shock radius, gain radius [cm]

        qheat: volume-integrated heating rate

        eta_acc: 'accretion efficiency'

        rho0, r0: initial density [g/cm^3], initial radius [cm] of
            mass shells

        v_shock: shock_velocity [cm/s]

        v_esc: escape velocity at initial location of mass shell
            [cm/s]

        e_diag: diagnositics explosion energy [erg] as a function of
            time

        m_init: mass shell [g] at which shock revival occurs

        m_by: 'final' baryonic neutron star mass [g]

        m_grav: gravitational neutron star mass [g]

        e_expl: 'final/ explosion energy [erg]

        xi17, xi27: compatctness parameter

        T_9s: post-shock temp in 10^9 K
        """

        self.setup_logger(silent)

        con_matzner = 0.794 # 1.03
        t9_oburn = 3.5
        t9_siburn = 5 # 4.8 # incomplete silicon burning
        # t9_siburn = 4.5 # complete silicon burning

        if ma_conv is None:
            ma_conv = np.zeros_like(rho)

        macc = self.mdot(r, m, rho)
        time = self.t_infall(r, m)
        rg = self.r_gain(macc, m, time, offset)
        lum0 = self.luminosity(macc, m, rg, time, t_15 = t_15, zeta = zeta, **kwargs)
        alpha = np.maximum(0, (1 - 2 * physconst.Kepler.gee * m / (5.0/7.0*rg * physconst.Kepler.c**2)))         # redshift factor
        if offset == 0:
            #lume2 = lum0 * 1e-52 * np.minimum (m / physconst.Kepler.solmass, 1.8) ** 2
            lume2 = lum0 * 1e-52 * (m / physconst.Kepler.solmass)**2 * alpha ** 1.5 #* 1.8 ** 0.2
#            lume2 = lum0 * 1e-52 * 1.8 * (m / physconst.Kepler.solmass)
#            lume2 = lum0 * 1e-52 * 1.5**2
        else:
            lume2 = interp1d(time, lum0)(np.maximum(time - offset, 0)) * 1e-52 * (m / physconst.Kepler.solmass)**2

        rsh = 0.55e5 * lume2**(4 / 9) * (rg * 1.e-6)**(16 / 9) / (macc**2 * m)**(1/3) * physconst.Kepler.solmass
        # ;with turbulent pressure:
        rsh = alpha_turb * rsh
        #rsh = 1.37557 * rsh
        #rsh = 1.6 * rsh
#        egain = (0.25 * physconst.Kepler.gee) * m / np.maximum(rsh, rg) + 7 * physconst.MEV / physconst.AMU * 0.75
        egain = (0.25 * physconst.Kepler.gee) * m / np.maximum(rsh, rg) + 8.8 * physconst.MEV / physconst.AMU * 0.75
        egain2 = (0.25 * physconst.Kepler.gee) * m / rg + 8.8 * physconst.MEV / physconst.AMU * 0.75
        tadv = 18e-3 * (np.maximum(rsh, 0.) * 1.e-7)**1.5 * np.log(np.maximum(rsh/rg, 1)) * (m / physconst.Kepler.solmass)**(-0.5)
        mgain = macc * tadv
        theat = 150.e-3 * (egain / 1e19) * (rg / 1.e7)**2 / np.maximum(lume2, 1e-80)
        qheat = (egain * mgain) / theat
        eta_acc = qheat / macc
        psi = np.pi * ma_conv / 2.0 * (macc * m * physconst.Kepler.solmass ** 2/ rg) / np.maximum(qheat,1.0)
        scr = 4.0 / 3.0 * 0.4649
        tadv_theat = tadv / theat * (1 + scr * (1 + psi) ** (2 / 3)) / (1 + scr)

        # ;; Compactness parameter
        if np.max(m/physconst.Kepler.solmass) > 2.5:
            xi25 = interp1d(m/physconst.Kepler.solmass,m/(r*1e-8*physconst.Kepler.solmass))(2.5).tolist() #CHECK-XXX
        else:
            xi25 = 0.0
        xi17 = interp1d(m/physconst.Kepler.solmass,m/(r*1e-8*physconst.Kepler.solmass))(1.75).tolist()
        mc4  = interp1d(stot,m/physconst.Kepler.solmass)(4.0).tolist()
        mu4  = interp1d(m/physconst.Kepler.solmass,r)(mc4+0.3).tolist() - \
            interp1d(m/physconst.Kepler.solmass,r)(mc4).tolist()
        mu4 = 0.6 / (mu4 / 1e8)
#        ebn_env = interp1d(m/physconst.Kepler.solmass,ybind)(core['iron core'].zm_sun).tolist()
        ebn_env = interp1d(m/physconst.Kepler.solmass,ybind)(mc4).tolist()

        # ;; Now estimate the explosion energy:

#        e_rec = 6 * physconst.MEV / physconst.AMU
        e_rec = 5 * physconst.MEV / physconst.AMU
        m_max = 2.05 * physconst.Kepler.solmass
        m_max_by  = m_max + 0.084 * m_max **2 / physconst.Kepler.solmass

        try:
            jj0 = np.where(np.logical_and (
                    tadv_theat > threshold,
                    m < m_max_by))[0][0]
            jj = np.arange(jj0, len(m))
        except IndexError:
            jj = np.array([len(m)-1], dtype = np.int)

        m_init = 0.0#1.e6 * physconst.Kepler.solmass

        if len(jj) > 1:
            i_expl = np.min(jj)
            m_init = m[i_expl - 1]
        else:
            i_final = len(m)

        v_esc = np.sqrt(2 * physconst.Kepler.gee * m/r)

        e_diag = np.zeros_like(m)
        v_shock = np.zeros_like(m)
        T_9s = np.zeros_like(m)

        eburn = 0.
        #ME = np.zeros((m.shape[0], 4))     #if we don't want to calculate burning energies
        try:
            ME = abu._ME
            Ni = abu._Ni
        except:
            #self.start_timer('X')
            ME = np.ndarray((m.shape[0], 4))
            Ni = np.ndarray((m.shape[0], 4))
            burners = [None, 'o_burn', 'si_burn', 'alpha_burn']
            for i, b in enumerate(burners):
                xE, xNi = abu.xENi56(b)
                ME[:, i] = xE[1:-1]
                Ni[:, i] = xNi[1:-1]
            abu._ME = ME
            abu._Ni = Ni
            #print('took ', time2human(self.finish_timer('X')))

        mNi56 = 0.
        mO16 = 0.
        m_by_rem = m_init
        e_delayed = 0.
        m_neutrino = 0.
        i_kick = 0
        for i in jj[:-1]:
#            e_diag[i] = e_diag[i-1] + (m[i] - m[i-1]) * (eta_acc[i] / (physconst.Kepler.gee * m[i] / rg[i])) * alpha_outflow * e_rec
#            e_diag[i] = e_diag[i-1] + (m[i] - m[i-1]) * (eta_acc[i] / egain[i]) * alpha_outflow * e_rec
            eta_outflow = 1.0 - alpha_outflow
            dot_m_sh = 4.0 * np.pi * v_shock[i-1] * r [i-1]**2 * rho [i-1]
            scr1 = 1.0
            if (eta_outflow * dot_m_sh > macc[i]):
                scr1 = macc[i] / dot_m_sh
#           eta_tmp = (eta_acc[i] / (physconst.Kepler.gee * m[i] / rg[i])) * alpha_outflow * scr1

            eta_1 = max( [(eta_acc[i] / np.abs(egain[i])), 0.0]) # * alpha_outflow
#            eta_1 = min( [eta_1,1.0])
            eta_2 =  eta_1 * scr1
#            eta_tmp = max ([eta_tmp*(1.0-eta_tmp),0.0])
#            alpha_outflow = eta_tmp
#            eta_tmp = eta_tmp * (1.0-alpha_outflow)

            m_out = (m[i] - m[i-1]) * eta_2
            e_delayed = e_delayed + (m[i] - m[i-1]) * (eta_outflow * eta_1 - eta_2) * e_rec
#            e_delayed = e_delayed + (m[i] - m[i-1]) * max([eta_outflow * eta_1 - eta_2,0.0]) * e_rec
            e_diag[i] = e_diag[i-1] + m_out * e_rec + (ybind [i-1] - ybind[i])  * alpha_outflow
            v_shock[i] = (con_matzner * np.sqrt(max(e_diag[i],0.0) / (m[i] - m_init)) * ((m[i] - m_init) / (rho[i] * r[i]**3))**0.19)

            m_by_rem = m_by_rem + (m[i] - m[i-1]) * (1 - alpha_outflow)*(1.0-eta_1) #- m_out
#            m_by_rem = m_by_rem + (m[i] - m[i-1]) * (1.0 - alpha_outflow)


            #if i == jj[0]:
             #   v_shock[i] = 0.5*v_esc[i]
            T_9s[i] = ((3/physconst.ARAD)*(beta/(beta-1))*rho[i]*v_shock[i]**2)**0.25*1e-9
            iregion = 0
            if T_9s[i] > t9_oburn:
                iregion = 1
            if T_9s[i] > t9_siburn:
                iregion = 2
            #print('*', i, T_9s[i], np.log10(rho[i]) ,11.62 + 1.5*np.log10(T_9s[i])-39.17/T_9s[i])
            if np.log10(beta*rho[i]) < 11.62 + 1.5 * np.log10(T_9s[i])-39.17/T_9s[i] and T_9s[i] > t9_siburn:
                iregion = 3
            eburn_i = ME[i, iregion] * alpha_outflow
            e_diag[i] += eburn_i
            mNi56 += Ni[i, iregion] * alpha_outflow
            if iregion == 0:
                mO16 += alpha_outflow * (m[i] - m [i-1]) * abu['O16'][i]
            eburn += eburn_i
            # print(i, iregion, v_shock[i], T_9s[i], beta*rho[i],  eburn_i)
            # determining final mass cut, v_shock ~ v_esc
            m_grav = (-1.0 + np.sqrt (1.0 + 4.0 * 0.084 * m_by_rem / physconst.Kepler.solmass)) / (2 * 0.084)  * physconst.Kepler.solmass
            if (((beta-1)/beta) * v_shock[i] > v_esc[i]):
                i_final = i
                e_diag [i_final] = e_diag [i_final] #+ e_delayed
                e_expl = e_diag[i_final]
                i_kick = i_final
                break
            if (m_by_rem > m_max_by):
                m_by_rem = np.max(m)
                m_grav = np.max(m)
                e_expl = 0.
                e_delayed = 0.
                i_final = jj[-1]
                mNi56 = 0.
                break
        else:
            m_by_rem = np.max(m)
            m_grav = np.max(m)
            e_expl = 0.
            mNi56 = 0.
            i_final = jj[-1]

        m_grav = (-1 + np.sqrt (1 + 4 * 0.084 * m_by_rem / physconst.Kepler.solmass)) / (2 * 0.084)  * physconst.Kepler.solmass
        m_wind = 0.0
        e_wind = 0.0
#         if addwind:
#            if (m_grav < m_max):
#                wind=self.add_wind(m_by_rem,lum0,rg,e_expl,m_grav,i_final,t_15=t_15,**kwargs)
#                e_diag[i_final] = wind['e_expl']
#                m_by_rem = wind['m_by']
#                m_grav = wind['m_grav']
#                m_wind = wind['m_wind']
#                e_wind = wind['e_wind']
        if m_by_rem > m_max_by:
            m_grav = m_by_rem

        # Continuing after break at i_final
        for i in range(i_final+1, jj[-1]):
            e_diag[i] = e_diag[i-1]
            eexp = e_diag[i] + (ybind[i-1] - ybind[i]) #+ (m[i]-m_by_rem)*m_grav*physconst.Kepler.gee/r[i]
            v_shock[i] = (con_matzner * np.sqrt(max([eexp, 0]) / (m[i] - m_init)) * ((m[i] - m_init) / (rho[i] * r[i]**3))**0.19)
            T_9s[i] = ((3 / physconst.ARAD)*(beta/(beta-1)) * rho[i] * v_shock[i]**2)**0.25*10**-9

            iregion = 0
            if T_9s[i] > t9_oburn:
                iregion = 1
            if T_9s[i] > t9_siburn:
                iregion = 2
            if T_9s[i] > t9_siburn and np.log10(beta*rho[i]) < 11.62 + 1.5 * np.log10(T_9s[i])-39.17/T_9s[i]:
                iregion = 3
#            if iregion == 0:
#                e_expl = e_diag [i] + ybind [i]
#                break
            eburn_i = ME [i, iregion]
            e_diag[i] = eexp + eburn_i
            e_expl = e_diag [i]
            eburn += eburn_i
            mNi56 += Ni[i, iregion]
            if iregion == 0:
                mO16 += alpha_outflow * (m[i] - m [i-1]) * abu['O16'][i]

#        mNi56 += np.sum(Ni[i:, 0])

        if (m_init > m_max_by):
            m_init = 0.0
        # m_init = (-1 + np.sqrt (1 + 4 * 0.084 * m_init / physconst.Kepler.solmass)) / (2 * 0.084)  * physconst.Kepler.solmass

        if m_grav < m_max:
            e_expl = e_expl + e_delayed
        if e_expl < 1e45:
            m_by_rem = np.max(m)
            m_grav = np.max(m)
            e_expl = 0.
            mNi56 = 0.
            i_final = jj[-1]
#            m_init = 0.

        #Calculate kick velocity
#        i_kick = i_final
#        v_kick = 0.25 * np.sqrt(2.0 * m_neutrino * e_expl) / m_grav
        v_kick = 0.16 * np.sqrt(max(0.0,2.0 * (m[i_kick]-m_init) * e_expl)) / m_grav
#        v_kick = 0.1 * np.sqrt(0.5 * physconst.Kepler.gee *  (m[i_kick]-m_init) / 1e7)#r [i_kick])
        if (e_expl < 1e45):
            v_kick = -1e5

        self.logger.info('star mass:         {:6.3f} [M_sun]'.format(m[-1] / physconst.Kepler.solmass))
        self.logger.info('Ni56 mass:         {:6.4f} [M_sun]'.format(mNi56 / physconst.Kepler.solmass))
        self.logger.info('Explosion energy:  {:6.3f} [B]'.format(e_expl * 1e-51))
        self.logger.info('Burning energy:    {:6.3f} [B]'.format(eburn * 1.e-51))
        if e_expl == 0:
            self.logger.info('Black hole mass:   {:6.3f} [M_sun]'.format(m_grav / physconst.Kepler.solmass))
        else:
            self.logger.info('Neutron star mass: {:6.3f} [M_sun]'.format(m_grav / physconst.Kepler.solmass))
        self.logger.info('-'*72)


        # results section

        result = {
            'crit' : tadv_theat,
            'lum' : lum0,
            'time' : time,
            'macc' : macc,
            'mass' : m,
            'tadv' : tadv,
            'theat' : theat,
            'rsh' : rsh,
            'rgain' : rg,
            'egain' : egain,
            'lume2' : lume2,
            'qheat' : qheat,
            'eta_acc' : eta_acc,
            'rho0' : rho,
            'r0' : r,
            'v_shock' : v_shock,
            'v_esc' : v_esc,
            'e_diag' : e_diag,
            'm_init' : m_init,
            'm_by' : m_by_rem,
            'm_grav' : m_grav,
            'e_expl' : e_expl,
            'e_wind' : e_wind,
            'm_wind' : m_wind,
            'xi25' : xi25,
            'xi17' : xi17,
            'mc4' : mc4,
            'mu4' : mu4,
            'm_star' : m[-1],
            'r_star' : r[-1],
            'm_ej' : m[-1]-m_by_rem,
            'eburn' : eburn,
            'Ni56' : mNi56,
            'O16' : mO16,
            'T_9s' : T_9s,
            'ebn_env': ebn_env,
            'i_final' : i_final,
            'fe_core' : core['iron core'].zm_sun,
            'si_core' : core['Si core'].zm_sun,
            'o_shell' : core['O shell'].zm_sun,
            'ye_core' : core['ye core'].zm_sun,
            'he_core' : core['He core'].zm_sun,
            'CO_core' : core['C/O core'].zm_sun,
            'm_h'     : m[-1]/physconst.Kepler.solmass-core['He core'].zm_sun,
            'v_kick'  : v_kick,
            }

#        print(eta_acc[i_expl] / np.abs(egain[i_expl]),np.abs(egain[i_expl])*1.66e-24/1.602e-6)

        self.close_logger()

        return result

#######################################################################


class SNSet(SN, Logged):
    def __init__(self, dumps = None):
        """
        Routine to load set of pre sn models.

        Enter the directory (string) of a single file to load that single dump.
        Enter nothing to load all models from default directory.
        Enter any other values to load nothing.
        Enter 0 to chose single model from the data directory.

        Note: If all models have already been loaded for another object (eg: sn), enter 'sn.dumps' in order to just copy these to the new
              object, rather than load them all again. This is virtually instantaneous, and the recalculation spee is much faster
              (i.e. saves a LOT of time).

        copy
        rhoav
        t_infall
        mdot
        r_gain
        luminosity
        criterion
        get_explosion
        update
        newfigure
        recalculate
        plot
        update
        plotsave
        hist
        add_wind
        add_plot
        dav_plot
        dav_plots
        ert_plot

        """
        if isinstance(dumps, str):
            dump = kepdump.loaddump(dumps)
            self.dumps = [dump]
        elif dumps is None:
            dumps = Presn()
            self.dumps = dumps
        elif dumps == 0:
            dump = kepdump.loaddump('/Users/bmueller/kepler/models/')
            self.dumps = [dump]
        else:
            self.dumps = dumps

#-------------------------------------------------------------------------------

    def update(self, *args, **kwargs):
        self.dumps.update()
        self.plot(*args, **kwargs)
        self.plotsave(*args, **kwargs)

#-------------------------------------------------------------------------------

    def recalculate(
            self,
            offset = 0,
            silent = False,
            threshold = 1,

            alpha_outflow = 0.5,
            alpha_turb = 1.18,
            beta = 4,
            eta_outflow = 0.5,
            t_15 = 1.2,
            zeta = 0.7,

            ma_conv = None,
            addwind = True,
            **kwargs):

        dumps = self.dumps
        if dumps is None:
            dumps = Presn()
            self.dumps = dumps

        explosions = {}
        for dump in dumps:
            explosion = self.get_explosion(
                dump,
                offset = offset,
                threshold = threshold,

                alpha_outflow = alpha_outflow,
                alpha_turb = alpha_turb,
                beta = beta,
                eta_outflow = eta_outflow,
                t_15=t_15,
                zeta = zeta,

                ma_conv = ma_conv,
                addwind = addwind,
                silent = silent,
                **kwargs)
            explosions[dump.mass] = explosion

        parameters = {
            'alpha_outflow' : alpha_outflow,
            'alpha_turb' : alpha_turb,
            'beta': beta,
            'threshold' : threshold,
            'zeta' : zeta
            }

        self.parameters = parameters

        self.explosions = explosions

#-------------------------------------------------------------------------------
    @classmethod
    def copy(cls, other):
        new = cls(other.dumps)
        try:
            new.f = other.f
            new.ax = other.ax
            new.ax2 = other.ax2
        except:
            pass
        return new

#-------------------------------------------------------------------------------

    def newfigure(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        self.ax = ax
        self.ax2 = ax.twinx()
        self.f = f

#-------------------------------------------------------------------------------

    def plot(self, **kwargs):
        try:
            f = self.f
            if f.canvas.manager.window is None:
                raise Exception()
            ax = self.ax
            ax2 = self.ax2
            xlim0 = ax.get_xlim()
            ylim0 = ax.get_ylim()
            ax.clear()
            ax2.clear()
        except:
            self.newfigure()
            f = self.f
            ax = self.ax
            ax2 = self.ax2
            xlim0 = None
            ylim0 = None

        xr = []
        xe = []
        m  = []
        xn = []
        e_expl = []
        m_grav = []

        for mass, explosion in self.explosions.items():
            m  += [mass]
            xn += [explosion['Ni56']]
            xe += [explosion['e_expl']]
            xr += [explosion['m_grav']]
            e_expl      += [explosion['e_expl']]          # Explosion energy
            m_grav += [explosion['m_grav']]     # Final (gravitational) mass of remenant neutron star

        xe = np.array(xe) * 1.e-51
        xr = np.array(xr) / physconst.Kepler.solmass
        xn = np.array(xn) / physconst.Kepler.solmass *100
        e_expl = np.array(e_expl) *1.e-51
        m_grav = np.array(m_grav) / physconst.Kepler.solmass
        m  = np.array(m)

        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_xlabel(r'initial mass / solar masses')
        '''
        ax.plot(m, xr, 'r+',)
        ax.set_ylabel(r'remnant mass / solar masses')#, color = 'r')

        ax2.plot(m, xe, 'g+')
        ax2.set_ylabel(r' explosion energy / B')#, color = 'g')
        '''
        ax.plot(m, m_grav, 'r+',)
        ax.set_ylabel(r'remnant mass / solar masses')

        ax2.plot(m, e_expl, 'g+')
        ax2.set_ylabel(r' explosion energy / B')

        # ax.set_ylim([1.2, 5.8])
        ax.set_xlim([min(m)-0.1, max(m)+0.1])
        # ax.legend(loc = 'best', numpoints = 1)
        # ax2.legend(loc = 'best', numpoints = 1)

        P = self.parameters
        plt.title('eta = {}, threshold = {}, beta = {}, zeta = {}'.format(P['alpha_outflow'],P['threshold'],P['beta'],P['zeta']))

        #f.tight_layout()

        plt.draw()

#-------------------------------------------------------------------------------

    def update(self, *args, **kwargs):
        self.dumps.update()
        self.plot(*args, **kwargs)

#-------------------------------------------------------------------------------

    def plotsave(self, **kwargs):
        try:
            f = self.f
            if f.canvas.manager.window is None:
                raise Exception()
            ax = self.ax
            ax2 = self.ax2
            xlim0 = ax.get_xlim()
            ylim0 = ax.get_ylim()
            ax.clear()
            ax2.clear()
        except:
            self.newfigure()
            f = self.f
            ax = self.ax
            ax2 = self.ax2
            xlim0 = None
            ylim0 = None

        dumps = self.dumps
        if dumps is None:
            dumps = Presn()
            self.dumps = dumps

        xr = []
        xe = []
        m  = []

        for dump in dumps:
            explosion = self.get_explosion(dump, silent = True, **kwargs)
            m += [dump.mass]
            xe += [explosion['e_expl']]
            xr += [explosion['m_grav']]
            explosions[dump.mass] = explosion
        self.explosions = explosions

        xe = np.array(xe) * 1.e-51
        xr = np.array(xr) / physconst.Kepler.solmass
        m  = np.array(m)

        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_xlabel(r'initial mass / solar masses')

        ax.plot(m, xr, 'r+',)
        ax.set_ylabel(r'remnant mass / solar masses', color = 'r')

        ax2.plot(m, xe, 'g+')
        ax2.set_ylabel(r' explosion energy / B', color = 'g')

#    ax.set_ylim([1.2, 5.8])
        ax.set_xlim([min(m)-0.1, max(m)+0.1])
        # ax.legend(loc = 'best', numpoints = 1)
        # ax2.legend(loc = 'best', numpoints = 1)
        f.tight_layout()

#	saving figure instead of drawing


#-------------------------------------------------------------------------------

    def hist_energy(self, weight = 'imf', stacking = True, **kwargs):

        xr = []
        xe = []
        mej = []
        mni = []
        xrout = []
        m = []

        for mass, explosion in self.explosions.items():
            m += [mass]
            xe += [explosion['e_expl']]
            xr += [explosion['m_grav']]
            mni += [explosion['Ni56']]
            mej += [explosion['m_star']]#-explosion['m_by']]
#            mej += [explosion['m_h']]
            xrout += [explosion['r_star']]

        xe = np.array(xe) * 1.e-51
        xr = np.array(xr) / physconst.Kepler.solmass
        m  = np.array(m)
        mni  = np.array(mni) /physconst.Kepler.solmass
        mej  = np.array(mej) /physconst.Kepler.solmass
        xrout  = np.array(xrout) / physconst.Kepler.solrad
        lum = xe ** (5.0/6.0) / np.sqrt(mej+1e-6) * xrout ** (2.0/3.0)
        #lum = mni**(3.53)

        ii = np.argsort(m)
        m = m[ii]
        xr = xr[ii]
        xe = xe[ii]

        mx = np.ndarray(len(m)+1)
        mx[1:-1] = 0.5 * (m[:-1] + m[1:])
        mx[[0,-1]] = m[[0,-1]]
        imf = -(mx[1:]**(-1.35) - mx[:-1]**(-1.35))/1.35
        if weight == 'lum':
            imf = imf * lum ** 1.5

        limits = [12, 15, 18, 20, 25, 1.e99]
        m_ = [[] for i in range(len(limits))]
        xe_ = [[] for i in range(len(limits))]
        xr_ = [[] for i in range(len(limits))]
        imf_ = [[] for i in range(len(limits))]

        for _m, _xe, _xr, _imf in zip(m, xe, xr, imf):
            for i,l in enumerate(limits):
                if (_m <= l):
                    m_[i] += [_m]
                    xe_[i] += [_xe]
                    xr_[i] += [_xr]
                    imf_[i] += [_imf]
                    break

        plt.clf()
        myfigsize = (5,4)
        plt.figure(figsize = myfigsize)

        #color=['red', xcolor.webcolors['Olive'],'yellow','green','blue','black']


        label = []
        for i in range(len(m_)):
            l = '$'
            if i >= 1:
                l += '{:d}<'.format(limits[i-1])
            l += 'm'
            if i < len(m_)-1:
#                l += r'\le{:d}'.format(limits[i])
                l += r'<{:d}'.format(limits[i])
            l += '$'
            label += [l]

        color = xcolor.isocolors(len(m_))
        color = xcolor.isoshadecolor(len(m_),hue=-120)

        c = xcolor.colormap('jet')
        from matplotlib.colors import rgb2hex
        color = [rgb2hex(c(x)) for x in np.linspace(0, 1, len(m_))]

        if stacking:
            plt.hist(xe_,bins=np.linspace(0.1,2,25),
                     weights=imf_,normed=True,cumulative=False,stacked=True,
                     color = color, label = label, rwidth = 1.0
                     )
        else:
            plt.hist(xe,bins=np.linspace
                     (1,2.2,25),weights=imf,normed=True,cumulative=False,rwidth=1.0)
            plt.xlim(1.0,2.2)
            #plt.title("Neutron star distribution")

        #plt.title("")
        plt.xlabel(r"$E_\mathrm{expl} \ [10^{51} \ \mathrm{erg}]$")
        plt.ylabel(r"$\mathrm{probability\ density} \ [10^{-51} \ \mathrm{erg}^{-1}]$")
        plt.legend(loc='best', prop={'size':12})
        plt.tight_layout()
        plt.show()
        plt.savefig('histogram_energy.pdf')


    def hist_frac(self, stacking = False, **kwargs):

        m = []
        xr = []
        xe = []

        for mass, explosion in self.explosions.items():
            m += [mass]
            xe += [explosion['e_expl']]
            xr += [explosion['m_grav']]

        xe = np.array(xe)
        xr = np.array(xr) / physconst.Kepler.solmass
        m  = np.array(m)

        ii = np.argsort(m)
        m = m[ii]
        xr = xr[ii]
        xe = xe[ii]

        limits = np.linspace (10,32,45)
        m_ = [[] for i in range(len(limits))]
        xt_ = [[] for i in range(len(limits))]
        xe_ = [[] for i in range(len(limits))]

        for _m, _xe, _xr in zip(m, xe, xr):
            for i,l in enumerate(limits.tolist()):
                if (_m <= l):
                    m_[i] += [_m]
                    xt_[i] += [_xe]
                    break

        for _m, _xe, _xr in zip(m, xe, xr):
            for i,l in enumerate(limits.tolist()):
                if (_m <= l and _xe >1e45):
                    m_[i] += [_m]
                    xe_[i] += [_xe]
                    break

        expl_frac = 1+np.zeros(len(xt_))
        for i in range(len(xt_)):
            if (len(xt_[i]) > 0):
                expl_frac [i] = len(xe_[i]) / float(len(xt_[i]))


#        limits = limits - 0.5 * (limits[1] - limits[0])
        plt.clf()
        myfigsize = (4.5,3.5)
        plt.figure(figsize = myfigsize)

        plt.bar(limits,expl_frac,width=0.5)
        plt.xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
        plt.ylabel(r'$P_\mathrm{expl}$')
        plt.xlim([10,32.5])

        plt.tight_layout()
        plt.show()
        plt.savefig('percentage.pdf')

        # pcum = np.zeros(len(xt_))
        # for i in range (len(xt_)):
        #     if i > 0:
        #         pcum [i] = pcum [i-1] + expl_frac [i] / limits [i] ** 2.3
        #     else:
        #         pcum [i] = expl_frac [i] / limits [i] ** 2.3


        # pcum = pcum / pcum [len(xt_) - 1]
        # print (pcum)

        # detec = np.array ([7,8,9,10,10,10,12,12,13,15,16,16])
        # detec = np.array ([8.,9,10,12,12,12,13,13,14,14,16,18,18,12,12,12,14,14,14,14,16,16,16,16,18,20])
        # detec = np.sort(detec)
        # pcumdet = np.linspace(0,1,len(detec))

        # plt.clf()
        # plt.plot(limits,pcum)
        # plt.plot(detec,pcumdet)
        # plt.savefig('cum_dist.pdf')

        return limits

    def hist (self, region = 'ns', stacking = True, **kwargs):

        m = []
        xr = []
        xe = []

        for mass, explosion in self.explosions.items():
            m += [mass]
            xe += [explosion['e_expl']]

            xr += [explosion['m_grav']]

        xe = np.array(xe) * 1.e-51
        xr = np.array(xr) / physconst.Kepler.solmass
        m  = np.array(m)

        ii = np.argsort(m)
        m = m[ii]
        xr = xr[ii]
        xe = xe[ii]

        mx = np.ndarray(len(m)+1)
        mx[1:-1] = 0.5 * (m[:-1] + m[1:])
        mx[[0,-1]] = m[[0,-1]]
        imf = -(mx[1:]**(-1.35) - mx[:-1]**(-1.35))/1.35

        limits = [12, 15, 18, 20, 25, 1.e99]

        m_ = [[] for i in range(len(limits))]
        xe_ = [[] for i in range(len(limits))]
        xr_ = [[] for i in range(len(limits))]
        imf_ = [[] for i in range(len(limits))]

        for _m, _xe, _xr, _imf in zip(m, xe, xr, imf):
            for i,l in enumerate(limits):
                if (_m <= l):
                    m_[i] += [_m]
                    xe_[i] += [_xe]
                    xr_[i] += [_xr]
                    imf_[i] += [_imf]
                    break

        plt.clf()
        myfigsize = (5,4)
        plt.figure(figsize = myfigsize)

        # color=['red', xcolor.webcolors['Olive'],'yellow','green','blue','black']

        label = []
        for i in range(len(m_)):
            l = '$'
            if i >= 1:
                l += '{:d}<'.format(limits[i-1])
            l += 'm'
            if i < len(m_)-1:
#                l += r'\le{:d}'.format(limits[i])
                l += r'<{:d}'.format(limits[i])
            l += '$'
            label += [l]

        color = xcolor.isocolors(len(m_))
        color = xcolor.isoshadecolor(len(m_),hue=-120)

        c = xcolor.colormap('jet')
        from matplotlib.colors import rgb2hex
        color = [rgb2hex(c(x)) for x in np.linspace(0, 1, len(m_))]

        if region == 'ns':
            if stacking:
                plt.hist(xr_,bins=np.linspace(1.025,2.225,25),
                         weights=imf_,normed=True,cumulative=False,stacked=True,
                         color = color, label = label, rwidth = 1.0
                         )
            else:
                plt.hist(xr,bins=np.linspace
                         (1,2.2,25),weights=imf,normed=True,cumulative=False)
            plt.xlim(1.0,2.2)
            #plt.title("Neutron star distribution")
        if region == 'all':
            if stacking:
                plt.hist(xr_,bins=np.linspace(1,20,96),
                weights=imf_,normed=True,cumulative=False,stacked=True,
                color = color, label = label, rwidth =1.0
                )
            else:
                plt.hist(xr,bins=np.linspace(1,20,96),weights=imf,normed=True,cumulative=False,rwidth=1.0)
            plt.xlim(1.0,20.0)
            #plt.title("Remnant mass distribution")
        if region == 'bh':
            if stacking:
                plt.hist(xr_,bins=np.linspace(12,18,13),
                weights=imf_,normed=True,cumulative=False,stacked=True,
                color = color, label = label, rwidth =1.0
                )
            else:
                plt.hist(xr,bins=np.linspace(12,18,13),weights=imf,normed=True,cumulative=False,rwidth=1.0)
            plt.xlim(12.0,18.0)
            #plt.title("Black hole distribution")

        #plt.title("")
        plt.xlabel(r"$M_\mathrm{NS} \ [M_\odot]$")
        plt.ylabel(r"$\mathrm{probability\ density} \ [M_\odot^{-1}]$")
        plt.legend(loc='best', prop={'size':12})
        P = self.parameters
#        plt.title(r"$\alpha_\mathrm{out}$ = {}, threshold = {}, beta = {}, zeta = {}".format(P['alpha_outflow'],P['threshold'],P['beta'],P['zeta']))
        plt.tight_layout()
        plt.show()
        plt.savefig('histogram.pdf')

#        xr=xr.tolist()
#        imf=imf.tolist()
        jj = np.where(xr < 3.)
        xr = xr [jj]
        imf = imf [jj]
        mns_av = np.average (xr, weights = imf)
        print ('Average neutron star mass',mns_av)

#-------------------------------------------------------------------------------
# Compare with and without wind contribution -----------------------------------

    def add_plot(self,**kwargs):
        """
        Routine to plot the modified explosion energy and rememant mass (due to neutrino driven wind phase).

        No argument needed.

        """

        # Initialise lists for grid
        m          = []
        m_grav = []
        e_expl      = []

        ax=self.ax
        ax2=self.ax2

        # Loop over all the models
        for mass, explosion in self.explosions.items():
            m          += [mass]                        # Zero-age progenitor mass
            e_expl      += [explosion['e_expl']]          # Explosion energy
            m_grav += [explosion['m_grav']]     # Final (gravitational) mass of remenant neutron star

        # Convert lists into arrays
        m          = np.array(m)
        e_expl      = np.array(e_expl) * 1.e-51
        m_grav = np.array(m_grav) / physconst.Kepler.solmass

        #ax.plot(m,m_grav,'r+')#,'y+')  # Plot values for left y-axis
        ax2.plot(m,e_expl,'c+')#,'c+')      # Plot values for right y-axis

        plt.draw()

#-------------------------------------------------------------------------------

    def dav_plot(self,extra=False,poznanski=False,**kwargs):

        xr = []
        xe = []
        m  = []
        xn = []
        e_expl = []
        m_grav = []

        for mass, explosion in self.explosions.items():
            m  += [mass]
            #m_grav  += [explosion['m_ej'] / physconst.Kepler.solmass]
            xn += [explosion['Ni56']]
            xe += [explosion['e_expl']]
            xr += [explosion['m_grav']]
            e_expl      += [explosion['e_expl']]          # Explosion energy
            m_grav += [explosion['m_grav']]     # Final (gravitational) mass of remenant neutron star

        xe = np.array(xe) * 1.e-51
        xr = np.array(xr) / physconst.Kepler.solmass
        xn = np.array(xn) / physconst.Kepler.solmass
        e_expl = np.array(e_expl) *1.e-51
        m_grav = np.array(m_grav) / physconst.Kepler.solmass
        m  = np.array(m)

        f, (ax1a, ax2) = plt.subplots(2, 1)
#        f, (ax1a) = plt.subplots(1, 1)

        ax1b = ax1a.twinx()
        ax1a.set_xlim([min(m)-0.1, max(m)+0.1])
        ax1b.set_xlim([min(m)-0.1, max(m)+0.1])
        ax2.set_xlim([min(m)-0.1, max(m)+0.1])

        ax2.plot(m, xn, 'bx')
        ax1a.plot(m, m_grav, 'r+',)
        ax1b.plot(m, e_expl, 'g+')

        if poznanski:
            ax1b.plot(np.sort(m), 0.2 * (np.sort(m) / 11.0) **3, 'g')
            ax1b.set_xlim([8, 28])
            ax1b.set_ylim([0, 1.5])


        # Option to add a third plot with a different gain radius
        if extra:
            ax1b.plot(m, xe, 'y+')
            self.recalculate(alpha_outflow=0.5,threshold=1,beta=4,zeta=0.8,radius=2.2e6)
            e_22km=[]
            for mass, explosion in self.explosions.items():
                e_22km += [explosion['e_expl']]
            e_22km = np.array(e_22km) *1.e-51
            print(e_22km)
            ax1b.plot(m, e_22km, 'c+')

        ax1a.set_ylabel(r'Remnant mass / solar masses')
        ax1b.set_ylabel(r'Explosion energy / B')
        ax1a.set_xlabel(r'initial mass / solar masses')
        ax2.set_ylabel(r'Nickel-56 mass / solar masses')
        ax2.set_xscale('linear')
        ax2.set_yscale('linear')
        ax2.set_xlabel(r'initial mass / solar masses')

        P=self.parameters
        f.suptitle('eta = {}, threshold = {}, beta = {}, zeta = {}'.format(P['alpha_outflow'],P['threshold'],P['beta'],P['zeta']))

#-------------------------------------------------------------------------------

    def various_plots(self,**kwargs):

        m = []
        m_init = []
        m_by = []
        m_star = []
        mc4 = []
        fe_core = []
        m_he = []
        m_co = []
        mu4 = []
        xi17 = []
        xi25 = []
        ebn_env = []
        e_expl = []
        print ("hic")

        for mass, explosion in self.explosions.items():
             m_init += [explosion['m_init']]
             m_by += [explosion['m_by']]
             m_star += [explosion['m_star']]
             mc4 += [explosion['mc4']]
             fe_core += [explosion['fe_core']]
             mu4 += [explosion['mu4']]
             m_he += [explosion['he_core']]
             m_co += [explosion['CO_core']]
             xi17 += [explosion['xi17']]
             xi25 += [explosion['xi25']]
             ebn_env += [explosion['ebn_env']]
             m += [mass]
             e_expl      += [explosion['e_expl']]          # Explosion energy

        colors = np.empty(len(m))

        e_expl = np.array(e_expl)
        m = np. array (m)
        m_star = np. array (m_star)
        m_init = np. array (m_init)
        m_by = np. array (m_by)
        xi17 = np.array(xi17)
        xi25 = np.array(xi25)
        ebn_env = np.array(ebn_env)
        mu4 = np.array(mu4)
        mc4 = np.array(mc4)
        fe_core = np.array(fe_core)
        m_he = np.array(m_he)
        m_co = np.array(m_co)
        jj1 = np.where(e_expl > 1e45)
        jj0 = np.where(e_expl <= 1e45)
        jj0a = np.where(np.logical_and(e_expl <= 1e45, m_init >= 0.01))
        jj0b = np.where(np.logical_and(e_expl <= 1e45, m_init < 0.01))

        plt.clf()
        myfigsize = (4,3)
        plt.figure(figsize = myfigsize)


        plt.scatter(m[jj1], xi25[jj1], color='r', s=2, marker ='.')
        plt.scatter(m[jj0a], xi25[jj0a], color='b', s=2, marker ='.')
        plt.scatter(m[jj0b], xi25[jj0b], color='k', s=2, marker ='.')

        plt.xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
#        plt.xlabel(r'$\xi_{17}$')
        plt.ylabel(r'$\xi_{2.5}$')
        plt.xlim ([10,32.6])
        plt.ylim ([0,0.6])
        plt.minorticks_on()
#        plt.set_xticks (np.arange(10,33,1), minor = True)
#        plt.set_yticks (np.arange(0,0.7,0.2), minor = True)
        plt.tight_layout()
        plt.savefig('xi_parameter.pdf')

        xicrit =0.278
        print ("xi25 false positives: ",(np.where (xi25[jj0] < xicrit))[0].size)
        print ("xi25 false negatives: ",(np.where (xi25[jj1] > xicrit))[0].size)
        print ("xi25 false: ",(np.where (xi25[jj0] < xicrit))[0].size + \
                   (np.where (xi25[jj1] > xicrit))[0].size)


        plt.clf()
        myfigsize = (4,3)
        plt.figure(figsize = myfigsize)


        plt.scatter(m_he[jj1], xi25[jj1], color='r', s=2, marker ='.')
        plt.scatter(m_he[jj0a], xi25[jj0a], color='b', s=2, marker ='.')
        plt.scatter(m_he[jj0b], xi25[jj0b], color='k', s=2, marker ='.')

        plt.xlabel(r'$M_\mathrm{He} \  [M_\odot]$')
#        plt.xlabel(r'$\xi_{17}$')
        plt.ylabel(r'$\xi_{2.5}$')
        plt.xlim ([2,12])
        plt.ylim ([0,0.6])
        plt.minorticks_on()
#        plt.set_xticks (np.arange(10,33,1), minor = True)
#        plt.set_yticks (np.arange(0,0.7,0.2), minor = True)
        plt.tight_layout()
        plt.savefig('xi_parameter_he.pdf')


        plt.clf()
        plt.figure(figsize = myfigsize)


        plt.scatter(m_co[jj1], xi25[jj1], color='r', s=2, marker ='.')
        plt.scatter(m_co[jj0a], xi25[jj0a], color='b', s=2, marker ='.')
        plt.scatter(m_co[jj0b], xi25[jj0b], color='k', s=2, marker ='.')

        plt.xlabel(r'$M_\mathrm{C/O} \  [M_\odot]$')
#        plt.xlabel(r'$\xi_{17}$')
        plt.ylabel(r'$\xi_{2.5}$')
        plt.xlim ([1,10])
        plt.ylim ([0,0.6])
        plt.minorticks_on()
#        plt.set_xticks (np.arange(10,33,1), minor = True)
#        plt.set_yticks (np.arange(0,0.7,0.2), minor = True)
        plt.tight_layout()
        plt.savefig('xi_parameter_co.pdf')


        plt.clf()

        plt.scatter(xi17[jj1], e_expl[jj1]/1e51, color='k', s=2, marker ='.')

        plt.ylabel(r'$E_\mathrm{expl} \  [10^{51} \ \mathrm{erg}]$')
        plt.xlabel(r'$\xi_{1.75}$')
        plt.xlim ([0,0.9])
        plt.ylim ([0,2.5])
        plt.tight_layout()
        plt.savefig('e_vs_xi.pdf')


        plt.clf()
#        plt.figure(figsize = myfigsize)
        yvar = m_star / physconst.Kepler.solmass - m_he
#        yvar=-ebn_env/(m_star)
        plt.scatter(m[jj1], yvar[jj1], color='r', s=2, marker ='.')
        plt.scatter(m[jj0a], yvar[jj0a], color='b', s=2, marker ='.')
        plt.scatter(m[jj0b], yvar[jj0b], color='k', s=2, marker ='.')
        plt.xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
        plt.ylabel(r'$M_\mathrm{env} \ [M_\odot]')
        plt.savefig('m_env.pdf')

        plt.clf()
#        plt.figure(figsize = myfigsize)
        yvar=-ebn_env/1e51
#        yvar=-ebn_env/(m_star)
        plt.scatter(m[jj1], yvar[jj1], color='r', s=2, marker ='.')
        plt.scatter(m[jj0a], yvar[jj0a], color='b', s=2, marker ='.')
        plt.scatter(m[jj0b], yvar[jj0b], color='k', s=2, marker ='.')
        plt.xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
        plt.ylabel(r'$E_\mathrm{bind,env} \ [10^{51}\mathrm{erg}]$')
        plt.savefig('e_bind_env.pdf')


        plt.clf()
#        plt.figure(figsize = myfigsize)
        yvar=e_expl/1e51#-ebn_env/1e51
        xvar=mc4
        plt.scatter(xvar[jj1], e_expl[jj1]/1e51, color='k', s=2, marker ='.')
#        plt.scatter(xvar[jj1], ebn_env[jj1]/1e51, color='r', s=2, marker ='.')
#        plt.xlabel(r'$M_4$')
        plt.ylabel(r'$E_\mathrm{expl} \ [10^{51}\mathrm{erg}]$')
        plt.savefig('e_vs_mc4.pdf')

#         plt.clf()
# #        plt.figure(figsize = myfigsize)
#         yvar= (m_by-m_init) / physconst.Kepler.solmass
# #        yvar=-ebn_env/(m_star)
#         plt.scatter(m[jj1], yvar[jj1], color='b', s=5, marker ='.')
#         plt.xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
#         plt.ylabel(r'$M_\mathrm{by}-M_\mathrm{init}} \ [M_\odot]')
#         plt.ylim([0,0.5])
#         plt.savefig('m_by_minus_m_init.pdf')


        plt.clf()
#        plt.figure(figsize = myfigsize)
#        f.subplots_adjust(hspace=0.0,top=0.97,bottom=0.1, left=0.15,right=0.97)
#        plt.plot(np.sort(m), mc4[np.argsort(m)], color='k')
#        plt.plot(np.sort(m), fe_core[np.argsort(m)], color='k')
        plt.scatter(m, mc4, color='k', s=7, marker ='.')
#        plt.scatter(m, fe_core, color='k', s=7, marker='.')
        plt.scatter(m[jj1], m_init[jj1] / physconst.Kepler.solmass, color='r', s=2, marker ='.')
        plt.scatter(m[jj0a], m_init[jj0a] / physconst.Kepler.solmass, color='b', s=2, marker ='.')
        plt.scatter(m[jj0b], m_init[jj0b] / physconst.Kepler.solmass, color='g', s=5, marker ='o')
        plt.xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
        plt.ylabel(r'$M_\mathrm{ini} \  [M_\odot]$')
        plt.ylim([0,2.5])
        plt.xlim ([10,32.6])
        plt.minorticks_on()

        plt.savefig('m_init.pdf')

        plt.clf()
        plt.scatter(mu4[jj1]*mc4[jj1]**1, mu4[jj1]*mc4[jj1]**0, color='r', s=2, marker ='.')
        plt.scatter(mu4[jj0a]*mc4[jj0a]**1, mu4[jj0a]*mc4[jj0a]**0, color='b', s=2, marker ='.')
        plt.scatter(mu4[jj0b]*mc4[jj0b]**1, mu4[jj0b]*mc4[jj0b]**0, color='k', s=2, marker ='.')

        k1 = 0.0
        k2 = 0.204

        xvar = np.sort(mu4*mc4)
        plt.plot(xvar,k1*(xvar)+k2,color='b')
        k1 = 0.33
        k2 = 0.09
        plt.plot(xvar,k1*(xvar)+k2,color='k')

        crit = k1 * (mu4*mc4)+k2 - mu4
#        crit = 0.3*(mu4*mc4)+0.107 - mu4
        mnoex=m[jj0]
        mex=m[jj1]
        print ("Ertl false positives: ",(np.where (crit[jj0] > 0.0))[0].size)
        print ("Ertl false positives: ",mnoex[(np.where (crit[jj0] > 0.0))[0]])
        print ("Ertl false negatives: ",(np.where (crit[jj1] < 0.0))[0].size)
        print ("Ertl false negatives: ",mex[(np.where (crit[jj1] < 0.0))[0]])
        print ("Ertl false: ",(np.where (crit[jj0] > 0.0))[0].size + \
                   (np.where (crit[jj1] < 0.0))[0].size)

        plt.xlabel(r'$M_4 \mu_4$')
        plt.ylabel(r'$\mu_4$')
        plt.xlim([0,1])
        plt.ylim([0,0.45])
        plt.tight_layout()
        plt.savefig('ertl.pdf')




    def ertl_test(self,**kwargs):


        m = []
        m_init = []
        m_by = []
        m_star = []
        mc4 = []
        fe_core = []
        mu4 = []
        xi17 = []
        xi25 = []
        ebn_env = []
        e_expl = []
        print ("hic")

        for mass, explosion in self.explosions.items():
             m_init += [explosion['m_init']]
             m_by += [explosion['m_by']]
             m_star += [explosion['m_star']]
             mc4 += [explosion['mc4']]
             fe_core += [explosion['fe_core']]
             mu4 += [explosion['mu4']]
             xi17 += [explosion['xi17']]
             xi25 += [explosion['xi25']]
             ebn_env += [explosion['ebn_env']]
             m += [mass]
             e_expl      += [explosion['e_expl']]          # Explosion energy

        colors = np.empty(len(m))

        e_expl = np.array(e_expl)
        m = np. array (m)
        m_star = np. array (m_star)
        m_init = np. array (m_init)
        m_by = np. array (m_by)
        xi17 = np.array(xi17)
        xi25 = np.array(xi25)
        ebn_env = np.array(ebn_env)
        mu4 = np.array(mu4)
        mc4 = np.array(mc4)
        fe_core = np.array(fe_core)
        jj1 = np.where(e_expl > 1e45)
        jj0 = np.where(e_expl <= 1e45)
        jj0a = np.where(np.logical_and(e_expl <= 1e45, m_init >= 0.01))
        jj0b = np.where(np.logical_and(e_expl <= 1e45, m_init < 0.01))

        k1 = 0.33
        k2 = 0.09

        plt.clf()
        plt.scatter(mu4[jj1]*mc4[jj1]**1, mu4[jj1]*mc4[jj1]**0, color='r', s=2, marker ='.')
        plt.scatter(mu4[jj0a]*mc4[jj0a]**1, mu4[jj0a]*mc4[jj0a]**0, color='b', s=2, marker ='.')
        plt.scatter(mu4[jj0b]*mc4[jj0b]**1, mu4[jj0b]*mc4[jj0b]**0, color='k', s=2, marker ='.')

        xvar = np.sort(mu4*mc4)
        plt.plot(xvar,k1*xvar+k2,color='k')
        crit = k1*mu4*mc4+k2 - mu4
#        crit = 0.3*(mu4*mc4)+0.107 - mu4
        mnoex=m[jj0]
        mex=m[jj1]

        print ("Ertl false positives: ",(np.where (crit[jj0] > 0.0))[0].size)
        print ("Ertl false positives: ",mnoex[(np.where (crit[jj0] > 0.0))[0]])
        print ("Ertl false negatives: ",(np.where (crit[jj1] < 0.0))[0].size)
        print ("Ertl false negatives: ",mex[(np.where (crit[jj1] < 0.0))[0]])

        print ("Ertl false: ",(np.where (crit[jj0] > 0.0))[0].size + \
                   (np.where (crit[jj1] < 0.0))[0].size)


        print ("---- Shock revival only -----")
        print ("Ertl false positives: ",(np.where (crit[jj0b] > 0.0))[0].size)
        print ("Ertl false negatives: ",(np.where (crit[jj1] < 0.0))[0].size+(np.where (crit[jj0a] > 0.0))[0].size)

        print ("Ertl false: ",(np.where (crit[jj0b] > 0.0))[0].size + \
                   (np.where (crit[jj1] < 0.0))[0].size + \
                   (np.where (crit[jj0a] < 0.0))[0].size)

        plt.xlabel(r'$M_4 \mu_4$')
        plt.ylabel(r'$\mu_4$')
        plt.xlim([0,1])
        plt.ylim([0,0.45])
        plt.tight_layout()
        plt.savefig('ertl.pdf')




#-------------------------------------------------------------------------------
# Generate some plots for a range of different values

    def dav_plots(self,**kwargs):
        #1
        self.recalculate(alpha_outflow=0.5,threshold=0.8,beta=6,zeta=0.8)
        self.dav_plot()
#        self.ertl_plot()

        self.recalculate(alpha_outflow=0.5,threshold=0.8,beta=6,zeta=0.8,addwind=True)
        self.dav_plot()


#        self.hist()
        #2
#        self.recalculate(alpha_outflow=0.5,threshold=1,beta=4,zeta=0.7)
#        self.dav_plot(extra=True)
#        self.hist()
        #3
#        self.recalculate(alpha_outflow=0.25,threshold=1,beta=4,zeta=0.7)
#        self.dav_plot()
#        self.hist()
        #4
#        self.recalculate(alpha_outflow=0.5,threshold=0.8,beta=4,zeta=0.7)
#        self.dav_plot()
#        self.hist()
        #5
#        self.recalculate(alpha_outflow=0.5,threshold=1,beta=5,zeta=0.7)
#        self.dav_plot()
#        self.hist()


#-------------------------------------------------------------------------------
# Plot for paper

    def paper_plot(self,extra=False,poznanski=False,mode = 1, **kwargs):

        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex = True,  figsize = (5,12) )
        f.subplots_adjust(hspace=0.0,top=0.97,bottom=0.1, left=0.15,right=0.97)

        betas = [3, 4, 5]
        zetas = [0.9, 0.7, 0.5]
        alpha_turbs = [1.28, 1.18, 1.08]
        alpha_outs = [0.33, 0.5, 0.67]
        t_cools = [0.7, 1.2, 1.6]
        thresholds = [0.6, 0.8, 1.0]

        myplots = []

        if mode <= 0:
            iplots = [1]
        else:
            iplots = [0, 1, 2]

        for iplot in iplots:

            if mode == 0:
                self.recalculate()
            if mode == 1:
                self.recalculate(beta=betas[iplot])
                mylabel = r'$\beta_\mathrm{expl}='+str(betas[iplot])+'$'
            if mode == 2:
                self.recalculate(alpha_turb=alpha_turbs[iplot])
                mylabel = r'$\alpha_\mathrm{turb}='+str(alpha_turbs[iplot])+'$'
            if mode == 3:
                self.recalculate(alpha_outflow=alpha_outs[iplot])
                mylabel = r'$\alpha_\mathrm{out}='+str(alpha_outs[iplot])+'$'
            if mode == 4:
                self.recalculate(zeta=zetas[iplot])
                mylabel = r'$\zeta='+str(zetas[iplot])+'$'
            if mode == 5:
                self.recalculate(threshold=thresholds[iplot])
                mylabel = r'$r_\mathrm{crit}='+str(thresholds[iplot])+'$'
            if mode == 6:
                self.recalculate(t_15=t_cools[iplot])
                mylabel = r'$\tau_\mathrm{1.5}='+str(t_cools[iplot])+'$'
            if mode == -1:
                self.recalculate(**kwargs)
            xr = []
            xe = []
            m  = []
            xn = []

            for mass, explosion in self.explosions.items():
                m  += [mass]
            #            m  += [explosion['m_star'] / physconst.Kepler.solmass]
                xn += [explosion['Ni56']]
                xe += [explosion['e_expl']]
                xr += [explosion['m_grav']]

            xe = np.array(xe) * 1.e-51
            xr = np.array(xr) / physconst.Kepler.solmass
            xn = np.array(xn) / physconst.Kepler.solmass
            m  = np.array(m)

            plotsyms = ['r.','k.','b.']
            if mode <= 0:
                ax1.plot(m, xe, plotsyms [iplot], ms=1)
            else:
                myplot, = ax1.plot(m, xe, plotsyms [iplot], ms=1, label=mylabel)
                myplots.append(myplot)
            ax2.plot(m, xr, plotsyms [iplot], ms=1)
            ax3.plot(m, xr, plotsyms [iplot], ms=1)
            ax4.plot(m, xn, plotsyms [iplot], ms=1)

        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax2.set_xscale('linear')
        ax2.set_yscale('linear')
        ax3.set_xscale('linear')
        ax3.set_yscale('linear')
        ax4.set_xscale('linear')
        ax4.set_yscale('linear')


#        ax1.set_xlim([10,32.54])
#        ax2.set_xlim([10,32.54])
#        ax3.set_xlim([10,32.54])
        ax4.set_xlim([10,32.54])

        if mode <= 0:
            ax1.set_ylim([0, 2.4])
        else:
            ax1.set_ylim([0, 3])
        ax3.set_ylim([1.05, 2.05])
        ax2.set_ylim([8, 18])
        if mode <= 0:
            ax4.set_ylim([0, 0.17])
        else:
            ax4.set_ylim([0, 0.38])

        ax1.set_ylabel(r'$E_\mathrm{expl} \  [10^{51} \ \mathrm{erg}]$')
        ax2.set_ylabel(r'$M_\mathrm{BH} \  [M_\odot]$')
        ax3.set_ylabel(r'$M_\mathrm{NS} \  [M_\odot]$')
        ax4.set_ylabel(r'$M_\mathrm{IG} \ [M_\odot]$')

#        ax1.set_xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
#        ax2.set_xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
        ax4.set_xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')


        ax1.set_yticks (np.arange (0, 3, 0.5))
        ax1.set_yticks (np.arange (0, 3, 0.1), minor = True)
        ax2.set_yticks (np.arange (10, 18, 2))
        ax2.set_yticks (np.arange (10, 18, 1), minor = True)
        ax3.set_yticks (np.arange (1.2, 2.1, 0.2))
        ax3.set_yticks (np.arange (1.2, 2.1, 0.1), minor = True)
        if (mode <=0):
            ax4.set_yticks (np.arange (0, 0.2, 0.05))
            ax4.set_yticks (np.arange (0, 0.2, 0.01), minor = True)
        else:
            ax4.set_yticks (np.arange (0, 0.4, 0.1))
            ax4.set_yticks (np.arange (0, 0.4, 0.05), minor = True)

        ax4.set_xticks (np.arange (10, 32.5, 5))
        ax4.set_xticks (np.arange (10, 32.5, 1), minor = True)

        ax1.text(0.05, 0.92, r'a)', transform=ax1.transAxes,
                fontsize=14, va='top')
        ax2.text(0.05, 0.92, r'b)', transform=ax2.transAxes,
                fontsize=14, va='top')
        ax3.text(0.05, 0.92, r'c)', transform=ax3.transAxes,
                fontsize=14, va='top')
        ax4.text(0.05, 0.92, r'd)', transform=ax4.transAxes,
                fontsize=14, va='top')

        if mode > 0:
            ax1.legend(handles = myplots, markerscale = 5, numpoints = 1, fontsize = 9)

        if mode == -1:
            f.savefig('non_standard.pdf')
        if mode == 0:
            f.savefig('standard.pdf')
        if mode == 1:
            f.savefig('beta.pdf')
        if mode == 2:
            f.savefig('alpha_turb.pdf')
        if mode == 3:
            f.savefig('alpha_out.pdf')
        if mode == 4:
            f.savefig('zeta.pdf')
        if mode == 5:
            f.savefig('threshold.pdf')
        if mode == 6 :
            f.savefig('t_cool.pdf')



#-------------------------------------------------------------------------------
# Plot explosion properties vs. He core mass instead of ZAMS mass

    def he_plot(self,extra=False,poznanski=False,mode = 1, **kwargs):

        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex = True,  figsize = (5,12) )
        f.subplots_adjust(hspace=0.0,top=0.97,bottom=0.1, left=0.15,right=0.97)

        betas = [3, 4, 5]
        zetas = [0.9, 0.7, 0.5]
        alpha_turbs = [1.28, 1.18, 1.08]
        alpha_outs = [0.33, 0.5, 0.67]
        t_cools = [0.7, 1.2, 1.6]
        thresholds = [0.6, 0.8, 1.0]

        myplots = []

        if mode <= 0:
            iplots = [1]
        else:
            iplots = [0, 1, 2]

        for iplot in iplots:

            if mode == 0:
                self.recalculate()
            if mode == 1:
                self.recalculate(beta=betas[iplot])
                mylabel = r'$\beta_\mathrm{expl}='+str(betas[iplot])+'$'
            if mode == 2:
                self.recalculate(alpha_turb=alpha_turbs[iplot])
                mylabel = r'$\alpha_\mathrm{turb}='+str(alpha_turbs[iplot])+'$'
            if mode == 3:
                self.recalculate(alpha_outflow=alpha_outs[iplot])
                mylabel = r'$\alpha_\mathrm{out}='+str(alpha_outs[iplot])+'$'
            if mode == 4:
                self.recalculate(zeta=zetas[iplot])
                mylabel = r'$\zeta='+str(zetas[iplot])+'$'
            if mode == 5:
                self.recalculate(threshold=thresholds[iplot])
                mylabel = r'$r_\mathrm{crit}='+str(thresholds[iplot])+'$'
            if mode == 6:
                self.recalculate(t_15=t_cools[iplot])
                mylabel = r'$\tau_\mathrm{1.5}='+str(t_cools[iplot])+'$'
            if mode == -1:
                self.recalculate(**kwargs)
            xr = []
            xe = []
            m  = []
            xn = []

            for mass, explosion in self.explosions.items():
                m  += [explosion['he_core']]
            #            m  += [explosion['m_star'] / physconst.Kepler.solmass]
                xn += [explosion['Ni56']]
                xe += [explosion['e_expl']]
                xr += [explosion['m_grav']]

            xe = np.array(xe) * 1.e-51
            xr = np.array(xr) / physconst.Kepler.solmass
            xn = np.array(xn) / physconst.Kepler.solmass
            m  = np.array(m)

            plotsyms = ['r.','k.','b.']
            if mode <= 0:
                ax1.plot(m, xe, plotsyms [iplot], ms=1)
            else:
                myplot, = ax1.plot(m, xe, plotsyms [iplot], ms=1, label=mylabel)
                myplots.append(myplot)
            ax2.plot(m, xr, plotsyms [iplot], ms=1)
            ax3.plot(m, xr, plotsyms [iplot], ms=1)
            ax4.plot(m, xn, plotsyms [iplot], ms=1)

        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax2.set_xscale('linear')
        ax2.set_yscale('linear')
        ax3.set_xscale('linear')
        ax3.set_yscale('linear')
        ax4.set_xscale('linear')
        ax4.set_yscale('linear')


#        ax1.set_xlim([10,32.54])
#        ax2.set_xlim([10,32.54])
#        ax3.set_xlim([10,32.54])
        ax4.set_xlim([2,12])

        if mode <= 0:
            ax1.set_ylim([0, 2.4])
        else:
            ax1.set_ylim([0, 3])
        ax3.set_ylim([1.05, 2.05])
        ax2.set_ylim([8, 18])
        if mode <= 0:
            ax4.set_ylim([0, 0.17])
        else:
            ax4.set_ylim([0, 0.38])

        ax1.set_ylabel(r'$E_\mathrm{expl} \  [10^{51} \ \mathrm{erg}]$')
        ax2.set_ylabel(r'$M_\mathrm{BH} \  [M_\odot]$')
        ax3.set_ylabel(r'$M_\mathrm{NS} \  [M_\odot]$')
        ax4.set_ylabel(r'$M_\mathrm{IG} \ [M_\odot]$')

#        ax1.set_xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
#        ax2.set_xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
        ax4.set_xlabel(r'$M_\mathrm{He} \  [M_\odot]$')


        ax1.set_yticks (np.arange (0, 3, 0.5))
        ax1.set_yticks (np.arange (0, 3, 0.1), minor = True)
        ax2.set_yticks (np.arange (10, 18, 2))
        ax2.set_yticks (np.arange (10, 18, 1), minor = True)
        ax3.set_yticks (np.arange (1.2, 2.1, 0.2))
        ax3.set_yticks (np.arange (1.2, 2.1, 0.1), minor = True)
        if (mode <=0):
            ax4.set_yticks (np.arange (0, 0.2, 0.05))
            ax4.set_yticks (np.arange (0, 0.2, 0.01), minor = True)
        else:
            ax4.set_yticks (np.arange (0, 0.4, 0.1))
            ax4.set_yticks (np.arange (0, 0.4, 0.05), minor = True)

        ax4.set_xticks (np.arange (2, 12, 1))
        ax4.set_xticks (np.arange (2, 12, 0.5), minor = True)

        ax1.text(0.05, 0.92, r'a)', transform=ax1.transAxes,
                fontsize=14, va='top')
        ax2.text(0.05, 0.92, r'b)', transform=ax2.transAxes,
                fontsize=14, va='top')
        ax3.text(0.05, 0.92, r'c)', transform=ax3.transAxes,
                fontsize=14, va='top')
        ax4.text(0.05, 0.92, r'd)', transform=ax4.transAxes,
                fontsize=14, va='top')

        if mode > 0:
            ax1.legend(handles = myplots, markerscale = 5, numpoints = 1, fontsize = 9)

        if mode == -1:
            f.savefig('non_standard_he.pdf')
        if mode == 0:
            f.savefig('standard_he.pdf')
        if mode == 1:
            f.savefig('beta_he.pdf')
        if mode == 2:
            f.savefig('alpha_turb_he.pdf')
        if mode == 3:
            f.savefig('alpha_out_he.pdf')
        if mode == 4:
            f.savefig('zeta_he.pdf')
        if mode == 5:
            f.savefig('threshold_he.pdf')
        if mode == 6 :
            f.savefig('t_cool_he.pdf')


#-------------------------------------------------------------------------------
# Plot explosion properties vs. C/O core mass instead of ZAMS mass

    def co_plot(self,extra=False,poznanski=False,mode = 1, **kwargs):

        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex = True,  figsize = (5,12) )
        f.subplots_adjust(hspace=0.0,top=0.97,bottom=0.1, left=0.15,right=0.97)

        betas = [3, 4, 5]
        zetas = [0.9, 0.7, 0.5]
        alpha_turbs = [1.28, 1.18, 1.08]
        alpha_outs = [0.33, 0.5, 0.67]
        t_cools = [0.7, 1.2, 1.6]
        thresholds = [0.6, 0.8, 1.0]

        myplots = []

        if mode <= 0:
            iplots = [1]
        else:
            iplots = [0, 1, 2]

        for iplot in iplots:

            if mode == 0:
                self.recalculate()
            if mode == 1:
                self.recalculate(beta=betas[iplot])
                mylabel = r'$\beta_\mathrm{expl}='+str(betas[iplot])+'$'
            if mode == 2:
                self.recalculate(alpha_turb=alpha_turbs[iplot])
                mylabel = r'$\alpha_\mathrm{turb}='+str(alpha_turbs[iplot])+'$'
            if mode == 3:
                self.recalculate(alpha_outflow=alpha_outs[iplot])
                mylabel = r'$\alpha_\mathrm{out}='+str(alpha_outs[iplot])+'$'
            if mode == 4:
                self.recalculate(zeta=zetas[iplot])
                mylabel = r'$\zeta='+str(zetas[iplot])+'$'
            if mode == 5:
                self.recalculate(threshold=thresholds[iplot])
                mylabel = r'$r_\mathrm{crit}='+str(thresholds[iplot])+'$'
            if mode == 6:
                self.recalculate(t_15=t_cools[iplot])
                mylabel = r'$\tau_\mathrm{1.5}='+str(t_cools[iplot])+'$'
            if mode == -1:
                self.recalculate(**kwargs)
            xr = []
            xe = []
            m  = []
            xn = []

            for mass, explosion in self.explosions.items():
                m  += [explosion['CO_core']]
            #            m  += [explosion['m_star'] / physconst.Kepler.solmass]
                xn += [explosion['Ni56']]
                xe += [explosion['e_expl']]
                xr += [explosion['m_grav']]

            xe = np.array(xe) * 1.e-51
            xr = np.array(xr) / physconst.Kepler.solmass
            xn = np.array(xn) / physconst.Kepler.solmass
            m  = np.array(m)

            plotsyms = ['r.','k.','b.']
            if mode <= 0:
                ax1.plot(m, xe, plotsyms [iplot], ms=1)
            else:
                myplot, = ax1.plot(m, xe, plotsyms [iplot], ms=1, label=mylabel)
                myplots.append(myplot)
            ax2.plot(m, xr, plotsyms [iplot], ms=1)
            ax3.plot(m, xr, plotsyms [iplot], ms=1)
            ax4.plot(m, xn, plotsyms [iplot], ms=1)

        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax2.set_xscale('linear')
        ax2.set_yscale('linear')
        ax3.set_xscale('linear')
        ax3.set_yscale('linear')
        ax4.set_xscale('linear')
        ax4.set_yscale('linear')


#        ax1.set_xlim([10,32.54])
#        ax2.set_xlim([10,32.54])
#        ax3.set_xlim([10,32.54])
        ax4.set_xlim([1,10])

        if mode <= 0:
            ax1.set_ylim([0, 2.4])
        else:
            ax1.set_ylim([0, 3])
        ax3.set_ylim([1.05, 2.05])
        ax2.set_ylim([8, 18])
        if mode <= 0:
            ax4.set_ylim([0, 0.17])
        else:
            ax4.set_ylim([0, 0.38])

        ax1.set_ylabel(r'$E_\mathrm{expl} \  [10^{51} \ \mathrm{erg}]$')
        ax2.set_ylabel(r'$M_\mathrm{BH} \  [M_\odot]$')
        ax3.set_ylabel(r'$M_\mathrm{NS} \  [M_\odot]$')
        ax4.set_ylabel(r'$M_\mathrm{IG} \ [M_\odot]$')

#        ax1.set_xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
#        ax2.set_xlabel(r'$M_\mathrm{ZAMS} \  [M_\odot]$')
        ax4.set_xlabel(r'$M_\mathrm{C/O} \  [M_\odot]$')


        ax1.set_yticks (np.arange (0, 3, 0.5))
        ax1.set_yticks (np.arange (0, 3, 0.1), minor = True)
        ax2.set_yticks (np.arange (10, 18, 2))
        ax2.set_yticks (np.arange (10, 18, 1), minor = True)
        ax3.set_yticks (np.arange (1.2, 2.1, 0.2))
        ax3.set_yticks (np.arange (1.2, 2.1, 0.1), minor = True)
        if (mode <=0):
            ax4.set_yticks (np.arange (0, 0.2, 0.05))
            ax4.set_yticks (np.arange (0, 0.2, 0.01), minor = True)
        else:
            ax4.set_yticks (np.arange (0, 0.4, 0.1))
            ax4.set_yticks (np.arange (0, 0.4, 0.05), minor = True)

        ax4.set_xticks (np.arange (2, 12, 1))
        ax4.set_xticks (np.arange (2, 12, 0.5), minor = True)

        ax1.text(0.05, 0.92, r'a)', transform=ax1.transAxes,
                fontsize=14, va='top')
        ax2.text(0.05, 0.92, r'b)', transform=ax2.transAxes,
                fontsize=14, va='top')
        ax3.text(0.05, 0.92, r'c)', transform=ax3.transAxes,
                fontsize=14, va='top')
        ax4.text(0.05, 0.92, r'd)', transform=ax4.transAxes,
                fontsize=14, va='top')

        if mode > 0:
            ax1.legend(handles = myplots, markerscale = 5, numpoints = 1, fontsize = 9)

        if mode == -1:
            f.savefig('non_standard_co.pdf')
        if mode == 0:
            f.savefig('standard_co.pdf')
        if mode == 1:
            f.savefig('beta_co.pdf')
        if mode == 2:
            f.savefig('alpha_turb_co.pdf')
        if mode == 3:
            f.savefig('alpha_out_co.pdf')
        if mode == 4:
            f.savefig('zeta_co.pdf')
        if mode == 5:
            f.savefig('threshold_co.pdf')
        if mode == 6 :
            f.savefig('t_cool_co.pdf')



#-------------------------------------------------------------------------------
# E_expl vs. M_IG

    def e_vs_mig(self,**kwargs):

        xe = []
        xn = []
        m_init = []

        for mass, explosion in self.explosions.items():
            xn += [explosion['Ni56']]
            xe += [explosion['e_expl']]
            m_init += [explosion['m_init']]

        e_expl = np.array(xe)
        xe = np.log10 (np.maximum (np.array(xe), 1e-5))
        xn = np.log10 (np.maximum (np.array(xn) / physconst.Kepler.solmass, 1e-5))
        m_init = np.array (m_init)

        f, (ax1) = plt.subplots(1, 1, figsize = (4,4))
        f.subplots_adjust (left=0.18, right=0.97, bottom=0.12, top = 0.95)

        ax1.plot(xe, xn, 'k.', ms=2)


        ax1.set_xlabel(r'$\log_{10} (E_\mathrm{expl} /  \mathrm{erg})$')
        ax1.set_ylabel(r'$\log_{10} (M_\mathrm{IG} / M_\odot)$')
        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax1.set_xlim([-1.0+51, 0.3+51])
        ax1.set_ylim([-2.0, -0.4])

#        ax1.plot(xe, 1.49 * xe - 2.90, 'r')
        ax1.plot(xe, 1.13 * (xe-50) - 2.45 ,'b')
        ax1.plot(xe, 1.13 * (xe-50) - 2.45 + np.log10(2) ,'b')


        f.savefig('e_vs_mig.pdf')


#-------------------------------------------------------------------------------
# E_expl vs. M_ej

    def e_vs_mej(self,extra=False,poznanski=False,**kwargs):

        xe = []
        xm = []

        for mass, explosion in self.explosions.items():
            xm += [explosion['m_ej']]
            xe += [explosion['e_expl']]

        xe = np.log10 (np.maximum (np.array(xe), 1e-5))
        xm = np.log10 (np.maximum (np.array(xm) / physconst.Kepler.solmass, 1e-5))


        f, (ax1) = plt.subplots(1, 1, figsize = (4,4))
        f.subplots_adjust (left=0.18, right=0.97, bottom=0.12, top = 0.97)


        ax1.plot(xm, xe, 'k.', ms=2)


        ax1.set_ylabel(r'$\log_{10} (E_\mathrm{expl} /  \mathrm{erg})$')
        ax1.set_xlabel(r'$\log_{10} (M_\mathrm{ej} / M_\odot)$')
        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax1.set_ylim([-1.0+51, 0.3+51])
        ax1.set_xlim([0.8, 1.2])

        ax1.plot(xm, 2.09 * xm - 1.78 + 50, 'r')
        ax1.plot(xm, 1.81 * xm - 1.12 + 50, 'b')

        f.savefig('e_vs_mej.pdf')



#-------------------------------------------------------------------------------
# E_expl vs. M_ej

    def e_vs_m_grav(self,extra=False,poznanski=False,**kwargs):

        xe = []
        xm = []

        for mass, explosion in self.explosions.items():
            xm += [explosion['m_grav']]
            xe += [explosion['e_expl']]

        xe = np.array(xe) * 1.e-51
        xm = np.array(xm) / physconst.Kepler.solmass

        f, (ax1) = plt.subplots(1, 1, figsize = (4,4))
        f.subplots_adjust (left=0.18, right=0.97, bottom=0.12, top = 0.96)


        ax1.plot(xm, xe, 'k.',ms=2)


        ax1.set_ylabel(r'$E_\mathrm{expl} \ [10^{51} \ \mathrm{erg}]$')
        ax1.set_xlabel(r'$M_\mathrm{NS} \ [M_\odot]$')
        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax1.set_ylim([0, 2])
        ax1.set_xlim([1.1, 2.05])

        f.savefig('e_vs_mgrav.pdf')
