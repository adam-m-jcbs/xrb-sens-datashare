"""
Collection of subroutines for gridb project
"""

import physconst
import os
import os.path
import math

import numpy as np

import kepdump

def get_J_M_exp(series = 'z',
              dump = 'hburn',
              rotation = '0100',
              base = '/u/alex/kepler/gridb',
              masses = (12,80)):
    """
    compute J index for const vvk
    """
    mass_str = [str(mass) for mass in masses]
    dumpfiles = [os.path.join(base,series+'gridb'+rotation,series+mass,series+mass+'#'+dump) for mass in mass_str]
    dumps = [kepdump.loaddump(dumpfile) for dumpfile in dumpfiles]
    for i,dump in enumerate(dumps):
        j = int(dump.j_conv_core)
        print(series+mass_str[i],': H1(1)=',dump.net.abu('h1')[1],' H(',j,')=',dump.net.abu('h1')[j])
    j0 = dumps[0].jm - 4
    j1 = dumps[1].jm - 4

    print('vvk(',series+mass_str[0],')=',dumps[0].angwwkn[j0],', vvk(',series+mass_str[1],')=',dumps[1].angwwkn[j1])
    print('J/J_ini(',series+mass_str[0],')=',dumps[0].qparm.anglt/dumps[0].qparm.anglint,', J/J_ini(',series+mass_str[1],')=',dumps[1].qparm.anglt/dumps[1].qparm.anglint)

    print('d ln (J_ini) / d ln (M) = ', math.log((dumps[0].qparm.anglint)/
                                            (dumps[1].qparm.anglint))/(
            math.log(masses[0]/masses[1])))
    print('d ln (J_ini/wwk) / d ln (M) = ', math.log((dumps[1].angwwkn[j1]*dumps[0].qparm.anglint)/
                                            (dumps[0].angwwkn[j0]*dumps[1].qparm.anglint))/(
            math.log(masses[0]/masses[1])))
    print('d ln (J/wwk) / d ln (M) = ', math.log((dumps[1].angwwkn[j1]*dumps[0].qparm.anglt)/
                                            (dumps[0].angwwkn[j0]*dumps[1].qparm.anglt))/(
            math.log(masses[0]/masses[1])))
    print('d ln (wwk) / d ln (M) = ', math.log((dumps[0].angwwkn[j0])/
                                            (dumps[1].angwwkn[j1]))/(
            math.log(masses[0]/masses[1])))
    print('d ln (I*wk) / d ln (M) = ', math.log((dumps[0].angwk[j0]*dumps[0].qparm.angit)/
                                            (dumps[1].angwk[j1]*dumps[1].qparm.angit))/(
            math.log(masses[0]/masses[1])))

def get_J_Z_exp(series = ('z','u','v','t','o','s'),
              dump = 'hburn',
              rotation = '0100',
              base = '/u/alex/kepler/gridb',
              mass = 20):
    """
    compute J index for const vvk
    """
    mass_str = str(mass)
    dumpfiles = [os.path.join(base,s+'gridb'+rotation,s+mass_str,s+mass_str+'#'+dump) for s in series]
    dumps = [kepdump.loaddump(dumpfile) for dumpfile in dumpfiles]
    for s,dump in zip(series, dumps):
        print(''.ljust(32,'='))
        print(s + mass_str,':')
        j = dump.jm - 4
        print('vvk = ',dump.angwwkn[j])
        print('J/J_ini = ',dump.qparm.anglt/dump.qparm.anglint)
        print('J_ini = ', dump.qparm.anglint)
        print('wwk = ', dump.angwwkn[j])
        print('J/wwk = ', dump.qparm.anglt/dump.angwwkn[j])
        print('wk = ', dump.angwk[j])
        print('I = ', dump.qparm.angit)
        print('I*wk = ', dump.angwk[j]*dump.qparm.angit)

    print(''.ljust(50,'='))
    for s,dump in zip(series, dumps):
        j = dump.jm - 4
        print('{:3s}: Z = {:12.5e}, I*wk/(J/J_ini) = {:12.5e}'.format(
              s + mass_str, dump.qparm.zinit,
              dump.angwk[j]*dump.qparm.angit/(
                dump.qparm.anglt/dump.qparm.anglint)))

    Z = [dump.qparm.zinit for dump in dumps]
    Z = np.array(Z[1:])
    J = [dump.angwk[-6]*dump.qparm.angit/(
                dump.qparm.anglt/dump.qparm.anglint)
         for dump in dumps]
    J = np.array(J[1:])
    opt_J_Z(Z,J)

import scipy.optimize

def opt_J_Z(Z = None, J = None):
    Z = np.log10(Z)
    J = J *1.e-53
    def fit(z,x):
        return x[0]*z**2 + x[1]*z + x[2]
    def residual(x):
        return sum((fit(Z,x) - J)**2)
    x0 = np.array([0,0,J[-2]])

    result = scipy.optimize.fmin(residual,x0,xtol=1.e-8,ftol=1.e-8)

    print('a logZ**2 + b logZ + c; a,b,c = ',result)

    for z,j in zip(Z,J):
        print("Z = {:12.5e}, J53_mod = {:12.5e}, J53_fit = {:12.5e}".format(
            10.**z,j,fit(z,result)))

import glob

def get_Teff_presn():
    files = glob.glob('/c/alex/kepler/gridb/?gridb????/???/*presn')
    for f in files:
        d = kepdump.loaddump(f,silent=True)
        print("{:s}: Teff = {:6.0f} K".format(
                f, float(d.qparm.teff)))
