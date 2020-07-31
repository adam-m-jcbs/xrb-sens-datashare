"""
Test routine for C12+C12 --> Mg23+n rate
"""

import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import strdata

def cc(t):
    t9 = t * 1.e-9
    t92 = t9**2
    t93 = t9**3
    t9m32 = t9**(-3/2)
    t9m3 = t9**(-3)

    rho = 1
    sc1212 = 1

    t9a = t9/(1 + 0.0396 * t9)
    t9a13 = t9a**(1/3)
    t9a56 = t9a**(5/6)

    r24 = 4.27e+26 * t9a56 * t9m32 * np.exp(-84.165 / t9a13 - 2.12e-3 * t93)
    r24 = 0.5 * rho * r24 * sc1212

    #     r24*pn(1)**2 is formation rate of mg24 cmpd. nucleus
    #...   branching ratios from fcz 1975
    #...     neutron branching from dayras switkowski and woosley
    #...     1976
    #
    #....
    #.... repair typographical error in dayras et al neutron branching
    #.... ratio. (t9**2 not t9**3)
    #....

    b24n = 0.055 * (1 - np.exp(-(0.789 * t9 - 0.976)))
    ii, = np.where(t9 <= 1.5)
    b24n[ii] = 0.859 * np.exp(-((0.766 * t9m3[ii]) *
                            (1 + 0.0789 * t9[ii] + 7.74 * t92[ii])))

    b24p = (1 - b24n) / 3
    b24a = 2 * b24p
    ii, = np.where(t9 <= 3)
    b24p[ii] = (1 - b24n[ii]) * 0.5
    b24a[ii] = b24p[ii]

    return r24 * b24n, r24 * b24p, r24 * b24a


#ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
# This is a rate based on recent ND measurement in the range of 3.1 - 6.3 MeV
# The lower energy part is estimated with Zickefoose's p0+p1 data * 23Mg/(p0+p1).
# The ratio, 23Mg/(p0+p1), is caluclated with TALYS.
# The fitting error is less than 2% when 0.5<T9<3 and increase to 4% when 3<T9<4.9.
# For more information, contact X. Tang/B. Bucher at Univ. of Notre Dame
# Updated at Aug. 2, 2013
#ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
def cn_rate(t):
    t9 = t * 1.e-9
    a1 = -3.569048E+02
    a2 = -4.664213E+01
    a3 =  1.638522E+02
    a4 =  2.339795E+02
    a5 = -2.708341E+01
    a6 =  2.028249E+00
    a7 = -1.207916E+01
    t913 = t9**(1 / 3)
    t953 = t9**(5 / 3)
    rate = (a1 + a2 / t9 + a3 / t913 + a4 * t913 + a5 * t9
                + a6 * t953 + a7 * np.log(t9))
    return np.exp(rate)

#ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
# Upper limit updated at Aug. 2, 2013
# The fitting was done in the range of 0.5<T9<5. The fitting error is less
# than 2% when T9<3 and increase to 8% when 3<t9<5
#ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

def cn_rate_up(t):
    t9 = t * 1.e-9

    a1 = -3.544816E+02
    a2 = -4.788872E+01
    a3 =  1.598118E+02
    a4 =  2.350152E+02
    a5 = -2.435667E+01
    a6 =  1.602213E+00
    a7 = -1.758802E+01
    t913 = t9**(1 / 3)
    t953 = t9**(5 / 3)
    rate = (a1 + a2 / t9 + a3 / t913 + a4 * t913 + a5 * t9
            + a6 * t953 + a7 * np.log(t9))
    return np.exp(rate)

#ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
# Lower limit updated at Aug. 2, 2013
# The fitting was done in the range of 0.5<T9<5. The fitting error is less
# than 2% when 0.5<T9<4.9.
#ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


def cn_rate_dn(t):
    t9 = t * 1.e-9

    a1 = -3.578855E+02
    a2 = -4.516047E+01
    a3 =  1.656794E+02
    a4 =  2.342141E+02
    a5 = -3.086552E+01
    a6 =  2.634970E+00
    a7 = -6.403464E+00

    t913 = t9**(1 / 3)
    t953 = t9**(5 / 3)
    rate = (a1 + a2 / t9 + a3 / t913 + a4 * t913 + a5 * t9
                + a6 * t953 + a7 * np.log(t9))
    return np.exp(rate)

def plot():
    t = 10**(np.arange(8,10.0001,.01))
    n,p,a = cc(t)
    c = (a+p+n)

    f = plt.figure()
    ax = f.add_subplot(111)

    ax.set_xscale('log')
    ax.set_xlabel('T/K')
    ax.set_ylabel('BR')

    ax.plot(t,n/c,label='CCN')
    ax.plot(t,p/c,label='CCP')
    ax.plot(t,a/c,label='CCA')

    c0 = cn_rate(t)
    cu = cn_rate_up(t)
    cd = cn_rate_dn(t)

    ax.plot(t,c0/c,'--',label='CCN0')
    ax.plot(t,cu/c,'--',label='CCNu')
    ax.plot(t,cd/c,'--',label='CCNd')

    ax.set_ylim(0,1)

    ax.legend(loc=3)


def brccn_dayras(t):

    n,p,a = cc(t)
    c = (a+p+n)

    return n/c


#ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#c This branching ratio is based on the PRELIMINARY rate
#c provided by B. Bucher (Aug. 2013) in the range of
#c 0.5 to 2.5 GK and the standard CF88 C+C fusion rate.
#c The old Dayras ratio is used in the range of
#c 2.5 to 5 GK after scaling up by a small factor (5.72/5.62).
#c For higher temperature, a constant ratio of 5.362%
#c is used.
#ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

def brccn_nd(t):

    t9 = t * 1.e-9

    onethird = 1/3

    t139=t9**onethird
    t239=t139*t139
    t9m1=1.0/t9
    t9m23=1.0/t239
    t129=np.sqrt(t9)
    t569=t139*t129
    rt9=11.60485*t9m1
    t932=t9*t129
    t913=t139
    t923=t239
    t943=t9*t913
    t953=t9*t923
    t9m32=1/t932
    t9m13=1/t913
    t92=t9*t9
    t93=t92*t9
    t9m3=1/t93

    b24n = np.zeros_like(t9)

    ii, = np.where(t9 < 1.5)
    b24n[ii] = \
             +1.1954e-01 * np.exp(-((
             +1.6446e-01 / t93[ii]) * (1.
             +1.5657e+01 * t9[ii]
             +1.1805e+01 * t92[ii])))
    ii, = np.where(np.logical_and(1.5 <= t9, t9 < 2.5))
    b24n[ii] = \
             +2.2120e-01 *(1. - np.exp(-(
             +1.3597e-01 * t9[ii]
             -1.5800e-01 )))
    ii, = np.where(np.logical_and(2.5 <= t9, t9 <= 5.))
    b24n[ii] = \
             +4.8811e-02 * (1. - np.exp(-(
             +2.1124e-00 * t9[ii]
             -3.8791e-00 )))
    ii, = np.where(t9 > 5.)
    b24n[ii] = \
             +4.8750e-02

    # c calculate upper and lower limit

    b24n_up = b24n * (
             +1.17261e-00)
    b24n_dn = b24n * (
             +8.34390e-01)

    ii, = np.where(t9 < 2.5)
    b24n_up[ii] = b24n[ii]*(\
             -0.0969e-00 * t93[ii]*t9[ii]
             +0.5957e-00 * t93[ii]
             -1.1456e-00 * t92[ii]
             +0.4849e-00 * t9[ii]
             +1.5977e-00 )
    b24n_dn[ii] = b24n[ii]*(\
             +0.1074e-00 * t93[ii]*t9[ii]
             -0.6440e-00 * t93[ii]
             +1.1915e-00 * t92[ii]
             -0.4478e-00 * t9[ii]
             +0.3742e-00 )
    return (b24n, b24n_up, b24n_dn)



def plot2():
    t = 10**(np.arange(8,10.0001,.01))
    n,u,d = brccn_nd(t)

    n_dayras = brccn_dayras(t)

    f = plt.figure()
    ax = f.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('T/K')
    ax.set_ylabel('BR')

    ax.plot(t,n,label='N')
    ax.plot(t,u,label='N\_up')
    ax.plot(t,d,label='N\_dn')

    ax.plot(t,n_dayras,'k',label='N\_Dayras')

    ax.set_ylim(1.e-5,1)

    ax.legend(loc=3)

# kepgen.MakeRun(composition='sollo09',mass=18,lane_rhoc=.001,dirbase='/home/alex/kepler/CCNrate',burn=True)
# kepgen.MakeRun(composition='zero',mass=18,lane_rhoc=.001,dirbase='/home/alex/kepler/CCNrate',burn=True)
# kepgen.MakeRun(composition='hez',mass=100,lane_rhoc=.01,dirbase='/home/alex/kepler/CCNrate',burn=True,bgdir='',special={'he-psn'})

# kepgen.TestExp(composition='sollo09',mass=18,dirbase='/home/alex/kepler/CCNrate',burn=True,exp='D',run=True)
# kepgen.TestExp(composition='zero',mass=18,dirbase='/home/alex/kepler/CCNrate',burn=True,exp='D',run=True)

# kepgen.BurnExp(composition='sollo09',mass=18,dirbase='/home/alex/kepler/CCNrate',burn=True,exp='D',run=True)
# kepgen.BurnExp(composition='zero',mass=18,dirbase='/home/alex/kepler/CCNrate',burn=True,exp='D',run=True)

def slice_data():
    dir = '/m/web/Download/xiaodong/slices/'
    source_dir = '~/kepler/CCNrate/'
    model = 'he100'
    masses = np.arange(10)*2.e34
    runs = ['','z','d','n','u']
    for r in runs:
        s = strdata.load(source_dir+model+r)
        time = s.time
        dt = s.dt
        for m in masses:
            dn = s.dn_zm(m)
            tn = s.tn_zm(m)
            filename = os.path.join(dir,"{:s}{:s}_trajectory_{:03d}.txt".format(model,r,int(m/2.e33)))
            with TextFile(filename, mode = 'w', compress=True) as f:
                f.write("{:>25s} {:>25s} {:>25s} {:>25s}\n".format('time','dt','density','temperature'))
                f.write("{:>25s} {:>25s} {:>25s} {:>25s}\n".format('(sec)','(sec)','(g/cm**3)','(K)'))
                for i in range(len(time)):
                    f.write("{:25.17e} {:25.17e} {:25.17e} {:25.17e}\n".format(time[i],dt[i],dn[i],tn[i]))

from kepdata import kepdata
import itertools

def slice_dump_data():
    dir = '/m/web/Download/xiaodong/slices/'
    source_dir = '~/kepler/CCNrate/'
    runs = ['','z','d','n','u']
    models = ['he100']
    dumps = ['cign','cdep','odep']
    for m,r,d in itertools.product(models,runs,dumps):
        kepdata(filename=os.path.join(source_dir,m+r,m+'#'+d),
                outfile=os.path.join(dir,m+r+'@@'+d),
                compress='gz',
                burn=True)

def dump_final_data():
    dir = '/m/web/Download/xiaodong/'
    source_dir = '~/kepler/CCNrate/'
    runs = ['','z','d','n','u']
    models = ['he100']
    dumps = ['final']
    types = ['burn','ely','deciso']
    for m,r,t,d in itertools.product(models,runs,types,dumps):
        kw = {t : True}
        kepdata(filename=os.path.join(source_dir,m+r,m+'#'+d),
                outfile=os.path.join(dir,t,m+r+'@@'+d),
                compress='gz',
                **kw)
