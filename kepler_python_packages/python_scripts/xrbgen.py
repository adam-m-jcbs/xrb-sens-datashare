#! /bin/env python3

"""
Module to generate 'generatort' files for KEPLER.
"""

import os
import os.path
import shutil

from kepgen import KepEnv

# define class xrbenv(KepEnv)

def xrbgen(run = 0,
           accrate = 1.,
           Z = 'solar',
           models = None,
           series = 'b',
           subset = '',
           accmass = 0):

    ke = KepEnv(silent = True)

    dir00 = ke.dir00
    xrbbase = 'XRB'
    dir0 = os.path.join(dir00, 'XRB')
    progfile = ke.progfile

    srun = '{:d}'.format(run)
    # we now drop 'xrb' from the run name
    run = series + srun + subset

    rundir = os.path.join(dir0, run)
    datadir = ke.datadir

    try:
        os.makedirs(rundir)
    except:
        raise Exception('directory already exists')

    p211 = 1.75e-8 * accrate
    p62 = 1.6e35 * accrate

    adapnet = 'adapnet.cfg'

    # change
    bdat = 'rath00_9.1.1.bdat'
    kepler = 'keplery'

    models0 = 20000

    if Z == 'solar':
        genburn = 'rpa1bg'
        abun = ['0.7048 h1 0.2752 he4 0.02 n14']
        models0 = 50000
        met = 0.02
        h1 = 0.7048
    elif Z == 'zero':
        genburn = 'rpa0bg'
        abun = ['0.76 h1 0.24 he4 0.00 n14']
        models0 = 20000
        met = 0.00
        h1 = 0.76
    elif Z == 'tenth':
        genburn = 'rpaz-1bg'
        abun = ['0.7545 h1  0.2435 he4 0.002 n14']
        models0 = 20000
        met = 0.002
        h1 = 0.7545
    elif Z == 'low-Z':
        genburn = 'rpa3bg'
        abun = ['0.759 h1 0.24 he4 0.001 n14']
        models0 = 20000
        met = 0.001
        h1 = 0.759
    elif z == 'fifth':
        genburn = 'rpa004bg'
        abun = ['0.749 h1 0.247 he4 0.004 n14']
        models0 = 30000
        met = 0.004
        h1 = 0.749
    elif z == 'half':
        genburn = 'rpa01bg'
        abun = ['0.7324 h1 0.2666 he4 0.01 n14']
        models0 = 30000
        met = 0.01
        h1 = 0.7324
    elif z == 'double':
        genburn = 'rpa04bg'
        abun = ['0.6496 h1 0.3104 he4 0.04 n14']
        models0 = 50000
        met = 0.04
        h1 = 0.6496
    elif z == 'five':
        genburn = 'rpaz5bg'
        abun = ['0.484 h1 0.416 he4 0.1 n14']
        models0 = 50000
        met = 0.1
        h1 = 0.484
    elif z == 'ten':
        genburn = 'rpaz10bg'
        abun = ['0.208 h1 0.592 he4 0.2 n14']
        models0 = 100000
        met = 0.2
        h1 = 0.208
    elif z == 'hesol':
        genburn = 'rpahesbg'
        abun = ['0.98 he4 0.02 n14']
        models0 = 50000
        met = 0.02
        h1 = 0.0e0
    elif z == 'h1sol':
        genburn='rpah1sbg'
        abun=['0.1 h1 0.88 he4 0.02 n14']
        models0 = 50000
        met = 0.02
        h1 = 0.1e0
    elif z == 'jlf':
        genburn = 'rpajlfbg'
        abun = ['0.706 h1 0.275 he4 0.0011 n14 0.003 c12 0.0096 o16 0.0053 fe54']
        models0 = 50000
        met = 0.019
        h1 = 0.706e0
    elif z == 'jxf':
        genburn = 'rpajxfbg'
        abun = ['0.709 h1 0.279 he4 0.013 n14']
        models0 = 50000
        met = 0.013
        h1 = 0.709e0

    if models is None: models = models0

    gendir = os.path.join(dir0, 'genburn')

    shutil.copy2(os.path.join(gendir, genburn), rundir)
    shutil.copy2(os.path.join(gendir, adapnet), rundir)
    shutil.copy2(os.path.join(datadir, bdat), rundir)
    shutil.copy2(os.path.join(progfile), os.path.join(rundir, 'k'))
    os.symlink(os.path.join(bdat), os.path.join(rundir, 'bdat'))

    genfile = os.path.join(rundir, run + 'g')

    abu = '\n'.join(['m acret {:s}'.format(a) for a in abun])

    with open(genfile, 'w') as f:
        f.write("""c
box a13 alex
net 1 h1 he3 he4 n14 c12 o16 ne20 mg24
net 1 si28 s32 ar36 ca40 ti44 cr48 fe52
net 1 ni56 fe54 pn1 nt1
m nstar 1.00 fe54
c solar abundances (weight %):
{abu:s}
c
g 0   2.0000e25  1 nstar  1.0e+8  1.0e+9
g 1   1.9000e25  1 nstar  1.0e+8  1.0e+9
g 40  1.0000e21  1 nstar  1.0e+8  1.0e+6
g 50  1.0000e20  1 nstar  1.0e+8  1.0e+6
g 51  8.0000e19  1 acret  5.0e+7  1.0e+6
g 54  2.0000e19  1 acret  2.5e+7  1.0e+8
g 55  0.         1 acret  1.1e+7  1.0e+4
dstat
genburn {genburn:s}
p 1 1.e-4
p 5 40
p 6 .05
p 7 .05
p 8 .10
p 9 .1
p 10 .99
p 14 1000000
p 16 100000
p 18 10
p 24 0.1
p 28 2
c p 29 1.5284 - BAD
c p 30 1.5284
c p 31 1.5284
c p 32 1.5284
c p 33 1.5284
p 39 50.
p 40 2.
p 46 .15
p 47 3.e-3
c p 48 1.  - DEFAULT
c p 49 1.e+50 - DEFAULT
p 52 10000
c p 53 .1 -DEFAULT
c p 54  2. -DEFAULT
p 55  10.
c p 59 .05 -DEFAULT
p 60 1.0e+06
p 61 2.8e+33
c accretion luminosity is 16E34*(L_acc/L_Edd)
p 62 {p62:e}
p 65 1.0e+99
p 70 1.e+99
p 73 1.e+99
p 75 1.e+99
p 80 .25
c - the following could be less
c p 82 (p 60) might work - or set 0.
c p 82 1.e+6 - check code for 1.d99
p 82 0.
p 83 1.e+4
c this may affect PRE bursts
p 84 2.e-5
p 86 0
p 87 0
p 93 51
c effectively default
p 88 1.e+14
c should be irrelevant
p 105 3.e+9
p 132 6
p 138 .33
p 139 .5
p 144 1.3
c zoning
p 150 .01
p 151 .015
p 152 .03
c dump saving
p 156 10
c p 159 5
c p 160 0 - DEFAULT
c p 189 .02 - switch off
p 189 1.
p 206 .003
c graphics window
p 42 14001000
p 199 -1.
c p 388 1
c opacity table - updated by Laurens
c p 377 0
p 377 4
p 299 10000000
c CHECK - we want to use
c p 265 -1
p 265 1
p 64 1
p 324 0.
p 325 0.1
p 326 0.01
c add more to disable
p 405 -1.d0
p 406 -1.d0
p 420 -1.d0
c no h/he burn dumps
p 454 -1.
p 456 -1.
c plotting
p 64 1
p 434 1
p 443 2
p 419 2.800000019999895D33
c convection - OVERWRITTEN LATER
p 147 1.
p 146 1.
p 148 0.01

p 233 1.d7
p 65 1.d7
c accretion rate is 17.5E-9*(L_acc/L_Edd)
p 211 {p211:e}
p 444 51
c plotting
p 119 40
p 132 4
p 128 1.d-4
c zoning
p 336 1.5d19
p 445 1.d20
p 437 10
p 376 1
p 11 1.d-8
p 12 1.d-8
c
p 137 1
c
c
""".format(genburn = genburn,
           p211 = p211,
           p62 = p62,
           abu = abu))

    if accmass != 0:
        p460 = 'p 460 {accmass:e}'.format(accmass)
    else:
        p460 = 'c'

    cmdfile = os.path.join(rundir, run + '.CMD')
    with open(cmdfile, 'w') as f:
        f.write("""c
p btempmin 1.d10
p tnucmin 1.d10
p 211 0.
@timesec>1.d15
zerotime
setcycle 0
p 1 1.d-4
c system "rm +run+.cnv"
c system "rm +run+.lc"
p tnucmin 1.d7
p btempmin 1.d7
% accretion rate is 17.5E-9*(L_acc/L_Edd)
p 211 {p211:e}
p 315 0.
p 453 0.001
{p460:s}
c
% rezone
@cycle>500
p 86 1
p 87 1
p 452 0
p 458 1.
@cycle=={models:d}
end""".format(
            models = models,
            p211 = p211,
            p460 = p460))

    shutil.copy2(os.path.join(rundir, run+'.CMD'), os.path.join(rundir, run+'.cmd'))

    print('{run:7s} {p211:13e} {met:7e} {h1:7e} {accrate:26e} {Z:s} ..{mod:d}k'.format(
        run = series+srun,
        p211 = p211,
        met = met,
        h1 = h1,
        accrate = accrate,
        Z = Z,
        mod = models // 1000))
