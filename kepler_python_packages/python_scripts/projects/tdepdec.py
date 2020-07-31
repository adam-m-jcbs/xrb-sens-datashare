"""
Project of T-dep decay rates with Maria Lugaro
"""

import kepgen
import os.path


project = 'Tdepdec'

runpar = dict(
    composition='solas12',
    #  mass=15,
    dirtarget = project + '/s{mass}{case}',
    burn=True,
    )

spec_w = dict(
    bdatcopy = True,
    bdat = 'rath00_10.1.bdat_jlf_mal',
    )

spec_w1 = dict(
    bdatcopy = True,
    bdat = '../rath00_10.1.bdat_jlf_mal-hf181',
    )

specials = {''   : dict(),
            'w'  : spec_w,
            'w1' : spec_w1,
        }


def gen(cases = None, **parm):
    """
    Make run
    """
    if cases is None:
        cases = list(specials.keys())
    
    for case in cases:
        p = dict(runpar)
        p.update(specials[case])
        p.update(parm)
        p['run'] = True
        p['dirtarget'] = p['dirtarget'].format(case = case, **p)
        kepgen.MakeRun(**p)
        

def testexp(**parm):
    exppar = dict(exp='D', **runpar)
    p = dict(exppar)

    p.update(parm)
    case = ''
    p['dirtarget'] = p['dirtarget'].format(case = case, **p)
    p['run'] = True
    kepgen.TestExp(**p)

def burnexp(cases = None, **parm):
    exppar = dict(exp='D', **runpar)

    if cases is None:
        cases = list(specials.keys())
        
    try:
        cases.delete('')
        cases.insert(0, '')
    except:
        pass

    for case in cases:
        p = dict(exppar)
        p.update(specials[case])
        p.update(parm)
        p['dirtarget'] = p['dirtarget'].format(case = case, **p)
        if case is not '':
            p['copygen'] = '../s{mass}'.format(**p)
        p['run'] = True
        kepgen.BurnExp(**p)

        

# tdepdec.gen(mass=18)



def make_link(dump = '#final'):
    from kepdata import kepdata
    outdir = '/m/web/Download/Tdepdec'
    indir = '/home/alex/kepler/Tdepdec'
    masses = [12,15,18,25]
    series = 's'
    masses = [series + str(m) for m in masses]
    exp = 'D'
    if not dump.startswith('#'):
        dump = '#' + dump
    for s in specials:
        for m in masses:
            kepdata(
                outfile = os.path.join(outdir,m+s+exp+'@@'+dump[1:]),
                filename = os.path.join(indir,m+s,m+exp+dump),
                molfrac = True,
                radiso = True)
