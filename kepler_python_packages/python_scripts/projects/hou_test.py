#! /bin/env python3

import kepdump
import os.path
import physconst
import numpy as np

def test():
    fn = '/home/alex/kepler/test/xrbhe8#10274'
    d = kepdump.loaddump(fn)
    xkn = d.xkn
    xln = d.xln

    xlx = ( physconst.Kepler.a*physconst.Kepler.c/3.*(4.*np.pi*d.rn[:-1]**2)**2
     *(xkn[:-1]+xkn[1:])/(xkn[:-1]*xkn[1:]*(d.xm[:-1]+d.xm[1:]))
     *(d.tn[:-1]**4-d.tn[1:]**4))

    k = ( 4.* physconst.Kepler.a * physconst.Kepler.c  *
          (d.tn[:-1]**3 + d.tn[1:]**3) / ( 3. * (d.dn[:-1] + d.dn[1:]) *
                                           (d.xkn[:-1] + d.xkn[1:]) *0.5))

    dtdr = -(d.tn[:-1] - d.tn[1:]) / (d.rn[:-1] - d.rn[1:])

    dedT = d.en / d.tn

    for x in zip(np.log10(d.ym[1:]/(4.*np.pi*d.rn[1:]**2)),
                 np.log10(k),
                 np.log10(dtdr),
                 np.log10(dedT[1:])):
        print(x)
        
    
if __name__ == "__main__":
    test()
