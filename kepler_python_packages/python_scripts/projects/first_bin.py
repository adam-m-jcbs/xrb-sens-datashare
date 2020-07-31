import convdata
import os.path
import physconst
import numpy as np
 
path = '/m/kepler/znuc'
masses = [15, 30, 45, 60]
models = ['z{:d}'.format(m) for m in masses]
dumps = [os.path.join(path,m,m+'.cnv') for m in models]

def get_times():
    for d in dumps:
        c = convdata.loadconv(d)
        t = c.time
        T = c.tc
        ims = ((np.where(T>1.5e8))[0])[0]
        tms = t[ims]/physconst.SEC*1.e-6
        tpms = (t[-1]-t[ims])/physconst.SEC*1.e-6
        print('{:s} {:6.3e} {:6.3e}'.format(d, tms, tpms))
