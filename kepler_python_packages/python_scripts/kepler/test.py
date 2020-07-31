from .main import Kepler, kd
from .dumpplot import DumpData, plot
from .plot.kep import kepplots

path = '/home/alex/kepler/test'

dump = 'xxx#presn'

def test1(**kwargs):
    k = Kepler('xxx', dump, 's')
    return kepplots[1](kd, **kwargs)

def test2(p = 1, **kwargs):
    return plot(dump, p, **kwargs)

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from .plot import gridscale

def test_grid_scale():
    x = np.arange(0,1000,1)
    y = x**2
    z = np.sqrt(x)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax3 = ax.twiny()
    ax.plot(z,y)
    ax3.set_xscale(
        'gridscale',
        refaxis = ax,
        )
    ax3.xaxis._scale.update_refvalues(z)
    f.show()


from .plot.trho import PlotTRho
def test3(p = 1, **kwargs):
    return PlotTRho(dump, **kwargs)
