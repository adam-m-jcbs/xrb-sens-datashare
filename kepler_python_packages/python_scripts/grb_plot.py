"""
Python module to plot GRB data

(under construction)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.legend import Legend
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

from kepdump import loaddump, KepDump
from physconst import Kepler
from color import isocolors


def plot_jj(dumpfile,
            xlim = None,
            ylim = None,
            figlabel = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dump = loaddump(dumpfile)
    s = dump.center_slice
    zm = dump.zmm_sun[s]
    ax.plot(zm, np.log10(dump.anglstn[s]), color='r',
            label=r'$j_{\mathrm{LSO}}$')
    ax.plot(zm, np.log10(dump.anglsts[s]), color='b',
            label=r'$j_{\mathrm{LSO,Schwarzschild}}$')
    ax.plot(zm, np.log10(dump.anglstk[s]), color='g',
            label=r'$j_{\mathrm{LSO,Kerr}}$')
    ax.plot(zm, np.log10(dump.angjeqn[s]), color='k',lw=2,
            label=r'$j_{\mathrm{eq}}$')
    ax.set_xlabel(r'$m / \mathrm{M}_\odot$')
    ax.set_ylabel(r'$\log\left(\,j\;/\;\mathrm{cm}^2\,\mathrm{s}^{-1}\right)$')
    plt.legend(loc=8)

    fp = FontProperties(size=10)
    fp = None
    # names={'C/O core': 'CO core',
    #        'Ne/Mg/O core': 'NeMgO core'}
    names = None

    show_core(dump, ax,
              names = names,
              fp = fp)
    ax.autoscale(enable=True,axis='x',tight=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if xlim is not None:
        if isinstance(xlim, dict):
            ax.set_xlim(**xlim)
        else:
            ax.set_xlim(xlim)
    if ylim is not None:
        if isinstance(ylim, dict):
            ax.set_ylim(**ylim)
        else:
            ax.set_ylim(ylim)

    # this seems to be the best way to it aligned with axis
    # here we should have a general utility or base class
    if figlabel is not None:
        proxy = Rectangle((0, 0), 1, 1, alpha = 0)
        ax.add_artist(Legend(ax, [proxy], [figlabel],
                             loc = 2,
                             handlelength=0,
                             handleheight=0,
                             borderpad = 0,
                             handletextpad = 0,
                             frameon = False,
                             prop = {'size':'x-large'}))
    plt.show()


def show_core(core,
              axes,
              coord = 'zm',
              scale = 1.,
              fp = None,
              selection = ['iron core',
                           'Si core',
                           'Ne/Mg/O core',
                           'C/O core',
                           'He core',
                           'star'],
              names = None # dict core_name:disp name
              ):
    # todo: make sure core colors are consistent between different plots?
    """
    Add core information to plot.
    """
    if isinstance(core, KepDump):
        core = core.core()
    colors = ['blue','green','red','cyan','magenta','yellow']
    colors = isocolors(12)[(np.arange(12)*17) % 12]
    alpha = 0.2
    x0 = 0.

    # add cores and labels for legend
    patches = []
    labels  = []
    for i,s in enumerate(selection):
        if s not in core:
            continue
        x1 = core[s].__getattribute__(coord)/scale
        patches += [axes.axvspan(x0,x1,
                                 color = colors[i],
                                 alpha = alpha,
                                 linewidth = 0)
                    ]
        if names is not None:
            label = names.get(s,s)
        else:
            label = s
        labels += [label]
        x0 = x1
    leg = Legend(axes, patches, labels,
                 loc = 4,
                 prop = fp)
    axes.add_artist(leg)
