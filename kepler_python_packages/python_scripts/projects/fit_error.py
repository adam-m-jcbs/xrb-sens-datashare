"""
Make plot for error fit
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

def plot_prop():
    f = plt.figure()
    ax = f.add_subplot(111)
    x = np.linspace(-10,10,1000)
    ax.plot(x,norm.cdf(-x),linewidth=3,label='cdf', color = 'b')
    ax.plot(x,norm.pdf(-x),linewidth=3,label='pdf', color = 'g')
    ax.plot(x,norm.pdf(-x)/norm.cdf(x+3),linewidth=3,label='pdf with threshold at -3', color = 'g', linestyle = '--')
    ax.legend(loc = 'best', fontsize = 'medium')
    ax.set_xlabel(r'deviation $x/\sigma$')
    ax.set_ylabel(r'likelihood $P\left(x/\sigma\right)$')
    f.tight_layout()
    f.show()


def plot_chi():
    f = plt.figure()
    ax = f.add_subplot(111)
    x = np.linspace(-10,10,1000)
    ax.plot(x,-2*norm.logcdf(-x),linewidth=3,label='limit', color = 'b')
    ax.plot(x,x**2,linewidth=3,label='detection', color = 'g')
    ax.plot(x,x**2+2*norm.logcdf(x-(-3)),linewidth=3, linestyle = '--', label='detection with threshold at $x/\sigma=-3$', color = 'g')
    ax.set_ylabel(r'badness $\chi^2\left(x/\sigma\right)')
    ax.set_xlabel(r'deviation $x/\sigma$')
    ax.legend(loc = 'best', fontsize = 'medium')
    ax.set_ylim((-5,105))
    f.tight_layout()
    f.show()
