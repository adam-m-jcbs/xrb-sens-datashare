"""
plots for 2012...2015 dusty SN project
"""
import convdata, convplot
import numpy as np
import os.path
import matplotlib.pyplot as plt
import kepdump


# for rev 1 use
#    kd(ymax = 6)
def kd(c = None, **kwargs):
    if c is None:
        c = convdata.load('~/kepler/sollo09/snuc/s15/s15.cnv.gz')
    p = convplot.plot(
        c,
        logtime = -8.2,
        stability = ['conv', 'semi', 'osht', 'thal'],
        solar = True,
        **kwargs)

# for rev 1 Figure we now call
#    presn(1.5, 6)
def presn(mmin = None, mmax = None):
    """
    Make composition plot
    """
    f = plt.figure(
        figsize = (8,6),
        dpi = 102,
        facecolor = 'white',
        edgecolor = 'white'
        )
    ax = f.add_subplot(111)
    d = kepdump.load('~/kepler/sollo09/snuc/s15/s15#presn')
    c = d.core()
    a = d.abu
    ax.set_yscale('log')
    if mmax == None:
        mmax = d.zm_sun[-2]
    if mmin == None:
        mmax = d.zm_sun[0]
    if mmin < 0:
        mmin = d.core()['iron core'].zm_sun + mmin

    ax.set_xlim(mmin, mmax)
    ax.set_ylim(1.e-4, 1)
    ax.axvspan(0, c['iron core'].zm_sun, color = '#808080', label = 'iron core')
    ax.axvspan(c['iron core'].zm_sun, c['O shell'].zm_sun, color = '#C0C0C0', label = 'piston')
    ax.set_xlabel('Mass coordinate (solar masses)')
    ax.set_ylabel('Mass fraction')
    j = c['O shell'].j
    ax.plot(d.zm_sun[j:], a.ion_abu('h1')[j:] , label = 'H')
    ax.plot(d.zm_sun[j:], a.ion_abu('he4')[j:], label = 'He')
    ax.plot(d.zm_sun[j:], a.ion_abu('c12')[j:], label = 'C', color = 'k', lw=2)
    ax.plot(d.zm_sun[j:], a.ion_abu('o16')[j:], label = 'O', lw = 2)
    ax.plot(d.zm_sun[j:], a.iron()[j:]        , label = 'iron group', lw = 2)

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(20)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(16)


    ax.legend(loc='best')
    f.tight_layout()
    f.savefig('s15_sollo09_presn.pdf')

    plt.draw()
