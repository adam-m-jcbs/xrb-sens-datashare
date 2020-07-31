"""
Python module to plot isotopic abundances data.
This is to replace IDL/yieldplot[mult]

(under construction)
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path

import convdata

class sundata(object):
    def __init__(self,
                 silent = False,
                 **kwargs):
        self.setup_logger(silent)
        dir0  = kwargs.get('dir0','/home/alex/kepler/solag89/sgrid')
        base  = kwargs.get('base', 's1')
        ext   = kwargs.get('ext', '.cnv')
        runs  = kwargs.get('runs', ('r1','m1', 'm2', 'm3', 'm4'))
        # runs  = ('m2', 'm3', 'm4')
        c = []
        for run in runs:
            file = os.path.join(dir0, base + run, base + ext)
            c += [convdata.loadconv(file)]
        self.data = np.array(c)
        self.runs = np.array(runs)
        self.dir0  = dir0
        self.base  = base
        self.ext   = ext
        self.close_logger(timing = 'Data loaded in')

    def add(self, runs, **kwargs):
        self.setup_logger(silent)
        if isinsrance(runs, str):
            runs = (runs,)
        dir0  = kwargs.get('dir0', self.dir0)
        base  = kwargs.get('base', self.base)
        ext   = kwargs.get('ext' , self.ext)
        c = []
        for run in runs:
            file = os.path.join(dir0, base + run, base + ext)
            c += [convdata.loadconv(file)]
        self.data = np.append(self.data, c)
        self.runs = np.append(self.runs, runs)
        self.close_logger(timing = 'Data loaded in')

    def nHz(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for run, data in zip(self.runs, self.data):
            # plot(c.time,c.awx)
            ax.plot(data.time, data.aw/(2* np.pi)*1.e9, label = run)
        ax.set_ylabel('central rotation / nHz')
        ax.set_xlabel('time / s')
        ax.set_yscale('log')
        plt.legend(loc='lower left')
        plt.draw()
        self.nHz_figure = fig

    def awx(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for run, data in zip(self.runs, self.data):
            ax.plot(data.time, data.awx * 10, label = run)
        ax.set_ylabel(r'$\widetilde{\omega}$ / $\mathrm{rad}\,\mathrm{cm}^2\,\mathrm{s}^{-1}\,\mathrm{g}^{-2/3}$')
        ax.set_xlabel('time / s')
        ax.set_yscale('log')
        plt.legend(loc='lower left')
        plt.draw()
        self.awx_figure = fig

def x7k():
    """
    plot model 7000 info
    """
    import convplot

    from utils import ergs2mbol
    import winddata
    import physconst
    f=figure()
    w = winddata.loadwind('s1.wnd')
    s = slice(6000,8001)
    plot(log10(w.data.teff[s]),log10(w.data.sl[s]/physconst.XLSUN))
    xlim(reversed(xlim()))
    s = 7000
    scatter(log10(w.data.teff[s]),log10(w.data.sl[s]/physconst.XLSUN))
    ylabel('log ( L / $L_\odot$ )')
    xlabel('log ( $T_\mathrm{eff}$ / K )')
    savefig("hrd_7000_6k-8k.pdf")

    c = convdata.loadconv('s1.cnv', firstmodel = 6000, lastmodel = 8000)
    p = convplot.plotconv(c)
    vlines(c.time[1001],0,2.e33)
    p.update(showmodels=True)
    savefig("kd_7000_6k-8k.pdf")


    import kepdump
    d = kepdump.loaddump('s1#7000')
    D = kepdump.loaddump('/home/alex/kepler/solag89/SNUCB/S1/S1#hign')

    f = figure()
    plot(d.zm_sun, d.angwcstn, label = 'hign')
    plot(D.zm_sun, D.angwcstn, label = '7000')
    yscale('log')
    ylabel('$\widetilde\omega$ / $\mathrm{rad}\,\mathrm{cm}^2\,\mathrm{s}^{-1}\,\mathrm{g}^{-2/3}$')
    xlabel('mass coordinte / $M_\odot$')
    legend(loc='lower right')
    savefig("wx_m.pdf")


    f = figure()
    plot(D.zm_sun, D.angjn, label = 'hign')
    plot(d.zm_sun, d.angjn, label = '7000')
    yscale('log')
    ylabel('specific angular momentum / $cm^2\,s^{-1}$')
    xlabel('mass coordinte / $M_\odot$')
    ylim(1.e12,1.e16)
    legend(loc='lower right')
    savefig("j_m.pdf")


    f = figure()
    plot(D.zm_sun, D.angjtn, label = 'hign')
    plot(d.zm_sun, d.angjtn, label = '7000')
    yscale('log')
    ylim(1.e43,1.e49)
    ylabel('angular velocity / erg$\,s$')
    xlabel('mass coordinte / $M_\odot$')
    legend(loc='lower right')
    savefig("jt_m.pdf")

    f = figure()
    plot(D.zm_sun, D.angwn, label = 'hign')
    plot(d.zm_sun, d.angwn, label = '7000')
    yscale('log')
    ylabel('angular velocity / $\,s^{-1}$')
    xlabel('mass coordinte / $M_\odot$')
    legend(loc='upper right')
    savefig("w_m.pdf")

    p = convplot.plotconv(c, radius = True, logarithmic = True)
    a = p.fig.axes[0]
    a.set_ylim(8,12)
    draw()
    vlines(c.time[1001],0,100)
    p.update(showmodels = True)
    savefig("kdr_7000_6k-8k.pdf")



    dm = kepdump.loaddump('s1#6000')
    dp = kepdump.loaddump('s1#8000')
    f = figure()
    plot(D.rn/D.rn[-1], D.angwn/(2*pi)*1e9, label = 'hign')
    plot(d.rn/d.rn[-1], d.angwn/(2*pi)*1e9, label = '7000')
    xscale('log')
    xlim(1.e-5,1.)
    yscale('log')
    xlabel('r/R')
    ylabel('rotation / nHz')
    plot(dm.rn/dm.rn[-1], dm.angwn/(2*pi)*1e9, label = '6000')
    plot(dp.rn/dp.rn[-1], dp.angwn/(2*pi)*1e9, label = '8000')
    legend(loc='lower left')
    savefig("freq.pdf")


def seq():
    import convplot
    from utils import ergs2mbol
    import winddata
    import physconst
    import kepdump
    from color import isocolors
    import os.path

    dir00 = os.path.expand_user("~")
    dir0 = os.path.join(dir00, "kepler","solag89","sgrid","s1m1")

    f = figure()
    models = np.arange(6000,8001,100, dtype=np.int64)
    colors = isocolors(models.size, stop=330)
    file0 = os.path.join(dir0,"s1")
    files = [file0+"#{:d}".format(i) for i in models]
    dumps = [kepdump.loaddump(file) for file in files]
    w = winddata.loadwind(file0+'.wnd')
    s = slice(6000,8001)
    plot(log10(w.data.teff[s]),
         log10(w.data.sl[s]/physconst.XLSUN),
         color='black')
    xlim(reversed(xlim()))
    for c,s in zip(colors,models):
        scatter(log10(w.data.teff[s]),
                log10(w.data.sl[s]/physconst.XLSUN),
                      color = c)
    ylabel('log ( L / $L_\odot$ )')
    xlabel('log ( $T_\mathrm{eff}$ / K )')
    savefig("hrd_spots.pdf")

    f = figure()
    for c,d in zip(colors, dumps):
        s = d.center_slice
        freq = d.angwn[s]/(2* np.pi)*1.e9
        r = d.rn[s] / d.qparm.radius
        plot(r, freq, color = c)
    ax = f.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ylabel('frquency / nHz')
    xlabel('r / R')
    f.gca().set_xlim(1e-5,1)
    draw()
    savefig("freq_spots.pdf")
