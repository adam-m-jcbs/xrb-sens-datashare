from human import time2human

from .dataplot import DataPlot

class PlotTRho(DataPlot):

    def __init__(self, *args, **kwargs):

        kwargs = kwargs.copy()

        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')

        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)

        xlabel = 'density (g/ccm)'
        ylabel = 'temperature (K)'

        super().__init__(*args, **kwargs)


        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.update()

    def draw(self):
        self.clear()
        data = self.data
        qparm = data.qparm
        parm = data.parm
        jm = data.jm
        iim = slice(1, jm+1)
        dn = data.dn[iim]
        tn = data.tn[iim]
        self.ax.plot(dn, tn, 'b')
        self.ax.text(0.99, 0.99, r'{:d} - {}'.format(qparm.ncyc, time2human(parm.time, 15)),
                     horizontalalignment = 'right',
                     verticalalignment = 'top',
                     transform = self.ax.transAxes)
