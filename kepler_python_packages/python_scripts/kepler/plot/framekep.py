import numpy as np
import matplotlib as mpl

from matplotlib.gridspec import GridSpec

from ..datainterface import DataInterface
from .frame import Frame

from .header import HeaderKep
from .kep import kepplots

class FrameKep(Frame):
    def __init__(self,
                 data = None,
                 plots = None,
                 sharex = True,
                 **kwargs):

        assert isinstance(data, DataInterface), 'Requite data source interface DataInterface instance.'
        self.data = data

        kw = kwargs.copy()
        kw.setdefault('style', 'kepler')
        kw.setdefault('update', False)
        kw.setdefault('figsize',
                      tuple(np.array(divmod(data.iwinsize, 10000)) / Frame.get_dpi()))
        title = r'KEPLER - {} - {}'.format(
            self.data.filename,
            self.data.runpath)
        kw.setdefault('title', title)
        super().__init__(**kw)
        # remove fig for sake of using other keywords in kwargs

        kw = kwargs.copy()
        kw['data'] = self.data
        kw['fig'] = self.fig

        showhead = kwargs.pop('showhead', 'default')
        if showhead == 'default':
            self.header = HeaderKep(**kw)
        else:
            self.header = None

        self.type = plots

        if plots is not None:
            nz = 0
            try:
                if int(plots) == plots:
                    px = []
                    while True:
                        plots, p = divmod(plots, 10)
                        if p == 0:
                            nz += 1
                        else:
                            px.insert(0, p)
                        if plots == 0:
                            break
                    plots = px
            except:

                raise Exception(f'Currently only interger plot numbers are supported: {plots}')

            if isinstance(plots, (list, tuple)):
                if len(plots) == 1:
                    gs = GridSpec(1, 1)
                    locs = [(0,0),]
                elif len(plots) == 2 and nz == 0:
                    gs = GridSpec(2, 1)
                    locs = [(0,0), (1,0)]
                elif len(plots) == 3 and nz == 0:
                    gs = GridSpec(2, 2)
                    locs = [(0, slice(None)), (1,0), (1,1)]
                elif len(plots) == 3 and nz == 2:
                    gs = GridSpec(3, 1)
                    locs = [(0,0), (1,0), (2,0)]
                elif len(plots) == 4 and nz == 0:
                    gs = GridSpec(2, 2)
                    locs = [(0,0), (0,1), (1,0), (1,1)]
                else:
                    print(f' [KepFrame] Multi-plot not implemented: {plots}')
                    self.close()
                    raise NotImplementedError()
                    return

                ax0 = None
                for p, l in zip(plots, locs):
                    if self.debug:
                        print(f' [{self.__class__.__name__}] adding plot {p}')
                    if sharex and ax0 is not None:
                        ax = self.fig.add_subplot(gs[l], sharex = ax0)
                    else:
                        ax = self.fig.add_subplot(gs[l])
                    p = kepplots[p](ax = ax, **kw)
                    self._plots.append(p)
                    ax0 = ax


        update = kwargs.get('update', True)
        if update:
            self.update()
