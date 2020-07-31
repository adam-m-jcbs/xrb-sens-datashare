import numpy as np

from matplotlib.pylab import get_backend

from .frame import Frame
from .framekep import FrameKep

class PlotManager():
    def __init__(self,
                 data = None,
                 default = None,
                 interactive = False,
                 debug = True,
                 style = None,
                 **kwargs,
                 ):
        self.data = data
        self.default = default
        self.plots = []
        self.interactive = interactive
        self.debug = debug
        self.style = style

    def plot(self, *args, **kwargs):
        kwargs.setdefault('debug', self.debug)
        if len(args) == 0:
            p = kwargs.pop('ipixtype', None)
            if p is not None:
                args = (p,)
        self.gc()
        if len(args) == 0:
            args = (self.data.parm.ipixtype,)
            kwargs.setdefault('duplicates', False)
        kwargs.setdefault('duplicates', True)
        for p in args:
            plot = self.add_plot(p, **kwargs)
        if len(self.plots) == 0 and len(args) == 0:
            if self.default is not None:
                plot = self.add_plot(self.default, **kwargs)
            else:
                plot = self.add_plot(self.data.parm.ipixtype, *args, **kwargs)
        elif len(args) == 0:
            self.update(**kwargs)

    def update(self, *args, **kwargs):
        self.gc()
        if len(args) == 0:
            args = range(len(self.plots))
        interactive = kwargs.pop('interactive', self.interactive)
        for i in args:
            p = self.plots[i]
            if isinstance(p, type):
                self.plots[i] = p(data, **kwargs)
            else:
                p.update(interactive = interactive)

    @staticmethod
    def is_closed(fig):
        # this should have been plt.fignum_exists(fig.number) but that fails.
        canvas = fig.canvas
        backend = get_backend()
        if backend in ('GTK3Agg', 'GTK3Cairo',):
            return not canvas.is_drawable()
        if backend in ('TkAgg', 'TkCairo',):
            return canvas.manager.window == None
        if backend in ('QT5Agg', 'QT5Cairo',):
            return not canvas.isVisible()
        raise AttributeError(f'backend {backend} not supported.')

    def gc(self):
        """
        garbage collect closed plots.
        """
        for p in self.plots.copy():
            if isinstance(p, type):
                continue
            if self.is_closed(p.fig):
                print(f' [manager.gc] Removing closed plot {p}')
                self.plots.remove(p)

    @staticmethod # could be in frame.py
    def compare_plot_type(p1, p2):
        o = object()
        t1 = t2 = o
        if isinstance(p1, Frame):
            t1 = getattr(p1, 'type', None)
        if isinstance(p2, Frame):
            t2 = getattr(p2, 'type', None)
        if not (t1 == t2 == o):
            if t1 == t2 or t1 == p2 or p1 == t2:
                return True
        if (isinstance(p2, type) and
            (not p2 == Frame) and
            (isinstance(p1, Frame)) and
            (type(p1) == p2)):
            return True
        if (isinstance(p1, type) and
            (not p1 == Frame) and
            (isinstance(p2, Frame)) and
            (type(p2) == p1)):
            return True
        return False


    def add_plot(self, *args, **kwargs):
        if len(args) > 0:
            plot, *args = args
        else:
            plot = kwargs.pop('plot', None)
        plot_ = plot

        duplicates = kwargs.pop('duplicates', False)
        self.gc()
        if plot is None and self.default is not None:
            plot = self.default
        if plot is None:
            print(f' [add_plot] No plot added for {plot_}.')
            return
        update = None
        if not duplicates:
            for p in self.plots:
                if self.compare_plot_type(p, plot):
                    update = p
                    break
        kw = {}
        if 'interactive' in kwargs:
            kw['interactive'] = kwargs['interactive']
        if update is not None:
            update.update(**kw)
            return
        if isinstance(plot, (int, np.int32, np.int64)):
            kwp = kwargs.copy()
            if not 'style' in kwp and self.style is not None:
                kwp['style'] = self.style
            try:
                plot = FrameKep(self.data, plot, **kwp)
            except Exception as e:
                print(f' [add_plot] Plot type {plot_} not supported.')
                if self.debug:
                    print(f' [add_plot] ERROR {e}')
                    raise
                return
        if isinstance(plot, type) and issubclass(plot, Frame):
            plot = plot(self.data, *args, **kwargs)
        assert isinstance(plot, Frame)
        self.plots.append(plot)
        # plot.show(**kw)
        return plot

    def closewin(self):
        for p in self.plots:
            p.close()
        self.plots = []

    def __len__(self):
        self.gc()
        return len(self.plots)
