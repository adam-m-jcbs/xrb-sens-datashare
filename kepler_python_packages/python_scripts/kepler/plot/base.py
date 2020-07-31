from .frame import Frame

class BasePlot():
    def __init__(self,
                 frame = None,
                 fig = None,
                 ax = None,
                 **kwargs):

        if ax is not None:
            self.ax = ax
            self.fig = ax.figure
        else:
            if fig is not None:
                self.fig = fig
            elif isinstance(frame, Frame):
                self.fig = frame.fig
            else:
                frame = self.make_default_frame(**kwargs)
            self.ax = self.fig.add_subplot(111)

        self.frame = frame

        self.debug = kwargs.get('debug', True)

        self.fig_legends = []

    def make_default_frame(self, *args, **kwargs):
        return Frame(self, *args, **kwargs)

    def draw(self):
        raise NotImplementedError()

    def clear(self):
        self.ax.lines.clear()
        self.ax.texts.clear()
        self.ax.patches.clear()

        try:
            self.ax2.lines.clear()
            self.ax2.texts.clear()
            self.ax2.patches.clear()
        except:
            pass

        try:
            self.ax.legend_.remove()
        except:
            pass
        try:
            self.ax2.legend_.remove()
        except:
            pass

        for l in self.fig_legends:
            self.fig.legends.remove(l)
        self.fig_legends.clear()

    def check_legend(self, legend):
        for leg in self.fig.legends:
            if hasattr(leg, 'legtype'):
                if leg.legtype == legend:
                    return True
        return False

    def add_legend(self, legtype, legend):
        legend.legtype = legtype
        self.fig_legends.append(legend)

    def update(self):
        pass
