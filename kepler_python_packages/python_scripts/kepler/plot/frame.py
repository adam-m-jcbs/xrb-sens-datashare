import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os.path
import time

import subprocess
import re

default_style = 'white'

class Frame():
    def __init__(self,
                 fig = None,
                 scale = 1,
                 style = default_style,
                 title = 'KEPLER',
                 update = True,
                 figsize = None,
                 **kwargs):

        self.set_style(style)

        self.title = title
        self.scale = scale
        if fig is not None:
            self.fig = fig
        else:
            self.set_figure(figsize = figsize)

        self.time = 0
        self.debug = kwargs.get('debug', True)
        self._plots = []
        self.header = None
        self.clear()
        if update:
            self.update()

    def set_style(self, style):
        self.style = style
        try:
            plt.style.use(os.path.join(os.path.dirname(__file__), f'{style}.mplstyle'))
        except OSError:
            pass
        else:
            return
        try:
            plt.style.use(style)
        except OSError:
            pass
        else:
            return
        if style != default_style:
            self.set_style(default_style)

    def _set_title(self, title):
        self.fig.canvas.set_window_title(title)

    def set_title(self):
        self._set_title(self.title)

    @staticmethod
    def get_dpi():
        # this part needs to be rewritten based on nucplot.py code
        # if xrandr is not available
        dpi = 100
        if mpl.get_backend() == 'TkAgg':
            output = subprocess.run(['xrandr'], stdout=subprocess.PIPE).stdout.decode()
            pixels = np.array(re.findall('(\d+)x(\d+) ', output)[0], dtype = np.int)
            mm = np.array(re.findall('(\d+)mm x (\d+)mm', output)[0], dtype = np.int)
            dpi = 25.4 * np.sqrt(np.sum(pixels**2) / np.sum(mm**2))
        return dpi

    def set_figure(self, figsize = None):
        if figsize == None:
            figsize = (6, 8)
        self.fig = plt.figure(
            figsize = figsize,
            )

        self.set_title()

        dpi0 = self.fig.get_dpi()
        dpi = self.get_dpi()
        if dpi > dpi0:
            print(' [Plot] setting dpi to {}'.format(dpi))
            self.fig.set_dpi(dpi * self.scale)

    def update(self, interactive = False):
        # if (not interactive and
        #     time.time() - self.time <  self.data.ipdtmin):
        #     return
        self.time = time.time()
        self.draw()
        self.show(interactive)

    def show(self, interactive = False):
        if interactive:
            return
        # plt.draw()
        self.fig.canvas.manager.show()
        plt.pause(0.001)
        #? self.fig.show(block = False)
        self.fig.canvas.draw()
        time.sleep(1e-6)
        self.fig.canvas.draw_idle()
        if self.debug:
            print(f' [Frame.show] Updating plot.')

    def draw(self):
        if self.header is not None:
            self.header.draw()
        if self._plots is not None:
            for p in self._plots:
                p.draw()

    def close(self):
        self.fig.canvas.manager.destroy()

    def save(self, filename):
        if self.debug:
            print(f' [Frame.save] facecolor: {self.fig.get_facecolor()}')
        self.fig.savefig(
            filename,
            facecolor = self.fig.get_facecolor(),
            transparent=True,
            )

    def clear(self):
        self.fig.legends.clear()
        self.legends = set()

    def check_legend(self, legend):
        check = legend in self.legends
        if not check:
            self.legends.add(legend)
        return check
