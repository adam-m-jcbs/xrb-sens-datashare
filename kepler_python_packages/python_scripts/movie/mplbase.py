"""
Some ideas how to combine MPL graphics with Movie framework
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pylab as plt

from PIL import Image

from .frames import MovieCanvasBase

class MPLBase(MovieCanvasBase):
    _canvas = 'agg'
    def __init__(self,
                 size = (800, 600),
                 dpi = 72,
                 bg = None,
                 # bg = '#00000000',
                 canvas = None,
                 restore = True,
                 alpha = None,
                 fig = None,
                 ):
        if fig is None:
            old_backend = mpl.get_backend()
            if canvas is None:
                canvas = self._canvas
            mpl.use(canvas)
            self.fig = plt.figure(
                figsize = np.array(size) / dpi,
                dpi = dpi,
                facecolor = bg,
                )
            if restore:
                mpl.use(old_backend)
            self.ownfig = True
        else:
            self.fig = fig
            self.ownfig = False
        if alpha is not None:
            self.fig.patch.set_alpha(alpha)

    def get_array(self):
        """
        Return figure data as array usable for Movie Writer

        This is the interface routine to be used by Movie Writer as
        'getter'.
        """
        return np.ndarray(
            self.get_buf_shape(),
            buffer = self.get_buffer(),
            dtype = np.uint8)

    def get_frame_size(self):
        return self.fig.canvas.get_width_height()[::-1]

    def get_buffer(self):
        """Helper function to get buffer data.r"""
        return self.fig.canvas.print_to_buffer()[0]

    @classmethod
    def from_arr(cls, arr, dpi = 72, bg = '#00000000'):
        """Constructor to create MPL figure from existing image."""
        size = arr.shape[1::-1]
        fig = cls(size = size, dpi = dpi, bg = bg)
        fig.fig.figimage(arr)
        return fig

    def close(self):
        if not self.ownfig:
            return
        # in case we made visible windows
        try:
            self.fig.canvas.manager.destroy()
        except:
            pass

    def get_canvas(self):
        return self.fig

    def clear(self):
        self.fig.clear()

class AggMPLBase(MPLBase):
    _canvas = 'agg'

class CairoMPLBase(MPLBase):
    _canvas = 'cairo'

class TkAggMPLBase(MPLBase):
    _canvas = 'TkAgg'

class TkCairoMPLBase(MPLBase):
    _canvas = 'TkCairo'
