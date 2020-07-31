import matplotlib.pylab as plt
import numpy as np

from . import models

_output_filter = None

_debug_ofilter = False

def set_output_filter(of):
    global _output_filter
    # maybe not needed, just RGB[A] --> RGB[A]
    #from .filters import ColorFilter
    #assert isinstance(of, ColorFilter)
    _output_filter = of

def get_output_filter(context = None):
    # in the future we may want to try to determine current output type
    if _debug_ofilter:
        print(plt.gcf().canvas.get_renderer(), plt.gcf().canvas.is_saving())
        print('O', _output_filter)
    return _output_filter

def clear_output_filter():
    _output_filter = None

#######################################################################
# just a test
class WPFilter():
    def __init__(self, sw = 'D65', dw = 'D65', normalize = True):
        self._M = models._BradfordMatrix(
            models.white_points_CIE1931_2[sw],
            models.white_points_CIE1931_2[dw])

        # normalize?
        if normalize:
            msum = np.max(np.sum(np.maximum(self._M, 0), axis=1))
            self._M /= msum

    def _filter(self, rgba):
        rgba[:, :3] = np.transpose(np.dot(self._M, np.transpose(rgba[:, :3])))
        return rgba
