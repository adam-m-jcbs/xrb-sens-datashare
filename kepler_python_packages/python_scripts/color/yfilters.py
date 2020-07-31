"""
YFilters pass an adutional data array on filter instantenation that is
used to modify the RGBA array
"""

import numpy as np

from .filters import ColorFilter
from .models import color_model
from .utils import color, rgb

class YFilter(ColorFilter):
    """
    Class to modify data color array before returned based separate
    data array provided on initialisation.

    The filter takes as call arguments RGBA normalized to 0...1.

    When used a _filter function parameters must be passes by
    keyword.
    """

    _filter = None

    def __init__(self, *args, **kwargs):
        kw = kwargs.copy()
        self._data = kwargs.pop('data')
        func = kw.pop('func', lambda x: x)
        if np.isscalar(func):
            self._func = lambda x : np.tile(func, x.shape)
        else:
            self._func = func
        super().__init__(*args, **kw)

    def _yfilter(self, rgba):
        """
        take rgba, filter, replace data in place, return modified array
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        # use as _yfilter
        print(args[0].shape, self._data.shape)

        if (self._color is None and
            len(args) == 1 and
            len(kwargs) == 0 and
            isinstance(args[0], np.ndarray) and
            isinstance(self._data, np.ndarray) and
            len(args[0].shape) == 3 and
            args[0].shape[-1] == 4 and
            np.allclose(args[0].shape[:-1], self._data.shape[:2])):
            return self._yfilter(*args)

        return super().__call__(*args, **kwargs)

class FuncAlphaYFilter(YFilter):
    """
    set alpha based on data array
    """
    def _yfilter(self, rgba):
        ii = (0 <= self._data) &  (sefl._data <= 1)
        rgba[ii, 3] = np.clip(self._func(self._data[ii]), 0, 1)
        return rgba
