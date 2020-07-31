"""
This module contains input filters that modify the data array before
it is sent to the underlying color function.
"""

import numpy as np

from types import FunctionType

from .functions import Color, colormap

########################################################################
########################################################################
########################################################################


class ColorScale(Color):
    """
    Process *data* array by function before passing to color function

    This is in contrast to 'filters' that process in the output
    pipeline.  This functionallity likely should be extended.

    """
    def __init__(self, *args, **kwargs):
        """
        paramaters:

        args[0] :
            color function or name

        args[1} :
            function

        keywords:

        normalize_func :
            function to normalize data

        """

        self._inherit_color(args[0])
        self._func = args[1]
        assert isinstance(self._func, FunctionType), 'need to provide function'
        self._normalize_func = kwargs.pop('normalize_func', False)
        super().__init__(*args, **kwargs)

    # _function returns rgba array
    _alpha = True

    def _function(self, data, *args, **kwargs):
        kwargs['bytes'] = False
        kwargs['return_data'] = False
        kwargs['normalize'] = self._normalize_func
        rgba = self._color(self._func(data), *args, **kwargs)
        return rgba

#######################################################################

class ColorScaleGamma(ColorScale):
    """
    Process *data* array by gamma function before passing to color function

    """
    def __init__(self, *args, **kwargs):
        """
        paramaters:

        args[0] :
            color function or name

        args[1} :
            gamma
            if gamma < 0, apply -gamma to (1-x)

        keywords:

        normalize_func :
            function to normalize data

        """

        gamma = args[1]
        if gamma >= 0:
            func = lambda x: x**gamma
        else:
            func = lambda x: 1-(1-x)**(-gamma)
        super().__init__(args[0], func, *(args[2:]), **kwargs)


#######################################################################

class ColorScaleExp(ColorScale):
    """
    Process *data* array by (exp(A * x) - 1) / (exp(A) - 1)

    """
    def __init__(self, *args, **kwargs):
        """
        paramaters:

        args[0] :
            color function or name

        args[1} :
            A
            scale factor, default is 1

        keywords:

        normalize_func :
            function to normalize data

        """

        A = args[1]
        if A == 0:
            func = lambda x: x
        else:
            func = lambda x: (np.exp(A * x) - 1) / (np.exp(A) - 1)
        super().__init__(args[0], func, *(args[2:]), **kwargs)


#######################################################################

class ColorScaleLog(ColorScale):
    """
    Process *data* array by (exp(A * x) - 1) / (exp(A) - 1)

    """
    def __init__(self, *args, **kwargs):
        """
        paramaters:

        args[0] :
            color function or name

        args[1} :
            A
            scale factor, default is 1

        keywords:

        normalize_func :
            function to normalize data

        """

        B = args[1]
        if B == 0:
            func = lambda x: x
        else:
            func = lambda x: np.log(x * (B - 1) + 1) / np.log(B)
        super().__init__(args[0], func, *(args[2:]), **kwargs)
