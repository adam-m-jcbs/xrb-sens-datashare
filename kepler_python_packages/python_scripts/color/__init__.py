"""
Python module for color functions.

"""

import numpy as np

if np.geterr()['under'] != 'ignore':
    print('[COLORFUNC] Warning: setting NumPy underflow error to \'ignore\'.')
    np.seterr(under='ignore')

# package imports

from .utils import color, rgb, colrgb
from .models import *
from .functions import *
from .levels import *
from .filters import *
from .xfilters import *
from .yfilters import *
from .ifilters import *
from .standard import *

from .compat import *

from .test import *
