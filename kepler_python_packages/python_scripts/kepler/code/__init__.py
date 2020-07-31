"""
KEPLER python package
"""

import numpy as np

# # this one does not work - code will just stop on starting kepler
# # kepler start maybe needs a check...
# os.environ['GFORTRAN_CONVERT_UNIT'] = 'big_endian'

np.seterr(over='ignore', invalid='ignore')

__all__ = []

from .main import Kepler
from ._build import clean
from .kepbin import build

# add kepler plots convience functions that use KeplerData
